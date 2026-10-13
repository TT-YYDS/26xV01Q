from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import uuid

import numpy as np
import open3d as o3d

from .pipeline import PipelineResult, vertical_plane_basis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNSTREAM_ROOT = PROJECT_ROOT
DEFAULT_DOWNSTREAM_URL = os.environ.get("POINTCLOUD_CAD_URL", "http://127.0.0.1:5000")
EXTERNAL_SECTION_MODE = "unified_final_section"


@dataclass(frozen=True)
class DownstreamHandoff:
    token: str
    analysis_url: str
    package_directory: Path
    section_cloud_path: Path
    processing_report_path: Path
    section_points: int


def _load_summary(result: PipelineResult) -> dict:
    try:
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取当前运行摘要：{exc}") from exc
    if not isinstance(summary, dict):
        raise RuntimeError("当前运行摘要格式无效")
    return summary


def _derive_processing_quality(summary: dict) -> dict[str, object]:
    """Summarize whether the stitched section is eligible for formal CAD results."""
    config = summary.get("config", {})
    registration_config = config.get("registration", {}) if isinstance(config, dict) else {}
    coverage_preserving_fusion = bool(
        registration_config.get("coverage_preserving_fusion", False)
    )
    metrics = summary.get("registration_metrics", [])
    if not isinstance(metrics, list):
        metrics = []

    total_views = len(metrics) or len(summary.get("input_files", []) or [])
    fused_view_count = int(summary.get("fused_view_count", 0) or 0)
    if not fused_view_count:
        fused_view_count = sum(
            bool(item.get("included_in_fusion"))
            for item in metrics
            if isinstance(item, dict)
        )
    try:
        minimum_required_views = int(
            registration_config.get("section_alignment_minimum_views", 3)
        )
    except (TypeError, ValueError):
        minimum_required_views = 3
    minimum_required_views = max(1, min(minimum_required_views, max(total_views, 1)))

    failures: list[str] = []
    insufficient: list[str] = []
    included_metrics = [
        item
        for item in metrics
        if isinstance(item, dict) and bool(item.get("included_in_fusion", True))
    ]
    failed_calibration = [
        item.get("file_name", "unknown")
        for item in included_metrics
        if item.get("calibration_quality_passed") is False
    ]
    if failed_calibration:
        failures.append("仍有工位的标定块质量未通过")
    if fused_view_count < minimum_required_views:
        insufficient.append(
            f"有效融合工位 {fused_view_count}/{minimum_required_views}，不足正式测量要求"
        )

    section_validation = summary.get("section_validation", {})
    if not isinstance(section_validation, dict):
        section_validation = {}
    section_validation_enabled = bool(
        section_validation.get(
            "enabled",
            registration_config.get("use_section_alignment_validation", False),
        )
    )
    section_status_counts: dict[str, int] = {}
    coverage_diagnostic_warnings: list[str] = []
    section_views = section_validation.get("views", [])
    if isinstance(section_views, list):
        for view in section_views:
            if not isinstance(view, dict) or not bool(view.get("included_in_fusion", True)):
                continue
            status = str(view.get("status", "not_checked"))
            section_status_counts[status] = section_status_counts.get(status, 0) + 1
            if coverage_preserving_fusion:
                if status == "mismatch":
                    coverage_diagnostic_warnings.append(
                        "多截面诊断发现共同覆盖区存在错层/重影差异，但互补视角仍保留参与覆盖融合"
                    )
                elif status in {"coverage_only", "insufficient_evidence", "not_checked"}:
                    coverage_diagnostic_warnings.append(
                        f"多截面诊断状态包含 {status}；该状态不作为删除互补视角的依据"
                    )
            elif status == "mismatch":
                failures.append("多截面验收发现仍参与融合的错层或重影工位")
            elif status in {"coverage_only", "insufficient_evidence", "not_checked"}:
                insufficient.append(f"多截面验收状态包含 {status}")
    elif section_validation_enabled and not section_views:
        if coverage_preserving_fusion:
            coverage_diagnostic_warnings.append("缺少多截面诊断明细")
        else:
            insufficient.append("缺少多截面验收明细")

    section_integrity = summary.get("section_integrity", {})
    if not isinstance(section_integrity, dict):
        section_integrity = {}
    integrity_status = str(section_integrity.get("status", "insufficient_evidence"))
    if integrity_status in {"cleanup_induced_gap", "mixed_structure_change"}:
        failures.append(f"截面清理造成结构变化（{integrity_status}）")
    elif integrity_status in {"source_coverage_gap_likely", "insufficient_evidence"}:
        insufficient.append(f"截面完整性证据不足（{integrity_status}）")

    # A nearby plane can look more continuous simply because it intersects a
    # different surface or a single station's unverified coverage.  In
    # coverage-first mode this remains an advisory warning; only the legacy
    # station-filter mode turns it into a formal-data gate.
    section_consensus = summary.get("section_consensus", {})
    if not isinstance(section_consensus, dict):
        section_consensus = {}
    section_offset_diagnostic = section_consensus.get("section_offset_diagnostic", {})
    if not isinstance(section_offset_diagnostic, dict):
        section_offset_diagnostic = {}
    section_offset_status = (
        str(section_offset_diagnostic.get("status", "")).strip().lower() or None
    )
    section_offset_gate_reasons = {
        "source_coverage_gap": (
            "截面偏移诊断发现候选轮廓缺少多工位共同支持，需重新采集覆盖缺口两侧的视角"
        ),
        "manual_review": (
            "截面偏移诊断存在未通过稳定性或平行双层复核的候选，需人工确认并重新处理"
        ),
        "auto_recommended": (
            "截面偏移诊断发现待复核候选，尚未重新处理并确认，当前 CAD 数值仅供参考"
        ),
    }
    if section_offset_status in section_offset_gate_reasons:
        if coverage_preserving_fusion:
            coverage_diagnostic_warnings.append(section_offset_gate_reasons[section_offset_status])
        else:
            insufficient.append(section_offset_gate_reasons[section_offset_status])

    multi_section_diagnostic = summary.get("multi_section_diagnostic", {})
    if not isinstance(multi_section_diagnostic, dict):
        multi_section_diagnostic = {}
    multi_section_status = (
        str(multi_section_diagnostic.get("diagnostic_status", "")).strip().lower()
        or None
    )
    multi_section_gate_reasons = {
        "neighbor_only_stable": (
            "多横截面诊断显示中心截面存在缺口，只有邻近截面较完整，当前 CAD 数值仅供参考"
        ),
        "persistent_gap": (
            "多横截面诊断显示多个截面都存在持续缺口，需重新采集覆盖目标区域的视角"
        ),
    }
    if multi_section_status in multi_section_gate_reasons:
        # Keep complementary source coverage in the fused cloud, but do not
        # allow a persistent section gap to produce a formal production result.
        insufficient.append(multi_section_gate_reasons[multi_section_status])

    section_points = int(summary.get("section_points", 0) or 0)
    if section_points < 100:
        insufficient.append(f"最终截面有效点数过少（{section_points}）")

    fusion_confidence = summary.get("section_consensus", {})
    if not isinstance(fusion_confidence, dict):
        fusion_confidence = {}
    fusion_confidence = fusion_confidence.get("fusion_confidence", {})
    if not isinstance(fusion_confidence, dict):
        fusion_confidence = {}
    confidence_score = fusion_confidence.get("confidence_score")
    try:
        confidence_value = float(confidence_score)
    except (TypeError, ValueError):
        confidence_value = None
    if confidence_value is not None and np.isfinite(confidence_value) and confidence_value < 0.35:
        insufficient.append(f"重叠区域融合置信度偏低（{confidence_value:.2f}）")

    if failures:
        status = "FAIL"
    elif insufficient:
        status = "INSUFFICIENT_DATA"
    else:
        status = "PASS"
    return {
        "status": status,
        "formal_measurement_allowed": status == "PASS",
        "total_views": total_views,
        "fused_view_count": fused_view_count,
        "minimum_required_views": minimum_required_views,
        "coverage_preserving_fusion": coverage_preserving_fusion,
        "section_validation_enabled": section_validation_enabled,
        "section_status_counts": section_status_counts,
        "section_integrity_status": integrity_status,
        "section_offset_diagnostic_status": section_offset_status,
        "section_offset_recommended_offset": section_offset_diagnostic.get(
            "recommended_offset"
        ),
        "multi_section_diagnostic_status": multi_section_status,
        "section_points": section_points,
        "fusion_confidence_score": confidence_value,
        "calibration_failed_views": failed_calibration,
        "failure_reasons": failures,
        "insufficient_data_reasons": insufficient,
        "coverage_diagnostic_warnings": coverage_diagnostic_warnings,
        "policy": (
            "覆盖优先模式：标定块质量合格视角全部参与融合；重叠区域和多截面差异只作诊断。"
            "标定块、有效点数或最终截面完整性不满足时仍显示 CAD 数值，但不得输出正式 PASS。"
            if coverage_preserving_fusion
            else "点云质量不满足时仍显示 CAD 数值，但不得输出正式 PASS。"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _section_profile_sha256(
    cloud: o3d.geometry.PointCloud,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> str:
    points = np.asarray(cloud.points, dtype=np.float64)
    horizontal = np.asarray(horizontal, dtype=np.float64).reshape(3)
    vertical = np.asarray(vertical, dtype=np.float64).reshape(3)
    finite_mask = np.isfinite(points).all(axis=1)
    profile = np.column_stack(
        (
            points[finite_mask] @ horizontal,
            points[finite_mask] @ vertical,
        )
    ).astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray([len(profile)], dtype="<u8").tobytes())
    digest.update(np.ascontiguousarray(profile).tobytes())
    return digest.hexdigest()


def _load_final_section_cloud(
    source_path: Path,
    angle_degrees: float,
    tilt_degrees: float,
    plane_offset: float,
) -> tuple[o3d.geometry.PointCloud, dict[str, list[float]]]:
    """Load the final section without changing its coordinates or point order."""
    source_cloud = o3d.io.read_point_cloud(str(source_path))
    points = np.asarray(source_cloud.points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise RuntimeError("最终截面点数不足，无法交给 CAD 偏差分析模块")
    finite_mask = np.isfinite(points).all(axis=1)
    if np.count_nonzero(finite_mask) < 3:
        raise RuntimeError("最终截面没有足够的有效坐标")

    normal, horizontal, vertical = vertical_plane_basis(angle_degrees, tilt_degrees)
    normal_xy = normal[:2]
    normal_xy_norm = float(np.linalg.norm(normal_xy))
    horizontal_xy = horizontal[:2]
    horizontal_xy_norm = float(np.linalg.norm(horizontal_xy))
    axis_xy = (
        normal_xy / normal_xy_norm
        if normal_xy_norm > 1e-12
        else np.array([1.0, 0.0], dtype=np.float64)
    )
    profile_axis_xy = (
        horizontal_xy / horizontal_xy_norm
        if horizontal_xy_norm > 1e-12
        else np.array([1.0, 0.0], dtype=np.float64)
    )

    basis = {
        "normal_xyz": normal.round(12).tolist(),
        "horizontal_xyz": horizontal.round(12).tolist(),
        "vertical_xyz": vertical.round(12).tolist(),
        "axis_xy": axis_xy.round(12).tolist(),
        "profile_axis_xy": profile_axis_xy.round(12).tolist(),
        "coordinate_origin_xyz": [0.0, 0.0, 0.0],
        "source_plane_offset": float(plane_offset),
    }
    return source_cloud, basis


def export_final_section(
    result: PipelineResult,
    downstream_root: Path = DEFAULT_DOWNSTREAM_ROOT,
    downstream_url: str = DEFAULT_DOWNSTREAM_URL,
) -> DownstreamHandoff:
    source_path = result.section_cloud_path
    if source_path is None or not source_path.is_file():
        raise RuntimeError("当前结果没有可用的最终截面点云，请先重新处理并生成截面")
    downstream_root = Path(downstream_root).resolve()
    if not (downstream_root / "media_web_app.py").is_file():
        raise RuntimeError(f"未找到本项目的 CAD 偏差分析模块：{downstream_root}")

    summary = _load_summary(result)
    section_config = summary.get("config", {}).get("section", {})
    try:
        angle_degrees = float(section_config["angle_degrees"])
        tilt_degrees = float(section_config.get("tilt_degrees", 0.0))
        plane_offset = float(section_config["plane_offset"])
        thickness = float(section_config["thickness"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("当前运行摘要缺少完整的三维截面参数") from exc

    final_section_cloud, basis = _load_final_section_cloud(
        source_path,
        angle_degrees,
        tilt_degrees,
        plane_offset,
    )
    section_points = len(final_section_cloud.points)
    section_profile_sha256 = _section_profile_sha256(
        final_section_cloud,
        basis["horizontal_xyz"],
        basis["vertical_xyz"],
    )
    source_section_sha256 = _sha256(source_path)
    token = uuid.uuid4().hex
    preview_root = downstream_root / "web_runs" / "pointcloud_previews"
    package_directory = preview_root / token
    section_directory = package_directory / "media_processing" / "vertical_section"
    section_directory.mkdir(parents=True, exist_ok=False)

    target_cloud_path = section_directory / "vertical_section_points.ply"
    shutil.copy2(source_path, target_cloud_path)

    analysis_section_sha256 = _sha256(target_cloud_path)
    if analysis_section_sha256 != source_section_sha256:
        raise RuntimeError("CAD 分析交接过程中截面文件发生变化，已停止后续分析")

    target_section_image = section_directory / "vertical_section_only.png"
    shutil.copy2(result.section_image_path, target_section_image)
    shutil.copy2(result.section_image_path, package_directory / "pointcloud_side_preview.png")

    compute = summary.get("compute", {})
    active_backend = compute.get("active_backend", result.compute_backend)
    gpu_capabilities = compute.get("gpu_capabilities", {})
    processing_report = {
        "source": {
            "program": "26xV01Q",
            "group_name": result.group_name,
            "run_directory": str(result.run_directory.resolve()),
            "run_summary": str(result.summary_path.resolve()),
            "source_section_cloud": str(source_path.resolve()),
            "source_merged_cloud": summary.get("section_source_cloud")
            or summary.get("output_files", {}).get("merged_cloud"),
            "source_section_sha256": source_section_sha256,
            "analysis_section_sha256": analysis_section_sha256,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        },
        "processing": {
            "mode": EXTERNAL_SECTION_MODE,
            "accelerator": {
                "backend": active_backend,
                "used": active_backend == "gpu",
                "gpu": {
                    "device_name": gpu_capabilities.get("device_name") or result.gpu_device_name,
                },
            },
            "timing_seconds": {},
        },
        "vertical_section": {
            "coordinate_system": "26xV01Q_final_section_xyz",
            "pointcloud_transform_applied": False,
            "point_order_preserved": True,
            "coordinates_preserved_from_source": True,
            "plane_center_xy": [0.0, 0.0],
            "section_points": section_points,
            "thickness": thickness,
            "source_angle_degrees": angle_degrees,
            "source_tilt_degrees": tilt_degrees,
            "source_plane_offset": plane_offset,
            **{key: value for key, value in basis.items() if key != "source_plane_offset"},
            "interface_image": str(target_section_image.resolve()),
            "section_cloud": str(target_cloud_path.resolve()),
            "source_section_sha256": source_section_sha256,
            "analysis_section_sha256": analysis_section_sha256,
            "section_profile_sha256": section_profile_sha256,
            "section_source": summary.get("section_source", "merged_fused_cloud"),
            "integrity_diagnostic": summary.get("section_integrity", {}),
        },
        "quality": _derive_processing_quality(summary),
    }
    processing_report_path = package_directory / "media_processing" / "processing_report.json"
    processing_report_path.write_text(
        json.dumps(processing_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "token": token,
        "files": [],
        "analysis_pointcloud": str(target_cloud_path.relative_to(package_directory)),
        "mode": EXTERNAL_SECTION_MODE,
        "pointcloud_transform_applied": False,
        "point_order_preserved": True,
        "coordinates_preserved_from_source": True,
        "section_points": section_points,
        "analysis_pointcloud_sha256": analysis_section_sha256,
        "section_profile_sha256": section_profile_sha256,
        "section_integrity": summary.get("section_integrity", {}),
        "quality": processing_report["quality"],
        "source_section_cloud": str(source_path.resolve()),
        "source_merged_cloud": processing_report["source"].get("source_merged_cloud"),
        "section_source": processing_report["vertical_section"].get("section_source"),
        "source_file_count": len(summary.get("input_files", [])),
        "processing": {
            "total_seconds": None,
            "registration_seconds": None,
            "fusion_seconds": None,
            "fusion_backend": active_backend,
            "gpu_used": active_backend == "gpu",
            "gpu_device": gpu_capabilities.get("device_name") or result.gpu_device_name,
        },
        "source": processing_report["source"],
    }
    (package_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    analysis_url = f"{downstream_url.rstrip('/')}?section_token={token}"
    return DownstreamHandoff(
        token=token,
        analysis_url=analysis_url,
        package_directory=package_directory,
        section_cloud_path=target_cloud_path,
        processing_report_path=processing_report_path,
        section_points=section_points,
    )


def prepare_final_section_for_cad(result: PipelineResult) -> DownstreamHandoff:
    return export_final_section(result)


def downstream_server_ready(downstream_url: str = DEFAULT_DOWNSTREAM_URL) -> bool:
    try:
        with urlopen(f"{downstream_url.rstrip('/')}/health", timeout=1.5) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def ensure_downstream_server(
    downstream_root: Path = DEFAULT_DOWNSTREAM_ROOT,
    downstream_url: str = DEFAULT_DOWNSTREAM_URL,
    timeout_seconds: float = 12.0,
) -> bool:
    if downstream_server_ready(downstream_url):
        return True
    application = Path(downstream_root).resolve() / "media_web_app.py"
    if not application.is_file():
        return False
    configured_python = os.environ.get("POINTCLOUD_PYTHON") or os.environ.get("26XV01Q_PYTHON")
    python_executable = Path(configured_python) if configured_python else Path(sys.executable)
    if not python_executable.is_file():
        python_executable = Path(sys.executable)
    environment = os.environ.copy()
    environment["PORT"] = "5000"
    environment["FLASK_DEBUG"] = "0"
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen(
        [str(python_executable), str(application)],
        cwd=str(application.parent),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if downstream_server_ready(downstream_url):
            return True
        time.sleep(0.35)
    return False
