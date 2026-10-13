"""UI-independent desktop tasks using the existing processing algorithms."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import zipfile
from collections import OrderedDict
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..models import Bounds2D, Bounds3D, CalibrationConfig, DenoiseConfig, ProcessingConfig, RegistrationConfig, SectionConfig
from ..pipeline import (
    CloudDenoiseStats, LevelingResult, PipelineResult, RegistrationMetric,
    _denoise_registered_views, _transform_cloud, build_preview,
    build_surface_preview, denoise_cloud,
    normal_plane_from_xyz_region,
    process_group, read_point_cloud, select_reference_file_with_leveling,
)
from ..profiles import apply_family_profile, classify_group_family
from ..gpu import resolve_compute_backend

ROOT = Path(__file__).resolve().parents[2]


class Cancelled(RuntimeError):
    pass


def file_signature(paths):
    return tuple((str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths)


def acquisition_signature(files):
    return file_signature(files)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def config_from_dict(payload):
    data = copy.deepcopy(payload.get("config", payload))
    allowed = {field.name for field in fields(ProcessingConfig)}
    if set(data) - allowed:
        raise ValueError("配置中包含未知字段")
    for name, cls in (("denoise", DenoiseConfig), ("calibration", CalibrationConfig),
                      ("registration", RegistrationConfig), ("section", SectionConfig)):
        values = data.get(name, {})
        if name == "calibration" and isinstance(values.get("bounds"), dict):
            values["bounds"] = Bounds3D(**values["bounds"])
        if name == "section" and isinstance(values.get("target_region"), dict):
            values["target_region"] = Bounds2D(**values["target_region"])
        data[name] = cls(**values)
    config = ProcessingConfig(**data)
    # Dataclass validation is complemented by rejecting NaN/Infinity.
    json.dumps(config.as_dict(), allow_nan=False)
    config.validate()
    return config


def default_config(group):
    config = apply_family_profile(ProcessingConfig(), classify_group_family(group))
    # Neighbor planes remain diagnostic; a picked center is never replaced.
    config.section.auto_offset_enabled = False
    config.registration.use_six_station_pose_prior = False
    config.registration.use_global_registration = True
    config.registration.station_pose_corrections = {}
    return config


def processing_quality(summary):
    from ..downstream_bridge import _derive_processing_quality
    quality = copy.deepcopy(_derive_processing_quality(summary))
    quality["reasons"] = [
        *quality.get("failure_reasons", []), *quality.get("insufficient_data_reasons", []),
        *quality.get("coverage_diagnostic_warnings", []),
    ]
    acquisition = summary.get("desktop_acquisition", {})
    registration = summary.get("config", {}).get("registration", {})
    automatic = (
        acquisition.get("registration_mode") == "automatic_calibration"
        and acquisition.get("input_verified") is True
        and registration.get("use_six_station_pose_prior") is False
        and registration.get("use_global_registration") is True
    )
    if not (automatic or acquisition.get("confirmed")):
        quality["formal_measurement_allowed"] = False
        if quality["status"] != "FAIL":
            quality["status"] = "INSUFFICIENT_DATA"
        reason = "历史结果缺少采集确认或自动配准输入校验记录，仅供参考"
        quality["insufficient_data_reasons"] = [*quality.get("insufficient_data_reasons", []), reason]
        quality["reasons"].append(reason)
    return quality


def fit_selection(points, section, bounds):
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        raise ValueError("框选范围无效")
    x0, x1 = sorted(bounds[:2])
    y0, y1 = sorted(bounds[2:])
    region = Bounds2D(x0, x1, y0, y1)
    region.validate()
    points = np.asarray(points)
    selected = points[
        np.isfinite(points).all(axis=1) & (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    ]
    if len(selected) < 30:
        raise ValueError("选区有效点不足 30 个")
    angle, tilt, offset, confidence = normal_plane_from_xyz_region(selected)
    updated = copy.deepcopy(section)
    updated.angle_degrees, updated.tilt_degrees, updated.plane_offset = angle, tilt, offset
    updated.target_region = region
    updated.auto_offset_enabled = False
    updated.validate()
    return updated, len(selected), confidence


def result_from_summary(path):
    path = Path(path).resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    output = summary["output_files"]

    def output_path(key, required=False):
        value = output.get(key)
        if not value:
            if required:
                raise ValueError(f"结果缺少 {key}")
            return None
        original = Path(value)
        local = path.parent / original.name
        return local if local.is_file() else original

    def from_fields(cls, value):
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})

    level = summary.get("leveling")
    if isinstance(level, dict) and "transformation" in level:
        level = from_fields(LevelingResult, level)
        level.transformation = np.asarray(level.transformation, dtype=float)
    else:
        level = None
    result = PipelineResult(
        group_name=summary["group_name"], run_directory=path.parent,
        merged_cloud_path=output_path("merged_cloud", True),
        raw_section_cloud_path=output_path("raw_section_cloud"),
        section_cloud_path=output_path("section_cloud"),
        raw_section_image_path=output_path("raw_section_image", True),
        section_image_path=output_path("section_image", True),
        top_preview_path=output_path("top_preview", True), summary_path=path,
        merged_points=int(summary.get("merged_points", 0)),
        section_points=int(summary.get("section_points", 0)),
        compute_backend=summary.get("compute", {}).get("active_backend", "cpu"),
        gpu_device_name=None,
        denoise_stats=[from_fields(CloudDenoiseStats, row) for row in summary.get("denoise_stats", [])],
        registration_metrics=[from_fields(RegistrationMetric, row) for row in summary.get("registration_metrics", [])],
        leveling=level, alignment_audit_path=output_path("alignment_audit"),
        section_alignment_audit_path=output_path("section_alignment_audit"),
        section_coverage_audit_path=output_path("section_coverage_audit"),
        multi_section_audit_path=output_path("multi_section_audit"),
        section_raw_points=int(summary.get("section_raw_points", 0)),
        section_source_points=int(summary.get("section_source_points", 0)),
        section_integrity=summary.get("section_integrity"),
    )
    return result


class Backend:
    """Small bounded cache; all calls are serialized by the desktop worker."""

    def __init__(self):
        self.cache = OrderedDict()

    def cached(self, key, create):
        if key not in self.cache:
            self.cache[key] = create()
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def preview(self, files, config, level, denoise, progress, result=None, all_views=False,
                *, protect_calibration=True):
        voxel, maximum = ((.20, 50_000), (.10, 100_000), (.06, 200_000), (.01, 350_000))[level]
        paths = [result.merged_cloud_path] if result else files
        stamp = file_signature(paths)
        # Section editing must not invalidate cloud loading/leveling.
        key = ("preview", stamp, json.dumps(config.denoise.__dict__, sort_keys=True),
               json.dumps(config.as_dict()["calibration"], sort_keys=True),
               config.registration.reference_vertex_count, config.compute_backend,
               level, denoise, bool(result), bool(all_views), bool(protect_calibration))

        def load():
            progress("读取点云")
            if result:
                reference, leveling = paths[0], None
                preview_files = paths
            else:
                reference, leveling = select_reference_file_with_leveling(
                    files, config.registration.reference_vertex_count, config.denoise,
                    config.calibration, config.compute_backend,
                )
                preview_files = files if all_views else [reference]
            progress("生成点云预览")
            # The exemption is only for calibration editing; inspection still
            # uses the same reference/leveling but filters points inside the ROI.
            data = build_preview(preview_files, voxel, maximum, config.compute_backend,
                                 config.denoise, denoise and not result, leveling,
                                 protected_bounds=(
                                     config.calibration.bounds.min_bound,
                                     config.calibration.bounds.max_bound,
                                 ) if protect_calibration and config.calibration.use_roi else None)
            return reference, data
        return self.cached(key, load)

    def single_preview(self, path, config, level, denoise, progress, result=None):
        path = Path(path)
        voxel, maximum = ((.20, 50_000), (.10, 100_000), (.06, 200_000), (.01, 350_000))[level]
        result_stamp = file_signature([result.summary_path]) if result else None
        key = ("single-preview", file_signature([path]), result_stamp, level, bool(denoise),
               json.dumps(config.denoise.__dict__, sort_keys=True), config.compute_backend)

        def load():
            progress(f"读取 {path.name}")
            # Inspection must filter ROI noise too; registration protects its
            # calibration samples separately in the processing pipeline.
            if denoise and result is not None:
                summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
                source_paths = [Path(item) for item in summary["input_files"]]
                selected_index = next(
                    (index for index, source_path in enumerate(source_paths)
                     if source_path.resolve() == path.resolve()),
                    None,
                )
                if selected_index is not None:
                    progress("应用已配准的跨视角重叠支持去噪")
                    transformations = [
                        np.asarray(metric["transformation"], dtype=np.float64)
                        for metric in summary["registration_metrics"]
                    ]
                    included_views = [
                        bool(metric.get("included_in_fusion", True))
                        for metric in summary["registration_metrics"]
                    ]
                    cloud_key = (
                        "registered-inspection-clouds",
                        tuple(file_signature(source_paths)),
                        json.dumps(config.denoise.__dict__, sort_keys=True),
                        tuple(tuple(np.asarray(matrix).round(12).flat) for matrix in transformations),
                        tuple(included_views),
                        json.dumps(summary.get("leveling", {}).get("transformation"), sort_keys=True),
                    )

                    def load_registered_clouds():
                        clouds = [
                            denoise_cloud(
                                read_point_cloud(source_path),
                                config.denoise,
                                source_path.name,
                                "cpu",
                            )[0]
                            for source_path in source_paths
                        ]
                        return _denoise_registered_views(
                            clouds,
                            transformations,
                            np.asarray(summary["leveling"]["transformation"], dtype=np.float64),
                            included_views,
                            config.denoise.voxel_size,
                            None,
                            str(summary.get("group_name", "")),
                        )[0]

                    filtered = self.cached(cloud_key, load_registered_clouds)
                    aligned_cloud = _transform_cloud(
                        filtered[selected_index],
                        np.asarray(summary["leveling"]["transformation"], dtype=np.float64)
                        @ transformations[selected_index],
                        "cpu",
                    )
                    return build_preview(
                        [path], voxel, maximum,
                        resolve_compute_backend(config.compute_backend),
                        config.denoise, False, None,
                        source_clouds=[aligned_cloud],
                    )
            return build_preview([path], voxel, maximum, resolve_compute_backend(config.compute_backend),
                                 config.denoise, bool(denoise), None)

        return self.cached(key, load)

    def surface(self, points, level, backend):
        digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
        return self.cached(("surface", digest, level, backend),
                           lambda: build_surface_preview(points, (.2, .1, .06, .01)[level],
                                                         100_000, 300_000, backend))

    def process(self, files, config, progress, output_root, *,
                pose_reference=None, capture_pose_reference=False):
        desktop_started_at = perf_counter()
        if len(files) not in (5, 6, 7, 8):
            raise ValueError("每组应包含 5 至 8 个 PLY")
        if config.section.target_region is None:
            raise ValueError("请先框选主轴区域")
        config = copy.deepcopy(config)
        # Desktop captures are registered from fixture geometry, never station labels.
        config.registration.use_six_station_pose_prior = False
        config.registration.use_global_registration = True
        config.registration.station_pose_corrections = {}
        config.validate()
        tracked = list(files)
        station_metadata = Path(files[0]).parent / "station_order.json"
        track_stations = pose_reference is not None or capture_pose_reference
        station_stamp = station_metadata.read_bytes() if track_stations and station_metadata.exists() else None
        if track_stations and station_metadata.exists():
            tracked.append(station_metadata)
        snapshot = file_signature(tracked)
        progress("记录输入文件指纹")
        hash_started_at = perf_counter()
        worker_count = min(4, max(1, len(files)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            hashes = list(executor.map(sha256, files))
        input_hashing_seconds = perf_counter() - hash_started_at
        fingerprints = {path.name: digest for path, digest in zip(files, hashes, strict=True)}
        if snapshot != file_signature(tracked):
            raise RuntimeError("输入文件在读取指纹期间发生变化")
        progress("根据标定块自动计算各视角位姿")
        processing_started_at = perf_counter()
        options = {}
        if pose_reference is not None or capture_pose_reference:
            options = dict(pose_reference=pose_reference,
                           capture_pose_reference=capture_pose_reference)
        result = process_group(files[0].parent.name, files, config, output_root, progress, **options)
        processing_seconds = perf_counter() - processing_started_at
        if snapshot != file_signature(tracked):
            raise RuntimeError("处理期间输入文件发生变化，本次结果不得测量")
        if track_stations and station_stamp != (
                station_metadata.read_bytes() if station_metadata.exists() else None):
            raise RuntimeError("处理期间工位编号发生变化，本次结果不得测量")
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        summary["desktop_acquisition"] = {
            "registration_mode": "automatic_calibration",
            "input_verified": True, "input_sha256": fingerprints,
            "verified_at": datetime.now().astimezone().isoformat(),
            "timing": {
                "input_hashing_seconds": round(input_hashing_seconds, 3),
                "processing_seconds": round(processing_seconds, 3),
                "total_seconds": round(perf_counter() - desktop_started_at, 3),
                "hash_workers": worker_count,
            },
        }
        result.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def alignment(self, result, progress):
        stamp = file_signature([result.summary_path])
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        paths = [Path(p) for p in summary["input_files"]]
        stamp += file_signature(paths)

        def load():
            items = []
            lookup = {p.name: p for p in paths}
            level = np.asarray((summary.get("leveling") or {}).get("transformation", np.eye(4)))
            denoise = config_from_dict(summary["config"]).denoise
            for index, metric in enumerate(summary["registration_metrics"]):
                path = lookup[metric["file_name"]]
                progress(f"读取视角 {index + 1}")
                cloud, _ = denoise_cloud(read_point_cloud(path), denoise, path.name, "cpu")
                points = np.asarray(cloud.points)
                points = points[np.isfinite(points).all(axis=1)]
                if len(points) > 18000:
                    points = points[np.linspace(0, len(points) - 1, 18000, dtype=int)]
                transform = level @ np.asarray(metric["transformation"])
                points = points @ transform[:3, :3].T + transform[:3, 3]
                station = {"reference": "0°", "minus10": "-10°", "minus28": "-28°",
                           "minus50": "-50°", "plus37": "+37°", "plus47": "+47°"}.get(
                               metric.get("pose_prior_label"), path.stem[-8:])
                label = f"第{index + 1}部分 ({station})"
                items.append((label, points, metric.get("included_in_fusion", True)))
            return items
        return self.cached(("alignment", stamp), load)

    def cad_sources(self, result, report, progress):
        """Trace frozen section samples to unchanged inputs, without moving points."""
        from scipy.spatial import cKDTree
        from ..registration_reference import valid_rigid

        summary_path = Path(result.summary_path)
        section_path = Path(report["source_pointcloud"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = {Path(p).name: Path(p) for p in summary["input_files"]}
        tracked = [summary_path, section_path, *paths.values()]

        def snapshot():
            stamps = []
            for path in tracked:
                try:
                    stamps.extend(file_signature([path]))
                except OSError:
                    stamps.append((str(path.resolve()), None, None))
            return tuple(stamps)

        stamp = snapshot()

        def load():
            points = np.asarray(read_point_cloud(section_path).points)
            if not np.isfinite(points).all():
                raise ValueError("最终截面包含非有限坐标，不能追溯来源")
            owners = np.full(len(points), -1, dtype=np.int32)
            fingerprints = (summary.get("desktop_acquisition") or {}).get("input_sha256", {})
            level = np.asarray((summary.get("leveling") or {}).get("transformation", np.eye(4)))
            metrics = summary["registration_metrics"]
            groups = []
            for index, metric in enumerate(metrics):
                name = metric["file_name"]
                path = paths.get(name)
                progress(f"追溯截面来源 {index + 1}/{len(metrics)} · {name}")
                if not metric.get("included_in_fusion", True) or path is None or not path.is_file():
                    continue
                expected_hash = fingerprints.get(name)
                if not expected_hash or sha256(path) != expected_hash:
                    continue
                transform = level @ np.asarray(metric["transformation"])
                if not valid_rigid(transform):
                    continue
                original = (points - transform[:3, 3]) @ transform[:3, :3]
                raw = np.asarray(read_point_cloud(path).points)
                raw = raw[np.isfinite(raw).all(axis=1)]
                if not len(raw):
                    continue
                # Only numerical roundoff is allowed, not ordinary nearest-surface matching.
                matches = cKDTree(raw).query(original)[0] <= 1e-6
                ambiguous = matches & (owners != -1)
                owners[matches & (owners == -1)] = index
                owners[ambiguous] = -2
            for index, metric in enumerate(metrics):
                groups.append((metric["file_name"], np.flatnonzero(owners == index)))
            for code, label in ((-2, "多源重合"), (-1, "未追溯")):
                indices = np.flatnonzero(owners == code)
                if len(indices):
                    groups.append((label, indices))
            if snapshot() != stamp:
                raise RuntimeError("来源文件在追溯期间发生变化，请重新加载结果")
            return groups

        return self.cached(("cad-sources", stamp), load)

    def analyze(self, result, dxf, selection, step, progress):
        from ..downstream_bridge import prepare_final_section_for_cad
        from ..cad_geometry import load_cad_entities, validate_cad_path_selection
        from media_analysis_core import analyze

        entities = load_cad_entities(dxf)
        selection = validate_cad_path_selection({e.index for e in entities}, selection, entities)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("CAD 采样间距必须大于 0")
        progress("冻结最终截面")
        handoff = prepare_final_section_for_cad(result)
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        processing = json.loads(handoff.processing_report_path.read_text(encoding="utf-8"))
        processing["quality"] = processing_quality(summary)
        handoff.processing_report_path.write_text(
            json.dumps(processing, ensure_ascii=False, indent=2), encoding="utf-8")
        directory = result.run_directory / "native_cad" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory.mkdir(parents=True)
        frozen_dxf = directory / dxf.name
        shutil.copy2(dxf, frozen_dxf)
        inputs = directory / "inputs"
        section_directory = inputs / "vertical_section"
        section_directory.mkdir(parents=True)
        frozen_section = section_directory / handoff.section_cloud_path.name
        shutil.copy2(handoff.section_cloud_path, frozen_section)
        shutil.copy2(handoff.processing_report_path, inputs / "processing_report.json")
        progress("CAD 配准与偏差测量")
        report = analyze(frozen_section, frozen_dxf, directory, step=step,
                         registration_paths=selection["registration_paths"],
                         measurement_paths=selection["measurement_paths"])
        report["desktop_input"] = {"dxf_sha256": sha256(frozen_dxf), "step": step,
                                   "selection": selection}
        report_path = directory / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        progress("测量完成")
        return report

    def export(self, result, report, destination, formal, progress):
        if formal:
            quality = (report or {}).get("measurement_quality", {})
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            if (not processing_quality(summary).get("formal_measurement_allowed")
                    or quality.get("status") != "PASS"):
                raise ValueError("当前质量未通过，只能导出诊断包")
        destination = Path(destination)
        candidates = list(result.run_directory.rglob("*"))
        if formal:
            selected_directory = Path(report["source_dxf"]).resolve().parent
            candidates = [p for p in candidates if p.parent == result.run_directory
                          or p.resolve().is_relative_to(selected_directory)]
        if destination.resolve().is_relative_to(result.run_directory.resolve()):
            raise ValueError("请将导出包保存到结果目录之外")
        temporary = destination.with_name(destination.name + ".partial")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for path in candidates:
                    if path.is_file():
                        progress(f"导出 {path.name}")
                        archive.write(path, path.relative_to(result.run_directory))
                archive.writestr("export_kind.json", json.dumps(
                    {"kind": "formal_measurement" if formal else "diagnostic",
                     "measurement_report": (report or {}).get("source_dxf")}, ensure_ascii=False))
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
