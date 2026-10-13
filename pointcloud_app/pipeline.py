from __future__ import annotations

import copy
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib
import numpy as np
import open3d as o3d
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_available_font_names = {font.name for font in font_manager.fontManager.ttflist}
for _font_name in ("Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC"):
    if _font_name in _available_font_names:
        plt.rcParams["font.sans-serif"] = [_font_name]
        break
plt.rcParams["axes.unicode_minus"] = False

from .gpu import (
    get_gpu_capabilities,
    gpu_bounds_mask,
    gpu_fuse_clouds,
    gpu_radius_inlier_mask,
    gpu_refine_icp,
    gpu_runtime_summary,
    gpu_statistical_inlier_mask,
    gpu_transform_points,
    gpu_vertical_section_mask,
    gpu_voxel_down_sample,
    resolve_compute_backend,
)
from .models import CalibrationConfig, DenoiseConfig, ProcessingConfig, RegistrationConfig, SectionConfig
from .overlap_registration import fixture_component_mask, refine_multiview_overlap


VERTICAL_SECTION_IMPLEMENTATION_VERSION = 36
SECTION_ALIGNMENT_MINIMUM_OVERLAP_RATIO = 0.30
SECTION_CANDIDATE_MINIMUM_PROFILE_POINTS = 64
SECTION_CANDIDATE_MINIMUM_OVERLAP_POINTS = 8
SECTION_ALIGNMENT_MINIMUM_LOCAL_MATCH_RATIO = 0.70
SECTION_ALIGNMENT_MINIMUM_BAD_BIN_RATIO = 0.30
SECTION_ALIGNMENT_MINIMUM_BAD_BIN_RUN = 3
SECTION_ALIGNMENT_STRONG_MISMATCH_COVERAGE_RATIO = 0.65
SECTION_ALIGNMENT_PROFILE_NEIGHBOR_COUNT = 16
SECTION_ALIGNMENT_MINIMUM_CONTOUR_MISMATCH_RATIO = 0.45
SECTION_ALIGNMENT_MINIMUM_CONTOUR_MISMATCH_BINS = 3
SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_RATIO = 0.30
SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_EXCESS = 0.12
SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_BINS = 10
SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_SUPPORT = 2
SECTION_ALIGNMENT_CONTOUR_CLUSTER_GAP_MINIMUM = 0.05
SECTION_ALIGNMENT_CONTOUR_CLUSTER_GAP_MAXIMUM = 0.10
SECTION_PARALLEL_TRACK_BIN_SIZE = 0.05
SECTION_PARALLEL_TRACK_CLUSTER_GAP = 0.06
SECTION_PARALLEL_TRACK_MINIMUM_RUN = 3
SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_MISMATCH_VIEWS = 2
SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_OVERLAP_RATIO = 0.60
SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_CONTOUR_BINS = 10
SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_CONTOUR_RATIO = 0.45
SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_PEAK_CONTOUR_RATIO = 0.60
# A peer cluster that only covers a local part of the trusted profile is a
# likely occlusion/complementary view, not a complete replacement contour.
SECTION_ALIGNMENT_LOCAL_COVERAGE_MAX_TRUSTED_SPAN_RATIO = 0.82
SECTION_ALIGNMENT_LOCAL_COVERAGE_MIN_COMMON_SPAN = 1.00
SECTION_CONSENSUS_MINIMUM_SOURCE_VIEWS = 3
SECTION_CONSENSUS_SOFT_REMOVAL_RATIO = 0.10
SECTION_CONSENSUS_HARD_REMOVAL_RATIO = 0.15
SECTION_CONSENSUS_MIN_SHARED_REMOVAL_RATIO = 0.18
SECTION_CONSENSUS_MINIMUM_EVALUATED_BINS = 4
SECTION_CONSENSUS_MINIMUM_EVIDENCE_POINTS = 80
SECTION_ALIGNMENT_PROBE_PERSISTENCE_MINIMUM = 2
SECTION_CONSENSUS_MINIMUM_ISOLATED_TRACK_BINS = 6
SECTION_CONSENSUS_ISOLATED_TRACK_DISTANCE = 0.12
SECTION_CONSENSUS_MAX_TRACK_GAP_BINS = 1
SECTION_INTEGRITY_MINIMUM_POINTS = 32
SECTION_INTEGRITY_MINIMUM_COMPONENT_SPAN = 0.60
SECTION_INTEGRITY_MINIMUM_CELL_SIZE = 0.05
SECTION_INTEGRITY_MAXIMUM_CELL_SIZE = 0.30
SECTION_INTEGRITY_MINIMUM_COMPONENT_RETENTION = 0.75
SECTION_INTEGRITY_MAXIMUM_LOST_CELL_RATIO = 0.25
# Offset recommendations are deliberately conservative.  A nearby slice is only
# useful for formal measurement when the newly exposed contour is repeatable in
# more than one source view; a single-view edge is treated as an occlusion or a
# different surface until an operator verifies it.
SECTION_OFFSET_MINIMUM_SOURCE_VIEWS = 3
SECTION_OFFSET_MINIMUM_NEW_MULTIVIEW_CELLS = 3
SECTION_OFFSET_MINIMUM_NEW_MULTIVIEW_RATIO = 0.50
SECTION_OFFSET_MINIMUM_STABLE_NEIGHBORS = 2
SECTION_OFFSET_MAX_LAYER_RATIO = 0.24
SECTION_OFFSET_MAX_LAYER_GROWTH = 0.10
# Recover only measured samples that were collapsed by the coarse single-view
# voxel pass.  The current retained cloud remains the geometric gate.
SECTION_DENSITY_RECOVERY_DISTANCE_FACTOR = 0.75


SIX_STATION_POSE_SEQUENCE = (
    'minus50',
    'minus28',
    'minus10',
    'reference',
    'plus37',
    'plus47',
)
# Fixture-anchored generic registration needs a stable traversal independent
# of display filenames; this order keeps all six views aligned in current scans.
SIX_STATION_GENERIC_REGISTRATION_ORDER = (
    'reference', 'minus28', 'minus50', 'plus47', 'plus37', 'minus10',
)
FIVE_STATION_POSE_SEQUENCE = (
    'minus50',
    'minus28',
    'minus10',
    'reference',
    'plus37',
)
SIX_STATION_OUTPUT_ORDER = (
    'reference',
    'minus10',
    'minus28',
    'plus37',
    'plus47',
    'minus50',
)
SIX_STATION_POSE_TRANSFORMS = {
    'reference': np.eye(4, dtype=np.float64),
    'plus37': np.array(
        [
            [0.997771349906, -0.066297949235, 0.00752623441, -1.072133685296],
            [0.057388100456, 0.795147685546, -0.603694097778, 0.96480230969],
            [0.034039217693, 0.60278069828, 0.79718044803, -2.075810405048],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    'minus10': np.array(
        [
            [0.999974646228, -0.002706583044, 0.006581001503, -0.247339173408],
            [0.0014566621, 0.983113923571, 0.182989209964, -1.927068278241],
            [-0.006965149069, -0.182974971757, 0.983092860301, 3.3144517693],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    'plus47': np.array(
        [
            [0.997540900338, -0.068878970251, 0.012965396246, 3.310575834962],
            [0.056476483979, 0.680383303122, -0.730674721886, 4.213869905505],
            [0.041506707502, 0.729610392606, 0.682600358366, 0.150352788099],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    'minus50': np.array(
        [
            [0.99995579568, -0.001418228511, -0.009291955254, -2.461843474552],
            [0.008078878487, 0.634968603805, 0.772495853542, 0.306290398071],
            [0.004804515589, -0.772536714037, 0.634951947794, 0.35152240194],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    'minus28': np.array(
        [
            [0.999755227098, -0.021113877102, -0.006609416824, -2.111915237536],
            [0.021764141798, 0.88489591861, 0.465280209847, 1.589279064906],
            [-0.003975224225, -0.465310172972, 0.885138840567, 4.517843999295],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}


ProgressCallback = Callable[[str], None]


class ProcessingTiming:
    """Record coarse processing stages without changing the pipeline data flow."""

    def __init__(self) -> None:
        self.started_at = perf_counter()
        self.stage_started_at = self.started_at
        self.current_stage: str | None = None
        self.stages: dict[str, float] = {}

    def start(self, stage: str) -> None:
        self.current_stage = stage
        self.stage_started_at = perf_counter()

    def checkpoint(self, stage: str) -> None:
        now = perf_counter()
        if self.current_stage is not None:
            self.stages[self.current_stage] = (
                self.stages.get(self.current_stage, 0.0)
                + now
                - self.stage_started_at
            )
        self.current_stage = stage
        self.stage_started_at = now

    def finish(self) -> dict[str, object]:
        now = perf_counter()
        if self.current_stage is not None:
            self.stages[self.current_stage] = (
                self.stages.get(self.current_stage, 0.0)
                + now
                - self.stage_started_at
            )
            self.current_stage = None
        return {
            "total_seconds": round(max(0.0, now - self.started_at), 3),
            "stages": {
                name: round(max(0.0, seconds), 3)
                for name, seconds in self.stages.items()
            },
            "measured_with": "perf_counter",
        }


def _processing_stage_for_message(message: str) -> str:
    text = str(message).lower()
    if any(token in text for token in ("去噪", "denoise")):
        return "single_view_denoise"
    if any(token in text for token in ("调平", "level")):
        return "leveling"
    if any(token in text for token in ("保存", "审计图", "render", "输出")):
        return "output_rendering"
    if ("融合" in text or "fusion" in text) and "已融合" not in text:
        return "fusion"
    if "截面" in text or "section" in text:
        return "section"
    if any(token in text for token in (
        "配准", "registration", "icp", "fpfh", "位姿", "姿态", "标定块",
        "重叠修正", "overlap", "闭环", "validation", "消歧",
    )):
        return "registration"
    return "preparation"


def _timed_progress_callback(
    callback: ProgressCallback | None,
    timing: ProcessingTiming,
) -> ProgressCallback:
    def emit(message: str) -> None:
        timing.checkpoint(_processing_stage_for_message(message))
        _emit(callback, message)

    return emit


@dataclass(frozen=True)
class PlyHeader:
    format_name: str
    data_offset: int
    vertex_count: int
    vertex_dtype: np.dtype


@dataclass
class CloudDenoiseStats:
    file_name: str
    input_points: int
    after_voxel_points: int
    after_statistical_points: int
    output_points: int
    after_component_points: int = 0
    after_surface_support_points: int = 0


@dataclass
class RegistrationMetric:
    file_name: str
    calibration_points: int
    global_fitness: float | None
    global_rmse: float | None
    icp_fitness: float
    icp_rmse: float
    transformation: list[list[float]]
    full_cloud_fitness: float | None = None
    full_cloud_rmse: float | None = None
    included_in_fusion: bool = True
    section_points: int | None = None
    section_overlap_points: int | None = None
    section_overlap_ratio: float | None = None
    section_median_distance: float | None = None
    section_local_p75_distance: float | None = None
    section_p90_distance: float | None = None
    section_alignment_passed: bool | None = None
    exclusion_reason: str | None = None
    pose_prior_label: str | None = None
    calibration_quality_passed: bool | None = None
    calibration_quality_reason: str | None = None
    section_supporting_views: int | None = None
    section_validation_status: str | None = None
    section_shared_span: float | None = None
    section_bad_bin_ratio: float | None = None
    section_bad_bin_run: int | None = None
    section_duplicate_contour_ratio: float | None = None
    section_missing_contour_ratio: float | None = None
    section_probe_count: int | None = None
    section_valid_probe_count: int | None = None
    section_compatible_probe_count: int | None = None
    section_mismatch_probe_count: int | None = None
    section_extra_contour_probe_count: int | None = None
    section_abnormal_evidence_count: int | None = None
    section_consensus_score: float | None = None
    section_consensus_evaluated_points: int | None = None
    section_consensus_removed_points: int | None = None
    section_consensus_removed_ratio: float | None = None
    section_consensus_shared_removed_ratio: float | None = None
    section_consensus_evaluated_bins: int | None = None
    calibration_geometry_status: str | None = None
    calibration_geometry_plane_count: int | None = None
    calibration_geometry_matched_planes: int | None = None
    calibration_geometry_plane_rmse_before: float | None = None
    calibration_geometry_plane_rmse_after: float | None = None
    calibration_geometry_correction_rotation_degrees: float | None = None
    calibration_geometry_correction_translation: float | None = None
    calibration_geometry_pairwise_rmse: float | None = None
    calibration_geometry_report: dict[str, object] | None = None
    overlap_refinement_report: dict[str, object] | None = None


@dataclass
class PoseGraphEdgeMetric:
    stage: str
    source_index: int
    target_index: int
    source_label: str
    target_label: str
    edge_type: str
    uncertain: bool
    initial_fitness: float
    initial_rmse: float
    fitness: float
    rmse: float
    correction_rotation_degrees: float
    correction_translation: float
    accepted: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class SectionAlignmentQuality:
    section_points: int
    overlap_points: int
    median_distance: float | None
    p90_distance: float | None
    overlap_ratio: float | None = None
    local_p75_distance: float | None = None


@dataclass
class LevelingResult:
    transformation: np.ndarray
    reference_plane_normal: np.ndarray
    reference_plane_center: np.ndarray
    plane_points: int
    tilt_degrees: float
    plane_rmse: float
    applied: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "reference_plane_normal": np.asarray(self.reference_plane_normal).round(8).tolist(),
            "reference_plane_center": np.asarray(self.reference_plane_center).round(8).tolist(),
            "plane_points": self.plane_points,
            "tilt_degrees": round(self.tilt_degrees, 8),
            "plane_rmse": round(self.plane_rmse, 8),
            "transformation": np.asarray(self.transformation).round(12).tolist(),
        }


@dataclass
class PreviewData:
    points: np.ndarray
    colors: np.ndarray
    source_indices: np.ndarray
    input_points: int = 0
    after_denoise_points: int = 0
    before_limit_points: int = 0
    leveling: LevelingResult | None = None


@dataclass
class SurfacePreviewData:
    vertices: np.ndarray
    triangles: np.ndarray
    source_points: int
    reconstruction_points: int
    voxel_size: float


@dataclass
class PipelineResult:
    group_name: str
    run_directory: Path
    merged_cloud_path: Path
    raw_section_cloud_path: Path | None
    section_cloud_path: Path | None
    raw_section_image_path: Path
    section_image_path: Path
    top_preview_path: Path
    summary_path: Path
    merged_points: int
    section_points: int
    compute_backend: str
    gpu_device_name: str | None
    denoise_stats: list[CloudDenoiseStats]
    registration_metrics: list[RegistrationMetric]
    section_source_points: int = 0
    section_raw_points: int = 0
    leveling: LevelingResult | None = None
    alignment_audit_path: Path | None = None
    section_alignment_audit_path: Path | None = None
    section_coverage_audit_path: Path | None = None
    multi_section_audit_path: Path | None = None
    section_integrity: dict[str, object] | None = None


PLY_TYPE_TO_NUMPY = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def discover_groups(data_root: Path) -> dict[str, list[Path]]:
    """Return PLY groups from direct folders or one-level class folders."""
    if not data_root.exists():
        return {}

    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return (0, f"{int(path.name):08d}")
        except ValueError:
            return (1, path.name.lower())

    groups: dict[str, list[Path]] = {}
    directories = (
        child
        for item in data_root.iterdir() if item.is_dir()
        for child in ((item,) if list(item.glob("*.ply")) else
                      tuple(child for child in item.iterdir() if child.is_dir()))
    )
    for directory in sorted(directories, key=sort_key):
        paths = sorted(directory.glob("*.ply"), key=lambda item: item.name.lower())
        if paths:
            groups[directory.name] = paths
    return groups


def select_reference_file(files: Sequence[Path], target_vertex_count: int) -> Path:
    """从一组 PLY 中选择最接近目标点数的固定物理视角作为参考。"""
    if not files:
        raise ValueError("无法从空文件列表中选择参考视角。")
    return min(
        files,
        key=lambda path: (abs(parse_ply_header(path).vertex_count - target_vertex_count), path.name.lower()),
    )


def order_files_with_reference(files: Sequence[Path], target_vertex_count: int) -> list[Path]:
    reference_file = select_reference_file(files, target_vertex_count)
    metadata = _station_order_metadata_path(files) if len(files) in (5, 6) else None
    ordered = files
    if metadata:
        from .registration_reference import station_ids
        labels = station_ids(files)
        by_station = dict(zip(labels, files, strict=True))
        if set(by_station) == set(SIX_STATION_GENERIC_REGISTRATION_ORDER):
            ordered = [by_station[label] for label in SIX_STATION_GENERIC_REGISTRATION_ORDER]
        else:
            by_name = {path.name: path for path in files}
            views = json.loads(metadata.read_text(encoding="utf-8"))["views"]
            ordered = [by_name[item["file"]] for item in views]
    return [reference_file, *(path for path in ordered if path != reference_file)]


def order_files_with_station_pose_prior(files: Sequence[Path]) -> tuple[list[Path], list[str]]:
    """Restore fixed-station order, preferring explicit metadata over legacy file times."""
    ordered_files, labels, _ = order_files_with_station_pose_prior_with_source(files)
    return ordered_files, labels


_STATION_ORDER_NUMBER_PATTERN = re.compile(
    r"(?:view|station|scan|camera|工位)[_-]?(\d+)",
    flags=re.IGNORECASE,
)


def _station_order_metadata_path(files: Sequence[Path]) -> Path | None:
    if not files:
        return None
    parents = {Path(path).resolve().parent for path in files}
    if len(parents) != 1:
        return None
    candidate = next(iter(parents)) / "station_order.json"
    return candidate if candidate.exists() else None


def _read_station_order_metadata(
    metadata_path: Path,
    files: Sequence[Path],
) -> tuple[list[Path], list[str]]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取工位顺序元数据 {metadata_path.name}: {exc}") from exc
    views = payload.get("views") if isinstance(payload, Mapping) else None
    if not isinstance(views, list) or len(views) != len(files):
        raise ValueError(
            f"{metadata_path.name} 的 views 必须包含当前目录全部 {len(files)} 个点云文件。"
        )
    files_by_name = {Path(path).name: Path(path) for path in files}
    valid_labels = set(SIX_STATION_POSE_TRANSFORMS)
    ordered_files: list[Path] = []
    labels: list[str] = []
    seen_files: set[str] = set()
    seen_labels: set[str] = set()
    for position, item in enumerate(views, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"{metadata_path.name} 的第 {position} 个 views 项必须是对象。")
        file_name = Path(str(item.get("file", ""))).name
        station = str(item.get("station", "")).strip()
        if file_name not in files_by_name:
            raise ValueError(f"{metadata_path.name} 引用了不存在的点云文件：{file_name}。")
        if station not in valid_labels:
            raise ValueError(f"{metadata_path.name} 包含未知工位标签：{station}。")
        if file_name in seen_files or station in seen_labels:
            raise ValueError(f"{metadata_path.name} 存在重复文件或重复工位：{file_name}/{station}。")
        seen_files.add(file_name)
        seen_labels.add(station)
        ordered_files.append(files_by_name[file_name])
        labels.append(station)
    if len(ordered_files) != len(files):
        raise ValueError(f"{metadata_path.name} 未覆盖当前目录中的全部点云文件。")
    if labels[0] != "reference":
        raise ValueError(
            f"{metadata_path.name} 的第一项必须是 reference，当前为 {labels[0]}；"
            "固定工位流程需要以参考视角作为第一个节点。"
        )
    return ordered_files, labels


def order_files_with_station_pose_prior_with_source(
    files: Sequence[Path],
) -> tuple[list[Path], list[str], str]:
    """Return ordered files, station labels, and the reproducible order source."""
    if len(files) == len(SIX_STATION_POSE_SEQUENCE):
        acquisition_sequence = SIX_STATION_POSE_SEQUENCE
    elif len(files) == len(FIVE_STATION_POSE_SEQUENCE):
        acquisition_sequence = FIVE_STATION_POSE_SEQUENCE
    else:
        raise ValueError(
            f'固定工位姿态先验要求 5 个或 6 个 PLY 文件，当前为 {len(files)} 个。'
        )

    metadata_path = _station_order_metadata_path(files)
    if metadata_path is not None:
        ordered_files, labels = _read_station_order_metadata(metadata_path, files)
        return ordered_files, labels, "station_order.json"

    direct_label_matches = {
        path: next(
            (
                label
                for label in SIX_STATION_POSE_TRANSFORMS
                if label.lower() in path.stem.lower()
            ),
            None,
        )
        for path in files
    }
    if all(direct_label_matches.values()) and len(set(direct_label_matches.values())) == len(files):
        files_by_label = {
            str(label): path
            for path, label in direct_label_matches.items()
            if label is not None
        }
        labels = [label for label in SIX_STATION_OUTPUT_ORDER if label in files_by_label]
        return [files_by_label[label] for label in labels], labels, "filename_station_label"

    numbered_matches = {
        path: _STATION_ORDER_NUMBER_PATTERN.search(path.stem)
        for path in files
    }
    if all(numbered_matches.values()):
        numbered_files = sorted(
            files,
            key=lambda path: (
                int(numbered_matches[path].group(1)),
                path.name.lower(),
            ),
        )
        files_by_label = dict(zip(acquisition_sequence, numbered_files, strict=True))
        labels = [label for label in SIX_STATION_OUTPUT_ORDER if label in files_by_label]
        return [files_by_label[label] for label in labels], labels, "filename_view_number"

    chronological_files = sorted(
        files,
        key=lambda path: (path.stat().st_mtime_ns, path.name.lower()),
    )
    files_by_label = dict(zip(acquisition_sequence, chronological_files, strict=True))
    labels = [label for label in SIX_STATION_OUTPUT_ORDER if label in files_by_label]
    return [files_by_label[label] for label in labels], labels, "file_mtime_legacy"


def order_files_with_six_station_pose_prior(files: Sequence[Path]) -> tuple[list[Path], list[str]]:
    """Backward-compatible alias for the fixed station pose-prior ordering."""
    return order_files_with_station_pose_prior(files)


def order_files_with_six_station_pose_prior_with_source(
    files: Sequence[Path],
) -> tuple[list[Path], list[str], str]:
    """Six-station ordering variant that also exposes the order source."""
    return order_files_with_station_pose_prior_with_source(files)


def station_pose_correction_transformation(correction: Mapping[str, float]) -> np.ndarray:
    """Build an XYZ Euler correction in the reference coordinate system."""
    rotation_x = math.radians(float(correction.get("rotation_x_degrees", 0.0)))
    rotation_y = math.radians(float(correction.get("rotation_y_degrees", 0.0)))
    rotation_z = math.radians(float(correction.get("rotation_z_degrees", 0.0)))
    cosine_x, sine_x = math.cos(rotation_x), math.sin(rotation_x)
    cosine_y, sine_y = math.cos(rotation_y), math.sin(rotation_y)
    cosine_z, sine_z = math.cos(rotation_z), math.sin(rotation_z)
    matrix_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine_x, -sine_x], [0.0, sine_x, cosine_x]],
        dtype=np.float64,
    )
    matrix_y = np.array(
        [[cosine_y, 0.0, sine_y], [0.0, 1.0, 0.0], [-sine_y, 0.0, cosine_y]],
        dtype=np.float64,
    )
    matrix_z = np.array(
        [[cosine_z, -sine_z, 0.0], [sine_z, cosine_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = matrix_z @ matrix_y @ matrix_x
    transformation[:3, 3] = [
        float(correction.get("translation_x", 0.0)),
        float(correction.get("translation_y", 0.0)),
        float(correction.get("translation_z", 0.0)),
    ]
    return transformation


def six_station_pose_transformations(
    labels: Sequence[str],
    corrections: Mapping[str, Mapping[str, float]] | None = None,
) -> list[np.ndarray]:
    """Return calibrated station transforms with optional user corrections."""
    unknown_labels = [label for label in labels if label not in SIX_STATION_POSE_TRANSFORMS]
    if unknown_labels:
        raise ValueError(f'未知固定工位姿态标签: {", ".join(unknown_labels)}')
    station_corrections = corrections or {}
    transformations: list[np.ndarray] = []
    for label in labels:
        transformation = SIX_STATION_POSE_TRANSFORMS[label].copy()
        if label in station_corrections:
            transformation = station_pose_correction_transformation(station_corrections[label]) @ transformation
        transformations.append(transformation)
    return transformations


def parse_ply_header(path: Path) -> PlyHeader:
    """解析顶点属性，兼容本项目 Sizector 二进制 PLY 的 float RGB。"""
    format_name: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    active_element: str | None = None

    with path.open("rb") as file:
        if file.readline().strip() != b"ply":
            raise ValueError(f"不是 PLY 文件: {path}")
        while True:
            raw_line = file.readline()
            if not raw_line:
                raise ValueError(f"PLY 头缺少 end_header: {path}")
            line = raw_line.decode("ascii", errors="strict").strip()
            if line == "end_header":
                data_offset = file.tell()
                break
            if not line or line.startswith("comment") or line.startswith("obj_info"):
                continue

            parts = line.split()
            if parts[0] == "format" and len(parts) >= 2:
                format_name = parts[1]
            elif parts[0] == "element" and len(parts) == 3:
                active_element = parts[1]
                if active_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and active_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(f"不支持的顶点属性定义: {line}")
                ply_type, property_name = parts[1], parts[2]
                if ply_type not in PLY_TYPE_TO_NUMPY:
                    raise ValueError(f"不支持的 PLY 数据类型 {ply_type}: {path}")
                vertex_properties.append((property_name, "<" + PLY_TYPE_TO_NUMPY[ply_type]))

    if format_name is None or vertex_count is None or not vertex_properties:
        raise ValueError(f"不完整的 PLY 头: {path}")
    return PlyHeader(format_name, data_offset, vertex_count, np.dtype(vertex_properties))


def read_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """读取 PLY，并将 RGB 统一为 Open3D 所需的 0~1 浮点颜色。"""
    header = parse_ply_header(path)
    if header.format_name == "binary_little_endian":
        records = np.fromfile(path, dtype=header.vertex_dtype, count=header.vertex_count, offset=header.data_offset)
    elif header.format_name == "ascii":
        records = np.loadtxt(path, dtype=header.vertex_dtype, skiprows=_count_header_lines(path))
    else:
        raise ValueError(f"不支持的 PLY 格式 {header.format_name}: {path}")

    names = records.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY 缺少 x/y/z 顶点坐标: {path}")
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64, copy=False)
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    if len(points) == 0:
        raise ValueError(f"PLY 中没有有效的三维点: {path}")

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    color_fields = _find_color_fields(names)
    if color_fields is not None:
        colors = np.column_stack([records[field] for field in color_fields]).astype(np.float64, copy=False)[valid_mask]
        maximum = float(np.nanmax(colors)) if len(colors) else 0.0
        if maximum > 1.0:
            colors /= 255.0
        colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return cloud


def _count_header_lines(path: Path) -> int:
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if raw_line.strip() == b"end_header":
                return line_number
    raise ValueError(f"PLY 头缺少 end_header: {path}")


def _find_color_fields(names: Iterable[str]) -> tuple[str, str, str] | None:
    field_names = set(names)
    for candidate in (("red", "green", "blue"), ("r", "g", "b")):
        if set(candidate).issubset(field_names):
            return candidate
    return None


def _cloud_from_arrays(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    normalized_points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if normalized_points.ndim != 2 or normalized_points.shape[1] != 3:
        raise ValueError("点数据必须是形状为 (N, 3) 的数组。")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normalized_points))
    if colors is not None:
        normalized_colors = np.ascontiguousarray(np.asarray(colors, dtype=np.float64))
        if normalized_colors.shape != normalized_points.shape:
            raise ValueError("颜色数据必须与点数据一一对应。")
        cloud.colors = o3d.utility.Vector3dVector(np.clip(normalized_colors, 0.0, 1.0))
    return cloud


def _cloud_colors_or_none(cloud: o3d.geometry.PointCloud) -> np.ndarray | None:
    colors = np.asarray(cloud.colors)
    return colors if len(colors) == len(cloud.points) else None


def _select_cloud_by_mask(cloud: o3d.geometry.PointCloud, mask: np.ndarray) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    normalized_mask = np.asarray(mask, dtype=bool)
    if normalized_mask.shape != (len(points),):
        raise ValueError("点云筛选掩码长度不匹配。")
    colors = _cloud_colors_or_none(cloud)
    return _cloud_from_arrays(points[normalized_mask], colors[normalized_mask] if colors is not None else None)


def _voxel_down_sample_cloud(cloud: o3d.geometry.PointCloud, voxel_size: float, backend: str) -> o3d.geometry.PointCloud:
    if backend == "gpu":
        points, colors = gpu_voxel_down_sample(np.asarray(cloud.points), _cloud_colors_or_none(cloud), voxel_size)
        return _cloud_from_arrays(points, colors)
    return cloud.voxel_down_sample(voxel_size)


def _transform_cloud(cloud: o3d.geometry.PointCloud, transformation: np.ndarray, backend: str) -> o3d.geometry.PointCloud:
    if backend == "gpu":
        points = gpu_transform_points(np.asarray(cloud.points), transformation)
        return _cloud_from_arrays(points, _cloud_colors_or_none(cloud))
    transformed_cloud = copy.deepcopy(cloud)
    transformed_cloud.transform(transformation)
    return transformed_cloud


def _merge_clouds(
    clouds: Sequence[o3d.geometry.PointCloud],
    transformations: Sequence[np.ndarray],
    voxel_size: float,
    backend: str,
) -> o3d.geometry.PointCloud:
    if backend == "gpu":
        points, colors = gpu_fuse_clouds(
            [np.asarray(cloud.points) for cloud in clouds],
            [_cloud_colors_or_none(cloud) for cloud in clouds],
            transformations,
            voxel_size,
        )
        return _cloud_from_arrays(points, colors)

    merged_cloud = o3d.geometry.PointCloud()
    for cloud, transformation in zip(clouds, transformations):
        merged_cloud += _transform_cloud(cloud, transformation, backend="cpu")
    return merged_cloud.voxel_down_sample(voxel_size)


def _representative_voxel_fusion(
    clouds: Sequence[o3d.geometry.PointCloud],
    transformations: Sequence[np.ndarray],
    voxel_size: float,
    backend: str,
    source_indices: Sequence[int] | None = None,
    *,
    clouds_are_aligned: bool = False,
    return_source_indices: bool = False,
) -> tuple[o3d.geometry.PointCloud, dict[str, object]] | tuple[
    o3d.geometry.PointCloud, dict[str, object], np.ndarray
    ] | tuple[o3d.geometry.PointCloud, dict[str, object], np.ndarray, np.ndarray]:
    """Fuse clouds by retaining real samples instead of emitting voxel means.

    Open3D's regular voxel sampler averages coordinates and colors.  That is
    useful for a preview, but it can create a false middle surface when two
    views are separated by a small registration error or when a thin edge is
    observed from different angles.  This routine uses the voxel centroid only
    to rank candidates, then emits one original measured point per voxel.  A
    voxel covered by one view is still retained, so complementary coverage is
    never converted to an overlap-only intersection.
    """
    if len(clouds) != len(transformations):
        raise ValueError("代表点融合的点云和位姿数量必须一致。")
    if voxel_size <= 0:
        raise ValueError("代表点融合体素大小必须大于 0。")
    if source_indices is None:
        normalized_sources = list(range(len(clouds)))
    else:
        normalized_sources = [int(value) for value in source_indices]
        if len(normalized_sources) != len(clouds):
            raise ValueError("代表点融合的来源索引数量必须与点云数量一致。")

    point_sets: list[np.ndarray] = []
    source_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray | None] = []
    colors_complete = True
    source_input_counts: dict[str, int] = {}
    for cloud, transformation, source_index in zip(
        clouds,
        transformations,
        normalized_sources,
        strict=True,
    ):
        aligned_cloud = (
            cloud
            if clouds_are_aligned
            else _transform_cloud(cloud, np.asarray(transformation, dtype=np.float64), backend)
        )
        raw_points = np.asarray(aligned_cloud.points, dtype=np.float64)
        finite_mask = np.isfinite(raw_points).all(axis=1)
        points = raw_points[finite_mask]
        source_input_counts[str(source_index)] = int(len(points))
        if not len(points):
            continue
        point_sets.append(points)
        source_sets.append(np.full(len(points), source_index, dtype=np.int64))
        colors = _cloud_colors_or_none(aligned_cloud)
        if colors is None or len(colors) != len(raw_points):
            colors_complete = False
            color_sets.append(None)
        else:
            color_sets.append(np.asarray(colors, dtype=np.float64)[finite_mask])

    empty_report = {
        "status": "insufficient_data",
        "method": "representative_sample_per_voxel",
        "voxel_size": float(voxel_size),
        "input_points_before_voxel": int(sum(source_input_counts.values())),
        "output_points_after_voxel": 0,
        "points_removed_by_voxel": int(sum(source_input_counts.values())),
        "voxel_count": 0,
        "multi_view_voxel_count": 0,
        "unique_view_voxel_count": 0,
        "multi_view_voxel_ratio": 0.0,
        "unique_view_voxel_ratio": 0.0,
        "source_input_counts": source_input_counts,
        "source_retained_voxel_counts": {},
        "representative_residual_median": None,
        "representative_residual_p95": None,
        "unique_coverage_retained": True,
        "coordinates_averaged": False,
    }
    if not point_sets:
        empty_cloud = o3d.geometry.PointCloud()
        if return_source_indices:
            return (
                empty_cloud,
                empty_report,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        return empty_cloud, empty_report

    points = np.concatenate(point_sets, axis=0)
    sources = np.concatenate(source_sets, axis=0)
    colors = np.concatenate(color_sets, axis=0) if colors_complete and color_sets else None

    voxel_keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1 if len(inverse) else 0
    # The centroid is used only as a stable representative ranking signal.  It
    # is never written to the output cloud, which keeps all emitted coordinates
    # on an actual measured sample.
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    centroid = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis], minlength=voxel_count) / counts
            for axis in range(3)
        ]
    )
    residuals = np.linalg.norm(points - centroid[inverse], axis=1)

    # Sort by voxel, then by distance to its centroid, and finally by source id
    # for deterministic tie-breaking.  The first point in each voxel is a real
    # sample and therefore cannot invent a mid-surface between two layers.
    order = np.lexsort((sources, residuals, inverse))
    sorted_inverse = inverse[order]
    first_mask = np.concatenate(
        ([True], sorted_inverse[1:] != sorted_inverse[:-1])
    )
    selected = order[first_mask]
    selected_sources = sources[selected]

    # Count distinct sources per voxel without materialising a second N×2
    # matrix.  This avoids a quadratic neighbourhood search and keeps peak
    # memory bounded for the 5-8 full-resolution scans used on site.
    pair_order = np.lexsort((sources, inverse))
    pair_inverse = inverse[pair_order]
    pair_sources = sources[pair_order]
    unique_pair_mask = np.concatenate(
        (
            [True],
            (pair_inverse[1:] != pair_inverse[:-1])
            | (pair_sources[1:] != pair_sources[:-1]),
        )
    )
    support_counts = np.bincount(
        pair_inverse[unique_pair_mask].astype(np.int64),
        minlength=voxel_count,
    ).astype(np.int64)
    selected_support = support_counts
    multi_view_mask = selected_support >= 2
    unique_view_mask = selected_support == 1
    retained_counts: dict[str, int] = {
        str(source): int(count)
        for source, count in zip(*np.unique(selected_sources, return_counts=True), strict=True)
    }
    supported_residuals = residuals[selected][multi_view_mask]
    output_points = points[selected]
    report = {
        "status": "ok",
        "method": "representative_sample_per_voxel",
        "voxel_size": float(voxel_size),
        "input_points_before_voxel": int(len(points)),
        "output_points_after_voxel": int(len(output_points)),
        "points_removed_by_voxel": int(len(points) - len(output_points)),
        "voxel_count": int(voxel_count),
        "multi_view_voxel_count": int(np.count_nonzero(multi_view_mask)),
        "unique_view_voxel_count": int(np.count_nonzero(unique_view_mask)),
        "multi_view_voxel_ratio": float(np.mean(multi_view_mask)) if len(multi_view_mask) else 0.0,
        "unique_view_voxel_ratio": float(np.mean(unique_view_mask)) if len(unique_view_mask) else 0.0,
        "source_input_counts": source_input_counts,
        "source_retained_voxel_counts": retained_counts,
        "representative_residual_median": (
            float(np.median(supported_residuals)) if len(supported_residuals) else None
        ),
        "representative_residual_p95": (
            float(np.quantile(supported_residuals, 0.95)) if len(supported_residuals) else None
        ),
        "unique_coverage_retained": True,
        "coordinates_averaged": False,
        "selection_policy": "nearest_to_voxel_centroid_then_source_id",
    }
    sampled_cloud = _cloud_from_arrays(
        output_points,
        colors[selected] if colors is not None else None,
    )
    if return_source_indices:
        return (
            sampled_cloud,
            report,
            selected_sources.astype(np.int64, copy=True),
            selected_support.astype(np.int64, copy=True),
        )
    return sampled_cloud, report


def _voxel_component_labels(voxel_keys: np.ndarray) -> tuple[int, np.ndarray]:
    """Label occupied cells using the exact 26-neighbor voxel relation."""
    pairs = cKDTree(voxel_keys).query_pairs(1, p=np.inf, output_type="ndarray")
    if not len(pairs):
        return len(voxel_keys), np.arange(len(voxel_keys), dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
        shape=(len(voxel_keys), len(voxel_keys)),
    ).tocsr()
    return connected_components(graph, directed=False)


def _spatial_query_workers() -> int:
    # Automatic workers already receive a per-process CPU budget.
    try:
        budget = int(os.environ.get("OMP_NUM_THREADS", "4"))
    except ValueError:
        budget = 1
    return max(1, min(4, budget, os.cpu_count() or 1))


def _final_fusion_component_cleanup(
    cloud: o3d.geometry.PointCloud,
    source_indices: np.ndarray | None,
    voxel_size: float,
    *,
    source_support_counts: np.ndarray | None = None,
    component_voxel_size: float | None = None,
) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    """Remove compact single-view islands from the final aligned fusion cloud.

    This is deliberately applied after all views share one coordinate frame.
    A fragment can look acceptable inside one scan yet become a floating island
    only after registration and voxel fusion.  Large unique regions and small
    regions observed by multiple stations are retained for coverage safety.
    """
    points = np.asarray(cloud.points, dtype=np.float64)
    normalized_sources = (
        np.asarray(source_indices, dtype=np.int64).reshape(-1)
        if source_indices is not None
        else np.full(len(points), -1, dtype=np.int64)
    )
    report: dict[str, object] = {
        "enabled": True,
        "stage": "final_aligned_fused_cloud",
        "method": "compact_weakly_supported_orphan_components_after_fusion_voxel",
        "input_points": int(len(points)),
        "output_points": int(len(points)),
        "removed_points": 0,
        "component_count": 0,
        "cleaned_component_count": 0,
        "component_voxel_size": float(
            max(float(component_voxel_size or 0.0), float(voxel_size) * 6.0, 0.12)
        ),
        "minimum_component_points": 64,
        "compact_span_limit": 1.5,
        "removed_components": [],
        "preserved_components": [],
    }
    if len(points) < 2:
        report["enabled"] = False
        report["reason"] = "insufficient_points"
        return cloud, report
    if normalized_sources.shape != (len(points),):
        raise ValueError("最终融合组件清理的来源索引数量必须与点数一致。")
    normalized_support = (
        np.asarray(source_support_counts, dtype=np.int64).reshape(-1)
        if source_support_counts is not None
        else np.ones(len(points), dtype=np.int64)
    )
    if normalized_support.shape != (len(points),):
        raise ValueError("最终融合组件清理的支持度数量必须与点数一致。")

    component_voxel = float(report["component_voxel_size"])
    voxel_keys, inverse = np.unique(
        np.floor(points / component_voxel).astype(np.int64),
        axis=0,
        return_inverse=True,
    )
    if len(voxel_keys) < 2:
        report["component_count"] = 1
        report["cleaned_component_count"] = 1
        return cloud, report

    component_count, labels = _voxel_component_labels(voxel_keys)
    point_labels = labels[inverse]
    component_sizes = np.bincount(point_labels, minlength=component_count)
    order = np.argsort(component_sizes)[::-1]
    largest_size = int(component_sizes[order[0]])
    size_limit = max(
        int(report["minimum_component_points"]),
        min(1024, int(max(largest_size, 1) * 0.005)),
    )
    compact_span_limit = float(max(component_voxel * 10.0, 1.5))
    report["component_count"] = int(component_count)
    report["minimum_component_points"] = int(size_limit)
    report["compact_span_limit"] = compact_span_limit

    keep = np.ones(len(points), dtype=bool)
    removed_components: list[dict[str, object]] = []
    preserved_components: list[dict[str, object]] = []
    for component_id in order:
        component_mask = point_labels == component_id
        count = int(np.count_nonzero(component_mask))
        if component_id == order[0]:
            continue
        component_points = points[component_mask]
        span = np.ptp(component_points, axis=0)
        sources = normalized_sources[component_mask]
        supports = normalized_support[component_mask]
        source_count = int(len(np.unique(sources[sources >= 0])))
        multi_view_support_ratio = float(np.mean(supports >= 2))
        compact = bool(float(np.max(span)) <= compact_span_limit)
        small = count <= size_limit
        weakly_supported = bool(
            source_count <= 1
            or multi_view_support_ratio < 0.20
        )
        remove = bool(small and compact and weakly_supported)
        item = {
            "component_id": int(component_id),
            "points": count,
            "span": np.asarray(span).round(6).tolist(),
            "source_view_count": source_count,
            "multi_view_support_ratio": multi_view_support_ratio,
            "compact": compact,
            "weakly_supported": weakly_supported,
        }
        if remove:
            keep[component_mask] = False
            removed_components.append(item)
        elif count <= size_limit:
            preserved_components.append(item)

    cleaned = _select_cloud_by_mask(cloud, keep)
    removed_count = int(np.count_nonzero(~keep))
    report.update({
        "output_points": int(len(cleaned.points)),
        "removed_points": removed_count,
        "cleaned_component_count": int(
            component_count - len(removed_components)
        ),
        "removed_components": removed_components,
        "preserved_components": preserved_components,
    })
    return cleaned, report


def _representative_voxel_down_sample_cloud(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    backend: str,
) -> o3d.geometry.PointCloud:
    """Downsample one scan while keeping measured coordinates and colors."""
    sampled, _ = _representative_voxel_fusion(
        [cloud],
        [np.eye(4, dtype=np.float64)],
        voxel_size,
        backend,
        source_indices=[0],
        clouds_are_aligned=True,
    )
    return sampled


def build_preview(
    files: Sequence[Path],
    voxel_size: float = 0.45,
    maximum_points: int = 75_000,
    compute_backend: str = "auto",
    denoise_config: DenoiseConfig | None = None,
    apply_denoise: bool = False,
    leveling: LevelingResult | None = None,
    protected_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    source_clouds: Sequence[o3d.geometry.PointCloud] | None = None,
) -> PreviewData:
    """加载低采样点，用于网页内的交互式标定块 ROI 预览。"""
    active_backend = resolve_compute_backend(compute_backend)
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    source_sets: list[np.ndarray] = []
    input_points = 0
    after_denoise_points = 0
    before_limit_points = 0
    preview_denoise_config = denoise_config or DenoiseConfig()
    if apply_denoise:
        preview_denoise_config.validate()
    if source_clouds is not None and len(source_clouds) != len(files):
        raise ValueError("预览源点云数量必须与文件数量一致。")
    for source_index, path in enumerate(files):
        cloud = (
            copy.deepcopy(source_clouds[source_index])
            if source_clouds is not None
            else read_point_cloud(path)
        )
        input_points += len(cloud.points)
        if apply_denoise:
            cloud, _ = denoise_cloud(
                cloud,
                preview_denoise_config,
                path.name,
                active_backend,
                protected_bounds=protected_bounds,
            )
        after_denoise_points += len(cloud.points)
        if leveling is not None and leveling.applied:
            cloud = _transform_cloud(cloud, leveling.transformation, active_backend)
        cloud = _voxel_down_sample_cloud(cloud, voxel_size, active_backend)
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        if len(colors) != len(points):
            colors = np.full((len(points), 3), 0.65, dtype=np.float64)
        before_limit_points += len(points)
        point_sets.append(points)
        color_sets.append(colors)
        source_sets.append(np.full(len(points), source_index, dtype=np.int16))

    points = np.concatenate(point_sets, axis=0)
    colors = np.concatenate(color_sets, axis=0)
    source_indices = np.concatenate(source_sets, axis=0)
    if len(points) > maximum_points:
        sample_indices = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
        points = points[sample_indices]
        colors = colors[sample_indices]
        source_indices = source_indices[sample_indices]
    return PreviewData(
        points=points,
        colors=colors,
        source_indices=source_indices,
        input_points=input_points,
        after_denoise_points=after_denoise_points,
        before_limit_points=before_limit_points,
        leveling=leveling,
    )


def build_surface_preview(
    points: np.ndarray,
    voxel_size: float,
    maximum_points: int = 35_000,
    maximum_triangles: int = 100_000,
    compute_backend: str = "auto",
) -> SurfacePreviewData:
    """将预览点云重建为受点数和三角面数约束的表面网格。"""
    source_points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("表面预览点数据必须是形状为 (N, 3) 的数组。")
    if voxel_size <= 0:
        raise ValueError("表面预览体素大小必须大于 0。")
    if maximum_points < 30:
        raise ValueError("表面预览最多重建点数至少为 30。")
    if maximum_triangles < 1:
        raise ValueError("表面预览最多三角面数至少为 1。")

    finite_mask = np.isfinite(source_points).all(axis=1)
    source_points = source_points[finite_mask]
    if len(source_points) < 30:
        raise ValueError("有效预览点不足，无法重建表面。")

    active_backend = resolve_compute_backend(compute_backend)
    source_cloud = _cloud_from_arrays(source_points)
    effective_voxel_size = float(voxel_size)
    reconstruction_cloud = _voxel_down_sample_cloud(source_cloud, effective_voxel_size, active_backend)
    for _ in range(4):
        reconstruction_count = len(reconstruction_cloud.points)
        if reconstruction_count <= maximum_points:
            break
        reduction_factor = max(1.15, math.sqrt(reconstruction_count / maximum_points) * 1.08)
        effective_voxel_size *= reduction_factor
        reconstruction_cloud = _voxel_down_sample_cloud(source_cloud, effective_voxel_size, active_backend)

    reconstruction_points = np.asarray(reconstruction_cloud.points)
    if len(reconstruction_points) > maximum_points:
        sample_indices = np.linspace(0, len(reconstruction_points) - 1, maximum_points, dtype=np.int64)
        reconstruction_cloud = _cloud_from_arrays(reconstruction_points[sample_indices])
        reconstruction_points = np.asarray(reconstruction_cloud.points)
    if len(reconstruction_points) < 30:
        raise ValueError("表面重建采样后的点数不足。")

    nearest_distances = np.asarray(reconstruction_cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
    valid_distances = nearest_distances[np.isfinite(nearest_distances) & (nearest_distances > 0)]
    if len(valid_distances) == 0:
        raise ValueError("无法估计点云间距，无法重建表面。")
    point_spacing = float(np.median(valid_distances))
    normal_radius = max(point_spacing * 3.0, effective_voxel_size * 1.5)
    reconstruction_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=40)
    )
    try:
        reconstruction_cloud.orient_normals_consistent_tangent_plane(
            min(20, len(reconstruction_points) - 1)
        )
    except RuntimeError:
        reconstruction_cloud.normalize_normals()

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        reconstruction_cloud,
        o3d.utility.DoubleVector([point_spacing * 1.5, point_spacing * 3.0]),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            reconstruction_cloud,
            o3d.utility.DoubleVector([point_spacing * 2.5, point_spacing * 4.0]),
        )
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise ValueError("表面重建未生成三角面，请切换回点云渲染或调整去噪参数。")

    if len(mesh.triangles) > maximum_triangles:
        mesh = mesh.simplify_quadric_decimation(maximum_triangles)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    return SurfacePreviewData(
        vertices=np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float32),
        triangles=np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.int32),
        source_points=len(source_points),
        reconstruction_points=len(reconstruction_points),
        voxel_size=effective_voxel_size,
    )


def denoise_cloud(
    cloud: o3d.geometry.PointCloud,
    config: DenoiseConfig,
    file_name: str,
    backend: str = "cpu",
    protected_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> tuple[o3d.geometry.PointCloud, CloudDenoiseStats]:
    """先保留真实测量样本做体素化，再按可选规则去除离群点。

    普通体素下采样会对一个体素内的坐标求平均；对扣型边缘和薄壁而言，
    这个平均值可能落在两个真实表面之间。去噪阶段先采用代表点策略，避免
    在进入标定块配准和最终融合前就把几何边缘抹平。
    """
    input_points = len(cloud.points)
    if protected_bounds is not None and len(cloud.points):
        lower, upper = (
            np.asarray(value, dtype=np.float64)
            for value in protected_bounds
        )
        points = np.asarray(cloud.points)
        protected_mask = np.all((points >= lower) & (points <= upper), axis=1)
        if protected_mask.any() and (~protected_mask).any():
            protected_cloud = _select_cloud_by_mask(cloud, protected_mask)
            workpiece_cloud = _select_cloud_by_mask(cloud, ~protected_mask)
            cleaned_workpiece, workpiece_stats = denoise_cloud(
                workpiece_cloud,
                config,
                file_name,
                backend,
                protected_bounds=None,
            )
            combined = _concatenate_clouds([cleaned_workpiece, protected_cloud])
            return combined, CloudDenoiseStats(
                file_name=file_name,
                input_points=input_points,
                after_voxel_points=workpiece_stats.after_voxel_points + len(protected_cloud.points),
                after_statistical_points=workpiece_stats.after_statistical_points + len(protected_cloud.points),
                output_points=len(combined.points),
                after_component_points=workpiece_stats.after_component_points + len(protected_cloud.points),
                after_surface_support_points=workpiece_stats.after_surface_support_points + len(protected_cloud.points),
            )

    cleaned = _representative_voxel_down_sample_cloud(cloud, config.voxel_size, backend)
    after_voxel = len(cleaned.points)

    after_statistical = after_voxel
    if config.use_statistical_filter and after_voxel > config.statistical_neighbors:
        neighbor_count = min(config.statistical_neighbors, after_voxel - 1)
        if backend == "gpu":
            cleaned = _select_cloud_by_mask(
                cleaned,
                gpu_statistical_inlier_mask(np.asarray(cleaned.points), neighbor_count, config.statistical_std_ratio),
            )
        else:
            cleaned, _ = cleaned.remove_statistical_outlier(
                nb_neighbors=neighbor_count,
                std_ratio=config.statistical_std_ratio,
            )
        after_statistical = len(cleaned.points)

    if config.use_radius_filter and len(cleaned.points) > config.radius_min_points:
        if backend == "gpu":
            cleaned = _select_cloud_by_mask(
                cleaned,
                gpu_radius_inlier_mask(np.asarray(cleaned.points), config.radius, config.radius_min_points),
            )
        else:
            cleaned, _ = cleaned.remove_radius_outlier(
                nb_points=config.radius_min_points,
                radius=config.radius,
            )

    if config.use_component_filter and len(cleaned.points) > config.secondary_min_points:
        secondary_source = cleaned
        secondary_input_count = len(secondary_source.points)
        secondary_radius = float(config.radius) * float(config.secondary_radius_factor)
        secondary_min_points = min(
            int(config.secondary_min_points),
            max(2, len(cleaned.points) - 1),
        )
        if backend == "gpu":
            secondary_mask = gpu_radius_inlier_mask(
                np.asarray(secondary_source.points),
                secondary_radius,
                secondary_min_points,
            )
            secondary_cleaned = _select_cloud_by_mask(secondary_source, secondary_mask)
        else:
            secondary_cleaned, _ = secondary_source.remove_radius_outlier(
                nb_points=secondary_min_points,
                radius=secondary_radius,
            )
        # Sparse scans can legitimately have fewer neighbours. Reject an
        # over-aggressive second pass instead of returning an empty cloud.
        if (
            len(secondary_cleaned.points) > 0
            and len(secondary_cleaned.points) >= max(
                30,
                int(secondary_input_count * 0.45),
            )
        ):
            cleaned = secondary_cleaned

    after_component = len(cleaned.points)
    if config.use_component_filter and len(cleaned.points) >= config.component_min_points:
        component_mask = _spatial_component_inlier_mask(
            np.asarray(cleaned.points),
            voxel_size=config.component_voxel_size,
            minimum_points=config.component_min_points,
            keep_largest=config.component_keep_largest,
            relative_size=config.component_relative_size,
            gap_factor=config.component_gap_factor,
            protected_bounds=protected_bounds,
        )
        if component_mask.any():
            component_cleaned = _select_cloud_by_mask(cleaned, component_mask)
            if len(component_cleaned.points):
                cleaned = component_cleaned
                after_component = len(cleaned.points)

    if config.use_surface_support_filter:
        surface_mask = _surface_support_inlier_mask(
            np.asarray(cleaned.points), config.voxel_size,
            config.surface_support_neighbors, config.surface_support_distance_factor,
        )
        cleaned = _select_cloud_by_mask(cleaned, surface_mask)

    if len(cleaned.points) == 0:
        raise ValueError(f"去噪后点云为空: {file_name}。请放宽去噪参数。")
    return cleaned, CloudDenoiseStats(
        file_name=file_name,
        input_points=input_points,
        after_voxel_points=after_voxel,
        after_statistical_points=after_statistical,
        output_points=len(cleaned.points),
        after_component_points=after_component,
        after_surface_support_points=len(cleaned.points),
    )


def parallel_denoise_clouds(
    files: Sequence[Path],
    config: DenoiseConfig,
    backend: str,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
) -> tuple[list[o3d.geometry.PointCloud], list[CloudDenoiseStats]]:
    """Read and denoise independent views concurrently, preserving file order."""
    if not files:
        return [], []

    def process_one(path: Path) -> tuple[o3d.geometry.PointCloud, CloudDenoiseStats]:
        return denoise_cloud(read_point_cloud(path), config, path.name, backend)

    if backend == "gpu" or len(files) == 1:
        values = [process_one(path) for path in files]
    else:
        try:
            thread_limit = int(os.environ.get("OMP_NUM_THREADS", "4"))
        except ValueError:
            thread_limit = 1
        worker_count = 1 if os.environ.get("POINTCLOUD_BATCH_WORKER") == "1" else max(
            1, min(len(files), 4, thread_limit)
        )
        values: list[tuple[o3d.geometry.PointCloud, CloudDenoiseStats] | None] = [None] * len(files)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_one, path): index
                for index, path in enumerate(files)
            }
            for future in as_completed(futures):
                index = futures[future]
                values[index] = future.result()
                _emit(
                    progress_callback,
                    f"第 {group_name} 组：并行去噪完成 {index + 1}/{len(files)} - "
                    f"{files[index].name}",
                )
        values = [value for value in values if value is not None]
    clouds, stats = zip(*values, strict=True)
    return list(clouds), list(stats)


def _surface_support_inlier_mask(
    points: np.ndarray,
    voxel_size: float,
    neighbors: int = 24,
    distance_factor: float = 0.4,
) -> np.ndarray:
    """Reject unstable single-view returns while retaining supported corners."""
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    keep = np.ones(count, dtype=bool)
    coarse_k = neighbors * 3
    if count < coarse_k:
        return keep
    tree = cKDTree(points)
    centers = np.empty_like(points)
    normals = np.empty_like(points)
    fine_scatter = np.empty(count)
    fine_width = np.empty(count)
    coarse_scatter = np.empty(count)
    residuals = np.empty(count)
    for start in range(0, count, 4096):
        stop = min(start + 4096, count)
        _, indices = tree.query(points[start:stop], k=coarse_k)
        local = points[indices]
        for k in (neighbors, coarse_k):
            samples = local[:, :k]
            center = samples.mean(axis=1)
            delta = samples - center[:, None, :]
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.einsum("nki,nkj->nij", delta, delta) / k
            )
            scatter = eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-15)
            if k == neighbors:
                centers[start:stop] = center
                normals[start:stop] = eigenvectors[:, :, 0]
                fine_scatter[start:stop] = scatter
                fine_width[start:stop] = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-15)
                residuals[start:stop] = np.abs(np.einsum(
                    "ni,ni->n", points[start:stop] - center, eigenvectors[:, :, 0]
                ))
            else:
                coarse_scatter[start:stop] = scatter
    tolerance = voxel_size * distance_factor
    suspect = (
        ((coarse_scatter > .04) & ((fine_scatter > .015) | (fine_width < .5)))
        | ((residuals > tolerance * 1.5) & (fine_scatter > .015))
    )
    # A long, thin return can look planar inside the fine neighborhood while
    # disagreeing with the broader surface. Require several independent
    # anchors for this second-tier suspect class; ordinary edges keep the
    # permissive rule above.
    extended_suspect = (
        ((coarse_scatter > .018) & ((fine_scatter > .006) | (fine_width < .35)))
        | (residuals > tolerance)
    )
    extended_suspect &= ~suspect
    suspect |= extended_suspect
    reliable = (fine_scatter < .012) & (coarse_scatter < .02) & (fine_width > .35)
    reliable_indices = np.flatnonzero(reliable)
    if not len(reliable_indices):
        # No locally planar evidence: this may be an intentionally rough scan.
        return keep
    support_tree = cKDTree(points[reliable_indices])
    for start in range(0, count, 4096):
        indices = np.flatnonzero(suspect[start:start + 4096]) + start
        if not len(indices):
            continue
        distances, local_ids = support_tree.query(
            points[indices], k=min(neighbors, len(reliable_indices)),
        )
        if distances.ndim == 1:
            distances, local_ids = distances[:, None], local_ids[:, None]
        support_ids = reliable_indices[local_ids]
        offsets = points[indices, None, :] - centers[support_ids]
        plane_distance = np.abs(np.einsum("nki,nki->nk", offsets, normals[support_ids]))
        support_matches = (
            (distances <= voxel_size * 6.) & (plane_distance <= tolerance)
        )
        supported = np.any(support_matches, axis=1)
        strict_supported = np.sum(support_matches, axis=1) >= min(3, support_matches.shape[1])
        keep[indices] = np.where(extended_suspect[indices], strict_supported, supported)
    keep &= _surface_patch_support_mask(
        points, centers, normals, reliable_indices, voxel_size, tolerance, neighbors,
    )
    return keep


def _surface_patch_support_mask(
    points: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    reliable_indices: np.ndarray,
    voxel_size: float,
    tolerance: float,
    neighbors: int,
) -> np.ndarray:
    """Stop small, self-supporting scan strips from acting as surface anchors."""
    keep = np.ones(len(points), dtype=bool)
    minimum_patch = neighbors * 4
    if len(reliable_indices) < minimum_patch:
        return keep
    anchors = points[reliable_indices]
    anchor_normals = normals[reliable_indices]
    anchor_tree = cKDTree(anchors)
    rows, columns = [], []
    # A bounded neighborhood graph joins only locally coplanar samples.
    # Long edges and curved faces can grow into patches; nearby parallel
    # layers cannot bridge merely because they are spatially close.
    for start in range(0, len(anchors), 4096):
        stop = min(start + 4096, len(anchors))
        distances, local = anchor_tree.query(anchors[start:stop], k=min(neighbors, len(anchors)))
        source = np.broadcast_to(np.arange(start, stop)[:, None], local.shape)
        delta = anchors[source] - anchors[local]
        compatible = (
            (source != local) & (distances <= voxel_size * 3.)
            & (np.abs(np.einsum("nki,nki->nk", anchor_normals[source],
                                anchor_normals[local])) >= math.cos(math.radians(15.)))
            & (np.abs(np.einsum("nki,nki->nk", delta, anchor_normals[source])) <= tolerance)
            & (np.abs(np.einsum("nki,nki->nk", delta, anchor_normals[local])) <= tolerance)
        )
        rows.append(source[compatible])
        columns.append(local[compatible])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, columns)),
        shape=(len(anchors), len(anchors)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    stable = sizes[labels] >= minimum_patch
    if stable.all() or not stable.any():
        return keep
    strong_indices = reliable_indices[stable]
    weak_indices = reliable_indices[~stable]
    strong_tree = cKDTree(points[strong_indices])
    weak_distance, _ = cKDTree(points[weak_indices]).query(points)
    strong_distance, _ = strong_tree.query(points)
    support_radius = voxel_size * 12.
    candidates = (
        (weak_distance <= support_radius)
        & (weak_distance < strong_distance + voxel_size * 1.5)
    )
    candidates[strong_indices] = False
    for start in range(0, len(points), 4096):
        indices = np.flatnonzero(candidates[start:start + 4096]) + start
        if not len(indices):
            continue
        distances, local = strong_tree.query(points[indices], k=min(neighbors, len(strong_indices)))
        support = strong_indices[local]
        offsets = points[indices, None] - centers[support]
        plane_distance = np.abs(np.einsum("nki,nki->nk", offsets, normals[support]))
        matches = (distances <= support_radius) & (plane_distance <= tolerance)
        keep[indices] = matches.sum(axis=1) >= 3
        # A closer parallel face can fill the k-nearest list and hide another
        # supporting patch. Check the remaining local anchors before removal.
        unresolved = indices[~keep[indices]]
        for offset in range(0, len(unresolved), 256):
            batch = unresolved[offset:offset + 256]
            neighborhoods = strong_tree.query_ball_point(points[batch], support_radius)
            for index, neighborhood in zip(batch, neighborhoods):
                if len(neighborhood) < 3:
                    continue
                support = strong_indices[neighborhood]
                plane_distance = np.abs(np.einsum(
                    "ni,ni->n", points[index] - centers[support], normals[support],
                ))
                keep[index] = np.count_nonzero(plane_distance <= tolerance) >= 3
    return keep


def _local_surface_planes(points, tree, indices, radius):
    """Fit only bounded, two-dimensional neighborhoods; edges remain uncertain."""
    count = min(12, len(points))
    if count < 6:
        return points[indices], np.zeros((len(indices), 3)), np.zeros(len(indices), dtype=bool)
    distances, neighbors = tree.query(points[indices], k=count)
    local = points[neighbors]
    centers = local.mean(axis=1)
    delta = local - centers[:, None, :]
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", delta, delta) / count)
    valid = (
        (distances[:, -1] <= radius)
        & (values[:, 0] <= .02 * np.maximum(values.sum(axis=1), 1e-15))
        & (values[:, 1] > .1 * values[:, 2])
        & (values[:, 1] > 1e-15)
    )
    return centers, vectors[:, :, 0], valid


def _cross_view_overlap_keep_masks(
    clouds: Sequence[o3d.geometry.PointCloud],
    transformations: Sequence[np.ndarray],
    leveling_transformation: np.ndarray,
    included_views: Sequence[bool],
    voxel_size: float,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Remove view-only returns inside neighborhoods covered by other views.

    A point is removed only when it is outside the close-support radius of every
    other included view but still lies in a neighborhood populated by those
    views. This preserves genuinely unique outer surfaces while rejecting
    floating strips or echoes inside an already observed target region.
    """
    if not (
        len(clouds) == len(transformations) == len(included_views)
        and np.isfinite(voxel_size) and voxel_size > 0
    ):
        raise ValueError("跨视角去噪输入数量或体素大小无效。")
    keep_masks = [np.ones(len(cloud.points), dtype=bool) for cloud in clouds]
    active_indices = [
        index for index, included in enumerate(included_views)
        if bool(included) and len(clouds[index].points)
    ]
    report: dict[str, object] = {
        "enabled": len(active_indices) >= 2,
        "stage": "registered_single_views_before_fusion",
        "method": "cross_view_close_support_with_observed_surface_coverage",
        "nonoverlap_policy": "preserve",
        "support_radius": float(max(voxel_size * 1.25, 0.12)),
        "context_radius": float(max(voxel_size * 4.5, 0.40)),
        "minimum_context_points": 8,
        "minimum_independent_support_views": 2,
        "strong_single_view_context_points": 48,
        "total_removed_points": 0,
        "views": [],
    }
    if len(active_indices) < 2:
        report["reason"] = "less_than_two_included_views"
        return keep_masks, report

    leveling = np.asarray(leveling_transformation, dtype=np.float64)
    aligned_points: dict[int, np.ndarray] = {}
    for index in active_indices:
        points = np.asarray(clouds[index].points, dtype=np.float64)
        final_transform = leveling @ np.asarray(transformations[index], dtype=np.float64)
        aligned_points[index] = points @ final_transform[:3, :3].T + final_transform[:3, 3]

    support_radius = float(report["support_radius"])
    context_radius = float(report["context_radius"])
    minimum_context_points = int(report["minimum_context_points"])
    strong_single_view_context_points = int(report["strong_single_view_context_points"])
    support_distances: dict[int, np.ndarray] = {}
    independent_support_counts: dict[int, np.ndarray] = {}
    maximum_context_counts: dict[int, np.ndarray] = {}
    peer_trees = {index: cKDTree(points) for index, points in aligned_points.items()}
    query_workers = _spatial_query_workers()
    for index in active_indices:
        peer_indices = [peer for peer in active_indices if peer != index]
        if not peer_indices:
            continue
        points = aligned_points[index]
        nearest_distances = []
        context_counts_by_peer = []
        for peer in peer_indices:
            peer_tree = peer_trees[peer]
            nearest_distances.append(peer_tree.query(points, k=1, workers=query_workers)[0])
            context_counts_by_peer.append(peer_tree.query_ball_point(
                points, context_radius, return_length=True, workers=query_workers,
            ))
        nearest_distance = np.min(np.asarray(nearest_distances), axis=0)
        context_counts_by_peer = np.asarray(context_counts_by_peer)
        independent_support_count = np.count_nonzero(
            context_counts_by_peer >= minimum_context_points, axis=0,
        )
        maximum_context_count = context_counts_by_peer.max(axis=0)
        strong_single_view_support = (
            (independent_support_count == 1)
            & (context_counts_by_peer.max(axis=0) >= strong_single_view_context_points)
        )
        unsupported = nearest_distance > support_radius
        support_distances[index] = nearest_distance
        independent_support_counts[index] = independent_support_count
        maximum_context_counts[index] = maximum_context_count
        unsupported_indices = np.flatnonzero(unsupported)
        candidates = unsupported_indices[
            (independent_support_count[unsupported_indices] >= 2)
            | strong_single_view_support[unsupported_indices]
        ]
        # Nearby faces are not proof of coverage at an occluded ramp. Require
        # the peer surface's tangential footprint to reach the candidate.
        initial_candidates = len(candidates)
        surface_context = np.zeros((len(peer_indices), len(candidates)), dtype=np.int64)
        for peer_position, peer in enumerate(peer_indices):
            eligible = context_counts_by_peer[peer_position, candidates] >= minimum_context_points
            positions = np.flatnonzero(eligible)
            if not len(positions):
                continue
            query = points[candidates[positions]]
            peer_points, tree = aligned_points[peer], peer_trees[peer]
            _, nearest = tree.query(query)
            anchors, inverse = np.unique(nearest, return_inverse=True)
            _, normals, valid = _local_surface_planes(
                peer_points, tree, anchors, context_radius,
            )
            delta = query - peer_points[nearest]
            normal_distance = np.einsum("ni,ni->n", delta, normals[inverse])
            tangent = delta - normal_distance[:, None] * normals[inverse]
            covered = valid[inverse] & (
                np.linalg.norm(tangent, axis=1) <= voxel_size * .75
            )
            selected = positions[covered]
            surface_context[peer_position, selected] = context_counts_by_peer[
                peer_position, candidates[selected]
            ]
        if len(candidates):
            covered_views = np.count_nonzero(surface_context >= minimum_context_points, axis=0)
            covered = (covered_views >= 2) | (
                surface_context.max(axis=0) >= strong_single_view_context_points
            )
            candidates = candidates[covered]
        keep_masks[index][candidates] = False
        removed = int(len(candidates))
        report["total_removed_points"] = int(report["total_removed_points"]) + removed
        report["views"].append({
            "input_index": index,
            "input_points": int(len(points)),
            "close_supported_points": int(np.count_nonzero(~unsupported)),
            "independent_overlap_supported_points": int(np.count_nonzero(
                independent_support_count >= 2,
            )),
            "strong_single_view_supported_points": int(np.count_nonzero(
                strong_single_view_support,
            )),
            "unsupported_without_context_points": int(len(unsupported_indices) - removed),
            "preserved_unobserved_surface_points": int(initial_candidates - removed),
            "removed_points": removed,
            "output_points": int(len(points) - removed),
            "removed_ratio": float(removed / len(points)) if len(points) else 0.0,
        })

    # Point-wise filtering can remove the narrow bridge that connected a small
    # floating island to a larger scan surface. Re-check connected components
    # after that pass; otherwise the island survives as a new orphan fragment.
    component_voxel_size = float(max(voxel_size * 1.5, 0.15))
    component_reports: list[dict[str, object]] = []
    component_removed_total = 0
    for index in active_indices:
        retained_indices = np.flatnonzero(keep_masks[index])
        if len(retained_indices) < 2:
            continue
        points = aligned_points[index][retained_indices]
        voxel_keys, inverse = np.unique(
            np.floor(points / component_voxel_size).astype(np.int64),
            axis=0,
            return_inverse=True,
        )
        if len(voxel_keys) < 2:
            continue
        _, labels = _voxel_component_labels(voxel_keys)
        voxel_component_sizes = np.bincount(labels)
        point_component_sizes = np.bincount(
            labels[inverse], minlength=len(voxel_component_sizes),
        )
        component_order = np.argsort(point_component_sizes)[::-1]
        largest_size = int(point_component_sizes[component_order[0]])
        size_limit = max(
            64,
            min(1024, int(max(largest_size, 1) * 0.05)),
        )
        nearest_distance = support_distances[index][retained_indices]
        independent_support_count = independent_support_counts[index][retained_indices]
        maximum_context_count = maximum_context_counts[index][retained_indices]
        remove_local = np.zeros(len(retained_indices), dtype=bool)
        removed_components: list[dict[str, object]] = []
        for component_id in component_order:
            local_mask = labels[inverse] == component_id
            count = int(np.count_nonzero(local_mask))
            if count > size_limit:
                continue
            component_points = points[local_mask]
            close_ratio = float(np.mean(
                nearest_distance[local_mask] <= support_radius,
            ))
            independent_ratio = float(np.mean(
                independent_support_count[local_mask] >= 2,
            ))
            span = np.ptp(component_points, axis=0)
            compact = float(np.max(span)) <= max(context_radius * 4.0, 1.8)
            unsupported_orphan = (
                close_ratio < 0.5
                and independent_ratio == 0.0
                and float(np.max(maximum_context_count[local_mask])) >= minimum_context_points
                and compact
            )
            if not unsupported_orphan:
                continue
            remove_local |= local_mask
            removed_components.append({
                "points": count,
                "span": np.asarray(span).round(6).tolist(),
                "close_support_ratio": close_ratio,
                "independent_support_ratio": independent_ratio,
            })
        if not np.any(remove_local):
            continue
        keep_masks[index][retained_indices[remove_local]] = False
        removed = int(np.count_nonzero(remove_local))
        component_removed_total += removed
        component_reports.append({
            "input_index": index,
            "component_voxel_size": component_voxel_size,
            "largest_component_points": largest_size,
            "component_size_limit": size_limit,
            "removed_points": removed,
            "removed_components": removed_components,
        })
    report["post_component_cleanup"] = {
        "enabled": True,
        "method": "unsupported_compact_orphan_components_after_point_filter",
        "component_voxel_size": component_voxel_size,
        "removed_points": component_removed_total,
        "views": component_reports,
    }
    report["total_removed_points"] = int(report["total_removed_points"]) + component_removed_total
    return keep_masks, report


def _denoise_registered_views(
    clouds: Sequence[o3d.geometry.PointCloud],
    transformations: Sequence[np.ndarray],
    leveling_transformation: np.ndarray,
    included_views: Sequence[bool],
    voxel_size: float,
    progress_callback: ProgressCallback | None,
    group_name: str,
) -> tuple[list[o3d.geometry.PointCloud], dict[str, object]]:
    masks, report = _cross_view_overlap_keep_masks(
        clouds, transformations, leveling_transformation, included_views, voxel_size,
    )
    filtered = [
        cloud.select_by_index(np.flatnonzero(mask).tolist()) if not mask.all() else cloud
        for cloud, mask in zip(clouds, masks, strict=True)
    ]
    _emit(
        progress_callback,
        f"第 {group_name} 组：配准后跨视角重叠支持去噪，融合前移除 "
        f"{report['total_removed_points']:,} 点；无重叠支持区域保留",
    )
    return filtered, report


def _spatial_component_inlier_mask(
    points: np.ndarray,
    *,
    voxel_size: float,
    minimum_points: int,
    keep_largest: int,
    relative_size: float = 0.02,
    gap_factor: float = 6.0,
    protected_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Keep the largest occupied 3D components after voxelization.

    This targets compact floating noise islands that survive distance filters.
    Components are built on occupied voxels, so the operation is bounded by the
    number of occupied cells rather than the full coordinate range.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or len(coordinates) < minimum_points
        or voxel_size <= 0
    ):
        return np.ones(len(coordinates), dtype=bool)

    protected = np.zeros(len(coordinates), dtype=bool)
    if protected_bounds is not None:
        lower, upper = (np.asarray(value, dtype=np.float64) for value in protected_bounds)
        protected = np.all(
            (coordinates >= lower) & (coordinates <= upper),
            axis=1,
        )
    candidates = ~protected
    if not np.any(candidates):
        return np.ones(len(coordinates), dtype=bool)

    candidate_indices = np.flatnonzero(candidates)
    candidate_coordinates = coordinates[candidate_indices]
    voxel_keys, inverse = np.unique(
        np.floor(candidate_coordinates / float(voxel_size)).astype(np.int64),
        axis=0,
        return_inverse=True,
    )
    if len(voxel_keys) <= 1:
        return np.ones(len(coordinates), dtype=bool)

    component_count, labels = _voxel_component_labels(voxel_keys)
    point_labels = labels[inverse]
    component_sizes = np.bincount(point_labels, minlength=component_count)
    ranked = sorted(
        range(component_count),
        key=lambda index: component_sizes[index],
        reverse=True,
    )
    largest_size = component_sizes[ranked[0]]
    relative_floor = max(int(minimum_points), int(largest_size * 0.03))
    component_centers = np.asarray([
        candidate_coordinates[point_labels == index].mean(axis=0)
        for index in range(component_count)
    ])
    largest_center = component_centers[ranked[0]]
    nearest_gaps = np.full(component_count, np.inf, dtype=np.float64)
    for index in range(component_count):
        if index == ranked[0]:
            nearest_gaps[index] = 0.0
            continue
        distances = np.linalg.norm(component_centers[index] - component_centers, axis=1)
        distances[index] = np.inf
        nearest_gaps[index] = float(np.min(distances))
    gap_limit = float(voxel_size) * float(gap_factor)
    retained = {
        index for index in ranked
        if component_sizes[index] >= relative_floor
        and component_sizes[index] >= int(largest_size * float(relative_size))
        and (
            index == ranked[0]
            or nearest_gaps[index] <= gap_limit
            or component_sizes[index] >= int(largest_size * 0.25)
        )
    }
    # The size threshold is adaptive. Keep_largest is retained in the config
    # for backward-compatible JSON files, but is no longer used as a hard cap.
    if not retained:
        return np.ones(len(coordinates), dtype=bool)
    retained_voxels = np.zeros(len(voxel_keys), dtype=bool)
    for index in retained:
        retained_voxels[labels == index] = True
    result = protected.copy()
    result[candidate_indices] = retained_voxels[inverse]
    return result


def crop_calibration_cloud(
    cloud: o3d.geometry.PointCloud,
    config: CalibrationConfig,
    file_name: str,
    backend: str = "cpu",
) -> o3d.geometry.PointCloud:
    """取多面体标定块的 ROI；关闭 ROI 时使用整个去噪点云。"""
    if not config.use_roi:
        return cloud
    if backend == "gpu":
        calibration_cloud = _select_cloud_by_mask(
            cloud,
            gpu_bounds_mask(np.asarray(cloud.points), config.bounds.min_bound, config.bounds.max_bound),
        )
    else:
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.asarray(config.bounds.min_bound, dtype=np.float64),
            max_bound=np.asarray(config.bounds.max_bound, dtype=np.float64),
        )
        calibration_cloud = cloud.crop(bounding_box)
    if len(calibration_cloud.points) < config.minimum_points:
        raise ValueError(
            f"{file_name} 在标定块 ROI 内仅有 {len(calibration_cloud.points)} 点，"
            f"少于要求的 {config.minimum_points} 点。请检查 ROI 或降低最少点数。"
        )
    return calibration_cloud


def _calibration_plane_features(
    cloud: o3d.geometry.PointCloud,
    config: RegistrationConfig,
) -> list[dict[str, object]]:
    """Extract stable planar patches from a calibration-block ROI.

    The fixture is intentionally treated as a geometric object rather than a
    generic texture cloud.  Several plane normals and plane-centre locations
    provide a much safer pose cue than a single loose ICP pass, especially when
    one view sees only part of a symmetric block.
    """
    if not config.use_calibration_geometry_constraints:
        return []
    sampled = _representative_voxel_down_sample_cloud(
        cloud,
        config.calibration_geometry_voxel_size,
        "cpu",
    )
    remaining = sampled
    features: list[dict[str, object]] = []
    maximum_iterations = 1200
    # Open3D's plane segmentation consumes process-global RNG state. Reset it
    # for every extracted plane so repeated runs do not depend on prior calls.
    base_seed = int(config.global_random_seed) + 104729
    for plane_index in range(int(config.calibration_geometry_max_planes)):
        if len(remaining.points) < int(config.calibration_geometry_minimum_plane_points):
            break
        o3d.utility.random.seed(base_seed + plane_index)
        model, indices = remaining.segment_plane(
            distance_threshold=float(config.calibration_geometry_plane_distance),
            ransac_n=3,
            num_iterations=maximum_iterations,
        )
        if len(indices) < int(config.calibration_geometry_minimum_plane_points):
            break
        plane_points = np.asarray(remaining.points)[np.asarray(indices, dtype=np.int64)]
        center = np.mean(plane_points, axis=0)
        normal = np.asarray(model[:3], dtype=np.float64)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-10:
            remaining = remaining.select_by_index(indices, invert=True)
            continue
        normal /= normal_length
        if normal[2] < 0.0:
            normal = -normal
        centered = plane_points - center
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        if len(singular_values) < 2:
            remaining = remaining.select_by_index(indices, invert=True)
            continue
        span = float(max(singular_values[0], singular_values[1]) * 2.0)
        plane_rmse = float(np.sqrt(np.mean(np.square(centered @ normal))))
        if span <= max(float(config.calibration_geometry_voxel_size) * 3.0, 0.15):
            remaining = remaining.select_by_index(indices, invert=True)
            continue
        features.append(
            {
                "normal": normal,
                "center": center,
                "offset": float(np.dot(normal, center)),
                "point_count": int(len(plane_points)),
                "rmse": plane_rmse,
                "span": span,
            }
        )
        remaining = remaining.select_by_index(indices, invert=True)

    features.sort(key=lambda item: (-int(item["point_count"]), float(item["rmse"])))
    return features


def _calibration_geometry_initial_correction(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    feature_cache: dict[int, tuple[o3d.geometry.PointCloud, list[dict[str, object]]]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate a bounded rigid correction from corresponding fixture planes.

    Matching is based on plane normal and centre geometry, then solved with a
    Kabsch transform over the matched plane centres plus normal directions.  A
    failed or weak match returns identity and a diagnostic status; the caller
    can still use the existing ICP path without losing a valid scan.
    """
    identity = np.eye(4, dtype=np.float64)
    if feature_cache is None:
        source_features = _calibration_plane_features(source, config)
        target_features = _calibration_plane_features(target, config)
    else:
        source_key, target_key = id(source), id(target)
        source_entry = feature_cache.get(source_key)
        if source_entry is None or source_entry[0] is not source:
            source_entry = (source, _calibration_plane_features(source, config))
            feature_cache[source_key] = source_entry
        target_entry = feature_cache.get(target_key)
        if target_entry is None or target_entry[0] is not target:
            target_entry = (target, _calibration_plane_features(target, config))
            feature_cache[target_key] = target_entry
        source_features = source_entry[1]
        target_features = target_entry[1]
    report: dict[str, object] = {
        "status": "disabled" if not config.use_calibration_geometry_constraints else "insufficient_planes",
        "source_plane_count": int(len(source_features)),
        "target_plane_count": int(len(target_features)),
        "matched_plane_count": 0,
        "plane_rmse_before": None,
        "plane_rmse_after": None,
        "correction_rotation_degrees": 0.0,
        "correction_translation": 0.0,
        "matched_plane_pairs": [],
    }
    minimum_planes = int(config.calibration_geometry_minimum_planes)
    if len(source_features) < minimum_planes or len(target_features) < minimum_planes:
        return identity, report

    source_normals = np.asarray([feature["normal"] for feature in source_features], dtype=np.float64)
    target_normals = np.asarray([feature["normal"] for feature in target_features], dtype=np.float64)
    source_centres = np.asarray([feature["center"] for feature in source_features], dtype=np.float64)
    target_centres = np.asarray([feature["center"] for feature in target_features], dtype=np.float64)
    source_spans = np.asarray([float(feature["span"]) for feature in source_features], dtype=np.float64)
    target_spans = np.asarray([float(feature["span"]) for feature in target_features], dtype=np.float64)
    source_counts = np.asarray([float(feature["point_count"]) for feature in source_features], dtype=np.float64)
    target_counts = np.asarray([float(feature["point_count"]) for feature in target_features], dtype=np.float64)
    normal_limit = math.radians(float(config.calibration_geometry_max_normal_angle_degrees))
    centre_limit = float(config.calibration_geometry_max_centroid_distance)
    span_ratio_limit = 3.5

    def build_feature_cost(
        evaluated_source_centres: np.ndarray,
        evaluated_source_normals: np.ndarray,
    ) -> np.ndarray:
        cost = np.full((len(source_features), len(target_features)), 1e6, dtype=np.float64)
        for source_index in range(len(source_features)):
            normal_cosines = np.clip(
                target_normals @ evaluated_source_normals[source_index],
                -1.0,
                1.0,
            )
            normal_angles = np.arccos(np.abs(normal_cosines))
            centre_distances = np.linalg.norm(
                target_centres - evaluated_source_centres[source_index],
                axis=1,
            )
            span_ratios = np.maximum(target_spans, source_spans[source_index]) / np.maximum(
                np.minimum(target_spans, source_spans[source_index]),
                1e-8,
            )
            count_ratios = np.maximum(target_counts, source_counts[source_index]) / np.maximum(
                np.minimum(target_counts, source_counts[source_index]),
                1.0,
            )
            valid = (
                (normal_angles <= normal_limit)
                & (centre_distances <= centre_limit)
                & (span_ratios <= span_ratio_limit)
            )
            cost[source_index, valid] = (
                normal_angles[valid] / max(normal_limit, 1e-8)
                + centre_distances[valid] / max(centre_limit, 1e-8)
                + 0.18 * np.abs(np.log(span_ratios[valid]))
                + 0.05 * np.abs(np.log(count_ratios[valid]))
                - 0.10 * np.log1p(target_counts[valid]) / max(np.log1p(1_000_000.0), 1.0)
            )
        return cost

    def solve_center_transform(
        assignment: list[tuple[int, int]],
    ) -> np.ndarray:
        if len(assignment) < 3:
            return identity.copy()
        source_points = np.asarray(
            [source_centres[source_index] for source_index, _ in assignment],
            dtype=np.float64,
        )
        target_points = np.asarray(
            [target_centres[target_index] for _, target_index in assignment],
            dtype=np.float64,
        )
        source_center = source_points.mean(axis=0)
        target_center = target_points.mean(axis=0)
        covariance = (source_points - source_center).T @ (target_points - target_center)
        left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance)
        rotation = right_vectors_transposed.T @ left_vectors.T
        if np.linalg.det(rotation) < 0.0:
            right_vectors_transposed[-1, :] *= -1.0
            rotation = right_vectors_transposed.T @ left_vectors.T
        transformation = identity.copy()
        transformation[:3, :3] = rotation
        transformation[:3, 3] = target_center - rotation @ source_center
        return transformation

    cost = build_feature_cost(source_centres, source_normals)
    source_assignment, target_assignment = linear_sum_assignment(cost)
    valid_assignment = [
        (int(source_index), int(target_index))
        for source_index, target_index in zip(source_assignment, target_assignment, strict=True)
        if cost[source_index, target_index] < 1e5
    ]
    # Re-score the same planes after a centre-only rigid solve.  This keeps a
    # locally ambiguous first assignment from becoming a globally inconsistent
    # correction while still supporting views that expose only a subset of the
    # fixture faces.
    for _ in range(3):
        if len(valid_assignment) < minimum_planes:
            break
        provisional = solve_center_transform(valid_assignment)
        evaluated_centres = source_centres @ provisional[:3, :3].T + provisional[:3, 3]
        evaluated_normals = source_normals @ provisional[:3, :3].T
        refined_cost = build_feature_cost(evaluated_centres, evaluated_normals)
        refined_source, refined_target = linear_sum_assignment(refined_cost)
        refined_assignment = [
            (int(source_index), int(target_index))
            for source_index, target_index in zip(refined_source, refined_target, strict=True)
            if refined_cost[source_index, target_index] < 1e5
        ]
        if len(refined_assignment) < minimum_planes or refined_assignment == valid_assignment:
            break
        valid_assignment = refined_assignment
    if len(valid_assignment) < minimum_planes:
        report["matched_plane_count"] = int(len(valid_assignment))
        return identity, report

    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    pair_report: list[dict[str, object]] = []
    for source_index, target_index in valid_assignment:
        source_feature = source_features[source_index]
        target_feature = target_features[target_index]
        source_points.append(np.asarray(source_feature["center"], dtype=np.float64))
        target_points.append(np.asarray(target_feature["center"], dtype=np.float64))
        # Normal endpoints make the solved transform respect plane orientation,
        # not only the centroid arrangement.
        normal_weight = min(
            0.35,
            max(
                0.08,
                float(config.calibration_geometry_voxel_size) * 1.5,
            ),
        )
        source_normal = np.asarray(source_feature["normal"], dtype=np.float64)
        target_normal = np.asarray(target_feature["normal"], dtype=np.float64)
        # Plane fitting has an arbitrary sign.  Orient the source normal toward
        # the matched target normal before adding the normal endpoint; without
        # this, a vertical plane fitted with the opposite sign can induce a
        # spurious 180°-like rotation in the Kabsch solve.
        if float(np.dot(source_normal, target_normal)) < 0.0:
            source_normal = -source_normal
        source_points.append(source_points[-1] + normal_weight * source_normal)
        target_points.append(target_points[-1] + normal_weight * target_normal)
        pair_report.append(
            {
                "source_plane": source_index,
                "target_plane": target_index,
                "normal_angle_degrees": float(
                    np.degrees(np.arccos(np.clip(abs(float(np.dot(source_feature["normal"], target_feature["normal"]))), -1.0, 1.0)))
                ),
                "centroid_distance": float(np.linalg.norm(source_feature["center"] - target_feature["center"])),
            }
        )

    source_array = np.asarray(source_points, dtype=np.float64)
    target_array = np.asarray(target_points, dtype=np.float64)
    source_center = source_array.mean(axis=0)
    target_center = target_array.mean(axis=0)
    covariance = (source_array - source_center).T @ (target_array - target_center)
    left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance)
    rotation = right_vectors_transposed.T @ left_vectors.T
    if np.linalg.det(rotation) < 0.0:
        right_vectors_transposed[-1, :] *= -1.0
        rotation = right_vectors_transposed.T @ left_vectors.T
    translation = target_center - rotation @ source_center
    correction = identity.copy()
    correction[:3, :3] = rotation
    correction[:3, 3] = translation
    before_residuals = np.linalg.norm(source_array - target_array, axis=1)
    transformed_source = source_array @ rotation.T + translation
    after_residuals = np.linalg.norm(transformed_source - target_array, axis=1)
    matched_source_centres = np.asarray(
        [source_centres[source_index] for source_index, _ in valid_assignment],
        dtype=np.float64,
    )
    matched_target_centres = np.asarray(
        [target_centres[target_index] for _, target_index in valid_assignment],
        dtype=np.float64,
    )
    if len(valid_assignment) >= 3:
        source_pairwise = np.linalg.norm(
            matched_source_centres[:, None, :] - matched_source_centres[None, :, :],
            axis=2,
        )
        target_pairwise = np.linalg.norm(
            matched_target_centres[:, None, :] - matched_target_centres[None, :, :],
            axis=2,
        )
        upper_triangle = np.triu_indices(len(valid_assignment), k=1)
        pairwise_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(source_pairwise[upper_triangle] - target_pairwise[upper_triangle])
                )
            )
        )
    else:
        pairwise_rmse = 0.0
    rotation_degrees = _rotation_angle_degrees(rotation)
    translation_norm = float(np.linalg.norm(translation))
    report.update(
        {
            "status": "ok",
            "matched_plane_count": int(len(valid_assignment)),
            "plane_rmse_before": float(np.sqrt(np.mean(np.square(before_residuals)))),
            "plane_rmse_after": float(np.sqrt(np.mean(np.square(after_residuals)))),
            "plane_pairwise_rmse": pairwise_rmse,
            "correction_rotation_degrees": rotation_degrees,
            "correction_translation": translation_norm,
            "matched_plane_pairs": pair_report,
        }
    )
    # Geometry is an initialiser, not a licence to move a station arbitrarily.
    # Large corrections indicate an ambiguous match and are rejected so the
    # existing bounded ICP/pose-prior path remains the authority.
    if rotation_degrees > max(float(config.pose_prior_rigid_max_rotation_degrees) * 1.5, 2.0):
        report["status"] = "rejected_rotation_bound"
        return identity, report
    if translation_norm > max(float(config.pose_prior_rigid_max_translation) * 1.5, 0.45):
        report["status"] = "rejected_translation_bound"
        return identity, report
    pairwise_limit = max(0.35, min(0.75, centre_limit * 0.25))
    if pairwise_rmse > pairwise_limit:
        report["status"] = "rejected_pairwise_geometry"
        return identity, report
    if not np.isfinite(report["plane_rmse_after"]) or report["plane_rmse_after"] > report["plane_rmse_before"] * 0.98:
        report["status"] = "rejected_no_improvement"
        return identity, report
    return correction, report


def _calibration_geometry_metric_report(report: Mapping[str, object]) -> dict[str, object]:
    """Keep the geometry decision compact and JSON serializable."""
    keys = (
        "status",
        "source_plane_count",
        "target_plane_count",
        "matched_plane_count",
        "plane_rmse_before",
        "plane_rmse_after",
        "plane_pairwise_rmse",
        "correction_rotation_degrees",
        "correction_translation",
        "matched_plane_pairs",
        "icp_baseline_fitness",
        "icp_baseline_rmse",
        "icp_final_fitness",
        "icp_final_rmse",
    )
    result: dict[str, object] = {}
    for key in keys:
        value = report.get(key)
        if value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = [
                _calibration_geometry_metric_report(item)
                if isinstance(item, Mapping)
                else float(item) if isinstance(item, (np.floating, np.integer)) else item
                for item in value
            ]
    if isinstance(report.get("global_pose_disambiguation"), Mapping):
        result["global_pose_disambiguation"] = dict(report["global_pose_disambiguation"])
    return result


def remove_calibration_region(
    cloud: o3d.geometry.PointCloud,
    config: CalibrationConfig,
    backend: str = "cpu",
    *,
    coordinate_transformation: np.ndarray | None = None,
    margin: float | None = None,
    component_voxel_size: float | None = None,
) -> o3d.geometry.PointCloud:
    """Remove the fixture ROI from an already globally aligned cloud.

    ``CalibrationConfig.bounds`` are defined in the unlevelled reference frame.
    When the cloud has already received the leveling transform, points are first
    mapped back through its inverse before testing the ROI.  This avoids turning
    a rotated axis-aligned ROI into a larger world-space box that could erase
    nearby workpiece points.
    """
    if not config.use_roi:
        return copy.deepcopy(cloud)
    points = np.asarray(cloud.points)
    if not len(points):
        return copy.deepcopy(cloud)
    local_points = np.asarray(points, dtype=np.float64)
    if coordinate_transformation is not None:
        transformation = np.asarray(coordinate_transformation, dtype=np.float64)
        if transformation.shape != (4, 4):
            raise ValueError("标定块排除坐标变换必须是形状为 (4, 4) 的矩阵。")
        inverse = np.linalg.inv(transformation)
        local_points = local_points @ inverse[:3, :3].T + inverse[:3, 3]
    normalized_margin = (
        float(config.exclusion_margin) if margin is None else float(margin)
    )
    if normalized_margin < 0:
        raise ValueError("标定块融合排除边界余量不能为负数。")
    minimum_bound = np.asarray(config.bounds.min_bound, dtype=np.float64) - normalized_margin
    maximum_bound = np.asarray(config.bounds.max_bound, dtype=np.float64) + normalized_margin
    inside = (
        gpu_bounds_mask(local_points, minimum_bound, maximum_bound)
        if backend == "gpu"
        else np.all(
            (local_points >= minimum_bound)
            & (local_points <= maximum_bound),
            axis=1,
        )
    )
    if component_voxel_size is not None:
        component_config = copy.deepcopy(config)
        component_config.exclusion_margin = normalized_margin
        inside = fixture_component_mask(local_points, component_config, component_voxel_size)
    return _select_cloud_by_mask(cloud, ~np.asarray(inside, dtype=bool))


def _identity_leveling_result() -> LevelingResult:
    return LevelingResult(
        transformation=np.eye(4, dtype=np.float64),
        reference_plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        reference_plane_center=np.zeros(3, dtype=np.float64),
        plane_points=0,
        tilt_degrees=0.0,
        plane_rmse=0.0,
        applied=False,
    )


def _rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_vector = np.asarray(source, dtype=np.float64).reshape(3)
    target_vector = np.asarray(target, dtype=np.float64).reshape(3)
    source_vector /= np.linalg.norm(source_vector)
    target_vector /= np.linalg.norm(target_vector)
    cross_product = np.cross(source_vector, target_vector)
    sine = float(np.linalg.norm(cross_product))
    cosine = float(np.clip(np.dot(source_vector, target_vector), -1.0, 1.0))
    if sine <= 1e-10:
        if cosine > 0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source_vector[0]) > 0.8:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis -= source_vector * np.dot(axis, source_vector)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    skew = np.array(
        [
            [0.0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0.0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(points, dtype=np.float64)
    center = coordinates.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(coordinates - center, full_matrices=False)
    normal = np.asarray(right_vectors[-1], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    residuals = (coordinates - center) @ normal
    return normal, center, residuals


def _estimate_horizontal_reference_plane(
    points: np.ndarray,
    distance_threshold: float,
    minimum_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    coordinates = np.asarray(points, dtype=np.float64)
    point_count = len(coordinates)
    if point_count < minimum_points:
        return None

    random_generator = np.random.default_rng(9137)
    candidate_count = min(2_000, max(500, point_count // 4))
    candidates: list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(candidate_count):
        sample_indices = random_generator.choice(point_count, size=3, replace=False)
        first, second, third = coordinates[sample_indices]
        normal = np.cross(second - first, third - first)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-10:
            continue
        normal /= normal_length
        if normal[2] < 0:
            normal = -normal
        if normal[2] < 0.80:
            continue

        initial_inlier_indices = np.flatnonzero(np.abs((coordinates - first) @ normal) <= distance_threshold)
        if len(initial_inlier_indices) < minimum_points:
            continue
        fitted_normal, fitted_center, _ = _fit_plane(coordinates[initial_inlier_indices])
        if fitted_normal[2] < 0.80:
            continue
        inlier_indices = np.flatnonzero(np.abs((coordinates - fitted_center) @ fitted_normal) <= distance_threshold)
        if len(inlier_indices) < minimum_points:
            continue
        fitted_normal, fitted_center, residuals = _fit_plane(coordinates[inlier_indices])
        if fitted_normal[2] < 0.80:
            continue
        inlier_indices = np.flatnonzero(np.abs((coordinates - fitted_center) @ fitted_normal) <= distance_threshold)
        if len(inlier_indices) < minimum_points:
            continue
        fitted_normal, fitted_center, residuals = _fit_plane(coordinates[inlier_indices])
        plane_rmse = float(np.sqrt(np.mean(np.square(residuals))))
        candidates.append((len(inlier_indices), plane_rmse, fitted_normal, fitted_center, inlier_indices))

    if not candidates:
        return None
    maximum_inliers = max(candidate[0] for candidate in candidates)
    supported_candidates = [
        candidate for candidate in candidates if candidate[0] >= maximum_inliers * 0.95
    ]
    _, _, normal, center, inlier_indices = min(
        supported_candidates,
        key=lambda candidate: (candidate[1], -candidate[0], -candidate[2][2]),
    )
    return normal, center, inlier_indices


def estimate_calibration_leveling(
    calibration_cloud: o3d.geometry.PointCloud,
    config: CalibrationConfig,
) -> LevelingResult:
    """从标定块中寻找近水平基准面，并生成绕该面中心的调平变换。"""
    if not config.level_reference_plane:
        return _identity_leveling_result()

    point_count = len(calibration_cloud.points)
    if point_count < config.leveling_minimum_points:
        raise ValueError(
            f"标定块调平需要至少 {config.leveling_minimum_points} 个点，当前只有 {point_count} 个点。"
        )

    points = np.asarray(calibration_cloud.points)
    fitted_plane = _estimate_horizontal_reference_plane(
        points,
        config.leveling_plane_distance,
        config.leveling_minimum_points,
    )
    if fitted_plane is None:
        raise ValueError(
            "标定块 ROI 中未找到可靠的近水平基准面，无法确认目标区域是否已调平。"
            "请检查标定块 ROI 是否完整覆盖其水平面。"
        )
    normal, center, inlier_indices = fitted_plane
    _, _, residuals = _fit_plane(points[inlier_indices])
    inlier_count = len(inlier_indices)
    plane_rmse = float(np.sqrt(np.mean(np.square(residuals))))

    rotation = _rotation_aligning_vectors(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = center - rotation @ center
    tilt_degrees = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    return LevelingResult(
        transformation=transformation,
        reference_plane_normal=normal,
        reference_plane_center=center,
        plane_points=inlier_count,
        tilt_degrees=tilt_degrees,
        plane_rmse=plane_rmse,
        applied=tilt_degrees > 1e-6,
    )


def estimate_reference_leveling(
    reference_file: Path,
    denoise_config: DenoiseConfig,
    calibration_config: CalibrationConfig,
    compute_backend: str = "auto",
) -> LevelingResult:
    """用与正式处理相同的去噪流程，预先计算参考视角的调平变换。"""
    if not calibration_config.level_reference_plane:
        return _identity_leveling_result()
    active_backend = resolve_compute_backend(compute_backend)
    raw_cloud = read_point_cloud(reference_file)
    cleaned_cloud, _ = denoise_cloud(raw_cloud, denoise_config, reference_file.name, active_backend)
    calibration_cloud = crop_calibration_cloud(cleaned_cloud, calibration_config, reference_file.name, active_backend)
    return estimate_calibration_leveling(calibration_cloud, calibration_config)


def select_reference_index_by_calibration_quality(
    qualities: Sequence[tuple[int, int, float] | None],
) -> int:
    """Keep the preferred reference when valid; otherwise use the best calibration-block coverage."""
    if not qualities:
        raise ValueError("至少需要一个候选参考视角。")
    if qualities[0] is not None:
        return 0

    candidates = [
        (index, quality)
        for index, quality in enumerate(qualities)
        if quality is not None
    ]
    if not candidates:
        raise ValueError("所有候选视角都无法提取可靠的标定块基准面。")
    return max(
        candidates,
        key=lambda item: (
            item[1][0],
            item[1][1],
            -item[1][2],
            -item[0],
        ),
    )[0]


def _select_calibration_reference(
    files: Sequence[Path],
    cleaned_clouds: Sequence[o3d.geometry.PointCloud],
    calibration_config: CalibrationConfig,
    backend: str,
) -> tuple[int, o3d.geometry.PointCloud, LevelingResult]:
    """Preserve the preferred frame unless its fixture coverage is severely truncated."""
    if len(files) != len(cleaned_clouds):
        raise ValueError("参考视角文件和去噪点云数量必须一致。")
    if not files:
        raise ValueError("无法从空点云列表中选择参考视角。")

    counts = []
    for cloud in cleaned_clouds:
        points = np.asarray(cloud.points)
        counts.append(int(np.count_nonzero(np.all(
            (points >= calibration_config.bounds.min_bound)
            & (points <= calibration_config.bounds.max_bound), axis=1)))
            if calibration_config.use_roi else len(points))
    order = sorted(range(len(files)), key=lambda index: (-counts[index], index))
    if counts[0] >= max(counts) * .25:
        order = [0, *(index for index in order if index != 0)]
    failures: list[str | None] = [None] * len(files)
    for index in order:
        path, cloud = files[index], cleaned_clouds[index]
        try:
            calibration_cloud = crop_calibration_cloud(cloud, calibration_config, path.name, backend)
            leveling = estimate_calibration_leveling(calibration_cloud, calibration_config)
        except ValueError as error:
            # Sparse fixture surfaces must not be judged by the workpiece
            # denoiser.  Retry only the calibration ROI on the raw scan.
            try:
                raw_calibration = crop_calibration_cloud(
                    read_point_cloud(path),
                    calibration_config,
                    path.name,
                    backend,
                )
                leveling = estimate_calibration_leveling(
                    raw_calibration,
                    calibration_config,
                )
                calibration_cloud = raw_calibration
            except ValueError:
                failures[index] = str(error)
                continue

        return index, calibration_cloud, leveling

    details = "；".join(
        f"{path.name}: {failure}"
        for path, failure in zip(files, failures)
        if failure is not None
    )
    raise ValueError(f"没有可用于对齐的标定块参考视角。{details}")


def select_reference_file_with_leveling(
    files: Sequence[Path],
    target_vertex_count: int,
    denoise_config: DenoiseConfig,
    calibration_config: CalibrationConfig,
    compute_backend: str = "auto",
) -> tuple[Path, LevelingResult]:
    """Choose a preview reference that has a usable calibration block and return its leveling transform."""
    ordered_files = order_files_with_reference(files, target_vertex_count)
    active_backend = resolve_compute_backend(compute_backend)
    cleaned_clouds: list[o3d.geometry.PointCloud] = []
    for path in ordered_files:
        raw_cloud = read_point_cloud(path)
        cleaned_cloud, _ = denoise_cloud(
            raw_cloud, denoise_config, path.name, active_backend,
            protected_bounds=(calibration_config.bounds.min_bound, calibration_config.bounds.max_bound)
            if calibration_config.use_roi else None)
        cleaned_clouds.append(cleaned_cloud)

    reference_index, _, leveling = _select_calibration_reference(
        ordered_files,
        cleaned_clouds,
        calibration_config,
        active_backend,
    )
    return ordered_files[reference_index], leveling


def prepare_registration_cloud(
    cloud: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    backend: str = "cpu",
) -> o3d.geometry.PointCloud:
    registration_cloud = _voxel_down_sample_cloud(cloud, config.registration_voxel_size, backend)
    if len(registration_cloud.points) < 30:
        raise ValueError("用于配准的点不足 30 个，请减小拼接体素大小或扩大标定块 ROI。")
    radius = config.registration_voxel_size * config.normal_radius_factor
    registration_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60))
    registration_cloud.normalize_normals()
    return registration_cloud


def _cached_registration_cloud(
    cloud: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    backend: str,
    cache: dict[str, dict] | None,
) -> o3d.geometry.PointCloud:
    if cache is None:
        return prepare_registration_cloud(cloud, config, backend)
    prepared = cache.setdefault("prepared", {})
    key = (id(cloud), float(config.registration_voxel_size),
           float(config.normal_radius_factor), backend)
    entry = prepared.get(key)
    if entry is None or entry[0] is not cloud:
        entry = (cloud, prepare_registration_cloud(cloud, config, backend))
        prepared[key] = entry
    return entry[1]


def _cached_fpfh_feature(
    cloud: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    backend: str,
    cache: dict[str, dict] | None,
):
    prepared = _cached_registration_cloud(cloud, config, backend, cache)
    if cache is None:
        feature_radius = config.registration_voxel_size * config.feature_radius_factor
        return o3d.pipelines.registration.compute_fpfh_feature(
            prepared,
            o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
        )
    features = cache.setdefault("features", {})
    key = (id(prepared), round(float(config.registration_voxel_size), 12),
           round(float(config.feature_radius_factor), 12), backend)
    entry = features.get(key)
    if entry is None or entry[0] is not prepared:
        feature_radius = config.registration_voxel_size * config.feature_radius_factor
        entry = (
            prepared,
            o3d.pipelines.registration.compute_fpfh_feature(
                prepared,
                o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
            ),
        )
        features[key] = entry
    return entry[1]


def coarse_global_register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    backend: str = "cpu",
    *,
    candidate_scorer: Callable[[np.ndarray], float] | None = None,
    registration_cache: dict[str, dict] | None = None,
) -> tuple[np.ndarray, float | None, float | None]:
    """用传入的配准点云执行 FPFH/RANSAC 初始定位。

    调用方决定 ``source`` 和 ``target`` 的语义：标定块初配准使用 ROI，
    整体朝向消歧与移出 ROI 的定位可以使用完整场景。
    """
    if not config.use_global_registration:
        return np.eye(4), None, None

    o3d.utility.random.seed(config.global_random_seed)
    source_registration = _cached_registration_cloud(source, config, backend, registration_cache)
    target_registration = _cached_registration_cloud(target, config, backend, registration_cache)
    source_feature = _cached_fpfh_feature(source, config, backend, registration_cache)
    target_feature = _cached_fpfh_feature(target, config, backend, registration_cache)
    global_distance = config.registration_voxel_size * config.global_distance_factor
    best_result: o3d.pipelines.registration.RegistrationResult | None = None
    best_score = -math.inf
    for _ in range(config.global_attempts):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_registration,
            target_registration,
            source_feature,
            target_feature,
            True,
            global_distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(global_distance),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(config.global_max_iterations, 0.999),
        )
        score = (
            candidate_scorer(np.asarray(result.transformation))
            if candidate_scorer is not None and result.fitness > 0
            else float(result.fitness)
        )
        if best_result is None or score > best_score or (
            score == best_score and result.inlier_rmse < best_result.inlier_rmse
        ):
            best_result = result
            best_score = score

    assert best_result is not None
    transformation = best_result.transformation if best_result.fitness > 0 else np.eye(4)
    return transformation, float(best_result.fitness), float(best_result.inlier_rmse)


def coarse_calibration_register(
    source_calibration: o3d.geometry.PointCloud,
    target_calibration: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    backend: str = "cpu",
    registration_cache: dict[str, dict] | None = None,
) -> tuple[np.ndarray, float | None, float | None]:
    """仅使用标定块点云做粗配准，明确隔离目标物体几何。

    该命名入口用于非固定工位数量（例如 7/8 个视角）的流程，避免
    调用处误把完整工件点云传给 FPFH/RANSAC。返回值与
    :func:`coarse_global_register` 保持一致。
    """
    return coarse_global_register(
        source_calibration, target_calibration, config, backend,
        registration_cache=registration_cache,
    )
def _refine_with_icp_diagnostics(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    backend: str = "cpu",
    registration_cache: dict[str, dict] | None = None,
    feature_cache: dict[int, tuple[o3d.geometry.PointCloud, list[dict[str, object]]]] | None = None,
) -> tuple[np.ndarray, float, float, dict[str, object]]:
    """Run geometry-seeded ICP and return an auditable registration report."""
    geometry_initial, geometry_report = _calibration_geometry_initial_correction(
        source,
        target,
        config,
        feature_cache,
    )
    source_registration = _cached_registration_cloud(source, config, backend, registration_cache)
    target_registration = _cached_registration_cloud(target, config, backend, registration_cache)
    if backend == "gpu":
        transformation, fitness, rmse = gpu_refine_icp(
            np.asarray(source_registration.points),
            np.asarray(target_registration.points),
            np.asarray(target_registration.normals),
            config.icp_max_correspondence_distance,
            config.icp_max_iterations,
            config.use_point_to_plane_icp,
            initial_transformation=geometry_initial,
        )
        geometry_report["icp_initial_transformation"] = geometry_initial.round(12).tolist()
        return transformation, fitness, rmse, geometry_report

    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
        if config.use_point_to_plane_icp
        else o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    if not config.use_multiscale_icp:
        result = o3d.pipelines.registration.registration_icp(
            source_registration,
            target_registration,
            config.icp_max_correspondence_distance,
            geometry_initial,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.icp_max_iterations),
        )
        geometry_report["icp_initial_transformation"] = geometry_initial.round(12).tolist()
        return result.transformation, float(result.fitness), float(result.inlier_rmse), geometry_report

    # A single loose ICP pass can bridge two parallel fixture faces.  Tighten
    # the correspondence gate in stages and only keep a stage when its quality
    # is not materially worse than the previous one.  The returned transform is
    # always rigid and remains anchored at the caller's input pose.
    stages = (
        config.icp_max_correspondence_distance,
        max(config.icp_max_correspondence_distance * 0.50, 0.06),
        max(config.icp_max_correspondence_distance * 0.25, 0.035),
    )
    transformation = np.asarray(geometry_initial, dtype=np.float64).copy()
    baseline = o3d.pipelines.registration.evaluate_registration(
        source_registration,
        target_registration,
        config.icp_max_correspondence_distance,
        transformation,
    )
    baseline_quality = full_cloud_alignment_quality(
        float(baseline.fitness),
        float(baseline.inlier_rmse),
        float(config.icp_max_correspondence_distance),
    )[0]
    best_quality = baseline_quality
    best_transformation = transformation.copy()
    for stage_distance in stages:
        result = o3d.pipelines.registration.registration_icp(
            source_registration,
            target_registration,
            float(stage_distance),
            transformation,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max(20, int(config.icp_max_iterations // (2 if stage_distance < stages[0] else 1)))
            ),
        )
        evaluated_stage = o3d.pipelines.registration.evaluate_registration(
            source_registration, target_registration,
            config.icp_max_correspondence_distance, result.transformation,
        )
        quality = full_cloud_alignment_quality(
            float(evaluated_stage.fitness),
            float(evaluated_stage.inlier_rmse),
            float(config.icp_max_correspondence_distance),
        )[0]
        if quality >= best_quality:
            transformation = np.asarray(result.transformation, dtype=np.float64)
            best_quality = quality
            best_transformation = transformation.copy()
    transformation = best_transformation
    # Report against the original configured gate so quality numbers stay
    # comparable with historical runs and the calibration quality gate.
    evaluated = o3d.pipelines.registration.evaluate_registration(
        source_registration,
        target_registration,
        config.icp_max_correspondence_distance,
        transformation,
    )
    geometry_report["icp_initial_transformation"] = geometry_initial.round(12).tolist()
    geometry_report["icp_baseline_fitness"] = float(baseline.fitness)
    geometry_report["icp_baseline_rmse"] = float(baseline.inlier_rmse)
    geometry_report["icp_baseline_quality"] = float(baseline_quality)
    geometry_report["icp_final_fitness"] = float(evaluated.fitness)
    geometry_report["icp_final_rmse"] = float(evaluated.inlier_rmse)
    return transformation, float(evaluated.fitness), float(evaluated.inlier_rmse), geometry_report


def refine_with_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: RegistrationConfig,
    backend: str = "cpu",
    registration_cache: dict[str, dict] | None = None,
    feature_cache: dict[int, tuple[o3d.geometry.PointCloud, list[dict[str, object]]]] | None = None,
) -> tuple[np.ndarray, float, float]:
    """在标定块区域上执行几何约束初始化后的多阶段 ICP。"""
    transformation, fitness, rmse, _ = _refine_with_icp_diagnostics(
        source,
        target,
        config,
        backend,
        registration_cache,
        feature_cache,
    )
    return transformation, fitness, rmse


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def refine_translation_only(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transformation: np.ndarray,
    maximum_correspondence_distance: float = 0.90,
) -> tuple[np.ndarray, float, float]:
    """Refine only translation so a symmetric calibration block cannot rotate to another face."""
    source_points = np.asarray(source.points, dtype=np.float64)
    target_points = np.asarray(target.points, dtype=np.float64)
    source_points = source_points[np.isfinite(source_points).all(axis=1)]
    target_points = target_points[np.isfinite(target_points).all(axis=1)]
    if len(source_points) < 30 or len(target_points) < 30:
        raise ValueError('锁旋转平移精调要求源点云和目标点云各至少包含 30 个有效点。')
    if maximum_correspondence_distance <= 0:
        raise ValueError('锁旋转平移精调的最大对应距离必须大于 0。')

    initial = np.asarray(initial_transformation, dtype=np.float64)
    if initial.shape != (4, 4):
        raise ValueError('锁旋转平移精调需要形状为 (4, 4) 的刚体变换。')

    provisional_result = o3d.pipelines.registration.registration_icp(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points)),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points)),
        maximum_correspondence_distance,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    provisional_transformation = np.asarray(provisional_result.transformation, dtype=np.float64)
    provisional_rotation_delta = provisional_transformation[:3, :3] @ initial[:3, :3].T
    provisional_translation_delta = provisional_transformation[:3, 3] - initial[:3, 3]
    if (
        provisional_result.fitness > 0
        and math.isfinite(float(provisional_result.inlier_rmse))
        and _rotation_angle_degrees(provisional_rotation_delta) <= 1.5
        and float(np.linalg.norm(provisional_translation_delta)) <= maximum_correspondence_distance
    ):
        transformation = initial.copy()
        transformation[:3, 3] = provisional_transformation[:3, 3]
        return (
            transformation,
            float(provisional_result.fitness),
            float(provisional_result.inlier_rmse),
        )

    rotated_source = source_points @ initial[:3, :3].T
    translation = initial[:3, 3].copy()
    target_search = o3d.core.nns.NearestNeighborSearch(
        o3d.core.Tensor(target_points.astype(np.float32, copy=False))
    )
    target_search.knn_index()

    def nearest_distances(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices, squared_distances = target_search.knn_search(
            o3d.core.Tensor(points.astype(np.float32, copy=False)),
            1,
        )
        return indices.numpy().reshape(-1), np.sqrt(squared_distances.numpy().reshape(-1))

    stage_distances = (
        maximum_correspondence_distance,
        maximum_correspondence_distance * 0.65,
        maximum_correspondence_distance * 0.40,
    )
    for stage_distance in stage_distances:
        for _ in range(80):
            aligned_source = rotated_source + translation
            nearest_indices, distances = nearest_distances(aligned_source)
            candidate_mask = distances <= stage_distance
            if int(candidate_mask.sum()) < 30:
                break
            robust_limit = max(float(np.quantile(distances[candidate_mask], 0.45)), 0.015)
            inlier_mask = candidate_mask & (distances <= min(robust_limit, stage_distance))
            if int(inlier_mask.sum()) < 30:
                break
            residuals = target_points[nearest_indices[inlier_mask]] - aligned_source[inlier_mask]
            residual_lengths = np.linalg.norm(residuals, axis=1)
            scale = max(float(np.quantile(residual_lengths, 0.80)), 0.015)
            weights = np.minimum(1.0, scale / np.maximum(residual_lengths, 1e-8))
            step = np.average(residuals, axis=0, weights=weights)
            step_length = float(np.linalg.norm(step))
            if step_length > 0.12:
                step *= 0.12 / step_length
                step_length = 0.12
            translation += step
            if step_length < 1e-4:
                break

    aligned_source = rotated_source + translation
    _, final_distances = nearest_distances(aligned_source)
    final_limit = min(maximum_correspondence_distance * 0.40, 0.35)
    inlier_mask = final_distances <= final_limit
    fitness = float(np.mean(inlier_mask))
    rmse = (
        float(np.sqrt(np.mean(np.square(final_distances[inlier_mask]))))
        if inlier_mask.any()
        else math.inf
    )
    transformation = initial.copy()
    transformation[:3, 3] = translation
    return transformation, fitness, rmse


def evaluate_full_cloud_alignment(
    source_cloud: o3d.geometry.PointCloud,
    reference_cloud: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    config: RegistrationConfig,
    backend: str = "cpu",
) -> tuple[float, float]:
    """只用于验收候选位姿，确认标定块对齐后工件全云没有落到错误位置。"""
    transformed_source = _transform_cloud(source_cloud, transformation, backend)
    validation_source = _voxel_down_sample_cloud(
        transformed_source,
        config.registration_voxel_size,
        backend,
    )
    if not len(validation_source.points) or not len(reference_cloud.points):
        return 0.0, math.inf
    result = o3d.pipelines.registration.evaluate_registration(
        validation_source,
        reference_cloud,
        config.icp_max_correspondence_distance,
        np.eye(4),
    )
    return float(result.fitness), float(result.inlier_rmse)


def full_cloud_alignment_quality(
    fitness: float,
    rmse: float,
    maximum_correspondence_distance: float,
) -> tuple[float, float, float]:
    if not np.isfinite(fitness) or not np.isfinite(rmse):
        return (-math.inf, -math.inf, -math.inf)
    normalized_rmse = min(max(rmse, 0.0) / maximum_correspondence_distance, 1.0)
    return (fitness * (1.0 - normalized_rmse), fitness, -rmse)


def calibration_registration_quality(
    calibration_points: int,
    fitness: float,
    rmse: float,
    calibration_config: CalibrationConfig,
    registration_config: RegistrationConfig,
) -> tuple[bool, str | None]:
    """Apply the fixture-only quality gate used before pose-graph fusion."""
    reasons: list[str] = []
    if int(calibration_points) < int(calibration_config.minimum_points):
        reasons.append(
            f"calibration points {int(calibration_points)} < "
            f"{int(calibration_config.minimum_points)}"
        )
    if not np.isfinite(fitness) or float(fitness) < registration_config.minimum_calibration_icp_fitness:
        reasons.append(
            f"calibration fitness {float(fitness):.3f} < "
            f"{registration_config.minimum_calibration_icp_fitness:.3f}"
        )
    if not np.isfinite(rmse) or float(rmse) > registration_config.maximum_calibration_icp_rmse:
        rendered_rmse = f"{float(rmse):.4f}" if np.isfinite(rmse) else "non-finite"
        reasons.append(
            f"calibration RMSE {rendered_rmse} > "
            f"{registration_config.maximum_calibration_icp_rmse:.4f}"
        )
    return (not reasons, "; ".join(reasons) if reasons else None)


def fine_full_cloud_registration_config(config: RegistrationConfig) -> RegistrationConfig:
    """为工件最终全云对齐创建比粗配准更细的 ICP 参数。"""
    fine_config = copy.deepcopy(config)
    fine_config.registration_voxel_size = min(config.registration_voxel_size, 0.15)
    fine_config.icp_max_correspondence_distance = min(config.icp_max_correspondence_distance, 0.45)
    fine_config.icp_max_iterations = max(config.icp_max_iterations, 200)
    return fine_config


def _pairwise_pose_graph_edge(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    source_transform: np.ndarray,
    target_transform: np.ndarray,
    registration_config: RegistrationConfig,
    stage: str,
    source_index: int,
    target_index: int,
    source_label: str,
    target_label: str,
    edge_type: str,
    uncertain: bool,
    maximum_rotation_degrees: float,
    maximum_translation: float,
    minimum_fitness: float,
    backend: str = "cpu",
    maximum_rmse: float | None = None,
    registration_cache: dict[str, dict] | None = None,
    feature_cache: dict[int, tuple[o3d.geometry.PointCloud, list[dict[str, object]]]] | None = None,
) -> tuple[o3d.pipelines.registration.PoseGraphEdge | None, PoseGraphEdgeMetric]:
    aligned_source = _transform_cloud(source_cloud, source_transform, backend)
    aligned_target = _transform_cloud(target_cloud, target_transform, backend)
    initial = o3d.pipelines.registration.evaluate_registration(
        aligned_source,
        aligned_target,
        registration_config.icp_max_correspondence_distance,
        np.eye(4),
    )
    correction, fitness, rmse = refine_with_icp(
        aligned_source,
        aligned_target,
        registration_config,
        backend,
        registration_cache,
        feature_cache,
    )
    rotation_degrees = _rotation_angle_degrees(correction[:3, :3])
    translation = float(np.linalg.norm(correction[:3, 3]))
    initial_quality = full_cloud_alignment_quality(
        float(initial.fitness),
        float(initial.inlier_rmse),
        registration_config.icp_max_correspondence_distance,
    )[0]
    candidate_quality = full_cloud_alignment_quality(
        fitness,
        rmse,
        registration_config.icp_max_correspondence_distance,
    )[0]
    rejection_reason: str | None = None
    if not np.isfinite(fitness) or fitness < minimum_fitness:
        rejection_reason = f"fitness {fitness:.3f} < {minimum_fitness:.3f}"
    elif not np.isfinite(rmse):
        rejection_reason = "RMSE is non-finite"
    elif maximum_rmse is not None and rmse > maximum_rmse:
        rejection_reason = f"RMSE {rmse:.4f} > {maximum_rmse:.4f}"
    elif rotation_degrees > maximum_rotation_degrees:
        rejection_reason = f"rotation {rotation_degrees:.3f} > {maximum_rotation_degrees:.3f}"
    elif translation > maximum_translation:
        rejection_reason = f"translation {translation:.4f} > {maximum_translation:.4f}"
    elif candidate_quality < initial_quality - 0.015:
        rejection_reason = "ICP quality regressed"

    metric = PoseGraphEdgeMetric(
        stage=stage,
        source_index=source_index,
        target_index=target_index,
        source_label=source_label,
        target_label=target_label,
        edge_type=edge_type,
        uncertain=uncertain,
        initial_fitness=float(initial.fitness),
        initial_rmse=float(initial.inlier_rmse),
        fitness=fitness,
        rmse=rmse,
        correction_rotation_degrees=rotation_degrees,
        correction_translation=translation,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
    )
    if rejection_reason is not None:
        return None, metric

    relative_transform = np.linalg.inv(target_transform) @ correction @ source_transform
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        aligned_source,
        aligned_target,
        registration_config.icp_max_correspondence_distance,
        correction,
    )
    edge = o3d.pipelines.registration.PoseGraphEdge(
        source_index,
        target_index,
        relative_transform,
        information,
        uncertain,
    )
    return edge, metric


def optimize_multiframe_pose_graph(
    initial_transforms: Sequence[np.ndarray],
    edges: Sequence[o3d.pipelines.registration.PoseGraphEdge],
    maximum_correspondence_distance: float,
    reference_index: int = 0,
    maximum_rotation_degrees: float | None = None,
    maximum_translation: float | None = None,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Optimize source-to-global poses and reject node corrections outside bounds."""
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for transformation in initial_transforms:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.asarray(transformation)))
    for edge in edges:
        pose_graph.edges.append(edge)
    if edges:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=maximum_correspondence_distance,
                edge_prune_threshold=0.25,
                preference_loop_closure=1.5,
                reference_node=reference_index,
            ),
        )

    optimized: list[np.ndarray] = []
    corrections: list[dict[str, object]] = []
    for index, (initial, node) in enumerate(zip(initial_transforms, pose_graph.nodes, strict=True)):
        candidate = np.asarray(node.pose)
        correction = candidate @ np.linalg.inv(initial)
        rotation_degrees = _rotation_angle_degrees(correction[:3, :3])
        translation = float(np.linalg.norm(correction[:3, 3]))
        accepted = index == reference_index or (
            (maximum_rotation_degrees is None or rotation_degrees <= maximum_rotation_degrees)
            and (maximum_translation is None or translation <= maximum_translation)
        )
        optimized.append(candidate.copy() if accepted else np.asarray(initial).copy())
        corrections.append(
            {
                "node_index": index,
                "rotation_degrees": rotation_degrees,
                "translation": translation,
                "accepted": accepted,
            }
        )
    return optimized, corrections


def build_and_optimize_pairwise_pose_graph(
    aligned_clouds: Sequence[o3d.geometry.PointCloud],
    labels: Sequence[str],
    registration_config: RegistrationConfig,
    stage: str,
    maximum_rotation_degrees: float,
    maximum_translation: float,
    minimum_adjacent_fitness: float,
    minimum_loop_fitness: float,
    backend: str = "cpu",
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
    *,
    active_indices: Sequence[int] | None = None,
    reference_label: str = "reference",
    maximum_adjacent_rmse: float | None = None,
    maximum_loop_rmse: float | None = None,
    registration_cache: dict[str, dict] | None = None,
    feature_cache: dict[int, tuple[o3d.geometry.PointCloud, list[dict[str, object]]]] | None = None,
) -> tuple[list[np.ndarray], list[PoseGraphEdgeMetric], list[dict[str, object]]]:
    """Build adjacent and overlap edges between globally aligned frames, then optimize corrections."""
    total_views = len(aligned_clouds)
    if len(labels) != total_views:
        raise ValueError("位姿图点云和标签数量必须一致。")
    if active_indices is None:
        working_indices = list(range(total_views))
    else:
        working_indices = list(dict.fromkeys(int(index) for index in active_indices))
        invalid_indices = [index for index in working_indices if index < 0 or index >= total_views]
        if invalid_indices:
            raise ValueError(f"位姿图活动节点索引越界：{invalid_indices[0]}。")
    if not working_indices:
        return [np.eye(4, dtype=np.float64) for _ in aligned_clouds], [], []
    if set(labels) == set(FIVE_STATION_POSE_SEQUENCE):
        acquisition_sequence = FIVE_STATION_POSE_SEQUENCE
        adjacent_pairs = {
            frozenset((left, right))
            for left, right in zip(
                acquisition_sequence[:-1],
                acquisition_sequence[1:],
                strict=True,
            )
        }
    elif set(labels) == set(SIX_STATION_POSE_SEQUENCE):
        acquisition_sequence = SIX_STATION_POSE_SEQUENCE
        adjacent_pairs = {
            frozenset((left, right))
            for left, right in zip(
                acquisition_sequence[:-1],
                acquisition_sequence[1:],
                strict=True,
            )
        }
    else:
        # Generic groups do not have built-in station names.  Their ordered
        # files still provide a useful chain of neighbouring fixture views;
        # the remaining pairs are treated as loop closures.
        adjacent_pairs = {
            frozenset((labels[left], labels[right]))
            for left, right in zip(
                working_indices[:-1],
                working_indices[1:],
                strict=True,
            )
        }
    edges: list[o3d.pipelines.registration.PoseGraphEdge] = []
    metrics: list[PoseGraphEdgeMetric] = []
    identity = np.eye(4, dtype=np.float64)
    total_pairs = len(working_indices) * (len(working_indices) - 1) // 2
    pair_index = 0
    for source_position, source_index in enumerate(working_indices):
        for target_position in range(source_position + 1, len(working_indices)):
            target_index = working_indices[target_position]
            pair_index += 1
            _emit(
                progress_callback,
                f'第 {group_name} 组：{stage} 位姿图边 {pair_index}/{total_pairs} - '
                f'{labels[source_index]} → {labels[target_index]}',
            )
            is_adjacent = frozenset((labels[source_index], labels[target_index])) in adjacent_pairs
            edge, metric = _pairwise_pose_graph_edge(
                aligned_clouds[source_index],
                aligned_clouds[target_index],
                identity,
                identity,
                registration_config,
                stage,
                source_position,
                target_position,
                labels[source_index],
                labels[target_index],
                "adjacent" if is_adjacent else "overlap_loop",
                not is_adjacent,
                maximum_rotation_degrees,
                maximum_translation,
                minimum_adjacent_fitness if is_adjacent else minimum_loop_fitness,
                backend,
                maximum_adjacent_rmse if is_adjacent else maximum_loop_rmse,
                registration_cache,
                feature_cache,
            )
            # Open3D edge node ids are local to the active-node subgraph, while
            # the persisted metric should continue to use the original view ids.
            metric.source_index = source_index
            metric.target_index = target_index
            metrics.append(metric)
            if edge is not None:
                edges.append(edge)

    local_reference_index = next(
        (position for position, index in enumerate(working_indices) if labels[index] == reference_label),
        0,
    )
    local_corrections, local_node_metrics = optimize_multiframe_pose_graph(
        [identity.copy() for _ in working_indices],
        edges,
        registration_config.icp_max_correspondence_distance,
        reference_index=local_reference_index,
        maximum_rotation_degrees=maximum_rotation_degrees,
        maximum_translation=maximum_translation,
    )
    corrections = [identity.copy() for _ in aligned_clouds]
    node_metrics: list[dict[str, object]] = []
    for local_position, original_index in enumerate(working_indices):
        corrections[original_index] = local_corrections[local_position]
        node_metric = dict(local_node_metrics[local_position])
        node_metric["node_index"] = original_index
        node_metric["active_node_position"] = local_position
        node_metrics.append(node_metric)
    return corrections, metrics, node_metrics


def registration_candidate_score(
    global_fitness: float | None,
    icp_fitness: float,
    icp_rmse: float,
    full_cloud_fitness: float,
    full_cloud_rmse: float,
    config: RegistrationConfig,
) -> tuple[float, float, float, float, float, float]:
    full_cloud_quality = full_cloud_alignment_quality(
        full_cloud_fitness,
        full_cloud_rmse,
        config.icp_max_correspondence_distance,
    )[0]
    return (
        full_cloud_quality,
        full_cloud_fitness,
        -full_cloud_rmse,
        icp_fitness,
        -icp_rmse,
        global_fitness if global_fitness is not None else -1.0,
    )


def calibration_candidate_score(
    global_fitness: float | None,
    icp_fitness: float,
    icp_rmse: float,
) -> tuple[float, float, float]:
    """Rank candidates by calibration-block registration, never by workpiece geometry."""
    return (
        float(icp_fitness),
        -float(icp_rmse) if np.isfinite(icp_rmse) else -math.inf,
        float(global_fitness) if global_fitness is not None else -1.0,
    )


def vertical_plane_basis(
    angle_degrees: float,
    tilt_degrees: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造截面法向及平面内横向、竖向正交基。"""
    azimuth = math.radians(angle_degrees)
    elevation = math.radians(tilt_degrees)
    horizontal_scale = math.cos(elevation)
    normal = np.array(
        [
            horizontal_scale * math.cos(azimuth),
            horizontal_scale * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)

    reference_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    vertical = reference_up - float(np.dot(reference_up, normal)) * normal
    vertical_length = float(np.linalg.norm(vertical))
    if vertical_length <= 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        vertical = fallback_up - float(np.dot(fallback_up, normal)) * normal
        vertical_length = float(np.linalg.norm(vertical))
    vertical /= vertical_length
    if vertical[2] < 0:
        vertical = -vertical

    horizontal = np.cross(vertical, normal)
    horizontal /= np.linalg.norm(horizontal)
    return normal, horizontal, vertical


def vertical_plane_from_xy_points(
    first_point: Sequence[float],
    second_point: Sequence[float],
) -> tuple[float, float]:
    """由顶视 XY 平面的两点生成经过该线的竖直平面角度和偏移。"""
    first = np.asarray(first_point, dtype=np.float64).reshape(-1)
    second = np.asarray(second_point, dtype=np.float64).reshape(-1)
    if len(first) < 2 or len(second) < 2:
        raise ValueError("手动选面需要两个包含 X、Y 坐标的点。")
    if not np.isfinite(first[:2]).all() or not np.isfinite(second[:2]).all():
        raise ValueError("手动选面的点坐标必须为有效数值。")

    direction = second[:2] - first[:2]
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= 1e-8:
        raise ValueError("两次选择的位置过于接近，无法确定竖直切割面方向。")

    normal_xy = np.array([-direction[1], direction[0]], dtype=np.float64) / direction_length
    angle_degrees = math.degrees(math.atan2(normal_xy[1], normal_xy[0]))
    plane_offset = float(np.dot(normal_xy, first[:2]))
    return angle_degrees, plane_offset


def normal_plane_from_xyz_region(points: np.ndarray) -> tuple[float, float, float, float]:
    """由框选点云的三维主轴生成垂直于主轴的法平面。"""
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        raise ValueError("框选区域必须包含形状为 (N, 3) 的三维点坐标。")

    coordinates = coordinates[:, :3]
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    coordinates = np.unique(coordinates, axis=0)
    if len(coordinates) < 3:
        raise ValueError("框选区域至少需要 3 个不同的有效点，才能拟合截面方向。")

    point_center = coordinates.mean(axis=0)
    centered_coordinates = coordinates - point_center
    _, singular_values, right_vectors = np.linalg.svd(centered_coordinates, full_matrices=False)
    if len(singular_values) < 2 or singular_values[0] <= 1e-8:
        raise ValueError("框选区域的点分布过于集中，无法确定截面方向。")

    principal = right_vectors[0]
    principal_length = float(np.linalg.norm(principal))
    if principal_length <= 1e-8:
        raise ValueError("框选区域无法确定稳定的主方向。")
    principal = principal / principal_length
    horizontal_projection = principal[:2]
    if (
        horizontal_projection[0] < 0
        or (np.isclose(horizontal_projection[0], 0.0) and horizontal_projection[1] < 0)
        or (np.linalg.norm(horizontal_projection) <= 1e-8 and principal[2] < 0)
    ):
        principal = -principal

    lower_bounds = coordinates.min(axis=0)
    upper_bounds = coordinates.max(axis=0)
    geometric_center = (lower_bounds + upper_bounds) / 2.0

    secondary_spread = max(float(singular_values[1]), 1e-12)
    directionality = float(singular_values[0] / secondary_spread)
    angle_degrees = math.degrees(math.atan2(principal[1], principal[0]))
    tilt_degrees = math.degrees(math.atan2(principal[2], np.linalg.norm(principal[:2])))
    plane_offset = float(np.dot(principal, geometric_center))
    return angle_degrees, tilt_degrees, plane_offset, directionality


def _vertical_section_mask(
    points: np.ndarray,
    config: SectionConfig,
    backend: str = "cpu",
) -> np.ndarray:
    coordinates = np.asarray(points)
    normal, horizontal, vertical = vertical_plane_basis(config.angle_degrees, config.tilt_degrees)
    if backend == "gpu":
        return np.asarray(
            gpu_vertical_section_mask(
                coordinates,
                normal,
                horizontal,
                vertical,
                config.plane_offset,
                config.thickness,
                config.horizontal_min,
                config.horizontal_max,
                config.vertical_min,
                config.vertical_max,
            ),
            dtype=bool,
        )
    signed_distance = coordinates @ normal - config.plane_offset
    horizontal_coordinate = coordinates @ horizontal
    vertical_coordinate = coordinates @ vertical
    return (
        (np.abs(signed_distance) <= config.thickness / 2.0)
        & (horizontal_coordinate >= config.horizontal_min)
        & (horizontal_coordinate <= config.horizontal_max)
        & (vertical_coordinate >= config.vertical_min)
        & (vertical_coordinate <= config.vertical_max)
    )


def select_vertical_section_indices(points: np.ndarray, config: SectionConfig) -> np.ndarray:
    """返回位于三维法向切片厚度和截面坐标范围内的点索引。"""
    return np.flatnonzero(_vertical_section_mask(points, config))


def _section_connected_component_mask(points: np.ndarray, config: SectionConfig) -> np.ndarray:
    coordinates = np.asarray(points, dtype=np.float64)
    keep_mask = np.zeros(len(coordinates), dtype=bool)
    if not len(coordinates):
        return keep_mask

    _, horizontal, vertical = vertical_plane_basis(config.angle_degrees, config.tilt_degrees)
    projected = np.column_stack((coordinates @ horizontal, coordinates @ vertical))
    finite_mask = np.isfinite(projected).all(axis=1)
    if not finite_mask.any():
        return keep_mask

    cell_coordinates = np.floor(projected / config.component_cell_size).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for point_index in np.flatnonzero(finite_mask):
        cell = (int(cell_coordinates[point_index, 0]), int(cell_coordinates[point_index, 1]))
        buckets.setdefault(cell, []).append(int(point_index))

    unvisited = set(buckets)
    minimum_component_span = max(
        config.component_cell_size * 2.0,
        config.thickness * 2.0,
    )
    while unvisited:
        root = unvisited.pop()
        pending = [root]
        component_indices: list[int] = []
        while pending:
            cell = pending.pop()
            component_indices.extend(buckets[cell])
            cell_x, cell_z = cell
            for delta_x in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbor = (cell_x + delta_x, cell_z + delta_z)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        pending.append(neighbor)
        component_projected = projected[np.asarray(component_indices, dtype=np.int64)]
        component_span = np.ptp(component_projected, axis=0)
        has_meaningful_span = float(np.max(component_span)) >= minimum_component_span
        if len(component_indices) >= config.component_min_points and has_meaningful_span:
            keep_mask[np.asarray(component_indices, dtype=np.int64)] = True
    return keep_mask


def clean_vertical_section(
    cloud: o3d.geometry.PointCloud,
    config: SectionConfig,
) -> o3d.geometry.PointCloud:
    """对截面薄片执行统计滤波和二维连通域清理，去除孤立飞点与紧凑噪声岛。"""
    original = cloud
    cleaned = cloud
    if config.use_statistical_filter and len(cleaned.points) > config.statistical_neighbors:
        neighbor_count = min(config.statistical_neighbors, len(cleaned.points) - 1)
        statistically_cleaned, _ = cleaned.remove_statistical_outlier(
            nb_neighbors=neighbor_count,
            std_ratio=config.statistical_std_ratio,
        )
        if len(statistically_cleaned.points):
            cleaned = statistically_cleaned

    if config.use_component_filter and len(cleaned.points) >= config.component_min_points:
        component_mask = _section_connected_component_mask(np.asarray(cleaned.points), config)
        if component_mask.any():
            component_cleaned = cleaned.select_by_index(np.flatnonzero(component_mask).tolist())
            if len(component_cleaned.points):
                cleaned = component_cleaned

    # Never accept a cleanup result that materially shortens a measured
    # contour.  The patent's neighbourhood merge is useful for duplicate
    # samples, but a section outlier filter must not turn an occluded edge into
    # a shorter edge.  The raw section remains the source of truth whenever the
    # cleanup would remove a structural component or too much occupied span.
    if len(original.points) >= SECTION_INTEGRITY_MINIMUM_POINTS and len(cleaned.points):
        raw_projected = _section_projected_coordinates(np.asarray(original.points), config)
        cleaned_projected = _section_projected_coordinates(np.asarray(cleaned.points), config)
        if len(raw_projected) and len(cleaned_projected):
            raw_grid = _section_integrity_grid_summary(
                raw_projected,
                min(SECTION_INTEGRITY_MAXIMUM_CELL_SIZE, max(SECTION_INTEGRITY_MINIMUM_CELL_SIZE, float(config.component_cell_size))),
                max(SECTION_INTEGRITY_MINIMUM_COMPONENT_SPAN, float(config.component_cell_size) * 2.0),
            )
            cleaned_grid = _section_integrity_grid_summary(
                cleaned_projected,
                min(SECTION_INTEGRITY_MAXIMUM_CELL_SIZE, max(SECTION_INTEGRITY_MINIMUM_CELL_SIZE, float(config.component_cell_size))),
                max(SECTION_INTEGRITY_MINIMUM_COMPONENT_SPAN, float(config.component_cell_size) * 2.0),
            )
            raw_span = float(raw_grid["largest_component_span"])
            cleaned_span = float(cleaned_grid["largest_component_span"])
            lost_ratio = len(set(raw_grid["cells"]) - set(cleaned_grid["cells"])) / max(len(raw_grid["cells"]), 1)
            if (
                int(cleaned_grid["component_count"]) > int(raw_grid["component_count"])
                or (raw_span > 0 and cleaned_span / raw_span < SECTION_INTEGRITY_MINIMUM_COMPONENT_RETENTION)
                or lost_ratio > SECTION_INTEGRITY_MAXIMUM_LOST_CELL_RATIO
            ):
                return copy.deepcopy(original)
    return cleaned


def _section_projected_coordinates(
    points: np.ndarray,
    config: SectionConfig,
) -> np.ndarray:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        return np.empty((0, 2), dtype=np.float64)
    coordinates = coordinates[:, :3]
    finite_mask = np.isfinite(coordinates).all(axis=1)
    if not finite_mask.any():
        return np.empty((0, 2), dtype=np.float64)
    _, horizontal, vertical = vertical_plane_basis(
        config.angle_degrees,
        config.tilt_degrees,
    )
    finite_coordinates = coordinates[finite_mask]
    projected = np.column_stack(
        (
            finite_coordinates @ horizontal,
            finite_coordinates @ vertical,
        )
    )
    return projected[np.isfinite(projected).all(axis=1)]


def _section_integrity_grid_summary(
    projected: np.ndarray,
    cell_size: float,
    minimum_component_span: float,
) -> dict[str, object]:
    coordinates = np.asarray(projected, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates):
        return {
            "cells": set(),
            "meaningful_components": [],
            "component_count": 0,
            "cell_count": 0,
            "largest_component_span": 0.0,
        }

    cell_coordinates = np.floor(coordinates / cell_size).astype(np.int64)
    cells = {
        (int(cell[0]), int(cell[1]))
        for cell in cell_coordinates
    }
    unvisited = set(cells)
    components: list[set[tuple[int, int]]] = []
    while unvisited:
        root = unvisited.pop()
        pending = [root]
        component = {root}
        while pending:
            cell_x, cell_y = pending.pop()
            for delta_x in (-1, 0, 1):
                for delta_y in (-1, 0, 1):
                    neighbor = (cell_x + delta_x, cell_y + delta_y)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        pending.append(neighbor)
                        component.add(neighbor)
        components.append(component)

    minimum_component_cells = max(
        3,
        int(math.ceil(minimum_component_span / cell_size)),
    )
    meaningful_components: list[dict[str, object]] = []
    for component in components:
        component_coordinates = np.asarray(sorted(component), dtype=np.float64)
        span = (np.ptp(component_coordinates, axis=0) + 1.0) * cell_size
        maximum_span = float(np.max(span))
        if len(component) < minimum_component_cells or maximum_span < minimum_component_span:
            continue
        meaningful_components.append(
            {
                "cells": component,
                "cell_count": len(component),
                "span_horizontal": float(span[0]),
                "span_vertical": float(span[1]),
                "maximum_span": maximum_span,
            }
        )

    meaningful_components.sort(
        key=lambda component: (
            float(component["maximum_span"]),
            int(component["cell_count"]),
        ),
        reverse=True,
    )
    return {
        "cells": cells,
        "meaningful_components": meaningful_components,
        "component_count": len(meaningful_components),
        "cell_count": len(cells),
        "largest_component_span": (
            float(meaningful_components[0]["maximum_span"])
            if meaningful_components
            else 0.0
        ),
    }


def scan_section_offsets(
    points: np.ndarray,
    config: SectionConfig,
    *,
    center_offset: float | None = None,
    half_range: float = 4.0,
    step: float = 0.25,
    grid_cell_size: float | None = None,
) -> list[dict[str, float | int | bool]]:
    """Measure nearby section planes without changing the point cloud or config.

    The scan is intentionally diagnostic only.  It reports how many preview points
    each parallel plane captures and how those points break into projected
    connected structures; callers still decide which offset, if any, to use for a
    later processing run.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        raise ValueError("截面偏移扫描需要形状为 (N, 3) 的点坐标。")
    if half_range < 0:
        raise ValueError("截面偏移扫描范围不能为负数。")
    if step <= 0:
        raise ValueError("截面偏移扫描步长必须大于 0。")

    center = float(config.plane_offset if center_offset is None else center_offset)
    start = center - float(half_range)
    stop = center + float(half_range)
    offsets = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    if len(offsets) == 0:
        offsets = np.asarray([center], dtype=np.float64)
    elif not np.any(np.isclose(offsets, center, atol=step * 1e-6)):
        offsets = np.sort(np.append(offsets, center))
    if len(offsets) > 161:
        raise ValueError("截面偏移扫描最多支持 161 个采样平面，请增大扫描步长。")

    finite_mask = np.isfinite(coordinates[:, :3]).all(axis=1)
    if finite_mask.any():
        finite_coordinates = coordinates[finite_mask, :3]
        normal, horizontal, vertical = vertical_plane_basis(
            config.angle_degrees,
            config.tilt_degrees,
        )
        signed_distances = finite_coordinates @ normal
        horizontal_coordinates = finite_coordinates @ horizontal
        vertical_coordinates = finite_coordinates @ vertical
        in_range = (
            (horizontal_coordinates >= config.horizontal_min)
            & (horizontal_coordinates <= config.horizontal_max)
            & (vertical_coordinates >= config.vertical_min)
            & (vertical_coordinates <= config.vertical_max)
        )
    else:
        signed_distances = np.empty(0, dtype=np.float64)
        horizontal_coordinates = np.empty(0, dtype=np.float64)
        vertical_coordinates = np.empty(0, dtype=np.float64)
        in_range = np.empty(0, dtype=bool)

    cell_size = float(
        config.component_cell_size if grid_cell_size is None else grid_cell_size
    )
    if cell_size <= 0:
        raise ValueError("截面偏移扫描网格大小必须大于 0。")
    minimum_component_span = max(
        SECTION_INTEGRITY_MINIMUM_COMPONENT_SPAN,
        cell_size * 2.0,
    )

    rows: list[dict[str, float | int | bool]] = []
    half_thickness = float(config.thickness) / 2.0
    target_horizontal_span = max(
        float(config.horizontal_max - config.horizontal_min),
        1e-9,
    )
    for raw_offset in offsets:
        offset = float(raw_offset)
        section_mask = in_range & (
            np.abs(signed_distances - offset) <= half_thickness
        )
        projected = np.column_stack(
            (
                horizontal_coordinates[section_mask],
                vertical_coordinates[section_mask],
            )
        )
        point_count = int(len(projected))
        if point_count:
            occupied_span = np.ptp(projected, axis=0)
            grid_summary = _section_integrity_grid_summary(
                projected,
                cell_size,
                minimum_component_span,
            )
            layer_summary = _audit_parallel_section_layers(
                projected,
                np.zeros(point_count, dtype=np.int64),
            )
            largest_component_span = float(grid_summary["largest_component_span"])
            component_count = int(grid_summary["component_count"])
            cell_count = int(grid_summary["cell_count"])
        else:
            occupied_span = np.zeros(2, dtype=np.float64)
            layer_summary = {
                "parallel_double_layer_ratio": 0.0,
                "parallel_layer_track_count": 0,
                "parallel_layer_max_tracks_per_bin": 0,
            }
            largest_component_span = 0.0
            component_count = 0
            cell_count = 0
        points_per_cell = float(point_count / max(cell_count, 1))
        horizontal_coverage_ratio = min(
            float(occupied_span[0]) / target_horizontal_span,
            1.0,
        )
        continuity_ratio = (
            min(largest_component_span / max(float(occupied_span[0]), 1e-9), 1.0)
            if point_count
            else 0.0
        )
        component_penalty = 1.0 / max(1.0 + 0.12 * max(component_count - 1, 0), 1.0)
        quality_score = float(
            np.clip(
                0.60 * horizontal_coverage_ratio
                + 0.30 * continuity_ratio
                + 0.10 * min(points_per_cell / 20.0, 1.0),
                0.0,
                1.0,
            )
            * component_penalty
        )
        rows.append(
            {
                "plane_offset": offset,
                "point_count": point_count,
                "component_count": component_count,
                "cell_count": cell_count,
                "points_per_cell": points_per_cell,
                "horizontal_span": float(occupied_span[0]),
                "vertical_span": float(occupied_span[1]),
                "largest_component_span": largest_component_span,
                "largest_component_ratio": float(
                    largest_component_span / max(float(occupied_span[0]), 1e-9)
                ),
                "horizontal_coverage_ratio": horizontal_coverage_ratio,
                "continuity_ratio": continuity_ratio,
                "quality_score": quality_score,
                "parallel_layer_ratio": float(
                    layer_summary["parallel_double_layer_ratio"]
                ),
                "parallel_layer_bin_count": int(
                    layer_summary["parallel_double_layer_bin_count"]
                    if "parallel_double_layer_bin_count" in layer_summary
                    else 0
                ),
                "parallel_layer_max_tracks": int(
                    layer_summary["parallel_layer_max_tracks_per_bin"]
                ),
                "is_current": bool(np.isclose(offset, center, atol=step * 0.25)),
            }
        )
    if rows:
        best_index = max(
            range(len(rows)),
            key=lambda index: (
                float(rows[index]["quality_score"]),
                -abs(float(rows[index]["plane_offset"]) - center),
            ),
        )
        for index, row in enumerate(rows):
            row["recommended"] = bool(index == best_index)
    return rows


def _sample_section_offset_points(
    points: np.ndarray,
    maximum_points: int = 180_000,
) -> np.ndarray:
    """Return a deterministic finite sample for offset diagnostics.

    Offset scanning is a diagnostic pass and must not turn a normal run into a
    second full-resolution fusion.  The selected section itself still uses the
    original fused cloud; only the scoring pass is sampled.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    coordinates = coordinates[:, :3]
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if len(coordinates) <= maximum_points:
        return coordinates
    # PLY point order is often grouped by station/surface.  Taking every
    # k-th record can therefore miss an entire thin plane.  A fixed-seed
    # sample is order-independent and keeps the diagnostic repeatable.
    generator = np.random.default_rng(20260903)
    sample_indices = generator.choice(
        len(coordinates),
        size=int(maximum_points),
        replace=False,
    )
    return coordinates[np.sort(sample_indices)]


def _prepare_section_offset_scan_points(
    points: np.ndarray,
    config: SectionConfig,
    center_offset: float,
    half_range: float,
    step: float,
    maximum_points: int,
    *,
    seed: int = 20260903,
) -> np.ndarray:
    """Keep a deterministic, normal-stratified sample for offset diagnostics.

    A uniform random sample of a broad 3-D band can miss a thin candidate plane
    completely.  Sampling each normal-distance bin separately keeps evidence for
    every scanned offset while leaving the original point coordinates untouched.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3 or maximum_points < 1:
        return np.empty((0, 3), dtype=np.float64)
    coordinates = coordinates[:, :3]
    finite_mask = np.isfinite(coordinates).all(axis=1)
    if not finite_mask.any():
        return np.empty((0, 3), dtype=np.float64)
    coordinates = coordinates[finite_mask]

    normal, horizontal, vertical = vertical_plane_basis(
        config.angle_degrees,
        config.tilt_degrees,
    )
    signed_distances = coordinates @ normal
    horizontal_coordinates = coordinates @ horizontal
    vertical_coordinates = coordinates @ vertical
    band_mask = (
        (np.abs(signed_distances - float(center_offset))
         <= float(half_range) + float(config.thickness) / 2.0)
        & (horizontal_coordinates >= config.horizontal_min)
        & (horizontal_coordinates <= config.horizontal_max)
        & (vertical_coordinates >= config.vertical_min)
        & (vertical_coordinates <= config.vertical_max)
    )
    band_indices = np.flatnonzero(band_mask)
    if not len(band_indices):
        return np.empty((0, 3), dtype=np.float64)
    if len(band_indices) <= maximum_points:
        return coordinates[band_indices]

    # Use the requested scan step as the stratification width.  The lower bound
    # prevents an extremely small UI step from creating thousands of tiny bins.
    bin_width = max(min(float(step), 0.50), 0.05)
    normal_bins = np.floor(
        (signed_distances[band_indices]
         - (float(center_offset) - float(half_range)))
        / bin_width
    ).astype(np.int64)
    unique_bins = np.unique(normal_bins)
    quota = max(1, int(math.ceil(maximum_points / max(len(unique_bins), 1))))
    generator = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for bin_index in unique_bins:
        members = band_indices[normal_bins == bin_index]
        if len(members) <= quota:
            selected.append(members)
            continue
        selected.append(
            np.sort(
                generator.choice(members, size=quota, replace=False)
            )
        )
    selected_indices = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if len(selected_indices) > maximum_points:
        selected_indices = np.sort(
            generator.choice(selected_indices, size=int(maximum_points), replace=False)
        )
    return coordinates[selected_indices]


def _source_section_offset_summary(
    source_clouds: Sequence[o3d.geometry.PointCloud],
    source_indices: Sequence[int],
    config: SectionConfig,
    offset: float,
    *,
    center_offset: float,
    half_range: float,
    step: float,
    maximum_points_per_source: int,
    cell_size: float,
) -> dict[str, object]:
    """Summarize per-view support for one candidate parallel section.

    The returned private set fields are consumed by the candidate evaluator and
    removed before the report is serialized.  They make it possible to reject a
    candidate whose apparent continuity comes from only one camera view.
    """
    if len(source_clouds) != len(source_indices):
        raise ValueError("截面偏移候选的源点云和来源索引数量必须一致。")
    normal, horizontal, vertical = vertical_plane_basis(
        config.angle_degrees,
        config.tilt_degrees,
    )
    half_thickness = float(config.thickness) / 2.0
    minimum_component_span = max(
        SECTION_INTEGRITY_MINIMUM_COMPONENT_SPAN,
        float(cell_size) * 2.0,
    )
    source_point_counts: list[int] = []
    projected_sets: list[np.ndarray] = []
    cell_support: dict[tuple[int, int], set[int]] = {}
    for source_position, (cloud, source_index) in enumerate(
        zip(source_clouds, source_indices, strict=True)
    ):
        sampled_points = _prepare_section_offset_scan_points(
            np.asarray(cloud.points),
            config,
            center_offset,
            half_range,
            step,
            maximum_points_per_source,
            seed=20260903 + int(source_index) * 101 + source_position,
        )
        if not len(sampled_points):
            source_point_counts.append(0)
            continue
        signed_distances = sampled_points @ normal
        horizontal_coordinates = sampled_points @ horizontal
        vertical_coordinates = sampled_points @ vertical
        section_mask = np.abs(signed_distances - float(offset)) <= half_thickness
        projected = np.column_stack(
            (horizontal_coordinates[section_mask], vertical_coordinates[section_mask])
        )
        projected = projected[np.isfinite(projected).all(axis=1)]
        source_point_counts.append(int(len(projected)))
        if not len(projected):
            continue
        projected_sets.append(projected)
        cells = np.floor(projected / float(cell_size)).astype(np.int64)
        for cell in {tuple(value) for value in cells}:
            cell_support.setdefault(cell, set()).add(int(source_index))

    union_projected = (
        np.concatenate(projected_sets, axis=0)
        if projected_sets
        else np.empty((0, 2), dtype=np.float64)
    )
    union_grid = _section_integrity_grid_summary(
        union_projected,
        float(cell_size),
        minimum_component_span,
    )
    union_cells = set(union_grid["cells"])
    multi_view_cells = {
        cell for cell, supports in cell_support.items() if len(supports) >= 2
    }
    horizontal_bins = {int(cell[0]) for cell in union_cells}
    multi_view_horizontal_bins = {
        int(cell[0]) for cell in multi_view_cells
    }
    source_view_count = sum(
        count >= SECTION_INTEGRITY_MINIMUM_POINTS for count in source_point_counts
    )
    return {
        "source_view_count": int(source_view_count),
        "source_view_point_counts": source_point_counts,
        "source_union_cell_count": int(len(union_cells)),
        "source_multi_view_cell_count": int(len(multi_view_cells)),
        "source_multi_view_cell_ratio": float(
            len(multi_view_cells) / max(len(union_cells), 1)
        ),
        "source_horizontal_bin_count": int(len(horizontal_bins)),
        "source_multi_view_horizontal_bin_count": int(
            len(multi_view_horizontal_bins)
        ),
        "source_multi_view_horizontal_bin_ratio": float(
            len(multi_view_horizontal_bins) / max(len(horizontal_bins), 1)
        ),
        "source_union_component_count": int(union_grid["component_count"]),
        "source_union_largest_component_span": float(
            union_grid["largest_component_span"]
        ),
        "_source_cells": union_cells,
        "_source_multi_view_cells": multi_view_cells,
    }


def evaluate_section_offset_candidates(
    points: np.ndarray,
    config: SectionConfig,
    *,
    center_offset: float | None = None,
    half_range: float | None = None,
    step: float | None = None,
    maximum_points: int = 180_000,
    source_clouds: Sequence[o3d.geometry.PointCloud] | None = None,
    source_indices: Sequence[int] | None = None,
    maximum_source_points_per_view: int = 60_000,
) -> dict[str, object]:
    """Evaluate nearby parallel planes and conservatively recommend one.

    The evaluator is intentionally independent of CAD.  It looks for a real
    improvement in the final fused cloud's connected contour while penalizing
    candidates that mainly increase parallel layers.  A recommendation is only
    marked safe when the candidate keeps most occupied section cells, gains a
    meaningful span/quality margin, and stays below the double-layer guardrail.
    """
    if not config.auto_offset_enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "status_label": "截面偏移自动诊断已关闭",
            "message": "当前配置关闭了附近平行截面诊断。",
            "recommendation": None,
            "recommended_offset": None,
            "current_offset": float(
                config.plane_offset if center_offset is None else center_offset
            ),
            "sampled_points": 0,
            "rows": [],
            "auto_applied": False,
        }

    normalized_half_range = (
        float(config.offset_scan_half_range)
        if half_range is None
        else float(half_range)
    )
    normalized_step = (
        float(config.offset_scan_step) if step is None else float(step)
    )
    center = float(config.plane_offset if center_offset is None else center_offset)
    sampled_points = _prepare_section_offset_scan_points(
        np.asarray(points, dtype=np.float64),
        config,
        center,
        normalized_half_range,
        normalized_step,
        maximum_points,
    )
    if not len(sampled_points):
        return {
            "enabled": True,
            "status": "insufficient_evidence",
            "status_label": "截面偏移证据不足",
            "message": "最终融合云没有可用于平行截面扫描的有限点。",
            "recommendation": None,
            "recommended_offset": None,
            "current_offset": float(
                config.plane_offset if center_offset is None else center_offset
            ),
            "sampled_points": 0,
            "rows": [],
        }

    rows = scan_section_offsets(
        sampled_points,
        config,
        center_offset=center_offset,
        half_range=normalized_half_range,
        step=normalized_step,
    )
    if not rows:
        return {
            "enabled": True,
            "status": "insufficient_evidence",
            "status_label": "截面偏移证据不足",
            "message": "附近没有生成有效的平行截面候选。",
            "recommendation": None,
            "recommended_offset": None,
            "current_offset": float(
                config.plane_offset if center_offset is None else center_offset
            ),
            "sampled_points": int(len(sampled_points)),
            "rows": [],
        }

    current = float(config.plane_offset if center_offset is None else center_offset)
    current_index = min(
        range(len(rows)),
        key=lambda index: abs(float(rows[index]["plane_offset"]) - current),
    )
    current_row = rows[current_index]
    current_cells = max(int(current_row.get("cell_count", 0)), 1)
    current_span = float(current_row.get("largest_component_span", 0.0))
    current_quality = float(current_row.get("quality_score", 0.0))
    current_components = int(current_row.get("component_count", 0))

    normalized_source_indices: list[int] = []
    source_summaries: list[dict[str, object]] = []
    source_support_available = bool(source_clouds)
    if source_support_available:
        normalized_source_indices = (
            [int(index) for index in source_indices]
            if source_indices is not None
            else list(range(len(source_clouds or [])))
        )
        if len(normalized_source_indices) != len(source_clouds or []):
            raise ValueError("截面偏移候选的源点云和来源索引数量必须一致。")
        source_summaries = [
            _source_section_offset_summary(
                source_clouds or [],
                normalized_source_indices,
                config,
                float(row["plane_offset"]),
                center_offset=center,
                half_range=normalized_half_range,
                step=normalized_step,
                maximum_points_per_source=max(
                    2_000,
                    int(maximum_source_points_per_view),
                ),
                cell_size=float(config.component_cell_size),
            )
            for row in rows
        ]
    current_source_summary = (
        source_summaries[current_index] if source_summaries else None
    )
    current_source_cells = (
        set(current_source_summary.get("_source_cells", set()))
        if current_source_summary is not None
        else set()
    )
    def _stable_neighbor_count(index: int) -> int:
        """Count adjacent scan planes that show the same structure."""
        row = rows[index]
        quality = float(row.get("quality_score", 0.0))
        span = float(row.get("largest_component_span", 0.0))
        layer_ratio = float(row.get("parallel_layer_ratio", 0.0))
        stable = 0
        for neighbor_index in (index - 2, index - 1, index + 1, index + 2):
            if not 0 <= neighbor_index < len(rows):
                continue
            neighbor = rows[neighbor_index]
            neighbor_span = float(neighbor.get("largest_component_span", 0.0))
            if (
                abs(float(neighbor.get("quality_score", 0.0)) - quality) <= 0.08
                and abs(neighbor_span - span) <= max(0.50, 0.12 * max(span, 1.0))
                and abs(float(neighbor.get("parallel_layer_ratio", 0.0)) - layer_ratio)
                <= 0.12
            ):
                stable += 1
        return stable

    for index, row in enumerate(rows):
        span = float(row.get("largest_component_span", 0.0))
        quality = float(row.get("quality_score", 0.0))
        layer_ratio = float(row.get("parallel_layer_ratio", 0.0))
        layer_tracks = int(row.get("parallel_layer_max_tracks", 0))
        cell_retention = float(int(row.get("cell_count", 0)) / current_cells)
        span_gain = float((span - current_span) / max(abs(current_span), 1e-9))
        quality_gain = float(quality - current_quality)
        structural_improvement = bool(
            int(row.get("component_count", 0)) < current_components
            or span_gain >= 0.15
        )
        evidence_ok = bool(
            int(row.get("point_count", 0)) >= SECTION_INTEGRITY_MINIMUM_POINTS
            and cell_retention >= 0.60
        )
        layer_ok = bool(
            layer_ratio <= SECTION_OFFSET_MAX_LAYER_RATIO
            and layer_tracks <= 6
            and layer_ratio <= float(current_row.get("parallel_layer_ratio", 0.0))
            + SECTION_OFFSET_MAX_LAYER_GROWTH
        )
        stable_neighbor_count = _stable_neighbor_count(index)
        stability_ok = bool(
            index == current_index
            or stable_neighbor_count >= min(
                SECTION_OFFSET_MINIMUM_STABLE_NEIGHBORS,
                max(len(rows) - 1, 0),
            )
        )
        source_summary = source_summaries[index] if source_summaries else None
        source_support_ok = True
        source_new_cell_count = 0
        source_new_multiview_cell_count = 0
        source_new_multiview_ratio = 0.0
        source_cell_retention = 0.0
        if source_summary is not None:
            source_cells = set(source_summary.get("_source_cells", set()))
            source_multi_view_cells = set(
                source_summary.get("_source_multi_view_cells", set())
            )
            source_new_cells = source_cells - current_source_cells
            source_new_multi_view_cells = source_new_cells & source_multi_view_cells
            # A cell that is already multi-view at the current plane is not new
            # evidence, even if a different surface occupies the same grid cell.
            source_new_multiview_cell_count = int(len(source_new_multi_view_cells))
            source_new_cell_count = int(len(source_new_cells))
            source_new_multiview_ratio = float(
                source_new_multiview_cell_count / max(source_new_cell_count, 1)
            )
            source_cell_retention = float(
                len(source_cells) / max(len(current_source_cells), 1)
            )
            source_view_count = int(source_summary.get("source_view_count", 0))
            source_support_ok = source_view_count >= min(
                SECTION_OFFSET_MINIMUM_SOURCE_VIEWS,
                len(source_clouds or []),
            )
            if source_new_cell_count >= SECTION_OFFSET_MINIMUM_NEW_MULTIVIEW_CELLS:
                source_support_ok = bool(
                    source_support_ok
                    and source_new_multiview_cell_count
                    >= SECTION_OFFSET_MINIMUM_NEW_MULTIVIEW_CELLS
                    and source_new_multiview_ratio
                    >= SECTION_OFFSET_MINIMUM_NEW_MULTIVIEW_RATIO
                )
            else:
                source_support_ok = bool(
                    source_support_ok
                    and float(source_summary.get("source_multi_view_cell_ratio", 0.0))
                    >= 0.40
                )
            row.update(
                {
                    key: value
                    for key, value in source_summary.items()
                    if not key.startswith("_")
                }
            )
        else:
            row.update(
                {
                    "source_view_count": None,
                    "source_view_point_counts": [],
                    "source_union_cell_count": None,
                    "source_multi_view_cell_count": None,
                    "source_multi_view_cell_ratio": None,
                    "source_horizontal_bin_count": None,
                    "source_multi_view_horizontal_bin_count": None,
                    "source_multi_view_horizontal_bin_ratio": None,
                    "source_union_component_count": None,
                    "source_union_largest_component_span": None,
                }
            )
        candidate_score = float(
            quality
            - 0.22 * min(max(layer_ratio, 0.0), 1.0)
            - 0.015 * max(layer_tracks - 2, 0)
            + 0.04 * min(max(cell_retention, 0.0), 1.0)
            + 0.04 * min(max(source_new_multiview_ratio, 0.0), 1.0)
        )
        row.update(
            {
                "cell_retention": cell_retention,
                "span_gain": span_gain,
                "quality_gain": quality_gain,
                "structural_improvement": structural_improvement,
                "evidence_ok": evidence_ok,
                "layer_guardrail_ok": layer_ok,
                "stable_neighbor_count": stable_neighbor_count,
                "stability_guardrail_ok": stability_ok,
                "source_support_available": source_summary is not None,
                "source_cell_retention": source_cell_retention,
                "source_new_cell_count": source_new_cell_count,
                "source_new_multiview_cell_count": source_new_multiview_cell_count,
                "source_new_multiview_ratio": source_new_multiview_ratio,
                "source_support_guardrail_ok": source_support_ok,
                "candidate_score": candidate_score,
                "auto_eligible": bool(
                    evidence_ok
                    and layer_ok
                    and structural_improvement
                    and stability_ok
                    and source_support_ok
                    and quality_gain >= float(config.offset_min_quality_gain)
                ),
            }
        )

    eligible_indices = [
        index
        for index, row in enumerate(rows)
        if index != current_index and bool(row.get("auto_eligible", False))
    ]
    best_index: int | None = None
    plateau_indices: list[int] = []
    if eligible_indices:
        best_score = max(float(rows[index]["candidate_score"]) for index in eligible_indices)
        plateau_indices = [
            index
            for index in eligible_indices
            if float(rows[index]["candidate_score"]) >= best_score - 0.005
        ]
        plateau_indices.sort(
            key=lambda index: float(rows[index]["plane_offset"])
        )
        # Use the center of a stable plateau rather than its first edge.  This
        # avoids chasing a single sampled plane when several adjacent planes
        # show the same physical contour.
        plateau_offset = float(
            np.median(
                [float(rows[index]["plane_offset"]) for index in plateau_indices]
            )
        )
        best_index = min(
            plateau_indices,
            key=lambda index: abs(float(rows[index]["plane_offset"]) - plateau_offset),
        )

    recommended_offset: float | None = None
    status = "current_is_best"
    status_label = "当前截面位置更稳妥"
    message = "当前平面没有发现证据充分且明显更完整的平行候选。"
    recommendation = None
    if best_index is not None:
        best_row = rows[best_index]
        recommended_offset = float(best_row["plane_offset"])
        status = "auto_recommended"
        status_label = "发现可复核的完整截面候选"
        message = (
            f"候选偏移 {recommended_offset:+.3f} 的连续跨度和覆盖评分高于当前平面，"
            "且未超过双层轮廓防误选阈值。"
        )
        recommendation = {
            "plane_offset": recommended_offset,
            "quality_gain": float(best_row["quality_gain"]),
            "span_gain": float(best_row["span_gain"]),
            "cell_retention": float(best_row["cell_retention"]),
            "parallel_layer_ratio": float(best_row["parallel_layer_ratio"]),
            "source_view_count": best_row.get("source_view_count"),
            "source_new_cell_count": int(best_row.get("source_new_cell_count", 0)),
            "source_new_multiview_cell_count": int(
                best_row.get("source_new_multiview_cell_count", 0)
            ),
            "source_new_multiview_ratio": float(
                best_row.get("source_new_multiview_ratio", 0.0)
            ),
            "plateau_offsets": [
                float(rows[index]["plane_offset"]) for index in plateau_indices
            ],
        }
    elif any(
        bool(row.get("structural_improvement", False))
        for index, row in enumerate(rows)
        if index != current_index
    ):
        source_rejected = [
            row
            for index, row in enumerate(rows)
            if index != current_index
            and bool(row.get("structural_improvement", False))
            and not bool(row.get("source_support_guardrail_ok", True))
        ]
        if source_support_available and source_rejected:
            status = "source_coverage_gap"
            status_label = "候选轮廓缺少多工位支持"
            message = (
                "附近存在跨度更大的候选截面，但新增网格主要只由单个工位支持，"
                "不能确认是同一实体轮廓；当前缺口不能由现有多工位数据可靠补齐。"
            )
            recommendation = {
                "action": "rescan",
                "reason": "candidate_new_coverage_not_multiview_supported",
                "minimum_source_views": min(
                    SECTION_OFFSET_MINIMUM_SOURCE_VIEWS,
                    len(source_clouds or []),
                ),
            }
        else:
            status = "manual_review"
            status_label = "候选截面需要人工复核"
            message = (
                "附近存在跨度更大的截面，但同时存在点数/覆盖不足、邻近平面不稳定或"
                "平行双层风险，程序不会自动替换当前平面。"
            )

    source_current_summary_public = None
    if current_source_summary is not None:
        source_current_summary_public = {
            key: value
            for key, value in current_source_summary.items()
            if not key.startswith("_")
        }

    for index, row in enumerate(rows):
        row["is_current"] = bool(index == current_index)
        row["recommended"] = bool(index == best_index)

    return {
        "enabled": True,
        "method": "final_fused_cloud_parallel_section_guarded_selection",
        "status": status,
        "status_label": status_label,
        "message": message,
        "recommendation": recommendation,
        "recommended_offset": recommended_offset,
        "current_offset": current,
        "current_row": dict(current_row),
        "source_support_available": source_support_available,
        "source_current": source_current_summary_public,
        "source_view_count": (
            int(current_source_summary.get("source_view_count", 0))
            if current_source_summary is not None
            else None
        ),
        "sampled_points": int(len(sampled_points)),
        "half_range": normalized_half_range,
        "step": normalized_step,
        "rows": rows,
        "auto_applied": False,
    }


def diagnose_section_integrity(
    raw_section_cloud: o3d.geometry.PointCloud,
    cleaned_section_cloud: o3d.geometry.PointCloud,
    config: SectionConfig,
    source_profiles: Mapping[int, np.ndarray] | None = None,
    source_profile_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Compare section structure without changing point coordinates."""
    raw_points = np.asarray(raw_section_cloud.points, dtype=np.float64)
    cleaned_points = np.asarray(cleaned_section_cloud.points, dtype=np.float64)
    raw_projected = _section_projected_coordinates(raw_points, config)
    cleaned_projected = _section_projected_coordinates(cleaned_points, config)
    raw_point_count = int(len(raw_projected))
    cleaned_point_count = int(len(cleaned_projected))
    removed_point_count = max(0, raw_point_count - cleaned_point_count)
    removed_ratio = (
        float(removed_point_count / raw_point_count)
        if raw_point_count
        else 0.0
    )

    base_result: dict[str, object] = {
        "enabled": True,
        "method": "raw_cleaned_projected_structure_comparison",
        "coordinates_changed": False,
        "interpolation_applied": False,
        "cad_feedback_applied": False,
        "raw_points": raw_point_count,
        "cleaned_points": cleaned_point_count,
        "removed_points": removed_point_count,
        "removed_ratio": removed_ratio,
        "grid_cell_size": 0.0,
        "minimum_component_span": 0.0,
        "raw_component_count": 0,
        "cleaned_component_count": 0,
        "raw_cell_count": 0,
        "cleaned_cell_count": 0,
        "shared_cell_count": 0,
        "lost_cell_count": 0,
        "lost_cell_ratio": 0.0,
        "component_match_ratio": 0.0,
        "largest_component_span_retention": 0.0,
        "source_profile_count": 0,
        "source_profile_component_counts": [],
        "source_profile_consistency_ratio": 0.0,
        "source_profile_gap_evidence": False,
        "structural_change_detected": False,
        "status": "insufficient_evidence",
        "status_label": "证据不足",
        "message": "当前截面点数或连续结构不足，暂不判断缺口来源。",
        "recommendation": "先调整截面区域或平面偏移，使基线截面包含完整目标轮廓。",
    }
    if raw_point_count < SECTION_INTEGRITY_MINIMUM_POINTS:
        return base_result

    cell_size = min(
        SECTION_INTEGRITY_MAXIMUM_CELL_SIZE,
        max(
            SECTION_INTEGRITY_MINIMUM_CELL_SIZE,
            float(config.component_cell_size),
            float(config.thickness) * 2.0,
        ),
    )
    minimum_component_span = max(
        SECTION_INTEGRITY_MINIMUM_COMPONENT_SPAN,
        cell_size * 2.0,
    )
    raw_grid = _section_integrity_grid_summary(
        raw_projected,
        cell_size,
        minimum_component_span,
    )
    cleaned_grid = _section_integrity_grid_summary(
        cleaned_projected,
        cell_size,
        minimum_component_span,
    )
    raw_cells = raw_grid["cells"]
    cleaned_cells = cleaned_grid["cells"]
    shared_cells = raw_cells & cleaned_cells
    lost_cells = raw_cells - cleaned_cells
    raw_components = raw_grid["meaningful_components"]
    cleaned_components = cleaned_grid["meaningful_components"]
    raw_component_count = int(raw_grid["component_count"])
    cleaned_component_count = int(cleaned_grid["component_count"])
    raw_largest_span = float(raw_grid["largest_component_span"])
    cleaned_largest_span = float(cleaned_grid["largest_component_span"])
    largest_span_retention = (
        min(1.0, cleaned_largest_span / raw_largest_span)
        if raw_largest_span > 0.0
        else 0.0
    )
    component_match_values: list[float] = []
    for cleaned_component in cleaned_components:
        cleaned_component_cells = cleaned_component["cells"]
        best_overlap = max(
            (
                len(cleaned_component_cells & raw_component["cells"])
                / max(1, len(cleaned_component_cells))
                for raw_component in raw_components
            ),
            default=0.0,
        )
        component_match_values.append(float(best_overlap))
    component_match_ratio = (
        float(np.mean(component_match_values))
        if component_match_values
        else 0.0
    )
    allowed_profile_indices = (
        {int(index) for index in source_profile_indices}
        if source_profile_indices is not None
        else None
    )
    source_profile_component_counts: list[int] = []
    if isinstance(source_profiles, Mapping):
        for profile_index, profile_value in source_profiles.items():
            try:
                normalized_index = int(profile_index)
            except (TypeError, ValueError):
                continue
            if (
                allowed_profile_indices is not None
                and normalized_index not in allowed_profile_indices
            ):
                continue
            profile_coordinates = np.asarray(profile_value, dtype=np.float64)
            if (
                profile_coordinates.ndim != 2
                or profile_coordinates.shape[1] != 2
                or len(profile_coordinates) < SECTION_INTEGRITY_MINIMUM_POINTS
            ):
                continue
            profile_grid = _section_integrity_grid_summary(
                profile_coordinates,
                cell_size,
                minimum_component_span,
            )
            component_count = int(profile_grid["component_count"])
            if component_count:
                source_profile_component_counts.append(component_count)
    source_profile_consistency_ratio = 0.0
    source_profile_gap_evidence = False
    if source_profile_component_counts:
        component_count_frequencies: dict[int, int] = {}
        for component_count in source_profile_component_counts:
            component_count_frequencies[component_count] = (
                component_count_frequencies.get(component_count, 0) + 1
            )
        modal_component_count = max(
            component_count_frequencies,
            key=component_count_frequencies.get,
        )
        source_profile_consistency_ratio = float(
            component_count_frequencies[modal_component_count]
            / len(source_profile_component_counts)
        )
        source_profile_gap_evidence = bool(
            len(source_profile_component_counts) >= 3
            and modal_component_count > 1
            and source_profile_consistency_ratio >= 0.75
        )
    lost_cell_ratio = float(len(lost_cells) / max(1, len(raw_cells)))
    structural_change_detected = bool(
        cleaned_component_count > raw_component_count
        or (
            raw_component_count > 0
            and cleaned_component_count == 0
        )
        or (
            raw_component_count > 0
            and largest_span_retention < SECTION_INTEGRITY_MINIMUM_COMPONENT_RETENTION
        )
        or lost_cell_ratio > SECTION_INTEGRITY_MAXIMUM_LOST_CELL_RATIO
    )

    if structural_change_detected:
        if cleaned_component_count > raw_component_count or cleaned_component_count == 0:
            status = "cleanup_induced_gap"
            status_label = "截面对比发现线段缺失"
            message = (
                "最终截面相对基线出现新的断线或主要连续结构丢失，具体原因需结合运行版本复核；"
                "请先对照基线截面和融合云组件清理记录复核。"
            )
            recommendation = (
                "先对照基线截面；正式测量不要用加厚、插值或 CAD 反向修正补线。"
            )
        else:
            status = "mixed_structure_change"
            status_label = "截面结构发生变化"
            message = (
                "基线截面与最终截面结构变化较大，无法把缺口单独归因于扫描覆盖；"
                "请对照基线截面和融合云组件清理记录。"
            )
            recommendation = (
                "先保留基线截面作为诊断依据，逐项复核融合云组件清理；不要插值补线或按 CAD 调整点云。"
            )
    elif raw_component_count > 1 or source_profile_gap_evidence:
        status = "source_coverage_gap_likely"
        status_label = "更可能是原始覆盖/截面位置缺口"
        if source_profile_gap_evidence:
            message = (
                "可信工位薄截面的主要结构分段具有一致性，且基线截面与最终截面结构基本保持；"
                "若图中仍有断线，优先检查扫描覆盖和截面平面位置；融合云清理的影响需另行复核。"
            )
        else:
            message = (
                "基线截面已经存在多个主要连续结构，最终截面结构数量和跨度基本保持；"
                "若图中仍有断线，优先检查扫描覆盖和截面平面位置；融合云清理的影响需另行复核。"
            )
        recommendation = (
            "正式测量保持当前截面厚度（当前推荐 0.15）；检查扫描覆盖和截面偏移，"
            "不要用增厚、插值或 CAD 反向调整补线。"
        )
    else:
        status = "stable_after_cleanup"
        status_label = "基线与最终截面结构一致"
        message = (
            "基线截面与最终截面的主要连续结构一致；不能据此单独判断融合云清理是否影响真实结构。"
        )
        recommendation = (
            "可以继续用当前固定点云进行 CAD 拟合；若仍看到断口，优先检查融合云基线截面、扫描覆盖和截面位置。"
        )

    base_result.update(
        {
            "grid_cell_size": cell_size,
            "minimum_component_span": minimum_component_span,
            "raw_component_count": raw_component_count,
            "cleaned_component_count": cleaned_component_count,
            "raw_cell_count": int(raw_grid["cell_count"]),
            "cleaned_cell_count": int(cleaned_grid["cell_count"]),
            "shared_cell_count": len(shared_cells),
            "lost_cell_count": len(lost_cells),
            "lost_cell_ratio": lost_cell_ratio,
            "component_match_ratio": component_match_ratio,
            "largest_component_span_retention": largest_span_retention,
            "source_profile_count": len(source_profile_component_counts),
            "source_profile_component_counts": source_profile_component_counts,
            "source_profile_consistency_ratio": source_profile_consistency_ratio,
            "source_profile_gap_evidence": source_profile_gap_evidence,
            "structural_change_detected": structural_change_detected,
            "status": status,
            "status_label": status_label,
            "message": message,
            "recommendation": recommendation,
        }
    )
    return base_result


def extract_vertical_section(
    cloud: o3d.geometry.PointCloud,
    config: SectionConfig,
    backend: str = "cpu",
    clean: bool = True,
) -> o3d.geometry.PointCloud:
    section = _select_cloud_by_mask(
        cloud,
        _vertical_section_mask(np.asarray(cloud.points), config, backend),
    )
    return clean_vertical_section(section, config) if clean else section


def _voxel_row_membership(
    points: np.ndarray,
    reference_points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Return which point voxel keys also occur in a reference point set."""
    coordinates = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    reference = np.ascontiguousarray(np.asarray(reference_points, dtype=np.float64))
    if not len(coordinates) or not len(reference):
        return np.zeros(len(coordinates), dtype=bool)
    point_keys = np.floor(coordinates / float(voxel_size)).astype(np.int64)
    reference_keys = np.unique(
        np.floor(reference / float(voxel_size)).astype(np.int64),
        axis=0,
    )
    key_dtype = np.dtype(
        [
            ("x", np.int64),
            ("y", np.int64),
            ("z", np.int64),
        ]
    )
    point_view = point_keys.view(key_dtype).reshape(-1)
    reference_view = reference_keys.view(key_dtype).reshape(-1)
    return np.isin(point_view, reference_view)


def _recover_section_density_from_raw(
    path: Path,
    retained_cloud: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    leveling_transformation: np.ndarray,
    section_config: SectionConfig,
    denoise_voxel_size: float,
    backend: str,
    calibration_config: CalibrationConfig | None = None,
    *,
    calibration_bounds_in_reference_frame: bool = True,
) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    """Recover local raw samples without bypassing the denoise decision.

    The raw scan is used only for the selected thin section.  A raw sample is
    eligible only when its coarse denoise voxel is still occupied by the
    retained source cloud and its measured coordinate is close to that cloud.
    Voxel membership must use the original scan frame, just like denoise_cloud;
    rotating the voxel grid could otherwise revive rejected cells. Output is
    aligned using the caller's fixed pose, without modifying the inputs.
    """
    if not np.isfinite(denoise_voxel_size) or denoise_voxel_size <= 0:
        raise ValueError("Section recovery voxel size must be finite and positive.")
    final_transformation = (
        np.asarray(leveling_transformation, dtype=np.float64)
        @ np.asarray(transformation, dtype=np.float64)
    )
    report: dict[str, object] = {
        "enabled": True,
        "method": "raw_section_samples_inside_retained_denoise_voxels",
        "source": "original_ply",
        "file_name": path.name,
        "voxel_size": float(denoise_voxel_size),
        "voxel_frame": "source_input",
        "distance_factor": float(SECTION_DENSITY_RECOVERY_DISTANCE_FACTOR),
        "maximum_distance": float(
            denoise_voxel_size * SECTION_DENSITY_RECOVERY_DISTANCE_FACTOR
        ),
        "interpolation_applied": False,
        "cad_feedback_applied": False,
        "raw_input_points": 0,
        "raw_section_candidates": 0,
        "retained_cloud_points": int(len(retained_cloud.points)),
        "voxel_matched_candidates": 0,
        "near_retained_candidates": 0,
        "recovered_points": 0,
        "status": "disabled",
    }
    retained_points = np.asarray(retained_cloud.points, dtype=np.float64)
    retained_points = retained_points[np.isfinite(retained_points).all(axis=1)]
    if not len(retained_points):
        report["status"] = "no_retained_points"
        return o3d.geometry.PointCloud(), report

    raw_cloud = read_point_cloud(path)
    report["raw_input_points"] = int(len(raw_cloud.points))
    raw_points = np.asarray(raw_cloud.points)
    raw_section_mask = np.zeros(len(raw_points), dtype=bool)
    # Transform bounded chunks for selection, then retain only the thin slab.
    for start in range(0, len(raw_points), 250_000):
        chunk = raw_points[start:start + 250_000]
        aligned = chunk @ final_transformation[:3, :3].T + final_transformation[:3, 3]
        raw_section_mask[start:start + len(chunk)] = _vertical_section_mask(
            aligned, section_config, backend,
        )
    raw_section = _select_cloud_by_mask(raw_cloud, raw_section_mask)
    raw_section_points = np.asarray(raw_section.points, dtype=np.float64)
    report["raw_section_candidates"] = int(len(raw_section_points))
    if not len(raw_section_points):
        report["status"] = "no_raw_section_points"
        return o3d.geometry.PointCloud(), report

    voxel_matched = _voxel_row_membership(
        raw_section_points,
        retained_points,
        denoise_voxel_size,
    )
    report["voxel_matched_candidates"] = int(np.count_nonzero(voxel_matched))
    if not np.any(voxel_matched):
        report["status"] = "no_shared_denoise_voxels"
        return o3d.geometry.PointCloud(), report

    retained_tree = cKDTree(retained_points)
    distances = retained_tree.query(
        raw_section_points[voxel_matched],
        k=1,
        workers=_spatial_query_workers(),
    )[0]
    near_retained = distances <= float(report["maximum_distance"])
    report["near_retained_candidates"] = int(np.count_nonzero(near_retained))
    # Exclude exact representatives already present in the fusion input.
    new_samples = near_retained & (distances > 1e-9)
    keep = np.zeros(len(raw_section_points), dtype=bool)
    keep[np.flatnonzero(voxel_matched)[new_samples]] = True
    recovered = _transform_cloud(
        _select_cloud_by_mask(raw_section, keep), final_transformation, backend,
    )
    if (
        calibration_config is not None
        and calibration_config.use_roi
        and calibration_config.exclude_from_fusion
    ):
        recovered = remove_calibration_region(
            recovered, calibration_config, backend,
            coordinate_transformation=(
                leveling_transformation
                if calibration_bounds_in_reference_frame else final_transformation
            ),
        )
    report["recovered_points"] = int(len(recovered.points))
    report["status"] = "recovered" if len(recovered.points) else "already_represented"
    return recovered, report


def _concatenate_clouds(clouds: Sequence[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    nonempty_clouds = [cloud for cloud in clouds if len(cloud.points)]
    if not nonempty_clouds:
        return o3d.geometry.PointCloud()

    point_sets = [np.asarray(cloud.points) for cloud in nonempty_clouds]
    color_sets = [_cloud_colors_or_none(cloud) for cloud in nonempty_clouds]
    points = np.concatenate(point_sets, axis=0)
    colors = np.concatenate(color_sets, axis=0) if all(colors is not None for colors in color_sets) else None
    return _cloud_from_arrays(points, colors)


def _audit_parallel_section_layers(
    profile: np.ndarray,
    source_indices: np.ndarray,
    *,
    bin_size: float = 0.05,
    cluster_gap: float = 0.06,
) -> dict[str, object]:
    """Report parallel contour tracks without deciding that they are defects.

    A second contour can be a real complementary surface, an occlusion artifact,
    or a registration ghost.  The station-level consensus filter decides what to
    remove; this helper only records the evidence that was present in the retained
    section slab.
    """
    coordinates = np.asarray(profile, dtype=np.float64)
    sources = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) != len(sources)
        or not len(coordinates)
    ):
        return {
            "parallel_double_layer_detected": False,
            "parallel_double_layer_bin_count": 0,
            "parallel_double_layer_supported_bin_count": 0,
            "parallel_double_layer_isolated_bin_count": 0,
            "parallel_double_layer_ratio": 0.0,
            "parallel_layer_evaluated_bins": 0,
            "parallel_layer_track_count": 0,
            "parallel_layer_max_tracks_per_bin": 0,
        }

    finite_mask = np.isfinite(coordinates).all(axis=1)
    coordinates = coordinates[finite_mask]
    sources = sources[finite_mask]
    if not len(coordinates) or bin_size <= 0 or cluster_gap <= 0:
        return {
            "parallel_double_layer_detected": False,
            "parallel_double_layer_bin_count": 0,
            "parallel_double_layer_supported_bin_count": 0,
            "parallel_double_layer_isolated_bin_count": 0,
            "parallel_double_layer_ratio": 0.0,
            "parallel_layer_evaluated_bins": 0,
            "parallel_layer_track_count": 0,
            "parallel_layer_max_tracks_per_bin": 0,
        }

    origin = float(np.floor(np.min(coordinates[:, 0]) / bin_size) * bin_size)
    bin_indices = np.floor((coordinates[:, 0] - origin) / bin_size).astype(np.int64)
    double_layer_bins = 0
    supported_double_layer_bins = 0
    isolated_double_layer_bins = 0
    track_count = 0
    maximum_tracks = 0
    evaluated_bins = int(len(np.unique(bin_indices)))

    for bin_index in np.unique(bin_indices):
        members = np.flatnonzero(bin_indices == bin_index)
        if len(members) < 2:
            continue
        ordered = members[np.argsort(coordinates[members, 1])]
        heights = coordinates[ordered, 1]
        split_positions = np.flatnonzero(np.diff(heights) > cluster_gap) + 1
        clusters = np.split(ordered, split_positions)
        clusters = [cluster for cluster in clusters if len(cluster) >= 2]
        if len(clusters) < 2:
            continue
        double_layer_bins += 1
        maximum_tracks = max(maximum_tracks, len(clusters))
        supports = [len(np.unique(sources[cluster])) for cluster in clusters]
        track_count += len(clusters)
        if all(support >= 2 for support in supports):
            supported_double_layer_bins += 1
        elif any(support == 1 for support in supports) and any(support >= 2 for support in supports):
            isolated_double_layer_bins += 1

    return {
        "parallel_double_layer_detected": bool(double_layer_bins),
        "parallel_double_layer_bin_count": int(double_layer_bins),
        "parallel_double_layer_supported_bin_count": int(supported_double_layer_bins),
        "parallel_double_layer_isolated_bin_count": int(isolated_double_layer_bins),
        "parallel_double_layer_ratio": float(double_layer_bins / max(evaluated_bins, 1)),
        "parallel_layer_evaluated_bins": evaluated_bins,
        "parallel_layer_track_count": int(track_count),
        "parallel_layer_max_tracks_per_bin": int(maximum_tracks),
    }


def _collapse_parallel_section_layers(
    cloud: o3d.geometry.PointCloud,
    source_indices: np.ndarray,
    support_counts: np.ndarray,
    section_config: SectionConfig,
    *,
    spacing: float = 0.025,
    radius: float = 0.075,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, dict[str, object]]:
    """Select real samples along local 2-D tangents without moving coordinates."""
    points = np.asarray(cloud.points, dtype=np.float64)
    sources = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    supports = np.asarray(support_counts, dtype=np.int64).reshape(-1)
    if len(points) != len(sources) or len(points) != len(supports):
        raise ValueError("截面层聚合的点、来源和支持度数量必须一致。")
    if not (np.isfinite(spacing) and np.isfinite(radius) and 0 < spacing < radius):
        raise ValueError("Contour spacing and radius must be finite and 0 < spacing < radius.")
    if not np.isfinite(points).all():
        raise ValueError("Contour points must be finite.")
    _, horizontal, vertical = vertical_plane_basis(
        section_config.angle_degrees,
        section_config.tilt_degrees,
    )
    profile = np.column_stack((points @ horizontal, points @ vertical))
    tree = cKDTree(profile)
    available = np.ones(len(points), dtype=bool)
    selected: list[int] = []
    for seed in np.lexsort((profile[:, 1], profile[:, 0])):
        if not available[seed]:
            continue
        neighbors = np.asarray(tree.query_ball_point(profile[seed], radius), dtype=np.int64)
        delta = profile[neighbors] - profile[seed]
        centered = delta - np.median(delta, axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        tangent, normal = vectors[:, 1], vectors[:, 0]
        along, across = delta @ tangent, delta @ normal
        # Ambiguous corners stay locally sampled; never choose one global branch.
        linear = values[1] > 4.0 * max(values[0], 1e-12)
        members_mask = (
            (np.abs(along) <= spacing / 2)
            & (np.abs(across - np.median(across)) <= radius / 2)
            if linear else np.linalg.norm(delta, axis=1) <= spacing / 2
        )
        members = neighbors[members_mask & available[neighbors]]
        if not len(members):
            members = np.asarray([seed])
        local = profile[members]
        normal_residual = np.abs(local @ normal - np.median(local @ normal))
        along_residual = np.abs(local @ tangent - np.median(local @ tangent))
        ranking = np.lexsort((
            members, -supports[members], along_residual, normal_residual,
        ))
        representative = int(members[ranking[0]])
        selected.append(representative)
        available[members] = False
        # Always make progress even if the seed lies outside the median band.
        if available[seed]:
            selected.append(int(seed))
            available[seed] = False

    keep = np.sort(np.asarray(selected, dtype=np.int64))
    colors = _cloud_colors_or_none(cloud)
    collapsed_cloud = _cloud_from_arrays(
        points[keep],
        colors[keep] if colors is not None else None,
    )
    report = {
        "parallel_layer_collapse_applied": len(keep) < len(points),
        "parallel_layer_collapse_method": "local_tangent_real_representative",
        "input_points": len(points),
        "output_points": len(keep),
        "parallel_layer_collapse_points_removed": len(points) - len(keep),
        "spacing": spacing,
        "neighborhood_radius": radius,
        "coordinates_changed": False,
        "interpolation_applied": False,
        "cad_feedback_applied": False,
    }
    return collapsed_cloud, sources[keep], supports[keep], report


def _support_weighted_section_fusion(
    section_parts: Sequence[o3d.geometry.PointCloud],
    source_indices: Sequence[int],
    voxel_size: float,
    section_config: SectionConfig,
    *,
    return_source_indices: bool = False,
) -> tuple[o3d.geometry.PointCloud, dict[str, object]] | tuple[
    o3d.geometry.PointCloud, dict[str, object], np.ndarray, np.ndarray
]:
    """Fuse retained section points by selecting real samples, never averaging layers.

    Open3D's voxel sampler averages all points in a cell.  In a thin section that
    can place the output between two parallel surfaces.  This selector keeps one
    original point per 3-D section voxel, preferring a point with cross-view support
    and the smallest local profile residual.  Voxels covered by only one station are
    deliberately retained so complementary views are not erased.
    """
    if len(section_parts) != len(source_indices):
        raise ValueError("截面支持度融合的点云和来源索引数量必须一致。")
    if voxel_size <= 0:
        raise ValueError("截面支持度融合体素大小必须大于 0。")

    point_sets: list[np.ndarray] = []
    source_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray | None] = []
    colors_complete = True
    for cloud, source_index in zip(section_parts, source_indices, strict=True):
        points = np.asarray(cloud.points, dtype=np.float64)
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if not len(points):
            continue
        point_sets.append(points)
        source_sets.append(np.full(len(points), int(source_index), dtype=np.int64))
        colors = _cloud_colors_or_none(cloud)
        if colors is None or len(colors) != len(np.asarray(cloud.points)):
            colors_complete = False
            color_sets.append(None)
        else:
            color_sets.append(np.asarray(colors, dtype=np.float64)[finite_mask])

    if not point_sets:
        empty_report = {
            "section_fusion_confidence": {
                "status": "insufficient_data",
                "confidence_score": 0.0,
                "section_voxel_count": 0,
                "section_voxel_support_counts": {},
                "section_voxel_multi_view_ratio": 0.0,
                "section_voxel_unique_view_ratio": 0.0,
                "section_local_residual_median": None,
                "section_local_residual_p95": None,
                "unique_coverage_retained": True,
            }
        }
        if return_source_indices:
            return (
                o3d.geometry.PointCloud(),
                empty_report,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        return o3d.geometry.PointCloud(), empty_report

    points = np.concatenate(point_sets, axis=0)
    sources = np.concatenate(source_sets, axis=0)
    colors = np.concatenate(color_sets, axis=0) if colors_complete and color_sets else None
    _, horizontal, vertical = vertical_plane_basis(
        section_config.angle_degrees,
        section_config.tilt_degrees,
    )
    profile = np.column_stack((points @ horizontal, points @ vertical))

    # The nearest point from another station is a local residual proxy.  It is
    # computed in the displayed section coordinates, so it remains interpretable.
    cross_view_distance = np.full(len(points), np.inf, dtype=np.float64)
    for source_index in np.unique(sources):
        own_mask = sources == source_index
        other_mask = ~own_mask
        if np.any(own_mask) and np.any(other_mask):
            tree = cKDTree(profile[other_mask])
            cross_view_distance[own_mask] = tree.query(profile[own_mask], k=1)[0]

    voxel_keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    split_positions = np.flatnonzero(np.diff(inverse[order])) + 1
    groups = np.split(order, split_positions)
    support_counts = np.asarray(
        [len(np.unique(sources[members])) for members in groups],
        dtype=np.int64,
    )

    selected_indices: list[int] = []
    for voxel_position, members in enumerate(groups):
        if not len(members):
            continue
        voxel_center = points[members].mean(axis=0)
        center_distance = np.linalg.norm(points[members] - voxel_center, axis=1)
        residual = cross_view_distance[members]
        if support_counts[voxel_position] < 2:
            residual = np.full(len(members), np.inf, dtype=np.float64)
        # lexsort's last key is primary: finite/small local residual first,
        # then a central point, then a stable source index tie-breaker.
        finite_priority = (~np.isfinite(residual)).astype(np.int8)
        ranking = np.lexsort(
            (
                sources[members],
                center_distance,
                residual,
                finite_priority,
            )
        )
        selected_indices.append(int(members[int(ranking[0])]))

    selected = np.asarray(selected_indices, dtype=np.int64)
    fused_cloud = _cloud_from_arrays(
        points[selected],
        colors[selected] if colors is not None else None,
    )

    support_values, support_frequencies = np.unique(support_counts, return_counts=True)
    support_distribution = {
        str(int(support)): int(frequency)
        for support, frequency in zip(support_values, support_frequencies, strict=True)
    }
    multi_view_mask = support_counts >= 2
    unique_view_mask = support_counts == 1
    # Residuals are indexed per selected voxel; recompute the selected support
    # vector so it stays aligned even when a voxel contained duplicate points.
    selected_support = support_counts
    selected_residuals = cross_view_distance[selected]
    finite_residuals = selected_residuals[
        np.isfinite(selected_residuals) & (selected_support >= 2)
    ]
    residual_median = float(np.median(finite_residuals)) if len(finite_residuals) else None
    residual_p95 = float(np.quantile(finite_residuals, 0.95)) if len(finite_residuals) else None
    maximum_support = max(int(np.max(support_counts)) if len(support_counts) else 1, 2)
    if np.any(multi_view_mask):
        support_strength = float(
            np.mean(
                np.clip(
                    (support_counts[multi_view_mask] - 1) / max(maximum_support - 1, 1),
                    0.0,
                    1.0,
                )
            )
        )
        residual_scale = max(float(section_config.thickness), float(voxel_size) * 4.0, 0.05)
        residual_score = (
            max(0.0, 1.0 - min(float(residual_p95) / residual_scale, 1.0))
            if residual_p95 is not None
            else 0.0
        )
        confidence_score = float(np.clip(0.65 * support_strength + 0.35 * residual_score, 0.0, 1.0))
    else:
        support_strength = 0.0
        residual_score = 0.0
        confidence_score = 0.0

    layer_report = _audit_parallel_section_layers(profile, sources)
    confidence_report = {
        "status": "ok" if len(selected) else "insufficient_data",
        "confidence_score": confidence_score,
        "support_strength": support_strength,
        "local_residual_score": residual_score,
        "section_voxel_size": float(voxel_size),
        "section_voxel_count": int(len(groups)),
        "section_voxel_support_counts": support_distribution,
        "section_voxel_multi_view_count": int(np.count_nonzero(multi_view_mask)),
        "section_voxel_unique_view_count": int(np.count_nonzero(unique_view_mask)),
        "section_voxel_multi_view_ratio": float(np.mean(multi_view_mask)) if len(multi_view_mask) else 0.0,
        "section_voxel_unique_view_ratio": float(np.mean(unique_view_mask)) if len(unique_view_mask) else 0.0,
        "section_points_before_selection": int(len(points)),
        "section_points_after_selection": int(len(selected)),
        "section_points_removed_by_selection": int(len(points) - len(selected)),
        "section_local_residual_median": residual_median,
        "section_local_residual_p95": residual_p95,
        "unique_coverage_retained": True,
        "selection_policy": "prefer_multi_view_support_then_local_cross_view_residual_then_voxel_center",
        **layer_report,
    }
    report = {"fusion_confidence": confidence_report}
    if return_source_indices:
        return (
            fused_cloud,
            report,
            sources[selected].astype(np.int64, copy=True),
            selected_support.astype(np.int64, copy=True),
        )
    return fused_cloud, report


def _local_consensus_profile_masks(
    profiles: Sequence[np.ndarray],
    profile_bin_size: float = 0.05,
    profile_cluster_gap: float = 0.06,
    profile_cluster_margin: float = 0.015,
    minimum_source_views: int = SECTION_CONSENSUS_MINIMUM_SOURCE_VIEWS,
    *,
    conservative: bool = False,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Find section tracks supported by multiple views and report unsupported points."""
    if profile_bin_size <= 0 or profile_cluster_gap <= 0 or profile_cluster_margin < 0:
        raise ValueError("截面共识的分箱、聚类间距和边界参数必须有效。")
    if minimum_source_views < 2:
        raise ValueError("截面共识至少需要两个来源工位。")

    normalized_profiles: list[np.ndarray] = []
    for profile in profiles:
        coordinates = np.asarray(profile, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            coordinates = np.empty((0, 2), dtype=np.float64)
        normalized_profiles.append(coordinates)

    keep_masks = [np.ones(len(profile), dtype=bool) for profile in normalized_profiles]
    evaluated_masks = [np.zeros(len(profile), dtype=bool) for profile in normalized_profiles]
    evaluated_bins = [set[int]() for _ in normalized_profiles]
    candidate_masks = [np.zeros(len(profile), dtype=bool) for profile in normalized_profiles]
    candidate_bins = [set[int]() for _ in normalized_profiles]
    candidate_bin_positions = [
        np.full(len(profile), np.iinfo(np.int64).min, dtype=np.int64)
        for profile in normalized_profiles
    ]
    empty_reports = [
        {
            "profile_points": int(len(profile)),
            "consensus_mode": "conservative" if conservative else "strict",
            "consensus_evaluated_points": 0,
            "consensus_supported_points": 0,
            "consensus_removed_points": 0,
            "consensus_removed_ratio": 0.0,
            "consensus_shared_removed_ratio": 0.0,
            "consensus_evaluated_bins": 0,
            "consensus_shared_horizontal_span": 0.0,
            "conservative_candidate_points": 0,
            "conservative_candidate_bins": 0,
            "conservative_qualified_track_bins": 0,
            "conservative_qualified_tracks": 0,
        }
        for profile in normalized_profiles
    ]
    if not normalized_profiles:
        return keep_masks, empty_reports

    finite_source_candidates = [
        index
        for index, profile in enumerate(normalized_profiles)
        if len(profile) and np.isfinite(profile).all(axis=1).any()
    ]
    if len(finite_source_candidates) < minimum_source_views:
        return keep_masks, empty_reports

    all_profiles = np.concatenate(normalized_profiles, axis=0)
    source_indices = np.concatenate(
        [
            np.full(len(profile), index, dtype=np.int32)
            for index, profile in enumerate(normalized_profiles)
        ]
    )
    finite_mask = np.isfinite(all_profiles).all(axis=1)
    if not np.any(finite_mask):
        return keep_masks, empty_reports

    finite_profiles = all_profiles[finite_mask]
    finite_source_indices = source_indices[finite_mask]
    finite_point_positions = np.flatnonzero(finite_mask)
    point_offsets = np.cumsum([0, *(len(profile) for profile in normalized_profiles)])
    bin_origin = float(
        np.floor(finite_profiles[:, 0].min() / profile_bin_size) * profile_bin_size
    )
    bin_indices = np.floor(
        (finite_profiles[:, 0] - bin_origin) / profile_bin_size
    ).astype(np.int64)

    for bin_index in np.unique(bin_indices):
        members = np.flatnonzero(bin_indices == bin_index)
        if not len(members):
            continue
        available_sources = np.unique(finite_source_indices[members])
        if len(available_sources) < minimum_source_views:
            continue

        ordered_members = members[np.argsort(finite_profiles[members, 1])]
        ordered_heights = finite_profiles[ordered_members, 1]
        split_positions = np.flatnonzero(np.diff(ordered_heights) > profile_cluster_gap) + 1
        clusters = np.split(ordered_members, split_positions)
        minimum_support = max(2, int(np.ceil(len(available_sources) / 2)))
        supported_clusters = [
            cluster
            for cluster in clusters
            if len(np.unique(finite_source_indices[cluster])) >= minimum_support
        ]
        if not supported_clusters:
            continue

        if conservative:
            for source_index in available_sources:
                source_members = members[finite_source_indices[members] == source_index]
                local_positions = (
                    finite_point_positions[source_members] - point_offsets[source_index]
                )
                evaluated_masks[source_index][local_positions] = True
                evaluated_bins[source_index].add(int(bin_index))

            supported_ranges = []
            for cluster in supported_clusters:
                cluster_heights = finite_profiles[cluster, 1]
                supported_ranges.append(
                    (
                        float(cluster_heights.min() - profile_cluster_margin),
                        float(cluster_heights.max() + profile_cluster_margin),
                    )
                )
            isolation_distance = max(
                SECTION_CONSENSUS_ISOLATED_TRACK_DISTANCE,
                profile_cluster_gap * 2.0,
                profile_cluster_margin * 4.0,
            )
            for cluster in clusters:
                cluster_sources = np.unique(finite_source_indices[cluster])
                if len(cluster_sources) != 1:
                    continue
                cluster_heights = finite_profiles[cluster, 1]
                cluster_center = float(np.median(cluster_heights))
                nearest_supported_distance = min(
                    max(lower - cluster_center, 0.0, cluster_center - upper)
                    for lower, upper in supported_ranges
                )
                if nearest_supported_distance < isolation_distance:
                    continue
                source_index = int(cluster_sources[0])
                source_members = cluster[
                    finite_source_indices[cluster] == source_index
                ]
                local_positions = (
                    finite_point_positions[source_members] - point_offsets[source_index]
                )
                candidate_masks[source_index][local_positions] = True
                candidate_bin_positions[source_index][local_positions] = int(bin_index)
                candidate_bins[source_index].add(int(bin_index))
        else:
            for source_index in available_sources:
                source_members = members[finite_source_indices[members] == source_index]
                local_positions = (
                    finite_point_positions[source_members] - point_offsets[source_index]
                )
                evaluated_masks[source_index][local_positions] = True
                keep_masks[source_index][local_positions] = False
                evaluated_bins[source_index].add(int(bin_index))

            for cluster in supported_clusters:
                cluster_sources = np.unique(finite_source_indices[cluster])
                cluster_heights = finite_profiles[cluster, 1]
                lower = float(cluster_heights.min() - profile_cluster_margin)
                upper = float(cluster_heights.max() + profile_cluster_margin)
                selected_members = members[
                    np.isin(finite_source_indices[members], cluster_sources)
                    & (finite_profiles[members, 1] >= lower)
                    & (finite_profiles[members, 1] <= upper)
                ]
                for source_index in np.unique(finite_source_indices[selected_members]):
                    source_members = selected_members[
                        finite_source_indices[selected_members] == source_index
                    ]
                    local_positions = (
                        finite_point_positions[source_members] - point_offsets[source_index]
                    )
                    keep_masks[source_index][local_positions] = True

    qualified_track_bins = [set[int]() for _ in normalized_profiles]
    qualified_track_counts = [0 for _ in normalized_profiles]
    if conservative:
        for source_index, source_bins in enumerate(candidate_bins):
            ordered_bins = sorted(source_bins)
            if not ordered_bins:
                continue
            current_run = [ordered_bins[0]]
            for bin_index in ordered_bins[1:]:
                if (
                    bin_index - current_run[-1]
                    <= SECTION_CONSENSUS_MAX_TRACK_GAP_BINS + 1
                ):
                    current_run.append(bin_index)
                else:
                    if len(current_run) >= SECTION_CONSENSUS_MINIMUM_ISOLATED_TRACK_BINS:
                        qualified_track_bins[source_index].update(current_run)
                        qualified_track_counts[source_index] += 1
                    current_run = [bin_index]
            if len(current_run) >= SECTION_CONSENSUS_MINIMUM_ISOLATED_TRACK_BINS:
                qualified_track_bins[source_index].update(current_run)
                qualified_track_counts[source_index] += 1

            if not qualified_track_bins[source_index]:
                continue
            remove_mask = candidate_masks[source_index] & np.isin(
                candidate_bin_positions[source_index],
                np.asarray(sorted(qualified_track_bins[source_index]), dtype=np.int64),
            )
            keep_masks[source_index][remove_mask] = False

    reports: list[dict[str, object]] = []
    for source_index, (profile, keep_mask, evaluated_mask, bin_set) in enumerate(zip(
        normalized_profiles,
        keep_masks,
        evaluated_masks,
        evaluated_bins,
        strict=True,
    )):
        evaluated_count = int(np.count_nonzero(evaluated_mask))
        removed_mask = evaluated_mask & ~keep_mask
        removed_count = int(np.count_nonzero(removed_mask))
        shared_span = 0.0
        if evaluated_count > 1:
            shared_span = float(np.ptp(profile[evaluated_mask, 0]))
        reports.append(
            {
                "profile_points": int(len(profile)),
                "consensus_mode": "conservative" if conservative else "strict",
                "consensus_evaluated_points": evaluated_count,
                "consensus_supported_points": int(
                    np.count_nonzero(evaluated_mask & keep_mask)
                ),
                "consensus_removed_points": removed_count,
                "consensus_removed_ratio": float(removed_count / len(profile))
                if len(profile)
                else 0.0,
                "consensus_shared_removed_ratio": float(removed_count / evaluated_count)
                if evaluated_count
                else 0.0,
                "consensus_evaluated_bins": int(len(bin_set)),
                "consensus_shared_horizontal_span": shared_span,
                "conservative_candidate_points": int(
                    np.count_nonzero(candidate_masks[source_index])
                ),
                "conservative_candidate_bins": int(len(candidate_bins[source_index])),
                "conservative_qualified_track_bins": int(
                    len(qualified_track_bins[source_index])
                ),
                "conservative_qualified_tracks": int(
                    qualified_track_counts[source_index]
                ),
            }
        )
    return keep_masks, reports


def _local_consensus_keep_masks(
    section_parts: Sequence[o3d.geometry.PointCloud],
    config: SectionConfig,
    profile_bin_size: float = 0.05,
    profile_cluster_gap: float = 0.06,
    profile_cluster_margin: float = 0.015,
    *,
    conservative: bool = False,
) -> list[np.ndarray]:
    """Return per-view masks that keep section tracks supported by other views."""
    _, horizontal, vertical = vertical_plane_basis(
        config.angle_degrees,
        config.tilt_degrees,
    )
    profiles = [
        np.column_stack(
            (
                np.asarray(cloud.points) @ horizontal,
                np.asarray(cloud.points) @ vertical,
            )
        )
        if len(cloud.points)
        else np.empty((0, 2), dtype=np.float64)
        for cloud in section_parts
    ]
    keep_masks, _ = _local_consensus_profile_masks(
        profiles,
        profile_bin_size,
        profile_cluster_gap,
        profile_cluster_margin,
        conservative=conservative,
    )
    return keep_masks


def _apply_section_consensus_station_filter(
    profiles: Mapping[int, np.ndarray],
    active_indices: Sequence[int],
    metrics: Sequence[RegistrationMetric],
    reference_index: int,
    registration_config: RegistrationConfig,
    mismatch_evidence_counts: Mapping[int, int] | None = None,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
    *,
    conservative: bool = False,
    apply_filter: bool = True,
) -> dict[str, object]:
    """Diagnose shared section support, optionally excluding anomalous scans.

    Coverage-first processing passes ``apply_filter=False``.  In that mode the
    same evidence is calculated for the audit report, but no station or point
    is removed because a complementary view is expected to contain unique
    target coverage.
    """
    initial_indices = list(dict.fromkeys(int(index) for index in active_indices))
    minimum_retained_views = min(
        registration_config.section_alignment_minimum_views,
        len(initial_indices),
    )
    mismatch_counts = {
        int(index): int(value)
        for index, value in (mismatch_evidence_counts or {}).items()
    }
    minimum_evidence_points = max(
        20,
        min(
            SECTION_CONSENSUS_MINIMUM_EVIDENCE_POINTS,
            int(registration_config.section_alignment_minimum_overlap_points),
        ),
    )

    def calculate(indices: Sequence[int]) -> dict[int, dict[str, object]]:
        empty_profile = np.empty((0, 2), dtype=np.float64)
        ordered_profiles = [
            np.asarray(profiles.get(index, empty_profile), dtype=np.float64)
            for index in indices
        ]
        _, reports = _local_consensus_profile_masks(
            ordered_profiles,
            conservative=conservative,
        )
        return {
            int(index): dict(report)
            for index, report in zip(indices, reports, strict=True)
        }

    def is_outlier(index: int, report: Mapping[str, object]) -> bool:
        if index == reference_index:
            return False
        if int(report.get("consensus_evaluated_points", 0)) < minimum_evidence_points:
            return False
        if int(report.get("consensus_evaluated_bins", 0)) < SECTION_CONSENSUS_MINIMUM_EVALUATED_BINS:
            return False
        shared_removed_ratio = float(
            report.get("consensus_shared_removed_ratio", 0.0)
        )
        removed_ratio = float(report.get("consensus_removed_ratio", 0.0))
        if conservative:
            if shared_removed_ratio < 0.30:
                return False
            if removed_ratio >= 0.40 and shared_removed_ratio >= 0.35:
                return True
            return bool(
                removed_ratio >= 0.25
                and shared_removed_ratio >= 0.40
                and mismatch_counts.get(index, 0) >= 2
            )
        if shared_removed_ratio < SECTION_CONSENSUS_MIN_SHARED_REMOVAL_RATIO:
            return False
        if removed_ratio >= SECTION_CONSENSUS_HARD_REMOVAL_RATIO:
            return True
        return bool(
            removed_ratio >= SECTION_CONSENSUS_SOFT_REMOVAL_RATIO
            and shared_removed_ratio >= 0.45
            and mismatch_counts.get(index, 0) >= 1
        )

    current_indices = initial_indices.copy()
    excluded_reports: dict[int, dict[str, object]] = {}
    iterations: list[dict[str, object]] = []
    blocked_candidates: list[int] = []
    diagnostic_excluded_indices: list[int] = []
    while apply_filter and len(current_indices) > minimum_retained_views:
        current_reports = calculate(current_indices)
        candidate_indices = [
            index
            for index in current_indices
            if is_outlier(index, current_reports.get(index, {}))
        ]
        if not candidate_indices:
            break

        def severity(index: int) -> tuple[float, float, int, int]:
            report = current_reports[index]
            return (
                float(report.get("consensus_removed_ratio", 0.0)),
                float(report.get("consensus_shared_removed_ratio", 0.0)),
                mismatch_counts.get(index, 0),
                int(report.get("consensus_evaluated_points", 0)),
            )

        excluded_index = max(candidate_indices, key=severity)
        excluded_report = dict(current_reports[excluded_index])
        excluded_report["mismatch_evidence_count"] = mismatch_counts.get(
            excluded_index,
            0,
        )
        excluded_reports[excluded_index] = excluded_report
        iterations.append(
            {
                "active_indices": current_indices.copy(),
                "excluded_index": excluded_index,
                "excluded_report": excluded_report,
            }
        )
        metric = metrics[excluded_index]
        metric.included_in_fusion = False
        metric.section_alignment_passed = False
        metric.section_validation_status = "mismatch"
        metric.exclusion_reason = "section_consensus"
        _emit(
            progress_callback,
            f"Group {group_name}: excluded {metric.file_name}; "
            f"shared contour consensus removed "
            f"{float(excluded_report.get('consensus_removed_ratio', 0.0)):.1%} "
            f"of its section points",
        )
        current_indices.remove(excluded_index)

    final_reports = calculate(current_indices)
    if not apply_filter:
        diagnostic_excluded_indices = [
            index
            for index in initial_indices
            if is_outlier(index, final_reports.get(index, {}))
        ]
    elif len(current_indices) <= minimum_retained_views:
        blocked_candidates = [
            index
            for index in current_indices
            if is_outlier(index, final_reports.get(index, {}))
        ]

    all_reports: dict[int, dict[str, object]] = {}
    for index in initial_indices:
        report = dict(final_reports.get(index, excluded_reports.get(index, {})))
        report["mismatch_evidence_count"] = mismatch_counts.get(index, 0)
        all_reports[index] = report
        metric = metrics[index]
        metric.section_consensus_evaluated_points = int(
            report.get("consensus_evaluated_points", 0)
        )
        metric.section_consensus_removed_points = int(
            report.get("consensus_removed_points", 0)
        )
        metric.section_consensus_removed_ratio = float(
            report.get("consensus_removed_ratio", 0.0)
        )
        metric.section_consensus_shared_removed_ratio = float(
            report.get("consensus_shared_removed_ratio", 0.0)
        )
        metric.section_consensus_evaluated_bins = int(
            report.get("consensus_evaluated_bins", 0)
        )
        metric.section_consensus_score = max(
            0.0,
            1.0
            - max(
                metric.section_consensus_removed_ratio,
                metric.section_consensus_shared_removed_ratio,
            ),
        )

    view_reports = []
    for index in initial_indices:
        metric = metrics[index]
        report = all_reports[index]
        view_reports.append(
            {
                "index": index,
                "file_name": metric.file_name,
                "pose_prior_label": metric.pose_prior_label,
                **report,
                "status": metric.section_validation_status,
                "included_in_fusion": bool(metric.included_in_fusion),
                "exclusion_reason": metric.exclusion_reason,
            }
        )

    return {
        "enabled": True,
        "method": (
            "conservative_shared_section_contour_consensus_station_filter"
            if conservative
            else "shared_section_contour_consensus_station_filter"
        ),
        "mode": "conservative" if conservative else "strict",
        "filter_applied": bool(apply_filter),
        "diagnostic_only": not bool(apply_filter),
        "coordinates_adjusted": False,
        "initial_active_indices": initial_indices,
        "final_active_indices": current_indices,
        "excluded_indices": [
            index for index in initial_indices if index not in current_indices
        ],
        "diagnostic_excluded_indices": diagnostic_excluded_indices,
        "minimum_retained_views": minimum_retained_views,
        "minimum_evidence_points": minimum_evidence_points,
        "minimum_evaluated_bins": SECTION_CONSENSUS_MINIMUM_EVALUATED_BINS,
        "minimum_source_views": SECTION_CONSENSUS_MINIMUM_SOURCE_VIEWS,
        "soft_removed_ratio": SECTION_CONSENSUS_SOFT_REMOVAL_RATIO,
        "hard_removed_ratio": SECTION_CONSENSUS_HARD_REMOVAL_RATIO,
        "minimum_shared_removed_ratio": SECTION_CONSENSUS_MIN_SHARED_REMOVAL_RATIO,
        "iterations": iterations,
        "blocked_candidates_by_minimum_views": blocked_candidates,
        "views": view_reports,
    }


def _filter_section_parts_by_local_consensus(
    section_parts: Sequence[o3d.geometry.PointCloud],
    config: SectionConfig,
    profile_bin_size: float = 0.05,
    profile_cluster_gap: float = 0.06,
    profile_cluster_margin: float = 0.015,
    *,
    conservative: bool = False,
) -> list[o3d.geometry.PointCloud]:
    """Remove isolated parallel section tracks while preserving unique coverage."""
    keep_masks = _local_consensus_keep_masks(
        section_parts,
        config,
        profile_bin_size,
        profile_cluster_gap,
        profile_cluster_margin,
        conservative=conservative,
    )
    filtered_parts = [
        _select_cloud_by_mask(cloud, keep_mask)
        for cloud, keep_mask in zip(section_parts, keep_masks, strict=True)
        if np.any(keep_mask)
    ]
    return filtered_parts or [cloud for cloud in section_parts if len(cloud.points)]


def _section_profile_coordinates(
    cloud: o3d.geometry.PointCloud,
    config: SectionConfig,
) -> np.ndarray:
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return np.empty((0, 2), dtype=np.float64)
    _, horizontal, vertical = vertical_plane_basis(
        config.angle_degrees,
        config.tilt_degrees,
    )
    profile = np.column_stack((points @ horizontal, points @ vertical))
    return profile[np.isfinite(profile).all(axis=1)]


def _trimmed_profile_distance(
    source_profile: np.ndarray,
    target_profile: np.ndarray,
    keep_ratio: float = 0.80,
) -> float:
    if not len(source_profile) or not len(target_profile):
        return float("inf")
    distances = cKDTree(target_profile).query(source_profile)[0]
    cutoff = float(np.quantile(distances, keep_ratio))
    return float(np.mean(np.minimum(distances, cutoff)))


def _estimate_section_pose_correction(
    source_profile: np.ndarray,
    target_profile: np.ndarray,
    maximum_correspondence_distance: float = 0.25,
    maximum_rotation_degrees: float = 1.25,
    maximum_translation: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    source_profile = np.asarray(source_profile, dtype=np.float64)
    target_profile = np.asarray(target_profile, dtype=np.float64)
    source_profile = source_profile[np.isfinite(source_profile).all(axis=1)]
    target_profile = target_profile[np.isfinite(target_profile).all(axis=1)]
    if len(source_profile) < 30 or len(target_profile) < 30:
        return None

    source_step = max(1, len(source_profile) // 12_000)
    target_step = max(1, len(target_profile) // 16_000)
    source_work = source_profile[::source_step]
    target_work = target_profile[::target_step]
    current_profile = source_work.copy()
    total_rotation = np.eye(2, dtype=np.float64)
    total_translation = np.zeros(2, dtype=np.float64)
    target_tree = cKDTree(target_work)
    before_score = _trimmed_profile_distance(source_work, target_work)

    for _ in range(8):
        distances, nearest_indices = target_tree.query(current_profile)
        correspondences = distances <= maximum_correspondence_distance
        if int(np.count_nonzero(correspondences)) < 24:
            break
        inlier_cutoff = float(np.quantile(distances[correspondences], 0.75))
        correspondences &= distances <= inlier_cutoff
        if int(np.count_nonzero(correspondences)) < 24:
            break

        source_matches = current_profile[correspondences]
        target_matches = target_work[nearest_indices[correspondences]]
        source_center = source_matches.mean(axis=0)
        target_center = target_matches.mean(axis=0)
        covariance = (source_matches - source_center).T @ (target_matches - target_center)
        left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance)
        incremental_rotation = right_vectors_transposed.T @ left_vectors.T
        if np.linalg.det(incremental_rotation) < 0:
            right_vectors_transposed[-1] *= -1.0
            incremental_rotation = right_vectors_transposed.T @ left_vectors.T

        incremental_angle = float(
            np.arctan2(incremental_rotation[1, 0], incremental_rotation[0, 0])
        )
        maximum_incremental_angle = np.radians(0.35)
        incremental_angle = float(
            np.clip(incremental_angle, -maximum_incremental_angle, maximum_incremental_angle)
        )
        incremental_rotation = np.array(
            [
                [np.cos(incremental_angle), -np.sin(incremental_angle)],
                [np.sin(incremental_angle), np.cos(incremental_angle)],
            ],
            dtype=np.float64,
        )
        incremental_translation = target_center - source_center @ incremental_rotation.T
        incremental_translation = np.clip(incremental_translation, -0.08, 0.08)
        current_profile = current_profile @ incremental_rotation.T + incremental_translation
        total_rotation = incremental_rotation @ total_rotation
        total_translation = total_translation @ incremental_rotation.T + incremental_translation

        if abs(incremental_angle) < np.radians(0.001) and np.linalg.norm(incremental_translation) < 0.001:
            break

    final_angle = float(np.arctan2(total_rotation[1, 0], total_rotation[0, 0]))
    if abs(final_angle) > np.radians(maximum_rotation_degrees):
        return None
    translation_norm = float(np.linalg.norm(total_translation))
    if translation_norm > maximum_translation:
        return None
    after_score = _trimmed_profile_distance(current_profile, target_work)
    if not np.isfinite(before_score) or not np.isfinite(after_score) or after_score >= before_score * 0.995:
        return None
    return total_rotation, total_translation, before_score, after_score


def refine_transforms_with_section_consensus(
    cleaned_clouds: Sequence[o3d.geometry.PointCloud],
    transformations: list[np.ndarray],
    leveling_transformation: np.ndarray,
    metrics: Sequence[RegistrationMetric],
    section_config: SectionConfig,
    backend: str,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
) -> list[dict[str, object]]:
    """Apply small in-plane pose corrections from reliable overlapping sections."""
    if len(cleaned_clouds) != len(transformations) or len(cleaned_clouds) != len(metrics):
        raise ValueError("截面姿态修正的点云、变换和指标数量必须一致。")
    active_indices = [index for index, metric in enumerate(metrics) if metric.included_in_fusion]
    if len(active_indices) < 3:
        return []

    section_profiles: dict[int, np.ndarray] = {}
    for index in active_indices:
        aligned_cloud = _transform_cloud(
            cleaned_clouds[index],
            np.asarray(leveling_transformation) @ np.asarray(transformations[index]),
            backend,
        )
        section_cloud = extract_vertical_section(aligned_cloud, section_config, backend, clean=False)
        section_profiles[index] = _section_profile_coordinates(section_cloud, section_config)

    reference_index = next(
        (
            index
            for index in active_indices
            if metrics[index].pose_prior_label == "reference"
        ),
        active_indices[0],
    )
    secondary_index = max(
        (index for index in active_indices if index != reference_index),
        key=lambda index: (
            float(metrics[index].full_cloud_fitness)
            if metrics[index].full_cloud_fitness is not None
            else -float("inf"),
            -index,
        ),
    )
    anchor_indices = [reference_index, secondary_index]
    target_profiles = np.concatenate(
        [section_profiles[index] for index in anchor_indices if len(section_profiles[index])],
        axis=0,
    )
    if len(target_profiles) < 30:
        return []

    normal, horizontal, vertical = vertical_plane_basis(
        section_config.angle_degrees,
        section_config.tilt_degrees,
    )
    basis = np.column_stack((horizontal, vertical, normal))
    leveling = np.asarray(leveling_transformation, dtype=np.float64)
    inverse_leveling = np.linalg.inv(leveling)
    diagnostics: list[dict[str, object]] = []
    for index in active_indices:
        if index in anchor_indices or len(section_profiles[index]) < 30:
            continue
        correction = _estimate_section_pose_correction(
            section_profiles[index],
            target_profiles,
            maximum_correspondence_distance=0.25,
            maximum_rotation_degrees=1.25,
            maximum_translation=0.20,
        )
        if correction is None:
            continue
        rotation_2d, translation_2d, before_score, after_score = correction
        rotation_3d_local = np.eye(3, dtype=np.float64)
        rotation_3d_local[:2, :2] = rotation_2d
        correction_3d = np.eye(4, dtype=np.float64)
        correction_3d[:3, :3] = basis @ rotation_3d_local @ basis.T
        correction_3d[:3, 3] = horizontal * translation_2d[0] + vertical * translation_2d[1]
        transformations[index] = inverse_leveling @ correction_3d @ leveling @ np.asarray(transformations[index])
        angle_degrees = float(np.degrees(np.arctan2(rotation_2d[1, 0], rotation_2d[0, 0])))
        translation_norm = float(np.linalg.norm(translation_2d))
        diagnostics.append(
            {
                "index": index,
                "pose_prior_label": metrics[index].pose_prior_label,
                "angle_degrees": angle_degrees,
                "translation": translation_2d.round(6).tolist(),
                "translation_norm": translation_norm,
                "score_before": before_score,
                "score_after": after_score,
                "anchor_indices": anchor_indices,
            }
        )
        _emit(
            progress_callback,
            f"Group {group_name}: section consensus refined {metrics[index].file_name} "
            f"({angle_degrees:+.3f} deg, {translation_norm:.4f})",
        )
    return diagnostics


def _reduce_section_profile(
    profile: np.ndarray,
    cell_size: float = 0.05,
) -> np.ndarray:
    coordinates = np.asarray(profile, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if not len(coordinates):
        return np.empty((0, 2), dtype=np.float64)
    if cell_size <= 0 or len(coordinates) <= 2:
        return coordinates
    grid = np.floor(coordinates / cell_size).astype(np.int64)
    _, indices = np.unique(grid, axis=0, return_index=True)
    return coordinates[np.sort(indices)]


def _score_section_pose_candidate(
    source_profile: np.ndarray,
    target_profile: np.ndarray,
    overlap_distance: float,
    minimum_overlap_points: int,
    maximum_local_distance: float,
) -> dict[str, object]:
    source = _reduce_section_profile(source_profile)
    target = _reduce_section_profile(target_profile)
    result: dict[str, object] = {
        "profile_points": int(len(source)),
        "overlap_points": 0,
        "overlap_ratio": 0.0,
        "median_distance": None,
        "local_p75_distance": None,
        "p90_distance": None,
        "score": float("inf"),
        "compatible": False,
    }
    if not len(source) or not len(target) or overlap_distance <= 0:
        return result

    distances = cKDTree(target).query(source)[0]
    overlap_mask = np.isfinite(distances) & (distances <= overlap_distance)
    overlap = distances[overlap_mask]
    if not len(overlap):
        return result

    overlap_ratio = float(len(overlap) / len(source))
    median_distance = float(np.median(overlap))
    local_p75_distance = float(np.quantile(overlap, 0.75))
    p90_distance = float(np.quantile(overlap, 0.90))
    ratio_penalty = max(0.0, SECTION_ALIGNMENT_MINIMUM_OVERLAP_RATIO - overlap_ratio)
    score = (
        local_p75_distance
        + 0.20 * median_distance
        + 0.10 * p90_distance
        + ratio_penalty * overlap_distance * 4.0
    )
    result.update(
        {
            "overlap_points": int(len(overlap)),
            "overlap_ratio": overlap_ratio,
            "median_distance": median_distance,
            "local_p75_distance": local_p75_distance,
            "p90_distance": p90_distance,
            "score": float(score),
            "compatible": bool(
                len(overlap) >= minimum_overlap_points
                and overlap_ratio >= SECTION_ALIGNMENT_MINIMUM_OVERLAP_RATIO
                and local_p75_distance <= maximum_local_distance
            ),
        }
    )
    return result


def select_section_consistent_pose_candidates(
    cleaned_clouds: Sequence[o3d.geometry.PointCloud],
    transformations: Sequence[np.ndarray],
    candidate_transformations: Mapping[str, Sequence[np.ndarray]],
    labels: Sequence[str],
    metrics: Sequence[RegistrationMetric],
    leveling_transformation: np.ndarray,
    section_config: SectionConfig,
    registration_config: RegistrationConfig,
    backend: str,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Choose the pose candidate whose common section is most self-consistent."""
    if not (
        len(cleaned_clouds)
        == len(transformations)
        == len(candidate_transformations.get("section_refined", ()))
        == len(labels)
        == len(metrics)
    ):
        raise ValueError("Section pose candidates, clouds, labels, and metrics must have matching sizes.")
    if len(cleaned_clouds) < 3:
        return [np.asarray(transformation).copy() for transformation in transformations], []

    active_indices = [index for index, metric in enumerate(metrics) if metric.included_in_fusion]
    if len(active_indices) < 3:
        return [np.asarray(transformation).copy() for transformation in transformations], []

    candidate_names = list(candidate_transformations)
    if "section_refined" not in candidate_names:
        raise ValueError("Section pose candidates must include section_refined.")
    for name, values in candidate_transformations.items():
        if len(values) != len(cleaned_clouds):
            raise ValueError(f"Section pose candidate set {name} has an invalid length.")

    preferred_anchor_labels = {"reference", "minus10", "minus28", "minus50"}
    anchor_indices = [
        index
        for index in active_indices
        if labels[index] in preferred_anchor_labels
    ]
    if len(anchor_indices) < 2:
        reference_index = next(
            (index for index in active_indices if labels[index] == "reference"),
            active_indices[0],
        )
        secondary_index = max(
            (index for index in active_indices if index != reference_index),
            key=lambda index: (
                float(metrics[index].full_cloud_fitness)
                if metrics[index].full_cloud_fitness is not None
                else -float("inf"),
                -index,
            ),
        )
        anchor_indices = [reference_index, secondary_index]
    candidate_indices = [index for index in active_indices if index not in anchor_indices]
    if not candidate_indices:
        return [np.asarray(transformation).copy() for transformation in transformations], []

    thin_section_config = copy.deepcopy(section_config)
    thin_section_config.thickness = min(
        section_config.thickness,
        registration_config.section_alignment_thickness,
    )
    thin_section_config.use_statistical_filter = False
    thin_section_config.use_component_filter = False
    level = np.asarray(leveling_transformation, dtype=np.float64)

    profile_cache: dict[tuple[str, int], np.ndarray] = {}

    def profile_for(candidate_name: str, index: int) -> np.ndarray:
        cache_key = (candidate_name, index)
        if cache_key in profile_cache:
            return profile_cache[cache_key]
        transformation = np.asarray(candidate_transformations[candidate_name][index])
        aligned_cloud = _transform_cloud(
            cleaned_clouds[index],
            level @ transformation,
            backend,
        )
        section_cloud = extract_vertical_section(
            aligned_cloud,
            thin_section_config,
            backend,
            clean=False,
        )
        profile = _section_profile_coordinates(section_cloud, section_config)
        profile_cache[cache_key] = profile
        return profile

    anchor_candidate_name = (
        "station_pose_prior"
        if "station_pose_prior" in candidate_transformations
        else "section_refined"
    )
    target_parts = [profile_for(anchor_candidate_name, index) for index in anchor_indices]
    target_parts = [part for part in target_parts if len(part)]
    if not target_parts:
        return [np.asarray(transformation).copy() for transformation in transformations], []
    target_profile = _reduce_section_profile(np.concatenate(target_parts, axis=0))

    candidate_priority = (
        "section_refined",
        "pose_graph",
        "registered_calibration",
        "station_pose_prior",
    )

    selected = [np.asarray(transformation).copy() for transformation in transformations]
    diagnostics: list[dict[str, object]] = []
    for index in candidate_indices:
        candidate_profiles = {
            name: _reduce_section_profile(profile_for(name, index))
            for name in candidate_names
        }
        candidate_metrics = {
            name: _score_section_pose_candidate(
                candidate_profiles[name],
                target_profile,
                registration_config.section_alignment_overlap_distance,
                max(
                    SECTION_CANDIDATE_MINIMUM_OVERLAP_POINTS,
                    min(
                        registration_config.section_alignment_minimum_overlap_points,
                        int(
                            np.ceil(
                                len(candidate_profiles[name])
                                * SECTION_ALIGNMENT_MINIMUM_OVERLAP_RATIO
                            )
                        ),
                    ),
                ),
                registration_config.section_alignment_max_p90_distance,
            )
            for name in candidate_names
        }
        compatible_names = [
            name for name in candidate_names if candidate_metrics[name]["compatible"]
        ]
        selection_reason = "compatible_candidate"
        if compatible_names:
            selection_pool = compatible_names
        elif (
            "station_pose_prior" in candidate_metrics
            and len(candidate_profiles["station_pose_prior"])
            < SECTION_CANDIDATE_MINIMUM_PROFILE_POINTS
        ):
            selection_pool = ["station_pose_prior"]
            selection_reason = "station_pose_prior_low_section_evidence"
        else:
            selection_pool = candidate_names
            selection_reason = "best_available_candidate"
        selected_name = min(
            selection_pool,
            key=lambda name: (
                float(candidate_metrics[name]["score"]),
                candidate_priority.index(name)
                if name in candidate_priority
                else len(candidate_priority),
            ),
        )
        selected[index] = np.asarray(candidate_transformations[selected_name][index]).copy()
        if selected_name != "section_refined":
            _emit(
                progress_callback,
                f"Group {group_name}: selected {selected_name} for {labels[index]} "
                f"using common-section consistency",
            )
        diagnostics.append(
            {
                "index": index,
                "pose_prior_label": labels[index],
                "selected_candidate": selected_name,
                "anchor_candidate": anchor_candidate_name,
                "anchor_indices": anchor_indices,
                "anchor_labels": [labels[anchor_index] for anchor_index in anchor_indices],
                "selection_reason": selection_reason,
                "candidate_minimum_profile_points": SECTION_CANDIDATE_MINIMUM_PROFILE_POINTS,
                "candidates": candidate_metrics,
            }
        )
    return selected, diagnostics


def extract_full_resolution_section(
    files: Sequence[Path],
    transformations: Sequence[np.ndarray],
    included_views: Sequence[bool],
    leveling_transformation: np.ndarray,
    config: SectionConfig,
    backend: str,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
    *,
    coverage_preserving: bool = True,
) -> tuple[o3d.geometry.PointCloud, int]:
    """从标定块质量合格的原始 PLY 点中提取截面。

    Coverage-first mode keeps every extracted section part.  The legacy local
    consensus mask remains available only when ``coverage_preserving=False``.
    """
    if len(files) != len(transformations) or len(files) != len(included_views):
        raise ValueError("截面提取的文件、位姿和视角状态数量必须一致。")

    section_parts: list[o3d.geometry.PointCloud] = []
    source_points = 0
    included_count = sum(bool(included) for included in included_views)
    current_index = 0
    for path, transformation, included in zip(files, transformations, included_views):
        if not included:
            continue
        current_index += 1
        group_prefix = f"第 {group_name} 组：" if group_name else ""
        _emit(
            progress_callback,
            f"{group_prefix}原始全分辨率截面 {current_index}/{included_count} - {path.name}",
        )
        raw_cloud = read_point_cloud(path)
        source_points += len(raw_cloud.points)
        final_transformation = np.asarray(leveling_transformation) @ np.asarray(transformation)
        aligned_cloud = _transform_cloud(raw_cloud, final_transformation, backend)
        section_part = extract_vertical_section(aligned_cloud, config, backend, clean=False)
        if len(section_part.points):
            section_parts.append(section_part)

    if not coverage_preserving:
        section_parts = _filter_section_parts_by_local_consensus(section_parts, config)
    return _concatenate_clouds(section_parts), source_points


def extract_section_from_merged_cloud(
    merged_cloud: o3d.geometry.PointCloud,
    config: SectionConfig,
    backend: str,
) -> tuple[o3d.geometry.PointCloud, int]:
    """Extract the section from the exact fused cloud shown to the user."""
    if not len(merged_cloud.points):
        return o3d.geometry.PointCloud(), 0
    section_cloud = extract_vertical_section(merged_cloud, config, backend, clean=False)
    return section_cloud, len(merged_cloud.points)


def _merge_clouds_with_section_consensus(
    files: Sequence[Path],
    transformations: Sequence[np.ndarray],
    included_views: Sequence[bool],
    leveling_transformation: np.ndarray,
    section_config: SectionConfig,
    voxel_size: float,
    backend: str,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
    *,
    conservative: bool = False,
    coverage_preserving: bool = True,
    section_recovery_voxel_size: float | None = None,
    calibration_config: CalibrationConfig | None = None,
    calibration_bounds_in_reference_frame: bool = True,
    source_clouds: Sequence[o3d.geometry.PointCloud] | None = None,
) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    """Fuse supplied cleaned clouds while preserving complementary target coverage.

    The section-consensus calculation remains available as a diagnostic, but
    coverage-first fusion uses an all-true mask for every calibration-valid
    view.  This keeps unique target points from a side or rear view instead of
    treating non-overlap as evidence that the view is wrong.
    """
    if len(files) != len(transformations) or len(files) != len(included_views):
        raise ValueError("Section-consensus fusion inputs must have matching sizes.")
    if voxel_size <= 0:
        raise ValueError("Fusion voxel size must be positive.")
    if source_clouds is not None and len(source_clouds) != len(files):
        raise ValueError("去噪输入点云数量必须与文件、位姿和工位数量一致。")
    if section_recovery_voxel_size is not None and (
        not np.isfinite(section_recovery_voxel_size) or section_recovery_voxel_size <= 0
    ):
        raise ValueError("Section recovery voxel size must be finite and positive.")

    included_indices = [
        index
        for index, included in enumerate(included_views)
        if bool(included)
    ]
    if not included_indices:
        return o3d.geometry.PointCloud(), {
            "enabled": True,
            "source_views": 0,
            "section_points_before": 0,
            "section_points_after": 0,
            "removed_section_points": 0,
            "fusion_confidence": {
                "status": "insufficient_data",
                "confidence_score": 0.0,
                "section_voxel_count": 0,
                "section_voxel_support_counts": {},
                "unique_coverage_retained": True,
            },
            "views": [],
        }

    leveling = np.asarray(leveling_transformation, dtype=np.float64)
    aligned_clouds: list[o3d.geometry.PointCloud] = []
    section_parts: list[o3d.geometry.PointCloud] = []
    section_density_recovery_views: list[dict[str, object]] = []
    fixture_removed_counts: list[int] = []
    source_points = 0
    calibration_points_removed = 0
    calibration_exclusion_enabled = bool(
        calibration_config is not None
        and calibration_config.exclude_from_fusion
        and calibration_config.use_roi
    )
    calibration_exclusion_margin = (
        float(calibration_config.exclusion_margin)
        if calibration_config is not None
        else None
    )
    calibration_bounds_frame = (
        "reference_before_leveling"
        if calibration_bounds_in_reference_frame
        else "source_input"
    )
    for position, index in enumerate(included_indices, start=1):
        path = files[index]
        _emit(
            progress_callback,
            f"Group {group_name}: section-consensus fusion {position}/{len(included_indices)} - {path.name}",
        )
        # Processing builds one cleaned cloud per view before registration.
        # Reuse it here so the final fusion cannot re-introduce raw outliers.
        source_cloud = source_clouds[index] if source_clouds is not None else read_point_cloud(path)
        source_points += len(source_cloud.points)
        final_transformation = leveling @ np.asarray(transformations[index], dtype=np.float64)
        fixture_removed = 0
        if calibration_exclusion_enabled:
            # Filter while retaining exact source coordinates for the denoise
            # voxel gate. The inverse maps source points into the ROI frame.
            source_roi_transformation = (
                np.linalg.inv(np.asarray(transformations[index], dtype=np.float64))
                if calibration_bounds_in_reference_frame
                else None
            )
            before_fixture_exclusion = len(source_cloud.points)
            source_cloud = remove_calibration_region(
                source_cloud,
                calibration_config,
                backend,
                coordinate_transformation=source_roi_transformation,
                component_voxel_size=max(voxel_size * 4, 0.24),
            )
            fixture_removed = before_fixture_exclusion - len(source_cloud.points)
            calibration_points_removed += fixture_removed
        fixture_removed_counts.append(fixture_removed)
        aligned_cloud = _transform_cloud(source_cloud, final_transformation, backend)
        aligned_clouds.append(aligned_cloud)
        current_section_part = extract_vertical_section(
            aligned_cloud,
            section_config,
            backend,
            clean=False,
        )
        recovered_section = o3d.geometry.PointCloud()
        recovery_report: dict[str, object] = {
            "enabled": False,
            "method": "raw_section_samples_inside_retained_denoise_voxels",
            "status": "source_clouds_not_denoised",
            "file_name": path.name,
            "recovered_points": 0,
        }
        if source_clouds is not None and section_recovery_voxel_size is not None:
            recovered_section, recovery_report = _recover_section_density_from_raw(
                path,
                source_cloud,
                np.asarray(transformations[index], dtype=np.float64),
                leveling,
                section_config,
                denoise_voxel_size=float(section_recovery_voxel_size),
                backend=backend,
                calibration_config=calibration_config,
                calibration_bounds_in_reference_frame=calibration_bounds_in_reference_frame,
            )
        elif source_clouds is not None:
            recovery_report["status"] = "recovery_voxel_size_not_supplied"
        section_density_recovery_views.append(recovery_report)
        recovery_report["input_index"] = index
        section_parts.append(
            _concatenate_clouds([current_section_part, recovered_section])
        )

    diagnostic_keep_masks = _local_consensus_keep_masks(
        section_parts,
        section_config,
        conservative=conservative,
    )
    keep_masks = (
        [np.ones(len(section_part.points), dtype=bool) for section_part in section_parts]
        if coverage_preserving
        else diagnostic_keep_masks
    )
    non_section_clouds: list[o3d.geometry.PointCloud] = []
    non_section_source_indices: list[int] = []
    retained_section_clouds: list[o3d.geometry.PointCloud] = []
    retained_section_source_indices: list[int] = []
    view_reports: list[dict[str, object]] = []
    section_points_before = 0
    section_points_after = 0
    for position, (index, aligned_cloud, section_part, keep_mask, diagnostic_keep_mask) in enumerate(
        zip(
            included_indices,
            aligned_clouds,
            section_parts,
            keep_masks,
            diagnostic_keep_masks,
            strict=True,
        ),
        start=1,
    ):
        section_mask = _vertical_section_mask(
            np.asarray(aligned_cloud.points),
            section_config,
            backend,
        )
        retained_section = _select_cloud_by_mask(section_part, keep_mask)
        non_section = _select_cloud_by_mask(aligned_cloud, ~section_mask)
        if len(non_section.points):
            non_section_clouds.append(non_section)
            non_section_source_indices.append(index)
        if len(retained_section.points):
            retained_section_clouds.append(retained_section)
            retained_section_source_indices.append(index)

        recovery_report = section_density_recovery_views[position - 1]
        recovered_count = int(recovery_report.get("recovered_points", 0) or 0)
        current_section_count = max(0, len(section_part.points) - recovered_count)
        before_count = len(section_part.points)
        after_count = len(retained_section.points)
        diagnostic_after_count = int(np.count_nonzero(diagnostic_keep_mask))
        section_points_before += before_count
        section_points_after += after_count
        view_reports.append(
            {
                "input_index": index,
                "file_name": files[index].name,
                "section_points_before": before_count,
                "section_points_before_recovery": current_section_count,
                "section_points_recovered": recovered_count,
                "section_points_after": after_count,
                "removed_section_points": before_count - after_count,
                "removed_section_ratio": float(
                    (before_count - after_count) / before_count
                )
                if before_count
                else 0.0,
                "diagnostic_section_points_after": diagnostic_after_count,
                "diagnostic_removed_section_points": before_count - diagnostic_after_count,
                "diagnostic_removed_section_ratio": float(
                    (before_count - diagnostic_after_count) / before_count
                )
                if before_count
                else 0.0,
                "non_section_points_before_voxel": len(non_section.points),
                "retained_section_points_before_voxel": after_count,
                "calibration_points_removed_before_fusion": fixture_removed_counts[position - 1],
            }
        )

    if non_section_clouds:
        non_section_result = _representative_voxel_fusion(
            non_section_clouds,
            [np.eye(4, dtype=np.float64) for _ in non_section_clouds],
            voxel_size,
            backend,
            source_indices=non_section_source_indices,
            clouds_are_aligned=True,
            return_source_indices=True,
        )
        (
            non_section_merged,
            whole_cloud_fusion,
            non_section_sources,
            non_section_support,
        ) = non_section_result
    else:
        non_section_merged = o3d.geometry.PointCloud()
        non_section_sources = np.empty(0, dtype=np.int64)
        non_section_support = np.empty(0, dtype=np.int64)
        whole_cloud_fusion = {
            "status": "insufficient_data",
            "method": "representative_sample_per_voxel",
            "voxel_size": float(voxel_size),
            "input_points_before_voxel": 0,
            "output_points_after_voxel": 0,
            "points_removed_by_voxel": 0,
            "voxel_count": 0,
            "multi_view_voxel_count": 0,
            "unique_view_voxel_count": 0,
            "multi_view_voxel_ratio": 0.0,
            "unique_view_voxel_ratio": 0.0,
            "source_input_counts": {},
            "source_retained_voxel_counts": {},
            "representative_residual_median": None,
            "representative_residual_p95": None,
            "unique_coverage_retained": True,
            "coordinates_averaged": False,
        }
    whole_cloud_fusion.update(
        {
            "calibration_exclusion_applied": calibration_exclusion_enabled,
            "calibration_exclusion_margin": calibration_exclusion_margin,
            "calibration_bounds_frame": calibration_bounds_frame,
            "calibration_points_removed_before_fusion": int(calibration_points_removed),
        }
    )
    section_fusion_voxel_size = min(max(voxel_size * 0.25, 0.0075), 0.01)
    section_result = _support_weighted_section_fusion(
        retained_section_clouds,
        retained_section_source_indices,
        section_fusion_voxel_size,
        section_config,
        return_source_indices=True,
    )
    (
        section_merged,
        fusion_confidence,
        section_sources,
        section_support,
    ) = section_result
    merged_cloud = _concatenate_clouds([non_section_merged, section_merged])
    merged_sources = np.concatenate([non_section_sources, section_sources])
    merged_support = np.concatenate([non_section_support, section_support])
    merged_cloud, final_component_cleanup = _final_fusion_component_cleanup(
        merged_cloud,
        merged_sources,
        voxel_size,
        source_support_counts=merged_support,
    )
    section_offset_diagnostic = evaluate_section_offset_candidates(
        np.asarray(merged_cloud.points),
        section_config,
        source_clouds=aligned_clouds,
        source_indices=included_indices,
    )
    fusion_points_before_voxel = sum(
        len(cloud.points) for cloud in non_section_clouds
    ) + sum(len(cloud.points) for cloud in retained_section_clouds)
    consensus_report = {
        "enabled": True,
        "method": (
            "conservative_local_section_consensus_with_dedicated_section_fusion"
            if conservative
            else "local_section_consensus_with_dedicated_section_fusion"
        ),
        "mode": "conservative" if conservative else "strict",
        "coverage_preserving": bool(coverage_preserving),
        "filter_applied": not bool(coverage_preserving),
        "diagnostic_only": bool(coverage_preserving),
        "pose_source": "caller_supplied_rigid_poses_plus_reference_leveling",
        "calibration_exclusion_applied": calibration_exclusion_enabled,
        "calibration_exclusion_margin": calibration_exclusion_margin,
        "calibration_bounds_frame": calibration_bounds_frame,
        "calibration_points_removed_before_fusion": int(calibration_points_removed),
        "fusion_input": "per_view_denoised_clouds" if source_clouds is not None else "raw_ply_clouds",
        "fusion_input_denoised": source_clouds is not None,
        "section_density_recovery": {
            "enabled": source_clouds is not None and section_recovery_voxel_size is not None,
            "method": "raw_section_samples_inside_retained_denoise_voxels",
            "source": "original_ply",
            "interpolation_applied": False,
            "cad_feedback_applied": False,
            "views": section_density_recovery_views,
            "raw_section_candidates": int(sum(
                int(item.get("raw_section_candidates", 0) or 0)
                for item in section_density_recovery_views
            )),
            "voxel_matched_candidates": int(sum(
                int(item.get("voxel_matched_candidates", 0) or 0)
                for item in section_density_recovery_views
            )),
            "near_retained_candidates": int(sum(
                int(item.get("near_retained_candidates", 0) or 0)
                for item in section_density_recovery_views
            )),
            "recovered_points": int(sum(
                int(item.get("recovered_points", 0) or 0)
                for item in section_density_recovery_views
            )),
        },
        "denoise_stage": (
            "single_view_before_registration_plus_local_raw_section_recovery_plus_final_fusion_component_cleanup"
            if source_clouds is not None and section_recovery_voxel_size is not None
            else "single_view_before_registration_plus_final_fusion_component_cleanup"
        ),
        "post_fusion_denoise_applied": bool(
            final_component_cleanup.get("removed_points", 0)
        ),
        "post_fusion_denoise_method": final_component_cleanup.get("method"),
        "post_fusion_denoise_stage": final_component_cleanup.get("stage"),
        "coordinates_adjusted_by_cad": False,
        "total_views": len(files),
        "source_views": len(included_indices),
        "included_indices": included_indices,
        "excluded_indices": [
            index for index, included in enumerate(included_views) if not included
        ],
        "source_points_before_voxel": source_points,
        "section_points_before": section_points_before,
        "section_points_after": section_points_after,
        "removed_section_points": section_points_before - section_points_after,
        "removed_section_ratio": float(
            (section_points_before - section_points_after) / section_points_before
        )
        if section_points_before
        else 0.0,
        "fusion_points_before_voxel": fusion_points_before_voxel,
        "whole_cloud_fusion": whole_cloud_fusion,
        "final_component_cleanup": final_component_cleanup,
        "whole_cloud_points_before_voxel": int(
            whole_cloud_fusion.get("input_points_before_voxel", 0)
        ),
        "whole_cloud_points_after_voxel": int(
            whole_cloud_fusion.get("output_points_after_voxel", 0)
        ),
        "non_section_points_before_voxel": sum(
            len(cloud.points) for cloud in non_section_clouds
        ),
        "non_section_points_after_voxel": len(non_section_merged.points),
        "section_fusion_voxel_size": section_fusion_voxel_size,
        "section_points_after_voxel": len(section_merged.points),
        "fusion_points_after_voxel": len(merged_cloud.points),
        "section_offset_diagnostic": section_offset_diagnostic,
        "profile_bin_size": 0.05,
        "profile_cluster_gap": 0.06,
        "profile_cluster_margin": 0.015,
        **fusion_confidence,
        "views": view_reports,
        "diagnostic_removed_section_points": int(
            sum(item["diagnostic_removed_section_points"] for item in view_reports)
        ),
        "diagnostic_removed_section_ratio": float(
            sum(item["diagnostic_removed_section_points"] for item in view_reports)
            / section_points_before
        )
        if section_points_before
        else 0.0,
    }
    _emit(
        progress_callback,
        (
            f"Group {group_name}: coverage-first fusion retained all "
            f"{section_points_before:,} section points before voxel fusion; "
            f"consensus diagnostic would remove "
            f"{consensus_report['diagnostic_removed_section_points']:,}; "
        )
        if coverage_preserving
        else (
            f"Group {group_name}: section consensus removed "
            f"{section_points_before - section_points_after:,} unsupported section points; "
        )
        + f"section fusion voxel {section_fusion_voxel_size:.4f}",
    )
    return merged_cloud, consensus_report


def _annotate_section_consensus_report(
    report: dict[str, object],
    metrics: Sequence[RegistrationMetric],
) -> None:
    """Attach station-level exclusion reasons to the fusion report."""
    report["total_views"] = len(metrics)
    report["included_indices"] = [
        index for index, metric in enumerate(metrics) if metric.included_in_fusion
    ]
    report["excluded_indices"] = [
        index for index, metric in enumerate(metrics) if not metric.included_in_fusion
    ]
    report["excluded_views"] = [
        {
            "input_index": index,
            "file_name": metric.file_name,
            "pose_prior_label": metric.pose_prior_label,
            "reason": metric.exclusion_reason,
        }
        for index, metric in enumerate(metrics)
        if not metric.included_in_fusion
    ]


def evaluate_section_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
    overlap_distance: float,
) -> SectionAlignmentQuality:
    """Measure a raw thin section against the overlapping contours from other views."""
    if overlap_distance <= 0:
        raise ValueError("Section-alignment overlap distance must be positive.")

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    source = source.reshape((-1, 3)) if source.size else np.empty((0, 3), dtype=np.float64)
    target = target.reshape((-1, 3)) if target.size else np.empty((0, 3), dtype=np.float64)
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if not len(source) or not len(target):
        return SectionAlignmentQuality(len(source), 0, None, None, 0.0, None)

    target_cloud = _cloud_from_arrays(target)
    tree = o3d.geometry.KDTreeFlann(target_cloud)
    distances = np.empty(len(source), dtype=np.float64)
    for index, point in enumerate(source):
        count, _, squared_distances = tree.search_knn_vector_3d(point, 1)
        distances[index] = math.sqrt(squared_distances[0]) if count else math.inf

    overlap_distances = distances[distances <= overlap_distance]
    if not len(overlap_distances):
        return SectionAlignmentQuality(len(source), 0, None, None, 0.0, None)
    return SectionAlignmentQuality(
        section_points=len(source),
        overlap_points=len(overlap_distances),
        median_distance=float(np.median(overlap_distances)),
        p90_distance=float(np.quantile(overlap_distances, 0.90)),
        overlap_ratio=float(len(overlap_distances) / len(source)),
        local_p75_distance=float(np.quantile(overlap_distances, 0.75)),
    )


def _profile_vertical_clusters(
    profile: np.ndarray,
    bin_size: float,
    cluster_gap: float,
    minimum_points: int = 3,
) -> dict[int, np.ndarray]:
    coordinates = np.asarray(profile, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        return {}
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if not len(coordinates) or bin_size <= 0 or cluster_gap <= 0:
        return {}
    bin_indices = np.floor(coordinates[:, 0] / bin_size).astype(np.int64)
    clusters: dict[int, np.ndarray] = {}
    for bin_index in np.unique(bin_indices):
        values = np.sort(coordinates[bin_indices == bin_index, 1])
        if len(values) < minimum_points:
            continue
        split_indices = np.flatnonzero(np.diff(values) > cluster_gap) + 1
        groups = np.split(values, split_indices)
        centers = [float(np.median(group)) for group in groups if len(group) >= minimum_points]
        if centers:
            clusters[int(bin_index)] = np.asarray(centers, dtype=np.float64)
    return clusters


def _profile_contour_mismatch(
    source_profile: np.ndarray,
    target_profile: np.ndarray,
    bin_size: float,
    mismatch_distance: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "compared_bins": 0,
        "duplicate_bins": 0,
        "missing_bins": 0,
        "duplicate_contour_ratio": 0.0,
        "missing_contour_ratio": 0.0,
        "source_cluster_count": 0,
        "target_cluster_count": 0,
    }
    if bin_size <= 0 or mismatch_distance <= 0:
        return result
    cluster_gap = min(
        SECTION_ALIGNMENT_CONTOUR_CLUSTER_GAP_MAXIMUM,
        max(
            SECTION_ALIGNMENT_CONTOUR_CLUSTER_GAP_MINIMUM,
            mismatch_distance * 0.80,
        ),
    )
    source_clusters = _profile_vertical_clusters(
        source_profile,
        bin_size,
        cluster_gap,
    )
    target_clusters = _profile_vertical_clusters(
        target_profile,
        bin_size,
        cluster_gap,
    )
    common_bins = sorted(set(source_clusters) & set(target_clusters))
    if not common_bins:
        return result
    duplicate_count = 0
    missing_count = 0
    source_count = 0
    target_count = 0
    for bin_index in common_bins:
        source_centers = source_clusters[bin_index]
        target_centers = target_clusters[bin_index]
        source_count += len(source_centers)
        target_count += len(target_centers)
        source_distances = np.abs(source_centers[:, None] - target_centers[None, :])
        target_distances = source_distances.T
        source_unmatched = np.count_nonzero(np.min(source_distances, axis=1) > cluster_gap)
        target_unmatched = np.count_nonzero(np.min(target_distances, axis=1) > cluster_gap)
        duplicate_count += int(source_unmatched)
        missing_count += int(target_unmatched)
    result.update(
        {
            "compared_bins": len(common_bins),
            "duplicate_bins": int(
                sum(
                    1
                    for bin_index in common_bins
                    if len(source_clusters[bin_index]) > len(target_clusters[bin_index])
                )
            ),
            "missing_bins": int(
                sum(
                    1
                    for bin_index in common_bins
                    if len(source_clusters[bin_index]) < len(target_clusters[bin_index])
                )
            ),
            "duplicate_contour_ratio": float(duplicate_count / source_count)
            if source_count
            else 0.0,
            "missing_contour_ratio": float(missing_count / target_count)
            if target_count
            else 0.0,
            "source_cluster_count": int(source_count),
            "target_cluster_count": int(target_count),
        }
    )
    return result


def _section_alignment_is_compatible(
    quality: SectionAlignmentQuality,
    registration_config: RegistrationConfig,
) -> bool:
    if quality.overlap_points < registration_config.section_alignment_minimum_overlap_points:
        return False
    if quality.overlap_ratio is None or quality.overlap_ratio < SECTION_ALIGNMENT_MINIMUM_OVERLAP_RATIO:
        return False
    if quality.p90_distance is not None and quality.p90_distance <= registration_config.section_alignment_max_p90_distance:
        return True
    return (
        quality.local_p75_distance is not None
        and quality.local_p75_distance <= registration_config.section_alignment_max_p90_distance
    )


def _profile_alignment_metrics(
    source_profile: np.ndarray,
    target_profile: np.ndarray,
    overlap_distance: float,
    mismatch_distance: float | None = None,
) -> dict[str, object]:
    """Compare one view against another only where their profiles overlap horizontally.

    A scan can legitimately contain only one end or one side of the target.  Its
    missing area must not be treated as a registration error.  This directional
    metric therefore asks a narrower question: within the horizontal area both
    views actually cover, does the *source* profile sit on the target profile?
    """
    source = np.asarray(source_profile, dtype=np.float64)
    target = np.asarray(target_profile, dtype=np.float64)
    effective_mismatch_distance = (
        float(mismatch_distance)
        if mismatch_distance is not None
        else float(overlap_distance)
    )
    if source.ndim != 2 or source.shape[1] != 2:
        source = np.empty((0, 2), dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 2:
        target = np.empty((0, 2), dtype=np.float64)
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    result: dict[str, object] = {
        "profile_points": int(len(source)),
        "common_profile_points": 0,
        "common_profile_ratio": 0.0,
        "compared_points": 0,
        "overlap_points": 0,
        "overlap_ratio": 0.0,
        "median_distance": None,
        "local_p75_distance": None,
        "p90_distance": None,
        "all_median_distance": None,
        "all_p75_distance": None,
        "all_p90_distance": None,
        "unmatched_ratio": 1.0,
        "common_horizontal_start": None,
        "common_horizontal_end": None,
        "common_horizontal_span": 0.0,
        "profile_bin_count": 0,
        "bad_bin_count": 0,
        "bad_bin_ratio": None,
        "largest_bad_bin_run": 0,
        "contour_compared_bins": 0,
        "duplicate_bins": 0,
        "missing_bins": 0,
        "duplicate_contour_ratio": 0.0,
        "missing_contour_ratio": 0.0,
        "source_cluster_count": 0,
        "target_cluster_count": 0,
    }
    if not len(source) or not len(target) or overlap_distance <= 0:
        return result

    common_start = max(float(source[:, 0].min()), float(target[:, 0].min()))
    common_end = min(float(source[:, 0].max()), float(target[:, 0].max()))
    common_span = max(0.0, common_end - common_start)
    result.update(
        {
            "common_horizontal_start": common_start,
            "common_horizontal_end": common_end,
            "common_horizontal_span": common_span,
        }
    )
    if common_span <= 1e-8:
        return result

    horizontal_margin = min(max(overlap_distance * 0.50, 0.02), 0.25)
    source_in_common = source[
        (source[:, 0] >= common_start - horizontal_margin)
        & (source[:, 0] <= common_end + horizontal_margin)
    ]
    target_in_common = target[
        (target[:, 0] >= common_start - horizontal_margin)
        & (target[:, 0] <= common_end + horizontal_margin)
    ]
    result["common_profile_points"] = int(len(source_in_common))
    result["common_profile_ratio"] = float(len(source_in_common) / len(source))
    if not len(source_in_common) or not len(target_in_common):
        return result

    bin_size = min(max(overlap_distance * 0.35, 0.05), 0.15)
    contour_mismatch = _profile_contour_mismatch(
        source_in_common,
        target_in_common,
        bin_size,
        effective_mismatch_distance,
    )
    result.update(
        {
            "contour_compared_bins": int(contour_mismatch["compared_bins"]),
            "duplicate_bins": int(contour_mismatch["duplicate_bins"]),
            "missing_bins": int(contour_mismatch["missing_bins"]),
            "duplicate_contour_ratio": float(contour_mismatch["duplicate_contour_ratio"]),
            "missing_contour_ratio": float(contour_mismatch["missing_contour_ratio"]),
            "source_cluster_count": int(contour_mismatch["source_cluster_count"]),
            "target_cluster_count": int(contour_mismatch["target_cluster_count"]),
        }
    )

    target_tree = cKDTree(target_in_common)
    neighbor_count = min(SECTION_ALIGNMENT_PROFILE_NEIGHBOR_COUNT, len(target_in_common))
    distances, neighbor_indices = target_tree.query(source_in_common, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, np.newaxis]
        neighbor_indices = neighbor_indices[:, np.newaxis]
    horizontal_distances = np.abs(
        source_in_common[:, np.newaxis, 0] - target_in_common[neighbor_indices, 0]
    )
    valid_neighbors = horizontal_distances <= overlap_distance
    nearest_distances = np.full(len(source_in_common), np.inf, dtype=np.float64)
    has_valid_neighbor = valid_neighbors.any(axis=1)
    if np.any(has_valid_neighbor):
        nearest_distances[has_valid_neighbor] = np.min(
            np.where(valid_neighbors[has_valid_neighbor], distances[has_valid_neighbor], np.inf),
            axis=1,
        )

    finite_mask = np.isfinite(nearest_distances)
    finite_distances = nearest_distances[finite_mask]
    result["compared_points"] = int(len(finite_distances))
    if not len(finite_distances):
        return result

    overlap_mask = finite_mask & (nearest_distances <= overlap_distance)
    overlap = nearest_distances[overlap_mask]
    if not len(overlap):
        result.update(
            {
                "all_median_distance": float(np.median(finite_distances)),
                "all_p75_distance": float(np.quantile(finite_distances, 0.75)),
                "all_p90_distance": float(np.quantile(finite_distances, 0.90)),
            }
        )
        return result

    bin_origin = math.floor((common_start - horizontal_margin) / bin_size) * bin_size
    compared_coordinates = source_in_common[finite_mask, 0]
    compared_bin_indices = np.floor(
        (compared_coordinates - bin_origin) / bin_size
    ).astype(np.int64)
    compared_distances = nearest_distances[finite_mask]
    bin_indices: list[int] = []
    bin_medians: list[float] = []
    minimum_points_per_bin = 1 if len(compared_distances) < 200 else 8
    for bin_index in np.unique(compared_bin_indices):
        bin_distances = compared_distances[compared_bin_indices == bin_index]
        if len(bin_distances) < minimum_points_per_bin:
            continue
        bin_indices.append(int(bin_index))
        bin_medians.append(float(np.median(bin_distances)))

    bad_bin_ratio: float | None = None
    largest_bad_bin_run = 0
    bad_bin_count = 0
    if bin_medians:
        bad_bins = np.asarray(bin_medians, dtype=np.float64) > effective_mismatch_distance
        bad_bin_count = int(np.count_nonzero(bad_bins))
        bad_bin_ratio = float(np.mean(bad_bins))
        current_run = 0
        previous_bin_index: int | None = None
        for bin_index, is_bad in zip(bin_indices, bad_bins, strict=True):
            if is_bad:
                current_run = (
                    current_run + 1
                    if previous_bin_index is not None and bin_index == previous_bin_index + 1
                    else 1
                )
                largest_bad_bin_run = max(largest_bad_bin_run, current_run)
            else:
                current_run = 0
            previous_bin_index = bin_index

    result.update(
        {
            "overlap_points": int(len(overlap)),
            "overlap_ratio": float(len(overlap) / len(finite_distances)),
            "median_distance": float(np.median(overlap)),
            "local_p75_distance": float(np.quantile(overlap, 0.75)),
            "p90_distance": float(np.quantile(overlap, 0.90)),
            "all_median_distance": float(np.median(finite_distances)),
            "all_p75_distance": float(np.quantile(finite_distances, 0.75)),
            "all_p90_distance": float(np.quantile(finite_distances, 0.90)),
            "unmatched_ratio": float(1.0 - len(overlap) / len(finite_distances)),
            "profile_bin_count": len(bin_medians),
            "bad_bin_count": bad_bin_count,
            "bad_bin_ratio": bad_bin_ratio,
            "largest_bad_bin_run": largest_bad_bin_run,
        }
    )
    return result


def _profile_alignment_is_compatible(
    quality: Mapping[str, object],
    minimum_overlap_points: int,
    maximum_local_distance: float,
) -> bool:
    p75_distance = quality.get("local_p75_distance")
    p90_distance = quality.get("p90_distance")
    return bool(
        int(quality.get("overlap_points", 0)) >= minimum_overlap_points
        and float(quality.get("overlap_ratio", 0.0))
        >= SECTION_ALIGNMENT_MINIMUM_LOCAL_MATCH_RATIO
        and p75_distance is not None
        and float(p75_distance) <= maximum_local_distance
        and p90_distance is not None
        and float(p90_distance) <= maximum_local_distance
    )


def _profile_alignment_has_mismatch_evidence(
    quality: Mapping[str, object],
    minimum_evidence_points: int,
    maximum_local_distance: float,
) -> bool:
    """Return whether one directional comparison shows a sustained wrong contour."""
    if int(quality.get("compared_points", 0)) < minimum_evidence_points:
        return False
    contour_mismatch_ratio = max(
        float(quality.get("duplicate_contour_ratio", 0.0)),
        float(quality.get("missing_contour_ratio", 0.0)),
    )
    if (
        int(quality.get("contour_compared_bins", 0))
        >= SECTION_ALIGNMENT_MINIMUM_CONTOUR_MISMATCH_BINS
        and contour_mismatch_ratio >= SECTION_ALIGNMENT_MINIMUM_CONTOUR_MISMATCH_RATIO
        and (
            float(quality.get("p90_distance") or 0.0)
            > maximum_local_distance * 0.75
            or float(quality.get("bad_bin_ratio") or 0.0)
            >= SECTION_ALIGNMENT_MINIMUM_BAD_BIN_RATIO * 0.65
        )
    ):
        return True
    bad_bin_ratio = quality.get("bad_bin_ratio")
    profile_bin_count = int(quality.get("profile_bin_count", 0))
    minimum_bad_bin_run = (
        1 if profile_bin_count <= 20 else SECTION_ALIGNMENT_MINIMUM_BAD_BIN_RUN
    )
    has_sustained_bad_bins = bool(
        bad_bin_ratio is not None
        and float(bad_bin_ratio) >= SECTION_ALIGNMENT_MINIMUM_BAD_BIN_RATIO
        and int(quality.get("largest_bad_bin_run", 0))
        >= minimum_bad_bin_run
    )
    if not has_sustained_bad_bins:
        return False

    p90_distance = quality.get("p90_distance")
    if p90_distance is not None and float(p90_distance) > maximum_local_distance:
        return True
    all_p75_distance = quality.get("all_p75_distance")
    return bool(
        all_p75_distance is not None
        and float(all_p75_distance) > maximum_local_distance
    )


def _profile_alignment_has_extra_contour_evidence(
    quality: Mapping[str, object],
    minimum_evidence_points: int,
) -> bool:
    """Detect a persistent extra contour without treating unique coverage as an error."""
    if int(quality.get("compared_points", 0)) < minimum_evidence_points:
        return False
    if int(quality.get("contour_compared_bins", 0)) < SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_BINS:
        return False
    duplicate_ratio = float(quality.get("duplicate_contour_ratio", 0.0) or 0.0)
    missing_ratio = float(quality.get("missing_contour_ratio", 0.0) or 0.0)
    return bool(
        duplicate_ratio >= SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_RATIO
        and duplicate_ratio - missing_ratio >= SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_EXCESS
        and float(quality.get("common_profile_ratio", 0.0) or 0.0) >= 0.60
        and float(quality.get("overlap_ratio", 0.0) or 0.0) >= 0.70
    )


def _parallel_contour_track_evidence(
    source_profile: np.ndarray,
    target_profile: np.ndarray,
    overlap_distance: float,
    maximum_local_distance: float,
) -> dict[str, object]:
    """Detect multiple long, parallel contour tracks with a stable offset."""
    result: dict[str, object] = {
        "qualified": False,
        "qualified_track_count": 0,
        "common_horizontal_span": 0.0,
        "matched_bin_count": 0,
        "track_reports": [],
    }
    source = np.asarray(source_profile, dtype=np.float64)
    target = np.asarray(target_profile, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        source = np.empty((0, 2), dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 2:
        target = np.empty((0, 2), dtype=np.float64)
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if not len(source) or not len(target) or overlap_distance <= 0:
        return result

    common_start = max(float(source[:, 0].min()), float(target[:, 0].min()))
    common_end = min(float(source[:, 0].max()), float(target[:, 0].max()))
    common_span = max(0.0, common_end - common_start)
    if common_span <= 1e-8:
        return result

    bin_size = min(max(overlap_distance * 0.35, 0.05), 0.15)
    cluster_gap = min(
        SECTION_ALIGNMENT_CONTOUR_CLUSTER_GAP_MAXIMUM,
        max(SECTION_ALIGNMENT_CONTOUR_CLUSTER_GAP_MINIMUM, maximum_local_distance * 0.80),
    )
    bin_origin = math.floor(common_start / bin_size) * bin_size

    def cluster_centers(values: np.ndarray) -> list[float]:
        ordered_values = np.sort(np.asarray(values, dtype=np.float64))
        if len(ordered_values) < 3:
            return []
        split_indices = np.flatnonzero(np.diff(ordered_values) > cluster_gap) + 1
        return [
            float(np.median(cluster))
            for cluster in np.split(ordered_values, split_indices)
            if len(cluster) >= 3
        ]

    records: list[tuple[float, list[float], list[float]]] = []
    bin_count = int(math.floor((common_end - bin_origin) / bin_size)) + 1
    for bin_number in range(bin_count):
        bin_start = bin_origin + bin_number * bin_size
        bin_end = bin_start + bin_size
        source_values = source[
            (source[:, 0] >= bin_start) & (source[:, 0] < bin_end),
            1,
        ]
        target_values = target[
            (target[:, 0] >= bin_start) & (target[:, 0] < bin_end),
            1,
        ]
        source_centers = cluster_centers(source_values)
        target_centers = cluster_centers(target_values)
        if source_centers and target_centers:
            records.append((bin_start, source_centers, target_centers))

    result["common_horizontal_span"] = common_span
    result["matched_bin_count"] = len(records)
    if not records:
        return result

    minimum_run_bins = max(6, int(math.ceil(common_span / bin_size * 0.35)))
    minimum_offset = max(0.12, maximum_local_distance * 1.05)
    maximum_offset = max(overlap_distance * 1.50, minimum_offset)
    residual_tolerance = max(0.04, maximum_local_distance * 0.40)
    maximum_absolute_p90 = max(0.25, maximum_local_distance * 2.0)
    maximum_rank = max(
        max(len(source_centers), len(target_centers))
        for _, source_centers, target_centers in records
    )
    track_reports: list[dict[str, object]] = []
    qualified_track_count = 0

    for rank in range(maximum_rank):
        samples: list[tuple[float, float]] = []
        for bin_start, source_centers, target_centers in records:
            if len(source_centers) == len(target_centers) and rank < len(source_centers):
                samples.append((bin_start, source_centers[rank] - target_centers[rank]))
        if len(samples) < 4:
            continue

        runs: list[list[tuple[float, float]]] = []
        current_run = [samples[0]]
        for sample in samples[1:]:
            if sample[0] - current_run[-1][0] <= bin_size * 1.5:
                current_run.append(sample)
            else:
                runs.append(current_run)
                current_run = [sample]
        runs.append(current_run)

        rank_reports: list[dict[str, object]] = []
        for run in runs:
            if len(run) < 4:
                continue
            run_positions = np.asarray([sample[0] for sample in run], dtype=np.float64)
            residuals = np.asarray([sample[1] for sample in run], dtype=np.float64)
            median_offset = float(np.median(residuals))
            absolute_residuals = np.abs(residuals)
            coverage_ratio = float(np.ptp(run_positions) / common_span)
            median_absolute_deviation = float(
                np.median(np.abs(residuals - median_offset))
            )
            absolute_p90 = float(np.quantile(absolute_residuals, 0.90))
            rank_reports.append(
                {
                    "track_rank": rank,
                    "run_bins": len(run),
                    "run_start": float(run_positions[0]),
                    "run_end": float(run_positions[-1]),
                    "run_span": float(np.ptp(run_positions)),
                    "coverage_ratio": coverage_ratio,
                    "median_offset": median_offset,
                    "median_absolute_deviation": median_absolute_deviation,
                    "absolute_p90": absolute_p90,
                    "qualified": bool(
                        len(run) >= minimum_run_bins
                        and coverage_ratio >= 0.65
                        and abs(median_offset) >= minimum_offset
                        and abs(median_offset) <= maximum_offset
                        and median_absolute_deviation <= residual_tolerance
                        and absolute_p90 <= maximum_absolute_p90
                    ),
                }
            )
        if not rank_reports:
            continue
        best_report = max(
            rank_reports,
            key=lambda report: (
                bool(report["qualified"]),
                float(report["coverage_ratio"]),
                int(report["run_bins"]),
            ),
        )
        track_reports.append(best_report)
        if bool(best_report["qualified"]):
            qualified_track_count += 1

    result["qualified_track_count"] = qualified_track_count
    result["qualified"] = qualified_track_count >= 2
    result["track_reports"] = track_reports
    return result


def _symmetric_profile_alignment_metrics(
    source_profile: np.ndarray,
    target_profile: np.ndarray,
    overlap_distance: float,
    minimum_overlap_points: int,
    maximum_local_distance: float,
) -> dict[str, object]:
    """Build a bidirectional report without requiring identical scan coverage.

    ``forward`` and ``reverse`` preserve which scan is being judged.  The caller
    accepts a candidate when the candidate's own covered profile agrees with a
    trusted view; it deliberately does not require the trusted view to contain
    every contour segment visible in the candidate.
    """
    forward = _profile_alignment_metrics(
        source_profile,
        target_profile,
        overlap_distance,
        maximum_local_distance,
    )
    reverse = _profile_alignment_metrics(
        target_profile,
        source_profile,
        overlap_distance,
        maximum_local_distance,
    )
    forward_ratio = float(forward["overlap_ratio"])
    reverse_ratio = float(reverse["overlap_ratio"])
    forward_p90 = forward["p90_distance"]
    reverse_p90 = reverse["p90_distance"]
    forward_p75 = forward["local_p75_distance"]
    reverse_p75 = reverse["local_p75_distance"]
    p90_values = [float(value) for value in (forward_p90, reverse_p90) if value is not None]
    p75_values = [float(value) for value in (forward_p75, reverse_p75) if value is not None]
    overlap_points = min(int(forward["overlap_points"]), int(reverse["overlap_points"]))
    overlap_ratio = min(forward_ratio, reverse_ratio)
    symmetric_p90 = max(p90_values) if p90_values else None
    symmetric_p75 = max(p75_values) if p75_values else None
    forward_compatible = _profile_alignment_is_compatible(
        forward,
        minimum_overlap_points,
        maximum_local_distance,
    )
    reverse_compatible = _profile_alignment_is_compatible(
        reverse,
        minimum_overlap_points,
        maximum_local_distance,
    )
    forward["compatible"] = forward_compatible
    reverse["compatible"] = reverse_compatible
    return {
        "forward": forward,
        "reverse": reverse,
        "overlap_points": overlap_points,
        "overlap_ratio": overlap_ratio,
        "median_distance": max(
            float(value)
            for value in (forward["median_distance"], reverse["median_distance"])
            if value is not None
        ) if forward["median_distance"] is not None and reverse["median_distance"] is not None else None,
        "local_p75_distance": symmetric_p75,
        "p90_distance": symmetric_p90,
        "unmatched_ratio": max(
            float(forward["unmatched_ratio"]),
            float(reverse["unmatched_ratio"]),
        ),
        "forward_compatible": forward_compatible,
        "reverse_compatible": reverse_compatible,
        "compatible": bool(forward_compatible or reverse_compatible),
        "mutual_compatible": bool(forward_compatible and reverse_compatible),
        "duplicate_contour_ratio": max(
            float(forward.get("duplicate_contour_ratio", 0.0)),
            float(reverse.get("duplicate_contour_ratio", 0.0)),
        ),
        "missing_contour_ratio": max(
            float(forward.get("missing_contour_ratio", 0.0)),
            float(reverse.get("missing_contour_ratio", 0.0)),
        ),
        "contour_compared_bins": max(
            int(forward.get("contour_compared_bins", 0)),
            int(reverse.get("contour_compared_bins", 0)),
        ),
    }


def _aggregate_probe_directional_metrics(
    qualities: Sequence[Mapping[str, object] | None],
    probe_offsets: Sequence[float],
    center_probe_index: int,
    minimum_overlap_points: int,
    maximum_local_distance: float,
) -> dict[str, object]:
    """Aggregate comparisons made at matching offsets without mixing slices."""
    minimum_evidence_points = max(1, min(30, int(minimum_overlap_points)))
    normalized_offsets = [float(offset) for offset in probe_offsets]
    probe_reports: list[dict[str, object]] = []
    comparable_indices: list[int] = []
    compatible_indices: list[int] = []
    mismatch_indices: list[int] = []
    extra_contour_indices: list[int] = []

    for probe_index, quality_value in enumerate(qualities):
        quality = dict(quality_value) if isinstance(quality_value, Mapping) else {}
        common_span = float(quality.get("common_horizontal_span", 0.0) or 0.0)
        comparable = bool(
            int(quality.get("compared_points", 0) or 0) >= minimum_evidence_points
            and common_span > 1e-8
        )
        compatible = bool(
            comparable
            and _profile_alignment_is_compatible(
                quality,
                minimum_overlap_points,
                maximum_local_distance,
            )
        )
        mismatch_evidence = bool(
            comparable
            and _profile_alignment_has_mismatch_evidence(
                quality,
                minimum_evidence_points,
                maximum_local_distance,
            )
        )
        extra_contour_evidence = bool(
            comparable
            and _profile_alignment_has_extra_contour_evidence(
                quality,
                minimum_evidence_points,
            )
        )
        if comparable:
            comparable_indices.append(probe_index)
        if compatible:
            compatible_indices.append(probe_index)
        if mismatch_evidence:
            mismatch_indices.append(probe_index)
        if extra_contour_evidence:
            extra_contour_indices.append(probe_index)
        if not comparable:
            status = "coverage_only"
        elif mismatch_evidence or extra_contour_evidence:
            status = "mismatch"
        elif compatible:
            status = "pass"
        else:
            status = "insufficient_evidence"
        annotated_quality = dict(quality)
        annotated_quality.update(
            {
                "probe_index": probe_index,
                "probe_offset": normalized_offsets[probe_index]
                if probe_index < len(normalized_offsets)
                else float(probe_index),
                "comparable": comparable,
                "compatible": compatible,
                "mismatch_evidence": mismatch_evidence,
                "extra_contour_evidence": extra_contour_evidence,
                "status": status,
            }
        )
        probe_reports.append(annotated_quality)

    comparable_count = len(comparable_indices)
    compatible_count = len(compatible_indices)
    strict_majority = comparable_count // 2 + 1 if comparable_count else 0
    aggregate_compatible = bool(
        compatible_count > 0
        and comparable_count > 0
        and compatible_count >= strict_majority
    )

    def persistent_evidence(indices: Sequence[int]) -> bool:
        if not indices or not comparable_count:
            return False
        required_count = (
            1
            if comparable_count == 1
            else max(SECTION_ALIGNMENT_PROBE_PERSISTENCE_MINIMUM, strict_majority)
        )
        return bool(len(indices) >= required_count)

    def numeric_values(key: str, source: Sequence[Mapping[str, object]]) -> list[float]:
        values: list[float] = []
        for item in source:
            value = item.get(key)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                values.append(numeric_value)
        return values

    comparable_qualities = [probe_reports[index] for index in comparable_indices]

    def robust_value(key: str, quantile: float = 0.75) -> float | None:
        values = numeric_values(key, comparable_qualities)
        if not values:
            return None
        return float(np.quantile(values, quantile))

    def robust_integer(key: str) -> int:
        value = robust_value(key, 0.50)
        return int(round(value)) if value is not None else 0

    center_quality = (
        probe_reports[center_probe_index]
        if 0 <= center_probe_index < len(probe_reports)
        else {}
    )
    center_profile_points = int(center_quality.get("profile_points", 0) or 0)
    if center_profile_points <= 0 and probe_reports:
        profile_point_values = numeric_values("profile_points", probe_reports)
        if profile_point_values:
            center_profile_points = int(round(float(np.median(profile_point_values))))

    p90_distance = robust_value("p90_distance")
    local_p75_distance = robust_value("local_p75_distance")
    median_distance = robust_value("median_distance")
    if p90_distance is None:
        p90_distance = robust_value("all_p90_distance")
    if local_p75_distance is None:
        local_p75_distance = robust_value("all_p75_distance")
    if median_distance is None:
        median_distance = robust_value("all_median_distance")

    abnormal_probe_indices = sorted(
        set(mismatch_indices) | set(extra_contour_indices)
    )
    return {
        "profile_points": center_profile_points,
        "common_profile_points": robust_integer("common_profile_points"),
        "common_profile_ratio": robust_value("common_profile_ratio", 0.50) or 0.0,
        "compared_points": robust_integer("compared_points"),
        "overlap_points": robust_integer("overlap_points"),
        "overlap_ratio": robust_value("overlap_ratio", 0.50) or 0.0,
        "median_distance": median_distance,
        "local_p75_distance": local_p75_distance,
        "p90_distance": p90_distance,
        "all_median_distance": robust_value("all_median_distance"),
        "all_p75_distance": robust_value("all_p75_distance"),
        "all_p90_distance": robust_value("all_p90_distance"),
        "unmatched_ratio": robust_value("unmatched_ratio", 0.75) or 0.0,
        "common_horizontal_start": (
            min(numeric_values("common_horizontal_start", comparable_qualities))
            if numeric_values("common_horizontal_start", comparable_qualities)
            else None
        ),
        "common_horizontal_end": (
            max(numeric_values("common_horizontal_end", comparable_qualities))
            if numeric_values("common_horizontal_end", comparable_qualities)
            else None
        ),
        "common_horizontal_span": robust_value("common_horizontal_span", 0.50) or 0.0,
        "profile_bin_count": robust_integer("profile_bin_count"),
        "bad_bin_count": robust_integer("bad_bin_count"),
        "bad_bin_ratio": robust_value("bad_bin_ratio", 0.75),
        "largest_bad_bin_run": max(
            (int(value) for value in numeric_values("largest_bad_bin_run", comparable_qualities)),
            default=0,
        ),
        "contour_compared_bins": max(
            (int(value) for value in numeric_values("contour_compared_bins", comparable_qualities)),
            default=0,
        ),
        "duplicate_bins": robust_integer("duplicate_bins"),
        "missing_bins": robust_integer("missing_bins"),
        "duplicate_contour_ratio": robust_value("duplicate_contour_ratio", 0.75) or 0.0,
        "missing_contour_ratio": robust_value("missing_contour_ratio", 0.75) or 0.0,
        "source_cluster_count": robust_integer("source_cluster_count"),
        "target_cluster_count": robust_integer("target_cluster_count"),
        "compatible": aggregate_compatible,
        "valid_probe_count": comparable_count,
        "compatible_probe_count": compatible_count,
        "coverage_only_probe_count": len(probe_reports) - comparable_count,
        "mismatch_probe_count": len(mismatch_indices),
        "extra_contour_probe_count": len(extra_contour_indices),
        "abnormal_evidence_count": len(abnormal_probe_indices),
        "valid_probe_indices": comparable_indices,
        "compatible_probe_indices": compatible_indices,
        "mismatch_probe_indices": mismatch_indices,
        "extra_contour_probe_indices": extra_contour_indices,
        "abnormal_probe_indices": abnormal_probe_indices,
        "persistent_mismatch_evidence": persistent_evidence(mismatch_indices),
        "persistent_extra_contour_evidence": persistent_evidence(extra_contour_indices),
        "center_probe_index": center_probe_index,
        "center_probe_status": center_quality.get("status", "coverage_only"),
        "probe_metrics": probe_reports,
    }


def _multi_probe_profile_alignment_metrics(
    source_profiles: Sequence[np.ndarray],
    target_profiles: Sequence[np.ndarray],
    probe_offsets: Sequence[float],
    center_probe_index: int,
    overlap_distance: float,
    minimum_overlap_points: int,
    maximum_local_distance: float,
) -> dict[str, object]:
    """Compare each probe only with the target probe at the same offset."""
    probe_count = max(len(source_profiles), len(target_profiles), len(probe_offsets))
    normalized_offsets = [float(offset) for offset in probe_offsets]
    forward_qualities: list[dict[str, object]] = []
    reverse_qualities: list[dict[str, object]] = []
    raw_probe_pairs: list[dict[str, object]] = []
    parallel_duplicate_reports: list[dict[str, object]] = []
    empty_profile = np.empty((0, 2), dtype=np.float64)
    for probe_index in range(probe_count):
        source_profile = (
            np.asarray(source_profiles[probe_index], dtype=np.float64)
            if probe_index < len(source_profiles)
            else empty_profile
        )
        target_profile = (
            np.asarray(target_profiles[probe_index], dtype=np.float64)
            if probe_index < len(target_profiles)
            else empty_profile
        )
        pair_quality = _symmetric_profile_alignment_metrics(
            source_profile,
            target_profile,
            overlap_distance,
            minimum_overlap_points,
            maximum_local_distance,
        )
        parallel_duplicate_reports.append(
            _parallel_contour_track_evidence(
                source_profile,
                target_profile,
                overlap_distance,
                maximum_local_distance,
            )
        )
        forward_quality = pair_quality.get("forward", {})
        reverse_quality = pair_quality.get("reverse", {})
        forward_qualities.append(
            dict(forward_quality) if isinstance(forward_quality, Mapping) else {}
        )
        reverse_qualities.append(
            dict(reverse_quality) if isinstance(reverse_quality, Mapping) else {}
        )
        raw_probe_pairs.append(pair_quality)

    forward = _aggregate_probe_directional_metrics(
        forward_qualities,
        normalized_offsets,
        center_probe_index,
        minimum_overlap_points,
        maximum_local_distance,
    )
    reverse = _aggregate_probe_directional_metrics(
        reverse_qualities,
        normalized_offsets,
        center_probe_index,
        minimum_overlap_points,
        maximum_local_distance,
    )
    forward_probe_reports = forward.pop("probe_metrics", [])
    reverse_probe_reports = reverse.pop("probe_metrics", [])
    probe_reports: list[dict[str, object]] = []
    for probe_index, pair_quality in enumerate(raw_probe_pairs):
        forward_report = (
            forward_probe_reports[probe_index]
            if probe_index < len(forward_probe_reports)
            and isinstance(forward_probe_reports[probe_index], Mapping)
            else {}
        )
        reverse_report = (
            reverse_probe_reports[probe_index]
            if probe_index < len(reverse_probe_reports)
            and isinstance(reverse_probe_reports[probe_index], Mapping)
            else {}
        )
        probe_reports.append(
            {
                "probe_index": probe_index,
                "probe_offset": normalized_offsets[probe_index]
                if probe_index < len(normalized_offsets)
                else float(probe_index),
                "forward": dict(forward_report),
                "reverse": dict(reverse_report),
                "compatible": bool(pair_quality.get("compatible", False)),
                "mutual_compatible": bool(pair_quality.get("mutual_compatible", False)),
                "mismatch_evidence": bool(
                    forward_report.get("mismatch_evidence", False)
                    or reverse_report.get("mismatch_evidence", False)
                ),
                "extra_contour_evidence": bool(
                    forward_report.get("extra_contour_evidence", False)
                    or reverse_report.get("extra_contour_evidence", False)
                ),
                "parallel_duplicate_evidence": bool(
                    parallel_duplicate_reports[probe_index].get("qualified", False)
                ),
                "parallel_duplicate_track_count": int(
                    parallel_duplicate_reports[probe_index].get(
                        "qualified_track_count",
                        0,
                    )
                ),
                "parallel_duplicate_track_reports": parallel_duplicate_reports[
                    probe_index
                ].get("track_reports", []),
            }
        )

    p90_values = [
        float(value)
        for value in (forward.get("p90_distance"), reverse.get("p90_distance"))
        if value is not None
    ]
    p75_values = [
        float(value)
        for value in (forward.get("local_p75_distance"), reverse.get("local_p75_distance"))
        if value is not None
    ]
    pair_mismatch_indices = sorted(
        set(forward.get("mismatch_probe_indices", []))
        | set(reverse.get("mismatch_probe_indices", []))
    )
    pair_extra_indices = sorted(
        set(forward.get("extra_contour_probe_indices", []))
        | set(reverse.get("extra_contour_probe_indices", []))
    )
    pair_abnormal_indices = sorted(set(pair_mismatch_indices) | set(pair_extra_indices))
    parallel_duplicate_indices = [
        probe_index
        for probe_index, evidence in enumerate(parallel_duplicate_reports)
        if bool(evidence.get("qualified", False))
    ]
    required_parallel_probe_count = (
        1
        if len(parallel_duplicate_reports) == 1
        else max(2, len(parallel_duplicate_reports) // 2 + 1)
    )
    parallel_duplicate_track_count = min(
        (
            int(parallel_duplicate_reports[probe_index].get("qualified_track_count", 0))
            for probe_index in parallel_duplicate_indices
        ),
        default=0,
    )
    return {
        "forward": forward,
        "reverse": reverse,
        "overlap_points": min(
            int(forward.get("overlap_points", 0)),
            int(reverse.get("overlap_points", 0)),
        ),
        "overlap_ratio": min(
            float(forward.get("overlap_ratio", 0.0) or 0.0),
            float(reverse.get("overlap_ratio", 0.0) or 0.0),
        ),
        "median_distance": max(
            [
                float(value)
                for value in (forward.get("median_distance"), reverse.get("median_distance"))
                if value is not None
            ],
            default=None,
        ),
        "local_p75_distance": max(p75_values, default=None),
        "p90_distance": max(p90_values, default=None),
        "unmatched_ratio": max(
            float(forward.get("unmatched_ratio", 0.0) or 0.0),
            float(reverse.get("unmatched_ratio", 0.0) or 0.0),
        ),
        "forward_compatible": bool(forward.get("compatible", False)),
        "reverse_compatible": bool(reverse.get("compatible", False)),
        "compatible": bool(
            forward.get("compatible", False) or reverse.get("compatible", False)
        ),
        "mutual_compatible": bool(
            forward.get("compatible", False) and reverse.get("compatible", False)
        ),
        "duplicate_contour_ratio": max(
            float(forward.get("duplicate_contour_ratio", 0.0) or 0.0),
            float(reverse.get("duplicate_contour_ratio", 0.0) or 0.0),
        ),
        "missing_contour_ratio": max(
            float(forward.get("missing_contour_ratio", 0.0) or 0.0),
            float(reverse.get("missing_contour_ratio", 0.0) or 0.0),
        ),
        "contour_compared_bins": max(
            int(forward.get("contour_compared_bins", 0)),
            int(reverse.get("contour_compared_bins", 0)),
        ),
        "valid_probe_count": min(
            int(forward.get("valid_probe_count", 0)),
            int(reverse.get("valid_probe_count", 0)),
        ),
        "compatible_probe_count": min(
            int(forward.get("compatible_probe_count", 0)),
            int(reverse.get("compatible_probe_count", 0)),
        ),
        "mismatch_probe_count": len(pair_mismatch_indices),
        "extra_contour_probe_count": len(pair_extra_indices),
        "abnormal_evidence_count": len(pair_abnormal_indices),
        "mismatch_probe_indices": pair_mismatch_indices,
        "extra_contour_probe_indices": pair_extra_indices,
        "abnormal_probe_indices": pair_abnormal_indices,
        "persistent_mismatch_evidence": bool(
            forward.get("persistent_mismatch_evidence", False)
            or reverse.get("persistent_mismatch_evidence", False)
        ),
        "persistent_extra_contour_evidence": bool(
            forward.get("persistent_extra_contour_evidence", False)
            or reverse.get("persistent_extra_contour_evidence", False)
        ),
        "parallel_duplicate_probe_indices": parallel_duplicate_indices,
        "parallel_duplicate_probe_count": len(parallel_duplicate_indices),
        "parallel_duplicate_track_count": parallel_duplicate_track_count,
        "persistent_parallel_duplicate_evidence": bool(
            len(parallel_duplicate_indices) >= required_parallel_probe_count
            and parallel_duplicate_track_count >= 2
        ),
        "probe_offsets": normalized_offsets,
        "center_probe_index": center_probe_index,
        "probe_metrics": probe_reports,
    }


def _isolated_peer_component_anomaly(
    component: Sequence[int],
    trusted_indices: Sequence[int],
    pair_metrics: Mapping[str, Mapping[str, object]],
    active_view_count: int,
    minimum_retained_views: int,
    profile_horizontal_spans: Mapping[int, float] | None = None,
) -> dict[str, object] | None:
    """Identify a mutually consistent peer cluster with a wrong shared contour.

    A peer cluster is only rejected as a station-level anomaly when it disagrees
    over most of the trusted profile span.  A scan that disagrees only over a
    shorter, well-covered interval is commonly an occlusion/complementary view
    and must remain available for section recovery.
    """
    component_indices = sorted({int(index) for index in component})
    trusted_view_indices = sorted({int(index) for index in trusted_indices})
    if len(component_indices) < 2 or not trusted_view_indices:
        return None
    if active_view_count - len(component_indices) < minimum_retained_views:
        return None
    if set(component_indices) & set(trusted_view_indices):
        return None

    minimum_mismatch_views = min(
        SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_MISMATCH_VIEWS,
        len(trusted_view_indices),
    )
    minimum_mismatch_views = max(1, minimum_mismatch_views)
    normalized_profile_spans = {
        int(index): max(0.0, float(span))
        for index, span in (profile_horizontal_spans or {}).items()
    }
    member_reports: list[dict[str, object]] = []
    anomalous_member_count = 0

    for component_index in component_indices:
        mismatch_indices: list[int] = []
        qualified_mismatch_indices: list[int] = []
        qualified_parallel_duplicate_indices: list[int] = []
        parallel_duplicate_track_counts: list[int] = []
        duplicate_ratios: list[float] = []
        missing_ratios: list[float] = []
        contour_bin_counts: list[int] = []
        trusted_coverage_ratios: list[float] = []
        common_spans: list[float] = []
        for trusted_index in trusted_view_indices:
            pair_key = f"{min(component_index, trusted_index)}:{max(component_index, trusted_index)}"
            quality = pair_metrics.get(pair_key)
            if not isinstance(quality, Mapping):
                continue
            if not bool(quality.get("persistent_mismatch_evidence", False)):
                continue
            mismatch_indices.append(trusted_index)
            overlap_ratio = float(quality.get("overlap_ratio", 0.0) or 0.0)
            contour_bin_count = int(quality.get("contour_compared_bins", 0) or 0)
            if (
                int(quality.get("valid_probe_count", 0) or 0)
                >= SECTION_ALIGNMENT_PROBE_PERSISTENCE_MINIMUM
                and overlap_ratio >= SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_OVERLAP_RATIO
                and contour_bin_count >= SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_CONTOUR_BINS
            ):
                qualified_mismatch_indices.append(trusted_index)
                duplicate_ratios.append(
                    float(quality.get("duplicate_contour_ratio", 0.0) or 0.0)
                )
                missing_ratios.append(
                    float(quality.get("missing_contour_ratio", 0.0) or 0.0)
                )
                contour_bin_counts.append(contour_bin_count)
                directional_quality = (
                    quality.get("reverse")
                    if component_index > trusted_index
                    else quality.get("forward")
                )
                if isinstance(directional_quality, Mapping):
                    common_span = float(
                        directional_quality.get("common_horizontal_span", 0.0)
                        or 0.0
                    )
                else:
                    common_span = 0.0
                trusted_span = normalized_profile_spans.get(trusted_index, 0.0)
                if trusted_span > 1e-8:
                    trusted_coverage_ratios.append(
                        min(1.0, max(0.0, common_span / trusted_span))
                    )
                common_spans.append(common_span)
                parallel_track_count = int(
                    quality.get("parallel_duplicate_track_count", 0) or 0
                )
                if (
                    bool(quality.get("persistent_parallel_duplicate_evidence", False))
                    and parallel_track_count >= 2
                ):
                    qualified_parallel_duplicate_indices.append(trusted_index)
                    parallel_duplicate_track_counts.append(parallel_track_count)

        minimum_duplicate_ratio = min(duplicate_ratios, default=0.0)
        minimum_missing_ratio = min(missing_ratios, default=0.0)
        peak_contour_ratio = max(
            [*duplicate_ratios, *missing_ratios],
            default=0.0,
        )
        median_trusted_coverage_ratio = float(
            np.median(trusted_coverage_ratios)
        ) if trusted_coverage_ratios else 1.0
        minimum_trusted_coverage_ratio = min(
            trusted_coverage_ratios,
            default=median_trusted_coverage_ratio,
        )
        maximum_common_span = max(common_spans, default=0.0)
        member_is_anomalous = bool(
            len(mismatch_indices) >= minimum_mismatch_views
            and len(qualified_mismatch_indices) >= minimum_mismatch_views
            and minimum_duplicate_ratio
            >= SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_CONTOUR_RATIO
            and minimum_missing_ratio
            >= SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_CONTOUR_RATIO
            and peak_contour_ratio
            >= SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_PEAK_CONTOUR_RATIO
        )
        if member_is_anomalous:
            anomalous_member_count += 1
        member_reports.append(
            {
                "index": component_index,
                "mismatch_trusted_indices": mismatch_indices,
                "qualified_mismatch_trusted_indices": qualified_mismatch_indices,
                "qualified_parallel_duplicate_trusted_indices": qualified_parallel_duplicate_indices,
                "minimum_parallel_duplicate_track_count": min(
                    parallel_duplicate_track_counts,
                    default=0,
                ),
                "minimum_duplicate_contour_ratio": minimum_duplicate_ratio,
                "minimum_missing_contour_ratio": minimum_missing_ratio,
                "peak_contour_ratio": peak_contour_ratio,
                "minimum_contour_compared_bins": min(contour_bin_counts, default=0),
                "median_trusted_coverage_ratio": median_trusted_coverage_ratio,
                "minimum_trusted_coverage_ratio": minimum_trusted_coverage_ratio,
                "maximum_common_span": maximum_common_span,
                "anomalous": member_is_anomalous,
            }
        )

    if anomalous_member_count != len(component_indices):
        return None

    # Do not throw away a mutually corroborated view merely because its
    # visible interval contains a different contour.  The interval itself is
    # the evidence of complementary coverage when it is substantially shorter
    # than the trusted views and no long parallel duplicate tracks were found.
    local_coverage_members = [
        report
        for report in member_reports
        if float(report.get("median_trusted_coverage_ratio", 1.0))
        <= SECTION_ALIGNMENT_LOCAL_COVERAGE_MAX_TRUSTED_SPAN_RATIO
        and float(report.get("maximum_common_span", 0.0))
        >= SECTION_ALIGNMENT_LOCAL_COVERAGE_MIN_COMMON_SPAN
        and not report.get("qualified_parallel_duplicate_trusted_indices")
    ]
    local_coverage_protected = bool(
        member_reports and len(local_coverage_members) == len(member_reports)
    )
    return {
        "component_indices": component_indices,
        "trusted_indices": trusted_view_indices,
        "minimum_mismatch_views": minimum_mismatch_views,
        "minimum_overlap_ratio": SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_OVERLAP_RATIO,
        "minimum_contour_compared_bins": SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_CONTOUR_BINS,
        "minimum_contour_ratio": SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_CONTOUR_RATIO,
        "minimum_peak_contour_ratio": SECTION_ALIGNMENT_ISOLATED_PEER_MINIMUM_PEAK_CONTOUR_RATIO,
        "local_coverage_max_trusted_span_ratio": SECTION_ALIGNMENT_LOCAL_COVERAGE_MAX_TRUSTED_SPAN_RATIO,
        "local_coverage_min_common_span": SECTION_ALIGNMENT_LOCAL_COVERAGE_MIN_COMMON_SPAN,
        "local_coverage_protected": local_coverage_protected,
        "member_reports": member_reports,
    }


def _profile_outside_trusted_span(
    profile: np.ndarray,
    trusted_profiles: Sequence[np.ndarray],
    margin: float,
) -> float:
    coordinates = np.asarray(profile, dtype=np.float64)
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if not len(coordinates) or not trusted_profiles:
        return 1.0
    spans = []
    for trusted_profile in trusted_profiles:
        trusted = np.asarray(trusted_profile, dtype=np.float64)
        trusted = trusted[np.isfinite(trusted).all(axis=1)]
        if len(trusted):
            spans.append((float(trusted[:, 0].min()) - margin, float(trusted[:, 0].max()) + margin))
    if not spans:
        return 1.0
    inside = np.zeros(len(coordinates), dtype=bool)
    for lower, upper in spans:
        inside |= (coordinates[:, 0] >= lower) & (coordinates[:, 0] <= upper)
    return float(1.0 - np.mean(inside))


def _render_section_view_alignment_audit(
    report: Mapping[str, object],
    labels: Sequence[str],
    metrics: Sequence[RegistrationMetric],
    section_config: SectionConfig,
    output_path: Path,
) -> None:
    if not bool(report.get("enabled", True)):
        _render_section_view_alignment_placeholder(
            report,
            labels,
            metrics,
            output_path,
            title="多工位截面质量诊断已关闭",
            message="本次未执行多截面质量诊断。\n仅使用标定块位姿图，并保留标定块质量合格视角的互补覆盖。",
        )
        return

    profiles_value = report.get("profiles", {})
    if not isinstance(profiles_value, Mapping):
        _render_section_view_alignment_placeholder(
            report,
            labels,
            metrics,
            output_path,
            title="多工位截面验收无可用报告",
            message="没有生成可用于审核的薄截面轮廓。\n请检查截面平面位置、厚度和目标区域。",
        )
        return
    profiles: dict[int, np.ndarray] = {}
    for key, value in profiles_value.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        profile = np.asarray(value, dtype=np.float64)
        if profile.ndim == 2 and profile.shape[1] == 2 and len(profile):
            profiles[index] = profile
    if not profiles:
        _render_section_view_alignment_placeholder(
            report,
            labels,
            metrics,
            output_path,
            title="多工位截面验收无可用轮廓",
            message="没有生成可用于审核的薄截面轮廓。\n请检查截面平面位置、厚度和目标区域。",
        )
        return

    view_reports_value = report.get("views", [])
    view_reports: dict[int, Mapping[str, object]] = {}
    if isinstance(view_reports_value, Sequence) and not isinstance(
        view_reports_value, (str, bytes)
    ):
        for view_report_value in view_reports_value:
            if not isinstance(view_report_value, Mapping):
                continue
            view_index = view_report_value.get("index")
            if isinstance(view_index, (int, np.integer)):
                view_reports[int(view_index)] = view_report_value
    probe_offsets_value = report.get("probe_offsets", [])
    if isinstance(probe_offsets_value, Sequence) and not isinstance(
        probe_offsets_value, (str, bytes)
    ):
        probe_offsets = [
            float(offset)
            for offset in probe_offsets_value
            if isinstance(offset, (int, float, np.integer, np.floating))
        ]
    else:
        probe_offsets = []
    center_probe_index = int(report.get("center_probe_index", 0) or 0)
    center_probe_offset = (
        probe_offsets[center_probe_index]
        if 0 <= center_probe_index < len(probe_offsets)
        else 0.0
    )

    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf")
    status_labels = {
        "reference": "参考",
        "pass": "通过",
        "peer_supported_coverage": "互证覆盖",
        "coverage_only": "局部覆盖",
        "insufficient_evidence": "点数不足",
        "mismatch": (
            "仅诊断"
            if bool(report.get("coverage_preserving_fusion", True))
            else "剔除"
        ),
        "not_checked": "未检查",
    }
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes_flat = list(np.asarray(axes).reshape(-1))
    all_coordinates = np.concatenate(list(profiles.values()), axis=0)
    x_limits = _section_plot_limits(all_coordinates[:, 0], section_config.horizontal_min, section_config.horizontal_max)
    y_limits = _section_plot_limits(all_coordinates[:, 1], section_config.vertical_min, section_config.vertical_max)

    overview = axes_flat[0]
    for index, profile in profiles.items():
        status = metrics[index].section_validation_status or "unknown"
        color = "#d62728" if not metrics[index].included_in_fusion else palette[index % len(palette)]
        overview.scatter(
            profile[:, 0],
            profile[:, 1],
            s=0.35,
            alpha=0.45,
            color=color,
            label=f"{labels[index]}:{status_labels.get(status, status)}",
            rasterized=True,
        )
    overview.set_title(
        f"多工位截面叠加（覆盖优先，仅标定块不合格工位可不参与） | center probe {center_probe_offset:+.3f}"
        if bool(report.get("coverage_preserving_fusion", True))
        else f"多工位截面叠加（红色为剔除工位） | center probe {center_probe_offset:+.3f}"
    )
    overview.legend(markerscale=8, fontsize=8, loc="best")
    overview.set_xlim(*x_limits)
    overview.set_ylim(*y_limits)
    overview.set_aspect("equal", adjustable="box")
    overview.grid(alpha=0.25)

    for axis, (index, profile) in zip(axes_flat[1:], sorted(profiles.items()), strict=False):
        metric = metrics[index]
        color = "#d62728" if not metric.included_in_fusion else palette[index % len(palette)]
        axis.scatter(profile[:, 0], profile[:, 1], s=0.45, alpha=0.65, color=color, rasterized=True)
        status = metric.section_validation_status or "unknown"
        axis.set_title(f"{labels[index]} | {status_labels.get(status, status)}")
        view_report = view_reports.get(index, {})
        probe_count = int(view_report.get("probe_count", len(probe_offsets)) or 0)
        valid_probe_count = int(view_report.get("valid_probe_count", 0) or 0)
        abnormal_offsets = [
            float(offset)
            for offset in (
                list(view_report.get("mismatch_probe_offsets", []))
                + list(view_report.get("extra_contour_probe_offsets", []))
            )
            if isinstance(offset, (int, float, np.integer, np.floating))
        ]
        unique_abnormal_offsets = sorted(set(abnormal_offsets))
        abnormal_label = (
            ", ".join(f"{offset:+.3f}" for offset in unique_abnormal_offsets)
            if unique_abnormal_offsets
            else "none"
        )
        axis.text(
            0.02,
            0.02,
            f"probe valid: {valid_probe_count}/{probe_count}\nabnormal offsets: {abnormal_label}",
            transform=axis.transAxes,
            fontsize=8,
            color="#8b0000" if unique_abnormal_offsets else "#455a64",
            bbox={"facecolor": "white", "alpha": 0.78, "pad": 3.0, "edgecolor": "none"},
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(profiles) + 1 :]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=210, facecolor="white")
    plt.close(figure)


def _render_section_view_alignment_placeholder(
    report: Mapping[str, object],
    labels: Sequence[str],
    metrics: Sequence[RegistrationMetric],
    output_path: Path,
    *,
    title: str,
    message: str,
) -> None:
    """Keep the audit artifact stable when thin-section validation is skipped."""
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.axis("off")
    lines = [message, "", f"工位数量：{len(metrics)}"]
    for index, metric in enumerate(metrics):
        label = labels[index] if index < len(labels) else f"view {index + 1}"
        status = metric.section_validation_status or "未执行"
        included = "参与融合" if metric.included_in_fusion else "未参与融合"
        lines.append(f"{label}：{status}，{included}")
    axis.text(
        0.04,
        0.92,
        title,
        transform=axis.transAxes,
        fontsize=19,
        fontweight="bold",
        color="#17365d",
        va="top",
    )
    axis.text(
        0.06,
        0.78,
        "\n".join(lines),
        transform=axis.transAxes,
        fontsize=12,
        color="#334e68",
        va="top",
        linespacing=1.65,
        bbox={
            "facecolor": "#f4f7fb",
            "edgecolor": "#b8c7d9",
            "boxstyle": "round,pad=0.8",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=210, facecolor="white")
    plt.close(figure)


def _render_section_coverage_audit(
    report: Mapping[str, object],
    labels: Sequence[str],
    metrics: Sequence[RegistrationMetric],
    output_path: Path,
) -> None:
    """Render horizontal coverage and support-count diagnostics without altering data."""
    profiles_value = report.get("profiles", {})
    profiles: dict[int, np.ndarray] = {}
    if isinstance(profiles_value, Mapping):
        for key, value in profiles_value.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            profile = np.asarray(value, dtype=np.float64)
            if profile.ndim == 2 and profile.shape[1] == 2 and len(profile):
                finite_profile = profile[np.isfinite(profile).all(axis=1)]
                if len(finite_profile):
                    profiles[index] = finite_profile

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": (1.15, 1.0)},
        constrained_layout=True,
    )
    if not profiles:
        axes[0].axis("off")
        axes[1].axis("off")
        axes[0].text(
            0.03,
            0.82,
            "截面覆盖诊断无可用薄截面轮廓",
            fontsize=18,
            fontweight="bold",
            color="#17365d",
        )
        axes[0].text(0.03, 0.62, "请检查截面平面位置、厚度和目标区域。", fontsize=12)
    else:
        ordered_indices = sorted(profiles)
        lower = min(float(np.min(profiles[index][:, 0])) for index in ordered_indices)
        upper = max(float(np.max(profiles[index][:, 0])) for index in ordered_indices)
        span = max(upper - lower, 1e-6)
        padding = max(span * 0.03, 0.05)
        lower -= padding
        upper += padding
        colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#8c564b")

        coverage_axis = axes[0]
        support_axis = axes[1]
        for row, index in enumerate(ordered_indices):
            profile = profiles[index]
            x_min = float(np.min(profile[:, 0]))
            x_max = float(np.max(profile[:, 0]))
            included = bool(metrics[index].included_in_fusion) if index < len(metrics) else True
            color = colors[index % len(colors)] if included else "#d62728"
            label = labels[index] if index < len(labels) else f"view {index + 1}"
            coverage_axis.plot(
                [x_min, x_max],
                [row, row],
                color=color,
                linewidth=8,
                solid_capstyle="butt",
                alpha=0.88,
            )
            coverage_axis.text(
                lower,
                row,
                f"{label}  {len(profile):,} 点",
                va="center",
                ha="left",
                fontsize=9,
                color="#263238",
                bbox={"facecolor": "white", "alpha": 0.82, "pad": 1.5, "edgecolor": "none"},
            )
        coverage_axis.set_xlim(lower, upper)
        coverage_axis.set_ylim(-1, len(ordered_indices))
        coverage_axis.set_yticks([])
        coverage_axis.set_title(
            "各工位截面横向覆盖范围（颜色区分来源；红色仅表示标定块质量闸门未通过）"
            if bool(report.get("coverage_preserving_fusion", True))
            else "各工位截面横向覆盖范围（红色表示未参与融合）"
        )
        coverage_axis.set_xlabel("平面内横向坐标")
        coverage_axis.grid(axis="x", alpha=0.25)

        bin_count = int(np.clip(np.ceil(span / 0.20), 40, 180))
        edges = np.linspace(lower, upper, bin_count + 1)
        centers = (edges[:-1] + edges[1:]) * 0.5
        support = np.zeros(bin_count, dtype=np.int32)
        for index in ordered_indices:
            x_values = profiles[index][:, 0]
            occupied = np.unique(np.clip(np.searchsorted(edges, x_values, side="right") - 1, 0, bin_count - 1))
            support[occupied] += 1
        support_axis.step(centers, support, where="mid", color="#17365d", linewidth=1.8)
        support_axis.fill_between(centers, support, step="mid", alpha=0.18, color="#4f81bd")
        support_axis.axhline(1, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7)
        support_axis.set_xlim(lower, upper)
        support_axis.set_ylim(0, max(int(np.max(support)) + 1, 2))
        support_axis.set_ylabel("支持工位数")
        support_axis.set_xlabel("平面内横向坐标")
        support_axis.set_title(
            f"横向覆盖支持数：共同覆盖 {int(np.sum(support >= 2))} 格，"
            f"单视角 {int(np.sum(support == 1))} 格，缺口 {int(np.sum(support == 0))} 格"
        )
        support_axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=210, facecolor="white")
    plt.close(figure)


def _validate_section_view_alignment_legacy(
    files: Sequence[Path],
    transformations: Sequence[np.ndarray],
    metrics: Sequence[RegistrationMetric],
    leveling_transformation: np.ndarray,
    section_config: SectionConfig,
    registration_config: RegistrationConfig,
    backend: str,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
) -> None:
    """Validate accepted poses with original full-resolution thin sections."""
    if not registration_config.use_section_alignment_validation:
        return
    if len(files) != len(transformations) or len(files) != len(metrics):
        raise ValueError("Section-alignment files, transformations, and metrics must have matching sizes.")

    active_indices = [index for index, metric in enumerate(metrics) if metric.included_in_fusion]
    if len(active_indices) < 3:
        return

    thin_section_config = copy.deepcopy(section_config)
    thin_section_config.thickness = min(
        section_config.thickness,
        registration_config.section_alignment_thickness,
    )
    thin_section_config.use_statistical_filter = False
    thin_section_config.use_component_filter = False

    probe_radius = registration_config.section_alignment_probe_count // 2
    probe_offsets = [
        probe_index * registration_config.section_alignment_probe_spacing
        for probe_index in range(-probe_radius, probe_radius + 1)
    ]
    section_parts: dict[int, np.ndarray] = {}
    for index in active_indices:
        path = files[index]
        _emit(
            progress_callback,
            f"Group {group_name}: original thin-section validation {index + 1}/{len(files)} - {path.name}",
        )
        raw_cloud = read_point_cloud(path)
        final_transformation = np.asarray(leveling_transformation) @ np.asarray(transformations[index])
        aligned_cloud = _transform_cloud(raw_cloud, final_transformation, backend)
        probe_parts: list[np.ndarray] = []
        for probe_offset in probe_offsets:
            probe_config = copy.deepcopy(thin_section_config)
            probe_config.plane_offset += probe_offset
            section_cloud = extract_vertical_section(aligned_cloud, probe_config, backend, clean=False)
            probe_points = np.asarray(section_cloud.points)
            if len(probe_points):
                probe_parts.append(probe_points.copy())
        section_parts[index] = (
            np.concatenate(probe_parts, axis=0)
            if probe_parts
            else np.empty((0, 3), dtype=np.float64)
        )

    pair_qualities: dict[tuple[int, int], SectionAlignmentQuality] = {}
    adjacency: dict[int, set[int]] = {index: set() for index in active_indices}

    for source_position, source_index in enumerate(active_indices):
        for target_index in active_indices[source_position + 1 :]:
            source_quality = evaluate_section_alignment(
                section_parts[source_index],
                section_parts[target_index],
                registration_config.section_alignment_overlap_distance,
            )
            target_quality = evaluate_section_alignment(
                section_parts[target_index],
                section_parts[source_index],
                registration_config.section_alignment_overlap_distance,
            )
            pair_qualities[(source_index, target_index)] = source_quality
            pair_qualities[(target_index, source_index)] = target_quality
            if _section_alignment_is_compatible(source_quality, registration_config):
                adjacency[target_index].add(source_index)
            if _section_alignment_is_compatible(target_quality, registration_config):
                adjacency[source_index].add(target_index)

    reference_index = next(
        (
            index
            for index in active_indices
            if metrics[index].pose_prior_label == "reference"
        ),
        active_indices[0],
    )
    trusted_indices = {reference_index}
    pending = [reference_index]
    while pending:
        current_index = pending.pop()
        for neighbor_index in adjacency[current_index]:
            if neighbor_index in trusted_indices:
                continue
            trusted_indices.add(neighbor_index)
            pending.append(neighbor_index)

    unverifiable_indices = {
        index
        for index in active_indices
        if len(section_parts[index]) < registration_config.section_alignment_minimum_overlap_points
    }
    trusted_indices.update(unverifiable_indices)

    minimum_views = min(registration_config.section_alignment_minimum_views, len(active_indices))
    while len(trusted_indices) < minimum_views:
        candidates = [index for index in active_indices if index not in trusted_indices]
        if not candidates:
            break

        def candidate_score(candidate_index: int) -> tuple[float, int]:
            qualities = [
                pair_qualities[(candidate_index, trusted_index)]
                for trusted_index in trusted_indices
                if trusted_index != candidate_index
            ]
            valid_p90 = [
                quality.p90_distance
                for quality in qualities
                if quality.p90_distance is not None
            ]
            best_p90 = min(valid_p90) if valid_p90 else math.inf
            best_overlap = max((quality.overlap_points for quality in qualities), default=0)
            return best_p90, -best_overlap

        trusted_indices.add(min(candidates, key=candidate_score))

    for index in active_indices:
        comparison_indices = [other for other in trusted_indices if other != index]
        qualities = [
            pair_qualities[(index, other)]
            for other in comparison_indices
            if (index, other) in pair_qualities
        ]
        if qualities:
            quality = min(
                qualities,
                key=lambda item: (
                    item.p90_distance if item.p90_distance is not None else math.inf,
                    -item.overlap_points,
                ),
            )
        else:
            quality = SectionAlignmentQuality(len(section_parts[index]), 0, None, None, 0.0, None)
        metric = metrics[index]
        metric.section_points = quality.section_points
        metric.section_overlap_points = quality.overlap_points
        metric.section_overlap_ratio = quality.overlap_ratio
        metric.section_median_distance = quality.median_distance
        metric.section_local_p75_distance = quality.local_p75_distance
        metric.section_p90_distance = quality.p90_distance
        if index in unverifiable_indices:
            metric.section_alignment_passed = None
        else:
            metric.section_alignment_passed = index in trusted_indices
        if index in trusted_indices:
            continue
        metric.included_in_fusion = False
        metric.exclusion_reason = "section_alignment"
        _emit(
            progress_callback,
            f"Group {group_name}: excluded {metric.file_name}; no compatible multi-section path "
            f"to the reference view within {registration_config.section_alignment_max_p90_distance:.4f}",
        )


def validate_section_view_alignment(
    files: Sequence[Path],
    transformations: Sequence[np.ndarray],
    metrics: Sequence[RegistrationMetric],
    leveling_transformation: np.ndarray,
    section_config: SectionConfig,
    registration_config: RegistrationConfig,
    backend: str,
    progress_callback: ProgressCallback | None = None,
    group_name: str = "",
    *,
    conservative: bool = False,
    source_clouds: Sequence[o3d.geometry.PointCloud] | None = None,
) -> dict[str, object]:
    """Use fixed calibration-block poses to identify anomalous scan views."""
    if len(files) != len(transformations) or len(files) != len(metrics):
        raise ValueError("Section-alignment files, transformations, and metrics must have matching sizes.")
    if source_clouds is not None and len(source_clouds) != len(files):
        raise ValueError("Section-alignment source clouds must match the input files.")

    coverage_preserving_fusion = bool(
        getattr(registration_config, "coverage_preserving_fusion", True)
    )

    report: dict[str, object] = {
        "enabled": bool(registration_config.use_section_alignment_validation),
        "reference_index": 0,
        "trusted_indices": [],
        "profiles": {},
        "profile_horizontal_spans": {},
        "pair_metrics": {},
        "probe_metrics": {},
        "probe_offsets": [],
        "center_probe_index": 0,
        "probe_count": 0,
        "views": [],
        "isolated_peer_components": [],
        "protected_peer_components": [],
        "section_consensus": {
            "enabled": False,
            "method": (
                "conservative_shared_section_contour_consensus_station_filter"
                if conservative
                else "shared_section_contour_consensus_station_filter"
            ),
            "mode": "conservative" if conservative else "strict",
            "coordinates_adjusted": False,
            "initial_active_indices": [],
            "final_active_indices": [],
            "excluded_indices": [],
            "filter_applied": False,
            "diagnostic_only": coverage_preserving_fusion,
            "views": [],
        },
        "coverage_preserving_fusion": coverage_preserving_fusion,
        "fusion_filter_applied": False,
        "input_source": (
            "single_view_denoised_clouds"
            if source_clouds is not None
            else "raw_ply_clouds"
        ),
    }
    if not registration_config.use_section_alignment_validation:
        return report

    active_indices = [index for index, metric in enumerate(metrics) if metric.included_in_fusion]
    if len(active_indices) < 3:
        for index in active_indices:
            metrics[index].section_validation_status = "not_checked"
        report["trusted_indices"] = active_indices.copy()
        return report

    thin_section_config = copy.deepcopy(section_config)
    thin_section_config.thickness = min(
        section_config.thickness,
        registration_config.section_alignment_thickness,
    )
    thin_section_config.use_statistical_filter = False
    thin_section_config.use_component_filter = False
    probe_radius = registration_config.section_alignment_probe_count // 2
    probe_offsets = [
        probe_index * registration_config.section_alignment_probe_spacing
        for probe_index in range(-probe_radius, probe_radius + 1)
    ]
    center_probe_index = len(probe_offsets) // 2
    _, horizontal, vertical = vertical_plane_basis(
        section_config.angle_degrees,
        section_config.tilt_degrees,
    )
    profiles: dict[int, np.ndarray] = {}
    probe_profiles: dict[int, list[np.ndarray]] = {}
    profile_horizontal_spans: dict[int, float] = {}
    maximum_profile_points = 24_000
    for index in active_indices:
        path = files[index]
        _emit(
            progress_callback,
            f"Group {group_name}: fixed-pose section validation {index + 1}/{len(files)} - {path.name}",
        )
        raw_cloud = source_clouds[index] if source_clouds is not None else read_point_cloud(path)
        final_transformation = np.asarray(leveling_transformation) @ np.asarray(transformations[index])
        aligned_cloud = _transform_cloud(raw_cloud, final_transformation, backend)
        view_probe_profiles: list[np.ndarray] = []
        for probe_offset in probe_offsets:
            probe_config = copy.deepcopy(thin_section_config)
            probe_config.plane_offset += probe_offset
            section_cloud = extract_vertical_section(aligned_cloud, probe_config, backend, clean=False)
            probe_points = np.asarray(section_cloud.points, dtype=np.float64)
            probe_points = (
                probe_points[np.isfinite(probe_points).all(axis=1)]
                if len(probe_points)
                else np.empty((0, 3), dtype=np.float64)
            )
            profile = (
                np.column_stack((probe_points @ horizontal, probe_points @ vertical))
                if len(probe_points)
                else np.empty((0, 2), dtype=np.float64)
            )
            if len(profile) > maximum_profile_points:
                sample_indices = np.linspace(
                    0,
                    len(profile) - 1,
                    maximum_profile_points,
                    dtype=np.int64,
                )
                profile = profile[sample_indices]
            view_probe_profiles.append(profile)
        probe_profiles[index] = view_probe_profiles
        display_profile = view_probe_profiles[center_probe_index]
        if not len(display_profile):
            nonempty_probe_indices = [
                probe_index
                for probe_index, probe_profile in enumerate(view_probe_profiles)
                if len(probe_profile)
            ]
            if nonempty_probe_indices:
                display_probe_index = min(
                    nonempty_probe_indices,
                    key=lambda probe_index: abs(probe_index - center_probe_index),
                )
                display_profile = view_probe_profiles[display_probe_index]
        profiles[index] = display_profile
        # Use the union of all probe slices for coverage decisions.  The
        # center probe can be sparse near an occlusion even when adjacent
        # probes still show the complete horizontal extent.
        finite_probe_profiles = [
            probe_profile[np.isfinite(probe_profile).all(axis=1)]
            for probe_profile in view_probe_profiles
            if len(probe_profile)
        ]
        if finite_probe_profiles:
            horizontal_coordinates = np.concatenate(
                [probe_profile[:, 0] for probe_profile in finite_probe_profiles]
            )
            profile_horizontal_spans[index] = (
                float(np.ptp(horizontal_coordinates))
                if len(horizontal_coordinates) > 1
                else 0.0
            )
        else:
            profile_horizontal_spans[index] = 0.0

    pair_metrics: dict[str, dict[str, object]] = {}
    probe_metrics: dict[str, list[dict[str, object]]] = {}
    for source_position, source_index in enumerate(active_indices):
        for target_index in active_indices[source_position + 1 :]:
            pair_quality = _multi_probe_profile_alignment_metrics(
                probe_profiles[source_index],
                probe_profiles[target_index],
                probe_offsets,
                center_probe_index,
                registration_config.section_alignment_overlap_distance,
                registration_config.section_alignment_minimum_overlap_points,
                registration_config.section_alignment_max_p90_distance,
            )
            pair_key = f"{source_index}:{target_index}"
            pair_metrics[pair_key] = pair_quality
            probe_metrics[pair_key] = list(pair_quality.get("probe_metrics", []))

    reference_index = next(
        (
            index
            for index in active_indices
            if metrics[index].pose_prior_label == "reference" and len(profiles[index])
        ),
        max(active_indices, key=lambda index: len(profiles[index])),
    )

    def pair_for(left: int, right: int) -> dict[str, object] | None:
        if left == right:
            return None
        key = f"{min(left, right)}:{max(left, right)}"
        return pair_metrics.get(key)

    def directional_quality(source_index: int, target_index: int) -> dict[str, object] | None:
        pair_quality = pair_for(source_index, target_index)
        if pair_quality is None:
            return None
        direction_key = "forward" if source_index < target_index else "reverse"
        direction_quality = pair_quality.get(direction_key)
        if not isinstance(direction_quality, dict):
            return None
        enriched_quality = dict(direction_quality)
        enriched_quality["mutual_compatible"] = bool(pair_quality.get("mutual_compatible"))
        enriched_quality["persistent_mismatch_evidence"] = bool(
            pair_quality.get("persistent_mismatch_evidence", False)
        )
        enriched_quality["persistent_extra_contour_evidence"] = bool(
            pair_quality.get("persistent_extra_contour_evidence", False)
        )
        enriched_quality["abnormal_evidence_count"] = int(
            pair_quality.get("abnormal_evidence_count", 0)
        )
        enriched_quality["mismatch_probe_indices"] = list(
            pair_quality.get("mismatch_probe_indices", [])
        )
        enriched_quality["extra_contour_probe_indices"] = list(
            pair_quality.get("extra_contour_probe_indices", [])
        )
        return enriched_quality

    trusted_indices: set[int] = {reference_index}
    while True:
        accepted_candidates: list[int] = []
        for candidate_index in active_indices:
            if candidate_index in trusted_indices:
                continue
            if any(
                quality is not None and bool(quality.get("mutual_compatible"))
                for trusted_index in trusted_indices
                if (quality := directional_quality(candidate_index, trusted_index)) is not None
            ):
                accepted_candidates.append(candidate_index)
        if not accepted_candidates:
            break
        trusted_indices.update(accepted_candidates)

    compatibility_neighbors: dict[int, set[int]] = {
        index: set() for index in active_indices
    }
    for source_position, source_index in enumerate(active_indices):
        for target_index in active_indices[source_position + 1 :]:
            quality = directional_quality(source_index, target_index)
            if quality is not None and bool(quality.get("mutual_compatible")):
                compatibility_neighbors[source_index].add(target_index)
                compatibility_neighbors[target_index].add(source_index)

    peer_component_by_index: dict[int, set[int]] = {}
    untrusted_indices = set(active_indices) - trusted_indices
    while untrusted_indices:
        seed_index = untrusted_indices.pop()
        component = {seed_index}
        pending_indices = [seed_index]
        while pending_indices:
            current_index = pending_indices.pop()
            adjacent_indices = (
                compatibility_neighbors[current_index] & untrusted_indices
            )
            if adjacent_indices:
                untrusted_indices.difference_update(adjacent_indices)
                component.update(adjacent_indices)
                pending_indices.extend(adjacent_indices)
        if len(component) >= 2:
            for component_index in component:
                peer_component_by_index[component_index] = component

    minimum_retained_views = min(
        registration_config.section_alignment_minimum_views,
        len(active_indices),
    )
    peer_components = sorted(
        {
            tuple(sorted(component))
            for component in peer_component_by_index.values()
        }
    )
    isolated_peer_component_by_index: dict[int, dict[str, object]] = {}
    protected_peer_component_by_index: dict[int, dict[str, object]] = {}
    isolated_peer_components: list[dict[str, object]] = []
    protected_peer_components: list[dict[str, object]] = []
    for component in peer_components:
        anomaly = _isolated_peer_component_anomaly(
            component,
            sorted(trusted_indices),
            pair_metrics,
            len(active_indices),
            minimum_retained_views,
            profile_horizontal_spans,
        )
        if anomaly is None:
            continue
        if bool(anomaly.get("local_coverage_protected", False)):
            protected_peer_components.append(anomaly)
            for component_index in component:
                protected_peer_component_by_index[component_index] = anomaly
            continue
        isolated_peer_components.append(anomaly)
        for component_index in component:
            isolated_peer_component_by_index[component_index] = anomaly

    def best_quality(left: int, candidates: Sequence[int]) -> dict[str, object] | None:
        qualities = [
            quality
            for candidate in candidates
            if (quality := directional_quality(left, candidate)) is not None
        ]
        if not qualities:
            return None
        return min(
            qualities,
            key=lambda quality: (
                0 if bool(quality.get("compatible")) else 1,
                float(quality["p90_distance"])
                if quality.get("p90_distance") is not None
                else math.inf,
                float(quality["all_p90_distance"])
                if quality.get("all_p90_distance") is not None
                else math.inf,
                -int(quality.get("overlap_points", 0)),
            ),
        )

    minimum_evidence_points = max(
        8,
        min(30, registration_config.section_alignment_minimum_overlap_points),
    )
    view_reports: list[dict[str, object]] = []
    trusted_profiles = [profiles[index] for index in trusted_indices if len(profiles[index])]
    for index in active_indices:
        metric = metrics[index]
        profile = profiles[index]
        comparison_indices = sorted(trusted_indices - {index})
        supporting_indices = [
            other
            for other in comparison_indices
            if (quality := directional_quality(index, other)) is not None
            and bool(quality.get("mutual_compatible"))
        ]
        peer_component = peer_component_by_index.get(index, set())
        peer_supporting_indices = sorted(
            compatibility_neighbors[index] & peer_component
        )
        if not supporting_indices and peer_supporting_indices:
            supporting_indices = peer_supporting_indices
        has_peer_supported_coverage = bool(peer_supporting_indices)
        peer_component_anomaly = isolated_peer_component_by_index.get(index)
        peer_component_protection = protected_peer_component_by_index.get(index)
        has_isolated_peer_component_anomaly = peer_component_anomaly is not None
        quality = best_quality(index, comparison_indices)
        evidence_qualities = [
            (other, candidate_quality)
            for other in comparison_indices
            if (candidate_quality := directional_quality(index, other)) is not None
            and int(candidate_quality.get("valid_probe_count", 0)) > 0
        ]
        mismatch_evidence_indices = [
            other
            for other, candidate_quality in evidence_qualities
            if bool(candidate_quality.get("persistent_mismatch_evidence", False))
        ]
        extra_contour_evidence_indices = [
            other
            for other, candidate_quality in evidence_qualities
            if bool(candidate_quality.get("persistent_extra_contour_evidence", False))
        ]
        has_strong_mismatch = any(
            other in mismatch_evidence_indices
            and float(candidate_quality.get("common_profile_ratio", 0.0))
            >= SECTION_ALIGNMENT_STRONG_MISMATCH_COVERAGE_RATIO
            for other, candidate_quality in evidence_qualities
        )
        has_consensus_mismatch = len(mismatch_evidence_indices) >= 2
        has_persistent_extra_contour = (
            len(extra_contour_evidence_indices) >= SECTION_ALIGNMENT_MINIMUM_EXTRA_CONTOUR_SUPPORT
            or any(
                other in extra_contour_evidence_indices
                and int(candidate_quality.get("extra_contour_probe_count", 0))
                >= SECTION_ALIGNMENT_PROBE_PERSISTENCE_MINIMUM
                for other, candidate_quality in evidence_qualities
            )
        )
        outside_ratio = _profile_outside_trusted_span(
            profile,
            trusted_profiles,
            registration_config.section_alignment_overlap_distance,
        )
        if index == reference_index:
            status = "reference"
            metric.section_alignment_passed = True
        elif has_persistent_extra_contour:
            status = "mismatch"
            metric.section_alignment_passed = False
            if not coverage_preserving_fusion:
                metric.included_in_fusion = False
                metric.exclusion_reason = "section_extra_contour"
            _emit(
                progress_callback,
                f"Group {group_name}: {('diagnostic only for ' if coverage_preserving_fusion else 'excluded ')}"
                f"{metric.file_name}; persistent extra contour in shared section coverage",
            )
        elif has_isolated_peer_component_anomaly:
            status = "mismatch"
            metric.section_alignment_passed = False
            if not coverage_preserving_fusion:
                metric.included_in_fusion = False
                metric.exclusion_reason = "section_peer_component"
            _emit(
                progress_callback,
                f"Group {group_name}: {('diagnostic only for ' if coverage_preserving_fusion else 'excluded ')}"
                f"{metric.file_name}; isolated peer cluster has persistent multi-view contour mismatch",
            )
        elif has_peer_supported_coverage:
            status = "peer_supported_coverage"
            metric.section_alignment_passed = True
            _emit(
                progress_callback,
                f"Group {group_name}: retained {metric.file_name}; "
                f"mutually corroborated coverage with {len(peer_component)} views",
            )
        elif has_strong_mismatch or has_consensus_mismatch:
            status = "mismatch"
            metric.section_alignment_passed = False
            if not coverage_preserving_fusion:
                metric.included_in_fusion = False
                metric.exclusion_reason = "section_alignment"
            _emit(
                progress_callback,
                f"Group {group_name}: {('diagnostic only for ' if coverage_preserving_fusion else 'excluded ')}"
                f"{metric.file_name}; persistent mismatch in shared section coverage",
            )
        elif index in trusted_indices:
            status = "pass"
            metric.section_alignment_passed = True
        elif len(profile) < minimum_evidence_points:
            status = "insufficient_evidence"
            metric.section_alignment_passed = None
        else:
            status = "coverage_only"
            metric.section_alignment_passed = None

        metric.section_points = int(len(profile))
        metric.section_supporting_views = len(supporting_indices)
        metric.section_probe_count = len(probe_offsets)
        metric.section_validation_status = status
        if quality is None:
            metric.section_valid_probe_count = 0
            metric.section_compatible_probe_count = 0
            metric.section_mismatch_probe_count = 0
            metric.section_extra_contour_probe_count = 0
            metric.section_abnormal_evidence_count = 0
            metric.section_overlap_points = 0
            metric.section_overlap_ratio = 0.0
            metric.section_median_distance = None
            metric.section_local_p75_distance = None
            metric.section_p90_distance = None
            metric.section_shared_span = None
            metric.section_bad_bin_ratio = None
            metric.section_bad_bin_run = None
        else:
            metric.section_valid_probe_count = int(quality.get("valid_probe_count", 0))
            metric.section_compatible_probe_count = int(
                quality.get("compatible_probe_count", 0)
            )
            metric.section_mismatch_probe_count = int(
                quality.get("mismatch_probe_count", 0)
            )
            metric.section_extra_contour_probe_count = int(
                quality.get("extra_contour_probe_count", 0)
            )
            metric.section_abnormal_evidence_count = int(
                quality.get("abnormal_evidence_count", 0)
            )
            metric.section_overlap_points = int(quality["overlap_points"])
            metric.section_overlap_ratio = float(quality["overlap_ratio"])
            metric.section_median_distance = (
                float(quality["median_distance"])
                if quality["median_distance"] is not None
                else None
            )
            metric.section_local_p75_distance = (
                float(quality["local_p75_distance"])
                if quality["local_p75_distance"] is not None
                else None
            )
            metric.section_p90_distance = (
                float(quality["p90_distance"])
                if quality["p90_distance"] is not None
                else None
            )
            metric.section_shared_span = float(quality["common_horizontal_span"])
            metric.section_bad_bin_ratio = (
                float(quality["bad_bin_ratio"])
                if quality["bad_bin_ratio"] is not None
                else None
            )
            metric.section_bad_bin_run = int(quality["largest_bad_bin_run"])
            metric.section_duplicate_contour_ratio = float(
                quality.get("duplicate_contour_ratio", 0.0)
            )
            metric.section_missing_contour_ratio = float(
                quality.get("missing_contour_ratio", 0.0)
            )
            metric.section_consensus_score = float(
                max(
                    0.0,
                    1.0
                    - max(
                        float(quality.get("duplicate_contour_ratio", 0.0)),
                        float(quality.get("missing_contour_ratio", 0.0)),
                        float(quality.get("bad_bin_ratio") or 0.0),
                    ),
                )
            )
        selected_quality = quality or {}
        mismatch_probe_indices = sorted(
            {
                int(probe_index)
                for _, candidate_quality in evidence_qualities
                for probe_index in candidate_quality.get("mismatch_probe_indices", [])
            }
        )
        extra_contour_probe_indices = sorted(
            {
                int(probe_index)
                for _, candidate_quality in evidence_qualities
                for probe_index in candidate_quality.get("extra_contour_probe_indices", [])
            }
        )
        view_reports.append(
            {
                "index": index,
                "file_name": metric.file_name,
                "pose_prior_label": metric.pose_prior_label,
                "profile_points": int(len(profile)),
                "supporting_views": len(supporting_indices),
                "supporting_indices": supporting_indices,
                "peer_supported_coverage": has_peer_supported_coverage,
                "peer_coverage_indices": sorted(peer_component),
                "peer_component_rejected": has_isolated_peer_component_anomaly,
                "peer_component_rejection": peer_component_anomaly,
                "peer_component_protected": peer_component_protection is not None,
                "peer_component_protection": peer_component_protection,
                "outside_trusted_span_ratio": outside_ratio,
                "shared_horizontal_span": metric.section_shared_span,
                "bad_bin_ratio": metric.section_bad_bin_ratio,
                "bad_bin_run": metric.section_bad_bin_run,
                "duplicate_contour_ratio": metric.section_duplicate_contour_ratio,
                "missing_contour_ratio": metric.section_missing_contour_ratio,
                "consensus_score": metric.section_consensus_score,
                "mismatch_evidence_indices": mismatch_evidence_indices,
                "extra_contour_evidence_indices": extra_contour_evidence_indices,
                "probe_count": len(probe_offsets),
                "valid_probe_count": int(selected_quality.get("valid_probe_count", 0)),
                "compatible_probe_count": int(
                    selected_quality.get("compatible_probe_count", 0)
                ),
                "mismatch_probe_count": int(
                    selected_quality.get("mismatch_probe_count", 0)
                ),
                "extra_contour_probe_count": int(
                    selected_quality.get("extra_contour_probe_count", 0)
                ),
                "abnormal_evidence_count": int(
                    selected_quality.get("abnormal_evidence_count", 0)
                ),
                "mismatch_probe_indices": mismatch_probe_indices,
                "mismatch_probe_offsets": [
                    float(probe_offsets[probe_index])
                    for probe_index in mismatch_probe_indices
                    if 0 <= probe_index < len(probe_offsets)
                ],
                "extra_contour_probe_indices": extra_contour_probe_indices,
                "extra_contour_probe_offsets": [
                    float(probe_offsets[probe_index])
                    for probe_index in extra_contour_probe_indices
                    if 0 <= probe_index < len(probe_offsets)
                ],
                "persistent_mismatch": any(
                    bool(candidate_quality.get("persistent_mismatch_evidence", False))
                    for _, candidate_quality in evidence_qualities
                ),
                "persistent_extra_contour": has_persistent_extra_contour,
                "status": status,
                "included_in_fusion": bool(metric.included_in_fusion),
            }
        )

    mismatch_evidence_counts = {
        int(view["index"]): len(
            set(view.get("mismatch_evidence_indices", []))
            | set(view.get("extra_contour_evidence_indices", []))
        )
        for view in view_reports
        if isinstance(view, Mapping) and isinstance(view.get("index"), (int, np.integer))
    }
    consensus_active_indices = [
        index for index in active_indices if metrics[index].included_in_fusion
    ]
    section_consensus = _apply_section_consensus_station_filter(
        profiles,
        consensus_active_indices,
        metrics,
        reference_index,
        registration_config,
        mismatch_evidence_counts,
        progress_callback,
        group_name,
        conservative=conservative,
        apply_filter=not coverage_preserving_fusion,
    )
    consensus_by_index = {
        int(view["index"]): view
        for view in section_consensus.get("views", [])
        if isinstance(view, Mapping) and isinstance(view.get("index"), (int, np.integer))
    }
    for view in view_reports:
        if not isinstance(view, dict):
            continue
        index = int(view["index"])
        metric = metrics[index]
        view["status"] = metric.section_validation_status
        view["included_in_fusion"] = bool(metric.included_in_fusion)
        view["exclusion_reason"] = metric.exclusion_reason
        consensus_view = consensus_by_index.get(index)
        if consensus_view is not None:
            view["section_consensus_removed_points"] = consensus_view.get(
                "consensus_removed_points", 0
            )
            view["section_consensus_removed_ratio"] = consensus_view.get(
                "consensus_removed_ratio", 0.0
            )
            view["section_consensus_shared_removed_ratio"] = consensus_view.get(
                "consensus_shared_removed_ratio", 0.0
            )
            view["section_consensus_evaluated_points"] = consensus_view.get(
                "consensus_evaluated_points", 0
            )
            view["section_consensus_evaluated_bins"] = consensus_view.get(
                "consensus_evaluated_bins", 0
            )
            view["section_consensus_score"] = metric.section_consensus_score

    report["reference_index"] = reference_index
    report["trusted_indices"] = [
        index
        for index in active_indices
        if metrics[index].included_in_fusion
        and metrics[index].section_alignment_passed is True
    ]
    report["profiles"] = profiles
    report["profile_horizontal_spans"] = profile_horizontal_spans
    report["probe_profiles"] = probe_profiles
    report["pair_metrics"] = pair_metrics
    report["probe_metrics"] = probe_metrics
    report["probe_offsets"] = [float(offset) for offset in probe_offsets]
    report["center_probe_index"] = center_probe_index
    report["probe_count"] = len(probe_offsets)
    report["views"] = view_reports
    report["isolated_peer_components"] = isolated_peer_components
    report["protected_peer_components"] = protected_peer_components
    report["section_consensus"] = section_consensus
    report["coverage_preserving_fusion"] = coverage_preserving_fusion
    report["fusion_filter_applied"] = bool(section_consensus.get("filter_applied", True))
    report["minimum_evidence_points"] = minimum_evidence_points
    report["minimum_retained_views"] = min(
        registration_config.section_alignment_minimum_views,
        len(active_indices),
    )
    return report


def _section_plot_limits(
    coordinates: np.ndarray,
    fallback_minimum: float,
    fallback_maximum: float,
) -> tuple[float, float]:
    finite_coordinates = np.asarray(coordinates, dtype=np.float64)
    finite_coordinates = finite_coordinates[np.isfinite(finite_coordinates)]
    if not len(finite_coordinates):
        return float(fallback_minimum), float(fallback_maximum)

    if len(finite_coordinates) >= 1_000:
        lower_quantile, upper_quantile = np.quantile(finite_coordinates, (0.002, 0.998))
        minimum = float(lower_quantile)
        maximum = float(upper_quantile)
        if maximum - minimum <= 1e-8:
            minimum = float(finite_coordinates.min())
            maximum = float(finite_coordinates.max())
    else:
        minimum = float(finite_coordinates.min())
        maximum = float(finite_coordinates.max())
    span = maximum - minimum
    padding = max(span * 0.08, 0.10) if span > 1e-8 else max(abs(minimum) * 0.05, 0.50)
    return minimum - padding, maximum + padding


def render_section_image(
    cloud: o3d.geometry.PointCloud,
    config: SectionConfig,
    output_path: Path,
    title_label: str | None = None,
) -> None:
    """保存三维法向横截面的正视 2D PNG，同时保留输入点的颜色。"""
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    _, horizontal, vertical = vertical_plane_basis(config.angle_degrees, config.tilt_degrees)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    horizontal_limits = (config.horizontal_min, config.horizontal_max)
    vertical_limits = (config.vertical_min, config.vertical_max)

    if len(points):
        horizontal_coordinate = points @ horizontal
        vertical_coordinate = points @ vertical
        display_mask = _section_display_contour_mask(
            horizontal_coordinate,
            vertical_coordinate,
        )
        horizontal_coordinate = horizontal_coordinate[display_mask]
        vertical_coordinate = vertical_coordinate[display_mask]
        if len(colors) == len(points):
            colors = colors[display_mask]
        horizontal_limits = _section_plot_limits(
            horizontal_coordinate,
            config.horizontal_min,
            config.horizontal_max,
        )
        vertical_limits = _section_plot_limits(
            vertical_coordinate,
            config.vertical_min,
            config.vertical_max,
        )
        rendered_colors: np.ndarray | str
        rendered_colors = np.clip(colors, 0.0, 1.0) if len(colors) == len(points) else vertical_coordinate
        # Keep sparse but valid measurements visible without fabricating or
        # resampling points.  The configured value remains available for
        # larger markers, while old configs below this floor get a readable
        # export size.
        render_point_size = max(float(config.point_size), 2.2)
        axis.scatter(
            horizontal_coordinate,
            vertical_coordinate,
            c=rendered_colors,
            s=render_point_size,
            linewidths=0,
            rasterized=True,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "当前切割参数未选中任何点\n请调整平面偏移或切片厚度",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=14,
        )

    axis.set_xlim(*horizontal_limits)
    axis.set_ylim(*vertical_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("平面内横向坐标")
    axis.set_ylabel("截面内竖向坐标")
    title_prefix = "三维法向横截面" if title_label is None else f"三维法向横截面（{title_label}）"
    axis.set_title(
        f"{title_prefix} | 方位 {config.angle_degrees:.2f}° | 俯仰 {config.tilt_degrees:.2f}° | "
        f"偏移 {config.plane_offset:.3f} | 厚度 {config.thickness:.3f}"
    )
    axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _section_display_contour_mask(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    *,
    bin_size: float = 0.05,
    cluster_gap: float = 0.025,
    maximum_layer_separation: float = 0.25,
) -> np.ndarray:
    """Thin only close parallel display layers; keep the measurement cloud intact."""
    horizontal = np.asarray(horizontal, dtype=np.float64)
    vertical = np.asarray(vertical, dtype=np.float64)
    if len(horizontal) != len(vertical):
        raise ValueError("截面渲染坐标数量必须一致。")
    keep = np.ones(len(horizontal), dtype=bool)
    if len(horizontal) < 2:
        return keep

    origin = float(np.floor(np.min(horizontal) / bin_size) * bin_size)
    bin_indices = np.floor((horizontal - origin) / bin_size).astype(np.int64)
    for bin_index in np.unique(bin_indices):
        members = np.flatnonzero(bin_indices == bin_index)
        ordered = members[np.argsort(vertical[members], kind="stable")]
        split_positions = np.flatnonzero(
            np.diff(vertical[ordered]) > cluster_gap
        ) + 1
        clusters = [
            cluster for cluster in np.split(ordered, split_positions)
            if len(cluster) >= 2
        ]
        if len(clusters) < 2:
            continue
        medians = np.asarray(
            [np.median(vertical[cluster]) for cluster in clusters],
            dtype=np.float64,
        )
        if float(np.max(medians) - np.min(medians)) > maximum_layer_separation:
            continue
        selected = max(clusters, key=len)
        keep[np.concatenate([cluster for cluster in clusters if cluster is not selected])] = False
    return keep


def _centered_section_probe_offsets(
    center_offset: float,
    probe_count: int,
    probe_spacing: float,
) -> list[float]:
    """Return an odd, center-inclusive set of independent parallel sections."""
    try:
        normalized_count = int(probe_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("多横截面诊断数量必须是整数。") from exc
    try:
        normalized_spacing = float(probe_spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError("多横截面诊断间距必须是数字。") from exc
    if normalized_count < 1 or normalized_count % 2 == 0:
        raise ValueError("多横截面诊断数量必须是正奇数。")
    if not np.isfinite(normalized_spacing) or normalized_spacing <= 0:
        raise ValueError("多横截面诊断间距必须大于 0。")
    if not np.isfinite(float(center_offset)):
        raise ValueError("多横截面诊断中心偏移必须是有限数值。")
    radius = normalized_count // 2
    return [
        float(center_offset + (index - radius) * normalized_spacing)
        for index in range(normalized_count)
    ]


def render_multi_section_audit(
    cloud: o3d.geometry.PointCloud,
    config: SectionConfig,
    output_path: Path,
    *,
    probe_count: int = 5,
    probe_spacing: float = 0.50,
    maximum_points_per_section: int = 60_000,
) -> dict[str, object]:
    """Render several independent parallel sections for coverage diagnosis.

    The returned metrics and image are diagnostic only.  The regular final
    section remains the sole CAD measurement input; sections are never merged
    across different longitudinal positions.
    """
    if maximum_points_per_section < 1:
        raise ValueError("多横截面诊断每片最大点数必须大于 0。")
    offsets = _centered_section_probe_offsets(
        float(config.plane_offset),
        probe_count,
        probe_spacing,
    )
    points = np.asarray(cloud.points, dtype=np.float64)
    finite_mask = (
        points.ndim == 2
        and points.shape[1] >= 3
        and np.isfinite(points[:, :3]).all(axis=1)
    ) if len(points) else np.zeros(0, dtype=bool)
    finite_points = points[finite_mask, :3] if len(points) else np.empty((0, 3), dtype=np.float64)
    colors = np.asarray(cloud.colors, dtype=np.float64)
    finite_colors = (
        colors[finite_mask]
        if len(colors) == len(points) and len(points)
        else None
    )
    normal, horizontal, vertical = vertical_plane_basis(
        config.angle_degrees,
        config.tilt_degrees,
    )
    signed_distances = finite_points @ normal if len(finite_points) else np.empty(0, dtype=np.float64)
    horizontal_coordinates = finite_points @ horizontal if len(finite_points) else np.empty(0, dtype=np.float64)
    vertical_coordinates = finite_points @ vertical if len(finite_points) else np.empty(0, dtype=np.float64)
    in_range = (
        (horizontal_coordinates >= config.horizontal_min)
        & (horizontal_coordinates <= config.horizontal_max)
        & (vertical_coordinates >= config.vertical_min)
        & (vertical_coordinates <= config.vertical_max)
    ) if len(finite_points) else np.zeros(0, dtype=bool)

    rows = scan_section_offsets(
        finite_points,
        config,
        center_offset=config.plane_offset,
        half_range=abs(offsets[-1] - offsets[0]) / 2.0,
        step=float(probe_spacing),
    ) if len(finite_points) or offsets else []
    # ``scan_section_offsets`` uses a regular grid; map its metrics back to the
    # exact offsets used for rendering so small floating-point differences do
    # not change the displayed order.
    row_by_offset = {
        round(float(row["plane_offset"]), 9): dict(row)
        for row in rows
    }
    rendered_profiles: list[np.ndarray] = []
    rendered_colors: list[np.ndarray | None] = []
    normalized_rows: list[dict[str, object]] = []
    for index, offset in enumerate(offsets):
        mask = in_range & (
            np.abs(signed_distances - float(offset)) <= float(config.thickness) / 2.0
        )
        profile = np.column_stack(
            (horizontal_coordinates[mask], vertical_coordinates[mask])
        ) if np.any(mask) else np.empty((0, 2), dtype=np.float64)
        profile_colors = finite_colors[mask] if finite_colors is not None and np.any(mask) else None
        if len(profile) > maximum_points_per_section:
            sample_indices = np.linspace(
                0,
                len(profile) - 1,
                maximum_points_per_section,
                dtype=np.int64,
            )
            profile = profile[sample_indices]
            if profile_colors is not None:
                profile_colors = profile_colors[sample_indices]
        rendered_profiles.append(profile)
        rendered_colors.append(profile_colors)
        row = row_by_offset.get(round(float(offset), 9))
        if row is None:
            row = {
                "plane_offset": float(offset),
                "point_count": int(np.count_nonzero(mask)),
                "component_count": 0,
                "cell_count": 0,
                "points_per_cell": 0.0,
                "horizontal_span": float(np.ptp(profile[:, 0])) if len(profile) > 1 else 0.0,
                "vertical_span": float(np.ptp(profile[:, 1])) if len(profile) > 1 else 0.0,
                "largest_component_span": 0.0,
                "horizontal_coverage_ratio": 0.0,
                "continuity_ratio": 0.0,
                "quality_score": 0.0,
                "parallel_layer_ratio": 0.0,
            }
        else:
            row = dict(row)
            row["plane_offset"] = float(offset)
        row.update(
            {
                "probe_index": int(index),
                "is_center": bool(index == len(offsets) // 2),
                "display_point_count": int(len(profile)),
                "diagnostic_status": (
                    "no_points"
                    if int(row.get("point_count", 0) or 0) == 0
                    else "split_structure"
                    if int(row.get("component_count", 0) or 0) > 1
                    else "single_structure"
                ),
            }
        )
        normalized_rows.append(row)

    stable_mask = [
        int(row.get("point_count", 0) or 0) >= SECTION_INTEGRITY_MINIMUM_POINTS
        and int(row.get("component_count", 0) or 0) <= 1
        and float(row.get("parallel_layer_ratio", 0.0) or 0.0) <= SECTION_OFFSET_MAX_LAYER_RATIO
        for row in normalized_rows
    ]
    maximum_stable_run = 0
    current_run = 0
    for stable in stable_mask:
        current_run = current_run + 1 if stable else 0
        maximum_stable_run = max(maximum_stable_run, current_run)
    center_index = len(offsets) // 2
    center_row = normalized_rows[center_index]
    if maximum_stable_run >= 3:
        diagnostic_status = "stable_neighbor_sections"
        diagnostic_message = "相邻多个横截面结构稳定，可用来确认目标位置；正式 CAD 仍只使用选定中心截面。"
    elif stable_mask[center_index]:
        diagnostic_status = "center_only_stable"
        diagnostic_message = "中心横截面暂时可用，但相邻截面证据不足，建议结合工位覆盖继续复核。"
    elif any(stable_mask):
        diagnostic_status = "neighbor_only_stable"
        diagnostic_message = "只有邻近横截面较完整，中心位置存在缺口；不能直接把邻近截面当作中心测量截面。"
    else:
        diagnostic_status = "persistent_gap"
        diagnostic_message = "多个横截面都未形成稳定单一轮廓，优先重新采集覆盖缺口区域的视角。"

    all_profiles = [profile for profile in rendered_profiles if len(profile)]
    if all_profiles:
        combined_profile = np.concatenate(all_profiles, axis=0)
        horizontal_limits = _section_plot_limits(
            combined_profile[:, 0], config.horizontal_min, config.horizontal_max
        )
        vertical_limits = _section_plot_limits(
            combined_profile[:, 1], config.vertical_min, config.vertical_max
        )
    else:
        horizontal_limits = (config.horizontal_min, config.horizontal_max)
        vertical_limits = (config.vertical_min, config.vertical_max)

    column_count = min(4, max(1, len(offsets)))
    row_count = int(math.ceil(len(offsets) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.0 * column_count, 4.2 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    axis_list = list(axes.flat)
    for index, (axis, profile, profile_colors, row) in enumerate(
        zip(
            axis_list[: len(offsets)],
            rendered_profiles,
            rendered_colors,
            normalized_rows,
            strict=True,
        )
    ):
        if len(profile):
            if profile_colors is not None and len(profile_colors) == len(profile):
                display_colors: np.ndarray | str = np.clip(profile_colors, 0.0, 1.0)
            else:
                display_colors = "#334e68"
            axis.scatter(
                profile[:, 0],
                profile[:, 1],
                c=display_colors,
                s=1.0,
                linewidths=0,
                rasterized=True,
            )
        axis.set_xlim(*horizontal_limits)
        axis.set_ylim(*vertical_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.22)
        axis.set_xlabel("平面内横向坐标")
        axis.set_ylabel("截面内竖向坐标")
        status = str(row.get("diagnostic_status", "unknown"))
        axis.set_title(
            f"{'中心' if row.get('is_center') else f'截面 {index + 1}'} | 偏移 {float(row['plane_offset']):+.3f}\n"
            f"{int(row.get('point_count', 0) or 0):,} 点 | "
            f"{int(row.get('component_count', 0) or 0)} 结构 | "
            f"跨度 {float(row.get('largest_component_span', 0.0) or 0.0):.2f}\n"
            f"{status}",
            color="#b42318" if status in {"split_structure", "no_points"} else "#176b3a",
            fontsize=10,
        )
    for axis in axis_list[len(offsets) :]:
        axis.axis("off")
    figure.suptitle(
        "多横截面诊断（各平面独立，不合并为一个 CAD 轮廓）\n"
        f"中心偏移 {config.plane_offset:+.3f} | 间距 {float(probe_spacing):.3f} | "
        f"{diagnostic_message}",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)

    return {
        "enabled": True,
        "method": "parallel_cross_section_diagnostic",
        "probe_count": len(offsets),
        "probe_spacing": float(probe_spacing),
        "center_index": center_index,
        "center_offset": float(config.plane_offset),
        "diagnostic_status": diagnostic_status,
        "diagnostic_message": diagnostic_message,
        "stable_slice_count": int(sum(stable_mask)),
        "maximum_stable_run": int(maximum_stable_run),
        "center_diagnostic_status": center_row.get("diagnostic_status"),
        "formal_measurement_uses_center_only": True,
        "rows": normalized_rows,
        "image_path": str(output_path.resolve()),
    }


def render_top_preview(cloud: o3d.geometry.PointCloud, output_path: Path, maximum_points: int = 180_000) -> None:
    """保存合并点云的顶视缩略图，便于快速检查拼接结果。"""
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) > maximum_points:
        sample_indices = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
        points = points[sample_indices]
        colors = colors[sample_indices] if len(colors) else colors

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    if len(points):
        render_colors: np.ndarray | str
        render_colors = np.clip(colors, 0.0, 1.0) if len(colors) == len(points) else "#2e86c1"
        axis.scatter(points[:, 0], points[:, 1], c=render_colors, s=0.25, linewidths=0, rasterized=True)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_title("合并点云顶视预览")
    axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)


def render_source_alignment_audit(
    clouds: Sequence[o3d.geometry.PointCloud],
    transformations: Sequence[np.ndarray],
    output_path: Path,
    leveling_transformation: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    maximum_points_per_view: int = 35_000,
) -> None:
    """Render source-colored projections to audit whether every scan uses the same pose."""
    if len(clouds) != len(transformations):
        raise ValueError('彩色对齐审计的点云和位姿数量必须一致。')
    if labels is not None and len(labels) != len(clouds):
        raise ValueError('彩色对齐审计的标签数量必须与点云数量一致。')

    palette = ('#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#00bfc4')
    leveling = (
        np.asarray(leveling_transformation, dtype=np.float64)
        if leveling_transformation is not None
        else np.eye(4, dtype=np.float64)
    )
    random_generator = np.random.default_rng(240826)
    aligned_points: list[np.ndarray] = []
    for cloud, transformation in zip(clouds, transformations):
        points = np.asarray(cloud.points, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > maximum_points_per_view:
            sample_indices = random_generator.choice(
                len(points),
                size=maximum_points_per_view,
                replace=False,
            )
            points = points[sample_indices]
        transformed = points @ np.asarray(transformation)[:3, :3].T + np.asarray(transformation)[:3, 3]
        transformed = transformed @ leveling[:3, :3].T + leveling[:3, 3]
        aligned_points.append(transformed)

    figure = plt.figure(figsize=(17, 13), constrained_layout=True)
    axis_xy = figure.add_subplot(2, 2, 1)
    axis_xz = figure.add_subplot(2, 2, 2)
    axis_yz = figure.add_subplot(2, 2, 3)
    axis_3d = figure.add_subplot(2, 2, 4, projection='3d')
    for index, points in enumerate(aligned_points):
        if not len(points):
            continue
        label = labels[index] if labels is not None else f'view {index + 1}'
        color = palette[index % len(palette)]
        axis_xy.scatter(points[:, 0], points[:, 1], c=color, s=0.18, alpha=0.55, linewidths=0, label=label, rasterized=True)
        axis_xz.scatter(points[:, 0], points[:, 2], c=color, s=0.18, alpha=0.55, linewidths=0, rasterized=True)
        axis_yz.scatter(points[:, 1], points[:, 2], c=color, s=0.18, alpha=0.55, linewidths=0, rasterized=True)
        axis_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=0.18, alpha=0.36, depthshade=False)

    for axis, title, x_label, y_label in (
        (axis_xy, 'XY source overlay', 'X', 'Y'),
        (axis_xz, 'XZ source overlay', 'X', 'Z'),
        (axis_yz, 'YZ source overlay', 'Y', 'Z'),
    ):
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=0.25)
    if aligned_points:
        axis_xy.legend(markerscale=14, loc='best')
    axis_3d.set_title('3D source overlay')
    axis_3d.view_init(elev=20, azim=-135)
    axis_3d.set_xlabel('X')
    axis_3d.set_ylabel('Y')
    axis_3d.set_zlabel('Z')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=210, facecolor='white')
    plt.close(figure)


def _recover_calibration_roi(cloud, reference_cloud, calibration, registration, file_name, backend):
    """Locate a missing native ROI geometrically, without station/file heuristics."""
    initial, fitness, _ = coarse_global_register(cloud, reference_cloud, registration, backend)
    if fitness is None or fitness <= 0:
        raise ValueError(f"{file_name} 无法自动定位移出原始 ROI 的标定块。")
    aligned = _transform_cloud(cloud, initial, backend)
    localized = crop_calibration_cloud(aligned, calibration, file_name, backend)
    return _transform_cloud(localized, np.linalg.inv(initial), backend)


def _disambiguate_global_pose(source, reference, reference_calibration, best,
                             config, refinement_config, file_name, backend,
                             *, reference_views=None, registration_cache=None,
                             feature_cache=None):
    """Resolve fixture symmetries with a full-scene hypothesis, then recheck the fixture."""
    if not config.registration.use_global_pose_validation:
        return best, None
    voxel = config.registration.registration_voxel_size
    source_sample = np.asarray(source.voxel_down_sample(voxel).points)
    reference_sample = np.asarray(reference.voxel_down_sample(voxel).points)
    reference_mask = fixture_component_mask(reference_sample, config.calibration, voxel * 2)
    reference_workpiece = reference_sample[~reference_mask]
    if len(reference_workpiece) < 100:
        return best, None
    target_tree = cKDTree(reference_workpiece)
    gate = config.registration.icp_max_correspondence_distance
    native_fixture = fixture_component_mask(source_sample, config.calibration, voxel * 2)
    if not native_fixture.any():
        initial_points = source_sample @ best[0][:3, :3].T + best[0][:3, 3]
        native_fixture = fixture_component_mask(initial_points, config.calibration, voxel * 2)
    source_workpiece = source_sample[~native_fixture]
    if len(source_workpiece) < 100:
        return best, None
    source_tree = cKDTree(source_workpiece)
    if registration_cache is None:
        registration_cache = {}

    def pose_score(transformation):
        # Keep the evaluated samples fixed across poses. Re-cropping every
        # candidate can reward moving a misaligned workpiece into the ROI.
        points = source_workpiece @ transformation[:3, :3].T + transformation[:3, 3]
        forward = target_tree.query(points)[0]
        # Rigid distances are invariant; query the fixed tree in source coordinates.
        reverse = source_tree.query(
            (reference_workpiece - transformation[:3, 3]) @ transformation[:3, :3])[0]
        return float(np.sqrt(np.mean(forward < gate) * np.mean(reverse < gate))
                     * (1 - .25 * (np.mean(np.minimum(forward, gate))
                                    + np.mean(np.minimum(reverse, gate))) / (2 * gate)))

    baseline = pose_score(best[0])
    fixture_passed, _ = calibration_registration_quality(
        best[5], best[3], best[4], config.calibration, config.registration)
    if (fixture_passed and baseline >= .9 and
            best[6] >= config.registration.minimum_full_cloud_fitness):
        best[8]["global_pose_disambiguation"] = {
            "accepted": False, "quality_before": baseline, "quality_after": baseline,
            "fixture_quality_passed": True, "search": "confident_existing_pose"}
        return best, None

    hypotheses = [coarse_global_register(
        source, reference, config.registration, backend,
        candidate_scorer=pose_score, registration_cache=registration_cache)]
    for peer in reference_views or ():
        hypotheses.append(coarse_global_register(
            source, peer, config.registration, backend,
            candidate_scorer=pose_score, registration_cache=registration_cache))
    if native_fixture.any() and len(source_workpiece) >= 100:
        hypotheses.append(coarse_global_register(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_workpiece)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference_workpiece)),
            config.registration, backend, candidate_scorer=pose_score,
            registration_cache=registration_cache))
    centered_reference = reference_workpiece - reference_workpiece.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(centered_reference.T @ centered_reference)
    if baseline < .75:
        pivot = np.asarray(reference_calibration.get_center())
        alternatives = []
        # Plates do not have a uniquely dominant long axis. Search all intrinsic
        # axes so their fixture symmetries receive the same checks as profiles.
        for axis in eigenvectors.T:
            for angle in np.linspace(-np.pi, np.pi, 12, endpoint=False):
                if abs(angle) < 1e-8:
                    continue
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                correction = np.eye(4)
                correction[:3, :3] = rotation
                correction[:3, 3] = pivot - rotation @ pivot
                proposal = correction @ best[0]
                alternatives.append((pose_score(proposal), proposal))
        for score, proposal in sorted(alternatives, key=lambda item: item[0], reverse=True)[:6]:
            hypotheses.append((proposal, score, 0.0))
    if len(source_workpiece) >= 100:
        source_body = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_workpiece))
        target_body = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference_workpiece))
        # Match Open3D GICP's default normal neighborhood, once for all hypotheses.
        source_body.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        target_body.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        coarse_gate = max(gate, float(np.linalg.norm(np.ptp(reference_workpiece, axis=0))) * .08)
        # A fixture can match a parallel face several voxels away. Refine the
        # global hypotheses over the workpiece before the independent fixture
        # check; this is not the bounded final micro-adjustment.
        refined_seeds = set()
        for initial, _, _ in [*hypotheses, (best[0], best[1], best[2])]:
            seed_key = np.asarray(initial).round(10).tobytes()
            if seed_key in refined_seeds:
                continue
            refined_seeds.add(seed_key)
            transformation = np.array(initial, copy=True)
            for distance in (coarse_gate, max(gate, coarse_gate * .5), gate):
                refined = o3d.pipelines.registration.registration_generalized_icp(
                    source_body, target_body, distance, transformation,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                transformation = np.array(refined.transformation, copy=True)
            if np.isfinite(transformation).all():
                hypotheses.append((transformation, float(refined.fitness), float(refined.inlier_rmse)))
    selected, selected_calibration = best, None
    best_score = baseline
    for initial, global_fitness, global_rmse in hypotheses:
        if global_fitness is None or global_fitness <= 0:
            continue
        aligned = _transform_cloud(source, initial, backend)
        try:
            localized = crop_calibration_cloud(aligned, config.calibration, file_name, backend)
        except ValueError:
            continue
        correction, fitness, rmse, geometry = _refine_with_icp_diagnostics(
            localized, reference_calibration, refinement_config, backend,
            registration_cache, feature_cache)
        passed, _ = calibration_registration_quality(
            len(localized.points), fitness, rmse, config.calibration, config.registration)
        if not passed:
            continue
        transform = correction @ initial
        quality = pose_score(transform)
        if quality <= best_score + .03:
            continue
        full_fitness, full_rmse = evaluate_full_cloud_alignment(
            source, reference, transform, config.registration, backend)
        geometry["global_pose_disambiguation"] = {
            "accepted": True, "quality_before": baseline, "quality_after": quality,
            "fixture_quality_passed": True,
        }
        selected = (transform, global_fitness, global_rmse, fitness, rmse,
                    len(localized.points), full_fitness, full_rmse, geometry)
        selected_calibration = _transform_cloud(localized, np.linalg.inv(initial), backend)
        best_score = quality
    return selected, selected_calibration


def _recover_multiview_poses(cleaned_clouds, transforms, metrics, config,
                            progress_callback, group_name):
    """Revisit weak single-reference poses using the other views' visible faces."""
    result = [np.array(t, copy=True) for t in transforms]
    active = [i for i, metric in enumerate(metrics) if metric.included_in_fusion]
    if (not config.registration.use_global_pose_validation
            or config.registration.use_six_station_pose_prior or len(active) < 3):
        return result, []
    voxel = config.registration.registration_voxel_size
    aligned = {
        i: _transform_cloud(
            cleaned_clouds[i].voxel_down_sample(
                config.registration.calibration_graph_voxel_size), transforms[i], "cpu")
        for i in active
    }
    refinement = copy.deepcopy(config.registration)
    refinement.registration_voxel_size = config.registration.calibration_graph_voxel_size
    refinement.icp_max_correspondence_distance = config.registration.calibration_graph_max_correspondence_distance
    refinement.icp_max_iterations = config.registration.calibration_graph_icp_max_iterations
    rows = []
    # All references use the same snapshot, so a newly recovered view cannot
    # validate another recovery in the same pass.
    for index in range(1, len(metrics)):
        reference = o3d.geometry.PointCloud()
        for peer in active:
            if peer != index:
                reference += aligned[peer]
        reference = reference.voxel_down_sample(refinement.registration_voxel_size)
        target_points = np.asarray(reference.voxel_down_sample(voxel).points)
        target_points = target_points[~fixture_component_mask(
            target_points, config.calibration, voxel * 2)]
        source_points = np.asarray(cleaned_clouds[index].voxel_down_sample(voxel).points)
        mask = fixture_component_mask(source_points, config.calibration, voxel * 2)
        source_points = source_points @ transforms[index][:3, :3].T + transforms[index][:3, 3]
        if not mask.any():
            mask = fixture_component_mask(source_points, config.calibration, voxel * 2)
        source_points = source_points[~mask]
        if min(len(source_points), len(target_points)) < 100:
            continue
        gate = config.registration.icp_max_correspondence_distance
        forward_support = float(np.mean(cKDTree(target_points).query(source_points)[0] < gate))
        reverse_support = float(np.mean(cKDTree(source_points).query(target_points)[0] < gate))
        support = float(np.sqrt(forward_support * reverse_support))
        if (metrics[index].included_in_fusion
                and support >= config.registration.workpiece_graph_minimum_loop_fitness):
            continue
        metric = metrics[index]
        _emit(progress_callback, f"第 {group_name} 组：联合其余视角复核低重叠位姿 - {metric.file_name}")
        try:
            reference_calibration = crop_calibration_cloud(
                reference, config.calibration, "multiview_reference", "cpu")
        except ValueError:
            continue
        best = (transforms[index], metric.global_fitness, metric.global_rmse,
                metric.icp_fitness, metric.icp_rmse, metric.calibration_points,
                metric.full_cloud_fitness, metric.full_cloud_rmse,
                metric.calibration_geometry_report or {})
        selected, localized = _disambiguate_global_pose(
            cleaned_clouds[index], reference, reference_calibration, best,
            config, refinement, metric.file_name, "cpu",
            reference_views=[aligned[peer] for peer in active if peer != index])
        row = {"view_index": index, "support_before": support,
               "forward_support_before": forward_support,
               "reverse_support_before": reverse_support,
               "reference_indices": [peer for peer in active if peer != index],
               "accepted": localized is not None}
        recovered_quality = (selected[8].get("global_pose_disambiguation", {})
                             .get("quality_after", 0.0))
        if localized is not None and (metric.included_in_fusion or
                recovered_quality >= config.registration.workpiece_graph_minimum_loop_fitness):
            result[index] = selected[0]
            (_, metric.global_fitness, metric.global_rmse, metric.icp_fitness,
             metric.icp_rmse, metric.calibration_points,
             metric.full_cloud_fitness, metric.full_cloud_rmse, geometry) = selected
            metric.calibration_geometry_report = _calibration_geometry_metric_report(geometry)
            row.update(geometry["global_pose_disambiguation"])
            metric.calibration_geometry_report["multiview_pose_recovery"] = row.copy()
            if not metric.included_in_fusion:
                metric.included_in_fusion = True
                metric.exclusion_reason = None
                metric.calibration_quality_passed = True
                metric.calibration_quality_reason = None
        else:
            row["accepted"] = False
        rows.append(row)
    return result, rows


def _refine_group_overlap(cleaned_clouds, transforms, metrics, config, progress_callback, group_name):
    transforms, recoveries = _recover_multiview_poses(
        cleaned_clouds, transforms, metrics, config, progress_callback, group_name)
    _emit(progress_callback, f"第 {group_name} 组：执行带退化方向保护的三维重叠配准")
    transforms, report = refine_multiview_overlap(
        cleaned_clouds, transforms, [m.included_in_fusion for m in metrics],
        config.calibration, config.registration,
    )
    report["multiview_pose_recovery"] = recoveries
    for row in report["views"]:
        metrics[row["view_index"]].overlap_refinement_report = row
        _emit(progress_callback, f"第 {group_name} 组：视角 {row['view_index'] + 1} "
              f"三维重叠修正：{row['reason']}")
    for metric, transformation in zip(metrics, transforms, strict=True):
        metric.transformation = np.asarray(transformation).round(12).tolist()
    return transforms, report


def _single_view_denoise_config(config: ProcessingConfig) -> ProcessingConfig:
    """Keep legacy saved settings from enabling cleanup after alignment."""
    effective = copy.deepcopy(config)
    effective.registration.coverage_preserving_fusion = True
    effective.section.use_statistical_filter = False
    effective.section.use_component_filter = False
    return effective


def process_group_with_six_station_pose_prior(
    group_name: str,
    files: Sequence[Path],
    config: ProcessingConfig,
    output_root: Path,
    progress_callback: ProgressCallback | None = None,
    *,
    _timing: ProcessingTiming | None = None,
) -> PipelineResult:
    """Process a fixed five- or six-station scan without symmetric fixture flips."""
    timing = _timing or ProcessingTiming()
    progress_callback = _timed_progress_callback(progress_callback, timing)
    config = _single_view_denoise_config(config)
    config.validate()
    active_backend = resolve_compute_backend(config.compute_backend)
    capabilities = get_gpu_capabilities()
    files, station_labels, station_order_source = order_files_with_six_station_pose_prior_with_source(files)
    transforms = six_station_pose_transformations(
        station_labels,
        config.registration.station_pose_corrections,
    )
    station_pose_prior_transforms = [transformation.copy() for transformation in transforms]
    backend_label = 'GPU' if active_backend == 'gpu' else 'CPU'
    _emit(
        progress_callback,
        f'第 {group_name} 组：使用 {len(files)} 工位姿态先验，'
        f'按 {station_order_source} 恢复站位顺序。',
    )
    if active_backend == 'gpu':
        _emit(progress_callback, f'第 {group_name} 组：CUDA GPU 加速已启用 - {capabilities.device_name}')

    cleaned_clouds, denoise_stats = parallel_denoise_clouds(
        files, config.denoise, active_backend, progress_callback, group_name,
    )

    reference_file = files[0]
    reference_calibration = crop_calibration_cloud(
        cleaned_clouds[0],
        config.calibration,
        reference_file.name,
        active_backend,
    )
    try:
        leveling = estimate_calibration_leveling(reference_calibration, config.calibration)
    except ValueError:
        # Protect sparse calibration faces from the stricter workpiece cleanup.
        reference_calibration = crop_calibration_cloud(
            read_point_cloud(reference_file),
            config.calibration,
            reference_file.name,
            active_backend,
        )
        leveling = estimate_calibration_leveling(reference_calibration, config.calibration)
    metrics: list[RegistrationMetric] = [
        RegistrationMetric(
            file_name=reference_file.name,
            calibration_points=len(reference_calibration.points),
            global_fitness=None,
            global_rmse=None,
            icp_fitness=1.0,
            icp_rmse=0.0,
            transformation=transforms[0].round(12).tolist(),
            full_cloud_fitness=None,
            full_cloud_rmse=None,
            included_in_fusion=True,
            pose_prior_label=station_labels[0],
            calibration_quality_passed=True,
        )
    ]
    aligned_calibrations: list[o3d.geometry.PointCloud] = [reference_calibration]
    calibration_refinement_config = copy.deepcopy(config.registration)
    calibration_refinement_config.registration_voxel_size = min(
        config.registration.pose_prior_calibration_voxel_size,
        config.registration.calibration_graph_voxel_size,
    )
    calibration_refinement_config.icp_max_correspondence_distance = min(
        config.registration.pose_prior_translation_max_correspondence_distance,
        config.registration.calibration_graph_max_correspondence_distance,
    )
    calibration_refinement_config.icp_max_iterations = max(
        300,
        config.registration.calibration_graph_icp_max_iterations,
    )
    registration_cache: dict[str, dict] = {}
    feature_cache: dict[int, tuple[o3d.geometry.PointCloud, list[dict[str, object]]]] = {}

    for index, (source_cloud, path, station_label, initial_transform) in enumerate(
        zip(cleaned_clouds[1:], files[1:], station_labels[1:], transforms[1:], strict=True),
        start=2,
    ):
        _emit(
            progress_callback,
            f'第 {group_name} 组：锁旋转精调 {index}/{len(files)} - {station_label} - {path.name}',
        )
        initially_aligned_cloud = _transform_cloud(source_cloud, initial_transform, active_backend)
        source_calibration = crop_calibration_cloud(
            initially_aligned_cloud,
            config.calibration,
            path.name,
            active_backend,
        )
        target_calibration = reference_calibration
        rigid_correction, rigid_fitness, rigid_rmse, geometry_report = _refine_with_icp_diagnostics(
            source_calibration,
            target_calibration,
            calibration_refinement_config,
            active_backend,
            registration_cache,
            feature_cache,
        )
        bounded_rotation = (
            _rotation_angle_degrees(rigid_correction[:3, :3])
            <= config.registration.pose_prior_rigid_max_rotation_degrees
        )
        bounded_translation = (
            float(np.linalg.norm(rigid_correction[:3, 3]))
            <= config.registration.pose_prior_rigid_max_translation
        )
        if bounded_rotation and bounded_translation:
            final_transform = rigid_correction @ initial_transform
            final_calibration = _transform_cloud(
                source_calibration,
                rigid_correction,
                active_backend,
            )
            icp_fitness, icp_rmse = rigid_fitness, rigid_rmse
        else:
            translation_correction, translation_fitness, translation_rmse = refine_translation_only(
                source_calibration,
                target_calibration,
                np.eye(4, dtype=np.float64),
                calibration_refinement_config.icp_max_correspondence_distance,
            )
            final_transform = translation_correction @ initial_transform
            final_calibration = _transform_cloud(
                source_calibration,
                translation_correction,
                active_backend,
            )
            icp_fitness, icp_rmse = translation_fitness, translation_rmse
        calibration_quality_passed, calibration_quality_reason = calibration_registration_quality(
            len(source_calibration.points),
            icp_fitness,
            icp_rmse,
            config.calibration,
            config.registration,
        )
        included_in_fusion = (
            calibration_quality_passed
            or not config.registration.skip_low_quality_calibration_views
        )
        transforms[index - 1] = final_transform
        aligned_calibrations.append(final_calibration)
        metrics.append(
            RegistrationMetric(
                file_name=path.name,
                calibration_points=len(source_calibration.points),
                global_fitness=None,
                global_rmse=None,
                icp_fitness=icp_fitness,
                icp_rmse=icp_rmse,
                transformation=np.asarray(final_transform).round(12).tolist(),
                full_cloud_fitness=None,
                full_cloud_rmse=None,
                included_in_fusion=included_in_fusion,
                pose_prior_label=station_label,
                calibration_quality_passed=calibration_quality_passed,
                calibration_quality_reason=calibration_quality_reason,
                calibration_geometry_status=str(geometry_report.get("status")),
                calibration_geometry_plane_count=int(
                    geometry_report.get("source_plane_count", 0)
                ),
                calibration_geometry_matched_planes=int(
                    geometry_report.get("matched_plane_count", 0)
                ),
                calibration_geometry_plane_rmse_before=(
                    float(geometry_report["plane_rmse_before"])
                    if geometry_report.get("plane_rmse_before") is not None
                    else None
                ),
                calibration_geometry_plane_rmse_after=(
                    float(geometry_report["plane_rmse_after"])
                    if geometry_report.get("plane_rmse_after") is not None
                    else None
                ),
                calibration_geometry_correction_rotation_degrees=float(
                    geometry_report.get("correction_rotation_degrees", 0.0)
                ),
                calibration_geometry_correction_translation=float(
                    geometry_report.get("correction_translation", 0.0)
                ),
                calibration_geometry_pairwise_rmse=(
                    float(geometry_report["plane_pairwise_rmse"])
                    if geometry_report.get("plane_pairwise_rmse") is not None
                    else None
                ),
                calibration_geometry_report=_calibration_geometry_metric_report(
                    geometry_report
                ),
                exclusion_reason=(
                    None
                    if included_in_fusion
                    else "calibration_icp_quality"
                ),
            )
        )
        if not calibration_quality_passed:
            _emit(
                progress_callback,
                f'第 {group_name} 组：标定块质量未通过 {path.name} - '
                f'{calibration_quality_reason}；'
                f'{"排除融合" if not included_in_fusion else "保留但标记"}',
            )

    _emit(
        progress_callback,
        f'第 {group_name} 组：仅使用多面体标定块建立全工位闭环位姿图（不使用工件或 CAD）',
    )
    calibration_graph_config = copy.deepcopy(config.registration)
    calibration_graph_config.registration_voxel_size = config.registration.calibration_graph_voxel_size
    calibration_graph_config.icp_max_correspondence_distance = (
        config.registration.calibration_graph_max_correspondence_distance
    )
    calibration_graph_config.icp_max_iterations = config.registration.calibration_graph_icp_max_iterations
    aligned_calibration_clouds = [
        crop_calibration_cloud(
            _transform_cloud(cloud, transformation, active_backend),
            config.calibration,
            path.name,
            active_backend,
        )
        for cloud, transformation, path in zip(cleaned_clouds, transforms, files, strict=True)
    ]
    calibration_active_indices = [
        index for index, metric in enumerate(metrics) if metric.included_in_fusion
    ]
    calibration_corrections, calibration_edge_metrics, calibration_node_metrics = (
        build_and_optimize_pairwise_pose_graph(
            aligned_calibration_clouds,
            station_labels,
            calibration_graph_config,
            "calibration",
            maximum_rotation_degrees=config.registration.calibration_graph_max_rotation_degrees,
            maximum_translation=config.registration.calibration_graph_max_translation,
            minimum_adjacent_fitness=config.registration.calibration_graph_minimum_adjacent_fitness,
            minimum_loop_fitness=config.registration.calibration_graph_minimum_loop_fitness,
            backend=active_backend,
            progress_callback=progress_callback,
            group_name=group_name,
            active_indices=calibration_active_indices,
            reference_label="reference",
            maximum_adjacent_rmse=config.registration.calibration_graph_maximum_adjacent_rmse,
            maximum_loop_rmse=config.registration.calibration_graph_maximum_loop_rmse,
            registration_cache=registration_cache,
            feature_cache=feature_cache,
        )
    )
    transforms = [
        correction @ transformation
        for correction, transformation in zip(calibration_corrections, transforms, strict=True)
    ]
    for metric, transformation in zip(metrics, transforms, strict=True):
        metric.transformation = np.asarray(transformation).round(12).tolist()

    transforms, overlap_refinement = _refine_group_overlap(
        cleaned_clouds, transforms, metrics, config, progress_callback, group_name)

    if config.registration.use_section_alignment_validation:
        _emit(progress_callback, f'第 {group_name} 组：执行多截面拼接质量诊断（不剔除互补视角）')
    else:
        _emit(progress_callback, f'第 {group_name} 组：跳过多截面质量诊断，保留全部标定块质量合格工位')
    section_validation = validate_section_view_alignment(
        files,
        transforms,
        metrics,
        leveling.transformation,
        config.section,
        config.registration,
        active_backend,
        progress_callback,
        group_name,
        conservative=True,
        source_clouds=cleaned_clouds,
    )
    included_views = [metric.included_in_fusion for metric in metrics]
    included_count = sum(included_views)
    if not included_count:
        raise ValueError('标定块质量闸门后没有可用于融合的工位，请检查标定块 ROI 和配准参数。')
    minimum_required_views = min(config.registration.section_alignment_minimum_views, len(files))
    if included_count < minimum_required_views:
        raise ValueError(
            f'标定块质量闸门后仅保留 {included_count}/{len(files)} 个工位，'
            f'低于当前流程建议的 {minimum_required_views} 个；'
            '请重新采集或调整标定块 ROI/配准参数。'
        )

    cleaned_clouds, cross_view_denoise = _denoise_registered_views(
        cleaned_clouds, transforms, leveling.transformation, included_views,
        config.denoise.voxel_size, progress_callback, group_name,
    )
    fusion_action = '应用全部标定块质量合格的' if config.registration.coverage_preserving_fusion else (
        '应用通过验收的' if config.registration.use_section_alignment_validation else '应用'
    )
    _emit(
        progress_callback,
        f'第 {group_name} 组：{fusion_action} {included_count}/{len(files)} 个工位并融合（{backend_label}）',
    )
    effective_fusion_voxel_size = min(max(config.fusion_voxel_size, 0.02), 0.035)
    merged_cloud, section_consensus = _merge_clouds_with_section_consensus(
        files,
        transforms,
        included_views,
        leveling.transformation,
        config.section,
        effective_fusion_voxel_size,
        active_backend,
        progress_callback,
        group_name,
        conservative=True,
        coverage_preserving=config.registration.coverage_preserving_fusion,
        section_recovery_voxel_size=config.denoise.voxel_size,
        calibration_config=config.calibration,
        calibration_bounds_in_reference_frame=True,
        source_clouds=cleaned_clouds,
    )
    _annotate_section_consensus_report(section_consensus, metrics)
    fusion_transforms = [
        transformation
        for transformation, included in zip(transforms, included_views, strict=True)
        if included
    ]
    if len(merged_cloud.points) == 0:
        raise ValueError('融合后点云为空，请检查去噪和六工位姿态先验。')

    _emit(progress_callback, f'第 {group_name} 组：从已融合点云提取三维法向横截面')
    raw_section_cloud, section_source_points = extract_section_from_merged_cloud(
        merged_cloud,
        config.section,
        active_backend,
    )
    section_cloud = copy.deepcopy(raw_section_cloud)
    section_integrity = diagnose_section_integrity(
        raw_section_cloud,
        section_cloud,
        config.section,
        source_profiles=section_validation.get("profiles"),
        source_profile_indices=(
            [index for index, included in enumerate(included_views) if included]
            if config.registration.coverage_preserving_fusion
            else section_validation.get("trusted_indices")
        ),
    )
    run_directory = _create_run_directory(output_root, group_name)
    merged_cloud_path = run_directory / 'merged_pointcloud.ply'
    raw_section_cloud_path = run_directory / 'vertical_section_raw.ply'
    section_cloud_path = run_directory / 'vertical_section.ply'
    raw_section_image_path = run_directory / 'vertical_section_raw.png'
    section_image_path = run_directory / 'vertical_section.png'
    top_preview_path = run_directory / 'merged_top_view.png'
    alignment_audit_path = run_directory / 'alignment_audit.png'
    section_alignment_audit_path = run_directory / 'section_alignment_audit.png'
    section_coverage_audit_path = run_directory / 'section_coverage_audit.png'
    multi_section_audit_path = run_directory / 'vertical_sections_multi.png'
    summary_path = run_directory / 'run_summary.json'

    _emit(progress_callback, f'第 {group_name} 组：保存点云、截面和彩色对齐审计图')
    _write_point_cloud(merged_cloud_path, merged_cloud)
    saved_raw_section_path: Path | None = None
    if len(raw_section_cloud.points):
        _write_point_cloud(raw_section_cloud_path, raw_section_cloud)
        saved_raw_section_path = raw_section_cloud_path
    saved_section_path: Path | None = None
    if len(section_cloud.points):
        _write_point_cloud(section_cloud_path, section_cloud)
        saved_section_path = section_cloud_path
    render_section_image(raw_section_cloud, config.section, raw_section_image_path, "融合云基线截面")
    render_section_image(section_cloud, config.section, section_image_path, "最终截面（来自已清理融合云）")
    multi_section_diagnostic = render_multi_section_audit(
        merged_cloud,
        config.section,
        multi_section_audit_path,
        probe_count=config.registration.section_alignment_probe_count,
        probe_spacing=config.registration.section_alignment_probe_spacing,
    )
    render_top_preview(merged_cloud, top_preview_path)
    _render_section_view_alignment_audit(
        section_validation,
        station_labels,
        metrics,
        config.section,
        section_alignment_audit_path,
    )
    _render_section_coverage_audit(
        section_validation,
        station_labels,
        metrics,
        section_coverage_audit_path,
    )
    render_source_alignment_audit(
        [cloud for cloud, included in zip(cleaned_clouds, included_views, strict=True) if included],
        fusion_transforms,
        alignment_audit_path,
        leveling.transformation,
        [
            f'{label}: {path.stem[-8:]}'
            for label, path, included in zip(station_labels, files, included_views, strict=True)
            if included
        ],
    )

    processing_timing = timing.finish()
    section_validation_enabled = bool(config.registration.use_section_alignment_validation)
    summary = {
        'group_name': group_name,
        'timing': processing_timing,
        'overlap_refinement': overlap_refinement,
        'cross_view_denoise': cross_view_denoise,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'alignment_method': (
            'fixed_station_prior_with_fixture_graph_and_validated_3d_overlap_fusion'
            if config.registration.coverage_preserving_fusion
            else (
                'fixed_station_prior_with_calibration_block_only_pose_graph_and_section_validation_representative_fusion'
                if section_validation_enabled
                else 'fixed_station_prior_with_calibration_block_only_pose_graph_and_local_section_consensus_representative_fusion'
            )
        ),
        'processing_flow': [
            '固定工位先验排序',
            '点云去噪',
            '仅多面体标定块 ROI 初调与闭环位姿图',
            '标定块建立统一坐标系，三维重叠区域执行有界刚体微调并验证',
            (
                '固定标定块位姿下的多截面拼接质量诊断（不修改位姿、不剔除互补视角）'
                if section_validation_enabled
                else '跳过多截面质量诊断'
            ),
            '配准后对各视角执行跨视角重叠支持去噪；无重叠区域保留',
            '标定块质量合格视角的已去噪目标点云参与覆盖融合；体素内保留真实代表点，并清理融合后弱支持浮空组件',
            '从最终融合点云提取侧面截面，不再去噪',
            '从最终融合点云生成多横截面覆盖诊断（不合并截面）',
            'CAD 仅在后续同页偏差分析步骤导入并拟合固定点云',
        ],
        'reference_view': reference_file.name,
        'station_pose_order': station_labels,
        'station_order_source': station_order_source,
        'input_files': [str(path.resolve()) for path in files],
        'config': config.as_dict(),
        'compute': gpu_runtime_summary(active_backend, config.compute_backend),
        'denoise_stats': [asdict(item) for item in denoise_stats],
        'registration_metrics': [asdict(item) for item in metrics],
        'pose_graph': {
            'active_indices': calibration_active_indices,
            'excluded_before_pose_graph': [
                index for index, metric in enumerate(metrics) if not metric.included_in_fusion
            ],
            'quality_thresholds': {
                'minimum_calibration_icp_fitness': config.registration.minimum_calibration_icp_fitness,
                'maximum_calibration_icp_rmse': config.registration.maximum_calibration_icp_rmse,
                'minimum_adjacent_fitness': config.registration.calibration_graph_minimum_adjacent_fitness,
                'minimum_loop_fitness': config.registration.calibration_graph_minimum_loop_fitness,
                'maximum_adjacent_rmse': config.registration.calibration_graph_maximum_adjacent_rmse,
                'maximum_loop_rmse': config.registration.calibration_graph_maximum_loop_rmse,
            },
            'calibration_edges': [asdict(item) for item in calibration_edge_metrics],
            'calibration_nodes': calibration_node_metrics,
        },
        'section_validation': {
            'enabled': section_validation.get('enabled', False),
            'role': (
                'diagnostic_only_coverage_preserving'
                if config.registration.coverage_preserving_fusion
                else 'validation_only_no_pose_writeback'
            ),
            'coverage_preserving_fusion': config.registration.coverage_preserving_fusion,
            'fusion_filter_applied': section_validation.get('fusion_filter_applied', False),
            'reference_index': section_validation.get('reference_index'),
            'trusted_indices': section_validation.get('trusted_indices', []),
            'minimum_evidence_points': section_validation.get('minimum_evidence_points'),
            'minimum_retained_views': section_validation.get('minimum_retained_views'),
            'probe_count': section_validation.get('probe_count', 0),
            'probe_offsets': section_validation.get('probe_offsets', []),
            'center_probe_index': section_validation.get('center_probe_index', 0),
            'probe_metrics': section_validation.get('probe_metrics', {}),
            'views': section_validation.get('views', []),
            'pair_metrics': section_validation.get('pair_metrics', {}),
            'isolated_peer_components': section_validation.get('isolated_peer_components', []),
            'protected_peer_components': section_validation.get('protected_peer_components', []),
            'section_consensus': section_validation.get('section_consensus', {}),
        },
        'leveling': leveling.as_dict(),
        'section_consensus': section_consensus,
        'multi_section_diagnostic': multi_section_diagnostic,
        'fused_view_count': included_count,
        'effective_fusion_voxel_size': effective_fusion_voxel_size,
        'fusion_source': (
            'calibration_valid_cross_view_denoised_clouds_nonoverlap_preserving_representative_samples'
            if config.registration.coverage_preserving_fusion
            else (
                'validated_cross_view_denoised_clouds_with_section_consensus_representative_samples'
                if section_validation_enabled
                else 'cross_view_denoised_clouds_with_local_section_consensus_representative_samples'
            )
        ),
        'merged_points': len(merged_cloud.points),
        'section_source': 'merged_fused_cloud',
        'section_source_cloud': str(merged_cloud_path.resolve()),
        'section_source_points': section_source_points,
        'section_raw_points': len(raw_section_cloud.points),
        'section_points': len(section_cloud.points),
        'section_integrity': section_integrity,
        'output_files': {
            'merged_cloud': str(merged_cloud_path.resolve()),
            'raw_section_cloud': str(saved_raw_section_path.resolve()) if saved_raw_section_path else None,
            'section_cloud': str(saved_section_path.resolve()) if saved_section_path else None,
            'raw_section_image': str(raw_section_image_path.resolve()),
            'section_image': str(section_image_path.resolve()),
            'top_preview': str(top_preview_path.resolve()),
            'alignment_audit': str(alignment_audit_path.resolve()),
            'section_alignment_audit': str(section_alignment_audit_path.resolve()),
            'section_coverage_audit': str(section_coverage_audit_path.resolve()),
            'multi_section_audit': str(multi_section_audit_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _emit(progress_callback, f'第 {group_name} 组：完成，融合后 {len(merged_cloud.points):,} 点')
    return PipelineResult(
        group_name=group_name,
        run_directory=run_directory,
        merged_cloud_path=merged_cloud_path,
        raw_section_cloud_path=saved_raw_section_path,
        section_cloud_path=saved_section_path,
        raw_section_image_path=raw_section_image_path,
        section_image_path=section_image_path,
        top_preview_path=top_preview_path,
        summary_path=summary_path,
        merged_points=len(merged_cloud.points),
        section_points=len(section_cloud.points),
        compute_backend=active_backend,
        gpu_device_name=capabilities.device_name if active_backend == 'gpu' else None,
        denoise_stats=denoise_stats,
        registration_metrics=metrics,
        section_source_points=section_source_points,
        section_raw_points=len(raw_section_cloud.points),
        section_integrity=section_integrity,
        leveling=leveling,
        alignment_audit_path=alignment_audit_path,
        section_alignment_audit_path=section_alignment_audit_path,
        section_coverage_audit_path=section_coverage_audit_path,
        multi_section_audit_path=multi_section_audit_path,
    )


def _refine_pose_seed(source, source_calibration, reference_calibration, reference_validation,
                      initial, config, refinement_config, backend, registration_cache,
                      feature_cache):
    """A failed seed is only a failed shortcut, never a failed acquisition."""
    started = perf_counter()
    diagnostic = {"reused": False}
    try:
        try:
            aligned = crop_calibration_cloud(_transform_cloud(source, initial, backend),
                config.calibration, "reference_seed", backend)
            diagnostic["localized_roi"] = True
        except ValueError:
            aligned = _transform_cloud(source_calibration, initial, backend)
            diagnostic["localized_roi"] = False
        correction, fitness, rmse, geometry = _refine_with_icp_diagnostics(
            aligned, reference_calibration, refinement_config,
            backend, registration_cache, feature_cache)
        passed, reason = calibration_registration_quality(
            len(aligned.points), fitness, rmse,
            config.calibration, config.registration)
        transform = correction @ initial
        full_fitness, full_rmse = evaluate_full_cloud_alignment(
            source, reference_validation, transform, config.registration, backend)
        rotation = _rotation_angle_degrees(correction[:3, :3])
        translation = float(np.linalg.norm(correction[:3, 3]))
        # These existing limits select the shortcut, not the final pose.
        # Failed seeds still receive the unrestricted original search.
        reused = bool(
            passed and np.isfinite(transform).all()
            and rotation <= config.registration.pose_prior_rigid_max_rotation_degrees
            and translation <= config.registration.pose_prior_rigid_max_translation
            and full_fitness >= config.registration.minimum_full_cloud_fitness
            and np.isfinite(full_rmse))
        diagnostic.update(reused=reused, fixture_passed=passed, fixture_reason=reason,
                          rotation_degrees=rotation, translation=translation,
                          full_fitness=full_fitness)
        if reused:
            return (transform, None, None, fitness, rmse, len(aligned.points),
                    full_fitness, full_rmse, geometry), diagnostic
        return None, diagnostic
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        diagnostic["reason"] = str(error)
        return None, diagnostic
    finally:
        diagnostic["seconds"] = round(perf_counter() - started, 6)


def process_group(
    group_name: str,
    files: Sequence[Path],
    config: ProcessingConfig,
    output_root: Path,
    progress_callback: ProgressCallback | None = None,
    *,
    pose_reference: dict | None = None,
    capture_pose_reference: bool = False,
) -> PipelineResult:
    """对一组点云完成去噪、标定块配准、融合和三维法向横截面保存。"""
    timing = ProcessingTiming()
    config = _single_view_denoise_config(config)
    if config.registration.use_six_station_pose_prior and len(files) in (
        len(FIVE_STATION_POSE_SEQUENCE), len(SIX_STATION_POSE_SEQUENCE)
    ):
        return process_group_with_six_station_pose_prior(
            group_name,
            files,
            config,
            output_root,
            progress_callback,
            _timing=timing,
        )
    if len(files) < 2:
        raise ValueError("每组至少需要两个 PLY 文件才能拼接。")
    progress_callback = _timed_progress_callback(progress_callback, timing)
    config.validate()
    active_backend = resolve_compute_backend(config.compute_backend)
    capabilities = get_gpu_capabilities()
    files = order_files_with_reference(files, config.registration.reference_vertex_count)
    if active_backend == "gpu":
        _emit(progress_callback, f"第 {group_name} 组：CUDA GPU 加速已启用 - {capabilities.device_name}")
    else:
        _emit(progress_callback, f"第 {group_name} 组：使用 CPU 后端处理")
    _emit(progress_callback, f"第 {group_name} 组：读取并去噪 {len(files)} 个点云")

    cleaned_clouds, denoise_stats = parallel_denoise_clouds(
        files, config.denoise, active_backend, progress_callback, group_name,
    )

    _emit(progress_callback, f"第 {group_name} 组：检查参考视角的标定块基准面")
    reference_index, reference_calibration, leveling = _select_calibration_reference(
        files,
        cleaned_clouds,
        config.calibration,
        active_backend,
    )
    if reference_index != 0:
        previous_reference = files[0]
        order = [reference_index, *(index for index in range(len(files)) if index != reference_index)]
        files = [files[index] for index in order]
        cleaned_clouds = [cleaned_clouds[index] for index in order]
        denoise_stats = [denoise_stats[index] for index in order]
        _emit(
            progress_callback,
            f"第 {group_name} 组：默认参考 {previous_reference.name} 的标定块不完整，"
            f"改用 {files[0].name} 作为参考视角",
        )
    # A rotated capture can move the fixture entirely outside the native ROI.
    # Locate that ROI geometrically before applying the unchanged fixture gate.
    calibration_clouds: list[o3d.geometry.PointCloud] = []
    recovered_calibration_rois: list[str] = []
    for path, cloud in zip(files, cleaned_clouds, strict=True):
        try:
            calibration_cloud = crop_calibration_cloud(
                cloud, config.calibration, path.name, active_backend)
        except ValueError:
            _emit(progress_callback, f"第 {group_name} 组：自动定位移出 ROI 的标定块 - {path.name}")
            calibration_cloud = _recover_calibration_roi(
                read_point_cloud(path), cleaned_clouds[0], config.calibration,
                config.registration, path.name, active_backend)
            recovered_calibration_rois.append(path.name)
        calibration_clouds.append(calibration_cloud)
    reference_calibration = calibration_clouds[0]
    reference_cloud = cleaned_clouds[0]
    reference_validation = _voxel_down_sample_cloud(
        reference_cloud,
        config.registration.registration_voxel_size,
        active_backend,
    )
    from .registration_reference import match_reference, make_reference
    pose_seeds, reuse_report = match_reference(
        pose_reference, files, calibration_clouds, config)
    reuse_report["views"] = []
    if pose_reference:
        _emit(progress_callback, f"第 {group_name} 组：位姿参考匹配 {reuse_report['status']} "
              f"{reuse_report.get('reason', '')}")
    _emit(progress_callback, f"第 {group_name} 组：拟合标定块水平基准面并调平")
    if leveling.applied:
        _emit(
            progress_callback,
            f"第 {group_name} 组：调平完成，基准面倾角 {leveling.tilt_degrees:.3f}°，"
            f"拟合点 {leveling.plane_points:,}，RMSE {leveling.plane_rmse:.4f}",
        )
    else:
        _emit(progress_callback, f"第 {group_name} 组：标定块已接近水平，无需旋转")

    transforms: list[np.ndarray] = [np.eye(4)]
    metrics: list[RegistrationMetric] = [
        RegistrationMetric(
            file_name=files[0].name,
            calibration_points=len(reference_calibration.points),
            global_fitness=None,
            global_rmse=None,
            icp_fitness=1.0,
            icp_rmse=0.0,
            transformation=np.eye(4).round(12).tolist(),
            full_cloud_fitness=1.0,
            full_cloud_rmse=0.0,
            included_in_fusion=True,
            calibration_quality_passed=True,
        )
    ]
    icp_backend_label = "GPU" if active_backend == "gpu" else "CPU"
    calibration_refinement_config = copy.deepcopy(config.registration)
    calibration_refinement_config.registration_voxel_size = min(
        config.registration.registration_voxel_size,
        config.registration.calibration_graph_voxel_size,
    )
    calibration_refinement_config.icp_max_correspondence_distance = min(
        config.registration.icp_max_correspondence_distance,
        config.registration.calibration_graph_max_correspondence_distance,
    )
    calibration_refinement_config.icp_max_iterations = max(
        config.registration.icp_max_iterations,
        config.registration.calibration_graph_icp_max_iterations,
    )
    registration_cache: dict[str, dict] = {}
    feature_cache: dict[int, tuple[o3d.geometry.PointCloud, list[dict[str, object]]]] = {}
    for index, (source_cloud, path) in enumerate(zip(cleaned_clouds[1:], files[1:]), start=2):
        best_candidate: tuple[
            np.ndarray,
            float | None,
            float | None,
            float,
            float,
            int,
            float,
            float,
            dict[str, object],
        ] | None = None
        source_calibration = calibration_clouds[index - 1]
        reused = False
        if pose_seeds is not None and pose_seeds[index - 1] is not None:
            seed_validation = copy.deepcopy(reference_validation)
            for peer in range(1, len(transforms)):
                if metrics[peer].included_in_fusion:
                    seed_validation += _transform_cloud(
                        cleaned_clouds[peer].voxel_down_sample(
                            config.registration.registration_voxel_size), transforms[peer], active_backend)
            seed_validation = seed_validation.voxel_down_sample(config.registration.registration_voxel_size)
            best_candidate, attempt = _refine_pose_seed(
                source_cloud, source_calibration, reference_calibration, seed_validation,
                pose_seeds[index - 1], config, calibration_refinement_config, active_backend,
                registration_cache, feature_cache)
            reused = best_candidate is not None
            if reused and attempt["localized_roi"]:
                aligned_roi = crop_calibration_cloud(
                    _transform_cloud(source_cloud, best_candidate[0], active_backend),
                    config.calibration, path.name, active_backend)
                calibration_clouds[index - 1] = _transform_cloud(
                    aligned_roi, np.linalg.inv(best_candidate[0]), active_backend)
            reuse_report["views"].append(dict(attempt, file=path.name))
            _emit(progress_callback, f"第 {group_name} 组：本组标定块参考精配准 {index}/{len(files)} "
                  + ("通过" if reused else "未通过快速路径检查，回退完整配准"))
        elif pose_seeds is not None:
            reuse_report["views"].append({
                "file": path.name, "reused": False, "reason": "unmatched_station",
            })
            _emit(progress_callback, f"第 {group_name} 组：视角 {index} 对应不明确，使用完整配准")
        for target_index in range(0 if reused else len(transforms)):
            if not metrics[target_index].included_in_fusion:
                continue
            target_calibration = calibration_clouds[target_index]
            target_transform = transforms[target_index]
            target_name = files[target_index].name
            stage_label = "参考视角" if target_index == 0 else f"已对齐视角 {target_index + 1}"
            _emit(
                progress_callback,
                f"第 {group_name} 组：标定块 FPFH + RANSAC 粗配准（CPU） {index}/{len(files)} → {stage_label} - {path.name}",
            )
            coarse_to_target, global_fitness, global_rmse = coarse_calibration_register(
                source_calibration,
                target_calibration,
                config.registration,
                active_backend,
                registration_cache=registration_cache,
            )
            initial_to_reference = target_transform @ coarse_to_target
            coarse_aligned_calibration = _transform_cloud(
                source_calibration,
                initial_to_reference,
                active_backend,
            )

            _emit(
                progress_callback,
                f"第 {group_name} 组：多面体标定块 ICP 精配准（{icp_backend_label}） {index}/{len(files)} ← {target_name}",
            )
            fine_transform, icp_fitness, icp_rmse, geometry_report = _refine_with_icp_diagnostics(
                coarse_aligned_calibration,
                _transform_cloud(target_calibration, target_transform, active_backend),
                calibration_refinement_config,
                active_backend,
                registration_cache,
                feature_cache,
            )
            calibration_transform = fine_transform @ initial_to_reference
            base_full_fitness, base_full_rmse = evaluate_full_cloud_alignment(
                source_cloud,
                reference_validation,
                calibration_transform,
                config.registration,
                active_backend,
            )
            _emit(
                progress_callback,
                f"第 {group_name} 组：评估该标定块候选的整体重叠质量 - {path.name}",
            )
            final_transform = calibration_transform
            full_cloud_fitness = base_full_fitness
            full_cloud_rmse = base_full_rmse
            candidate = (
                final_transform,
                global_fitness,
                global_rmse,
                icp_fitness,
                icp_rmse,
                len(source_calibration.points),
                full_cloud_fitness,
                full_cloud_rmse,
                geometry_report,
            )
            if best_candidate is None or calibration_candidate_score(
                candidate[1],
                candidate[3],
                candidate[4],
            ) > calibration_candidate_score(
                best_candidate[1],
                best_candidate[3],
                best_candidate[4],
            ):
                best_candidate = candidate

        if best_candidate is None:
            raise ValueError(
                f"{path.name} 无法通过参考或已对齐相邻视角定位标定块。"
                "请扩大标定块 ROI、提高全局配准重试次数或调整参考视角目标点数。"
            )
        localized_calibration = None
        if not reused:
            _emit(progress_callback, f"第 {group_name} 组：验证整体朝向，消除对称标定块歧义 - {path.name}")
            best_candidate, localized_calibration = _disambiguate_global_pose(
                source_cloud, reference_cloud, reference_calibration, best_candidate,
                config, calibration_refinement_config, path.name, active_backend,
                registration_cache=registration_cache,
                feature_cache=feature_cache)
        if localized_calibration is not None:
            calibration_clouds[index - 1] = localized_calibration
        (
            final_transform,
            global_fitness,
            global_rmse,
            icp_fitness,
            icp_rmse,
            calibration_points,
            full_cloud_fitness,
            full_cloud_rmse,
            geometry_report,
        ) = best_candidate
        calibration_quality_passed, calibration_quality_reason = calibration_registration_quality(
            calibration_points,
            icp_fitness,
            icp_rmse,
            config.calibration,
            config.registration,
        )
        included_in_fusion = True
        exclusion_reason: str | None = None
        if (
            config.registration.skip_low_quality_calibration_views
            and not calibration_quality_passed
        ):
            included_in_fusion = False
            exclusion_reason = "calibration_icp_quality"
        elif (
            config.registration.skip_low_quality_views
            and not config.registration.coverage_preserving_fusion
            and full_cloud_fitness < config.registration.minimum_full_cloud_fitness
        ):
            included_in_fusion = False
            exclusion_reason = "full_cloud_fitness"
        transforms.append(final_transform)
        metrics.append(
            RegistrationMetric(
                file_name=path.name,
                calibration_points=calibration_points,
                global_fitness=global_fitness,
                global_rmse=global_rmse,
                icp_fitness=icp_fitness,
                icp_rmse=icp_rmse,
                transformation=np.asarray(final_transform).round(12).tolist(),
                full_cloud_fitness=full_cloud_fitness,
                full_cloud_rmse=full_cloud_rmse,
                included_in_fusion=included_in_fusion,
                exclusion_reason=exclusion_reason,
                calibration_quality_passed=calibration_quality_passed,
                calibration_quality_reason=calibration_quality_reason,
                calibration_geometry_status=str(geometry_report.get("status")),
                calibration_geometry_plane_count=int(
                    geometry_report.get("source_plane_count", 0)
                ),
                calibration_geometry_matched_planes=int(
                    geometry_report.get("matched_plane_count", 0)
                ),
                calibration_geometry_plane_rmse_before=(
                    float(geometry_report["plane_rmse_before"])
                    if geometry_report.get("plane_rmse_before") is not None
                    else None
                ),
                calibration_geometry_plane_rmse_after=(
                    float(geometry_report["plane_rmse_after"])
                    if geometry_report.get("plane_rmse_after") is not None
                    else None
                ),
                calibration_geometry_correction_rotation_degrees=float(
                    geometry_report.get("correction_rotation_degrees", 0.0)
                ),
                calibration_geometry_correction_translation=float(
                    geometry_report.get("correction_translation", 0.0)
                ),
                calibration_geometry_pairwise_rmse=(
                    float(geometry_report["plane_pairwise_rmse"])
                    if geometry_report.get("plane_pairwise_rmse") is not None
                    else None
                ),
                calibration_geometry_report=_calibration_geometry_metric_report(
                    geometry_report
                ),
            )
        )
        if not calibration_quality_passed:
            _emit(
                progress_callback,
                f"第 {group_name} 组：标定块质量未通过 {path.name} - "
                f"{calibration_quality_reason}；"
                f"{'排除融合' if not included_in_fusion else '保留但标记'}",
            )
        elif not included_in_fusion:
            _emit(
                progress_callback,
                f"第 {group_name} 组：跳过低质量视角 {path.name}，"
                f"全云 fitness {full_cloud_fitness:.4f} 低于 {config.registration.minimum_full_cloud_fitness:.4f}",
            )
        elif config.registration.coverage_preserving_fusion:
            _emit(
                progress_callback,
                f"第 {group_name} 组：保留互补覆盖视角 {path.name}；"
                f"全云 fitness {full_cloud_fitness:.4f} 仅作诊断，不作为删除依据",
            )

    # Establish a fixture-anchored graph before bounded overlap refinement.
    calibration_graph_edges: list[PoseGraphEdgeMetric] = []
    calibration_graph_nodes: list[dict[str, object]] = []
    calibration_graph_active_indices = [
        index for index, metric in enumerate(metrics) if metric.included_in_fusion
    ]
    if len(calibration_graph_active_indices) >= 3:
        _emit(
            progress_callback,
            f"第 {group_name} 组：仅使用标定块执行 {len(calibration_graph_active_indices)} 视角闭环位姿图优化",
        )
        calibration_graph_config = copy.deepcopy(config.registration)
        calibration_graph_config.registration_voxel_size = (
            config.registration.calibration_graph_voxel_size
        )
        calibration_graph_config.icp_max_correspondence_distance = (
            config.registration.calibration_graph_max_correspondence_distance
        )
        calibration_graph_config.icp_max_iterations = (
            config.registration.calibration_graph_icp_max_iterations
        )
        aligned_calibration_clouds = [
            _transform_cloud(
                calibration_cloud,
                transforms[index],
                active_backend,
            )
            for calibration_cloud, index in zip(
                calibration_clouds,
                range(len(calibration_clouds)),
                strict=True,
            )
        ]
        generic_labels = [f"view-{index + 1}" for index in range(len(files))]
        calibration_corrections, calibration_graph_edges, calibration_graph_nodes = (
            build_and_optimize_pairwise_pose_graph(
                aligned_calibration_clouds,
                generic_labels,
                calibration_graph_config,
                "calibration",
                maximum_rotation_degrees=config.registration.calibration_graph_max_rotation_degrees,
                maximum_translation=config.registration.calibration_graph_max_translation,
                minimum_adjacent_fitness=config.registration.calibration_graph_minimum_adjacent_fitness,
                minimum_loop_fitness=config.registration.calibration_graph_minimum_loop_fitness,
                backend=active_backend,
                progress_callback=progress_callback,
                group_name=group_name,
                active_indices=calibration_graph_active_indices,
                reference_label=generic_labels[0],
                maximum_adjacent_rmse=config.registration.calibration_graph_maximum_adjacent_rmse,
                maximum_loop_rmse=config.registration.calibration_graph_maximum_loop_rmse,
                registration_cache=registration_cache,
                feature_cache=feature_cache,
            )
        )
        transforms = [
            correction @ transformation
            for correction, transformation in zip(
                calibration_corrections,
                transforms,
                strict=True,
            )
        ]
        for metric, transformation in zip(metrics, transforms, strict=True):
            metric.transformation = np.asarray(transformation).round(12).tolist()

    transforms, overlap_refinement = _refine_group_overlap(
        cleaned_clouds, transforms, metrics, config, progress_callback, group_name)
    learned_reference = (make_reference(files, calibration_clouds, transforms, config, metrics)
                         if capture_pose_reference and pose_reference is None else None)
    _emit(progress_callback, f"第 {group_name} 组：应用位姿并融合（{icp_backend_label}）")
    section_validation = validate_section_view_alignment(
        files,
        transforms,
        metrics,
        leveling.transformation,
        config.section,
        config.registration,
        active_backend,
        progress_callback,
        group_name,
        source_clouds=cleaned_clouds,
    )

    effective_fusion_voxel_size = float(config.fusion_voxel_size)
    included_views = [metric.included_in_fusion for metric in metrics]
    included_count = sum(included_views)
    if not included_count:
        raise ValueError("标定块质量闸门后没有可用于融合的工位，请检查标定块和配准参数。")
    minimum_required_views = min(config.registration.section_alignment_minimum_views, len(files))
    if included_count < minimum_required_views:
        raise ValueError(
            f"标定块质量闸门后仅保留 {included_count}/{len(files)} 个工位，"
            f"低于当前流程建议的 {minimum_required_views} 个。"
        )
    cleaned_clouds, cross_view_denoise = _denoise_registered_views(
        cleaned_clouds, transforms, leveling.transformation, included_views,
        config.denoise.voxel_size, progress_callback, group_name,
    )
    merged_cloud, section_consensus = _merge_clouds_with_section_consensus(
        files,
        transforms,
        included_views,
        leveling.transformation,
        config.section,
        effective_fusion_voxel_size,
        active_backend,
        progress_callback,
        group_name,
        coverage_preserving=config.registration.coverage_preserving_fusion,
        section_recovery_voxel_size=config.denoise.voxel_size,
        calibration_config=config.calibration,
        calibration_bounds_in_reference_frame=True,
        source_clouds=cleaned_clouds,
    )
    _annotate_section_consensus_report(section_consensus, metrics)
    if len(merged_cloud.points) == 0:
        raise ValueError("融合后点云为空，请检查去噪和拼接参数。")
    _emit(
        progress_callback,
        f"第 {group_name} 组：从已融合点云提取三维法向平面截面（{icp_backend_label}）",
    )
    raw_section_cloud, section_source_points = extract_section_from_merged_cloud(
        merged_cloud,
        config.section,
        active_backend,
    )
    section_cloud = copy.deepcopy(raw_section_cloud)
    section_integrity = diagnose_section_integrity(
        raw_section_cloud,
        section_cloud,
        config.section,
        source_profiles=section_validation.get("profiles"),
        source_profile_indices=(
            [index for index, included in enumerate(included_views) if included]
            if config.registration.coverage_preserving_fusion
            else section_validation.get("trusted_indices")
        ),
    )
    run_directory = _create_run_directory(output_root, group_name)
    merged_cloud_path = run_directory / "merged_pointcloud.ply"
    raw_section_cloud_path = run_directory / "vertical_section_raw.ply"
    section_cloud_path = run_directory / "vertical_section.ply"
    raw_section_image_path = run_directory / "vertical_section_raw.png"
    section_image_path = run_directory / "vertical_section.png"
    multi_section_audit_path = run_directory / "vertical_sections_multi.png"
    top_preview_path = run_directory / "merged_top_view.png"
    summary_path = run_directory / "run_summary.json"

    _emit(progress_callback, f"第 {group_name} 组：保存点云和截面图")
    _write_point_cloud(merged_cloud_path, merged_cloud)
    saved_raw_section_path: Path | None = None
    if len(raw_section_cloud.points):
        _write_point_cloud(raw_section_cloud_path, raw_section_cloud)
        saved_raw_section_path = raw_section_cloud_path
    saved_section_path: Path | None = None
    if len(section_cloud.points):
        _write_point_cloud(section_cloud_path, section_cloud)
        saved_section_path = section_cloud_path
    render_section_image(raw_section_cloud, config.section, raw_section_image_path, "融合云基线截面")
    render_section_image(section_cloud, config.section, section_image_path, "最终截面（来自已清理融合云）")
    multi_section_diagnostic = render_multi_section_audit(
        merged_cloud,
        config.section,
        multi_section_audit_path,
        probe_count=config.registration.section_alignment_probe_count,
        probe_spacing=config.registration.section_alignment_probe_spacing,
    )
    render_top_preview(merged_cloud, top_preview_path)

    processing_timing = timing.finish()
    summary = {
        "group_name": group_name,
        "timing": processing_timing,
        "overlap_refinement": overlap_refinement,
        "cross_view_denoise": cross_view_denoise,
        "recovered_calibration_rois": recovered_calibration_rois,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alignment_method": (
            "fixture_anchored_global_pose_validation_and_3d_overlap_fusion"
            if config.registration.coverage_preserving_fusion
            else (
                "calibration_block_only_coarse_icp_with_fixture_pose_graph_and_section_validation"
                if config.registration.use_section_alignment_validation
                else "calibration_block_only_coarse_icp_with_fixture_pose_graph_and_local_section_consensus"
            )
        ),
        "processing_flow": [
            "按点云点数选择参考视角",
            "点云去噪",
            "提取标定块 ROI，移出原始范围时自动定位",
            "标定块初配准与整体朝向消歧，候选必须通过标定块质量检查",
            "对已定位视角再次仅用标定块建立多视角闭环位姿图",
            "三维重叠区域执行带退化方向保护的有界刚体微调，双向验证后采用",
            (
                "固定标定块位姿下的多截面拼接质量诊断（不修改位姿、不剔除互补视角）"
                if config.registration.use_section_alignment_validation
                else "跳过多截面质量诊断"
            ),
            "配准后对各视角执行跨视角重叠支持去噪；无重叠区域保留",
            "标定块质量合格视角的已去噪目标点云参与覆盖融合，并清理融合后弱支持浮空组件",
            "从最终融合点云提取侧面截面，不再去噪",
            "CAD 仅在后续同页偏差分析步骤导入并拟合固定点云",
        ],
        "input_files": [str(path.resolve()) for path in files],
        "config": config.as_dict(),
        "compute": gpu_runtime_summary(active_backend, config.compute_backend),
        "denoise_stats": [asdict(item) for item in denoise_stats],
        "registration_metrics": [asdict(item) for item in metrics],
        "pose_reference": reuse_report,
        "pose_reference_candidate": learned_reference,
        "pose_graph": {
            "active_indices": calibration_graph_active_indices,
            "quality_thresholds": {
                "minimum_calibration_icp_fitness": config.registration.minimum_calibration_icp_fitness,
                "maximum_calibration_icp_rmse": config.registration.maximum_calibration_icp_rmse,
                "minimum_adjacent_fitness": config.registration.calibration_graph_minimum_adjacent_fitness,
                "minimum_loop_fitness": config.registration.calibration_graph_minimum_loop_fitness,
                "maximum_adjacent_rmse": config.registration.calibration_graph_maximum_adjacent_rmse,
                "maximum_loop_rmse": config.registration.calibration_graph_maximum_loop_rmse,
            },
            "calibration_edges": [asdict(item) for item in calibration_graph_edges],
            "calibration_nodes": calibration_graph_nodes,
            "source": "calibration_block_only",
        },
        "section_alignment_validation": {
            "enabled": config.registration.use_section_alignment_validation,
            "thin_section_thickness": min(
                config.section.thickness,
                config.registration.section_alignment_thickness,
            ),
            "overlap_distance": config.registration.section_alignment_overlap_distance,
            "maximum_p90_distance": config.registration.section_alignment_max_p90_distance,
            "minimum_overlap_points": config.registration.section_alignment_minimum_overlap_points,
        },
        "section_validation": {
            "enabled": section_validation.get("enabled", False),
            "role": (
                "diagnostic_only_coverage_preserving"
                if config.registration.coverage_preserving_fusion
                else "validation_only_no_pose_writeback"
            ),
            "coverage_preserving_fusion": config.registration.coverage_preserving_fusion,
            "fusion_filter_applied": section_validation.get("fusion_filter_applied", False),
            "reference_index": section_validation.get("reference_index"),
            "trusted_indices": section_validation.get("trusted_indices", []),
            "minimum_evidence_points": section_validation.get("minimum_evidence_points"),
            "minimum_retained_views": section_validation.get("minimum_retained_views"),
            "probe_count": section_validation.get("probe_count", 0),
            "probe_offsets": section_validation.get("probe_offsets", []),
            "center_probe_index": section_validation.get("center_probe_index", 0),
            "probe_metrics": section_validation.get("probe_metrics", {}),
            "views": section_validation.get("views", []),
            "pair_metrics": section_validation.get("pair_metrics", {}),
            "isolated_peer_components": section_validation.get(
                "isolated_peer_components", []
            ),
            "protected_peer_components": section_validation.get(
                "protected_peer_components", []
            ),
            "section_consensus": section_validation.get("section_consensus", {}),
        },
        "leveling": leveling.as_dict(),
        "section_consensus": section_consensus,
        "multi_section_diagnostic": multi_section_diagnostic,
        "fused_view_count": sum(item.included_in_fusion for item in metrics),
        "effective_fusion_voxel_size": effective_fusion_voxel_size,
        "fusion_source": (
            "calibration_valid_cross_view_denoised_clouds_nonoverlap_preserving_representative_samples"
            if config.registration.coverage_preserving_fusion
            else "validated_cross_view_denoised_clouds_with_section_consensus_representative_samples"
        ),
        "merged_points": len(merged_cloud.points),
        "section_source": "merged_fused_cloud",
        "section_source_cloud": str(merged_cloud_path.resolve()),
        "section_source_points": section_source_points,
        "section_raw_points": len(raw_section_cloud.points),
        "section_points": len(section_cloud.points),
        "section_integrity": section_integrity,
        "output_files": {
            "merged_cloud": str(merged_cloud_path.resolve()),
            "raw_section_cloud": str(saved_raw_section_path.resolve()) if saved_raw_section_path else None,
            "section_cloud": str(saved_section_path.resolve()) if saved_section_path else None,
            "raw_section_image": str(raw_section_image_path.resolve()),
            "section_image": str(section_image_path.resolve()),
            "multi_section_audit": str(multi_section_audit_path.resolve()),
            "top_preview": str(top_preview_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit(progress_callback, f"第 {group_name} 组：完成，融合后 {len(merged_cloud.points):,} 点")
    return PipelineResult(
        group_name=group_name,
        run_directory=run_directory,
        merged_cloud_path=merged_cloud_path,
        raw_section_cloud_path=saved_raw_section_path,
        section_cloud_path=saved_section_path,
        raw_section_image_path=raw_section_image_path,
        section_image_path=section_image_path,
        top_preview_path=top_preview_path,
        summary_path=summary_path,
        merged_points=len(merged_cloud.points),
        section_points=len(section_cloud.points),
        compute_backend=active_backend,
        gpu_device_name=capabilities.device_name if active_backend == "gpu" else None,
        denoise_stats=denoise_stats,
        registration_metrics=metrics,
        section_source_points=section_source_points,
        section_raw_points=len(raw_section_cloud.points),
        section_integrity=section_integrity,
        multi_section_audit_path=multi_section_audit_path,
        leveling=leveling,
    )
def _create_run_directory(output_root: Path, group_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = output_root / f"group_{group_name}" / timestamp
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _write_point_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> None:
    success = o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)
    if not success:
        raise IOError(f"无法写入点云文件: {path}")


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
