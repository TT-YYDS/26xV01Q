from __future__ import annotations

from pointcloud_app.profiles import (
    DATA_FAMILY_PROFILES,
    classify_group_family,
    _apply_config_overrides,
    apply_family_profile,
)
import copy
import hashlib
import importlib
import json
import os
import time
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cad_analysis_panel import render_cad_analysis_panel
from pointcloud_app import gpu as gpu_module
from pointcloud_app import models as models_module
from pointcloud_app import pipeline as pipeline_module

_registration_fields = getattr(models_module.RegistrationConfig, "__dataclass_fields__", {})
_calibration_fields = getattr(models_module.CalibrationConfig, "__dataclass_fields__", {})
_section_fields = getattr(models_module.SectionConfig, "__dataclass_fields__", {})
_processing_fields = getattr(models_module.ProcessingConfig, "__dataclass_fields__", {})
_models_need_reload = (
    not {
        "skip_low_quality_views",
        "minimum_full_cloud_fitness",
        "minimum_calibration_icp_fitness",
        "maximum_calibration_icp_rmse",
        "skip_low_quality_calibration_views",
        "coverage_preserving_fusion",
        "use_section_alignment_validation",
        "global_random_seed",
        "calibration_graph_voxel_size",
        "calibration_graph_maximum_adjacent_rmse",
        "calibration_graph_maximum_loop_rmse",
        "workpiece_graph_voxel_size",
        "station_pose_corrections",
        "use_multiscale_icp",
        "use_calibration_geometry_constraints",
        "calibration_geometry_voxel_size",
        "calibration_geometry_plane_distance",
        "calibration_geometry_minimum_plane_points",
        "calibration_geometry_minimum_planes",
        "calibration_geometry_max_planes",
        "calibration_geometry_max_normal_angle_degrees",
        "calibration_geometry_max_centroid_distance",
    }.issubset(_registration_fields)
    or not {"exclude_from_fusion", "exclusion_margin"}.issubset(_calibration_fields)
    or not {
        "tilt_degrees",
        "auto_offset_enabled",
        "offset_scan_half_range",
        "offset_scan_step",
        "offset_min_quality_gain",
    }.issubset(_section_fields)
    or not {"data_family", "profile_name", "profile_source"}.issubset(_processing_fields)
)
_gpu_needs_reload = getattr(gpu_module, "GPU_IMPLEMENTATION_VERSION", 0) < 2
_pipeline_needs_reload = (
    not hasattr(pipeline_module, "estimate_reference_leveling")
    or getattr(pipeline_module, "VERTICAL_SECTION_IMPLEMENTATION_VERSION", 0) < 36
)
if _models_need_reload:
    models_module = importlib.reload(models_module)
if _gpu_needs_reload:
    gpu_module = importlib.reload(gpu_module)
if _models_need_reload or _gpu_needs_reload or _pipeline_needs_reload:
    pipeline_module = importlib.reload(pipeline_module)

from pointcloud_app.gpu import get_gpu_capabilities
from pointcloud_app.models import (
    Bounds2D,
    Bounds3D,
    CalibrationConfig,
    DenoiseConfig,
    ProcessingConfig,
    RegistrationConfig,
    SectionConfig,
)
from pointcloud_app.pipeline import (
    PipelineResult,
    PreviewData,
    build_preview,
    build_surface_preview,
    discover_groups,
    process_group,
    select_reference_file,
    select_reference_file_with_leveling,
    normal_plane_from_xyz_region,
    read_point_cloud,
    scan_section_offsets,
    six_station_pose_transformations,
    vertical_plane_basis,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("POINTCLOUD_DATA_ROOT", str(PROJECT_ROOT / "media"))
).resolve()
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

PREVIEW_SAMPLING_LEVELS: dict[str, tuple[float, int, str]] = {
    "1 级｜极速": (0.20, 50_000, "快速查看主要结构与标定块位置"),
    "2 级｜标准": (0.10, 100_000, "清楚查看整体轮廓与主要表面"),
    "3 级｜精细": (0.06, 200_000, "推荐，清晰检查多面体边缘与表面细节"),
    "4 级｜超清": (0.01, 2_000_000, "最多显示 2,000,000 点，适合精细检查；加载、重建和旋转会明显变慢"),
}

COMPUTE_BACKEND_OPTIONS: dict[str, str] = {
    "自动｜优先 GPU": "auto",
    "强制 GPU｜不可用则停止": "gpu",
    "仅 CPU": "cpu",
}

STATION_POSE_CORRECTION_OPTIONS: dict[str, str] = {
    "-10° 工位": "minus10",
    "-28° 工位": "minus28",
    "-50° 工位": "minus50",
    "+37° 工位": "plus37",
    "+47° 工位": "plus47",
}

SIMPLE_STITCHING_PRESETS: dict[str, dict[str, float | int | str]] = {
    "标准自动（推荐）": {
        "description": "适用于当前设备和大多数正常点云；以标定块求位姿并保留各角度互补覆盖，重叠区域只做质量诊断。",
        "denoise_voxel": 0.15,
        "registration_voxel": 0.35,
        "graph_voxel": 0.12,
        "calibration_distance": 0.40,
        "graph_iterations": 200,
        "calibration_adjacent_fitness": 0.45,
        "calibration_loop_fitness": 0.50,
        "section_overlap": 0.35,
        "section_p90": 0.115,
        "section_overlap_points": 500,
        "fusion_voxel": 0.03,
    },
    "高精度｜截面有重影": {
        "description": "减小配准体素和对应距离，适合截面出现双线、厚边或轻微错层。",
        "denoise_voxel": 0.12,
        "registration_voxel": 0.28,
        "graph_voxel": 0.08,
        "calibration_distance": 0.30,
        "graph_iterations": 300,
        "calibration_adjacent_fitness": 0.50,
        "calibration_loop_fitness": 0.55,
        "section_overlap": 0.30,
        "section_p90": 0.10,
        "section_overlap_points": 600,
        "fusion_voxel": 0.025,
    },
    "宽松保留｜工位容易缺失": {
        "description": "对局部覆盖更宽容，适合有效视角只扫描到局部轮廓的情况。",
        "denoise_voxel": 0.15,
        "registration_voxel": 0.35,
        "graph_voxel": 0.12,
        "calibration_distance": 0.45,
        "graph_iterations": 200,
        "calibration_adjacent_fitness": 0.35,
        "calibration_loop_fitness": 0.45,
        "section_overlap": 0.45,
        "section_p90": 0.20,
        "section_overlap_points": 300,
        "fusion_voxel": 0.03,
    },
    "快速检查｜先看整体": {
        "description": "降低计算量，适合先确认区域和大致拼接；正式结果建议改回标准自动。",
        "denoise_voxel": 0.20,
        "registration_voxel": 0.45,
        "graph_voxel": 0.16,
        "calibration_distance": 0.45,
        "graph_iterations": 120,
        "calibration_adjacent_fitness": 0.40,
        "calibration_loop_fitness": 0.48,
        "section_overlap": 0.40,
        "section_p90": 0.18,
        "section_overlap_points": 350,
        "fusion_voxel": 0.035,
    },
}

# The fixed five/six-station data is not homogeneous.  These profiles keep
# the user-selected ROI and section geometry, while tuning the parts of the
# pipeline that are sensitive to point density and surface texture.

DATA_FAMILY_TUNING_MODES: dict[str, str] = {
    "按数据族自动推荐（推荐）": "auto",
    "沿用当前通用参数": "common",
}








def build_family_processing_configs(
    config: ProcessingConfig,
    group_names: list[str],
) -> dict[str, ProcessingConfig]:
    """Build one independent processing config per selected group."""
    tuning_mode = st.sidebar.selectbox(
        "三组点云独立调优",
        list(DATA_FAMILY_TUNING_MODES),
        key="data_family_tuning_mode",
        help="数字组、基础木板组和木板变体组使用不同的去噪、标定块位姿图和局部共识参数；ROI、截面方向和工位修正仍沿用当前设置。",
    )
    source = DATA_FAMILY_TUNING_MODES[tuning_mode]
    configs: dict[str, ProcessingConfig] = {}
    families_in_selection: dict[str, list[str]] = {}
    for group_name in group_names:
        family = classify_group_family(group_name)
        families_in_selection.setdefault(family, []).append(group_name)
        configs[group_name] = apply_family_profile(config, family, source=source)

    profile_lines = [
        f"{DATA_FAMILY_PROFILES[family]['label']}：{configs[group_names_for_family[0]].profile_name}"
        for family, group_names_for_family in families_in_selection.items()
    ]
    st.sidebar.caption("；".join(profile_lines))
    return configs

PREVIEW_WEBGL_BATCH_SIZE = 60_000
SURFACE_PREVIEW_MAXIMUM_POINTS = 500_000
SURFACE_PREVIEW_MAXIMUM_TRIANGLES = 1_500_000
# The picker only needs enough points to recover the main axis. Keeping this
# below the render limit makes box selection and Streamlit reruns responsive.
SECTION_PICKER_MAXIMUM_POINTS = 20_000
SECTION_OFFSET_SCAN_MAXIMUM_POINTS = 120_000
PREVIEW_RENDER_REVISION = "surface-webgl-v3-orthographic"
POINTCLOUD_DATA_SIGNATURE_STATE_KEY = "active_pointcloud_data_signature"
POINTCLOUD_TARGET_GROUP_STATE_KEY = "pointcloud_target_group_index"


def _selected_group_names_from_index(
    group_names: Sequence[str],
    selected_index: int,
) -> list[str]:
    """Resolve the sidebar selection without parsing its human-readable label."""
    names = list(group_names)
    if not names:
        return []
    if selected_index == 0:
        return names
    if 1 <= selected_index <= len(names):
        return [names[selected_index - 1]]
    raise ValueError(f"点云组选择索引无效：{selected_index}")


st.set_page_config(
    page_title="FCMS | Floor Clip Measurement System",
    page_icon=str(Path(__file__).parent / "pointcloud_app" / "native" / "fcms_logo.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --tech-bg: #080d18;
    --tech-panel: rgba(16, 25, 43, 0.78);
    --tech-panel-strong: rgba(20, 32, 55, 0.94);
    --tech-border: rgba(94, 154, 255, 0.24);
    --tech-cyan: #43d9ff;
    --tech-blue: #5b8cff;
    --tech-purple: #a277ff;
    --tech-text: #e7efff;
    --tech-muted: #8d9ab5;
}

.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(66, 111, 255, 0.16), transparent 31rem),
        radial-gradient(circle at 10% 15%, rgba(38, 213, 255, 0.08), transparent 27rem),
        var(--tech-bg);
    color: var(--tech-text);
}

.block-container {
    max-width: 1680px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(17, 29, 51, 0.98), rgba(8, 14, 27, 0.98)),
        var(--tech-bg);
    border-right: 1px solid var(--tech-border);
    box-shadow: 12px 0 40px rgba(0, 0, 0, 0.22);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] .stExpander {
    margin: 0.55rem 0;
    border: 1px solid rgba(111, 151, 220, 0.22);
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(28, 43, 73, 0.72), rgba(11, 20, 37, 0.72));
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035), 0 8px 24px rgba(0, 0, 0, 0.12);
}

[data-testid="stSidebar"] .stExpander:first-of-type {
    border-color: rgba(67, 217, 255, 0.52);
    box-shadow: 0 0 0 1px rgba(67, 217, 255, 0.08), 0 0 26px rgba(67, 217, 255, 0.09);
}

[data-testid="stSidebar"] .stExpander summary p {
    color: var(--tech-text);
    font-weight: 700;
    letter-spacing: 0.02em;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #c7d2e8;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: var(--tech-muted);
}

[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    border: 1px solid rgba(117, 157, 230, 0.38);
    border-radius: 10px;
    background: rgba(36, 52, 84, 0.62);
    color: #dce8ff;
    transition: all 160ms ease;
}

[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    border-color: var(--tech-cyan);
    color: #ffffff;
    box-shadow: 0 0 18px rgba(67, 217, 255, 0.18);
    transform: translateY(-1px);
}

[data-testid="stSidebar"] button[kind="primary"] {
    border: 0;
    background: linear-gradient(100deg, #ff5360, #ff356f 52%, #a961ff);
    box-shadow: 0 9px 24px rgba(255, 67, 105, 0.26);
    font-weight: 800;
    letter-spacing: 0.04em;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(91, 140, 255, 0.22);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    background: linear-gradient(145deg, rgba(19, 34, 61, 0.82), rgba(10, 18, 34, 0.74));
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

[data-testid="stAlert"] {
    border-radius: 12px;
    border-color: rgba(91, 140, 255, 0.28);
    background: rgba(18, 30, 54, 0.72);
}

.app-hero {
    position: relative;
    overflow: hidden;
    margin-bottom: 1.6rem;
    padding: 1.65rem 2rem;
    border: 1px solid rgba(91, 140, 255, 0.3);
    border-radius: 20px;
    background:
        linear-gradient(115deg, rgba(22, 41, 77, 0.92), rgba(12, 21, 40, 0.86) 54%, rgba(36, 24, 73, 0.82)),
        var(--tech-panel);
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

.app-hero::after {
    position: absolute;
    top: -7rem;
    right: 8%;
    width: 18rem;
    height: 18rem;
    border: 1px solid rgba(67, 217, 255, 0.17);
    border-radius: 50%;
    box-shadow: 0 0 0 2rem rgba(67, 217, 255, 0.035), 0 0 0 4rem rgba(162, 119, 255, 0.025);
    content: "";
}

.app-hero-kicker {
    position: relative;
    z-index: 1;
    margin-bottom: 0.45rem;
    color: var(--tech-cyan);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.app-hero-title {
    position: relative;
    z-index: 1;
    margin: 0;
    color: #f4f8ff;
    font-size: clamp(1.65rem, 3vw, 2.55rem);
    font-weight: 800;
    letter-spacing: 0.02em;
}

.app-hero-subtitle {
    position: relative;
    z-index: 1;
    max-width: 1000px;
    margin: 0.7rem 0 0;
    color: #aebbd3;
    line-height: 1.65;
}
.workflow-window-title {
    margin: 0 0 1rem;
    padding: 0.7rem 1rem;
    border: 1px solid #c9dbe2;
    border-radius: 8px;
    background: #ffffff;
    color: #087f8c;
    font-size: 1.05rem;
    font-weight: 700;
}

/* Desktop-safe light theme: Streamlit native controls can otherwise inherit
   low-contrast dark-theme text when rendered by Qt WebEngine. */
:root {
    --tech-bg: #f4f7f9;
    --tech-panel: #ffffff;
    --tech-panel-strong: #ffffff;
    --tech-border: #d5e0e6;
    --tech-cyan: #087f8c;
    --tech-blue: #1769aa;
    --tech-purple: #5b6470;
    --tech-text: #17232d;
    --tech-muted: #5b6b76;
}
.stApp, [data-testid="stAppViewContainer"], .main, .block-container {
    background: #f4f7f9 !important;
    color: #17232d !important;
}
.block-container { padding-top: 1.25rem; }
.stApp p, .stApp label, .stApp span, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
    color: #17232d;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #d5e0e6 !important;
    box-shadow: 2px 0 10px rgba(27, 55, 71, 0.08) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stExpander summary p {
    color: #243642 !important;
}
[data-testid="stSidebar"] .stExpander,
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #d5e0e6 !important;
    box-shadow: 0 2px 8px rgba(27, 55, 71, 0.06) !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #ffffff !important;
    color: #17232d !important;
    border-color: #b8c8d1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] input::placeholder {
    color: #17232d !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background: #eef4f6 !important;
    color: #17313d !important;
    border: 1px solid #b9cbd3 !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    background: #dceff1 !important;
    color: #075e69 !important;
    border-color: #087f8c !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    background: #087f8c !important;
    color: #ffffff !important;
    border: 0 !important;
    box-shadow: 0 3px 10px rgba(8, 127, 140, 0.22) !important;
}
[data-testid="stMetric"], [data-testid="stAlert"] {
    background: #ffffff !important;
    color: #17232d !important;
    border-color: #d5e0e6 !important;
    box-shadow: 0 2px 8px rgba(27, 55, 71, 0.06) !important;
}
.app-hero {
    background: #ffffff !important;
    border: 1px solid #c9dbe2 !important;
    box-shadow: 0 3px 12px rgba(27, 55, 71, 0.08) !important;
}
.app-hero::after { display: none; }
.app-hero-kicker { color: #087f8c !important; }
.app-hero-title { color: #17232d !important; }
.app-hero-subtitle { color: #52636e !important; }
[data-testid="stDataFrame"], .stDataFrame {
    background: #ffffff !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

PARAMETER_HELP: dict[str, str] = {
    "compute_backend": "计算后端。自动优先使用 CUDA GPU；强制 GPU 在不可用时直接报错；仅 CPU 用于排查 GPU 或兼容性问题。推荐：自动｜优先 GPU。",
    "denoise_voxel": "去噪前的体素网格边长，单位跟随 PLY 坐标单位。每个体素内的点会聚合为一个代表点。当前数据推荐 0.15；减小可保留细节但更慢，增大可提高速度但可能抹平薄边。",
    "use_statistical": "统计离群点滤波开关。根据每个点到 K 个邻居的平均距离，删除明显远离整体分布的飞点。推荐开启。",
    "statistical_neighbors": "统计滤波的邻居 K。K 越大越稳定但越容易把局部细节视为噪声。当前数据推荐 24；稀疏点云可用 12–20，密集点云可用 24–40。",
    "statistical_std": "统计距离阈值倍率。保留平均邻居距离不超过全局均值 + 倍率×标准差的点。当前推荐 1.5；调小到 1.0–1.3 会更严格，调大到 2.0–2.5 可减少误删边缘。",
    "use_radius": "半径离群点滤波开关。在给定半径内邻居不足的点会被删除，特别适合处理孤立飞点。当前推荐开启；它只作用于每个独立视角，不会强行删除融合后的互补覆盖。",
    "radius": "半径滤波搜索半径，单位跟随 PLY 坐标单位。必须与点间距同量级。当前推荐 0.25；太小滤不掉飞点，太大可能删除薄结构。",
    "radius_min_points": "半径内要求的最少邻居数。当前推荐 6；增大过滤更严格，减小更保守。应和半径一起调，不能只看一个值。",
    "use_roi": "是否只用视野中的多面体标定块做 ICP 精配准。开启可避免把被测物表面误当成匹配依据。推荐开启。",
    "roi_min": "多面体标定块 ROI 的最小边界。固定 5/6 工位会先应用工位先验再按参考坐标裁剪；7/8 等通用数量路径会在每个输入坐标系直接裁剪，因此每个视角都必须落在同一 ROI 约定内。不要大范围包含主体被测物。当前推荐 X=4.0、Y=2.0、Z=-6.5。",
    "roi_max": "多面体标定块 ROI 的最大边界。当前推荐 X=17.5、Y=17.0、Z=8.0。若某个视角 ROI 点数不足，优先沿对应方向扩大 0.5–1.0；若混入主体，缩小范围。",
    "roi_minimum_points": "ROI 内允许配准的最少点数。点数太少无法稳定估计位姿。当前推荐 800；ROI 较稀疏可降到 300–500，但不建议低于 30。",
    "exclude_calibration_from_fusion": "标定块只用于计算各视角刚体位姿，默认不写入最终被测件融合云。关闭后可用于排查 ROI，但标定块会污染顶视图和截面，正式测量不建议关闭。",
    "exclusion_margin": "在标定块 ROI 外额外排除的边界余量，单位跟随 PLY。用于避免标定块边缘和反射散点残留；过大可能误删靠近标定块的工件点。当前推荐 0.03。",
    "level_reference_plane": "使用标定块 ROI 中最接近水平的基准面，自动旋转整个融合点云，使该基准面与 XY 平面平行；这是后续真实竖直横截面的必要步骤。",
    "leveling_distance": "标定块基准面 RANSAC 的最大点到平面距离。应略大于标定块表面噪声；过大可能把相邻面混进来。当前推荐 0.12。",
    "leveling_min_points": "用于调平的单个水平面至少需要的点数。过低会误选小噪声平面；当前推荐 500。",
    "registration_voxel": "标定块精配准专用体素大小，只影响多面体标定块 ICP 的计算点数，不直接改变最终融合分辨率。当前推荐 0.35；减小可提高精度但显著变慢，增大可提速但可能错过细节。",
    "use_global": "通用数量入口的标定块粗配准开关。启用时 FPFH/RANSAC 只在各视角的多面体标定块 ROI 上运行；工件全云只做质量诊断，不参与位姿求解。",
    "six_station_pose_prior": "固定五/六工位先验：五帧使用 -50°、-28°、-10°、参考、+37°；六帧额外包含 +47°。先验只由多面体标定块 ROI 小范围修正和闭环优化，不使用工件或 CAD。",
    "reference_count": "选择参考视角的规则：程序选择顶点数最接近该目标值的 PLY 作为固定参考。当前数据推荐 1,114,000；换数据后应填最接近目标参考扫描点数的值。",
    "normal_factor": "法线搜索半径 = 配准体素大小 × 此倍率。法线用于点到平面 ICP。当前推荐 2.5；表面噪声大可提高到 3–4，细节密集可降到 1.5–2.0。",
    "feature_factor": "FPFH 特征搜索半径 = 配准体素大小 × 此倍率。半径太小特征不稳定，太大特征变得不局部。当前推荐 5.0；一般可在 4–8 调整。",
    "global_distance": "RANSAC 的对应距离 = 配准体素大小 × 此倍率。用于判断两个特征匹配是否几何一致。当前推荐 1.5；初始位姿差大可提高到 2–2.5，但误匹配也会增加。",
    "global_iterations": "每次 RANSAC 粗配准的最大迭代次数。次数越多找到正确姿态的概率越高，但 CPU 时间越长。当前推荐 100,000；失败时可提高到 200,000–500,000。",
    "global_attempts": "RANSAC 重试次数。每次使用不同随机采样寻找更好的粗配准结果。当前推荐 3；偶发失败可提高到 5，速度优先可降到 1–2。",
    "fallback_threshold": "低质量 ICP 的回退阈值，使用 ICP fitness 判断是否继续尝试其他已对齐视角。当前推荐 0.75；提高更可靠但更慢，降低更快但可能接受较差连接。",
    "skip_low_quality_views": "兼容旧入口。覆盖优先模式不会因工件全云重叠不足而删除视角；全云 fitness 只作为诊断指标。",
    "full_cloud_fitness": "工件全云一致性诊断指标。多角度拍摄用于补齐遮挡，覆盖不足不能作为删除视角或修改位姿的依据。",
    "calibration_quality_fitness": "标定块 ICP 的最低 fitness。它只评价固定标定块的重合程度；低于该值的视角不会进入位姿图或正式融合。当前建议 0.60–0.70。",
    "calibration_quality_rmse": "标定块 ICP 的最大 RMSE。RMSE 越大说明对应点残差越大，通常意味着标定块未正确重合。当前设备建议 0.16–0.20。",
    "skip_low_quality_calibration_views": "启用后，标定块 ICP 未通过 fitness 或 RMSE 闸门的视角会被排除，不参与位姿图和最终融合；建议正式测量保持开启。",
    "section_alignment_validation": "以目标截面为中心抽取多张薄截面，只在共同覆盖区域做拼接质量诊断。覆盖优先模式下不会因为重叠不足、局部轮廓不同或互补视角而剔除工位；只保留标定块位姿质量结果。",
    "section_alignment_p90": "共同横向覆盖区内的局部轮廓 P90 距离阈值。当前设备标准方案使用 0.115 mm；减小会更严格，增大会保留更多视角。它只用于验收，不会反向修正点云。",
    "station_pose_correction": "在内置标定工位矩阵基础上增加小范围 XYZ 欧拉角和 XYZ 平移修正。旋转和平移均在参考坐标系中应用；正常情况下保持 0。",
    "pose_calibration_voxel": "固定工位初调时，累计标定块目标点云的体素大小。减小可保留更多标定块细节，但计算更慢。",
    "pose_translation_distance": "固定工位锁旋转平移 ICP 的最大对应距离。先验偏差大时可适当增大，错配时应减小。",
    "pose_rigid_limit": "标定块初调允许刚性 ICP 在先验基础上修正的最大幅度。范围过大可能使对称标定块翻到错误面。",
    "graph_voxel": "该阶段位姿图边配准使用的体素大小。越小越精细但越慢；应小于或等于基础配准体素。",
    "graph_distance": "该阶段位姿图 ICP 的最大点对距离。过大容易跨层匹配，过小可能没有足够对应点。",
    "graph_iterations": "该阶段每条位姿图边执行 ICP 的最大迭代次数。",
    "graph_motion_limit": "位姿图优化后允许单工位接受的最大旋转和平移修正，超过限制时回退到优化前位姿。",
    "graph_fitness": "位姿图边的最低 fitness。相邻边负责保证图连通；闭环边通常应设置得更严格。",
    "graph_rmse": "位姿图边的最大 RMSE。边的 fitness 和 RMSE 必须同时通过；超出阈值的边只保留诊断指标，不会进入全局优化。",
    "section_alignment_thickness": "用于工位验收的单张薄截面厚度，不改变最终输出截面厚度。越薄越能暴露错层，但点数会减少；它只用于验收，不会补齐最终截面缺口。",
    "section_alignment_overlap": "比较两个工位截面时判定为有效重叠的最大距离。必须大于或等于 P90 阈值。",
    "section_alignment_points": "待验工位在共同横向区域内至少需要多少匹配截面点才可建立可信连接。点云稀疏时可降低。",
    "section_alignment_probes": "以目标截面为中心抽取的平行薄截面数量，必须为奇数。增加可避免仅凭单张截面误判。",
    "section_alignment_spacing": "相邻验收截面的平面偏移间距。用于检查目标位置前后的拼接一致性。",
    "section_alignment_views": "多截面诊断中记录的最低工位数。覆盖优先模式下不用于删除互补视角，只用于提示证据是否充分。",
    "icp_distance": "ICP 最大对应距离，单位跟随 PLY 坐标单位。只有距离小于该值的点对才参与精配准。当前推荐 0.90；匹配不上可提高，错配多或结果漂移可降低。",
    "icp_iterations": "ICP 最大迭代次数。每次迭代重新寻找对应点并更新刚体变换。当前推荐 100；收敛不足可提高到 150–300，已经稳定时不必继续增大。",
    "multiscale_icp": "先用较宽对应距离捕获整体，再逐步收紧到细节的多阶段 ICP，可降低跨面错配和局部重影；关闭后退回单阶段 ICP。正式测量建议开启。",
    "point_to_plane": "ICP 误差模型。点到平面通常比点到点收敛更快、更适合平面/多面体标定块，但依赖法线质量。当前推荐开启；法线异常时可关闭进行排查。",
    "fusion_voxel": "最终原始全分辨率点云的统一体素融合尺寸。固定 5/6 和通用 7/8 视角都会保留所有标定块质量合格视角，默认 0.03 mm。",
    "section_angle": "截面法向在 XY 平面投影的方位角。框选后由三维主轴自动回填；它只表示水平转向，不能单独描述点云的上下倾斜。",
    "section_tilt": "截面法向相对 XY 平面的俯仰角。0° 表示法向完全水平；非零表示截面会随目标点云的三维倾角一起倾斜。框选后自动拟合。",
    "section_offset": "切平面沿法线方向的有符号偏移，平面满足 点·法线 = 偏移。当前推荐 9.0；向目标截面方向移动平面直到切中工件。",
    "section_thickness": "切片带宽。保留距离切平面不超过厚度/2 的点。当前设备正式测量推荐 0.15；增大厚度不会补回缺失数据，超过约 0.20 还可能把前后相邻型面叠加成重影。",
    "section_horizontal": "截面平面内横向筛选范围。当前推荐 -30 到 30；应覆盖完整目标截面。PNG 会根据实际截面点自动缩放，不再因范围过大而把轮廓缩成小点。",
    "section_vertical": "截面平面内竖向坐标范围，同时也是竖向筛选范围。该方向位于倾斜截面内，并尽量朝向全局 +Z；当前推荐 -8 到 10。",
    "section_point_size": "保存 PNG 截面图中的点大小，不改变点云数据。当前推荐 0.45；点过密可减小，点稀疏或图像分辨率高可增大。",
    "section_region_selection": "开启后，在 3D 预览下方点击“选择区域”，再沿被测件走向拖拽矩形框选择一段主轴拟合区域，并排除标定块。程序会取框内完整 XYZ 点拟合三维主轴，以该主轴作为截面法向；青色框只用于定方向和位置，不会裁掉最终截面轮廓。",
    "show_preview": "首次处理前显示单帧参考云用于框选截面区域；该组处理完成后自动切换为拼接后融合点云的交互式 3D 预览。",
    "preview_level": "预览采样等级：1级 50,000点/0.20体素；2级 100,000点/0.10；3级 200,000点/0.06；4级最多 2,000,000点/0.01体素。推荐先用3级，最终核查用4级。实际显示点数还受原始文件和去噪结果限制，点径会按实际点数自动缩放。",
    "preview_denoise": "是否在 3D 预览中实时应用当前去噪参数。开启后，预览会先执行去噪体素、统计滤波和可选半径滤波，再按预览等级采样；参数提交后自动重算。默认开启，超清 2,000,000 点上限预览会更慢。",
    "preview_surface": "点击后在“点云渲染”和“表面渲染”之间切换。表面渲染会从当前预览点云重建三角网格，并按高度着色；最高档原始预览最多 2,000,000 点，网格重建最多使用 500,000 个点和 1,500,000 个三角面。采样仍优先使用 CUDA，网格重建由 Open3D CPU 完成，浏览器 WebGL 使用 GPU 绘制表面。",
    "preview_projection": "默认开启正交等比例投影：X、Y、Z 坐标单位保持 1:1:1，且没有透视造成的近大远小，可避免工件看起来被拉伸。关闭后使用透视投影，立体感更强，但近处会显得更大。",
}


def render_parameter_manual() -> None:
    with st.expander("处理流程与调参顺序（点击展开）", expanded=False):
        st.markdown(
            """
**点云流程**：

1. 点云去噪；
2. 用旁边的多面体标定块 ROI 建立固定工位或通用视角刚体位姿；
3. 固定 5/6 工位仅用标定块进行闭环位姿图优化；
4. 将各个标定块质量合格视角的目标物体点云全部变换到统一坐标系；
5. 对所有互补视角执行原始全分辨率覆盖融合，重叠区域只生成质量诊断；
6. 仅清理融合云中小型、紧凑、弱支持且与主体断开的孤立组件；
7. 从清理后的融合云生成多截面诊断并直接导出最终截面，不再清噪；
8. 才导入 CAD，使 CAD 拟合固定点云并输出偏差。

**原则**：步骤 1～7 不读取 CAD；截面只用于验收，不会旋转、平移、拉伸或重排点云。出现明显双线时，优先检查标定块 ROI 和截面审核图，而不是用 CAD 反向改点云。

**普通用户只需调整**：

| 情况 | 建议操作 |
| --- | --- |
| 常规数据 | 使用“标准自动（推荐）”，框选目标区域后处理。 |
| 截面有平行双线或重影 | 选择“高精度｜截面有重影”，查看标定块位姿和重叠区域诊断；互补覆盖不会因不重叠而删除。 |
| 合理局部轮廓缺失 | 首先检查是否所有标定块质量合格视角都参与融合，再检查相机是否真正看到缺失区域。 |
| 轮廓断裂 | 先查看“截面缺口来源诊断”和融合云基线截面；不要用增大厚度、插值或 CAD 反向调整来补线。 |
| 标定块配准不稳定 | 在专家模式检查 ROI 是否完整覆盖多面体且不混入被测件，再小幅调整标定块 ICP 参数。 |
            """
        )
    return

    with st.expander("参数原理、推荐值与调节方法（点击展开）", expanded=False):
        st.markdown(
            """
**说明**：数值单位跟随 PLY 文件坐标单位；如果扫描数据以毫米保存，下面的距离参数就是毫米。推荐值针对当前 `media/1` 至 `media/4` 数据，换设备、换扫描距离或换点密度后应重新检查。

### 0. GPU 与预览

| 参数 | 当前推荐值 | 原理与调节建议 |
| --- | --- | --- |
| 处理计算后端 | 自动｜优先 GPU | GPU 加速体素、去噪、ROI、ICP、原始点融合和切片；Open3D CPU 执行两阶段鲁棒位姿图全局优化。 |
| 显示当前组低采样 3D 预览 | 关闭；检查时开启 | 只影响网页预览，不改变正式结果。建议先开预览确认红色 ROI。 |
| 3D 预览采样等级 | 3 级｜精细 | 1级最快，2级常规，3级推荐，4级最多 2,000,000 点、0.01 体素，适合最终检查但更占内存和浏览器显存。 |
| 实时应用去噪参数到 3D 预览 | 开启 | 预览先按当前去噪体素、统计滤波和半径滤波处理，再按预览等级采样；调整参数并提交后自动重算。超清档较慢时可暂时关闭。 |
| 正交等比例显示（防止拉伸） | 开启 | X、Y、Z 坐标单位按 1:1:1 显示，且没有透视近大远小；适合核对点云真实形状。关闭后使用透视投影，立体感更强但近处会显得更大。 |
| 切换为表面渲染 | 默认点云渲染 | 点击按钮将散点预览切换为三角网格表面，适合观察连续形状。最高档原始预览最多 2,000,000 点；网格最多使用 500,000 个重建点与 1,500,000 个三角面，网格重建由 Open3D CPU 完成，WebGL 表面显示使用 GPU。 |

### 1. 去噪

| 参数 | 当前推荐值 | 原理与调节建议 |
| --- | --- | --- |
| 去噪体素大小 | 0.15 | 先将空间划分为体素并聚合点。减小保留细节但变慢；增大更快但可能抹平薄边。 |
| 统计离群点滤波 | 开启 | 依据点到邻居的平均距离清除孤立飞点，建议保持开启。 |
| 统计邻居数 | 24 | K 越大越稳定但越不敏感于局部细节。稀疏点云 12–20，密集点云 24–40。 |
| 统计标准差倍率 | 1.5 | 阈值为均值 + 倍率×标准差。飞点多降到 1.0–1.3；边缘被误删升到 2.0–2.5。 |
| 半径离群点滤波 | 开启 | 在每个独立视角拼接前清理局部孤立飞点；只删除邻域极少的点，不对融合后的互补覆盖做强制清理。 |
| 半径滤波半径 | 0.25 | 搜索邻域半径。应与正常点间距同量级；太小无效，太大可能删除薄结构。 |
| 半径内最少点数 | 6 | 半径内少于该数量的点会被删除。增大更严格，减小更保守。 |

### 2. 多面体标定块 ROI

| 参数 | 当前推荐值 | 原理与调节建议 |
| --- | --- | --- |
| 仅使用多面体标定块 | 开启 | 在参考坐标系的轴对齐框内提取多面体，仅用它做 ICP，避免主体表面干扰。 |
| X 范围 | 4.0 ～ 17.5 | ROI 最小/最大 X。必须完整覆盖标定块；点数不足时沿对应方向扩大 0.5～1.0。 |
| Y 范围 | 2.0 ～ 17.0 | 同上。范围过大混入主体会降低配准可靠性。 |
| Z 范围 | -6.5 ～ 8.0 | 同上。应覆盖标定块的全部高度。 |
| ROI 最少点数 | 800 | 低于此数量认为标定块不足。稀疏数据可降到 300～500，不建议低于 30。 |
| 使用标定块基准面自动调平 | 开启 | 从标定块 ROI 中拟合近水平基准面，并旋转全部点云建立统一参考坐标。横截面法向仍由框选区域的三维主轴拟合，不会被强制锁定到全局 Z。 |
| 调平基准面距离阈值 | 0.12 | RANSAC 拟合标定块基准面的最大点到平面距离。应略大于表面噪声；过大会把相邻面混入。 |
| 调平基准面最少点数 | 500 | 基准面至少需要的拟合点数。过低可能把小噪声面误认为调平基准。 |

### 3. 拼接与配准

| 参数 | 当前推荐值 | 原理与调节建议 |
| --- | --- | --- |
| 配准体素大小 | 0.35 | 只用于配准阶段降采样。越小越精细、越慢；越大越快、越容易丢细节。 |
| 固定五/六工位位姿图 | 开启 | 5 帧使用 -50°、-28°、-10°、参考、+37°；6 帧增加 +47°。7/8 等其他数量不套用缺失的先验，改用各视角标定块 ROI 做粗定位。 |
| 标定块 FPFH + RANSAC 粗配准 | 开启 | 通用数量路径只在各视角标定块 ROI 上做粗定位；固定 5/6 工位先用内置姿态先验，再由标定块 ICP 修正。工件全云不参与位姿求解。 |
| 参考视角目标点数 | 1,114,000 | 选择点数最接近该值的文件作固定参考，当前数据建议保持。 |
| 法线半径倍率 | 2.5 | 法线半径 = 配准体素×倍率。噪声大可升到 3～4，细节密集可降到 1.5～2。 |
| FPFH 半径倍率 | 5.0 | FPFH 特征邻域半径 = 配准体素×倍率。通常在 4～8 调整。 |
| RANSAC 距离倍率 | 1.5 | 允许的特征对应距离 = 配准体素×倍率。初始差异大可升到 2～2.5，但误匹配会增加。 |
| RANSAC 最大迭代 | 100,000 | 随机采样次数上限。粗配准偶发失败可升到 200,000～500,000。 |
| 全局配准重试次数 | 3 | 使用不同随机采样重复 RANSAC。偶发失败可升到 5，速度优先可降到 1～2。 |
| 低质量回退 ICP 阈值 | 0.75 | ICP fitness 低于此值时尝试其他已对齐视角。提高更保守可靠，降低更快。 |
| 跳过标定块质量不合格的视角 | 开启 | 只有标定块 ROI 的点数、fitness 或 RMSE 未通过时才排除视角；工件全云 fitness 仅作诊断，不因遮挡或互补覆盖而删除。 |
| 全云最低 fitness | 0.70 | 工件全云一致性诊断参考值。它不再作为覆盖优先融合的删除闸门；用于发现残余错位、重影和可能的错误位姿。 |
| ICP 最大对应距离 | 0.90 | 小于该距离的点对参与精配准。匹配不上升大；错配或漂移降低。 |
| ICP 最大迭代 | 100 | 重复寻找对应点并更新刚体变换。未收敛可升到 150～300。 |
| 使用点到平面 ICP | 开启 | 利用法线计算点到平面的误差，适合多面体平面；法线异常时可关闭。 |
| 融合输出体素大小 | 0.03 | 使用各视角已经去噪的点云进行统一融合；重叠区域只做诊断，互补独有点保留。 |

### 4. 三维法向横截面

| 参数 | 当前推荐值 | 原理与调节建议 |
| --- | --- | --- |
| 法向方位角 | 0° | 截面法向在 XY 平面投影的方向；框选后根据区域三维主轴自动回填。 |
| 法向俯仰角 | 0° | 截面法向相对 XY 平面的上下倾角；点云整体倾斜时该值会自动为非零，使横截面真正垂直于目标三维主轴。 |
| 平面偏移 | 9.0 | 平面满足 `点·法线 = 偏移`。沿法线移动到目标位置。 |
| 顶视矩形框选自动选面 | 开启 | 在预览下方顶视图中框选目标区域。程序会回取框内完整 XYZ 点拟合三维主轴，以主轴作为法向，生成穿过框选几何中心并垂直于倾斜点云的横截面。 |
| 切片厚度 | 0.15 | 保留距离平面 ±厚度/2 内的点。当前正式测量保持 0.15；增大厚度不会补回缺失数据，超过约 0.20 可能叠加前后相邻型面。 |
| 截面横向范围 | -30 ～ 30 | 控制横向筛选和 PNG 显示窗口，应覆盖目标截面。 |
| 截面竖向范围 | -8 ～ 10 | 控制 Z 向筛选和 PNG 显示窗口，应覆盖被测高度。 |
| 截面图点大小 | 0.45 | 只影响 PNG 的绘制点大小，不改变切片点云数据。 |
| 融合后孤立组件清理 | 自动执行 | 只清理小型、紧凑、弱支持且与主体断开的融合云组件；多视角支撑的小组件和单视角独有覆盖保留。最终截面直接从清理后的融合云提取。 |

**推荐调参顺序**：先固定 GPU、单视角去噪和标定块 ROI；确认已自动调平；打开预览并仅框选被测件、排除标定块；再确认每个参与融合视角的全云 fitness；最后微调截面厚度、范围和偏移。不要同时大幅修改多个配准参数。
            """
        )


def _initialize_section_control_state() -> None:
    defaults = {
        "section_region_selection": True,
        "section_angle_degrees": 0.0,
        "section_tilt_degrees": 0.0,
        "section_plane_offset": 9.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _pointcloud_data_signature(
    groups: Mapping[str, list[Path]],
    selected_group_names: list[str],
) -> str:
    files: list[dict[str, object]] = []
    for group_name in selected_group_names:
        for path in groups.get(group_name, []):
            try:
                metadata = path.stat()
                size = int(metadata.st_size)
                modified_ns = int(metadata.st_mtime_ns)
            except OSError:
                size = -1
                modified_ns = -1
            files.append(
                {
                    "group": group_name,
                    "path": str(path.resolve()),
                    "size": size,
                    "modified_ns": modified_ns,
                }
            )
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clear_pointcloud_page_state(state: MutableMapping[str, Any]) -> None:
    picker_revision = int(state.get("section_picker_revision", 0)) + 1
    for key in (
        "pipeline_results",
        "pipeline_result_data_signature",
        "pipeline_completion_message",
        "section_picker_group",
        "section_picker_region",
        "section_picker_locked",
        "section_picker_drawing",
        "section_picker_last_signature",
        "pending_section_plane",
        "pending_section_offset",
    ):
        state.pop(key, None)
    for key in list(state):
        if str(key).startswith(
            ("cad-", "section-offset-scan-", "section_offset_scan_")
        ):
            state.pop(key, None)
    state["section_picker_revision"] = picker_revision
    state["section_angle_degrees"] = 0.0
    state["section_tilt_degrees"] = 0.0
    state["section_plane_offset"] = 9.0


def _section_target_region_from_state() -> Bounds2D | None:
    region = st.session_state.get("section_picker_region")
    if not isinstance(region, dict):
        return None
    if int(region.get("implementation_version", 0)) < 3:
        return None
    try:
        target_region = Bounds2D(
            x_min=float(region["x_min"]),
            x_max=float(region["x_max"]),
            y_min=float(region["y_min"]),
            y_max=float(region["y_max"]),
        )
        target_region.validate("目标截面区域")
    except (KeyError, TypeError, ValueError):
        return None
    return target_region


def _build_simple_processing_config() -> ProcessingConfig:
    st.sidebar.subheader("自动拼接")
    st.sidebar.caption("选择处理方案后，程序会自动配置去噪、标定块位姿图、覆盖优先融合与截面诊断。")
    preset_label = st.sidebar.selectbox(
        "处理方案",
        list(SIMPLE_STITCHING_PRESETS),
        key="simple_stitching_preset",
    )
    preset = SIMPLE_STITCHING_PRESETS[preset_label]
    st.sidebar.caption(str(preset["description"]))
    strictness = st.sidebar.slider(
        "截面异常识别严格度",
        min_value=1,
        max_value=5,
        value=3,
        help="只影响重叠区域质量诊断的严格度。覆盖优先模式不会因为工件不重叠或局部轮廓不同而删除互补视角。一般保持 3。",
        key="simple_stitching_strictness",
    )
    strictness_labels = {
        1: "诊断宽松",
        2: "偏宽松",
        3: "质量诊断（推荐）",
        4: "偏严格",
        5: "诊断严格",
    }
    st.sidebar.caption(f"当前验收策略：{strictness_labels[strictness]}")
    section_thickness = st.sidebar.number_input(
        "最终截面厚度",
        min_value=0.02,
        max_value=2.0,
        value=0.15,
        step=0.01,
        help="只影响最终输出截面的取点厚度，不改变用于错层验收的薄截面厚度。当前正式测量推荐 0.15；加厚不能补回原始缺口，超过约 0.20 可能产生重影。",
        key="simple_section_thickness",
    )
    if section_thickness > 0.20:
        st.sidebar.warning("当前厚度超过 0.20，仅建议用于复核；正式测量建议保持 0.15，避免前后型面叠加成重影。")
    st.sidebar.caption("操作顺序：选择数据 → 在 3D 预览中框选目标区域 → 开始处理。CAD 仅在结果后的第 8 步导入。")

    st.session_state["section_region_selection"] = True
    target_region = _section_target_region_from_state()
    if target_region is None:
        st.sidebar.warning("还未框选截面区域，请在右侧 3D 预览中点击“选择区域”。")
    else:
        st.sidebar.success("截面区域已选择，可以开始处理。")
    st.sidebar.number_input(
        "截面平面偏移微调",
        min_value=-50.0,
        max_value=50.0,
        step=0.1,
        format="%.3f",
        key="section_plane_offset",
        help=PARAMETER_HELP["section_offset"],
    )
    st.sidebar.caption("仅沿当前三维法向平移切片平面；不会旋转、平移或重排点云。")

    strictness_offset = strictness - 3
    p90_factors = {1: 1.35, 2: 1.15, 3: 1.0, 4: 0.85, 5: 0.70}
    section_p90 = max(0.02, float(preset["section_p90"]) * p90_factors[strictness])
    section_overlap = max(float(preset["section_overlap"]), section_p90)
    section_overlap_points = max(
        100,
        int(preset["section_overlap_points"]) + strictness_offset * 100,
    )

    def adjusted_fitness(parameter_name: str) -> float:
        return float(np.clip(float(preset[parameter_name]) + strictness_offset * 0.03, 0.05, 0.95))

    denoise_voxel = float(preset["denoise_voxel"])
    graph_iterations = int(preset["graph_iterations"])
    return ProcessingConfig(
        denoise=DenoiseConfig(
            voxel_size=denoise_voxel,
            use_statistical_filter=True,
            statistical_neighbors=24,
            statistical_std_ratio=1.5,
            use_radius_filter=True,
            radius=0.25,
            radius_min_points=6,
        ),
        calibration=CalibrationConfig(
            use_roi=True,
            bounds=Bounds3D(
                x_min=4.0,
                x_max=17.5,
                y_min=2.0,
                y_max=17.0,
                z_min=-6.5,
                z_max=8.0,
            ),
            minimum_points=800,
            exclude_from_fusion=True,
            exclusion_margin=0.03,
            level_reference_plane=True,
            leveling_plane_distance=0.12,
            leveling_minimum_points=500,
        ),
        registration=RegistrationConfig(
            registration_voxel_size=float(preset["registration_voxel"]),
            use_global_registration=True,
            reference_vertex_count=1_114_000,
            normal_radius_factor=2.5,
            feature_radius_factor=5.0,
            global_distance_factor=1.5,
            global_max_iterations=100_000,
            global_attempts=3,
            global_random_seed=5,
            fallback_icp_fitness_threshold=0.75,
            skip_low_quality_views=True,
            minimum_full_cloud_fitness=float(np.clip(0.70 + strictness_offset * 0.03, 0.50, 0.90)),
            minimum_calibration_icp_fitness=float(
                np.clip(0.65 + strictness_offset * 0.02, 0.50, 0.90)
            ),
            maximum_calibration_icp_rmse=float(np.clip(0.16 - strictness_offset * 0.01, 0.05, 0.50)),
            skip_low_quality_calibration_views=True,
            coverage_preserving_fusion=True,
            use_section_alignment_validation=True,
            section_alignment_thickness=0.05,
            section_alignment_overlap_distance=section_overlap,
            section_alignment_max_p90_distance=section_p90,
            section_alignment_minimum_overlap_points=section_overlap_points,
            section_alignment_probe_count=5,
            section_alignment_probe_spacing=0.50,
            section_alignment_minimum_views=3,
            icp_max_correspondence_distance=0.90,
            icp_max_iterations=100,
            use_point_to_plane_icp=True,
            use_multiscale_icp=True,
            use_six_station_pose_prior=True,
            station_pose_corrections={},
            pose_prior_calibration_voxel_size=min(denoise_voxel, 0.15),
            pose_prior_translation_max_correspondence_distance=0.90,
            pose_prior_rigid_max_rotation_degrees=1.50,
            pose_prior_rigid_max_translation=0.30,
            calibration_graph_voxel_size=float(preset["graph_voxel"]),
            calibration_graph_max_correspondence_distance=float(preset["calibration_distance"]),
            calibration_graph_icp_max_iterations=graph_iterations,
            calibration_graph_max_rotation_degrees=1.50,
            calibration_graph_max_translation=0.30,
            calibration_graph_minimum_adjacent_fitness=adjusted_fitness(
                "calibration_adjacent_fitness"
            ),
            calibration_graph_minimum_loop_fitness=adjusted_fitness(
                "calibration_loop_fitness"
            ),
            calibration_graph_maximum_adjacent_rmse=0.14,
            calibration_graph_maximum_loop_rmse=0.16,
            use_calibration_geometry_constraints=True,
            calibration_geometry_voxel_size=0.08,
            calibration_geometry_plane_distance=0.05,
            calibration_geometry_minimum_plane_points=80,
            calibration_geometry_minimum_planes=3,
            calibration_geometry_max_planes=8,
            calibration_geometry_max_normal_angle_degrees=15.0,
            calibration_geometry_max_centroid_distance=2.50,
        ),
        section=SectionConfig(
            angle_degrees=float(st.session_state.get("section_angle_degrees", 0.0)),
            tilt_degrees=float(st.session_state.get("section_tilt_degrees", 0.0)),
            plane_offset=float(st.session_state.get("section_plane_offset", 9.0)),
            thickness=float(section_thickness),
            horizontal_min=-30.0,
            horizontal_max=30.0,
            vertical_min=-8.0,
            vertical_max=10.0,
            point_size=0.45,
            target_region=target_region,
            use_statistical_filter=False,
            statistical_neighbors=12,
            statistical_std_ratio=1.0,
            use_component_filter=False,
            component_cell_size=0.30,
            component_min_points=12,
        ),
        fusion_voxel_size=float(preset["fusion_voxel"]),
        compute_backend="auto",
    )


def build_processing_config() -> ProcessingConfig:
    _initialize_section_control_state()
    st.sidebar.header("处理参数")
    advanced_parameter_mode = st.sidebar.checkbox(
        "专家参数模式",
        value=False,
        key="advanced_parameter_mode",
        help="普通用户建议关闭。开启后才显示去噪、标定块 ROI、固定工位先验、标定块位姿图和截面验收的工程参数。",
    )
    if not advanced_parameter_mode:
        return _build_simple_processing_config()
    st.sidebar.warning("专家模式会显示全部工程参数；不确定参数含义时建议返回普通模式。")
    with st.sidebar.expander("0. GPU 计算后端", expanded=False):
        capabilities = get_gpu_capabilities()
        backend_label = st.selectbox("处理计算后端", list(COMPUTE_BACKEND_OPTIONS), index=0, help=PARAMETER_HELP["compute_backend"])
        compute_backend = COMPUTE_BACKEND_OPTIONS[backend_label]
        if capabilities.available:
            st.success(
                f"CUDA GPU 已就绪：{capabilities.device_name}（计算能力 {capabilities.compute_capability}，CUDA {capabilities.cuda_runtime_version}）"
            )
            st.caption("GPU 处理：体素采样、去噪、ROI、标定块 ICP、融合与切片；标定块位姿图全局优化保持 Open3D CPU 后端以保证稳定性。")
        elif compute_backend == "gpu":
            st.error(f"强制 GPU 模式不可用：{capabilities.reason}")
        else:
            st.warning(f"CUDA GPU 当前不可用，将使用 CPU 回退：{capabilities.reason}")
        st.caption("3D 预览采用 Plotly WebGL；浏览器启用硬件加速时，旋转、缩放和高度彩色渲染由 GPU 完成。")

    with st.sidebar.expander("1. 去噪", expanded=False):
        voxel_size = st.number_input("去噪体素大小", min_value=0.01, max_value=2.0, value=0.15, step=0.01, help=PARAMETER_HELP["denoise_voxel"])
        use_statistical_filter = st.checkbox("启用统计离群点滤波", value=True, help=PARAMETER_HELP["use_statistical"])
        statistical_neighbors = st.number_input("统计邻居数", min_value=2, max_value=200, value=24, step=1, help=PARAMETER_HELP["statistical_neighbors"])
        statistical_std_ratio = st.number_input("统计标准差倍率", min_value=0.05, max_value=10.0, value=1.5, step=0.05, help=PARAMETER_HELP["statistical_std"])
        use_radius_filter = st.checkbox("启用半径离群点滤波", value=True, help=PARAMETER_HELP["use_radius"])
        radius = st.number_input("半径滤波半径", min_value=0.01, max_value=5.0, value=0.25, step=0.01, help=PARAMETER_HELP["radius"])
        radius_min_points = st.number_input("半径内最少点数", min_value=1, max_value=200, value=6, step=1, help=PARAMETER_HELP["radius_min_points"])

    with st.sidebar.expander("2. 标定块 ROI", expanded=False):
        use_roi = st.checkbox("仅使用多面体标定块进行配准", value=True, help=PARAMETER_HELP["use_roi"])
        st.caption("该范围位于参考工位坐标系；程序先应用固定工位先验，再只在统一坐标系内裁剪多面体标定块。")
        column_left, column_right = st.columns(2)
        with column_left:
            x_min = st.number_input("X 最小", value=4.0, step=0.1, help=PARAMETER_HELP["roi_min"])
            y_min = st.number_input("Y 最小", value=2.0, step=0.1, help=PARAMETER_HELP["roi_min"])
            z_min = st.number_input("Z 最小", value=-6.5, step=0.1, help=PARAMETER_HELP["roi_min"])
        with column_right:
            x_max = st.number_input("X 最大", value=17.5, step=0.1, help=PARAMETER_HELP["roi_max"])
            y_max = st.number_input("Y 最大", value=17.0, step=0.1, help=PARAMETER_HELP["roi_max"])
            z_max = st.number_input("Z 最大", value=8.0, step=0.1, help=PARAMETER_HELP["roi_max"])
        minimum_points = st.number_input("ROI 最少点数", min_value=30, max_value=200_000, value=800, step=50, help=PARAMETER_HELP["roi_minimum_points"])
        exclude_calibration_from_fusion = st.checkbox(
            "标定块不进入最终融合点云",
            value=True,
            help=PARAMETER_HELP["exclude_calibration_from_fusion"],
        )
        exclusion_margin = st.number_input(
            "标定块排除边界余量",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.01,
            format="%.3f",
            disabled=not exclude_calibration_from_fusion,
            help=PARAMETER_HELP["exclusion_margin"],
        )
        level_reference_plane = st.checkbox(
            "使用标定块基准面自动调平",
            value=True,
            disabled=not use_roi,
            help=PARAMETER_HELP["level_reference_plane"],
        )
        leveling_plane_distance = st.number_input(
            "调平基准面距离阈值",
            min_value=0.01,
            max_value=1.0,
            value=0.12,
            step=0.01,
            disabled=not use_roi or not level_reference_plane,
            help=PARAMETER_HELP["leveling_distance"],
        )
        leveling_minimum_points = st.number_input(
            "调平基准面最少点数",
            min_value=30,
            max_value=50_000,
            value=500,
            step=50,
            disabled=not use_roi or not level_reference_plane,
            help=PARAMETER_HELP["leveling_min_points"],
        )
        if level_reference_plane and use_roi:
            st.caption("正式处理会先用标定块近水平面把整体坐标调平，再计算目标区域横截面；标定块不会进入目标区域截面。")

    with st.sidebar.expander("3. 标定块拼接", expanded=False):
        use_six_station_pose_prior = True
        st.success("5/6 视角使用工位先验 + 标定块位姿图；7/8 等其他数量使用标定块 ROI 粗配准 + ICP。")
        st.caption("步骤 1～7 只使用多面体标定块确定刚体位姿；工件全云只做诊断，不会参与位姿求解，也不会读取 CAD。")
        registration_voxel_size = st.number_input(
            "标定块精配准体素大小",
            min_value=0.02,
            max_value=1.0,
            value=0.35,
            step=0.01,
            help=PARAMETER_HELP["registration_voxel"],
        )
        icp_max_correspondence_distance = st.number_input(
            "标定块 ICP 最大对应距离",
            min_value=0.01,
            max_value=5.0,
            value=0.90,
            step=0.01,
            help=PARAMETER_HELP["icp_distance"],
        )
        icp_max_iterations = st.number_input(
            "标定块 ICP 最大迭代",
            min_value=1,
            max_value=1_000,
            value=100,
            step=5,
            help=PARAMETER_HELP["icp_iterations"],
        )
        use_multiscale_icp = st.checkbox(
            "启用粗到细多阶段 ICP",
            value=True,
            help=PARAMETER_HELP["multiscale_icp"],
        )
        use_calibration_geometry_constraints = st.checkbox(
            "启用标定块平面几何约束",
            value=True,
            help="先匹配标定块的多个稳定平面，再以该小范围位姿作为 ICP 初值；可降低多面体跨面错配。",
        )
        geometry_left, geometry_right = st.columns(2)
        with geometry_left:
            calibration_geometry_plane_distance = st.number_input(
                "平面拟合距离",
                min_value=0.005,
                max_value=1.0,
                value=0.05,
                step=0.005,
                format="%.3f",
                disabled=not use_calibration_geometry_constraints,
                help="标定块平面 RANSAC 的点到平面距离阈值；应略大于测量噪声。",
            )
            calibration_geometry_voxel_size = st.number_input(
                "几何约束体素",
                min_value=0.01,
                max_value=1.0,
                value=0.08,
                step=0.01,
                disabled=not use_calibration_geometry_constraints,
                help="只用于提取平面特征，不改变最终融合点云分辨率。",
            )
        with geometry_right:
            calibration_geometry_max_normal_angle_degrees = st.number_input(
                "平面最大法向角（°）",
                min_value=1.0,
                max_value=89.0,
                value=15.0,
                step=1.0,
                disabled=not use_calibration_geometry_constraints,
                help="源/目标标定块平面匹配的法向角门限。",
            )
            calibration_geometry_max_centroid_distance = st.number_input(
                "平面中心最大距离",
                min_value=0.05,
                max_value=10.0,
                value=2.50,
                step=0.05,
                disabled=not use_calibration_geometry_constraints,
                help="源/目标平面中心匹配门限；过小会丢掉局部可见的有效平面。",
            )
        quality_left, quality_right = st.columns(2)
        with quality_left:
            minimum_calibration_icp_fitness = st.number_input(
                "标定块最低 fitness",
                min_value=0.05,
                max_value=1.0,
                value=0.65,
                step=0.01,
                format="%.3f",
                help=PARAMETER_HELP["calibration_quality_fitness"],
            )
        with quality_right:
            maximum_calibration_icp_rmse = st.number_input(
                "标定块最大 RMSE",
                min_value=0.001,
                max_value=5.0,
                value=0.16,
                step=0.01,
                format="%.3f",
                help=PARAMETER_HELP["calibration_quality_rmse"],
            )
        skip_low_quality_calibration_views = st.checkbox(
            "标定块质量不合格时排除视角",
            value=True,
            help=PARAMETER_HELP["skip_low_quality_calibration_views"],
        )
        fusion_voxel_size = st.number_input(
            "最终融合体素大小",
            min_value=0.02,
            max_value=0.035,
            value=0.03,
            step=0.005,
            format="%.3f",
            help=PARAMETER_HELP["fusion_voxel"],
            key="fusion_voxel_size_calibration_only_v1",
        )
        st.caption("标定块质量合格的视角会重新读取原始全分辨率点云并全部参与覆盖融合；重叠区和截面共识只生成诊断，不删除互补覆盖。")

        use_global_registration = True
        reference_vertex_count = 1_114_000
        normal_radius_factor = 2.5
        feature_radius_factor = 5.0
        global_distance_factor = 1.5
        global_max_iterations = 100_000
        global_attempts = 3
        fallback_icp_fitness_threshold = 0.75
        skip_low_quality_views = True
        minimum_full_cloud_fitness = 0.70
        use_point_to_plane_icp = True

    with st.sidebar.expander("4. 工位先验与标定块初调", expanded=False):
        st.caption("默认值对应当前设备标定结果。只有设备角度、安装位置或采集工位发生变化时才需要修改。")
        pose_prior_calibration_voxel_size = st.number_input(
            "初调标定块体素大小",
            min_value=0.01,
            max_value=1.0,
            value=0.15,
            step=0.01,
            disabled=not use_six_station_pose_prior,
            help=PARAMETER_HELP["pose_calibration_voxel"],
        )
        pose_prior_translation_max_correspondence_distance = st.number_input(
            "锁旋转平移 ICP 最大对应距离",
            min_value=0.01,
            max_value=5.0,
            value=0.90,
            step=0.01,
            disabled=not use_six_station_pose_prior,
            help=PARAMETER_HELP["pose_translation_distance"],
        )
        pose_limit_left, pose_limit_right = st.columns(2)
        with pose_limit_left:
            pose_prior_rigid_max_rotation_degrees = st.number_input(
                "初调最大旋转（°）",
                min_value=0.01,
                max_value=20.0,
                value=1.50,
                step=0.05,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["pose_rigid_limit"],
            )
        with pose_limit_right:
            pose_prior_rigid_max_translation = st.number_input(
                "初调最大平移",
                min_value=0.01,
                max_value=10.0,
                value=0.30,
                step=0.01,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["pose_rigid_limit"],
            )
        use_custom_station_pose_corrections = st.checkbox(
            "启用各工位先验矩阵修正",
            value=False,
            disabled=not use_six_station_pose_prior,
            help=PARAMETER_HELP["station_pose_correction"],
        )
        station_pose_corrections: dict[str, dict[str, float]] = {}
        if use_custom_station_pose_corrections and use_six_station_pose_prior:
            station_items = list(STATION_POSE_CORRECTION_OPTIONS.items())
            station_tabs = st.tabs([display_name for display_name, _ in station_items])
            for station_tab, (_, station_label) in zip(station_tabs, station_items, strict=True):
                with station_tab:
                    rotation_column, translation_column = st.columns(2)
                    with rotation_column:
                        st.number_input(
                            "绕 X 修正（°）",
                            min_value=-10.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.05,
                            format="%.3f",
                            key=f"station_pose_{station_label}_rotation_x_degrees",
                        )
                        st.number_input(
                            "绕 Y 修正（°）",
                            min_value=-10.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.05,
                            format="%.3f",
                            key=f"station_pose_{station_label}_rotation_y_degrees",
                        )
                        st.number_input(
                            "绕 Z 修正（°）",
                            min_value=-10.0,
                            max_value=10.0,
                            value=0.0,
                            step=0.05,
                            format="%.3f",
                            key=f"station_pose_{station_label}_rotation_z_degrees",
                        )
                    with translation_column:
                        st.number_input(
                            "X 平移修正",
                            min_value=-5.0,
                            max_value=5.0,
                            value=0.0,
                            step=0.01,
                            format="%.3f",
                            key=f"station_pose_{station_label}_translation_x",
                        )
                        st.number_input(
                            "Y 平移修正",
                            min_value=-5.0,
                            max_value=5.0,
                            value=0.0,
                            step=0.01,
                            format="%.3f",
                            key=f"station_pose_{station_label}_translation_y",
                        )
                        st.number_input(
                            "Z 平移修正",
                            min_value=-5.0,
                            max_value=5.0,
                            value=0.0,
                            step=0.01,
                            format="%.3f",
                            key=f"station_pose_{station_label}_translation_z",
                        )
            correction_fields = (
                "rotation_x_degrees",
                "rotation_y_degrees",
                "rotation_z_degrees",
                "translation_x",
                "translation_y",
                "translation_z",
            )
            for station_label in STATION_POSE_CORRECTION_OPTIONS.values():
                correction = {
                    field_name: float(st.session_state.get(f"station_pose_{station_label}_{field_name}", 0.0))
                    for field_name in correction_fields
                }
                if any(abs(value) > 1e-12 for value in correction.values()):
                    station_pose_corrections[station_label] = correction
            st.caption("修正量会叠加到内置工位标定矩阵；各工位标签页中的参数会同时保存并参与处理。")

    with st.sidebar.expander("5. 标定块位姿图", expanded=False):
        calibration_graph_left, calibration_graph_right = st.columns(2)
        with calibration_graph_left:
            calibration_graph_voxel_size = st.number_input(
                "标定图体素",
                min_value=0.01,
                max_value=1.0,
                value=0.12,
                step=0.01,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_voxel"],
            )
            calibration_graph_max_correspondence_distance = st.number_input(
                "标定图对应距离",
                min_value=0.01,
                max_value=5.0,
                value=0.40,
                step=0.01,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_distance"],
            )
            calibration_graph_icp_max_iterations = st.number_input(
                "标定图 ICP 迭代",
                min_value=1,
                max_value=2_000,
                value=200,
                step=10,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_iterations"],
            )
        with calibration_graph_right:
            calibration_graph_max_rotation_degrees = st.number_input(
                "标定图最大旋转（°）",
                min_value=0.01,
                max_value=20.0,
                value=1.50,
                step=0.05,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_motion_limit"],
            )
            calibration_graph_max_translation = st.number_input(
                "标定图最大平移",
                min_value=0.01,
                max_value=10.0,
                value=0.30,
                step=0.01,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_motion_limit"],
            )
        calibration_fitness_left, calibration_fitness_right = st.columns(2)
        with calibration_fitness_left:
            calibration_graph_minimum_adjacent_fitness = st.number_input(
                "相邻边最低 fitness",
                min_value=0.01,
                max_value=1.0,
                value=0.35,
                step=0.01,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_fitness"],
                key="calibration_graph_adjacent_fitness",
            )
        with calibration_fitness_right:
            calibration_graph_minimum_loop_fitness = st.number_input(
                "闭环边最低 fitness",
                min_value=0.01,
                max_value=1.0,
                value=0.45,
                step=0.01,
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_fitness"],
                key="calibration_graph_loop_fitness",
            )
        calibration_rmse_left, calibration_rmse_right = st.columns(2)
        with calibration_rmse_left:
            calibration_graph_maximum_adjacent_rmse = st.number_input(
                "相邻边最大 RMSE",
                min_value=0.001,
                max_value=5.0,
                value=0.14,
                step=0.01,
                format="%.3f",
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_rmse"],
                key="calibration_graph_adjacent_rmse",
            )
        with calibration_rmse_right:
            calibration_graph_maximum_loop_rmse = st.number_input(
                "闭环边最大 RMSE",
                min_value=0.001,
                max_value=5.0,
                value=0.16,
                step=0.01,
                format="%.3f",
                disabled=not use_six_station_pose_prior,
                help=PARAMETER_HELP["graph_rmse"],
                key="calibration_graph_loop_rmse",
            )
        st.caption("边必须同时通过 fitness 和 RMSE；不合格边会保留在运行摘要中，但不会用来强行连接位姿图节点。")

    with st.sidebar.expander("6. 多截面拼接质量诊断", expanded=False):
        use_section_alignment_validation = st.checkbox(
            "启用多截面拼接质量诊断",
            value=True,
            help=PARAMETER_HELP["section_alignment_validation"],
        )
        section_validation_left, section_validation_right = st.columns(2)
        with section_validation_left:
            section_alignment_thickness = st.number_input(
                "验收切片厚度",
                min_value=0.005,
                max_value=1.0,
                value=0.05,
                step=0.005,
                format="%.3f",
                disabled=not use_section_alignment_validation,
                help=PARAMETER_HELP["section_alignment_thickness"],
            )
            section_alignment_overlap_distance = st.number_input(
                "有效重叠距离",
                min_value=0.01,
                max_value=5.0,
                value=0.35,
                step=0.01,
                disabled=not use_section_alignment_validation,
                help=PARAMETER_HELP["section_alignment_overlap"],
            )
            section_alignment_max_p90_distance = st.number_input(
                "对齐 P90 阈值",
                min_value=0.005,
                max_value=5.0,
                value=0.115,
                step=0.005,
                format="%.3f",
                key="section_alignment_p90_v3",
                disabled=not use_section_alignment_validation,
                help=PARAMETER_HELP["section_alignment_p90"],
            )
            section_alignment_minimum_overlap_points = st.number_input(
                "最少重叠点数",
                min_value=1,
                max_value=200_000,
                value=500,
                step=50,
                disabled=not use_section_alignment_validation,
                help=PARAMETER_HELP["section_alignment_points"],
            )
        with section_validation_right:
            section_alignment_probe_count = st.select_slider(
                "平行切片数量",
                options=[1, 3, 5, 7, 9, 11],
                value=5,
                disabled=not use_section_alignment_validation,
                help=PARAMETER_HELP["section_alignment_probes"],
            )
            section_alignment_probe_spacing = st.number_input(
                "平行切片间距",
                min_value=0.01,
                max_value=10.0,
                value=0.50,
                step=0.05,
                disabled=not use_section_alignment_validation,
                help=PARAMETER_HELP["section_alignment_spacing"],
            )
            section_alignment_minimum_views = st.number_input(
                "最低诊断工位数",
                min_value=1,
                max_value=8,
                value=3,
                step=1,
                disabled=not use_section_alignment_validation,
                help=PARAMETER_HELP["section_alignment_views"],
            )
        if section_alignment_max_p90_distance > section_alignment_overlap_distance:
            st.warning("对齐 P90 阈值不能大于有效重叠距离，否则处理时会拒绝该参数组合。")
        st.caption("默认检查 5 张平行薄截面。只在共同覆盖区域生成错层、重影和覆盖差异诊断；覆盖优先模式不会因不重叠或局部轮廓不同而删除标定块合格视角，也不会回写任何位姿。")

    with st.sidebar.expander("7. 三维法向截面", expanded=False):
        section_region_selection = st.checkbox(
            "启用顶视矩形框选自动选面",
            key="section_region_selection",
            help=PARAMETER_HELP["section_region_selection"],
        )
        if section_region_selection:
            st.caption("打开 3D 预览后，点击下方“选择区域”，再沿被测件走向拖拽矩形框；青色框不会裁掉最终截面轮廓。")
        st.caption("框选会从区域内完整 XYZ 点拟合三维主轴，并自动回填法向方位、俯仰和偏移；截面不再强制使用全局 Z 竖直坐标面。")
        target_region = _section_target_region_from_state() if section_region_selection else None
        if target_region is not None:
            st.caption(
                f"当前主轴拟合区域：X {target_region.x_min:.2f}～{target_region.x_max:.2f}，"
                f"Y {target_region.y_min:.2f}～{target_region.y_max:.2f}；仅用于拟合截面法向和确定截面中心。"
            )
        elif section_region_selection:
            st.warning("尚未框选目标区域；为避免把标定块或其他部件当成横截面，正式处理前必须完成框选。")
        angle_degrees = st.number_input(
            "法向方位角（度）",
            min_value=-180.0,
            max_value=180.0,
            step=1.0,
            key="section_angle_degrees",
            help=PARAMETER_HELP["section_angle"],
        )
        tilt_degrees = st.number_input(
            "法向俯仰角（度）",
            min_value=-89.0,
            max_value=89.0,
            step=0.5,
            key="section_tilt_degrees",
            help=PARAMETER_HELP["section_tilt"],
        )
        plane_offset = st.number_input(
            "平面偏移",
            min_value=-50.0,
            max_value=50.0,
            step=0.1,
            key="section_plane_offset",
            help=PARAMETER_HELP["section_offset"],
        )
        thickness = st.number_input("切片厚度", min_value=0.01, max_value=5.0, value=0.15, step=0.01, help=PARAMETER_HELP["section_thickness"])
        if thickness > 0.20:
            st.warning("当前厚度超过 0.20：只能作为复核参数，不能用来补齐断线；正式测量建议保持 0.15。")
        section_column_left, section_column_right = st.columns(2)
        with section_column_left:
            horizontal_min = st.number_input("截面横向最小", value=-30.0, step=0.5, help=PARAMETER_HELP["section_horizontal"])
            vertical_min = st.number_input("截面竖向最小", value=-8.0, step=0.5, help=PARAMETER_HELP["section_vertical"])
        with section_column_right:
            horizontal_max = st.number_input("截面横向最大", value=30.0, step=0.5, help=PARAMETER_HELP["section_horizontal"])
            vertical_max = st.number_input("截面竖向最大", value=10.0, step=0.5, help=PARAMETER_HELP["section_vertical"])
        point_size = st.number_input("截面图点大小", min_value=0.1, max_value=10.0, value=0.45, step=0.05, help=PARAMETER_HELP["section_point_size"])

    return ProcessingConfig(
        denoise=DenoiseConfig(
            voxel_size=float(voxel_size),
            use_statistical_filter=use_statistical_filter,
            statistical_neighbors=int(statistical_neighbors),
            statistical_std_ratio=float(statistical_std_ratio),
            use_radius_filter=use_radius_filter,
            radius=float(radius),
            radius_min_points=int(radius_min_points),
        ),
        calibration=CalibrationConfig(
            use_roi=use_roi,
            bounds=Bounds3D(
                x_min=float(x_min), x_max=float(x_max), y_min=float(y_min),
                y_max=float(y_max), z_min=float(z_min), z_max=float(z_max),
            ),
            minimum_points=int(minimum_points),
            exclude_from_fusion=bool(exclude_calibration_from_fusion),
            exclusion_margin=float(exclusion_margin),
            level_reference_plane=level_reference_plane,
            leveling_plane_distance=float(leveling_plane_distance),
            leveling_minimum_points=int(leveling_minimum_points),
        ),
        registration=RegistrationConfig(
            registration_voxel_size=float(registration_voxel_size),
            use_global_registration=use_global_registration,
            use_six_station_pose_prior=use_six_station_pose_prior,
            reference_vertex_count=int(reference_vertex_count),
            normal_radius_factor=float(normal_radius_factor),
            feature_radius_factor=float(feature_radius_factor),
            global_distance_factor=float(global_distance_factor),
            global_max_iterations=int(global_max_iterations),
            global_attempts=int(global_attempts),
            global_random_seed=5,
            fallback_icp_fitness_threshold=float(fallback_icp_fitness_threshold),
            skip_low_quality_views=skip_low_quality_views,
            minimum_full_cloud_fitness=float(minimum_full_cloud_fitness),
            minimum_calibration_icp_fitness=float(minimum_calibration_icp_fitness),
            maximum_calibration_icp_rmse=float(maximum_calibration_icp_rmse),
            skip_low_quality_calibration_views=skip_low_quality_calibration_views,
            coverage_preserving_fusion=True,
            use_section_alignment_validation=use_section_alignment_validation,
            section_alignment_thickness=float(section_alignment_thickness),
            section_alignment_overlap_distance=float(section_alignment_overlap_distance),
            section_alignment_max_p90_distance=float(section_alignment_max_p90_distance),
            section_alignment_minimum_overlap_points=int(section_alignment_minimum_overlap_points),
            section_alignment_probe_count=int(section_alignment_probe_count),
            section_alignment_probe_spacing=float(section_alignment_probe_spacing),
            section_alignment_minimum_views=int(section_alignment_minimum_views),
            icp_max_correspondence_distance=float(icp_max_correspondence_distance),
            icp_max_iterations=int(icp_max_iterations),
            use_point_to_plane_icp=use_point_to_plane_icp,
            use_multiscale_icp=bool(use_multiscale_icp),
            station_pose_corrections=station_pose_corrections,
            pose_prior_calibration_voxel_size=float(pose_prior_calibration_voxel_size),
            pose_prior_translation_max_correspondence_distance=float(
                pose_prior_translation_max_correspondence_distance
            ),
            pose_prior_rigid_max_rotation_degrees=float(pose_prior_rigid_max_rotation_degrees),
            pose_prior_rigid_max_translation=float(pose_prior_rigid_max_translation),
            calibration_graph_voxel_size=float(calibration_graph_voxel_size),
            calibration_graph_max_correspondence_distance=float(
                calibration_graph_max_correspondence_distance
            ),
            calibration_graph_icp_max_iterations=int(calibration_graph_icp_max_iterations),
            calibration_graph_max_rotation_degrees=float(calibration_graph_max_rotation_degrees),
            calibration_graph_max_translation=float(calibration_graph_max_translation),
            calibration_graph_minimum_adjacent_fitness=float(
                calibration_graph_minimum_adjacent_fitness
            ),
            calibration_graph_minimum_loop_fitness=float(calibration_graph_minimum_loop_fitness),
            calibration_graph_maximum_adjacent_rmse=float(
                calibration_graph_maximum_adjacent_rmse
            ),
            calibration_graph_maximum_loop_rmse=float(calibration_graph_maximum_loop_rmse),
            use_calibration_geometry_constraints=bool(use_calibration_geometry_constraints),
            calibration_geometry_voxel_size=float(calibration_geometry_voxel_size),
            calibration_geometry_plane_distance=float(calibration_geometry_plane_distance),
            calibration_geometry_minimum_plane_points=80,
            calibration_geometry_minimum_planes=3,
            calibration_geometry_max_planes=8,
            calibration_geometry_max_normal_angle_degrees=float(
                calibration_geometry_max_normal_angle_degrees
            ),
            calibration_geometry_max_centroid_distance=float(
                calibration_geometry_max_centroid_distance
            ),
        ),
        section=SectionConfig(
            angle_degrees=float(angle_degrees), tilt_degrees=float(tilt_degrees),
            plane_offset=float(plane_offset), thickness=float(thickness),
            horizontal_min=float(horizontal_min), horizontal_max=float(horizontal_max),
            vertical_min=float(vertical_min), vertical_max=float(vertical_max), point_size=float(point_size),
            target_region=target_region,
            # Retain legacy fields and the grid size used by diagnostics.
            use_statistical_filter=False,
            statistical_neighbors=12,
            statistical_std_ratio=1.0,
            use_component_filter=False,
            component_cell_size=0.30,
            component_min_points=12,
        ),
        fusion_voxel_size=float(fusion_voxel_size),
        compute_backend=compute_backend,
    )


def _preview_axis_range(values: np.ndarray) -> list[float]:
    """为高密度预览锁定稳定的坐标范围，避免浏览器自动范围异常。"""
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return [-1.0, 1.0]
    lower_bound, upper_bound = np.percentile(finite_values, (0.1, 99.9))
    coordinate_span = max(float(upper_bound - lower_bound), 0.1)
    margin = coordinate_span * 0.05
    return [float(lower_bound - margin), float(upper_bound + margin)]


def _preview_axis_ranges(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "x": _preview_axis_range(points[:, 0]),
        "y": _preview_axis_range(points[:, 1]),
        "z": _preview_axis_range(points[:, 2]),
    }


def _preview_aspect_ratio(axis_ranges: dict[str, list[float]]) -> dict[str, float]:
    spans = np.array(
        [
            axis_ranges["x"][1] - axis_ranges["x"][0],
            axis_ranges["y"][1] - axis_ranges["y"][0],
            axis_ranges["z"][1] - axis_ranges["z"][0],
        ],
        dtype=np.float64,
    )
    largest_span = max(float(np.max(spans)), 1e-6)
    normalized_spans = np.clip(spans / largest_span, 0.08, 1.0)
    return {
        "x": float(normalized_spans[0]),
        "y": float(normalized_spans[1]),
        "z": float(normalized_spans[2]),
    }


def _preview_marker_size(point_count: int) -> float:
    """按实际显示点数自动缩放点径，避免稀疏预览看起来过于粗大。"""
    if point_count >= 1_000_000:
        return 0.65
    if point_count >= 500_000:
        return 0.75
    if point_count >= 200_000:
        return 0.90
    if point_count >= 100_000:
        return 1.05
    return 0.85


def _transform_points(points: np.ndarray, transformation: np.ndarray | None) -> np.ndarray:
    normalized_points = np.asarray(points, dtype=np.float64)
    if transformation is None:
        return normalized_points
    homogeneous = np.column_stack((normalized_points, np.ones(len(normalized_points), dtype=np.float64)))
    return (homogeneous @ np.asarray(transformation, dtype=np.float64).T)[:, :3]


def roi_wireframe(
    bounds: Bounds3D,
    transformation: np.ndarray | None = None,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    vertices = np.array(
        [
            [bounds.x_min, bounds.y_min, bounds.z_min], [bounds.x_max, bounds.y_min, bounds.z_min],
            [bounds.x_max, bounds.y_max, bounds.z_min], [bounds.x_min, bounds.y_max, bounds.z_min],
            [bounds.x_min, bounds.y_min, bounds.z_max], [bounds.x_max, bounds.y_min, bounds.z_max],
            [bounds.x_max, bounds.y_max, bounds.z_max], [bounds.x_min, bounds.y_max, bounds.z_max],
        ]
    )
    vertices = _transform_points(vertices, transformation)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in edges:
        x_values.extend([float(vertices[start, 0]), float(vertices[end, 0]), None])
        y_values.extend([float(vertices[start, 1]), float(vertices[end, 1]), None])
        z_values.extend([float(vertices[start, 2]), float(vertices[end, 2]), None])
    return x_values, y_values, z_values


def target_region_wireframe(
    bounds: Bounds2D,
    vertical_range: list[float],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    vertices = np.array(
        [
            [bounds.x_min, bounds.y_min, vertical_range[0]],
            [bounds.x_max, bounds.y_min, vertical_range[0]],
            [bounds.x_max, bounds.y_max, vertical_range[0]],
            [bounds.x_min, bounds.y_max, vertical_range[0]],
            [bounds.x_min, bounds.y_min, vertical_range[1]],
            [bounds.x_max, bounds.y_min, vertical_range[1]],
            [bounds.x_max, bounds.y_max, vertical_range[1]],
            [bounds.x_min, bounds.y_max, vertical_range[1]],
        ],
        dtype=np.float64,
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in edges:
        x_values.extend([float(vertices[start, 0]), float(vertices[end, 0]), None])
        y_values.extend([float(vertices[start, 1]), float(vertices[end, 1]), None])
        z_values.extend([float(vertices[start, 2]), float(vertices[end, 2]), None])
    return x_values, y_values, z_values


def _section_plane_vertices(config: SectionConfig) -> np.ndarray:
    normal, horizontal, vertical = vertical_plane_basis(config.angle_degrees, config.tilt_degrees)
    plane_origin = normal * config.plane_offset
    return np.array(
        [
            plane_origin + horizontal * config.horizontal_min + vertical * config.vertical_min,
            plane_origin + horizontal * config.horizontal_max + vertical * config.vertical_min,
            plane_origin + horizontal * config.horizontal_max + vertical * config.vertical_max,
            plane_origin + horizontal * config.horizontal_min + vertical * config.vertical_max,
        ],
        dtype=np.float64,
    )


def _section_plane_edges(vertices: np.ndarray) -> tuple[list[float | None], list[float | None], list[float | None]]:
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for start, end in edges:
        x_values.extend([float(vertices[start, 0]), float(vertices[end, 0]), None])
        y_values.extend([float(vertices[start, 1]), float(vertices[end, 1]), None])
        z_values.extend([float(vertices[start, 2]), float(vertices[end, 2]), None])
    return x_values, y_values, z_values


def _sample_preview_points(points: np.ndarray, maximum_points: int) -> np.ndarray:
    if len(points) <= maximum_points:
        return points
    sample_indices = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
    return points[sample_indices]


@st.cache_data(
    show_spinner=False,
    ttl=3600,
    hash_funcs={ProcessingConfig: lambda config: json.dumps(config.as_dict(), sort_keys=True)},
)
def load_preview_data(
    files: list[Path],
    config: ProcessingConfig,
    preview_voxel_size: float,
    preview_maximum_points: int,
    apply_denoise: bool,
) -> tuple[Path, PreviewData]:
    reference_file, leveling = select_reference_file_with_leveling(
        files,
        config.registration.reference_vertex_count,
        config.denoise,
        config.calibration,
        config.compute_backend,
    )
    preview = build_preview(
        [reference_file],
        voxel_size=preview_voxel_size,
        maximum_points=preview_maximum_points,
        compute_backend=config.compute_backend,
        denoise_config=config.denoise,
        apply_denoise=apply_denoise,
        leveling=leveling,
    )
    return reference_file, preview


@st.cache_data(
    show_spinner=False,
    ttl=3600,
    hash_funcs={ProcessingConfig: lambda config: json.dumps(config.as_dict(), sort_keys=True)},
)
def load_merged_preview_data(
    result: PipelineResult,
    config: ProcessingConfig,
    preview_voxel_size: float,
    preview_maximum_points: int,
) -> tuple[Path, PreviewData]:
    """Load the completed merged cloud for the interactive result preview."""
    merged_file = result.merged_cloud_path
    preview = build_preview(
        [merged_file],
        voxel_size=preview_voxel_size,
        maximum_points=preview_maximum_points,
        compute_backend=config.compute_backend,
        apply_denoise=False,
        leveling=None,
    )
    return merged_file, preview


def make_preview_figure(
    files: list[Path],
    config: ProcessingConfig,
    preview_voxel_size: float,
    preview_maximum_points: int,
    apply_denoise: bool = True,
    render_mode: str = "points",
    preview_data: PreviewData | None = None,
    reference_file: Path | None = None,
    use_orthographic_projection: bool = True,
) -> go.Figure:
    if render_mode not in {"points", "surface"}:
        raise ValueError(f"不支持的预览渲染模式: {render_mode}")

    if preview_data is None:
        reference_file, preview = load_preview_data(
            files,
            config,
            preview_voxel_size,
            preview_maximum_points,
            apply_denoise,
        )
    else:
        preview = preview_data
        reference_file = reference_file or select_reference_file(files, config.registration.reference_vertex_count)
    points = np.ascontiguousarray(preview.points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError("当前采样参数未得到可预览的点云。")

    heights = points[:, 2]
    color_min, color_max = np.percentile(heights, (1.0, 99.0))
    if np.isclose(color_min, color_max):
        color_min = float(heights.min())
        color_max = float(heights.max())
    if np.isclose(color_min, color_max):
        color_max = color_min + 0.1

    axis_ranges = _preview_axis_ranges(points)
    figure = go.Figure()
    if render_mode == "surface":
        surface_voxel_size = preview_voxel_size
        surface = build_surface_preview(
            points,
            voxel_size=surface_voxel_size,
            maximum_points=SURFACE_PREVIEW_MAXIMUM_POINTS,
            maximum_triangles=SURFACE_PREVIEW_MAXIMUM_TRIANGLES,
            compute_backend=config.compute_backend,
        )
        vertices = surface.vertices
        triangles = surface.triangles
        figure.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=vertices[:, 2],
            intensitymode="vertex",
            colorscale="Turbo",
            cmin=float(color_min),
            cmax=float(color_max),
            showscale=True,
            colorbar={"title": "高度 Z", "thickness": 16, "len": 0.75},
            flatshading=False,
            opacity=0.98,
            lighting={"ambient": 0.34, "diffuse": 0.84, "specular": 0.18, "roughness": 0.62, "fresnel": 0.08},
            lightposition={"x": 120, "y": 120, "z": 160},
            name=f"表面渲染（{len(vertices):,} 顶点 / {len(triangles):,} 面）",
            hoverinfo="skip",
        ))
    else:
        is_high_density = len(points) > PREVIEW_WEBGL_BATCH_SIZE
        batch_size = PREVIEW_WEBGL_BATCH_SIZE if is_high_density else len(points)
        marker_size = _preview_marker_size(len(points))
        opacity = 0.94 if preview_maximum_points >= 300_000 else 0.86
        source_name = reference_file.name
        for batch_start in range(0, len(points), batch_size):
            batch_end = min(batch_start + batch_size, len(points))
            batch_points = points[batch_start:batch_end]
            batch_heights = heights[batch_start:batch_end]
            is_first_batch = batch_start == 0
            marker: dict[str, object] = {
                "size": marker_size,
                "color": batch_heights,
                "colorscale": "Turbo",
                "cmin": float(color_min),
                "cmax": float(color_max),
                "opacity": opacity,
                "showscale": is_first_batch,
            }
            if is_first_batch:
                marker["colorbar"] = {"title": "高度 Z", "thickness": 16, "len": 0.75}

            trace_options: dict[str, object] = {
                "x": batch_points[:, 0],
                "y": batch_points[:, 1],
                "z": batch_points[:, 2],
                "mode": "markers",
                "marker": marker,
                "name": f"按高度着色（{len(points):,} 点）",
                "legendgroup": "height-colored-cloud",
                "showlegend": is_first_batch,
            }
            if is_high_density:
                trace_options["hoverinfo"] = "skip"
            else:
                trace_options["hovertemplate"] = (
                    f"{source_name}<br>X=%{{x:.2f}}<br>Y=%{{y:.2f}}<br>Z=%{{z:.2f}}<extra></extra>"
                )
            figure.add_trace(go.Scatter3d(**trace_options))

    section_vertices = _section_plane_vertices(config.section)
    figure.add_trace(go.Mesh3d(
        x=section_vertices[:, 0],
        y=section_vertices[:, 1],
        z=section_vertices[:, 2],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color="#ff4b4b",
        opacity=0.28,
        flatshading=True,
        lighting={"ambient": 0.78, "diffuse": 0.35, "specular": 0.12, "roughness": 0.75},
        name="实际三维法向横截面",
        hovertemplate=(
            f"实际三维法向横截面<br>方位={config.section.angle_degrees:.2f}°"
            f"<br>俯仰={config.section.tilt_degrees:.2f}°"
            f"<br>偏移={config.section.plane_offset:.3f}"
            f"<br>厚度={config.section.thickness:.3f}<extra></extra>"
        ),
    ))
    section_x, section_y, section_z = _section_plane_edges(section_vertices)
    figure.add_trace(go.Scatter3d(
        x=section_x,
        y=section_y,
        z=section_z,
        mode="lines",
        line={"color": "#ff4b4b", "width": 7},
        name="实际截面边界",
        hoverinfo="skip",
    ))

    leveling_transformation = preview.leveling.transformation if preview.leveling is not None else None
    if config.calibration.use_roi:
        x_values, y_values, z_values = roi_wireframe(config.calibration.bounds, leveling_transformation)
        figure.add_trace(go.Scatter3d(
            x=x_values, y=y_values, z=z_values, mode="lines",
            line={"color": "#a78bfa", "width": 6}, name="标定块配准 ROI（非截面）",
        ))
    if config.section.target_region is not None:
        x_values, y_values, z_values = target_region_wireframe(
            config.section.target_region,
            axis_ranges["z"],
        )
        figure.add_trace(go.Scatter3d(
            x=x_values, y=y_values, z=z_values, mode="lines",
            line={"color": "#00e5ff", "width": 5, "dash": "dash"},
            name="目标横截面区域",
        ))
    figure.update_layout(
        height=640,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={
            "aspectmode": "manual",
            "aspectratio": _preview_aspect_ratio(axis_ranges),
            "camera": {
                "projection": {"type": "orthographic" if use_orthographic_projection else "perspective"},
                "eye": {"x": 1.55, "y": 1.55, "z": 1.25},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
            "xaxis": {"title": "X", "range": axis_ranges["x"]},
            "yaxis": {"title": "Y", "range": axis_ranges["y"]},
            "zaxis": {"title": "Z", "range": axis_ranges["z"]},
        },
        legend={"orientation": "h"},
        hovermode="closest",
        meta={
            "input_points": preview.input_points,
            "after_denoise_points": preview.after_denoise_points,
            "before_limit_points": preview.before_limit_points,
            "display_points": len(points),
            "render_mode": render_mode,
        },
        uirevision=(
            f"{PREVIEW_RENDER_REVISION}-{render_mode}-{preview_voxel_size:.4f}-"
            f"orthographic-{int(use_orthographic_projection)}-{len(points)}"
        ),
    )
    return figure


@st.cache_data(show_spinner=False, ttl=1800)
def make_alignment_audit_figure(
    input_files: tuple[str, ...],
    transformations: tuple[tuple[tuple[float, ...], ...], ...],
    leveling: tuple[tuple[float, ...], ...] | None,
    labels: tuple[str, ...],
    maximum_points_per_view: int = 18_000,
) -> go.Figure:
    """Interactive four-view source-alignment audit.

    The layout mirrors the original audit image: XY, XZ, YZ projections and a
    rotatable 3D view. Every panel uses the same transformed source samples.
    """
    palette = ("#e45756", "#4c78a8", "#59a14f", "#b279a2", "#f28e2b", "#4cc3c7")
    leveling_matrix = np.asarray(leveling, dtype=np.float64) if leveling else np.eye(4)
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("XY source overlay", "XZ source overlay", "YZ source overlay", "3D source overlay"),
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )
    for index, (file_name, transform_data, label) in enumerate(
        zip(input_files, transformations, labels, strict=True)
    ):
        cloud = read_point_cloud(Path(file_name))
        points = np.asarray(cloud.points, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > maximum_points_per_view:
            sample_indices = np.linspace(0, len(points) - 1, maximum_points_per_view, dtype=np.int64)
            points = points[sample_indices]
        if not len(points):
            continue
        transform = np.asarray(transform_data, dtype=np.float64)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        points = points @ leveling_matrix[:3, :3].T + leveling_matrix[:3, 3]
        trace_name = f"第 {index + 1} 部分点云 · {label}"
        color = palette[index % len(palette)]
        common = {
            "mode": "markers",
            "marker": {"size": 3.0, "color": color, "opacity": 0.42},
            "name": trace_name,
            "legendgroup": f"source-{index}",
            "showlegend": True,
            "hovertemplate": (
                f"{trace_name}<br>"
                "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>"
            ),
        }
        figure.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            **{**common, "showlegend": True},
        ), row=1, col=1)
        figure.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 2],
            **{**common, "showlegend": False, "hovertemplate": f"{trace_name}<br>X=%{{x:.3f}}<br>Z=%{{y:.3f}}<extra></extra>"},
        ), row=1, col=2)
        figure.add_trace(go.Scatter(
            x=points[:, 1], y=points[:, 2],
            **{**common, "showlegend": False, "hovertemplate": f"{trace_name}<br>Y=%{{x:.3f}}<br>Z=%{{y:.3f}}<extra></extra>"},
        ), row=2, col=1)
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker={"size": 3.0, "color": color, "opacity": 0.48},
            name=trace_name,
            legendgroup=f"source-{index}",
            showlegend=False,
            hovertemplate=common["hovertemplate"],
        ), row=2, col=2)
    figure.update_layout(
        height=860,
        margin={"l": 10, "r": 10, "t": 58, "b": 10},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis={"title": "X", "scaleanchor": "y", "gridcolor": "#dbe3e8", "zerolinecolor": "#8c9aa3"},
        yaxis={"title": "Y", "gridcolor": "#dbe3e8", "zerolinecolor": "#8c9aa3"},
        xaxis2={"title": "X", "scaleanchor": "y2", "gridcolor": "#dbe3e8", "zerolinecolor": "#8c9aa3"},
        yaxis2={"title": "Z", "gridcolor": "#dbe3e8", "zerolinecolor": "#8c9aa3"},
        xaxis3={"title": "Y", "scaleanchor": "y3", "gridcolor": "#dbe3e8", "zerolinecolor": "#8c9aa3"},
        yaxis3={"title": "Z", "gridcolor": "#dbe3e8", "zerolinecolor": "#8c9aa3"},
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "X", "backgroundcolor": "#ffffff", "gridcolor": "#dbe3e8"},
            "yaxis": {"title": "Y", "backgroundcolor": "#ffffff", "gridcolor": "#dbe3e8"},
            "zaxis": {"title": "Z", "backgroundcolor": "#ffffff", "gridcolor": "#dbe3e8"},
            "camera": {"projection": {"type": "orthographic"}, "eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
        },
        legend={
            "orientation": "h",
            "itemsizing": "constant",
            "itemwidth": 175,
            "font": {"size": 14, "color": "#17232d"},
        },
        hovermode="closest",
        uirevision="alignment-audit-four-view",
    )
    return figure


def _section_plane_top_view_line(
    points: np.ndarray,
    config: SectionConfig,
    reference_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    normal, horizontal, _ = vertical_plane_basis(config.angle_degrees, config.tilt_degrees)
    horizontal_coordinates = points @ horizontal
    finite_coordinates = horizontal_coordinates[np.isfinite(horizontal_coordinates)]
    if len(finite_coordinates):
        lower_bound, upper_bound = np.percentile(finite_coordinates, (0.5, 99.5))
        span = max(float(upper_bound - lower_bound), 1.0)
        horizontal_min = float(lower_bound - span * 0.08)
        horizontal_max = float(upper_bound + span * 0.08)
    else:
        horizontal_min = config.horizontal_min
        horizontal_max = config.horizontal_max

    if reference_z is None:
        finite_z = points[:, 2][np.isfinite(points[:, 2])]
        reference_z = float(np.median(finite_z)) if len(finite_z) else 0.0
    normal_xy = normal[:2]
    normal_xy_squared = float(np.dot(normal_xy, normal_xy))
    if normal_xy_squared <= 1e-12:
        plane_origin = (normal * config.plane_offset)[:2]
    else:
        horizontal_offset = config.plane_offset - normal[2] * float(reference_z)
        plane_origin = normal_xy * (horizontal_offset / normal_xy_squared)
    line_points = np.array(
        [
            plane_origin + horizontal[:2] * horizontal_min,
            plane_origin + horizontal[:2] * horizontal_max,
        ],
        dtype=np.float64,
    )
    return line_points[:, 0], line_points[:, 1]


def _section_picker_region_points(selection_event: object) -> np.ndarray:
    try:
        selected_points = selection_event.selection.points
    except (AttributeError, KeyError, TypeError):
        return np.empty((0, 2), dtype=np.float64)

    selected_coordinates: list[tuple[float, float]] = []
    for point in reversed(selected_points):
        try:
            # Plotly versions embedded by desktop WebEngine may omit
            # curve_number for box selections. Missing metadata is safe here
            # because the picker cloud is the first trace.
            curve_number = point.get("curve_number")
            if curve_number is not None and int(curve_number) != 0:
                continue
            x_coordinate = float(point["x"])
            y_coordinate = float(point["y"])
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if np.isfinite(x_coordinate) and np.isfinite(y_coordinate):
            selected_coordinates.append((x_coordinate, y_coordinate))
    if not selected_coordinates:
        return np.empty((0, 2), dtype=np.float64)
    return np.unique(np.asarray(selected_coordinates, dtype=np.float64), axis=0)


def _section_picker_region_metadata(
    selected_coordinates: np.ndarray,
    selected_points: np.ndarray,
    directionality: float,
) -> dict[str, float | int]:
    lower_bounds = selected_coordinates.min(axis=0)
    upper_bounds = selected_coordinates.max(axis=0)
    center = (lower_bounds + upper_bounds) / 2.0
    point_lower_bounds = selected_points.min(axis=0)
    point_upper_bounds = selected_points.max(axis=0)
    return {
        "x_min": float(lower_bounds[0]),
        "x_max": float(upper_bounds[0]),
        "y_min": float(lower_bounds[1]),
        "y_max": float(upper_bounds[1]),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float((point_lower_bounds[2] + point_upper_bounds[2]) / 2.0),
        "point_count": int(len(selected_points)),
        "directionality": float(directionality),
        "implementation_version": 3,
    }


def _section_points_inside_picker_region(
    points: np.ndarray,
    selected_coordinates: np.ndarray,
) -> np.ndarray:
    lower_bounds = selected_coordinates.min(axis=0)
    upper_bounds = selected_coordinates.max(axis=0)
    coordinates = np.asarray(points, dtype=np.float64)
    finite_mask = np.isfinite(coordinates).all(axis=1)
    region_mask = (
        finite_mask
        & (coordinates[:, 0] >= lower_bounds[0])
        & (coordinates[:, 0] <= upper_bounds[0])
        & (coordinates[:, 1] >= lower_bounds[1])
        & (coordinates[:, 1] <= upper_bounds[1])
    )
    return coordinates[region_mask]


def _section_picker_region_signature(selected_coordinates: np.ndarray) -> tuple[float | int, ...]:
    lower_bounds = selected_coordinates.min(axis=0)
    upper_bounds = selected_coordinates.max(axis=0)
    return (
        round(float(lower_bounds[0]), 6),
        round(float(lower_bounds[1]), 6),
        round(float(upper_bounds[0]), 6),
        round(float(upper_bounds[1]), 6),
        int(len(selected_coordinates)),
    )


def _make_section_picker_figure(
    points: np.ndarray,
    config: SectionConfig,
    selected_region: dict[str, float | int] | None,
    selection_enabled: bool = False,
) -> go.Figure:
    picker_points = _sample_preview_points(points, SECTION_PICKER_MAXIMUM_POINTS)
    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=picker_points[:, 0],
        y=picker_points[:, 1],
        mode="markers",
        marker={
            "size": 3.0,
            "color": picker_points[:, 2],
            "colorscale": "Turbo",
            "opacity": 0.78,
            "showscale": False,
        },
        name=f"顶视点云（{len(picker_points):,} 点）",
        hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<extra></extra>",
    ))

    reference_z_value = selected_region.get("center_z") if selected_region is not None else None
    reference_z = float(reference_z_value) if reference_z_value is not None else None
    plane_x, plane_y = _section_plane_top_view_line(picker_points, config, reference_z)
    figure.add_trace(go.Scatter(
        x=plane_x,
        y=plane_y,
        mode="lines",
        line={"color": "#ff4b4b", "width": 4},
        name="三维法平面在选区中心高度的投影",
        hoverinfo="skip",
    ))
    if selected_region is not None:
        region_x_min = float(selected_region["x_min"])
        region_x_max = float(selected_region["x_max"])
        region_y_min = float(selected_region["y_min"])
        region_y_max = float(selected_region["y_max"])
        figure.add_shape(
            type="rect",
            x0=region_x_min,
            x1=region_x_max,
            y0=region_y_min,
            y1=region_y_max,
            line={"color": "#00e5ff", "width": 3},
            fillcolor="rgba(0, 229, 255, 0.14)",
            layer="above",
        )
        figure.add_trace(go.Scatter(
            x=[float(selected_region["center_x"])],
            y=[float(selected_region["center_y"])],
            mode="markers",
            marker={"size": 11, "symbol": "x", "color": "#00e5ff", "line": {"color": "white", "width": 1.5}},
            name="框选区域中心",
            hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<extra></extra>",
        ))

    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        title=(
            "顶视图：正在选择区域，请拖拽矩形框"
            if selection_enabled
            else "顶视图：点击“选择区域”后可拖拽矩形框"
        ),
        dragmode="select" if selection_enabled else "pan",
        xaxis={"title": "X", "range": _preview_axis_range(picker_points[:, 0])},
        yaxis={
            "title": "Y",
            "range": _preview_axis_range(picker_points[:, 1]),
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h"},
        hovermode="closest",
    )
    return figure


def _reset_section_region_picker(group_name: str) -> None:
    st.session_state["section_picker_group"] = group_name
    st.session_state["section_picker_region"] = None
    st.session_state["section_picker_locked"] = False
    st.session_state["section_picker_drawing"] = False
    st.session_state["section_picker_last_signature"] = None
    st.session_state["section_picker_revision"] = int(st.session_state.get("section_picker_revision", 0)) + 1


def _set_section_region_drawing(group_name: str, enabled: bool) -> None:
    st.session_state["section_picker_group"] = group_name
    st.session_state["section_picker_drawing"] = enabled
    st.session_state["section_picker_last_signature"] = None
    st.session_state["section_picker_revision"] = int(st.session_state.get("section_picker_revision", 0)) + 1


def _clear_section_region_picker(group_name: str) -> None:
    _reset_section_region_picker(group_name)
    st.session_state.pop("pending_section_plane", None)
    st.session_state.pop("pending_section_offset", None)
    st.session_state["section_angle_degrees"] = 0.0
    st.session_state["section_tilt_degrees"] = 0.0
    st.session_state["section_plane_offset"] = 9.0


def _apply_pending_section_plane() -> None:
    pending_plane = st.session_state.pop("pending_section_plane", None)
    if pending_plane is None:
        return
    st.session_state["section_angle_degrees"] = float(pending_plane["angle_degrees"])
    st.session_state["section_tilt_degrees"] = float(pending_plane.get("tilt_degrees", 0.0))
    st.session_state["section_plane_offset"] = float(pending_plane["plane_offset"])


def _apply_pending_section_offset() -> None:
    pending_offset = st.session_state.pop("pending_section_offset", None)
    if pending_offset is None:
        return
    st.session_state["section_plane_offset"] = float(pending_offset)


def _section_offset_scan_signature(
    group_name: str,
    preview_data: PreviewData,
    section_config: SectionConfig,
    half_range: float,
    step: float,
) -> tuple[object, ...]:
    sampled_points = _sample_preview_points(
        np.asarray(preview_data.points),
        SECTION_OFFSET_SCAN_MAXIMUM_POINTS,
    )
    digest = hashlib.sha1(
        np.ascontiguousarray(sampled_points, dtype=np.float32).tobytes()
    ).hexdigest()
    return (
        group_name,
        digest,
        round(float(section_config.angle_degrees), 6),
        round(float(section_config.tilt_degrees), 6),
        round(float(section_config.plane_offset), 6),
        round(float(section_config.thickness), 6),
        round(float(section_config.horizontal_min), 6),
        round(float(section_config.horizontal_max), 6),
        round(float(section_config.vertical_min), 6),
        round(float(section_config.vertical_max), 6),
        round(float(section_config.component_cell_size), 6),
        round(float(half_range), 6),
        round(float(step), 6),
    )


def _normalize_section_offset_scan_rows(
    rows: object,
) -> list[dict[str, float | int | bool]]:
    """Normalize scan rows retained by an older Streamlit session.

    The diagnostic schema gained ``points_per_cell`` after the first release. A
    browser session can outlive that code change, so rendering must not assume
    every cached row was produced by the current implementation.
    """
    if not isinstance(rows, (list, tuple)):
        return []

    normalized: list[dict[str, float | int | bool]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        try:
            plane_offset = float(raw_row["plane_offset"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(plane_offset):
            continue

        def _float_value(key: str, default: float = 0.0) -> float:
            try:
                value = float(raw_row.get(key, default))
            except (TypeError, ValueError):
                return default
            return value if np.isfinite(value) else default

        def _int_value(key: str, default: int = 0) -> int:
            try:
                value = int(raw_row.get(key, default))
            except (TypeError, ValueError):
                return default
            return max(0, value)

        def _optional_float(key: str) -> float | None:
            value = raw_row.get(key)
            if value is None:
                return None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        def _optional_int(key: str) -> int | None:
            value = raw_row.get(key)
            if value is None:
                return None
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return None

        point_count = _int_value("point_count")
        cell_count = _int_value("cell_count")
        points_per_cell = _float_value("points_per_cell", float("nan"))
        if not np.isfinite(points_per_cell) or points_per_cell < 0:
            points_per_cell = float(point_count / max(cell_count, 1))

        normalized.append(
            {
                "plane_offset": plane_offset,
                "point_count": point_count,
                "component_count": _int_value("component_count"),
                "cell_count": cell_count,
                "points_per_cell": points_per_cell,
                "horizontal_span": _float_value("horizontal_span"),
                "vertical_span": _float_value("vertical_span"),
                "largest_component_span": _float_value("largest_component_span"),
                "largest_component_ratio": _float_value("largest_component_ratio"),
                "horizontal_coverage_ratio": _float_value("horizontal_coverage_ratio"),
                "continuity_ratio": _float_value("continuity_ratio"),
                "quality_score": _float_value("quality_score"),
                "candidate_score": _float_value("candidate_score"),
                "quality_gain": _float_value("quality_gain"),
                "span_gain": _float_value("span_gain"),
                "cell_retention": _float_value("cell_retention"),
                "parallel_layer_ratio": _float_value("parallel_layer_ratio"),
                "parallel_layer_max_tracks": _int_value("parallel_layer_max_tracks"),
                "stable_neighbor_count": _int_value("stable_neighbor_count"),
                "source_view_count": _optional_int("source_view_count"),
                "source_new_cell_count": _int_value("source_new_cell_count"),
                "source_new_multiview_cell_count": _int_value(
                    "source_new_multiview_cell_count"
                ),
                "source_new_multiview_ratio": _optional_float(
                    "source_new_multiview_ratio"
                ),
                "source_support_guardrail_ok": bool(
                    raw_row.get("source_support_guardrail_ok", False)
                ),
                "stability_guardrail_ok": bool(
                    raw_row.get("stability_guardrail_ok", False)
                ),
                "auto_eligible": bool(raw_row.get("auto_eligible", False)),
                "is_current": bool(raw_row.get("is_current", False)),
                "recommended": bool(raw_row.get("recommended", False)),
            }
        )
    return normalized


def render_section_offset_diagnostic(
    group_name: str,
    preview_data: PreviewData,
    section_config: SectionConfig,
) -> None:
    """Show a read-only nearby-plane scan and let the user choose an offset."""
    state_key = f"section-offset-scan-{group_name}"
    with st.expander("截面平面位置诊断（只读）", expanded=False):
        st.caption(
            "围绕当前平面扫描平行薄片，比较点数、二维连续结构和覆盖跨度。"
            "扫描只读取预览点，不修改点云、工位位姿或 CAD；写入偏移后仍需重新点击“开始处理”。"
        )
        range_column, step_column, action_column = st.columns((1.0, 1.0, 1.2))
        with range_column:
            half_range = st.number_input(
                "扫描半范围",
                min_value=0.5,
                max_value=20.0,
                value=4.0,
                step=0.5,
                format="%.2f",
                key=f"section_offset_scan_half_range_{group_name}",
            )
        with step_column:
            step = st.number_input(
                "扫描步长",
                min_value=0.05,
                max_value=2.0,
                value=0.25,
                step=0.05,
                format="%.2f",
                key=f"section_offset_scan_step_{group_name}",
            )
        with action_column:
            st.write("")
            scan_requested = st.button(
                "扫描附近平面",
                key=f"section_offset_scan_button_{group_name}",
                use_container_width=True,
            )

        current_signature = _section_offset_scan_signature(
            group_name,
            preview_data,
            section_config,
            float(half_range),
            float(step),
        )
        scan_state = st.session_state.get(state_key)
        if scan_requested:
            sampled_points = _sample_preview_points(
                np.asarray(preview_data.points),
                SECTION_OFFSET_SCAN_MAXIMUM_POINTS,
            )
            try:
                with st.spinner("正在扫描附近截面平面..."):
                    rows = scan_section_offsets(
                        sampled_points,
                        section_config,
                        half_range=float(half_range),
                        step=float(step),
                    )
            except ValueError as error:
                st.error(f"截面偏移扫描失败：{error}")
                return
            st.session_state[state_key] = {
                "signature": current_signature,
                "rows": rows,
            }
            scan_state = st.session_state[state_key]

        raw_rows = scan_state.get("rows", []) if isinstance(scan_state, Mapping) else []
        rows = _normalize_section_offset_scan_rows(raw_rows)
        if rows and rows != raw_rows and isinstance(scan_state, Mapping):
            # Persist the normalized form so a long-lived session is upgraded
            # once instead of taking the compatibility path on every rerun.
            st.session_state[state_key] = {**scan_state, "rows": rows}
            scan_state = st.session_state[state_key]
        signature_matches = bool(
            isinstance(scan_state, Mapping)
            and scan_state.get("signature") == current_signature
        )
        if not rows:
            st.info("点击“扫描附近平面”后查看当前截面位置附近的取点和结构变化。")
            return
        if not signature_matches:
            st.warning("截面角度、厚度、范围或预览数据已变化；请重新扫描以得到最新诊断。")
            return

        offsets = np.asarray([float(row["plane_offset"]) for row in rows])
        point_counts = np.asarray([int(row["point_count"]) for row in rows])
        occupied_cell_counts = np.asarray([int(row["cell_count"]) for row in rows])
        largest_spans = np.asarray(
            [float(row["largest_component_span"]) for row in rows]
        )
        points_per_cell = np.asarray(
            [float(row["points_per_cell"]) for row in rows]
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=offsets,
                y=occupied_cell_counts,
                mode="lines+markers",
                name="占用网格数",
                line={"color": "#2563eb", "width": 2.5},
                marker={"size": 7},
                customdata=np.column_stack((point_counts, points_per_cell)),
                hovertemplate=(
                    "偏移=%{x:.3f}<br>占用网格=%{y:,}"
                    "<br>预览点数=%{customdata[0]:,}"
                    "<br>每格点数=%{customdata[1]:.1f}<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=offsets,
                y=largest_spans,
                mode="lines+markers",
                name="最大连续跨度",
                yaxis="y2",
                line={"color": "#d97706", "width": 2.0},
                marker={"size": 6},
                hovertemplate="偏移=%{x:.3f}<br>最大跨度=%{y:.2f}<extra></extra>",
            )
        )
        figure.add_vline(
            x=float(section_config.plane_offset),
            line={"color": "#dc2626", "dash": "dash", "width": 2},
            annotation_text="当前偏移",
            annotation_position="top",
        )
        figure.update_layout(
            height=360,
            margin={"l": 0, "r": 0, "t": 28, "b": 0},
            xaxis={"title": "平面偏移（点·法线 = 偏移）"},
            yaxis={"title": "占用网格数", "rangemode": "tozero"},
            yaxis2={
                "title": "最大连续跨度",
                "overlaying": "y",
                "side": "right",
                "rangemode": "tozero",
            },
            legend={"orientation": "h"},
            hovermode="x unified",
        )
        st.plotly_chart(
            figure,
            width="stretch",
            key=f"section-offset-scan-plot-{group_name}",
            config={"displaylogo": False, "responsive": True},
        )

        table_rows = [
            {
                "平面偏移": f"{float(row['plane_offset']):.3f}",
                "预览点数（辅助）": int(row["point_count"]),
                "连续结构数": int(row["component_count"]),
                "占用网格": int(row["cell_count"]),
                "每格点数（辅助）": f"{float(row['points_per_cell']):.1f}",
                "横向跨度": f"{float(row['horizontal_span']):.2f}",
                "竖向跨度": f"{float(row['vertical_span']):.2f}",
                "最大连续跨度": f"{float(row['largest_component_span']):.2f}",
                "覆盖评分": f"{float(row['quality_score']):.3f}",
                "推荐": "是" if row.get("recommended", False) else "",
                "当前": "是" if row["is_current"] else "",
            }
            for row in rows
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        st.caption(
            "优先比较占用网格、连续结构和跨度；预览点数及每格点数会受采样级别、融合体素和截面厚度影响，仅作辅助。"
            "覆盖评分用于推荐横向覆盖更完整、连续性更好的候选平面；扫描不会修改点云或自动替代人工确认。"
        )

        offset_options = [float(row["plane_offset"]) for row in rows]
        recommended_indices = [
            index for index, row in enumerate(rows) if bool(row.get("recommended", False))
        ]
        current_index = (
            recommended_indices[0]
            if recommended_indices
            else int(np.argmin(np.abs(offsets - float(section_config.plane_offset))))
        )
        selected_offset = st.selectbox(
            "选择要写入的平面偏移",
            offset_options,
            index=current_index,
            format_func=lambda value: f"{float(value):.3f}",
            key=f"section_offset_scan_selected_{group_name}",
        )
        if st.button(
            "将选中偏移写入截面参数",
            key=f"section_offset_scan_apply_{group_name}",
            use_container_width=True,
        ):
            st.session_state["pending_section_offset"] = float(selected_offset)
            st.rerun()


def render_section_region_selector(
    group_name: str,
    preview_data: PreviewData,
    section_config: SectionConfig,
) -> None:
    if st.session_state.get("section_picker_group") != group_name:
        _reset_section_region_picker(group_name)

    region_state = st.session_state.get("section_picker_region")
    selected_region = region_state if isinstance(region_state, dict) else None
    if selected_region is not None and int(selected_region.get("implementation_version", 0)) < 3:
        _reset_section_region_picker(group_name)
        st.info("截面算法已升级为三维主轴法平面，旧框选角度已失效；请重新框选目标区域。")
        st.rerun()
    picker_locked = bool(st.session_state.get("section_picker_locked", False))
    picker_drawing = bool(st.session_state.get("section_picker_drawing", False))
    st.subheader("框选主轴区域自动生成三维法向横截面")
    st.caption("沿被测件走向框选一段区域，排除标定块和其他部件。系统会取框内完整 XYZ 点拟合三维主轴，并生成法向沿该主轴、垂直于倾斜点云的完整横截面；青色框只定方向和位置，不裁剪截面轮廓。")

    action_left, action_center, action_right = st.columns((1.0, 1.0, 1.2))
    with action_left:
        button_label = "取消选择区域" if picker_drawing else "选择区域"
        if st.button(button_label, key="toggle_section_region_picker", use_container_width=True):
            _set_section_region_drawing(group_name, not picker_drawing)
            st.rerun()
    with action_center:
        if st.button(
            "清空区域",
            key="clear_section_region_picker",
            use_container_width=True,
            disabled=selected_region is None and not picker_drawing,
        ):
            _clear_section_region_picker(group_name)
            st.rerun()
    with action_right:
        if picker_drawing:
            st.caption("已开启绘制：请在下方拖拽矩形框")
        elif selected_region is not None:
            st.caption(f"已框选 {int(selected_region['point_count']):,} 个预览点")
        else:
            st.caption("尚未框选区域")

    picker_figure = _make_section_picker_figure(
        preview_data.points,
        section_config,
        selected_region,
        selection_enabled=picker_drawing,
    )
    picker_key = f"section-picker-{group_name}-{st.session_state.get('section_picker_revision', 0)}"
    picker_config = {"displaylogo": False, "scrollZoom": True, "responsive": True}
    if picker_drawing:
        selection_event = st.plotly_chart(
            picker_figure,
            width="stretch",
            key=picker_key,
            on_select="rerun",
            selection_mode="box",
            config=picker_config,
        )
    else:
        picker_config["modeBarButtonsToRemove"] = ["select2d", "lasso2d"]
        st.plotly_chart(
            picker_figure,
            width="stretch",
            key=picker_key,
            on_select="ignore",
            config=picker_config,
        )
        selection_event = None

    selected_coordinates = _section_picker_region_points(selection_event)
    if len(selected_coordinates) and picker_drawing:
        selected_points = _section_points_inside_picker_region(preview_data.points, selected_coordinates)
        selection_signature = _section_picker_region_signature(selected_coordinates)
        if selection_signature != st.session_state.get("section_picker_last_signature"):
            st.session_state["section_picker_last_signature"] = selection_signature
            try:
                if len(selected_points) < 30:
                    raise ValueError("框选区域内有效点太少，请沿主轴方向框选更长的一段真实点云。")
                angle_degrees, tilt_degrees, plane_offset, directionality = normal_plane_from_xyz_region(selected_points)
            except ValueError as error:
                st.warning(str(error))
            else:
                st.session_state["section_picker_region"] = _section_picker_region_metadata(
                    selected_coordinates,
                    selected_points,
                    directionality,
                )
                st.session_state["pending_section_plane"] = {
                    "angle_degrees": angle_degrees,
                    "tilt_degrees": tilt_degrees,
                    "plane_offset": plane_offset,
                }
                st.session_state["section_picker_locked"] = True
                st.session_state["section_picker_drawing"] = False
                st.session_state["section_auto_generation_message"] = (
                    "已根据框选区域拟合三维主轴，并自动生成垂直于主轴的法向横截面。"
                )
                st.rerun()

    if picker_locked and selected_region is not None and not picker_drawing:
        auto_message = st.session_state.pop("section_auto_generation_message", None)
        if auto_message:
            st.success(auto_message)
        st.success(
            f"已生成三维法向横截面：法向方位 {section_config.angle_degrees:.2f}°，"
            f"俯仰 {section_config.tilt_degrees:.2f}°，"
            f"偏移 {section_config.plane_offset:.3f}。可继续在侧栏微调。"
        )
        directionality = float(selected_region["directionality"])
        if directionality < 1.2:
            st.warning("选区在顶视图中的主方向不够明显。建议框住目标部件的完整长度，同时排除标定块，以获得稳定的横截面方向。")
        else:
            st.caption(f"拟合稳定性：主方向长宽比 {directionality:.2f}。数值越大，截面法向越稳定。")
        render_section_offset_diagnostic(group_name, preview_data, section_config)


def render_result_section_offset_diagnostic(
    result: PipelineResult,
    section_consensus_summary: Mapping[str, Any],
) -> None:
    """Render final-cloud offset evidence and keep unsafe candidates manual-only."""
    diagnostic = section_consensus_summary.get("section_offset_diagnostic")
    if not isinstance(diagnostic, Mapping) or not diagnostic.get("enabled", False):
        return

    run_summary = {}
    try:
        run_summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        run_summary = {}
    config_summary = run_summary.get("config", {})
    registration_summary = (
        config_summary.get("registration", {})
        if isinstance(config_summary, Mapping)
        else {}
    )
    coverage_preserving = bool(
        isinstance(registration_summary, Mapping)
        and registration_summary.get("coverage_preserving_fusion", True)
    )

    status = str(diagnostic.get("status", "insufficient_evidence"))
    status_labels = {
        "auto_recommended": "发现可复核的完整截面候选",
        "current_is_best": "当前截面位置更稳妥",
        "manual_review": "候选截面需要人工复核",
        "source_coverage_gap": "候选轮廓缺少多工位支持",
        "insufficient_evidence": "截面偏移证据不足",
    }
    title = status_labels.get(status, str(diagnostic.get("status_label", status)))
    message = str(diagnostic.get("message", ""))
    if status == "auto_recommended":
        st.info(f"最终融合云偏移诊断：{title}。{message}")
    elif status == "source_coverage_gap":
        st.warning(f"最终融合云偏移诊断：{title}。{message}")
    elif status == "manual_review":
        st.warning(f"最终融合云偏移诊断：{title}。{message}")
    else:
        st.info(f"最终融合云偏移诊断：{title}。{message}")

    current_offset = float(diagnostic.get("current_offset", 0.0) or 0.0)
    recommended_offset = diagnostic.get("recommended_offset")
    source_view_count = diagnostic.get("source_view_count")
    metric_columns = st.columns(4)
    metric_columns[0].metric("当前偏移", f"{current_offset:.3f}")
    metric_columns[1].metric(
        "候选偏移",
        f"{float(recommended_offset):.3f}" if recommended_offset is not None else "无",
    )
    metric_columns[2].metric(
        "多工位证据",
        str(int(source_view_count)) if source_view_count is not None else "未提供",
    )
    metric_columns[3].metric(
        "诊断采样点",
        f"{int(diagnostic.get('sampled_points', 0) or 0):,}",
    )

    rows_value = diagnostic.get("rows", [])
    rows = _normalize_section_offset_scan_rows(rows_value)
    if not rows:
        return
    offsets = np.asarray([float(row["plane_offset"]) for row in rows], dtype=np.float64)
    span_values = np.asarray(
        [float(row["largest_component_span"]) for row in rows], dtype=np.float64
    )
    score_values = np.asarray(
        [float(row.get("candidate_score", row.get("quality_score", 0.0))) for row in rows],
        dtype=np.float64,
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=offsets,
            y=span_values,
            mode="lines+markers",
            name="最大连续跨度",
            line={"color": "#d97706", "width": 2.3},
            marker={"size": 7},
            customdata=np.column_stack((score_values,)),
            hovertemplate="偏移=%{x:.3f}<br>最大连续跨度=%{y:.2f}<br>候选评分=%{customdata[0]:.3f}<extra></extra>",
        )
    )
    source_new_values = [
        int(row.get("source_new_multiview_cell_count", 0) or 0) for row in rows
    ]
    figure.add_trace(
        go.Scatter(
            x=offsets,
            y=source_new_values,
            mode="lines+markers",
            name="新增多工位网格",
            yaxis="y2",
            line={"color": "#2563eb", "width": 1.8},
            marker={"size": 6},
            hovertemplate="偏移=%{x:.3f}<br>新增多工位网格=%{y:,}<extra></extra>",
        )
    )
    figure.add_vline(
        x=current_offset,
        line={"color": "#dc2626", "dash": "dash", "width": 2},
        annotation_text="当前偏移",
        annotation_position="top",
    )
    if recommended_offset is not None:
        figure.add_vline(
            x=float(recommended_offset),
            line={"color": "#16a34a", "dash": "dot", "width": 2},
            annotation_text="候选偏移",
            annotation_position="bottom",
        )
    figure.update_layout(
        height=340,
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        xaxis={"title": "平面偏移（点·法线 = 偏移）"},
        yaxis={"title": "最大连续跨度", "rangemode": "tozero"},
        yaxis2={
            "title": "新增多工位网格",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
        },
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    with st.expander("查看最终融合云截面偏移诊断", expanded=status == "source_coverage_gap"):
        st.plotly_chart(
            figure,
            width="stretch",
            key=f"final-section-offset-diagnostic-{result.group_name}",
            config={"displaylogo": False, "responsive": True},
        )
        table_rows = []
        for row in rows:
            table_rows.append(
                {
                    "平面偏移": f"{float(row['plane_offset']):.3f}",
                    "连续结构数": int(row.get("component_count", 0) or 0),
                    "最大连续跨度": f"{float(row.get('largest_component_span', 0.0) or 0.0):.2f}",
                    "候选评分": f"{float(row.get('candidate_score', 0.0) or 0.0):.3f}",
                    "稳定邻近平面": int(row.get("stable_neighbor_count", 0) or 0),
                    "源工位数": (
                        int(row["source_view_count"])
                        if row.get("source_view_count") is not None
                        else "-"
                    ),
                    "新增网格": int(row.get("source_new_cell_count", 0) or 0),
                    "新增多工位网格": int(
                        row.get("source_new_multiview_cell_count", 0) or 0
                    ),
                    "新增多工位比例": (
                        f"{float(row['source_new_multiview_ratio']):.1%}"
                        if row.get("source_new_multiview_ratio") is not None
                        else "-"
                    ),
                    "自动采用": "是" if row.get("auto_eligible", False) else "否",
                    "当前": "是" if row.get("is_current", False) else "",
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        if status == "auto_recommended" and recommended_offset is not None:
            st.caption(
                "候选偏移只提供复核入口，不会在当前结果上直接改写点云或测量值；"
                "采用后必须重新处理并重新确认 CAD 质量。"
            )
            if st.button(
                "采用候选偏移并重新处理",
                key=f"apply-final-section-offset-{result.group_name}",
                use_container_width=True,
            ):
                st.session_state["pending_section_offset"] = float(recommended_offset)
                st.session_state.pop("pipeline_results", None)
                st.session_state.pop("pipeline_result_data_signature", None)
                st.rerun()
        elif status == "source_coverage_gap":
            if coverage_preserving:
                st.caption(
                    "候选新增轮廓没有达到多工位共同支持阈值。该结论只用于覆盖风险诊断，"
                    "不会删除互补视角；正式 CAD 是否可用仍由最终截面完整性和测量质量闸门判断。"
                )
            else:
                st.caption(
                    "候选新增轮廓没有达到多工位共同支持阈值。当前结果不允许正式测量；"
                    "请重新采集能覆盖缺口两侧的视角，或重新框选真正需要测量的截面。"
                )


def render_result(result: PipelineResult, show_cad: bool = False) -> None:
    st.subheader(f"第 {result.group_name} 组结果")
    try:
        run_summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run_summary = {}
    config_summary = run_summary.get("config", {})
    if isinstance(config_summary, Mapping):
        profile_name = config_summary.get("profile_name")
        profile_source = config_summary.get("profile_source")
        family_name = config_summary.get("data_family")
        if profile_name:
            source_label = "自动数据族推荐" if profile_source == "auto" else "通用参数"
            st.info(
                f"本组采用：{profile_name}（{source_label}；数据族 {family_name or 'default'}）。"
            )
    pose_graph_summary = run_summary.get("pose_graph")
    if len(result.denoise_stats) == 6 and not pose_graph_summary:
        st.warning("当前显示的是网页更新前生成的旧结果，尚未包含“仅标定块位姿图 + 截面审核后原始点重融合”。请重新点击“开始处理”。")
    fused_view_count = sum(item.included_in_fusion for item in result.registration_metrics)
    metric_left, metric_center, metric_right, metric_backend, metric_leveling = st.columns(5)
    metric_left.metric("融合后点数", f"{result.merged_points:,}")
    metric_center.metric("截面点数", f"{result.section_points:,}")
    metric_right.metric("融合视角", f"{fused_view_count}/{len(result.denoise_stats)}")
    metric_backend.metric("计算后端", "GPU" if result.compute_backend == "gpu" else "CPU")
    if result.leveling is not None:
        metric_leveling.metric("调平倾角", f"{result.leveling.tilt_degrees:.3f}°")
        if result.leveling.applied:
            st.success(
                f"已使用标定块近水平基准面调平：拟合 {result.leveling.plane_points:,} 点，"
                f"平面 RMSE {result.leveling.plane_rmse:.4f}。"
            )
        else:
            st.info("标定块基准面已接近水平，未额外旋转坐标。")
    if pose_graph_summary:
        calibration_edges = pose_graph_summary.get("calibration_edges", [])
        accepted_calibration = sum(bool(item.get("accepted")) for item in calibration_edges)
        st.success(
            f"标定块位姿图已应用：{accepted_calibration}/{len(calibration_edges)} 条标定块边被接受。"
        )
        effective_voxel = float(run_summary.get("effective_fusion_voxel_size", 0.03))
        section_validation_enabled = bool(
            isinstance(config_summary, Mapping)
            and isinstance(config_summary.get("registration"), Mapping)
            and config_summary["registration"].get("use_section_alignment_validation", True)
        )
        if section_validation_enabled:
            st.caption(
                "步骤 1～7 只依据多面体标定块确定刚体位姿；多截面只做拼接质量诊断，不删除互补视角。"
                f"标定块质量合格的原始全分辨率点云已以 {effective_voxel:.3f} mm 全部参与覆盖融合；体素内保留真实代表点，最终截面直接从该融合云提取（{result.section_source_points:,} 点）。"
            )
        else:
            st.caption(
                "步骤 1～7 只依据多面体标定块确定刚体位姿；本组关闭多截面质量诊断，"
                f"标定块质量合格的原始全分辨率点云已以 {effective_voxel:.3f} mm 全部参与覆盖融合；体素内保留真实代表点（{result.section_source_points:,} 点）。"
            )
    else:
        st.caption(
            f"截面直接从上方已融合点云提取，输入共 {result.section_source_points:,} 点。"
        )
    section_consensus_summary = run_summary.get("section_consensus", {})
    if isinstance(section_consensus_summary, Mapping):
        final_cleanup = section_consensus_summary.get("final_component_cleanup", {})
        if isinstance(final_cleanup, Mapping):
            removed_points = int(final_cleanup.get("removed_points", 0) or 0)
            if removed_points:
                st.caption(
                    f"融合云孤立组件清理：删除 {removed_points:,} 个弱支持浮空点，"
                    f"保留 {int(final_cleanup.get('output_points', result.merged_points) or 0):,} 个融合点；"
                    "最终截面直接从清理后的融合云提取，截面本身未再执行清噪。"
                )
    if result.section_raw_points != result.section_points:
        st.caption(
            f"兼容旧结果：截面结构清理前 {result.section_raw_points:,} 点，"
            f"清理后 {result.section_points:,} 点。"
        )
    exclusion_reason_labels = {
        "section_alignment": "共同覆盖区域持续错层/重影",
        "section_extra_contour": "多工位一致检测到额外轮廓",
        "section_consensus": "截面轮廓共识判定为异常工位",
        "full_cloud_fitness": "全云配准质量不足",
    }
    exclusion_reason_labels["section_peer_component"] = "互证工位簇存在持续轮廓不一致"
    excluded_views = [
        f"{item.pose_prior_label or item.file_name}（"
        f"{exclusion_reason_labels.get(item.exclusion_reason or '', '质量验收未通过')}）"
        for item in result.registration_metrics
        if not item.included_in_fusion
    ]
    if excluded_views:
        coverage_preserving = bool(
            isinstance(config_summary, Mapping)
            and isinstance(config_summary.get("registration"), Mapping)
            and config_summary["registration"].get("coverage_preserving_fusion", True)
        )
        st.warning(
            (
                "以下工位因标定块位姿质量未通过，未参与覆盖融合："
                if coverage_preserving
                else "以下工位未参与重新融合："
            )
            + "; ".join(excluded_views)
        )
    section_validation_summary = run_summary.get("section_validation", {})
    if isinstance(section_validation_summary, dict):
        nested_consensus = section_validation_summary.get("section_consensus", {})
        if isinstance(nested_consensus, dict) and nested_consensus:
            if not isinstance(section_consensus_summary, dict):
                section_consensus_summary = {}
            section_consensus_summary = {
                **section_consensus_summary,
                "station_filter": nested_consensus,
            }
        protected_peer_components = section_validation_summary.get(
            "protected_peer_components", []
        )
        if isinstance(protected_peer_components, list) and protected_peer_components:
            protected_labels: list[str] = []
            for component in protected_peer_components:
                if not isinstance(component, Mapping):
                    continue
                indices = component.get("component_indices", [])
                if not isinstance(indices, list):
                    continue
                labels = [
                    result.registration_metrics[index].pose_prior_label
                    or result.registration_metrics[index].file_name
                    for index in indices
                    if isinstance(index, int)
                    and 0 <= index < len(result.registration_metrics)
                ]
                if labels:
                    protected_labels.append("/".join(labels))
            if protected_labels:
                st.info(
                    "局部互补覆盖保护：保留 "
                    + "; ".join(protected_labels)
                    + "，其轮廓只覆盖局部区段且由同簇工位互相印证，未按整站异常剔除。"
                )
    if isinstance(section_consensus_summary, Mapping):
        render_result_section_offset_diagnostic(result, section_consensus_summary)
    if isinstance(section_consensus_summary, dict) and section_consensus_summary.get("enabled"):
        actual_removed_points = int(
            section_consensus_summary.get("removed_section_points", 0) or 0
        )
        actual_removed_ratio = float(
            section_consensus_summary.get("removed_section_ratio", 0.0) or 0.0
        )
        diagnostic_removed_points = int(
            section_consensus_summary.get(
                "diagnostic_removed_section_points", actual_removed_points
            )
            or 0
        )
        diagnostic_removed_ratio = float(
            section_consensus_summary.get(
                "diagnostic_removed_section_ratio", actual_removed_ratio
            )
            or 0.0
        )
        excluded_count = len(section_consensus_summary.get("excluded_indices", []) or [])
        coverage_preserving = bool(
            isinstance(config_summary, Mapping)
            and isinstance(config_summary.get("registration"), Mapping)
            and config_summary["registration"].get("coverage_preserving_fusion", True)
        )
        if coverage_preserving:
            st.caption(
                f"覆盖优先融合：{len(result.registration_metrics) - excluded_count}/{len(result.registration_metrics)} 个标定块质量合格视角参与融合；"
                f"截面共识仅做诊断，实际删除局部点 {actual_removed_points:,} 个；"
                f"若按共识掩码计算，可能标记 {diagnostic_removed_points:,} 个（{diagnostic_removed_ratio:.1%}）。"
                "所有保留点只做刚体坐标变换和体素融合，不按 CAD 重排、拉伸或校正。"
            )
        else:
            st.caption(
                f"截面共识筛查：局部不支持点 {actual_removed_points:,} 个（{actual_removed_ratio:.1%}）；"
                f"整工位剔除 {excluded_count} 个。保留点只做坐标变换和体素融合，"
                "不会按 CAD 重排、拉伸或校正。"
            )
        fusion_confidence = section_consensus_summary.get("fusion_confidence", {})
        if isinstance(fusion_confidence, Mapping):
            confidence_value = fusion_confidence.get("confidence_score")
            try:
                confidence_text = f"{float(confidence_value):.3f}"
            except (TypeError, ValueError):
                confidence_text = "-"
            st.caption(
                "重叠区域融合诊断："
                f"置信度 {confidence_text}；"
                f"多工位体素 {int(fusion_confidence.get('section_voxel_multi_view_count', 0) or 0):,}；"
                f"单视角独有体素 {int(fusion_confidence.get('section_voxel_unique_view_count', 0) or 0):,}；"
                f"局部残差 P95 "
                + (
                    f"{float(fusion_confidence.get('section_local_residual_p95')):.4f}"
                    if fusion_confidence.get("section_local_residual_p95") is not None
                    else "-"
                )
                + "。"
            )
            with st.expander("查看重叠区域融合置信度详情", expanded=False):
                detail = {
                    key: value
                    for key, value in fusion_confidence.items()
                    if key != "selection_policy"
                }
                st.json(detail)
        whole_cloud_fusion = section_consensus_summary.get("whole_cloud_fusion", {})
        if isinstance(whole_cloud_fusion, Mapping) and whole_cloud_fusion.get("status") == "ok":
            multi_view_voxels = int(whole_cloud_fusion.get("multi_view_voxel_count", 0) or 0)
            unique_view_voxels = int(whole_cloud_fusion.get("unique_view_voxel_count", 0) or 0)
            voxel_count = int(whole_cloud_fusion.get("voxel_count", 0) or 0)
            representative_p95 = whole_cloud_fusion.get("representative_residual_p95")
            p95_text = (
                f"{float(representative_p95):.4f}"
                if representative_p95 is not None
                else "-"
            )
            st.caption(
                "全云覆盖优先融合："
                f"{multi_view_voxels:,} 个体素由多个视角支持，"
                f"{unique_view_voxels:,} 个体素为单视角独有覆盖，"
                f"共 {voxel_count:,} 个输出体素；代表点残差 P95 {p95_text}。"
                "体素内选择真实测量点，不输出坐标平均值。"
            )
            with st.expander("查看全云覆盖优先融合详情", expanded=False):
                detail = {
                    key: value
                    for key, value in whole_cloud_fusion.items()
                    if key != "selection_policy"
                }
                st.json(detail)
    section_integrity_summary = run_summary.get("section_integrity")
    if not isinstance(section_integrity_summary, Mapping) and result.section_integrity:
        section_integrity_summary = result.section_integrity
    if isinstance(section_integrity_summary, Mapping):
        integrity_status = str(section_integrity_summary.get("status", "insufficient_evidence"))
        integrity_message = str(
            section_integrity_summary.get(
                "message",
                "当前截面证据不足，暂不判断缺口来源。",
            )
        )
        integrity_recommendation = str(
            section_integrity_summary.get("recommendation", "")
        )
        if integrity_status == "cleanup_induced_gap":
            st.warning(f"截面缺口来源诊断：{integrity_message}")
        elif integrity_status == "mixed_structure_change":
            st.warning(f"截面缺口来源诊断：{integrity_message}")
        elif integrity_status == "source_coverage_gap_likely":
            st.info(f"截面缺口来源诊断：{integrity_message}")
        elif integrity_status == "stable_after_cleanup":
            st.success(f"截面缺口来源诊断：{integrity_message}")
        else:
            st.info(f"截面缺口来源诊断：{integrity_message}")
        if integrity_recommendation:
            st.caption(integrity_recommendation)
        with st.expander("查看截面缺口来源诊断详情", expanded=False):
            detail_columns = st.columns(4)
            detail_columns[0].metric(
                "基线截面点数",
                f"{int(section_integrity_summary.get('raw_points', result.section_raw_points) or 0):,}",
            )
            detail_columns[1].metric(
                "最终截面点数",
                f"{int(section_integrity_summary.get('cleaned_points', result.section_points) or 0):,}",
            )
            detail_columns[2].metric(
                "截面点数变化率",
                f"{float(section_integrity_summary.get('removed_ratio', 0.0) or 0.0):.2%}",
            )
            detail_columns[3].metric(
                "主要结构（基线→最终）",
                f"{int(section_integrity_summary.get('raw_component_count', 0) or 0)}→"
                f"{int(section_integrity_summary.get('cleaned_component_count', 0) or 0)}",
            )
            st.write(
                f"二维结构网格：{float(section_integrity_summary.get('grid_cell_size', 0.0) or 0.0):.3f}；"
                f"基线/最终共同网格：{int(section_integrity_summary.get('shared_cell_count', 0) or 0):,}；"
                f"占用网格差异率：{float(section_integrity_summary.get('lost_cell_ratio', 0.0) or 0.0):.2%}。"
            )
            source_profile_count = int(
                section_integrity_summary.get("source_profile_count", 0) or 0
            )
            if source_profile_count and bool(
                section_integrity_summary.get("source_profile_gap_evidence", False)
            ):
                st.write(
                    f"可信工位薄截面共识样本：{source_profile_count} 个；"
                    f"结构分段一致率：{float(section_integrity_summary.get('source_profile_consistency_ratio', 0.0) or 0.0):.1%}。"
                )
            st.caption(
                "该诊断只比较基线截面和最终截面的结构变化；两者都来自清理后的融合云，"
                "不能单独证明融合云组件清理无损。融合清理影响需另行核对组件记录和源点云；"
                "截面本身不再做统计或连通域清噪。程序不会插值、连线、拉伸点云，"
                "也不会把 CAD 结果回写到点云位姿。"
            )
    elif pose_graph_summary:
        st.info("当前运行摘要没有截面缺口诊断字段，请重新点击“开始处理”生成新版结果。")
    if result.gpu_device_name is not None:
        if pose_graph_summary:
            st.caption(f"本次处理 CUDA GPU：{result.gpu_device_name}。位姿图鲁棒全局优化由 Open3D CPU 后端执行。")
        else:
            st.caption(f"本次处理 CUDA GPU：{result.gpu_device_name}。标定块位姿图全局优化由 Open3D CPU 后端执行。")

    image_left, image_raw, image_clean = st.columns(3)
    with image_left:
        st.image(str(result.top_preview_path), caption="融合点云顶视预览", use_container_width=True)
    with image_raw:
        st.image(
            str(result.raw_section_image_path),
            caption="截面汇总图（来自融合云的基线截面）",
            use_container_width=True,
        )
    with image_clean:
        st.image(
            str(result.section_image_path),
            caption="最终三维法向横截面（来自已清理融合云）",
            use_container_width=True,
        )
    multi_section_path = result.multi_section_audit_path
    if multi_section_path is None:
        output_files = run_summary.get("output_files", {})
        if isinstance(output_files, Mapping):
            candidate_path = Path(str(output_files.get("multi_section_audit", "")))
            multi_section_path = candidate_path if candidate_path.is_file() else None
    multi_section_diagnostic = run_summary.get("multi_section_diagnostic", {})
    if multi_section_path is not None and multi_section_path.is_file():
        with st.expander("多横截面诊断（只读）", expanded=False):
            st.image(
                str(multi_section_path),
                caption="最终融合云的多个独立横截面；不把不同位置的截面合并为一个 CAD 轮廓。",
                use_container_width=True,
            )
            if isinstance(multi_section_diagnostic, Mapping):
                status_labels = {
                    "stable_neighbor_sections": "相邻截面稳定",
                    "center_only_stable": "中心截面暂时稳定",
                    "neighbor_only_stable": "仅邻近截面较完整",
                    "persistent_gap": "多截面持续缺口",
                }
                diagnostic_status = str(
                    multi_section_diagnostic.get("diagnostic_status", "")
                )
                metric_columns = st.columns(4)
                metric_columns[0].metric(
                    "诊断截面数",
                    str(int(multi_section_diagnostic.get("probe_count", 0) or 0)),
                )
                metric_columns[1].metric(
                    "截面间距",
                    f"{float(multi_section_diagnostic.get('probe_spacing', 0.0) or 0.0):.3f}",
                )
                metric_columns[2].metric(
                    "稳定截面数",
                    str(int(multi_section_diagnostic.get("stable_slice_count", 0) or 0)),
                )
                metric_columns[3].metric(
                    "诊断结论",
                    status_labels.get(diagnostic_status, diagnostic_status or "未提供"),
                )
                rows_value = multi_section_diagnostic.get("rows", [])
                if isinstance(rows_value, list) and rows_value:
                    table_rows = []
                    for row in rows_value:
                        if not isinstance(row, Mapping):
                            continue
                        table_rows.append(
                            {
                                "截面": (
                                    "中心"
                                    if row.get("is_center")
                                    else str(int(row.get("probe_index", 0)) + 1)
                                ),
                                "偏移": f"{float(row.get('plane_offset', 0.0) or 0.0):+.3f}",
                                "点数": int(row.get("point_count", 0) or 0),
                                "连续结构数": int(row.get("component_count", 0) or 0),
                                "最大连续跨度": f"{float(row.get('largest_component_span', 0.0) or 0.0):.2f}",
                                "双层比例": f"{float(row.get('parallel_layer_ratio', 0.0) or 0.0):.1%}",
                                "状态": status_labels.get(
                                    str(row.get("diagnostic_status", "")),
                                    str(row.get("diagnostic_status", "")),
                                ),
                            }
                        )
                    if table_rows:
                        st.dataframe(table_rows, use_container_width=True, hide_index=True)
                message = str(multi_section_diagnostic.get("diagnostic_message", ""))
                if message:
                    st.caption(message)
    if result.section_alignment_audit_path is not None:
        section_validation_enabled = bool(
            isinstance(config_summary, Mapping)
            and isinstance(config_summary.get("registration"), Mapping)
            and config_summary["registration"].get("use_section_alignment_validation", True)
        )
        st.image(
            str(result.section_alignment_audit_path),
            caption=(
                "第 6 步多工位截面叠加诊断图：显示共同覆盖区内的错层、重影和互补覆盖；不据此剔除视角，也不会改动点云位姿。"
                if section_validation_enabled
                else "多工位截面诊断图：本组关闭多截面诊断；不会改动点云位姿。"
            ),
            use_container_width=True,
        )
    if result.section_coverage_audit_path is not None:
        st.image(
            str(result.section_coverage_audit_path),
            caption="截面横向覆盖诊断图：显示各工位覆盖范围、共同覆盖、单视角覆盖和缺口位置。",
            use_container_width=True,
        )
    if result.alignment_audit_path is not None:
        metrics = run_summary.get("registration_metrics", [])
        metrics = [item for item in metrics if isinstance(item, Mapping)]
        input_files = tuple(
            str(path)
            for path in run_summary.get("input_files", [])
            if path
        )
        transforms = tuple(
            tuple(tuple(float(value) for value in row) for row in item["transformation"])
            for item in metrics
            if item.get("transformation")
        )
        labels = tuple(
            str(item.get("pose_prior_label") or f"view {index + 1}")
            for index, item in enumerate(metrics)
        )
        leveling = (
            tuple(tuple(float(value) for value in row) for row in result.leveling.transformation)
            if result.leveling is not None
            else None
        )
        if len(input_files) == len(transforms) == len(labels):
            st.subheader("按原始扫描着色的标定块位姿叠加审核")
            st.caption("与顶部融合点云预览一致：可旋转、缩放、平移，并点击图例隐藏或显示工位。")
            st.plotly_chart(
                make_alignment_audit_figure(
                    input_files,
                    transforms,
                    leveling,
                    labels,
                ),
                width="stretch",
                key=f"interactive-alignment-audit-{result.group_name}",
                config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            )
        st.download_button(
            "下载静态位姿审计图",
            data=result.alignment_audit_path.read_bytes(),
            file_name=f"group_{result.group_name}_alignment_audit.png",
            mime="image/png",
            key=f"download-alignment-audit-{result.run_directory}",
        )

    section_status_labels = {
        "reference": "参考工位",
        "pass": "通过",
        "peer_supported_coverage": "互证覆盖（保留）",
        "coverage_only": "局部覆盖不足（保留）",
        "insufficient_evidence": "截面点不足（保留）",
                "mismatch": "持续错层/重影（仅诊断）",
        "not_checked": "未检查",
    }

    section_view_reports_by_index: dict[int, Mapping[str, Any]] = {}
    if isinstance(section_validation_summary, Mapping):
        section_view_values = section_validation_summary.get("views", [])
        if isinstance(section_view_values, list):
            for view_value in section_view_values:
                if not isinstance(view_value, Mapping):
                    continue
                view_index = view_value.get("index")
                if isinstance(view_index, int):
                    section_view_reports_by_index[view_index] = view_value

    def section_probe_summary(metric_index: int, metric: Any) -> tuple[str, str, str, str, str]:
        report = section_view_reports_by_index.get(metric_index, {})
        probe_count = report.get("probe_count", metric.section_probe_count)
        valid_count = report.get("valid_probe_count", metric.section_valid_probe_count)
        compatible_count = report.get(
            "compatible_probe_count",
            metric.section_compatible_probe_count,
        )
        abnormal_count = report.get(
            "abnormal_evidence_count",
            metric.section_abnormal_evidence_count,
        )
        try:
            probe_count_value = int(probe_count) if probe_count is not None else 0
        except (TypeError, ValueError):
            probe_count_value = 0
        try:
            valid_count_value = int(valid_count) if valid_count is not None else 0
        except (TypeError, ValueError):
            valid_count_value = 0
        try:
            compatible_count_value = int(compatible_count) if compatible_count is not None else 0
        except (TypeError, ValueError):
            compatible_count_value = 0
        try:
            abnormal_count_value = int(abnormal_count) if abnormal_count is not None else 0
        except (TypeError, ValueError):
            abnormal_count_value = 0
        abnormal_offsets: list[float] = []
        for key in ("mismatch_probe_offsets", "extra_contour_probe_offsets"):
            values = report.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_value):
                    abnormal_offsets.append(numeric_value)
        unique_offsets = sorted(set(abnormal_offsets))
        offset_text = (
            ", ".join(f"{offset:+.3f}" for offset in unique_offsets)
            if unique_offsets
            else "-"
        )
        probe_text = f"{valid_count_value}/{probe_count_value}" if probe_count_value else "-"
        compatible_text = (
            f"{compatible_count_value}/{probe_count_value}" if probe_count_value else "-"
        )
        abnormal_text = str(abnormal_count_value) if probe_count_value else "-"
        evidence_indices: set[int] = set()
        if report:
            for key in ("mismatch_evidence_indices", "extra_contour_evidence_indices"):
                values = report.get(key, [])
                if not isinstance(values, list):
                    continue
                for value in values:
                    if isinstance(value, int):
                        evidence_indices.add(value)
        evidence_text = str(len(evidence_indices)) if report else "-"
        return probe_text, compatible_text, abnormal_text, offset_text, evidence_text

    st.markdown("**标定块配准与截面验收摘要**")
    rows = []
    for metric_index, item in enumerate(result.registration_metrics):
        valid_probe_text, compatible_probe_text, abnormal_probe_text, abnormal_offset_text, evidence_text = section_probe_summary(
            metric_index,
            item,
        )
        rows.append(
            {
                "文件": item.file_name,
                "工位": item.pose_prior_label or "-",
                "标定块点数": item.calibration_points,
                "标定块 ICP fitness": f"{item.icp_fitness:.4f}",
                "标定块 ICP RMSE": f"{item.icp_rmse:.5f}",
                "几何约束": item.calibration_geometry_status or "-",
                "几何平面": (
                    "-"
                    if item.calibration_geometry_matched_planes is None
                    else f"{item.calibration_geometry_matched_planes}/"
                    f"{item.calibration_geometry_plane_count or 0}"
                ),
                "几何残差": (
                    "-"
                    if item.calibration_geometry_plane_rmse_before is None
                    else f"{item.calibration_geometry_plane_rmse_before:.4f}"
                    "→"
                    f"{item.calibration_geometry_plane_rmse_after or 0.0:.4f}"
                ),
                "几何修正": (
                    "-"
                    if item.calibration_geometry_correction_rotation_degrees is None
                    else f"{item.calibration_geometry_correction_rotation_degrees:.2f}°/"
                    f"{item.calibration_geometry_correction_translation or 0.0:.3f}"
                ),
                "截面状态": section_status_labels.get(item.section_validation_status or "not_checked", "未检查"),
                "支持工位数": "-" if item.section_supporting_views is None else item.section_supporting_views,
                "共同横向跨度": "-" if item.section_shared_span is None else f"{item.section_shared_span:.3f}",
                "局部匹配点": "-" if item.section_overlap_points is None else item.section_overlap_points,
                "局部 P90": "-" if item.section_p90_distance is None else f"{item.section_p90_distance:.5f}",
                "异常横向格比例": "-" if item.section_bad_bin_ratio is None else f"{item.section_bad_bin_ratio:.1%}",
                "截面共识删除率": "-" if item.section_consensus_removed_ratio is None else f"{item.section_consensus_removed_ratio:.1%}",
                "有效探针": valid_probe_text,
                "兼容探针": compatible_probe_text,
                "异常探针": abnormal_probe_text,
                "异常偏移": abnormal_offset_text,
                "异常工位证据": evidence_text,
                "参与融合": "是" if item.included_in_fusion else "否",
            }
        )
    summary_rows = [
        {
            "工位": row["工位"],
            "截面状态": row["截面状态"],
            "标定块 fitness": row["标定块 ICP fitness"],
            "标定块 RMSE": row["标定块 ICP RMSE"],
            "局部 P90": row["局部 P90"],
            "参与融合": row["参与融合"],
        }
        for row in rows
    ]
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    with st.expander("查看逐工位完整配准与截面指标", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if pose_graph_summary:
        with st.expander("查看标定块位姿图边指标", expanded=False):
            edge_rows = []
            for item in pose_graph_summary.get("calibration_edges", []):
                edge_rows.append(
                    {
                        "源工位": item.get("source_label"),
                        "目标工位": item.get("target_label"),
                        "边类型": item.get("edge_type"),
                        "fitness": f"{float(item.get('fitness', 0.0)):.4f}",
                        "RMSE": f"{float(item.get('rmse', 0.0)):.5f}",
                        "旋转修正": f"{float(item.get('correction_rotation_degrees', 0.0)):.3f}°",
                        "平移修正": f"{float(item.get('correction_translation', 0.0)):.4f}",
                        "接受": "是" if item.get("accepted") else "否",
                        "拒绝原因": item.get("rejection_reason") or "-",
                    }
                )
            st.dataframe(edge_rows, use_container_width=True, hide_index=True)

    section_alignment_rows = []
    for metric_index, item in enumerate(result.registration_metrics):
        if item.section_points is None:
            continue
        valid_probe_text, compatible_probe_text, abnormal_probe_text, abnormal_offset_text, evidence_text = section_probe_summary(
            metric_index,
            item,
        )
        section_alignment_rows.append(
            {
                "工位": item.pose_prior_label or item.file_name,
                "薄截面点数": item.section_points,
                "共同横向跨度": "-" if item.section_shared_span is None else f"{item.section_shared_span:.3f}",
                "局部匹配点": item.section_overlap_points,
                "局部中位距离": "-" if item.section_median_distance is None else f"{item.section_median_distance:.5f}",
                "局部 P90": "-" if item.section_p90_distance is None else f"{item.section_p90_distance:.5f}",
                "异常横向格比例": "-" if item.section_bad_bin_ratio is None else f"{item.section_bad_bin_ratio:.1%}",
                "截面共识删除率": "-" if item.section_consensus_removed_ratio is None else f"{item.section_consensus_removed_ratio:.1%}",
                "有效探针": valid_probe_text,
                "兼容探针": compatible_probe_text,
                "异常探针": abnormal_probe_text,
                "异常偏移": abnormal_offset_text,
                "异常工位证据": evidence_text,
                "验收结论": section_status_labels.get(item.section_validation_status or "not_checked", "未检查"),
            }
        )
    if section_alignment_rows:
        with st.expander("查看第 6 步原始薄截面逐工位验收详情", expanded=False):
            st.caption("使用原始全分辨率点云的多张薄截面，只比较共同横向覆盖区域。这里的结果仅用于判断标定块位姿是否稳定、记录错层/重影和覆盖差异；覆盖优先模式下，局部覆盖不足和互补轮廓不会删除工位或删除独有点。")
            st.dataframe(section_alignment_rows, use_container_width=True, hide_index=True)

    st.markdown("**输出文件**")
    st.code(str(result.run_directory.resolve()), language=None)
    st.write(f"合并点云：`{result.merged_cloud_path.resolve()}`")
    if result.raw_section_cloud_path is not None:
        st.write(f"融合云基线截面点云：`{result.raw_section_cloud_path.resolve()}`")
    st.write(f"融合云基线截面图：`{result.raw_section_image_path.resolve()}`")
    if result.section_cloud_path is not None:
        st.write(f"最终截面点云：`{result.section_cloud_path.resolve()}`")
    else:
        st.warning("当前截面参数未选中点，已保存提示图片；请调整平面偏移或切片厚度后重试。")
    if result.section_alignment_audit_path is not None:
        st.write(f"多工位截面验收图：`{result.section_alignment_audit_path.resolve()}`")
    if result.section_coverage_audit_path is not None:
        st.write(f"截面横向覆盖诊断图：`{result.section_coverage_audit_path.resolve()}`")
    if multi_section_path is not None and multi_section_path.is_file():
        st.write(f"多横截面诊断图：`{multi_section_path.resolve()}`")
    if result.alignment_audit_path is not None:
        st.write(f"标定块位姿叠加审核图：`{result.alignment_audit_path.resolve()}`")
    st.write(f"运行参数与位姿：`{result.summary_path.resolve()}`")

    download_raw, download_clean, download_audit = st.columns(3)
    with download_raw:
        st.download_button(
            "下载基线截面 PNG",
            data=result.raw_section_image_path.read_bytes(),
            file_name=f"group_{result.group_name}_vertical_section_raw.png",
            mime="image/png",
            key=f"download-raw-section-{result.run_directory}",
            use_container_width=True,
        )
    with download_clean:
        st.download_button(
            "下载最终截面 PNG",
            data=result.section_image_path.read_bytes(),
            file_name=f"group_{result.group_name}_vertical_section.png",
            mime="image/png",
            key=f"download-section-{result.run_directory}",
            use_container_width=True,
        )
    with download_audit:
        if result.section_alignment_audit_path is not None:
            st.download_button(
                "下载工位截面审核 PNG",
                data=result.section_alignment_audit_path.read_bytes(),
                file_name=f"group_{result.group_name}_section_alignment_audit.png",
                mime="image/png",
                key=f"download-section-audit-{result.run_directory}",
                use_container_width=True,
            )
    if result.section_coverage_audit_path is not None:
        st.download_button(
            "下载截面覆盖诊断 PNG",
            data=result.section_coverage_audit_path.read_bytes(),
            file_name=f"group_{result.group_name}_section_coverage_audit.png",
            mime="image/png",
            key=f"download-section-coverage-{result.run_directory}",
            use_container_width=True,
        )
    if multi_section_path is not None and multi_section_path.is_file():
        st.download_button(
            "下载多横截面诊断 PNG",
            data=multi_section_path.read_bytes(),
            file_name=f"group_{result.group_name}_vertical_sections_multi.png",
            mime="image/png",
            key=f"download-multi-section-{result.run_directory}",
            use_container_width=True,
        )
    st.download_button(
        "下载运行摘要 JSON",
        data=result.summary_path.read_bytes(),
        file_name=f"group_{result.group_name}_run_summary.json",
        mime="application/json",
        key=f"download-summary-{result.run_directory}",
    )

    if show_cad:
        render_cad_analysis_panel(result)


def main() -> None:
    st.markdown(
        """
        <section class="app-hero">
            <div class="app-hero-kicker">POINT CLOUD ALIGNMENT · 3D INSPECTION</div>
            <h1 class="app-hero-title">点云拼接、侧面截面与 CAD 偏差分析</h1>
            <p class="app-hero-subtitle">标定块刚体定位 → 各角度目标点云覆盖优先融合 → 多截面质量诊断 → 最终侧面截面 → 第 8 步 CAD 偏差分析</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    groups = discover_groups(DATA_ROOT)
    if not groups:
        st.error(f"未在 `{DATA_ROOT}` 找到分组 PLY 数据。")
        st.stop()

    workflow_view = st.sidebar.radio(
        "工作窗口",
        ("预览与框选", "开始处理", "结果审核", "CAD 测量"),
        index=0,
        key="workflow_view",
        horizontal=False,
    )
    st.markdown(
        f'<div class="workflow-window-title">当前工作窗口：{workflow_view}</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("◈ 3D 预览工作台", expanded=True):
        show_preview = st.checkbox(
            "显示当前组低采样 3D 预览",
            value=True,
            key="show_preview",
            disabled=workflow_view != "预览与框选",
            help=PARAMETER_HELP["show_preview"],
        )
        preview_sampling_level = st.select_slider(
            "预览采样等级",
            options=list(PREVIEW_SAMPLING_LEVELS),
            value="3 级｜精细",
            disabled=not show_preview,
            help=PARAMETER_HELP["preview_level"],
        )
        preview_apply_denoise = st.checkbox(
            "实时应用去噪参数（GPU）",
            value=True,
            disabled=not show_preview,
            help=PARAMETER_HELP["preview_denoise"],
        )
        preview_orthographic_projection = st.checkbox(
            "正交等比例显示（防止拉伸）",
            value=True,
            disabled=not show_preview,
            help=PARAMETER_HELP["preview_projection"],
        )
        preview_render_mode = st.session_state.setdefault("preview_render_mode", "points")
        surface_button_label = "切换为表面渲染" if preview_render_mode == "points" else "切换回点云渲染"
        if st.button(
            surface_button_label,
            key="preview_surface_render_button",
            use_container_width=True,
            disabled=not show_preview,
            help=PARAMETER_HELP["preview_surface"],
        ):
            st.session_state["preview_render_mode"] = "surface" if preview_render_mode == "points" else "points"
            st.rerun()
        preview_render_mode = st.session_state["preview_render_mode"]
        preview_voxel_size, preview_maximum_points, _ = PREVIEW_SAMPLING_LEVELS[preview_sampling_level]
        if show_preview and workflow_view == "预览与框选":
            if preview_maximum_points >= 300_000:
                st.caption(f"超清档会拆分为每批最多 {PREVIEW_WEBGL_BATCH_SIZE:,} 点的 WebGL 图层，避免浏览器因单图层过大而空白。")
            if preview_maximum_points >= 1_000_000:
                st.warning("200 万点是最高上限；实际显示点数不会超过原始数据和去噪后的有效点数。")

    st.sidebar.header("数据集")
    group_names = list(groups)
    target_labels = ["全部分组"] + [f"第 {name} 组（{len(groups[name])} 个 PLY）" for name in group_names]
    target_index = st.sidebar.selectbox(
        "待处理数据",
        options=range(len(target_labels)),
        index=1 if group_names else 0,
        format_func=lambda index: target_labels[index],
        key=POINTCLOUD_TARGET_GROUP_STATE_KEY,
        help="选择要处理的点云组。推荐先选择单个组验证参数，确认 ICP 指标和截面图后再选择全部分组批处理。",
    )
    selected_group_names = _selected_group_names_from_index(group_names, int(target_index))
    if selected_group_names:
        st.sidebar.caption(
            "本次实际处理："
            + ("全部分组" if target_index == 0 else f"第 {selected_group_names[0]} 组")
        )

    pointcloud_data_signature = _pointcloud_data_signature(groups, selected_group_names)
    previous_data_signature = st.session_state.get(POINTCLOUD_DATA_SIGNATURE_STATE_KEY)
    if previous_data_signature is not None and previous_data_signature != pointcloud_data_signature:
        _clear_pointcloud_page_state(st.session_state)
    stored_result_signature = st.session_state.get("pipeline_result_data_signature")
    if st.session_state.get("pipeline_results") and stored_result_signature != pointcloud_data_signature:
        _clear_pointcloud_page_state(st.session_state)
    st.session_state[POINTCLOUD_DATA_SIGNATURE_STATE_KEY] = pointcloud_data_signature

    _apply_pending_section_plane()
    _apply_pending_section_offset()
    config = build_processing_config()
    configs_by_group = build_family_processing_configs(config, selected_group_names)
    if st.session_state.get("advanced_parameter_mode", False):
        render_parameter_manual()
    start_processing = st.sidebar.button(
        "开始处理当前数据",
        type="primary",
        use_container_width=True,
        disabled=workflow_view not in {"预览与框选", "开始处理"},
        help=(
            "框选主轴区域后可直接开始处理；也可以切换到“开始处理”窗口再执行。"
            if workflow_view == "预览与框选"
            else "执行当前已确认的点云、截面和配准参数。"
        ),
    )

    stored_results = st.session_state.get("pipeline_results", [])
    if show_preview and workflow_view == "预览与框选":
        preview_group = selected_group_names[0]
        preview_config = configs_by_group[preview_group]
        preview_result = next(
            (item for item in stored_results if item.group_name == preview_group),
            None,
        )
        use_merged_preview = (
            preview_result is not None
            and preview_result.merged_cloud_path.is_file()
        )
        preview_mode_label = "表面渲染" if preview_render_mode == "surface" else "点云渲染"
        try:
            with st.spinner(f"加载第 {preview_group} 组 {preview_sampling_level} {preview_mode_label}..."):
                if use_merged_preview:
                    reference_file, preview_data = load_merged_preview_data(
                        preview_result,
                        preview_config,
                        preview_voxel_size,
                        preview_maximum_points,
                    )
                    preview_files = [reference_file]
                    effective_preview_denoise = False
                else:
                    reference_file, preview_data = load_preview_data(
                        groups[preview_group],
                        preview_config,
                        preview_voxel_size,
                        preview_maximum_points,
                        preview_apply_denoise,
                    )
                    preview_files = groups[preview_group]
                    effective_preview_denoise = preview_apply_denoise
                figure = make_preview_figure(
                    preview_files,
                    preview_config,
                    preview_voxel_size,
                    preview_maximum_points,
                    effective_preview_denoise,
                    preview_render_mode,
                    preview_data,
                    reference_file,
                    preview_orthographic_projection,
                )
        except Exception as error:
            st.error(f"无法加载 3D 预览：{error}")
        else:
            if use_merged_preview:
                st.subheader(f"第 {preview_group} 组拼接后融合点云预览")
            else:
                st.subheader(f"第 {preview_group} 组单个参考 PLY 预览（尚未拼接）")
            st.plotly_chart(
                figure,
                width="stretch",
                key=(
                    f"{PREVIEW_RENDER_REVISION}-{preview_group}-{preview_sampling_level}-"
                    f"merged-{int(use_merged_preview)}-denoise-{int(effective_preview_denoise)}-{preview_render_mode}-"
                    f"orthographic-{int(preview_orthographic_projection)}"
                ),
                config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            )
            if st.session_state.get("section_region_selection", True):
                render_section_region_selector(preview_group, preview_data, config.section)

    if start_processing and workflow_view in {"预览与框选", "开始处理"}:
        if st.session_state.get("section_region_selection", True) and config.section.target_region is None:
            st.error("请先打开 3D 预览并框选最终要测量的目标区域；否则无法确认输出是目标部件的真实横截面。")
            st.stop()
        try:
            for group_config in configs_by_group.values():
                group_config.validate()
        except ValueError as error:
            st.error(str(error))
            st.stop()

        progress = st.progress(0, text="准备处理")
        status = st.empty()
        elapsed_status = st.empty()
        processing_started = time.monotonic()

        def format_elapsed(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            minutes, remainder = divmod(int(seconds), 60)
            hours, minutes = divmod(minutes, 60)
            if hours:
                return f"{hours}小时{minutes:02d}分{remainder:02d}秒"
            return f"{minutes}分{remainder:02d}秒"

        def show_elapsed() -> None:
            elapsed_status.caption(f"本次处理已用时间：{format_elapsed(time.monotonic() - processing_started)}")

        show_elapsed()
        results: list[PipelineResult] = []
        for index, group_name in enumerate(selected_group_names):
            base_progress = index / len(selected_group_names)
            progress.progress(int(base_progress * 100), text=f"处理中：第 {group_name} 组")

            def update_status(message: str, group_index: int = index) -> None:
                status.caption(message)
                show_elapsed()
                running_progress = (group_index + 0.45) / len(selected_group_names)
                progress.progress(min(99, int(running_progress * 100)), text=message)

            try:
                group_config = configs_by_group[group_name]
                result = process_group(
                    group_name=group_name,
                    files=groups[group_name],
                    config=group_config,
                    output_root=OUTPUT_ROOT,
                    progress_callback=update_status,
                )
                results.append(result)
            except Exception as error:
                st.exception(error)
                st.warning(f"第 {group_name} 组处理失败，已停止后续任务；可调整参数后重新运行。")
                break
        elapsed_seconds = time.monotonic() - processing_started
        progress.progress(100, text=f"处理结束（耗时 {format_elapsed(elapsed_seconds)}）")
        elapsed_status.caption(f"本次处理总耗时：{format_elapsed(elapsed_seconds)}")
        if results:
            st.session_state["pipeline_results"] = results
            st.session_state["pipeline_result_data_signature"] = pointcloud_data_signature
            st.session_state["pipeline_completion_message"] = (
                f"已完成 {len(results)} 组处理，总耗时 {format_elapsed(elapsed_seconds)}。"
                f"输出位于 `{OUTPUT_ROOT.resolve()}`。"
            )
            st.rerun()

    results = st.session_state.get("pipeline_results", [])
    completion_message = st.session_state.pop("pipeline_completion_message", None)
    if completion_message:
        st.success(completion_message)
    if results and workflow_view in {"结果审核", "CAD 测量"}:
        st.divider()
        for result in results:
            render_result(result, show_cad=workflow_view == "CAD 测量")


if __name__ == "__main__":
    main()
