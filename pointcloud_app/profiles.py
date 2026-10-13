"""Shared acquisition settings; family names are display metadata only."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from .models import ProcessingConfig


COMMON_OVERRIDES: dict[str, object] = {
    "denoise": {
        "voxel_size": 0.12,
        "statistical_neighbors": 24,
        "statistical_std_ratio": 1.50,
        "use_radius_filter": True,
        "radius": 0.25,
        "radius_min_points": 6,
        "use_component_filter": True,
        "component_voxel_size": 0.18,
        "component_min_points": 400,
        "component_keep_largest": 8,
    },
    "registration": {
        "registration_voxel_size": 0.28,
        "pose_prior_calibration_voxel_size": 0.12,
        "pose_prior_translation_max_correspondence_distance": 0.75,
        "pose_prior_rigid_max_rotation_degrees": 1.50,
        "pose_prior_rigid_max_translation": 0.30,
        "icp_max_correspondence_distance": 0.75,
        "icp_max_iterations": 150,
        "calibration_graph_voxel_size": 0.08,
        "calibration_graph_max_correspondence_distance": 0.32,
        "calibration_graph_icp_max_iterations": 300,
        "minimum_calibration_icp_fitness": 0.65,
        "maximum_calibration_icp_rmse": 0.16,
        "calibration_graph_minimum_adjacent_fitness": 0.45,
        "calibration_graph_minimum_loop_fitness": 0.50,
        "calibration_graph_maximum_adjacent_rmse": 0.14,
        "calibration_graph_maximum_loop_rmse": 0.16,
        "section_alignment_overlap_distance": 0.45,
        "section_alignment_max_p90_distance": 0.18,
        "section_alignment_minimum_overlap_points": 250,
        "use_section_alignment_validation": True,
        "use_overlap_refinement": True,
    },
    "fusion_voxel_size": 0.03,
}

def _family_overrides(
    *,
    denoise: Mapping[str, object] | None = None,
    registration: Mapping[str, object] | None = None,
    fusion_voxel_size: float | None = None,
) -> dict[str, object]:
    """Build a family profile from a shared baseline with bounded material tuning."""
    result = copy.deepcopy(COMMON_OVERRIDES)
    if denoise:
        result["denoise"].update(denoise)
    if registration:
        result["registration"].update(registration)
    if fusion_voxel_size is not None:
        result["fusion_voxel_size"] = fusion_voxel_size
    return result


DATA_FAMILY_PROFILES = {
    # c1: dense, repeatable numbered captures. Preserve small details and use
    # a tighter overlap gate because these views have the strongest overlap.
    "c1": {
        "label": "c1 数字组（1～4）",
        "name": "c1｜高重叠精细",
        "description": "数字组使用较细体素和较紧重叠门限，减少边缘重影。",
        "overrides": _family_overrides(
            denoise={
                "voxel_size": 0.10,
                "statistical_neighbors": 30,
                "statistical_std_ratio": 1.20,
                "radius": 0.20,
                "radius_min_points": 8,
                "secondary_radius_factor": 1.35,
                "secondary_min_points": 12,
                "component_voxel_size": 0.15,
                "component_min_points": 350,
                "component_relative_size": 0.035,
                "component_gap_factor": 4.0,
            },
            registration={
                "registration_voxel_size": 0.24,
                "icp_max_correspondence_distance": 0.60,
                "workpiece_graph_max_correspondence_distance": 0.28,
                "section_alignment_overlap_distance": 0.36,
            },
            fusion_voxel_size=0.025,
        ),
    },
    # c2: broad wood faces with more sparse/uneven returns. A slightly broader
    # gate prevents rejecting valid complementary views while stronger
    # component cleanup suppresses isolated surface noise.
    "c2": {
        "label": "c2 木板组（木板、木板2）",
        "name": "c2｜木板稀疏表面",
        "description": "木板组放宽稀疏表面的对应容差，同时加强离散噪声抑制。",
        "overrides": _family_overrides(
            denoise={
                "voxel_size": 0.14,
                "statistical_std_ratio": 1.70,
                "radius": 0.30,
                "radius_min_points": 5,
                "component_voxel_size": 0.20,
                "component_min_points": 300,
            },
            registration={
                "registration_voxel_size": 0.32,
                "icp_max_correspondence_distance": 0.85,
                "workpiece_graph_max_correspondence_distance": 0.38,
                "section_alignment_overlap_distance": 0.50,
                "section_alignment_max_p90_distance": 0.22,
            },
            fusion_voxel_size=0.035,
        ),
    },
    # c3: variant wood captures have fewer views and weaker overlap. Keep
    # unique coverage, but make pose correction conservative to avoid sliding
    # along a visually flat board surface.
    "c3": {
        "label": "c3 木板变体组（木板-2～木板-6）",
        "name": "c3｜低重叠保守",
        "description": "木板变体组优先保留互补覆盖，并限制低重叠方向的过度修正。",
        "overrides": _family_overrides(
            denoise={
                "voxel_size": 0.13,
                "statistical_std_ratio": 1.65,
                "component_voxel_size": 0.19,
                "component_min_points": 280,
            },
            registration={
                "registration_voxel_size": 0.30,
                "icp_max_correspondence_distance": 0.78,
                "workpiece_graph_max_correspondence_distance": 0.32,
                "workpiece_graph_max_rotation_degrees": 0.90,
                "workpiece_graph_max_translation": 0.22,
                "section_alignment_overlap_distance": 0.46,
                "section_alignment_max_p90_distance": 0.20,
            },
            fusion_voxel_size=0.032,
        ),
    },
    "numeric": {},
    "wood_base": {},
    "wood_variant": {},
    "default": {},
}


def classify_group_family(group_name: str) -> str:
    """Classify groups into the three media families, with legacy aliases."""
    name = str(group_name).strip()
    if name in {"1", "2", "3", "4"}:
        return "c1"
    if name in {"木板", "木板2"}:
        return "c2"
    if name.startswith("木板-"):
        return "c3"
    return "default"


def _apply_config_overrides(config: ProcessingConfig, overrides: Mapping[str, object]) -> None:
    for section_name, section_overrides in overrides.items():
        if section_name == "fusion_voxel_size":
            config.fusion_voxel_size = float(section_overrides)
            continue
        target = getattr(config, section_name, None)
        if target is None or not isinstance(section_overrides, Mapping):
            continue
        for field_name, value in section_overrides.items():
            if hasattr(target, field_name):
                setattr(target, field_name, value)


def apply_family_profile(config: ProcessingConfig, family: str, *,
                         source: str = "auto") -> ProcessingConfig:
    family = {"numeric": "c1", "wood_base": "c2", "wood_variant": "c3"}.get(family, family)
    family = family if family in DATA_FAMILY_PROFILES else "default"
    tuned = copy.deepcopy(config)
    if source == "auto":
        profile = DATA_FAMILY_PROFILES[family]
        overrides = profile.get("overrides") or COMMON_OVERRIDES
        _apply_config_overrides(tuned, overrides)
    tuned.data_family = family
    tuned.profile_name = DATA_FAMILY_PROFILES[family].get("name") if source == "auto" else "通用参数"
    tuned.profile_source = source
    return tuned
