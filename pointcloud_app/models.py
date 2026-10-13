from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Bounds3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def validate(self, label: str = "范围") -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"{label}的 X 最小值必须小于最大值。")
        if self.y_min >= self.y_max:
            raise ValueError(f"{label}的 Y 最小值必须小于最大值。")
        if self.z_min >= self.z_max:
            raise ValueError(f"{label}的 Z 最小值必须小于最大值。")

    @property
    def min_bound(self) -> tuple[float, float, float]:
        return (self.x_min, self.y_min, self.z_min)

    @property
    def max_bound(self) -> tuple[float, float, float]:
        return (self.x_max, self.y_max, self.z_max)


@dataclass
class Bounds2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def validate(self, label: str = "范围") -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"{label}的 X 最小值必须小于最大值。")
        if self.y_min >= self.y_max:
            raise ValueError(f"{label}的 Y 最小值必须小于最大值。")


@dataclass
class DenoiseConfig:
    voxel_size: float = 0.15
    use_statistical_filter: bool = True
    statistical_neighbors: int = 24
    statistical_std_ratio: float = 1.5
    use_radius_filter: bool = True
    radius: float = 0.25
    radius_min_points: int = 6
    use_component_filter: bool = True
    component_voxel_size: float = 0.18
    component_min_points: int = 400
    component_keep_largest: int = 8
    component_relative_size: float = 0.02
    component_gap_factor: float = 6.0
    secondary_radius_factor: float = 1.5
    secondary_min_points: int = 10
    use_surface_support_filter: bool = True
    surface_support_neighbors: int = 24
    surface_support_distance_factor: float = 0.4

    def validate(self) -> None:
        if self.surface_support_neighbors < 8:
            raise ValueError("Surface support requires at least 8 neighbors.")
        if not 0 < self.surface_support_distance_factor <= 1:
            raise ValueError("Surface support distance factor must be in (0, 1].")
        if self.voxel_size <= 0:
            raise ValueError("去噪体素大小必须大于 0。")
        if self.statistical_neighbors < 2:
            raise ValueError("统计滤波邻居数至少为 2。")
        if self.statistical_std_ratio <= 0:
            raise ValueError("统计滤波标准差倍率必须大于 0。")
        if self.use_radius_filter and (self.radius <= 0 or self.radius_min_points < 1):
            raise ValueError("半径滤波参数无效。")
        if self.use_component_filter:
            if self.component_voxel_size <= 0:
                raise ValueError("连通区域体素大小必须大于 0。")
            if self.component_min_points < 1 or self.component_keep_largest < 1:
                raise ValueError("连通区域保留参数必须大于 0。")
            if not 0 < self.component_relative_size <= 1:
                raise ValueError("连通区域相对规模必须在 0 到 1 之间。")
            if self.component_gap_factor <= 0:
                raise ValueError("连通区域间隙倍率必须大于 0。")
            if self.secondary_radius_factor <= 1.0 or self.secondary_min_points < 2:
                raise ValueError("二次局部密度过滤参数无效。")


@dataclass
class CalibrationConfig:
    use_roi: bool = True
    bounds: Bounds3D = field(
        default_factory=lambda: Bounds3D(
            x_min=4.0,
            x_max=17.5,
            y_min=2.0,
            y_max=17.0,
            z_min=-6.5,
            z_max=8.0,
        )
    )
    minimum_points: int = 800
    # The fixture is a calibration target, not part of the measured workpiece.
    # Keep the switch explicit so operators can temporarily retain it for
    # troubleshooting without changing the registration ROI itself.
    exclude_from_fusion: bool = True
    exclusion_margin: float = 0.03
    level_reference_plane: bool = True
    leveling_plane_distance: float = 0.12
    leveling_minimum_points: int = 500

    def validate(self) -> None:
        if self.use_roi:
            self.bounds.validate("标定块 ROI")
        if self.minimum_points < 30:
            raise ValueError("标定块最少点数至少为 30。")
        if self.exclusion_margin < 0:
            raise ValueError("标定块融合排除边界余量不能为负数。")
        if self.level_reference_plane and not self.use_roi:
            raise ValueError("启用标定块自动调平时，必须启用标定块 ROI。")
        if self.leveling_plane_distance <= 0:
            raise ValueError("标定块调平平面距离阈值必须大于 0。")
        if self.leveling_minimum_points < 30:
            raise ValueError("标定块调平基准面最少点数至少为 30。")


@dataclass
class RegistrationConfig:
    registration_voxel_size: float = 0.35
    use_global_registration: bool = True
    reference_vertex_count: int = 1_114_000
    normal_radius_factor: float = 2.5
    feature_radius_factor: float = 5.0
    global_distance_factor: float = 1.5
    global_max_iterations: int = 100_000
    global_attempts: int = 3
    global_random_seed: int = 5
    fallback_icp_fitness_threshold: float = 0.75
    skip_low_quality_views: bool = True
    minimum_full_cloud_fitness: float = 0.70
    # Quality gates for the calibration block are intentionally independent
    # from workpiece/full-cloud metrics.  A view that cannot register the
    # fixture reliably must not be allowed to move the pose graph or produce
    # a formal measurement result.
    minimum_calibration_icp_fitness: float = 0.65
    maximum_calibration_icp_rmse: float = 0.16
    skip_low_quality_calibration_views: bool = True
    # Multi-view captures are intended to complement each other.  Calibration
    # block quality may gate a pose, but workpiece overlap diagnostics must not
    # remove a view or erase its unique coverage from the fused cloud.
    coverage_preserving_fusion: bool = True
    use_section_alignment_validation: bool = True
    section_alignment_thickness: float = 0.05
    section_alignment_overlap_distance: float = 0.35
    section_alignment_max_p90_distance: float = 0.115
    section_alignment_minimum_overlap_points: int = 500
    section_alignment_probe_count: int = 5
    section_alignment_probe_spacing: float = 0.50
    section_alignment_minimum_views: int = 3
    icp_max_correspondence_distance: float = 0.90
    icp_max_iterations: int = 100
    use_point_to_plane_icp: bool = True
    # A broad first pass followed by tighter ICP stages is more stable on
    # partially visible polyhedral calibration blocks than one loose pass.
    use_multiscale_icp: bool = True
    use_six_station_pose_prior: bool = True
    station_pose_corrections: dict[str, dict[str, float]] = field(default_factory=dict)
    pose_prior_calibration_voxel_size: float = 0.15
    pose_prior_translation_max_correspondence_distance: float = 0.90
    pose_prior_rigid_max_rotation_degrees: float = 1.50
    pose_prior_rigid_max_translation: float = 0.30
    calibration_graph_voxel_size: float = 0.12
    calibration_graph_max_correspondence_distance: float = 0.40
    calibration_graph_icp_max_iterations: int = 200
    calibration_graph_max_rotation_degrees: float = 1.50
    calibration_graph_max_translation: float = 0.30
    calibration_graph_minimum_adjacent_fitness: float = 0.35
    calibration_graph_minimum_loop_fitness: float = 0.45
    calibration_graph_maximum_adjacent_rmse: float = 0.14
    calibration_graph_maximum_loop_rmse: float = 0.16
    workpiece_graph_voxel_size: float = 0.12
    use_overlap_refinement: bool = True
    use_global_pose_validation: bool = True
    workpiece_graph_max_correspondence_distance: float = 0.35
    workpiece_graph_icp_max_iterations: int = 200
    workpiece_graph_max_rotation_degrees: float = 1.25
    workpiece_graph_max_translation: float = 0.30
    workpiece_graph_minimum_adjacent_fitness: float = 0.20
    workpiece_graph_minimum_loop_fitness: float = 0.30
    # Use the calibration block's measured planes as a geometric constraint
    # before ICP.  This is the 3D equivalent of matching the patent's common
    # contour/edge field instead of allowing a single loose surface match to
    # decide the complete pose.
    use_calibration_geometry_constraints: bool = True
    calibration_geometry_voxel_size: float = 0.08
    calibration_geometry_plane_distance: float = 0.05
    calibration_geometry_minimum_plane_points: int = 80
    calibration_geometry_minimum_planes: int = 3
    calibration_geometry_max_planes: int = 8
    calibration_geometry_max_normal_angle_degrees: float = 15.0
    calibration_geometry_max_centroid_distance: float = 2.50

    def validate(self) -> None:
        if self.registration_voxel_size <= 0:
            raise ValueError("拼接体素大小必须大于 0。")
        if self.reference_vertex_count < 1:
            raise ValueError("参考视角目标点数必须大于 0。")
        if self.normal_radius_factor <= 0 or self.feature_radius_factor <= 0:
            raise ValueError("法线和特征半径倍率必须大于 0。")
        if self.global_distance_factor <= 0:
            raise ValueError("全局配准距离倍率必须大于 0。")
        if self.global_max_iterations < 1:
            raise ValueError("全局配准迭代次数必须大于 0。")
        if self.global_attempts < 1:
            raise ValueError("全局配准重试次数必须大于 0。")
        if self.global_random_seed < 0:
            raise ValueError("Global-registration random seed must be non-negative.")
        if not 0 < self.fallback_icp_fitness_threshold <= 1:
            raise ValueError("回退 ICP fitness 阈值必须在 (0, 1] 内。")
        if not 0 < self.minimum_full_cloud_fitness <= 1:
            raise ValueError("全云融合最低 fitness 阈值必须在 (0, 1] 内。")
        if not 0 < self.minimum_calibration_icp_fitness <= 1:
            raise ValueError("标定块 ICP 最低 fitness 阈值必须在 (0, 1] 内。")
        if self.maximum_calibration_icp_rmse <= 0:
            raise ValueError("标定块 ICP 最大 RMSE 必须大于 0。")
        if self.section_alignment_thickness <= 0:
            raise ValueError("Section-alignment slice thickness must be positive.")
        if self.section_alignment_overlap_distance <= 0:
            raise ValueError("Section-alignment overlap distance must be positive.")
        if not 0 < self.section_alignment_max_p90_distance <= self.section_alignment_overlap_distance:
            raise ValueError("Section-alignment p90 threshold must be within the overlap distance.")
        if self.section_alignment_minimum_overlap_points < 1:
            raise ValueError("Section-alignment minimum overlap count must be at least one.")
        if self.section_alignment_probe_count < 1 or self.section_alignment_probe_count % 2 == 0:
            raise ValueError("Section-alignment probe count must be a positive odd number.")
        if self.section_alignment_probe_spacing <= 0:
            raise ValueError("Section-alignment probe spacing must be positive.")
        if self.section_alignment_minimum_views < 1:
            raise ValueError("Section-alignment minimum retained views must be positive.")
        if self.icp_max_correspondence_distance <= 0 or self.icp_max_iterations < 1:
            raise ValueError("ICP 参数无效。")
        correction_fields = {
            "rotation_x_degrees",
            "rotation_y_degrees",
            "rotation_z_degrees",
            "translation_x",
            "translation_y",
            "translation_z",
        }
        valid_station_labels = {"reference", "minus10", "minus28", "minus50", "plus37", "plus47"}
        for station_label, correction in self.station_pose_corrections.items():
            if station_label not in valid_station_labels:
                raise ValueError(f"未知工位先验修正标签：{station_label}。")
            if set(correction) - correction_fields:
                raise ValueError(f"工位 {station_label} 包含未知的先验修正字段。")
            if any(not isinstance(value, (int, float)) for value in correction.values()):
                raise ValueError(f"工位 {station_label} 的先验修正值必须是数字。")
        positive_pose_parameters = {
            "工位初调标定块体素": self.pose_prior_calibration_voxel_size,
            "锁旋转平移 ICP 最大对应距离": self.pose_prior_translation_max_correspondence_distance,
            "工位刚性修正最大旋转": self.pose_prior_rigid_max_rotation_degrees,
            "工位刚性修正最大平移": self.pose_prior_rigid_max_translation,
            "标定块位姿图体素": self.calibration_graph_voxel_size,
            "标定块位姿图最大对应距离": self.calibration_graph_max_correspondence_distance,
            "标定块位姿图最大旋转": self.calibration_graph_max_rotation_degrees,
            "标定块位姿图最大平移": self.calibration_graph_max_translation,
            "工件位姿图体素": self.workpiece_graph_voxel_size,
            "工件位姿图最大对应距离": self.workpiece_graph_max_correspondence_distance,
            "工件位姿图最大旋转": self.workpiece_graph_max_rotation_degrees,
            "工件位姿图最大平移": self.workpiece_graph_max_translation,
        }
        invalid_parameters = [label for label, value in positive_pose_parameters.items() if value <= 0]
        if invalid_parameters:
            raise ValueError(f"{invalid_parameters[0]}必须大于 0。")
        if self.calibration_graph_icp_max_iterations < 1 or self.workpiece_graph_icp_max_iterations < 1:
            raise ValueError("位姿图 ICP 最大迭代次数必须大于 0。")
        geometry_parameters = {
            "标定块几何体素": self.calibration_geometry_voxel_size,
            "标定块平面距离": self.calibration_geometry_plane_distance,
            "标定块最大法向角": self.calibration_geometry_max_normal_angle_degrees,
            "标定块最大平面中心距离": self.calibration_geometry_max_centroid_distance,
        }
        invalid_geometry = [label for label, value in geometry_parameters.items() if value <= 0]
        if invalid_geometry:
            raise ValueError(f"{invalid_geometry[0]}必须大于 0。")
        if self.calibration_geometry_minimum_plane_points < 30:
            raise ValueError("标定块几何约束的单平面最少点数至少为 30。")
        if self.calibration_geometry_minimum_planes < 2:
            raise ValueError("标定块几何约束至少需要 2 个平面。")
        if self.calibration_geometry_max_planes < self.calibration_geometry_minimum_planes:
            raise ValueError("标定块几何约束的最大平面数不能小于最少平面数。")
        if self.calibration_geometry_max_normal_angle_degrees >= 90:
            raise ValueError("标定块几何约束的最大法向角必须小于 90°。")
        graph_fitness_parameters = {
            "标定块相邻边最低 fitness": self.calibration_graph_minimum_adjacent_fitness,
            "标定块闭环边最低 fitness": self.calibration_graph_minimum_loop_fitness,
            "工件相邻边最低 fitness": self.workpiece_graph_minimum_adjacent_fitness,
            "工件闭环边最低 fitness": self.workpiece_graph_minimum_loop_fitness,
        }
        invalid_fitness = [
            label for label, value in graph_fitness_parameters.items() if not 0 < value <= 1
        ]
        if invalid_fitness:
            raise ValueError(f"{invalid_fitness[0]}必须在 (0, 1] 内。")
        graph_rmse_parameters = {
            "标定块相邻边最大 RMSE": self.calibration_graph_maximum_adjacent_rmse,
            "标定块闭环边最大 RMSE": self.calibration_graph_maximum_loop_rmse,
        }
        invalid_rmse = [label for label, value in graph_rmse_parameters.items() if value <= 0]
        if invalid_rmse:
            raise ValueError(f"{invalid_rmse[0]}必须大于 0。")


@dataclass
class SectionConfig:
    angle_degrees: float = 0.0
    tilt_degrees: float = 0.0
    plane_offset: float = 9.0
    thickness: float = 0.15
    horizontal_min: float = -30.0
    horizontal_max: float = 30.0
    vertical_min: float = -8.0
    vertical_max: float = 10.0
    # Exported section images need a readable marker at the current physical
    # sampling density; this changes display only, never the stored points.
    point_size: float = 2.2
    target_region: Bounds2D | None = None
    use_statistical_filter: bool = True
    statistical_neighbors: int = 12
    statistical_std_ratio: float = 1.0
    use_component_filter: bool = True
    component_cell_size: float = 0.30
    component_min_points: int = 12
    # A nearby parallel plane can recover a complete physical edge when the
    # initial top-view pick lands between two surfaces.  The optimizer only
    # changes the section plane offset; it never moves, interpolates, or edits
    # point coordinates.
    auto_offset_enabled: bool = True
    offset_scan_half_range: float = 4.0
    offset_scan_step: float = 0.25
    offset_min_quality_gain: float = 0.05

    def validate(self) -> None:
        if not -89.9 <= self.tilt_degrees <= 89.9:
            raise ValueError("截面法向俯仰角必须在 -89.9° 到 89.9° 之间。")
        if self.thickness <= 0:
            raise ValueError("切片厚度必须大于 0。")
        if self.horizontal_min >= self.horizontal_max:
            raise ValueError("截面横向最小值必须小于最大值。")
        if self.vertical_min >= self.vertical_max:
            raise ValueError("截面竖向最小值必须小于最大值。")
        if self.point_size <= 0:
            raise ValueError("截面点大小必须大于 0。")
        if self.target_region is not None:
            self.target_region.validate("目标截面区域")
        if self.statistical_neighbors < 2:
            raise ValueError("截面统计滤波邻居数至少为 2。")
        if self.statistical_std_ratio <= 0:
            raise ValueError("截面统计滤波标准差倍率必须大于 0。")
        if self.component_cell_size <= 0:
            raise ValueError("截面连通域网格大小必须大于 0。")
        if self.component_min_points < 1:
            raise ValueError("截面连通域最少点数必须至少为 1。")
        if self.offset_scan_half_range < 0:
            raise ValueError("截面偏移自动优化范围不能为负数。")
        if self.offset_scan_step <= 0:
            raise ValueError("截面偏移自动优化步长必须大于 0。")
        if self.offset_scan_step > max(self.offset_scan_half_range, 0.25) * 2.0:
            raise ValueError("截面偏移自动优化步长过大，无法覆盖有效扫描范围。")
        if not 0 <= self.offset_min_quality_gain <= 1:
            raise ValueError("截面偏移自动优化最低质量提升必须在 0 到 1 之间。")


@dataclass
class ProcessingConfig:
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    section: SectionConfig = field(default_factory=SectionConfig)
    fusion_voxel_size: float = 0.03
    compute_backend: str = "auto"
    # These fields are metadata for reproducibility when one batch contains
    # several point-cloud families with different stitching behavior.
    data_family: str | None = None
    profile_name: str | None = None
    profile_source: str | None = None

    def validate(self) -> None:
        self.denoise.validate()
        self.calibration.validate()
        self.registration.validate()
        self.section.validate()
        if self.fusion_voxel_size <= 0:
            raise ValueError("融合体素大小必须大于 0。")
        if self.compute_backend not in {"auto", "gpu", "cpu"}:
            raise ValueError("计算后端必须是自动、GPU 或 CPU。")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
