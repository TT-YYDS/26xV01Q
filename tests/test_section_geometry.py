import unittest
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import open3d as o3d

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
    clean_vertical_section,
    diagnose_section_integrity,
    evaluate_section_alignment,
    evaluate_section_offset_candidates,
    estimate_calibration_leveling,
    extract_vertical_section,
    extract_full_resolution_section,
    _filter_section_parts_by_local_consensus,
    _apply_section_consensus_station_filter,
    _profile_alignment_has_extra_contour_evidence,
    _local_consensus_profile_masks,
    _collapse_parallel_section_layers,
    _section_display_contour_mask,
    _support_weighted_section_fusion,
    _representative_voxel_fusion,
    _representative_voxel_down_sample_cloud,
    _spatial_component_inlier_mask,
    _voxel_component_labels,
    _merge_clouds_with_section_consensus,
    _recover_section_density_from_raw,
    coarse_calibration_register,
    full_cloud_alignment_quality,
    fine_full_cloud_registration_config,
    _pairwise_pose_graph_edge,
    calibration_registration_quality,
    build_and_optimize_pairwise_pose_graph,
    normal_plane_from_xyz_region,
    optimize_multiframe_pose_graph,
    process_group,
    RegistrationMetric,
    registration_candidate_score,
    refine_translation_only,
    render_multi_section_audit,
    remove_calibration_region,
    order_files_with_six_station_pose_prior,
    order_files_with_reference,
    select_reference_index_by_calibration_quality,
    select_section_consistent_pose_candidates,
    select_vertical_section_indices,
    scan_section_offsets,
    six_station_pose_transformations,
    vertical_plane_basis,
    vertical_plane_from_xy_points,
    validate_section_view_alignment,
)


class VerticalSectionTests(unittest.TestCase):
    def test_generic_registration_uses_stable_station_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / f"{number}.ply" for number in range(1, 7)]
            order = (1, 5, 6, 4, 3, 2)
            registration_order = (1, 6, 3, 2, 4, 5)
            labels = ("reference", "minus10", "minus28", "plus37", "minus50", "plus47")
            (root / "station_order.json").write_text(json.dumps({
                "views": [
                    {"file": f"{number}.ply", "station": label}
                    for number, label in zip(order, labels)
                ],
            }), encoding="utf-8")
            with patch("pointcloud_app.pipeline.select_reference_file", return_value=files[0]):
                self.assertEqual(
                    order_files_with_reference(files, 0),
                    [root / f"{number}.ply" for number in registration_order],
                )
            (root / "station_order.json").write_text(json.dumps({
                "views": [
                    {"file": f"{number}.ply", "station": str(index)}
                    for index, number in enumerate(order)
                ],
            }), encoding="utf-8")
            with patch("pointcloud_app.pipeline.select_reference_file", return_value=files[0]):
                self.assertEqual(
                    order_files_with_reference(files, 0),
                    [root / f"{number}.ply" for number in order],
                )
            (root / "station_order.json").unlink()
            with patch("pointcloud_app.pipeline.select_reference_file", return_value=files[0]):
                self.assertEqual(order_files_with_reference(files, 0), files)

    def test_raw_section_density_recovery_is_gated_by_retained_voxel_and_distance(self) -> None:
        retained_points = np.array(
            [[0.01, 0.01, 0.0], [0.51, 0.51, 0.0]],
            dtype=np.float64,
        )
        raw_points = np.array(
            [
                [0.01, 0.01, 0.0],
                [0.02, 0.03, 0.0],
                [0.04, 0.06, 0.0],
                [0.08, 0.08, 0.0],
                [0.51, 0.51, 0.0],
                [0.90, 0.90, 0.0],
            ],
            dtype=np.float64,
        )
        section = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.15,
            horizontal_min=-2.0,
            horizontal_max=2.0,
            vertical_min=-2.0,
            vertical_max=2.0,
            use_statistical_filter=False,
            use_component_filter=False,
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "raw.ply"
            o3d.io.write_point_cloud(
                str(path),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_points)),
            )
            recovered, report = _recover_section_density_from_raw(
                path,
                o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(retained_points)
                ),
                np.eye(4),
                np.eye(4),
                section,
                denoise_voxel_size=0.1,
                backend="cpu",
            )

        recovered_points = np.asarray(recovered.points)
        self.assertEqual(report["status"], "recovered")
        self.assertEqual(report["raw_section_candidates"], 3)
        self.assertGreater(report["voxel_matched_candidates"], report["recovered_points"])
        self.assertEqual(report["recovered_points"], 2)
        self.assertFalse(np.any(np.all(np.isclose(recovered_points, [0.90, 0.90, 0.0]), axis=1)))

    def test_fusion_can_restore_raw_section_sampling_without_reintroducing_far_raw_points(self) -> None:
        clean_points = np.column_stack(
            (
                np.full(11, 0.01),
                np.linspace(0.01, 1.01, 11),
                np.zeros(11),
            )
        )
        raw_points = np.column_stack(
            (
                np.full(101, 0.01),
                np.linspace(0.01, 1.01, 101),
                np.zeros(101),
            )
        )
        raw_points = np.vstack((raw_points, [50.0, 50.0, 50.0]))
        section = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.15,
            horizontal_min=-2.0,
            horizontal_max=2.0,
            vertical_min=-2.0,
            vertical_max=2.0,
            use_statistical_filter=False,
            use_component_filter=False,
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "raw.ply"
            o3d.io.write_point_cloud(
                str(path),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_points)),
            )
            merged, report = _merge_clouds_with_section_consensus(
                [path],
                [np.eye(4)],
                [True],
                np.eye(4),
                section,
                voxel_size=0.025,
                backend="cpu",
                coverage_preserving=True,
                section_recovery_voxel_size=0.1,
                source_clouds=[
                    o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(clean_points)
                    )
                ],
            )

        merged_points = np.asarray(merged.points)
        recovery = report["section_density_recovery"]
        self.assertTrue(recovery["enabled"])
        self.assertGreater(recovery["recovered_points"], 0)
        self.assertGreater(len(merged_points), len(clean_points))
        self.assertFalse(np.any(np.linalg.norm(merged_points - [50.0, 50.0, 50.0], axis=1) < 1.0))

    def test_voxel_component_labels_handles_disconnected_cells(self) -> None:
        keys = np.asarray([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.int64)
        count, labels = _voxel_component_labels(keys)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_spatial_component_filter_removes_small_floating_islands(self) -> None:
        main = np.column_stack(
            (
                np.linspace(0.0, 4.0, 500),
                np.zeros(500),
                np.zeros(500),
            )
        )
        island = np.column_stack(
            (
                np.linspace(20.0, 20.4, 25),
                np.ones(25),
                np.ones(25),
            )
        )
        points = np.vstack((main, island))
        mask = _spatial_component_inlier_mask(
            points,
            voxel_size=0.2,
            minimum_points=100,
            keep_largest=1,
        )
        self.assertEqual(int(np.count_nonzero(mask)), len(main))
        self.assertFalse(mask[-1])

    def test_section_validation_reuses_supplied_clouds(self) -> None:
        points = np.column_stack((
            np.zeros(40),
            np.linspace(0.0, 3.9, 40),
            np.sin(np.linspace(0.0, 3.9, 40)),
        ))
        clouds = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.copy())),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.copy())),
        ]
        paths = [Path(f"missing-{index}.ply") for index in range(3)]
        metrics = [
            RegistrationMetric(
                file_name=path.name,
                calibration_points=0,
                global_fitness=None,
                global_rmse=None,
                icp_fitness=1.0,
                icp_rmse=0.0,
                transformation=np.eye(4).tolist(),
                pose_prior_label="reference" if index == 0 else f"view-{index}",
            )
            for index, path in enumerate(paths)
        ]
        section_config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.05,
            horizontal_min=-2.0,
            horizontal_max=5.0,
            vertical_min=-3.0,
            vertical_max=3.0,
            use_statistical_filter=False,
            use_component_filter=False,
        )
        registration_config = RegistrationConfig(
            use_section_alignment_validation=True,
            section_alignment_thickness=0.05,
            section_alignment_overlap_distance=0.35,
            section_alignment_max_p90_distance=0.10,
            section_alignment_minimum_overlap_points=8,
            section_alignment_probe_count=3,
            section_alignment_probe_spacing=0.50,
            section_alignment_minimum_views=2,
            coverage_preserving_fusion=True,
        )
        with patch("pointcloud_app.pipeline.read_point_cloud",
                   side_effect=AssertionError("should reuse supplied clouds")):
            report = validate_section_view_alignment(
                paths,
                [np.eye(4) for _ in paths],
                metrics,
                np.eye(4),
                section_config,
                registration_config,
                "cpu",
                source_clouds=clouds,
            )
        self.assertEqual(report["input_source"], "single_view_denoised_clouds")

    @staticmethod
    def _multi_probe_cloud(
        layers: list[tuple[float, float, float]],
        horizontal_maximum: float = 3.9,
        sample_count: int = 40,
    ) -> np.ndarray:
        horizontal = np.linspace(0.0, horizontal_maximum, sample_count)
        point_sets = []
        for plane_offset, vertical_offset, coverage_end in layers:
            covered_horizontal = horizontal[horizontal <= coverage_end]
            point_sets.append(
                np.column_stack(
                    (
                        np.full(len(covered_horizontal), plane_offset),
                        covered_horizontal,
                        np.sin(covered_horizontal) + vertical_offset,
                    )
                )
            )
        return np.concatenate(point_sets, axis=0) if point_sets else np.empty((0, 3))

    def _validate_multi_probe_clouds(
        self,
        point_sets: list[np.ndarray],
        maximum_p90_distance: float = 0.10,
        coverage_preserving_fusion: bool = False,
    ) -> tuple[list[RegistrationMetric], dict[str, object]]:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / f"view-{index}.ply" for index in range(len(point_sets))]
            for path, points in zip(paths, point_sets, strict=True):
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                )
            metrics = [
                RegistrationMetric(
                    file_name=path.name,
                    calibration_points=0,
                    global_fitness=None,
                    global_rmse=None,
                    icp_fitness=1.0,
                    icp_rmse=0.0,
                    transformation=np.eye(4).tolist(),
                    pose_prior_label="reference" if index == 0 else f"view-{index}",
                )
                for index, path in enumerate(paths)
            ]
            section_config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.05,
                horizontal_min=-2.0,
                horizontal_max=5.0,
                vertical_min=-3.0,
                vertical_max=3.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )
            registration_config = RegistrationConfig(
                section_alignment_thickness=0.05,
                section_alignment_overlap_distance=0.35,
                section_alignment_max_p90_distance=maximum_p90_distance,
                section_alignment_minimum_overlap_points=8,
                section_alignment_probe_count=3,
                section_alignment_probe_spacing=0.50,
                section_alignment_minimum_views=2,
                coverage_preserving_fusion=coverage_preserving_fusion,
            )
            report = validate_section_view_alignment(
                paths,
                [np.eye(4) for _ in paths],
                metrics,
                np.eye(4),
                section_config,
                registration_config,
                "cpu",
            )
        return metrics, report

    def test_extra_contour_evidence_requires_sufficient_shared_coverage(self) -> None:
        quality = {
            "compared_points": 120,
            "contour_compared_bins": 24,
            "duplicate_contour_ratio": 0.34,
            "missing_contour_ratio": 0.08,
            "common_profile_ratio": 0.82,
            "overlap_ratio": 0.86,
        }
        self.assertTrue(_profile_alignment_has_extra_contour_evidence(quality, 30))

        low_coverage = dict(quality)
        low_coverage["overlap_ratio"] = 0.62
        self.assertFalse(_profile_alignment_has_extra_contour_evidence(low_coverage, 30))

        balanced_contours = dict(quality)
        balanced_contours["missing_contour_ratio"] = 0.25
        self.assertFalse(_profile_alignment_has_extra_contour_evidence(balanced_contours, 30))

    def test_profile_consensus_excludes_anomalous_station(self) -> None:
        horizontal = np.linspace(0.0, 3.9, 40)
        good_profile = np.column_stack((horizontal, np.sin(horizontal)))
        anomalous_profile = good_profile.copy()
        anomalous_profile[:, 1] += 0.50
        profiles = {
            0: good_profile,
            1: good_profile.copy(),
            2: good_profile.copy(),
            3: anomalous_profile,
        }
        metrics = [
            RegistrationMetric(
                file_name=f"view-{index}.ply",
                calibration_points=100,
                global_fitness=None,
                global_rmse=None,
                icp_fitness=1.0,
                icp_rmse=0.0,
                transformation=np.eye(4).tolist(),
                included_in_fusion=True,
                pose_prior_label=label,
            )
            for index, label in enumerate(("reference", "minus10", "minus28", "plus37"))
        ]
        report = _apply_section_consensus_station_filter(
            profiles,
            [0, 1, 2, 3],
            metrics,
            reference_index=0,
            registration_config=RegistrationConfig(
                section_alignment_minimum_overlap_points=8,
                section_alignment_minimum_views=3,
            ),
        )

        self.assertEqual(report["excluded_indices"], [3])
        self.assertEqual(report["final_active_indices"], [0, 1, 2])
        self.assertFalse(metrics[3].included_in_fusion)
        self.assertEqual(metrics[3].section_validation_status, "mismatch")
        self.assertEqual(metrics[3].exclusion_reason, "section_consensus")
        self.assertGreater(metrics[3].section_consensus_removed_ratio or 0.0, 0.90)

    def test_profile_consensus_keeps_unique_coverage_bins(self) -> None:
        base = np.column_stack((np.linspace(0.0, 1.9, 20), np.zeros(20)))
        extended = np.column_stack((np.linspace(0.0, 3.9, 40), np.zeros(40)))
        masks, reports = _local_consensus_profile_masks(
            [base, base.copy(), extended],
        )

        self.assertTrue(np.all(masks[0]))
        self.assertTrue(np.all(masks[1]))
        self.assertTrue(np.all(masks[2]))
        self.assertEqual(reports[2]["consensus_removed_points"], 0)

    def test_local_consensus_removes_isolated_parallel_track(self) -> None:
        config = SectionConfig(
            angle_degrees=0.0,
            tilt_degrees=0.0,
            plane_offset=0.0,
            thickness=0.10,
        )
        horizontal_positions = np.linspace(0.0, 1.0, 21)
        good = np.column_stack(
            (
                np.zeros_like(horizontal_positions),
                horizontal_positions,
                np.zeros_like(horizontal_positions),
            )
        )
        good_offset = good.copy()
        good_offset[:, 2] = 0.01
        isolated = good.copy()
        isolated[:, 2] = 0.35
        filtered = _filter_section_parts_by_local_consensus(
            [
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(good)),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(good_offset)),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(isolated)),
            ],
            config,
        )
        self.assertEqual(len(filtered), 2)
        self.assertEqual(sum(len(cloud.points) for cloud in filtered), 42)

    def test_support_weighted_section_fusion_keeps_real_layers_without_averaging(self) -> None:
        horizontal = np.linspace(0.0, 1.0, 61)
        base = np.column_stack(
            (
                np.zeros_like(horizontal),
                horizontal,
                np.zeros_like(horizontal),
            )
        )
        matching = base.copy()
        matching[:, 2] = 0.002
        unique = base.copy()
        unique[:, 2] = 0.30
        clouds = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            for points in (base, matching, unique)
        ]

        fused, report = _support_weighted_section_fusion(
            clouds,
            [0, 1, 2],
            voxel_size=0.01,
            section_config=SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.50,
                use_statistical_filter=False,
                use_component_filter=False,
            ),
        )

        fused_z = np.asarray(fused.points)[:, 2]
        self.assertFalse(np.any(np.isclose(fused_z, 0.15, atol=0.01)))
        self.assertTrue(np.any(np.isclose(fused_z, 0.30, atol=0.01)))
        confidence = report["fusion_confidence"]
        self.assertTrue(confidence["unique_coverage_retained"])
        self.assertGreater(confidence["section_voxel_multi_view_count"], 0)
        self.assertGreater(confidence["parallel_double_layer_isolated_bin_count"], 0)

    def test_parallel_layer_collapse_preserves_real_samples_and_separate_branches(self) -> None:
        horizontal = np.repeat(np.linspace(0, 1, 201), 6)
        vertical = np.tile(np.array([-0.01, 0, 0.01, 0.29, 0.30, 0.31]), 201)
        points = np.column_stack((np.zeros_like(horizontal), horizontal, vertical))
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        colors = np.random.default_rng(12).random(points.shape)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        collapsed, sources, supports, report = _collapse_parallel_section_layers(
            cloud,
            np.arange(len(points)),
            np.ones(len(points), dtype=np.int64),
            SectionConfig(
                angle_degrees=0.0,
                tilt_degrees=0.0,
                plane_offset=0.0,
                thickness=0.50,
                use_statistical_filter=False,
                use_component_filter=False,
            ),
        )

        collapsed_points = np.asarray(collapsed.points)
        self.assertTrue(report["parallel_layer_collapse_applied"])
        self.assertLess(len(collapsed_points), len(points) // 3)
        self.assertGreater(np.count_nonzero(collapsed_points[:, 2] > 0.2), 20)
        self.assertGreater(np.count_nonzero(collapsed_points[:, 2] < 0.1), 20)
        np.testing.assert_array_equal(collapsed_points, points[sources])
        np.testing.assert_array_equal(np.asarray(collapsed.colors), colors[sources])
        self.assertEqual(len(sources), len(supports))

    def test_display_contour_mask_thins_close_layers_but_keeps_far_branches(self) -> None:
        horizontal = np.repeat(np.linspace(0.0, 1.0, 21), 6)
        vertical = np.tile(np.array([0.00, 0.01, 0.02, 0.20, 0.21, 1.00]), 21)
        mask = _section_display_contour_mask(horizontal, vertical)

        self.assertLess(int(np.count_nonzero(mask)), len(mask))
        self.assertGreater(
            int(np.count_nonzero(mask & (vertical > 0.9))),
            0,
        )
        self.assertGreater(
            int(np.count_nonzero(mask & (vertical < 0.05))),
            0,
        )

    def test_representative_voxel_fusion_never_emits_coordinate_mean(self) -> None:
        first = np.array(
            [[0.000, 0.0, 0.0], [1.000, 0.0, 0.0]],
            dtype=np.float64,
        )
        second = np.array(
            [[0.020, 0.0, 0.0], [1.020, 0.0, 0.0]],
            dtype=np.float64,
        )
        clouds = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            for points in (first, second)
        ]

        fused, report = _representative_voxel_fusion(
            clouds,
            [np.eye(4), np.eye(4)],
            voxel_size=0.05,
            backend="cpu",
            source_indices=[0, 1],
            clouds_are_aligned=True,
        )

        fused_points = np.asarray(fused.points)
        self.assertFalse(np.any(np.isclose(fused_points[:, 0], 0.010, atol=1e-9)))
        self.assertFalse(np.any(np.isclose(fused_points[:, 0], 1.010, atol=1e-9)))
        self.assertTrue(report["coordinates_averaged"] is False)
        self.assertEqual(report["multi_view_voxel_count"], 2)
        self.assertTrue(report["unique_coverage_retained"])

    def test_representative_voxel_downsample_keeps_complementary_points(self) -> None:
        points = np.array(
            [
                [0.001, 0.0, 0.0],
                [0.009, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [1.00, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        sampled = _representative_voxel_down_sample_cloud(cloud, 0.02, "cpu")

        sampled_points = np.asarray(sampled.points)
        self.assertEqual(len(sampled_points), 3)
        self.assertTrue(np.any(np.isclose(sampled_points[:, 0], 0.001, atol=1e-9)))
        self.assertTrue(np.any(np.isclose(sampled_points[:, 0], 0.50, atol=1e-9)))
        self.assertTrue(np.any(np.isclose(sampled_points[:, 0], 1.00, atol=1e-9)))

    def test_calibration_roi_exclusion_uses_inverse_alignment_transform(self) -> None:
        points = np.array(
            [
                [5.0, 5.0, 5.0],
                [20.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points + [1.0, 0.0, 0.0]))
        transformation = np.eye(4, dtype=np.float64)
        transformation[0, 3] = 1.0
        filtered = remove_calibration_region(
            cloud,
            CalibrationConfig(
                bounds=Bounds3D(4.0, 17.5, 2.0, 17.0, -6.5, 8.0),
                exclusion_margin=0.0,
            ),
            backend="cpu",
            coordinate_transformation=transformation,
        )

        filtered_points = np.asarray(filtered.points)
        self.assertEqual(len(filtered_points), 1)
        np.testing.assert_allclose(filtered_points[0], [21.0, 5.0, 5.0])

    def test_section_cleanup_reverts_when_filter_breaks_measured_structure(self) -> None:
        horizontal = np.linspace(-2.0, 2.0, 81)
        points = np.column_stack(
            (
                np.zeros_like(horizontal),
                horizontal,
                np.sin(horizontal),
            )
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        config = SectionConfig(
            plane_offset=0.0,
            thickness=0.15,
            horizontal_min=-3.0,
            horizontal_max=3.0,
            vertical_min=-2.0,
            vertical_max=2.0,
            use_statistical_filter=False,
            use_component_filter=True,
            component_cell_size=0.10,
            component_min_points=3,
        )
        broken_mask = np.ones(len(points), dtype=bool)
        broken_mask[30:52] = False
        with patch(
            "pointcloud_app.pipeline._section_connected_component_mask",
            return_value=broken_mask,
        ):
            cleaned = clean_vertical_section(cloud, config)

        np.testing.assert_array_equal(np.asarray(cleaned.points), points)

    def test_coverage_first_fusion_keeps_complementary_section_extent(self) -> None:
        """A view that adds a unique side of the target must survive fusion."""
        first_horizontal = np.linspace(-1.0, 1.0, 101)
        second_horizontal = np.linspace(0.5, 2.5, 101)
        first_points = np.column_stack(
            (
                np.zeros_like(first_horizontal),
                first_horizontal,
                np.zeros_like(first_horizontal),
            )
        )
        second_points = np.column_stack(
            (
                np.zeros_like(second_horizontal),
                second_horizontal,
                np.zeros_like(second_horizontal),
            )
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / "first.ply", root / "complementary.ply"]
            for path, points in zip(paths, (first_points, second_points), strict=True):
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                )
            section_config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.10,
                horizontal_min=-2.0,
                horizontal_max=3.0,
                vertical_min=-1.0,
                vertical_max=1.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )
            merged, report = _merge_clouds_with_section_consensus(
                paths,
                [np.eye(4), np.eye(4)],
                [True, True],
                np.eye(4),
                section_config,
                voxel_size=0.05,
                backend="cpu",
                coverage_preserving=True,
            )

        merged_points = np.asarray(merged.points)
        self.assertEqual(report["source_views"], 2)
        self.assertEqual(report["removed_section_points"], 0)
        self.assertFalse(report["filter_applied"])
        self.assertTrue(report["diagnostic_only"])
        self.assertGreater(float(np.max(merged_points[:, 1])), 2.45)
        self.assertTrue(np.any(np.isclose(merged_points[:, 1], 2.5, atol=0.02)))

    def test_fusion_reuses_cleaned_per_view_clouds(self) -> None:
        """Final fusion must not re-introduce raw points removed before registration."""
        clean_points = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float64,
        )
        raw_points = np.vstack(
            [clean_points, [[50.0, 50.0, 50.0], [60.0, -60.0, 40.0]]]
        )

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / "first.ply", root / "second.ply"]
            for path in paths:
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_points)),
                )
            cleaned = [
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clean_points)),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clean_points)),
            ]
            section_config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.10,
                horizontal_min=-1.0,
                horizontal_max=1.0,
                vertical_min=-1.0,
                vertical_max=1.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )
            merged, report = _merge_clouds_with_section_consensus(
                paths,
                [np.eye(4), np.eye(4)],
                [True, True],
                np.eye(4),
                section_config,
                voxel_size=0.05,
                backend="cpu",
                coverage_preserving=True,
                source_clouds=cleaned,
            )

        merged_points = np.asarray(merged.points)
        self.assertTrue(report["fusion_input_denoised"])
        self.assertEqual(report["fusion_input"], "per_view_denoised_clouds")
        self.assertLess(float(np.max(np.abs(merged_points))), 1.0)

    def test_matching_sections_do_not_claim_fusion_cleanup_is_lossless(self) -> None:
        horizontal = np.linspace(-2.0, 2.0, 81)
        points = np.column_stack(
            (np.zeros_like(horizontal), horizontal, np.sin(horizontal))
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        diagnostic = diagnose_section_integrity(
            cloud, cloud, SectionConfig(component_cell_size=0.10)
        )

        self.assertEqual(diagnostic["status"], "stable_after_cleanup")
        self.assertFalse(diagnostic["structural_change_detected"])
        self.assertIn("不能据此单独判断融合云清理", diagnostic["message"])

    def test_section_integrity_marks_existing_gap_as_source_gap(self) -> None:
        horizontal = np.linspace(-2.0, 2.0, 81)
        points = np.column_stack(
            (
                np.zeros_like(horizontal),
                horizontal,
                np.sin(horizontal),
            )
        )
        source_gap = points[np.abs(horizontal) > 0.45]
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.15,
            component_cell_size=0.10,
            use_statistical_filter=False,
            use_component_filter=False,
        )

        diagnostic = diagnose_section_integrity(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_gap)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_gap.copy())),
            config,
            source_profiles={
                0: np.column_stack((source_gap[:, 1], source_gap[:, 2])),
                1: np.column_stack((source_gap[:, 1], source_gap[:, 2])),
                2: np.column_stack((source_gap[:, 1], source_gap[:, 2])),
            },
            source_profile_indices=[0, 1, 2],
        )

        self.assertEqual(diagnostic["status"], "source_coverage_gap_likely")
        self.assertEqual(diagnostic["raw_component_count"], diagnostic["cleaned_component_count"])
        self.assertFalse(diagnostic["structural_change_detected"])
        self.assertTrue(diagnostic["source_profile_gap_evidence"])
        self.assertEqual(diagnostic["source_profile_count"], 3)
        self.assertFalse(diagnostic["interpolation_applied"])
        self.assertFalse(diagnostic["cad_feedback_applied"])

    def test_section_integrity_detects_gap_created_by_cleanup(self) -> None:
        horizontal = np.linspace(-2.0, 2.0, 81)
        raw_points = np.column_stack(
            (
                np.zeros_like(horizontal),
                horizontal,
                np.sin(horizontal),
            )
        )
        cleaned_points = raw_points[np.abs(horizontal) > 0.45]
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.15,
            component_cell_size=0.10,
            use_statistical_filter=False,
            use_component_filter=False,
        )

        diagnostic = diagnose_section_integrity(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_points)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cleaned_points)),
            config,
        )

        self.assertEqual(diagnostic["status"], "cleanup_induced_gap")
        self.assertGreater(diagnostic["cleaned_component_count"], diagnostic["raw_component_count"])
        self.assertTrue(diagnostic["structural_change_detected"])

    def test_section_cleanup_does_not_mutate_source_coordinates(self) -> None:
        horizontal = np.linspace(-1.0, 1.0, 41)
        points = np.column_stack(
            (
                np.zeros_like(horizontal),
                horizontal,
                np.cos(horizontal),
            )
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        original_points = np.asarray(cloud.points).copy()
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.15,
            use_statistical_filter=False,
            use_component_filter=True,
            component_cell_size=0.10,
            component_min_points=3,
        )

        clean_vertical_section(cloud, config)

        np.testing.assert_array_equal(np.asarray(cloud.points), original_points)

    def test_section_validation_excludes_peer_cluster_without_parallel_tracks(self) -> None:
        horizontal = np.linspace(0.0, 3.9, 160)

        def multi_track_cloud(peer: bool) -> np.ndarray:
            point_sets = []
            for plane_offset in (-0.5, 0.0, 0.5):
                if peer:
                    wobble = 0.13 * np.sin(2.7 * horizontal)
                    curves = (
                        np.sin(horizontal) + 0.15 + wobble,
                        np.sin(horizontal) + 0.45 - wobble,
                    )
                else:
                    curves = (
                        np.sin(horizontal),
                        np.sin(horizontal) + 0.30,
                    )
                for curve in curves:
                    point_sets.append(
                        np.column_stack(
                            (
                                np.full(len(horizontal), plane_offset),
                                horizontal,
                                curve,
                            )
                        )
                    )
            return np.concatenate(point_sets, axis=0)

        metrics, report = self._validate_multi_probe_clouds(
            [
                multi_track_cloud(False),
                multi_track_cloud(False),
                multi_track_cloud(True),
                multi_track_cloud(True),
            ]
        )

        self.assertFalse(metrics[2].included_in_fusion)
        self.assertFalse(metrics[3].included_in_fusion)
        self.assertEqual(metrics[2].exclusion_reason, "section_peer_component")
        self.assertEqual(metrics[3].exclusion_reason, "section_peer_component")
        self.assertEqual(report["isolated_peer_components"][0]["component_indices"], [2, 3])
        self.assertTrue(report["pair_metrics"]["0:2"]["persistent_mismatch_evidence"])
        self.assertFalse(report["pair_metrics"]["0:2"]["persistent_parallel_duplicate_evidence"])
        self.assertEqual(report["section_consensus"]["final_active_indices"], [0, 1])

    def test_section_validation_keeps_mismatched_peer_cluster_with_local_coverage(self) -> None:
        horizontal = np.linspace(0.0, 3.9, 160)

        def multi_track_cloud(peer: bool) -> np.ndarray:
            point_sets = []
            for plane_offset in (-0.5, 0.0, 0.5):
                if peer:
                    covered_horizontal = horizontal[horizontal <= 2.5]
                    wobble = 0.13 * np.sin(2.7 * covered_horizontal)
                    curves = (
                        np.sin(covered_horizontal) + 0.15 + wobble,
                        np.sin(covered_horizontal) + 0.45 - wobble,
                    )
                else:
                    covered_horizontal = horizontal
                    curves = (
                        np.sin(covered_horizontal),
                        np.sin(covered_horizontal) + 0.30,
                    )
                for curve in curves:
                    point_sets.append(
                        np.column_stack(
                            (
                                np.full(len(covered_horizontal), plane_offset),
                                covered_horizontal,
                                curve,
                            )
                        )
                    )
            return np.concatenate(point_sets, axis=0)

        metrics, report = self._validate_multi_probe_clouds(
            [
                multi_track_cloud(False),
                multi_track_cloud(False),
                multi_track_cloud(True),
                multi_track_cloud(True),
            ]
        )

        self.assertTrue(metrics[2].included_in_fusion)
        self.assertTrue(metrics[3].included_in_fusion)
        self.assertEqual(metrics[2].section_validation_status, "peer_supported_coverage")
        self.assertEqual(metrics[3].section_validation_status, "peer_supported_coverage")
        self.assertEqual(report["isolated_peer_components"], [])
        self.assertEqual(
            report["protected_peer_components"][0]["component_indices"],
            [2, 3],
        )
        self.assertTrue(
            report["protected_peer_components"][0]["local_coverage_protected"]
        )
        self.assertEqual(report["section_consensus"]["final_active_indices"], [0, 1, 2, 3])

    def test_pose_graph_uses_source_to_target_edge_direction(self) -> None:
        def translated_pose(x_value: float) -> np.ndarray:
            transformation = np.eye(4, dtype=np.float64)
            transformation[0, 3] = x_value
            return transformation

        expected = [translated_pose(0.0), translated_pose(1.0), translated_pose(2.0)]
        information = np.eye(6, dtype=np.float64) * 100.0
        edges = [
            o3d.pipelines.registration.PoseGraphEdge(
                source_index,
                target_index,
                np.linalg.inv(expected[target_index]) @ expected[source_index],
                information,
                uncertain,
            )
            for source_index, target_index, uncertain in (
                (0, 1, False),
                (1, 2, False),
                (0, 2, True),
            )
        ]

        optimized, corrections = optimize_multiframe_pose_graph(
            [translated_pose(0.0), translated_pose(0.8), translated_pose(2.2)],
            edges,
            maximum_correspondence_distance=0.20,
            maximum_rotation_degrees=1.0,
            maximum_translation=0.30,
        )

        np.testing.assert_allclose([item[0, 3] for item in optimized], [0.0, 1.0, 2.0], atol=1e-6)
        self.assertTrue(all(bool(item["accepted"]) for item in corrections))

    def test_six_station_prior_restores_timestamp_order_and_returns_copies(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chronological_names = [
                'scan_minus50.ply',
                'scan_minus28.ply',
                'scan_minus10.ply',
                'scan_reference.ply',
                'scan_plus37.ply',
                'scan_plus47.ply',
            ]
            files: list[Path] = []
            for index, name in enumerate(chronological_names):
                path = root / name
                path.touch()
                timestamp = 1_700_000_000 + index
                os.utime(path, (timestamp, timestamp))
                files.append(path)

            ordered_files, labels = order_files_with_six_station_pose_prior(list(reversed(files)))
            transformations = six_station_pose_transformations(labels)

            self.assertEqual(labels, ['reference', 'minus10', 'minus28', 'plus37', 'plus47', 'minus50'])
            self.assertEqual([path.name for path in ordered_files], [
                'scan_reference.ply',
                'scan_minus10.ply',
                'scan_minus28.ply',
                'scan_plus37.ply',
                'scan_plus47.ply',
                'scan_minus50.ply',
            ])
            np.testing.assert_allclose(transformations[0], np.eye(4))
            transformations[0][0, 0] = 0.0
            self.assertEqual(six_station_pose_transformations(['reference'])[0][0, 0], 1.0)

    def test_five_station_prior_omits_plus47_and_keeps_plus37(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chronological_names = [
                'scan_minus50.ply',
                'scan_minus28.ply',
                'scan_minus10.ply',
                'scan_reference.ply',
                'scan_plus37.ply',
            ]
            files: list[Path] = []
            for index, name in enumerate(chronological_names):
                path = root / name
                path.touch()
                timestamp = 1_700_100_000 + index
                os.utime(path, (timestamp, timestamp))
                files.append(path)

            ordered_files, labels = order_files_with_six_station_pose_prior(list(reversed(files)))

            self.assertEqual(labels, ['reference', 'minus10', 'minus28', 'plus37', 'minus50'])
            self.assertEqual(
                [path.name for path in ordered_files],
                [
                    'scan_reference.ply',
                    'scan_minus10.ply',
                    'scan_minus28.ply',
                    'scan_plus37.ply',
                    'scan_minus50.ply',
                ],
            )

    def test_station_order_metadata_takes_precedence_over_file_times(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            files = [root / f"capture_{index}.ply" for index in range(6)]
            for index, path in enumerate(files):
                path.touch()
                timestamp = 1_700_200_000 + (len(files) - index)
                os.utime(path, (timestamp, timestamp))
            (root / "station_order.json").write_text(
                json.dumps(
                    {
                        "views": [
                            {"file": "capture_3.ply", "station": "reference"},
                            {"file": "capture_2.ply", "station": "minus10"},
                            {"file": "capture_1.ply", "station": "minus28"},
                            {"file": "capture_4.ply", "station": "plus37"},
                            {"file": "capture_5.ply", "station": "plus47"},
                            {"file": "capture_0.ply", "station": "minus50"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            ordered_files, labels = order_files_with_six_station_pose_prior(list(reversed(files)))

            self.assertEqual(labels, ["reference", "minus10", "minus28", "plus37", "plus47", "minus50"])
            self.assertEqual(
                [path.name for path in ordered_files],
                [
                    "capture_3.ply",
                    "capture_2.ply",
                    "capture_1.ply",
                    "capture_4.ply",
                    "capture_5.ply",
                    "capture_0.ply",
                ],
            )

    def test_station_pose_prior_applies_user_translation_correction(self) -> None:
        base = six_station_pose_transformations(["plus37"])[0]
        corrected = six_station_pose_transformations(
            ["plus37"],
            {
                "plus37": {
                    "translation_x": 0.12,
                    "translation_y": -0.08,
                    "translation_z": 0.03,
                }
            },
        )[0]

        np.testing.assert_allclose(corrected[:3, :3], base[:3, :3], atol=1e-12)
        np.testing.assert_allclose(corrected[:3, 3] - base[:3, 3], [0.12, -0.08, 0.03], atol=1e-12)

    def test_low_evidence_section_keeps_station_pose_prior(self) -> None:
        section_config = SectionConfig(
            plane_offset=0.0,
            thickness=0.20,
            horizontal_min=-10.0,
            horizontal_max=10.0,
            vertical_min=-10.0,
            vertical_max=10.0,
            use_statistical_filter=False,
            use_component_filter=False,
        )
        profile = np.column_stack(
            (
                np.zeros(20, dtype=np.float64),
                np.linspace(-1.0, 1.0, 20),
                np.zeros(20, dtype=np.float64),
            )
        )
        shifted_profile = profile.copy()
        shifted_profile[:, 1] += 2.0
        clouds = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(profile)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(profile)),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(shifted_profile)),
        ]

        def translated_pose(x_value: float) -> np.ndarray:
            transformation = np.eye(4, dtype=np.float64)
            transformation[0, 3] = x_value
            return transformation

        station_prior = [translated_pose(0.0), translated_pose(0.0), translated_pose(0.0)]
        section_refined = [translated_pose(0.0), translated_pose(0.0), translated_pose(0.5)]
        pose_graph = [translated_pose(0.0), translated_pose(0.0), translated_pose(0.5)]
        registered_calibration = [translated_pose(0.0), translated_pose(0.0), translated_pose(0.5)]
        metrics = [
            RegistrationMetric(
                file_name=f"{label}.ply",
                calibration_points=100,
                global_fitness=None,
                global_rmse=None,
                icp_fitness=1.0,
                icp_rmse=0.0,
                transformation=np.eye(4).tolist(),
                full_cloud_fitness=1.0,
                included_in_fusion=True,
                pose_prior_label=label,
            )
            for label in ("reference", "minus10", "plus37")
        ]

        selected, diagnostics = select_section_consistent_pose_candidates(
            clouds,
            [pose.copy() for pose in station_prior],
            {
                "section_refined": section_refined,
                "pose_graph": pose_graph,
                "registered_calibration": registered_calibration,
                "station_pose_prior": station_prior,
            },
            ["reference", "minus10", "plus37"],
            metrics,
            np.eye(4, dtype=np.float64),
            section_config,
            RegistrationConfig(
                section_alignment_overlap_distance=0.35,
                section_alignment_max_p90_distance=0.15,
                section_alignment_minimum_overlap_points=500,
            ),
            "cpu",
        )

        np.testing.assert_allclose(selected[2], station_prior[2])
        self.assertEqual(diagnostics[0]["selected_candidate"], "station_pose_prior")
        self.assertEqual(
            diagnostics[0]["selection_reason"],
            "station_pose_prior_low_section_evidence",
        )

    def test_registration_config_validates_user_adjustable_pose_graph_parameters(self) -> None:
        config = RegistrationConfig(
            calibration_graph_voxel_size=0.10,
            calibration_graph_max_correspondence_distance=0.32,
            calibration_graph_icp_max_iterations=260,
            workpiece_graph_voxel_size=0.08,
            workpiece_graph_max_correspondence_distance=0.24,
            workpiece_graph_icp_max_iterations=320,
            section_alignment_thickness=0.04,
            section_alignment_overlap_distance=0.30,
            section_alignment_max_p90_distance=0.12,
            section_alignment_probe_count=7,
            section_alignment_probe_spacing=0.40,
        )

        config.validate()
        self.assertEqual(config.calibration_graph_icp_max_iterations, 260)
        self.assertEqual(config.workpiece_graph_icp_max_iterations, 320)
        self.assertEqual(config.section_alignment_probe_count, 7)

    def test_calibration_quality_gate_checks_fitness_and_rmse(self) -> None:
        config = RegistrationConfig(
            minimum_calibration_icp_fitness=0.65,
            maximum_calibration_icp_rmse=0.16,
        )
        calibration = CalibrationConfig(minimum_points=800)

        passed, reason = calibration_registration_quality(900, 0.70, 0.10, calibration, config)
        self.assertTrue(passed)
        self.assertIsNone(reason)

        passed, reason = calibration_registration_quality(900, 0.64, 0.10, calibration, config)
        self.assertFalse(passed)
        self.assertIn("fitness", reason or "")

        passed, reason = calibration_registration_quality(900, 0.70, 0.17, calibration, config)
        self.assertFalse(passed)
        self.assertIn("RMSE", reason or "")

    def test_pose_graph_rejects_high_rmse_edge_without_pose_change(self) -> None:
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        config = RegistrationConfig(
            icp_max_correspondence_distance=0.40,
            calibration_graph_maximum_adjacent_rmse=0.05,
        )
        with patch(
            "pointcloud_app.pipeline.o3d.pipelines.registration.evaluate_registration",
            return_value=type("Result", (), {"fitness": 0.90, "inlier_rmse": 0.02})(),
        ), patch(
            "pointcloud_app.pipeline.refine_with_icp",
            return_value=(np.eye(4), 0.90, 0.08),
        ):
            edge, metric = _pairwise_pose_graph_edge(
                source,
                target,
                np.eye(4),
                np.eye(4),
                config,
                "calibration",
                0,
                1,
                "reference",
                "minus10",
                "adjacent",
                False,
                2.0,
                1.0,
                0.30,
                "cpu",
                0.05,
            )
        self.assertIsNone(edge)
        self.assertFalse(metric.accepted)
        self.assertIn("RMSE", metric.rejection_reason or "")

    def test_pose_graph_uses_only_accepted_edges_and_keeps_rejected_nodes_at_prior(self) -> None:
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        config = RegistrationConfig(icp_max_correspondence_distance=0.40)
        with patch(
            "pointcloud_app.pipeline.o3d.pipelines.registration.evaluate_registration",
            return_value=type("Result", (), {"fitness": 0.90, "inlier_rmse": 0.02})(),
        ), patch(
            "pointcloud_app.pipeline.refine_with_icp",
            return_value=(np.eye(4), 0.20, 0.08),
        ):
            corrections, metrics, nodes = build_and_optimize_pairwise_pose_graph(
                [source, target],
                ["reference", "minus10"],
                config,
                "calibration",
                maximum_rotation_degrees=2.0,
                maximum_translation=1.0,
                minimum_adjacent_fitness=0.30,
                minimum_loop_fitness=0.30,
                maximum_adjacent_rmse=0.05,
                maximum_loop_rmse=0.05,
            )
        self.assertEqual(len(metrics), 1)
        self.assertFalse(metrics[0].accepted)
        self.assertEqual(len(nodes), 2)
        np.testing.assert_allclose(corrections[0], np.eye(4))
        np.testing.assert_allclose(corrections[1], np.eye(4))

    def test_translation_only_refinement_does_not_change_rotation(self) -> None:
        random_generator = np.random.default_rng(42)
        source_points = random_generator.normal(size=(600, 3)).astype(np.float64)
        expected_translation = np.array([1.25, -0.70, 0.35], dtype=np.float64)
        target_points = source_points + expected_translation
        source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
        initial_transformation = np.eye(4, dtype=np.float64)
        initial_transformation[:3, 3] = np.array([1.0, -0.5, 0.2], dtype=np.float64)

        transformation, fitness, rmse = refine_translation_only(
            source,
            target,
            initial_transformation,
            maximum_correspondence_distance=0.60,
        )

        np.testing.assert_allclose(transformation[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(transformation[:3, 3], expected_translation, atol=1e-3)
        self.assertGreater(fitness, 0.99)
        self.assertLess(rmse, 1e-3)

    def test_section_validation_excludes_misaligned_view(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            profile = np.array(
                [[0.0, float(index), float(index) * 0.1] for index in range(12)],
                dtype=np.float64,
            )
            shifted_profile = profile.copy()
            shifted_profile[:, 1] += 0.20
            paths = [root / name for name in ("reference.ply", "matching.ply", "shifted.ply")]
            for path, points in zip(paths, (profile, profile, shifted_profile)):
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                )

            metrics = [
                RegistrationMetric(
                    file_name=path.name,
                    calibration_points=0,
                    global_fitness=None,
                    global_rmse=None,
                    icp_fitness=1.0,
                    icp_rmse=0.0,
                    transformation=np.eye(4).tolist(),
                    full_cloud_fitness=1.0,
                    full_cloud_rmse=0.0,
                )
                for path in paths
            ]
            section_config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.05,
                horizontal_min=-20.0,
                horizontal_max=20.0,
                vertical_min=-10.0,
                vertical_max=10.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )
            registration_config = RegistrationConfig(
                section_alignment_thickness=0.05,
                section_alignment_overlap_distance=0.35,
                section_alignment_max_p90_distance=0.10,
                section_alignment_minimum_overlap_points=8,
                section_alignment_minimum_views=2,
                coverage_preserving_fusion=False,
            )

            validate_section_view_alignment(
                paths,
                [np.eye(4) for _ in paths],
                metrics,
                np.eye(4),
                section_config,
                registration_config,
                "cpu",
            )

            self.assertTrue(metrics[0].included_in_fusion)
            self.assertTrue(metrics[1].included_in_fusion)
            self.assertFalse(metrics[2].included_in_fusion)
            self.assertEqual(metrics[2].exclusion_reason, "section_alignment")
            self.assertFalse(metrics[2].section_alignment_passed)
            self.assertGreater(metrics[2].section_p90_distance or 0.0, 0.10)

    def test_section_validation_compares_matching_probe_offsets_only(self) -> None:
        layers = [(-0.50, 0.00, 3.9), (0.00, 0.40, 3.9), (0.50, 0.80, 3.9)]
        metrics, report = self._validate_multi_probe_clouds(
            [
                self._multi_probe_cloud(layers),
                self._multi_probe_cloud(layers),
                self._multi_probe_cloud(layers),
            ]
        )

        self.assertTrue(all(metric.included_in_fusion for metric in metrics))
        self.assertEqual(report["probe_offsets"], [-0.5, 0.0, 0.5])
        self.assertEqual(report["center_probe_index"], 1)
        self.assertEqual(len(report["profiles"][0]), 40)
        self.assertEqual(len(report["probe_metrics"]["0:1"]), 3)
        pair_quality = report["pair_metrics"]["0:1"]
        self.assertEqual(pair_quality["valid_probe_count"], 3)
        self.assertEqual(pair_quality["abnormal_evidence_count"], 0)

    def test_coverage_first_section_validation_keeps_complementary_view(self) -> None:
        matching_layers = [(-0.50, 0.00, 3.9), (0.00, 0.40, 3.9), (0.50, 0.80, 3.9)]
        shifted_layers = [(-0.50, 0.25, 3.9), (0.00, 0.65, 3.9), (0.50, 1.05, 3.9)]
        metrics, report = self._validate_multi_probe_clouds(
            [
                self._multi_probe_cloud(matching_layers),
                self._multi_probe_cloud(matching_layers),
                self._multi_probe_cloud(shifted_layers),
            ],
            coverage_preserving_fusion=True,
        )

        self.assertTrue(metrics[2].included_in_fusion)
        self.assertIsNone(metrics[2].exclusion_reason)
        self.assertFalse(metrics[2].section_alignment_passed)
        self.assertEqual(metrics[2].section_validation_status, "mismatch")
        self.assertFalse(report["fusion_filter_applied"])
        self.assertTrue(report["coverage_preserving_fusion"])
        self.assertEqual(report["section_consensus"]["excluded_indices"], [])
        self.assertIn(2, report["section_consensus"]["diagnostic_excluded_indices"])

    def test_section_validation_excludes_persistent_multi_probe_mismatch(self) -> None:
        matching_layers = [(-0.50, 0.00, 3.9), (0.00, 0.40, 3.9), (0.50, 0.80, 3.9)]
        shifted_layers = [(-0.50, 0.25, 3.9), (0.00, 0.65, 3.9), (0.50, 1.05, 3.9)]
        metrics, report = self._validate_multi_probe_clouds(
            [
                self._multi_probe_cloud(matching_layers),
                self._multi_probe_cloud(matching_layers),
                self._multi_probe_cloud(shifted_layers),
            ]
        )

        self.assertFalse(metrics[2].included_in_fusion)
        self.assertFalse(metrics[2].section_alignment_passed)
        self.assertEqual(metrics[2].section_validation_status, "mismatch")
        self.assertEqual(metrics[2].section_valid_probe_count, 3)
        self.assertEqual(metrics[2].section_mismatch_probe_count, 3)
        self.assertEqual(metrics[2].section_abnormal_evidence_count, 3)
        self.assertEqual(
            report["views"][2]["mismatch_probe_offsets"],
            [-0.5, 0.0, 0.5],
        )

    def test_section_validation_excludes_persistent_extra_contour(self) -> None:
        matching_layers = [(-0.50, 0.00, 3.9), (0.00, 0.40, 3.9), (0.50, 0.80, 3.9)]
        extra_layers = matching_layers + [
            (plane_offset, vertical_offset + 0.25, coverage_end)
            for plane_offset, vertical_offset, coverage_end in matching_layers
        ]
        metrics, report = self._validate_multi_probe_clouds(
            [
                self._multi_probe_cloud(matching_layers, sample_count=160),
                self._multi_probe_cloud(matching_layers, sample_count=160),
                self._multi_probe_cloud(extra_layers, sample_count=160),
            ],
            maximum_p90_distance=0.40,
        )

        self.assertFalse(metrics[2].included_in_fusion)
        self.assertFalse(metrics[2].section_alignment_passed)
        self.assertEqual(metrics[2].section_validation_status, "mismatch")
        self.assertEqual(metrics[2].exclusion_reason, "section_extra_contour")
        self.assertEqual(metrics[2].section_extra_contour_probe_count, 3)
        self.assertEqual(metrics[2].section_abnormal_evidence_count, 3)
        self.assertEqual(
            report["views"][2]["extra_contour_probe_offsets"],
            [-0.5, 0.0, 0.5],
        )

    def test_section_validation_keeps_mutually_supported_coverage_cluster(self) -> None:
        matching_layers = [(-0.50, 0.00, 3.9), (0.00, 0.40, 3.9), (0.50, 0.80, 3.9)]
        peer_supported_layers = [
            (plane_offset, vertical_offset + 0.45, coverage_end)
            for plane_offset, vertical_offset, coverage_end in matching_layers
        ]
        metrics, report = self._validate_multi_probe_clouds(
            [
                self._multi_probe_cloud(matching_layers),
                self._multi_probe_cloud(matching_layers),
                self._multi_probe_cloud(peer_supported_layers),
                self._multi_probe_cloud(peer_supported_layers),
            ]
        )

        self.assertTrue(metrics[2].included_in_fusion)
        self.assertTrue(metrics[3].included_in_fusion)
        self.assertTrue(metrics[2].section_alignment_passed)
        self.assertTrue(metrics[3].section_alignment_passed)
        self.assertEqual(metrics[2].section_validation_status, "peer_supported_coverage")
        self.assertEqual(metrics[3].section_validation_status, "peer_supported_coverage")
        self.assertEqual(report["views"][2]["peer_coverage_indices"], [2, 3])
        self.assertEqual(report["views"][3]["peer_coverage_indices"], [2, 3])

    def test_section_validation_excludes_isolated_peer_cluster_with_wrong_contour(self) -> None:
        horizontal = np.linspace(0.0, 3.9, 160)

        def multi_track_cloud(offsets: tuple[float, float]) -> np.ndarray:
            point_sets = []
            for plane_offset in (-0.5, 0.0, 0.5):
                for vertical_offset in offsets:
                    point_sets.append(
                        np.column_stack(
                            (
                                np.full(len(horizontal), plane_offset),
                                horizontal,
                                np.sin(horizontal) + vertical_offset,
                            )
                        )
                    )
            return np.concatenate(point_sets, axis=0)

        metrics, report = self._validate_multi_probe_clouds(
            [
                multi_track_cloud((0.0, 0.30)),
                multi_track_cloud((0.0, 0.30)),
                multi_track_cloud((0.15, 0.45)),
                multi_track_cloud((0.15, 0.45)),
            ]
        )

        self.assertFalse(metrics[2].included_in_fusion)
        self.assertFalse(metrics[3].included_in_fusion)
        self.assertEqual(metrics[2].exclusion_reason, "section_peer_component")
        self.assertEqual(metrics[3].exclusion_reason, "section_peer_component")
        self.assertTrue(report["views"][2]["peer_component_rejected"])
        self.assertTrue(report["views"][3]["peer_component_rejected"])
        self.assertEqual(report["isolated_peer_components"][0]["component_indices"], [2, 3])
        self.assertEqual(report["section_consensus"]["final_active_indices"], [0, 1])

    def test_section_validation_keeps_missing_probe_coverage_view(self) -> None:
        all_layers = [(-0.50, 0.00, 3.9), (0.00, 0.40, 3.9), (0.50, 0.80, 3.9)]
        center_only_layer = [(0.00, 0.40, 3.9)]
        metrics, report = self._validate_multi_probe_clouds(
            [
                self._multi_probe_cloud(all_layers),
                self._multi_probe_cloud(all_layers),
                self._multi_probe_cloud(center_only_layer),
            ]
        )

        self.assertTrue(metrics[2].included_in_fusion)
        self.assertTrue(metrics[2].section_alignment_passed)
        self.assertEqual(metrics[2].section_validation_status, "pass")
        self.assertEqual(metrics[2].section_valid_probe_count, 1)
        self.assertEqual(metrics[2].section_mismatch_probe_count, 0)
        pair_quality = report["pair_metrics"]["0:2"]
        self.assertEqual(pair_quality["valid_probe_count"], 1)
        self.assertEqual(pair_quality["forward"]["coverage_only_probe_count"], 2)

    def test_section_validation_keeps_matching_partial_coverage_view(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_profile = np.array(
                [[0.0, float(index) * 0.10, float(index) * 0.01] for index in range(120)],
                dtype=np.float64,
            )
            partial_profile = reference_profile[:20].copy()
            paths = [root / name for name in ("reference.ply", "matching.ply", "partial.ply")]
            for path, points in zip(
                paths,
                (reference_profile, reference_profile, partial_profile),
                strict=True,
            ):
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                )

            metrics = [
                RegistrationMetric(
                    file_name=path.name,
                    calibration_points=0,
                    global_fitness=None,
                    global_rmse=None,
                    icp_fitness=1.0,
                    icp_rmse=0.0,
                    transformation=np.eye(4).tolist(),
                    pose_prior_label=label,
                )
                for path, label in zip(paths, ("reference", "minus10", "plus37"), strict=True)
            ]
            section_config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.05,
                horizontal_min=-20.0,
                horizontal_max=50.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )
            registration_config = RegistrationConfig(
                section_alignment_thickness=0.05,
                section_alignment_overlap_distance=0.35,
                section_alignment_max_p90_distance=0.10,
                section_alignment_minimum_overlap_points=8,
                section_alignment_probe_count=1,
                coverage_preserving_fusion=False,
            )

            validate_section_view_alignment(
                paths,
                [np.eye(4) for _ in paths],
                metrics,
                np.eye(4),
                section_config,
                registration_config,
                "cpu",
            )

            self.assertTrue(metrics[2].included_in_fusion)
            self.assertTrue(metrics[2].section_alignment_passed)
            self.assertEqual(metrics[2].section_validation_status, "pass")
            self.assertLess(metrics[2].section_shared_span or 0.0, 2.0)

    def test_section_validation_marks_disjoint_profile_as_coverage_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_profile = np.array(
                [[0.0, float(index) * 0.10, float(index) * 0.01] for index in range(120)],
                dtype=np.float64,
            )
            disjoint_profile = reference_profile.copy()
            disjoint_profile[:, 1] += 30.0
            paths = [root / name for name in ("reference.ply", "matching.ply", "disjoint.ply")]
            for path, points in zip(
                paths,
                (reference_profile, reference_profile, disjoint_profile),
                strict=True,
            ):
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                )

            metrics = [
                RegistrationMetric(
                    file_name=path.name,
                    calibration_points=0,
                    global_fitness=None,
                    global_rmse=None,
                    icp_fitness=1.0,
                    icp_rmse=0.0,
                    transformation=np.eye(4).tolist(),
                    pose_prior_label=label,
                )
                for path, label in zip(paths, ("reference", "minus10", "plus37"), strict=True)
            ]
            section_config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.05,
                horizontal_min=-20.0,
                horizontal_max=50.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )
            registration_config = RegistrationConfig(
                section_alignment_thickness=0.05,
                section_alignment_overlap_distance=0.35,
                section_alignment_max_p90_distance=0.10,
                section_alignment_minimum_overlap_points=8,
                section_alignment_probe_count=1,
                coverage_preserving_fusion=False,
            )

            validate_section_view_alignment(
                paths,
                [np.eye(4) for _ in paths],
                metrics,
                np.eye(4),
                section_config,
                registration_config,
                "cpu",
            )

            self.assertTrue(metrics[2].included_in_fusion)
            self.assertIsNone(metrics[2].section_alignment_passed)
            self.assertEqual(metrics[2].section_validation_status, "coverage_only")

    def test_section_alignment_quality_uses_only_overlapping_points(self) -> None:
        source = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        target = np.array(
            [[0.08, 0.0, 0.0], [1.08, 0.0, 0.0], [2.08, 0.0, 0.0]],
            dtype=np.float64,
        )

        quality = evaluate_section_alignment(source, target, overlap_distance=0.35)

        self.assertEqual(quality.section_points, 4)
        self.assertEqual(quality.overlap_points, 3)
        self.assertAlmostEqual(quality.median_distance or 0.0, 0.08, places=6)
        self.assertAlmostEqual(quality.p90_distance or 0.0, 0.08, places=6)

    def test_section_validation_rejects_view_with_insufficient_overlap(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matching_profile = np.array(
                [[0.0, float(index), float(index) * 0.1] for index in range(12)],
                dtype=np.float64,
            )
            low_overlap_profile = matching_profile.copy()
            low_overlap_profile[3:, 2] += 2.0
            paths = [root / f"view-{index}.ply" for index in range(4)]
            profiles = [matching_profile, matching_profile, matching_profile, low_overlap_profile]
            for path, points in zip(paths, profiles, strict=True):
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)),
                )
            metrics = [
                RegistrationMetric(
                    file_name=path.name,
                    calibration_points=0,
                    global_fitness=None,
                    global_rmse=None,
                    icp_fitness=1.0,
                    icp_rmse=0.0,
                    transformation=np.eye(4).tolist(),
                )
                for path in paths
            ]
            section_config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.05,
                horizontal_min=-20.0,
                horizontal_max=20.0,
                vertical_min=-10.0,
                vertical_max=10.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )
            registration_config = RegistrationConfig(
                section_alignment_thickness=0.05,
                section_alignment_overlap_distance=0.35,
                section_alignment_max_p90_distance=0.10,
                section_alignment_minimum_overlap_points=8,
                section_alignment_probe_count=1,
                coverage_preserving_fusion=False,
            )

            validate_section_view_alignment(
                paths,
                [np.eye(4) for _ in paths],
                metrics,
                np.eye(4),
                section_config,
                registration_config,
                "cpu",
            )

            self.assertFalse(metrics[3].included_in_fusion)
            self.assertFalse(metrics[3].section_alignment_passed)
            self.assertLess(metrics[3].section_overlap_points or 0, 8)

    def test_full_cloud_alignment_quality_prefers_lower_rmse_when_overlap_is_similar(self) -> None:
        original = full_cloud_alignment_quality(0.84, 0.31, 0.90)
        refined = full_cloud_alignment_quality(0.835, 0.29, 0.90)

        self.assertGreater(refined, original)

    def test_fine_full_cloud_registration_uses_tighter_alignment_parameters(self) -> None:
        config = RegistrationConfig(
            registration_voxel_size=0.35,
            icp_max_correspondence_distance=0.90,
            icp_max_iterations=100,
        )

        fine_config = fine_full_cloud_registration_config(config)

        self.assertEqual(fine_config.registration_voxel_size, 0.15)
        self.assertEqual(fine_config.icp_max_correspondence_distance, 0.45)
        self.assertEqual(fine_config.icp_max_iterations, 200)

    def test_reference_selection_falls_back_to_best_calibration_coverage(self) -> None:
        fallback_index = select_reference_index_by_calibration_quality(
            [
                None,
                (2_200, 700, 0.03),
                (9_400, 1_400, 0.04),
            ]
        )
        preferred_index = select_reference_index_by_calibration_quality(
            [
                (1_100, 600, 0.03),
                (9_400, 1_400, 0.04),
            ]
        )

        self.assertEqual(fallback_index, 2)
        self.assertEqual(preferred_index, 0)

    def test_full_resolution_section_keeps_original_points_after_registration(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            included_path = root / "included.ply"
            excluded_path = root / "excluded.ply"
            included_points = np.array(
                [[1.00, 0.0, 0.0], [1.01, 0.2, 0.1], [2.0, 0.0, 0.0]],
                dtype=np.float64,
            )
            excluded_points = np.array([[0.0, 5.0, 0.0]], dtype=np.float64)
            o3d.io.write_point_cloud(
                str(included_path),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(included_points)),
            )
            o3d.io.write_point_cloud(
                str(excluded_path),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(excluded_points)),
            )
            registration = np.eye(4)
            registration[0, 3] = -1.0
            config = SectionConfig(
                angle_degrees=0.0,
                plane_offset=0.0,
                thickness=0.10,
                horizontal_min=-10.0,
                horizontal_max=10.0,
                vertical_min=-10.0,
                vertical_max=10.0,
                use_statistical_filter=False,
                use_component_filter=False,
            )

            section, source_points = extract_full_resolution_section(
                [included_path, excluded_path],
                [registration, np.eye(4)],
                [True, False],
                np.eye(4),
                config,
                "cpu",
            )

            self.assertEqual(source_points, len(included_points))
            np.testing.assert_allclose(
                np.asarray(section.points),
                [[0.0, 0.0, 0.0], [0.01, 0.2, 0.1]],
                atol=1e-6,
            )

    def test_zero_degree_plane_uses_x_as_normal(self) -> None:
        normal, horizontal, vertical = vertical_plane_basis(0.0)
        np.testing.assert_allclose(normal, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(horizontal, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(vertical, [0.0, 0.0, 1.0])

        points = np.array(
            [
                [0.05, 2.0, 1.0],
                [0.16, 2.0, 1.0],
                [-0.08, 2.0, 1.0],
                [0.03, 50.0, 1.0],
            ]
        )
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.20,
            horizontal_min=-10.0,
            horizontal_max=10.0,
            vertical_min=-1.0,
            vertical_max=3.0,
        )
        np.testing.assert_array_equal(select_vertical_section_indices(points, config), [0, 2])

    def test_ninety_degree_plane_uses_y_as_normal(self) -> None:
        points = np.array(
            [
                [2.0, 1.02, 0.0],
                [2.0, 0.87, 0.0],
                [2.0, 1.20, 0.0],
            ]
        )
        config = SectionConfig(
            angle_degrees=90.0,
            plane_offset=1.0,
            thickness=0.30,
            horizontal_min=-10.0,
            horizontal_max=10.0,
            vertical_min=-1.0,
            vertical_max=1.0,
        )
        np.testing.assert_array_equal(select_vertical_section_indices(points, config), [0, 1])

    def test_tilted_plane_basis_uses_three_dimensional_normal(self) -> None:
        normal, horizontal, vertical = vertical_plane_basis(0.0, 30.0)

        np.testing.assert_allclose(normal, [np.sqrt(3.0) / 2.0, 0.0, 0.5], atol=1e-8)
        self.assertAlmostEqual(float(np.dot(normal, horizontal)), 0.0, places=8)
        self.assertAlmostEqual(float(np.dot(normal, vertical)), 0.0, places=8)
        self.assertAlmostEqual(float(np.dot(horizontal, vertical)), 0.0, places=8)
        self.assertGreater(vertical[2], 0.0)

    def test_two_top_view_points_define_a_vertical_plane(self) -> None:
        angle_degrees, plane_offset = vertical_plane_from_xy_points((2.0, 5.0), (8.0, 5.0))
        normal, horizontal, vertical = vertical_plane_basis(angle_degrees)

        self.assertAlmostEqual(abs(angle_degrees), 90.0, places=6)
        self.assertAlmostEqual(plane_offset, 5.0, places=6)
        self.assertAlmostEqual(float(np.dot(normal, [2.0, 5.0, 0.0])), plane_offset, places=6)
        self.assertAlmostEqual(float(np.dot(normal, [8.0, 5.0, 0.0])), plane_offset, places=6)
        self.assertAlmostEqual(abs(float(np.dot(horizontal, [6.0, 0.0, 0.0]))), 6.0, places=6)
        np.testing.assert_allclose(vertical, [0.0, 0.0, 1.0])

    def test_xyz_region_generates_normal_plane_for_tilted_principal_axis(self) -> None:
        center = np.array([4.0, -3.0, 2.0])
        principal_direction = np.array([0.8, 0.3, 0.45], dtype=np.float64)
        principal_direction /= np.linalg.norm(principal_direction)
        reference_up = np.array([0.0, 0.0, 1.0])
        vertical_direction = reference_up - np.dot(reference_up, principal_direction) * principal_direction
        vertical_direction /= np.linalg.norm(vertical_direction)
        lateral_direction = np.cross(vertical_direction, principal_direction)
        lateral_direction /= np.linalg.norm(lateral_direction)
        longitudinal_coordinates = np.repeat(np.linspace(-6.0, 6.0, 25), 5)
        lateral_coordinates = np.tile(np.linspace(-0.3, 0.3, 5), 25)
        xyz_coordinates = (
            center
            + np.outer(longitudinal_coordinates, principal_direction)
            + np.outer(lateral_coordinates, lateral_direction)
        )

        angle_degrees, tilt_degrees, plane_offset, directionality = normal_plane_from_xyz_region(xyz_coordinates)
        normal, horizontal, vertical = vertical_plane_basis(angle_degrees, tilt_degrees)

        self.assertGreater(directionality, 10.0)
        self.assertAlmostEqual(float(np.dot(normal, principal_direction)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot(normal, center)), plane_offset, places=6)
        self.assertAlmostEqual(float(np.dot(horizontal, principal_direction)), 0.0, places=6)
        self.assertAlmostEqual(float(np.dot(vertical, principal_direction)), 0.0, places=6)

    def test_xyz_region_uses_geometric_center_instead_of_point_density(self) -> None:
        full_span = np.column_stack((np.linspace(-10.0, 10.0, 41), np.zeros(41), np.zeros(41)))
        dense_left = np.column_stack((np.linspace(-10.0, -6.0, 401), np.zeros(401), np.zeros(401)))
        region_points = np.vstack((full_span, dense_left))

        angle_degrees, tilt_degrees, plane_offset, _ = normal_plane_from_xyz_region(region_points)
        normal, _, _ = vertical_plane_basis(angle_degrees, tilt_degrees)

        self.assertAlmostEqual(float(np.dot(normal, [1.0, 0.0, 0.0])), 1.0, places=6)
        self.assertAlmostEqual(plane_offset, 0.0, places=6)

    def test_scan_section_offsets_reports_parallel_planes_without_mutating_config(self) -> None:
        points = np.vstack(
            (
                np.column_stack(
                    (
                        np.zeros(80),
                        np.linspace(-2.0, 2.0, 80),
                        np.zeros(80),
                    )
                ),
                np.column_stack(
                    (
                        np.full(80, 0.5),
                        np.linspace(-2.0, 2.0, 80),
                        np.zeros(80),
                    )
                ),
            )
        )
        config = SectionConfig(
            angle_degrees=0.0,
            tilt_degrees=0.0,
            plane_offset=0.0,
            thickness=0.10,
            horizontal_min=-3.0,
            horizontal_max=3.0,
            vertical_min=-3.0,
            vertical_max=3.0,
            component_cell_size=0.20,
            use_statistical_filter=False,
            use_component_filter=False,
        )
        rows = scan_section_offsets(points, config, half_range=0.5, step=0.25)
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(config.plane_offset, 0.0)
        center = next(row for row in rows if row["is_current"])
        shifted = next(row for row in rows if np.isclose(row["plane_offset"], 0.5))
        self.assertEqual(center["point_count"], 80)
        self.assertEqual(shifted["point_count"], 80)
        self.assertGreater(float(center["points_per_cell"]), 0.0)
        self.assertAlmostEqual(
            float(center["points_per_cell"]),
            float(center["point_count"]) / max(int(center["cell_count"]), 1),
        )
        self.assertGreater(float(center["largest_component_span"]), 3.0)

    def test_scan_section_offsets_validates_sampling_limits(self) -> None:
        points = np.zeros((3, 3), dtype=float)
        config = SectionConfig(use_statistical_filter=False, use_component_filter=False)
        with self.assertRaises(ValueError):
            scan_section_offsets(points, config, half_range=-1.0)
        with self.assertRaises(ValueError):
            scan_section_offsets(points, config, step=0.0)

    def test_multi_section_audit_keeps_slices_independent_and_center_explicit(self) -> None:
        offsets = np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0))
        full_profile = np.linspace(0.0, 4.0, 80)
        split_profile = np.concatenate((np.linspace(0.0, 1.5, 40), np.linspace(2.5, 4.0, 40)))
        point_sets = []
        for offset in offsets:
            profile = split_profile if np.isclose(offset, 0.0) else full_profile
            point_sets.append(
                np.column_stack(
                    (
                        np.full(len(profile), offset),
                        profile,
                        np.zeros(len(profile)),
                    )
                )
            )
        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.concatenate(point_sets, axis=0))
        )
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.10,
            horizontal_min=-1.0,
            horizontal_max=5.0,
            vertical_min=-1.0,
            vertical_max=1.0,
            component_cell_size=0.20,
            use_statistical_filter=False,
            use_component_filter=False,
        )

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "multi_sections.png"
            diagnostic = render_multi_section_audit(
                cloud,
                config,
                output_path,
                probe_count=5,
                probe_spacing=0.5,
            )
            self.assertTrue(output_path.is_file())

        self.assertEqual(diagnostic["probe_count"], 5)
        self.assertTrue(diagnostic["formal_measurement_uses_center_only"])
        self.assertEqual(diagnostic["center_diagnostic_status"], "split_structure")
        self.assertEqual(diagnostic["diagnostic_status"], "neighbor_only_stable")
        self.assertAlmostEqual(config.plane_offset, 0.0)

    def test_offset_candidate_rejects_single_view_bridge(self) -> None:
        horizontal_left = np.linspace(0.0, 2.0, 80)
        horizontal_right = np.linspace(4.0, 6.0, 80)
        base_profile = np.vstack(
            (
                np.column_stack((np.zeros(80), horizontal_left, np.zeros(80))),
                np.column_stack((np.zeros(80), horizontal_right, np.zeros(80))),
            )
        )
        # This bridge makes a nearby slice look continuous, but it is visible in
        # only one source view and must not be recommended automatically.
        bridge_offsets = np.repeat(np.asarray((0.75, 1.0, 1.25)), 80)
        bridge_horizontal = np.tile(np.linspace(2.0, 4.0, 80), 3)
        bridge = np.column_stack(
            (bridge_offsets, bridge_horizontal, np.zeros(len(bridge_offsets)))
        )
        source_arrays = [
            np.vstack((base_profile, bridge)),
            base_profile.copy(),
            base_profile.copy(),
        ]
        source_clouds = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            for points in source_arrays
        ]
        merged_points = np.concatenate(source_arrays, axis=0)
        config = SectionConfig(
            angle_degrees=0.0,
            tilt_degrees=0.0,
            plane_offset=0.0,
            thickness=0.12,
            horizontal_min=-1.0,
            horizontal_max=7.0,
            vertical_min=-1.0,
            vertical_max=1.0,
            component_cell_size=0.20,
            use_statistical_filter=False,
            use_component_filter=False,
            offset_min_quality_gain=0.01,
        )

        result = evaluate_section_offset_candidates(
            merged_points,
            config,
            half_range=1.5,
            step=0.25,
            maximum_points=100_000,
            source_clouds=source_clouds,
            source_indices=[0, 1, 2],
            maximum_source_points_per_view=100_000,
        )

        self.assertEqual(result["status"], "source_coverage_gap")
        self.assertIsNone(result["recommended_offset"])
        candidate_rows = [
            row
            for row in result["rows"]
            if float(row["plane_offset"]) > 0.5
            and bool(row.get("structural_improvement", False))
        ]
        self.assertTrue(candidate_rows)
        self.assertTrue(
            all(
                int(row.get("source_view_count", 0)) < 3
                or not bool(row.get("source_support_guardrail_ok", False))
                for row in candidate_rows
            )
        )

    def test_tilted_normal_plane_selects_points_by_full_xyz_distance(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, -np.sqrt(3.0)],
                [0.30, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        config = SectionConfig(
            angle_degrees=0.0,
            tilt_degrees=30.0,
            plane_offset=0.0,
            thickness=0.20,
            horizontal_min=-10.0,
            horizontal_max=10.0,
            vertical_min=-10.0,
            vertical_max=10.0,
        )

        np.testing.assert_array_equal(select_vertical_section_indices(points, config), [0, 1])

    def test_target_region_guides_plane_without_clipping_section_profile(self) -> None:
        points = np.array(
            [
                [0.02, 0.0, 0.0],
                [-0.01, 0.5, 0.5],
                [0.03, 1.5, 0.5],
                [0.02, 1.5, 2.0],
            ],
            dtype=np.float64,
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.20,
            horizontal_min=-10.0,
            horizontal_max=10.0,
            vertical_min=-1.0,
            vertical_max=3.0,
            target_region=Bounds2D(x_min=-1.0, x_max=1.0, y_min=-0.1, y_max=1.0),
            use_statistical_filter=False,
            use_component_filter=False,
        )

        section = extract_vertical_section(cloud, config, backend="cpu", clean=False)

        np.testing.assert_array_equal(np.asarray(section.points), points)

    def test_section_component_filter_removes_small_islands(self) -> None:
        main_component = np.array(
            [[0.0, y, z] for y in np.linspace(-1.0, 1.0, 21) for z in np.linspace(0.0, 1.0, 3)],
            dtype=np.float64,
        )
        secondary_component = np.array(
            [[0.0, y, z] for y in np.linspace(3.0, 3.8, 5) for z in np.linspace(0.0, 0.8, 4)],
            dtype=np.float64,
        )
        isolated_points = np.array([[0.0, 6.0, 0.0], [0.0, 8.0, 2.0]], dtype=np.float64)
        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.vstack((main_component, secondary_component, isolated_points)))
        )
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.20,
            horizontal_min=-10.0,
            horizontal_max=10.0,
            vertical_min=-1.0,
            vertical_max=3.0,
            use_statistical_filter=False,
            use_component_filter=True,
            component_cell_size=0.25,
            component_min_points=8,
        )

        cleaned = clean_vertical_section(cloud, config)

        self.assertEqual(len(cleaned.points), len(main_component) + len(secondary_component))

    def test_section_component_filter_removes_dense_compact_island(self) -> None:
        main_component = np.array(
            [[0.0, y, z] for y in np.linspace(-1.0, 1.0, 21) for z in np.linspace(0.0, 1.0, 3)],
            dtype=np.float64,
        )
        compact_island = np.array(
            [[0.0, y, z] for y in np.linspace(3.0, 3.16, 8) for z in np.linspace(0.0, 0.16, 8)],
            dtype=np.float64,
        )
        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.concatenate((main_component, compact_island), axis=0))
        )
        config = SectionConfig(
            angle_degrees=0.0,
            plane_offset=0.0,
            thickness=0.20,
            horizontal_min=-10.0,
            horizontal_max=10.0,
            vertical_min=-1.0,
            vertical_max=3.0,
            use_statistical_filter=False,
            use_component_filter=True,
            component_cell_size=0.25,
            component_min_points=8,
        )

        cleaned = clean_vertical_section(cloud, config)

        self.assertEqual(len(cleaned.points), len(main_component))

    def test_calibration_plane_leveling_aligns_normal_to_z(self) -> None:
        generator = np.random.default_rng(20260826)
        xy = generator.uniform(-4.0, 4.0, size=(2_000, 2))
        expected_normal = np.array([0.12, -0.08, 1.0], dtype=np.float64)
        expected_normal /= np.linalg.norm(expected_normal)
        z_values = -(expected_normal[0] * xy[:, 0] + expected_normal[1] * xy[:, 1]) / expected_normal[2]
        plane_points = np.column_stack((xy, z_values))
        plane_points += generator.normal(0.0, 0.006, size=plane_points.shape)
        clutter = generator.uniform(-3.0, 3.0, size=(700, 3))
        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.vstack((plane_points, clutter)))
        )

        result = estimate_calibration_leveling(
            cloud,
            CalibrationConfig(leveling_plane_distance=0.05, leveling_minimum_points=700),
        )
        leveled_normal = result.transformation[:3, :3] @ result.reference_plane_normal

        self.assertTrue(result.applied)
        self.assertGreater(result.plane_points, 1_500)
        self.assertLess(result.plane_rmse, 0.02)
        np.testing.assert_allclose(leveled_normal, [0.0, 0.0, 1.0], atol=1e-8)

    def test_full_cloud_alignment_dominates_registration_candidate_ranking(self) -> None:
        config = RegistrationConfig(icp_max_correspondence_distance=0.90)
        calibration_only_candidate = registration_candidate_score(
            global_fitness=0.95,
            icp_fitness=0.98,
            icp_rmse=0.10,
            full_cloud_fitness=0.42,
            full_cloud_rmse=0.42,
            config=config,
        )
        globally_consistent_candidate = registration_candidate_score(
            global_fitness=0.72,
            icp_fitness=0.85,
            icp_rmse=0.24,
            full_cloud_fitness=0.84,
            full_cloud_rmse=0.30,
            config=config,
        )

        self.assertGreater(globally_consistent_candidate, calibration_only_candidate)

    def test_generic_group_uses_calibration_block_for_coarse_registration(self) -> None:
        self._assert_automatic_calibration_pipeline(7)

    def test_automatic_five_view_group_uses_calibration_geometry(self) -> None:
        self._assert_automatic_calibration_pipeline(5)

    def test_automatic_six_view_group_uses_calibration_geometry(self) -> None:
        self._assert_automatic_calibration_pipeline(6)

    def _assert_automatic_calibration_pipeline(self, view_count) -> None:
        """Each ICP candidate must refine against its matched fixture view."""
        marker = np.array(
            [
                [x, y, z]
                for x in np.linspace(0.10, 0.90, 4)
                for y in np.linspace(0.10, 0.90, 4)
                for z in np.linspace(0.10, 0.90, 4)
            ],
            dtype=np.float64,
        )
        workpiece = np.array(
            [
                [2.0, y, z]
                for y in np.linspace(0.0, 4.0, 11)
                for z in np.linspace(-0.5, 1.5, 5)
            ],
            dtype=np.float64,
        )
        all_points = np.vstack((marker, workpiece))
        observed_registration_inputs: list[tuple[np.ndarray, np.ndarray]] = []

        def fake_global_register(source, target, _config, _backend, **_kwargs):
            observed_registration_inputs.append(
                (np.asarray(source.points).copy(), np.asarray(target.points).copy())
            )
            return np.eye(4, dtype=np.float64), 0.95, 0.01

        def fake_icp(_source, target, refinement, _backend, *_args):
            np.testing.assert_allclose(
                np.asarray(target.points), observed_registration_inputs[-1][1]
            )
            self.assertLessEqual(refinement.registration_voxel_size, config.registration.calibration_graph_voxel_size)
            self.assertLessEqual(refinement.icp_max_correspondence_distance,
                                 config.registration.calibration_graph_max_correspondence_distance)
            return np.eye(4, dtype=np.float64), 0.99, 0.01, {}

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for index in range(view_count):
                path = root / f"view-{index}.ply"
                o3d.io.write_point_cloud(
                    str(path),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((marker[index:], workpiece)))),
                )
                paths.append(path)

            config = ProcessingConfig(
                denoise=DenoiseConfig(
                    voxel_size=0.01,
                    use_statistical_filter=False,
                    use_radius_filter=False,
                ),
                calibration=CalibrationConfig(
                    use_roi=True,
                    bounds=Bounds3D(
                        x_min=0.0,
                        x_max=1.0,
                        y_min=0.0,
                        y_max=1.0,
                        z_min=0.0,
                        z_max=1.0,
                    ),
                    minimum_points=30,
                    level_reference_plane=False,
                ),
                registration=RegistrationConfig(
                    use_six_station_pose_prior=False,
                    registration_voxel_size=0.05,
                    use_global_registration=True,
                    global_attempts=1,
                    global_max_iterations=10,
                    use_global_pose_validation=False,
                    minimum_calibration_icp_fitness=0.50,
                    maximum_calibration_icp_rmse=0.20,
                    skip_low_quality_calibration_views=True,
                    use_section_alignment_validation=False,
                    coverage_preserving_fusion=True,
                ),
                section=SectionConfig(
                    plane_offset=2.0,
                    thickness=0.20,
                    horizontal_min=-1.0,
                    horizontal_max=5.0,
                    vertical_min=-1.0,
                    vertical_max=2.0,
                    use_statistical_filter=False,
                    use_component_filter=False,
                ),
                fusion_voxel_size=0.03,
                compute_backend="cpu",
            )

            with patch(
                "pointcloud_app.pipeline.coarse_global_register",
                side_effect=fake_global_register,
            ), patch(
                "pointcloud_app.pipeline._refine_with_icp_diagnostics",
                side_effect=fake_icp,
            ), patch(
                "pointcloud_app.pipeline.refine_with_icp",
                return_value=(np.eye(4, dtype=np.float64), 0.99, 0.01),
            ), patch(
                "pointcloud_app.pipeline.evaluate_full_cloud_alignment",
                return_value=(0.90, 0.02),
            ):
                result = process_group(f"generic-{view_count}", paths, config, root / "outputs")

        self.assertTrue(observed_registration_inputs)
        self.assertTrue(
            all(
                len(source_points) < len(all_points)
                and len(target_points) < len(all_points)
                and np.all((source_points >= 0.0) & (source_points <= 1.0))
                and np.all((target_points >= 0.0) & (target_points <= 1.0))
                for source_points, target_points in observed_registration_inputs
            )
        )
        self.assertGreater(result.merged_points, 0)


if __name__ == "__main__":
    unittest.main()
