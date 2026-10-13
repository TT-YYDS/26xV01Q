import copy
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import open3d as o3d

from pointcloud_app.models import Bounds3D, CalibrationConfig, ProcessingConfig, RegistrationConfig
from pointcloud_app.overlap_registration import (
    fixture_component_mask, refine_overlap_pair, refine_multiview_overlap,
)
from pointcloud_app.profiles import apply_family_profile
from pointcloud_app.pipeline import (
    _disambiguate_global_pose, _recover_calibration_roi,
    _select_calibration_reference, _identity_leveling_result,
    _merge_clouds_with_section_consensus, _refine_with_icp_diagnostics,
    _recover_multiview_poses, RegistrationMetric,
)


def cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


class OverlapRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.config = RegistrationConfig(workpiece_graph_voxel_size=.08)
        x, y = np.meshgrid(np.linspace(-3, 3, 70), np.linspace(-2, 2, 60))
        self.points = np.column_stack((x.ravel(), y.ravel(),
                                      (.18 * y ** 2 + .08 * np.sin(x * 2)).ravel()))

    def test_recovers_side_offset_without_modifying_input(self):
        source = cloud(self.points + [0, 0, .12])
        before = np.asarray(source.points).copy()
        correction, report = refine_overlap_pair(source, cloud(self.points), self.config)
        self.assertTrue(report["accepted"], report)
        self.assertLess(report["rmse_after"], report["rmse_before"] * .6)
        np.testing.assert_array_equal(np.asarray(source.points), before)
        self.assertAlmostEqual(np.linalg.det(correction[:3, :3]), 1, places=8)
        self.assertAlmostEqual(correction[2, 3], -.12, delta=.025)

    def test_overlap_refinement_generalizes_across_frames_and_sampling(self):
        rng = np.random.default_rng(51)
        for angle in (0., .7, 1.4):
            with self.subTest(angle=angle):
                rotation = o3d.geometry.get_rotation_matrix_from_xyz((angle, angle / 2, -.3))
                target = self.points @ rotation.T + [23, -11, 5]
                sparse = target[::2] + rng.normal(0, .003, target[::2].shape)
                shifted = sparse + np.array([0, 0, .12]) @ rotation.T
                outliers = rng.uniform(40, 45, (300, 3))
                transform, report = refine_overlap_pair(
                    cloud(np.vstack((shifted, outliers))), cloud(target), self.config)
                self.assertTrue(report["accepted"], report)
                recovered = shifted @ transform[:3, :3].T + transform[:3, 3]
                self.assertLess(np.sqrt(np.mean(np.sum((recovered - sparse) ** 2, axis=1))), .035)

    def test_no_overlap_keeps_pose(self):
        transform, report = refine_overlap_pair(
            cloud(self.points + [0, 0, 20]), cloud(self.points), self.config)
        np.testing.assert_array_equal(transform, np.eye(4))
        self.assertFalse(report["accepted"])

    def test_aligned_points_do_not_drift(self):
        transform, report = refine_overlap_pair(cloud(self.points), cloud(self.points), self.config)
        self.assertFalse(report["accepted"])
        np.testing.assert_array_equal(transform, np.eye(4))

    def test_rejects_excessive_correction(self):
        self.config.workpiece_graph_max_translation = .01
        transform, report = refine_overlap_pair(
            cloud(self.points + [0, 0, .12]), cloud(self.points), self.config)
        self.assertEqual(report["reason"], "correction_limit")
        np.testing.assert_array_equal(transform, np.eye(4))

    def test_reference_excluded_views_and_unique_samples_are_preserved(self):
        clouds = [cloud(self.points), cloud(self.points + [0, 0, .12]),
                  cloud(self.points + [0, 0, 20])]
        snapshots = [np.asarray(c.points).copy() for c in clouds]
        result, report = refine_multiview_overlap(
            clouds, [np.eye(4)] * 3, [True, True, False],
            CalibrationConfig(use_roi=False), self.config)
        np.testing.assert_array_equal(result[0], np.eye(4))
        np.testing.assert_array_equal(result[2], np.eye(4))
        self.assertEqual(len(report["views"]), 1)
        for c, original in zip(clouds, snapshots):
            np.testing.assert_array_equal(np.asarray(c.points), original)

    def test_fixture_growth_preserves_separate_workpiece(self):
        x, y = np.meshgrid(np.arange(0, 2, .1), np.arange(0, 2, .1))
        fixture = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        workpiece = fixture + [-5, 0, 0]
        config = CalibrationConfig(bounds=Bounds3D(-.1, 2.1, .8, 2.1, -.2, .2))
        mask = fixture_component_mask(np.vstack((fixture, workpiece)), config, .15)
        self.assertTrue(mask[:len(fixture)].all())
        self.assertFalse(mask[len(fixture):].any())

    def test_dense_fixture_seed_cannot_swallow_connected_workpiece(self):
        x, y = np.meshgrid(np.arange(-5, 2, .1), np.arange(0, 2, .1))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        config = CalibrationConfig(bounds=Bounds3D(0, 2.1, -.1, 2.1, -.2, .2))
        mask = fixture_component_mask(points, config, .15)
        self.assertFalse(mask[points[:, 0] < -1].any())

    def test_families_have_distinct_stable_algorithm_parameters(self):
        base = ProcessingConfig()
        results = [apply_family_profile(base, family) for family in
                   ("numeric", "wood_base", "wood_variant", "default")]
        self.assertEqual([result.data_family for result in results],
                         ["c1", "c2", "c3", "default"])
        self.assertNotEqual(results[0].registration, results[1].registration)
        self.assertNotEqual(results[1].registration, results[2].registration)
        self.assertNotEqual(results[0].denoise, results[2].denoise)

    def test_global_pose_alternative_still_requires_fixture_quality(self):
        config = ProcessingConfig()
        best = (np.eye(4), .5, .1, .9, .1, 1000, .3, .2, {})
        fixture = cloud(np.vstack((np.tile([5., 5., 0.], (1000, 1)),
                                   self.points + [-10, 0, 0])))
        with patch("pointcloud_app.pipeline.coarse_global_register",
                   return_value=(np.eye(4), .9, .1)), patch(
                "pointcloud_app.pipeline._refine_with_icp_diagnostics",
                return_value=(np.eye(4), .1, .5, {})), patch(
                "pointcloud_app.pipeline.evaluate_full_cloud_alignment") as evaluate:
            result, localized = _disambiguate_global_pose(
                fixture, fixture, fixture, best, config, config.registration, "view", "cpu")
        self.assertIs(result, best)
        self.assertIsNone(localized)
        evaluate.assert_not_called()

    def test_global_pose_replaces_symmetric_match_only_with_better_evidence(self):
        config = ProcessingConfig()
        best = (np.eye(4), .5, .1, .9, .1, 1000, .3, .2, {})
        fixture = cloud(np.vstack((np.tile([5., 5., 0.], (1000, 1)),
                                   self.points + [-10, 0, 0])))
        for displacement, expected in ((20, True), (0, False)):
            baseline_pose = np.eye(4)
            baseline_pose[0, 3] = displacement
            best = (baseline_pose, *best[1:])
            with self.subTest(displacement=displacement), patch(
                    "pointcloud_app.pipeline.coarse_global_register",
                    return_value=(np.eye(4), .9, .1)), patch(
                    "pointcloud_app.pipeline._refine_with_icp_diagnostics",
                    return_value=(np.eye(4), .9, .05, {})), patch(
                    "pointcloud_app.pipeline.evaluate_full_cloud_alignment",
                    return_value=(.9, .05)):
                result, localized = _disambiguate_global_pose(
                    fixture, fixture, fixture, best, config, config.registration, "view", "cpu")
                self.assertEqual(localized is not None, expected)
                self.assertEqual(result is not best, expected)

    def test_missing_roi_is_transformed_back_to_source_coordinates(self):
        config = ProcessingConfig()
        reference = cloud(np.tile([5., 5., 0.], (1000, 1)))
        source = cloud(np.tile([5., -5., 0.], (1000, 1)))
        transform = np.eye(4)
        transform[1, 3] = 10
        with patch("pointcloud_app.pipeline.coarse_global_register",
                   return_value=(transform, .9, .05)):
            localized = _recover_calibration_roi(
                source, reference, config.calibration, config.registration, "view", "cpu")
        np.testing.assert_allclose(np.asarray(localized.points), np.asarray(source.points))

    def test_reference_completeness_check_does_not_fit_every_view(self):
        config = CalibrationConfig()
        for first_count, expected in ((50, 1), (400, 0)):
            inputs = [cloud(np.tile([5., 5., 0.], (first_count, 1))),
                      cloud(np.tile([5., 5., 0.], (1000, 1)))]
            config.minimum_points = 30
            with self.subTest(first_count=first_count), patch(
                    "pointcloud_app.pipeline.estimate_calibration_leveling",
                    return_value=_identity_leveling_result()) as fit:
                index, _, _ = _select_calibration_reference(
                    [Path("a.ply"), Path("b.ply")], inputs, config, "cpu")
            self.assertEqual(index, expected)
            self.assertEqual(fit.call_count, 1)

    def test_fusion_removes_fixture_in_reference_not_rotated_source_frame(self):
        config = ProcessingConfig()
        points = np.array([[5., 5., 0.], [-1., 0., 0.]])
        transform = np.eye(4)
        transform[0, 3] = 10
        merged, report = _merge_clouds_with_section_consensus(
            [Path("a.ply"), Path("b.ply")], [np.eye(4), transform], [True, True],
            np.eye(4), config.section, .03, "cpu",
            source_clouds=[cloud(points), cloud(points - [10, 0, 0])],
            calibration_config=config.calibration, calibration_bounds_in_reference_frame=True)
        self.assertTrue(np.all(np.asarray(merged.points)[:, 0] < 0))
        self.assertEqual(report["calibration_bounds_frame"], "reference_before_leveling")

    def test_multiscale_candidates_are_compared_at_one_fixed_gate(self):
        from types import SimpleNamespace
        config = RegistrationConfig(use_multiscale_icp=True)
        transforms = []
        for distance in (.03, .02, .01):
            transform = np.eye(4)
            transform[2, 3] = distance
            transforms.append(transform)
        # Tight-stage fitness alone appears worse, but its fixed-gate residual
        # improves. A final regressing stage must not replace that good pose.
        evaluated = [
            SimpleNamespace(fitness=.9, inlier_rmse=.10),
            SimpleNamespace(fitness=.9, inlier_rmse=.08),
            SimpleNamespace(fitness=.9, inlier_rmse=.03),
            SimpleNamespace(fitness=.9, inlier_rmse=.06),
            SimpleNamespace(fitness=.9, inlier_rmse=.03),
        ]
        stages = [SimpleNamespace(transformation=t, fitness=f, inlier_rmse=.02)
                  for t, f in zip(transforms, (.9, .3, .95))]
        with patch("pointcloud_app.pipeline._calibration_geometry_initial_correction",
                   return_value=(np.eye(4), {})), patch(
                "pointcloud_app.pipeline.prepare_registration_cloud",
                side_effect=lambda c, *_: c), patch(
                "open3d.pipelines.registration.registration_icp",
                side_effect=stages), patch(
                "open3d.pipelines.registration.evaluate_registration",
                side_effect=evaluated) as evaluate:
            result, _, _, _ = _refine_with_icp_diagnostics(
                cloud(self.points), cloud(self.points), config)
        np.testing.assert_array_equal(result, transforms[1])
        self.assertEqual(evaluate.call_count, 5)
        for call in evaluate.call_args_list:
            self.assertEqual(call.args[2], config.icp_max_correspondence_distance)

    def test_global_scoring_never_recrops_samples_for_candidate_poses(self):
        config = ProcessingConfig()
        fixture = cloud(np.vstack((np.tile([5., 5., 0.], (1000, 1)),
                                   self.points + [-10, 0, 0])))
        baseline_pose = np.eye(4)
        baseline_pose[0, 3] = 20
        best = (baseline_pose, .9, .05, .9, .05, 1000, .9, .05, {})
        def coarse(*args, candidate_scorer, **kwargs):
            identity_score = candidate_scorer(np.eye(4))
            inside_roi = np.eye(4)
            inside_roi[:3, 3] = [17, 8, 0]
            self.assertGreater(identity_score, candidate_scorer(inside_roi))
            return np.eye(4), .9, .05
        with patch("pointcloud_app.pipeline.fixture_component_mask",
                   wraps=fixture_component_mask) as masking, patch(
                "pointcloud_app.pipeline.coarse_global_register", side_effect=coarse), patch(
                "pointcloud_app.pipeline._refine_with_icp_diagnostics",
                return_value=(np.eye(4), .9, .05, {})):
            _disambiguate_global_pose(
                fixture, fixture, fixture, best, config, config.registration, "view", "cpu")
        self.assertEqual(masking.call_count, 2)

    def test_cached_pose_scoring_matches_rebuilt_trees_for_rigid_poses(self):
        from scipy.spatial import cKDTree
        config = ProcessingConfig()
        source = cloud(self.points)
        target = cloud(self.points + [.4, -.2, .1])
        source_points = np.asarray(source.voxel_down_sample(
            config.registration.registration_voxel_size).points)
        target_points = np.asarray(target.voxel_down_sample(
            config.registration.registration_voxel_size).points)
        gate = config.registration.icp_max_correspondence_distance
        baseline = np.eye(4)
        baseline[0, 3] = 40

        def coarse(*args, candidate_scorer, **kwargs):
            for angle in (0., .3, 1.2):
                pose = np.eye(4)
                pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((angle, -.1, .2))
                pose[:3, 3] = [.4, -.2, .1]
                points = source_points @ pose[:3, :3].T + pose[:3, 3]
                forward = cKDTree(target_points).query(points)[0]
                reverse = cKDTree(points).query(target_points)[0]
                expected = np.sqrt(np.mean(forward < gate) * np.mean(reverse < gate)) * (
                    1 - .25 * (np.mean(np.minimum(forward, gate)) +
                               np.mean(np.minimum(reverse, gate))) / (2 * gate))
                self.assertAlmostEqual(candidate_scorer(pose), expected, places=12)
            return np.eye(4), .9, .1

        with patch("pointcloud_app.pipeline.fixture_component_mask",
                   side_effect=lambda p, *_: np.zeros(len(p), dtype=bool)), patch(
                "pointcloud_app.pipeline.coarse_global_register", side_effect=coarse) as coarse_call, patch(
                "pointcloud_app.pipeline.crop_calibration_cloud", side_effect=ValueError):
            _disambiguate_global_pose(source, target, target,
                (baseline, .9, .1, .9, .1, 1000, .9, .1, {}),
                config, config.registration, "test", "cpu")
        self.assertGreater(coarse_call.call_count, 0)

    def test_confident_existing_pose_skips_search_only_with_fixture_and_scene_quality(self):
        config = ProcessingConfig()
        fixture = cloud(np.vstack((np.tile([5., 5., 0.], (1000, 1)),
                                   self.points + [-10, 0, 0])))
        for fitness, full_fitness, expected_calls in ((.9, .95, 0), (.1, .95, 1), (.9, .1, 1)):
            with self.subTest(fitness=fitness, full_fitness=full_fitness), patch(
                    "pointcloud_app.pipeline.coarse_global_register",
                    return_value=(np.eye(4), .9, .05)) as coarse, patch(
                    "pointcloud_app.pipeline._refine_with_icp_diagnostics",
                    return_value=(np.eye(4), .9, .05, {})):
                best = (np.eye(4), .9, .05, fitness, .05, 1000, full_fitness, .05, {})
                _disambiguate_global_pose(fixture, fixture, fixture, best,
                    config, config.registration, "test", "cpu")
            self.assertEqual(bool(coarse.call_count), bool(expected_calls))

    def test_precomputed_gicp_normals_preserve_native_result(self):
        source = cloud(self.points[::4] + [0, 0, .1])
        target = cloud(self.points[::4])
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        before = o3d.pipelines.registration.registration_generalized_icp(
            source, target, .5, np.eye(4), criteria=criteria)
        source.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        after = o3d.pipelines.registration.registration_generalized_icp(
            source, target, .5, np.eye(4), criteria=criteria)
        np.testing.assert_allclose(before.transformation, after.transformation, atol=1e-8)
        self.assertAlmostEqual(before.fitness, after.fitness, places=10)

    def test_multiview_recovery_uses_other_views_and_preserves_anchor(self):
        config = ProcessingConfig()
        config.registration.use_six_station_pose_prior = False
        clouds = [cloud(self.points), cloud(self.points + [0, 0, 10]), cloud(self.points)]
        metrics = [RegistrationMetric(
            file_name=f"{i}.ply", calibration_points=1000, global_fitness=.9,
            global_rmse=.05, icp_fitness=.9, icp_rmse=.05,
            transformation=np.eye(4).tolist()) for i in range(3)]
        correction = np.eye(4)
        correction[2, 3] = -10
        def recover(source, reference, target, best, *args, reference_views):
            self.assertLess(np.asarray(reference.points)[:, 2].max(), 5)
            self.assertEqual(len(reference_views), 2)
            for peer in reference_views:
                self.assertLess(np.asarray(peer.points)[:, 2].max(), 5)
            report = {"global_pose_disambiguation": {
                "accepted": True, "quality_before": 0., "quality_after": .9,
                "fixture_quality_passed": True}}
            return (correction, *best[1:8], report), target
        with patch("pointcloud_app.pipeline.crop_calibration_cloud",
                   return_value=cloud(self.points)), patch(
                "pointcloud_app.pipeline._disambiguate_global_pose",
                side_effect=recover) as retry:
            result, rows = _recover_multiview_poses(
                clouds, [np.eye(4)] * 3, metrics, config, None, "test")
        self.assertEqual(retry.call_count, 1)
        np.testing.assert_array_equal(result[0], np.eye(4))
        np.testing.assert_array_equal(result[1], correction)
        np.testing.assert_array_equal(result[2], np.eye(4))
        self.assertEqual(rows[0]["reference_indices"], [0, 2])
        np.testing.assert_array_equal(np.asarray(clouds[1].points), self.points + [0, 0, 10])

        from types import SimpleNamespace
        def support_tree(fraction):
            distances = np.r_[np.zeros(int(fraction * 1000)),
                              np.full(1000 - int(fraction * 1000), 100.)]
            return SimpleNamespace(query=lambda _: (distances, None))
        with patch("pointcloud_app.pipeline.crop_calibration_cloud",
                   return_value=cloud(self.points)), patch(
                "pointcloud_app.pipeline._disambiguate_global_pose",
                side_effect=recover) as retry, patch(
                "pointcloud_app.pipeline.cKDTree",
                side_effect=[support_tree(.4), support_tree(.1),
                             support_tree(1.), support_tree(1.)]):
            _, rows = _recover_multiview_poses(
                clouds, [np.eye(4)] * 3, metrics, config, None, "test")
        self.assertEqual(retry.call_count, 1)
        self.assertAlmostEqual(rows[0]["support_before"], .2)

        metrics[2].included_in_fusion = False
        with patch("pointcloud_app.pipeline._disambiguate_global_pose") as retry:
            result, rows = _recover_multiview_poses(
                clouds, [np.eye(4)] * 3, metrics, config, None, "test")
        retry.assert_not_called()
        self.assertEqual(rows, [])

    def test_multiview_recovers_excluded_view_only_with_independent_support(self):
        config = ProcessingConfig()
        config.registration.use_six_station_pose_prior = False
        clouds = [cloud(self.points), cloud(self.points),
                  cloud(self.points + [0, 0, 10]), cloud(self.points)]
        metrics = [RegistrationMetric(
            file_name=f"{i}.ply", calibration_points=1000, global_fitness=.9,
            global_rmse=.05, icp_fitness=.9, icp_rmse=.05,
            transformation=np.eye(4).tolist()) for i in range(4)]
        metrics[2].included_in_fusion = False
        metrics[2].exclusion_reason = "calibration_icp_quality"
        metrics[2].calibration_quality_passed = False
        correction = np.eye(4)
        correction[2, 3] = -10

        def recover(source, reference, target, best, *args, reference_views):
            self.assertEqual(len(reference_views), 3)
            report = {"global_pose_disambiguation": {"accepted": True,
                      "quality_before": 0., "quality_after": .9,
                      "fixture_quality_passed": True}}
            return (correction, *best[1:8], report), target

        with patch("pointcloud_app.pipeline.crop_calibration_cloud",
                   return_value=cloud(self.points)), patch(
                "pointcloud_app.pipeline._disambiguate_global_pose",
                side_effect=recover) as retry:
            result, rows = _recover_multiview_poses(
                clouds, [np.eye(4)] * 4, metrics, config, None, "test")
        self.assertEqual(retry.call_count, 1)
        self.assertEqual(rows[0]["view_index"], 2)
        self.assertTrue(rows[0]["accepted"])
        self.assertTrue(metrics[2].included_in_fusion)
        self.assertIsNone(metrics[2].exclusion_reason)
        np.testing.assert_array_equal(result[2], correction)

        metrics[2].included_in_fusion = False
        metrics[2].exclusion_reason = "calibration_icp_quality"
        metrics[2].calibration_quality_passed = False
        def weak_recovery(source, reference, target, best, *args, reference_views):
            report = {"global_pose_disambiguation": {
                "accepted": True, "quality_after": .2}}
            return (correction, *best[1:8], report), target
        with patch("pointcloud_app.pipeline.crop_calibration_cloud",
                   return_value=cloud(self.points)), patch(
                "pointcloud_app.pipeline._disambiguate_global_pose",
                side_effect=weak_recovery):
            result, rows = _recover_multiview_poses(
                clouds, [np.eye(4)] * 4, metrics, config, None, "test")
        self.assertFalse(rows[0]["accepted"])
        self.assertFalse(metrics[2].included_in_fusion)
        self.assertEqual(metrics[2].exclusion_reason, "calibration_icp_quality")
        np.testing.assert_array_equal(result[2], np.eye(4))


if __name__ == "__main__":
    unittest.main()
