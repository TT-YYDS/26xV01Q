import copy
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from pointcloud_app.models import DenoiseConfig
from pointcloud_app.native.backend import config_from_dict
from pointcloud_app.pipeline import (
    _surface_patch_support_mask, _surface_support_inlier_mask, denoise_cloud,
)


class SurfaceSupportTests(unittest.TestCase):
    def plane(self):
        x, y = np.meshgrid(np.arange(-2, 2.01, .08), np.arange(-2, 2.01, .08))
        return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

    def test_plane_and_boundary_are_not_eroded(self):
        points = self.plane()
        self.assertTrue(_surface_support_inlier_mask(points, .1).all())

    def test_sharp_corner_preserves_both_faces_and_edge(self):
        a, b = np.meshgrid(np.arange(0, 2.01, .08), np.arange(-2, 2.01, .08))
        face = np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))
        wall = face[:, [2, 1, 0]]
        points = np.unique(np.vstack((face, wall)), axis=0)
        mask = _surface_support_inlier_mask(points, .1)
        edge = (points[:, 0] < .1) & (points[:, 2] < .1)
        self.assertGreater(mask.mean(), .98)
        self.assertGreater(mask[edge].mean(), .98)

    def test_parallel_thin_surfaces_are_not_averaged_or_erased(self):
        plane = self.plane()
        points = np.vstack((plane, plane + [0, 0, .08]))
        mask = _surface_support_inlier_mask(points, .1)
        self.assertGreater(mask.mean(), .98)
        self.assertEqual(set(np.unique(points[mask, 2])), {0., .08})

    def test_round_surface_is_retained(self):
        angle, y = np.meshgrid(np.linspace(0, np.pi, 80), np.arange(-2, 2.01, .08))
        points = np.column_stack((np.cos(angle.ravel()), y.ravel(), np.sin(angle.ravel())))
        self.assertGreater(_surface_support_inlier_mask(points, .1).mean(), .98)

    def test_flying_points_are_removed_without_changing_original(self):
        plane = self.plane()
        rng = np.random.default_rng(47)
        noise = np.column_stack((rng.uniform(-1, 1, (100, 2)), rng.uniform(.12, .22, 100)))
        points = np.vstack((plane, noise))
        original = points.copy()
        keep = _surface_support_inlier_mask(points, .1)
        self.assertGreater(keep[:len(plane)].mean(), .98)
        self.assertLess(keep[len(plane):].mean(), .2)
        np.testing.assert_array_equal(points, original)

    def test_nearby_thin_return_is_removed_while_plane_is_preserved(self):
        plane = self.plane()
        rng = np.random.default_rng(18)
        thin_return = np.column_stack((
            rng.uniform(-1, 1, (100, 2)),
            rng.uniform(.07, .08, 100),
        ))
        points = np.vstack((plane, thin_return))
        keep = _surface_support_inlier_mask(points, .1)
        self.assertGreater(keep[:len(plane)].mean(), .995)
        self.assertLess(keep[len(plane):].mean(), .05)

    def strip_scene(self):
        x, y = np.meshgrid(np.arange(-3, 3.01, .08), np.arange(-2, 2.01, .08))
        face = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        x, z = np.meshgrid(np.arange(-.12, .121, .06), np.arange(.10, .81, .05))
        strip = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        strips = np.vstack([strip + [offset, 0, 0] for offset in (-1.5, 0, 1.5)])
        return face, strips

    def test_small_planar_strips_cannot_support_themselves(self):
        face, strips = self.strip_scene()
        points = np.vstack((face, strips))
        original = points.copy()
        with patch("pointcloud_app.pipeline._surface_patch_support_mask",
                   side_effect=lambda p, *args: np.ones(len(p), dtype=bool)):
            before = _surface_support_inlier_mask(points, .1)
        after = _surface_support_inlier_mask(points, .1)
        self.assertGreater(np.sum(before[len(face):]), 150)
        self.assertEqual(np.sum(after[len(face):]), 0)
        self.assertGreater(after[:len(face)].mean(), .995)
        np.testing.assert_array_equal(points, original)

    def test_long_thin_wall_retains_continuous_support(self):
        face, _ = self.strip_scene()
        x, z = np.meshgrid(np.arange(-2.5, 2.51, .08), np.arange(.08, .81, .08))
        wall = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
        points = np.vstack((face, wall))
        keep = _surface_support_inlier_mask(points, .1)
        self.assertGreater(keep[:len(face)].mean(), .99)
        self.assertGreater(keep[len(face):].mean(), .98)
        self.assertGreater(np.unique(points[len(face):][keep[len(face):], 0]).size, 60)

    def test_coplanar_small_patch_across_gap_is_not_mistaken_for_a_strip(self):
        face = self.plane()
        x, y = np.meshgrid(np.arange(2.15, 2.36, .05), np.arange(-.2, .21, .05))
        patch_points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        points = np.vstack((face, patch_points))
        keep = _surface_support_inlier_mask(points, .1)
        self.assertGreater(keep[len(face):].mean(), .98)

    def test_closer_parallel_face_cannot_hide_local_coplanar_support(self):
        x, y = np.meshgrid(np.arange(-1., 1.01, .08), np.arange(-1., 1.01, .08))
        upper = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .12)))
        x, y = np.meshgrid(np.arange(.8, 2.01, .08), np.arange(-1., 1.01, .08))
        lower = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        x, y = np.meshgrid(np.linspace(-.1, .1, 5), np.linspace(-.1, .1, 5))
        patch_points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        points = np.vstack((upper, lower, patch_points))
        normals = np.tile([0., 0., 1.], (len(points), 1))
        keep = _surface_patch_support_mask(
            points, points, normals, np.arange(len(points)), .1, .04, 24,
        )
        self.assertTrue(keep.all())

    def test_strip_rejection_is_rotation_and_scale_invariant(self):
        face, strips = self.strip_scene()
        points = np.vstack((face, strips))
        rng = np.random.default_rng(37)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        before = _surface_support_inlier_mask(points, .1)
        after = _surface_support_inlier_mask((points @ rotation + [4, -9, 2]) * 3, .3)
        self.assertGreater(np.mean(before == after), .995)
        self.assertLess(after[len(face):].mean(), .02)

    def test_unanchored_small_scan_is_not_emptied(self):
        _, strips = self.strip_scene()
        self.assertTrue(_surface_support_inlier_mask(strips, .1).all())

    def test_small_and_sparse_planar_inputs_are_preserved(self):
        for points in (np.empty((0, 3)), self.plane()[:20], self.plane()[::5] * 3):
            self.assertTrue(_surface_support_inlier_mask(points, .1).all())

    def test_rotating_and_scaling_preserves_the_filter_decision(self):
        plane = self.plane()
        rng = np.random.default_rng(11)
        noise = np.column_stack((rng.uniform(-1, 1, (100, 2)), rng.uniform(.12, .22, 100)))
        points = np.vstack((plane, noise))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        expected = _surface_support_inlier_mask(points, .1)
        actual = _surface_support_inlier_mask((points @ rotation + [3, -4, 7]) * 2, .2)
        self.assertGreater(np.mean(expected == actual), .995)

    def test_enabled_stage_preserves_retained_point_attributes(self):
        points = self.plane()
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(np.full_like(points, .25))
        config = DenoiseConfig(use_statistical_filter=False, use_radius_filter=False,
                               use_component_filter=False)
        cleaned, stats = denoise_cloud(cloud, config, "plane")
        self.assertTrue(np.all(np.asarray(cleaned.colors) == .25))
        originals = set(map(tuple, points))
        self.assertTrue(all(tuple(p) in originals for p in np.asarray(cleaned.points)))
        self.assertEqual(stats.after_surface_support_points, stats.output_points)
        self.assertLessEqual(stats.output_points, stats.after_component_points)

    def test_filter_can_be_disabled_and_reports_stage_counts(self):
        points = self.plane()
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        config = DenoiseConfig(use_statistical_filter=False, use_radius_filter=False,
                               use_component_filter=False, use_surface_support_filter=False)
        with patch("pointcloud_app.pipeline._surface_support_inlier_mask") as filtering:
            cleaned, stats = denoise_cloud(cloud, config, "plane")
        filtering.assert_not_called()
        self.assertEqual(stats.after_surface_support_points, len(cleaned.points))
        np.testing.assert_array_equal(np.asarray(cloud.points), points)

    def test_old_config_loads_defaults_and_new_config_roundtrips(self):
        config = config_from_dict({"denoise": {"voxel_size": .1}})
        self.assertTrue(config.denoise.use_surface_support_filter)
        restored = config_from_dict(config.as_dict())
        self.assertEqual(config.denoise, restored.denoise)
        disabled = copy.deepcopy(config)
        disabled.denoise.use_surface_support_filter = False
        self.assertFalse(config_from_dict(disabled.as_dict()).denoise.use_surface_support_filter)

    def test_parameters_are_validated(self):
        for values in ({"surface_support_neighbors": 7},
                       {"surface_support_distance_factor": 0},
                       {"surface_support_distance_factor": float("nan")}):
            with self.assertRaises(ValueError):
                DenoiseConfig(**values).validate()


if __name__ == "__main__":
    unittest.main()
