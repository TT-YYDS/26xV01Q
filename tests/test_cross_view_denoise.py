import unittest

import numpy as np
import open3d as o3d

from pointcloud_app.pipeline import (
    _cross_view_overlap_keep_masks, _denoise_registered_views,
    _final_fusion_component_cleanup,
)


class CrossViewDenoiseTests(unittest.TestCase):
    def scene(self):
        x, y = np.meshgrid(np.arange(-1, 1.01, .05), np.arange(-1, 1.01, .05))
        plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        strip = np.column_stack((
            np.linspace(-.3, .3, 30), np.zeros(30), np.full(30, .24),
        ))
        unique = plane[::10] + [5, 0, 1]
        return plane, strip, unique

    def cloud(self, points):
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    def masks(self, clouds, included=None, transforms=None, leveling=None):
        return _cross_view_overlap_keep_masks(
            clouds, transforms if transforms is not None else [np.eye(4)] * len(clouds),
            leveling if leveling is not None else np.eye(4),
            included if included is not None else [True] * len(clouds), .1,
        )

    def test_removes_view_only_strip_and_preserves_overlap_plane(self):
        plane, strip, _ = self.scene()
        masks, report = self.masks([
            self.cloud(np.vstack((plane, strip))), self.cloud(plane + [.01, .01, 0]),
        ])
        self.assertTrue(masks[0][:len(plane)].all())
        self.assertFalse(masks[0][len(plane):].any())
        self.assertTrue(masks[1].all())
        self.assertEqual(report["total_removed_points"], len(strip))
        self.assertEqual(report["stage"], "registered_single_views_before_fusion")

    def test_preserves_real_region_without_other_view_context(self):
        plane, strip, unique = self.scene()
        masks, report = self.masks([
            self.cloud(np.vstack((plane, strip, unique))), self.cloud(plane),
        ])
        self.assertTrue(masks[0][-len(unique):].all())
        self.assertEqual(report["views"][0]["unsupported_without_context_points"], len(unique))

    def test_registration_and_leveling_are_used_without_mutating_inputs(self):
        plane, strip, _ = self.scene()
        points = np.vstack((plane, strip))
        angle = .7
        pose = np.eye(4)
        pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]
        pose[:3, 3] = [4, -3, 2]
        local = (plane - pose[:3, 3]) @ pose[:3, :3]
        level = pose.copy()
        clouds = [self.cloud(points), self.cloud(local)]
        expected, _ = self.masks([self.cloud(points), self.cloud(plane)])
        masks, _ = self.masks(clouds, transforms=[np.eye(4), pose], leveling=level)
        for actual, keep in zip(masks, expected):
            np.testing.assert_array_equal(actual, keep)
        np.testing.assert_array_equal(np.asarray(clouds[0].points), points)
        np.testing.assert_array_equal(np.asarray(clouds[1].points), local)

    def test_application_preserves_coordinates_colors_normals_and_inputs(self):
        plane, strip, _ = self.scene()
        points = np.vstack((plane, strip))
        cloud = self.cloud(points)
        colors = np.random.default_rng(17).uniform(0, 1, points.shape)
        normals = np.tile([0., 0., 1.], (len(points), 1))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        cloud.normals = o3d.utility.Vector3dVector(normals)
        filtered, report = _denoise_registered_views(
            [cloud, self.cloud(plane)], [np.eye(4)] * 2, np.eye(4),
            [True, True], .1, None, "synthetic",
        )
        np.testing.assert_array_equal(np.asarray(filtered[0].points), plane)
        np.testing.assert_array_equal(np.asarray(filtered[0].colors), colors[:len(plane)])
        np.testing.assert_array_equal(np.asarray(filtered[0].normals), normals[:len(plane)])
        np.testing.assert_array_equal(np.asarray(cloud.points), points)
        np.testing.assert_array_equal(np.asarray(cloud.colors), colors)
        self.assertEqual(report["total_removed_points"], len(strip))

    def test_less_than_two_nonempty_included_views_does_not_filter(self):
        plane, strip, _ = self.scene()
        cloud = self.cloud(np.vstack((plane, strip)))
        for clouds, included in (
            ([cloud], [True]),
            ([cloud, self.cloud(plane)], [True, False]),
            ([cloud, self.cloud(np.empty((0, 3)))], [True, True]),
            ([], []),
        ):
            with self.subTest(included=included, sizes=[len(c.points) for c in clouds]):
                masks, report = self.masks(clouds, included=included)
                self.assertTrue(all(mask.all() for mask in masks))
                self.assertFalse(report["enabled"])
                self.assertEqual(report["total_removed_points"], 0)

    def test_excluded_view_cannot_support_noise_or_be_filtered(self):
        plane, strip, _ = self.scene()
        clouds = [self.cloud(np.vstack((plane, strip))), self.cloud(plane), self.cloud(strip)]
        masks, report = self.masks(clouds, included=[True, True, False])
        self.assertFalse(masks[0][len(plane):].any())
        self.assertTrue(masks[2].all())
        self.assertEqual(len(report["views"]), 2)

    def test_close_support_from_one_peer_is_sufficient(self):
        plane, strip, _ = self.scene()
        clouds = [self.cloud(np.vstack((plane, strip))), self.cloud(plane), self.cloud(strip)]
        masks, _ = self.masks(clouds)
        self.assertTrue(all(mask.all() for mask in masks))

    def test_two_independent_views_can_reject_an_unsupported_return(self):
        plane, strip, _ = self.scene()
        masks, report = self.masks([
            self.cloud(np.vstack((plane, strip))),
            self.cloud(plane),
            self.cloud(plane),
        ])
        self.assertTrue(masks[0][:len(plane)].all())
        self.assertFalse(masks[0][len(plane):].any())
        self.assertGreaterEqual(
            report["views"][0]["independent_overlap_supported_points"],
            len(strip),
        )

    def test_post_filter_orphan_component_is_removed(self):
        plane, _, _ = self.scene()
        x, y = np.meshgrid(np.linspace(-.25, .25, 9), np.linspace(-.25, .25, 9))
        island = np.column_stack((x.ravel() + .1, y.ravel() + .1, np.full(x.size, .32)))
        support_x, support_y = np.meshgrid(
            np.linspace(-.18, .38, 4), np.linspace(-.18, .38, 4),
        )
        sparse_peer = np.column_stack((
            support_x.ravel(), support_y.ravel(), np.zeros(support_x.size),
        ))
        masks, report = self.masks([
            self.cloud(np.vstack((plane, island))),
            self.cloud(sparse_peer),
        ])
        self.assertTrue(masks[0][:len(plane)].all())
        self.assertFalse(masks[0][len(plane):].any())
        self.assertGreater(
            report["post_component_cleanup"]["removed_points"],
            0,
        )

    def test_supported_disconnected_component_is_preserved(self):
        plane, _, _ = self.scene()
        x, y = np.meshgrid(np.linspace(-.25, .25, 9), np.linspace(-.25, .25, 9))
        island = np.column_stack((x.ravel() + .1, y.ravel() + .1, np.full(x.size, .3)))
        clouds = [
            self.cloud(np.vstack((plane, island))),
            self.cloud(plane),
            self.cloud(plane),
        ]
        # The island is genuinely observed by the second and third views.
        clouds[1] += self.cloud(island)
        clouds[2] += self.cloud(island)
        masks, _ = self.masks(clouds)
        self.assertTrue(masks[0][-len(island):].all())

    def test_sparse_context_does_not_count_as_overlap(self):
        _, strip, _ = self.scene()
        masks, _ = self.masks([self.cloud(strip), self.cloud(np.array([[0, 0, 0.]]))])
        self.assertTrue(masks[0].all())

    def test_final_fusion_cleanup_removes_single_view_floating_island(self):
        x, y = np.meshgrid(np.linspace(-1.0, 1.0, 31), np.linspace(-1.0, 1.0, 31))
        main = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        ix, iy = np.meshgrid(np.linspace(3.0, 3.4, 7), np.linspace(0.0, 0.4, 7))
        island = np.column_stack((ix.ravel(), iy.ravel(), np.full(ix.size, 0.8)))
        points = np.vstack((main, island))
        cleaned, report = _final_fusion_component_cleanup(
            self.cloud(points),
            np.concatenate((
                np.zeros(len(main), dtype=np.int64),
                np.ones(len(island), dtype=np.int64),
            )),
            voxel_size=0.05,
        )
        self.assertEqual(len(cleaned.points), len(main))
        self.assertEqual(report["removed_points"], len(island))
        self.assertEqual(len(report["removed_components"]), 1)

    def test_final_fusion_cleanup_preserves_multi_view_island_and_unique_extent(self):
        x, y = np.meshgrid(np.linspace(-1.0, 1.0, 31), np.linspace(-1.0, 1.0, 31))
        main = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        ix, iy = np.meshgrid(np.linspace(3.0, 3.4, 7), np.linspace(0.0, 0.4, 7))
        island = np.column_stack((ix.ravel(), iy.ravel(), np.full(ix.size, 0.8)))
        extent = np.column_stack((
            np.linspace(5.0, 8.0, 80),
            np.zeros(80),
            np.zeros(80),
        ))
        points = np.vstack((main, island, extent))
        sources = np.concatenate((
            np.zeros(len(main), dtype=np.int64),
            np.tile(np.array([1, 2], dtype=np.int64), len(island) // 2 + 1)[:len(island)],
            np.ones(len(extent), dtype=np.int64),
        ))
        cleaned, report = _final_fusion_component_cleanup(
            self.cloud(points),
            sources,
            voxel_size=0.05,
            source_support_counts=np.concatenate((
                np.ones(len(main), dtype=np.int64),
                np.full(len(island), 2, dtype=np.int64),
                np.ones(len(extent), dtype=np.int64),
            )),
        )
        self.assertEqual(len(cleaned.points), len(points))
        self.assertEqual(report["removed_points"], 0)
        self.assertTrue(any(
            item["source_view_count"] >= 2
            for item in report["preserved_components"]
        ))

    def test_invalid_sizes_and_voxel_are_rejected(self):
        for voxel in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _cross_view_overlap_keep_masks([], [], np.eye(4), [], voxel)
        with self.assertRaises(ValueError):
            _cross_view_overlap_keep_masks([], [], np.eye(4), [True], .1)


if __name__ == "__main__":
    unittest.main()
