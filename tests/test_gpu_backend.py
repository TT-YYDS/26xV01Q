import unittest

import numpy as np
import open3d as o3d

from pointcloud_app.gpu import get_gpu_capabilities, gpu_refine_icp, gpu_voxel_down_sample


GPU_AVAILABLE = get_gpu_capabilities().available


@unittest.skipUnless(GPU_AVAILABLE, "需要可用的 CUDA GPU 才能执行 GPU 后端测试。")
class GpuBackendTests(unittest.TestCase):
    def test_gpu_voxel_down_sample_averages_points_and_colors(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.00],
                [0.02, 0.02, 0.02],
                [1.00, 0.00, 0.00],
            ],
            dtype=np.float64,
        )
        colors = np.array(
            [
                [1.00, 0.00, 0.00],
                [0.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ],
            dtype=np.float64,
        )

        sampled_points, sampled_colors = gpu_voxel_down_sample(points, colors, voxel_size=0.10)

        self.assertEqual(sampled_points.shape, (2, 3))
        self.assertIsNotNone(sampled_colors)
        np.testing.assert_allclose(sampled_points[0], [0.01, 0.01, 0.01], atol=1e-6)
        np.testing.assert_allclose(sampled_colors[0], [0.50, 0.50, 0.00], atol=1e-6)

    def test_gpu_point_to_plane_icp_recovers_translation(self) -> None:
        generator = np.random.default_rng(20260825)
        source_points = generator.uniform(-1.0, 1.0, size=(1_200, 3))
        expected_translation = np.array([0.04, -0.03, 0.02])
        target_points = source_points + expected_translation
        target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
        target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

        transformation, fitness, rmse = gpu_refine_icp(
            source_points,
            target_points,
            np.asarray(target_cloud.normals),
            max_correspondence_distance=0.25,
            max_iterations=35,
            use_point_to_plane=True,
        )

        np.testing.assert_allclose(transformation[:3, 3], expected_translation, atol=2e-3)
        self.assertGreater(fitness, 0.99)
        self.assertLess(rmse, 2e-3)


if __name__ == "__main__":
    unittest.main()