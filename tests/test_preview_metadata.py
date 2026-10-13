import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

from pointcloud_app.pipeline import build_preview


class PreviewMetadataTests(unittest.TestCase):
    def test_preview_reports_raw_denoised_and_display_counts(self) -> None:
        raw_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                np.array(
                    [
                        [0.00, 0.00, 0.00],
                        [0.10, 0.00, 0.00],
                        [0.20, 0.00, 0.00],
                        [0.30, 0.00, 0.00],
                    ],
                    dtype=np.float64,
                )
            )
        )
        denoised_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                np.array(
                    [
                        [0.00, 0.00, 0.00],
                        [0.20, 0.00, 0.00],
                        [0.30, 0.00, 0.00],
                    ],
                    dtype=np.float64,
                )
            )
        )

        with patch("pointcloud_app.pipeline.read_point_cloud", return_value=raw_cloud), patch(
            "pointcloud_app.pipeline.denoise_cloud", return_value=(denoised_cloud, None)
        ):
            preview = build_preview(
                [Path("sample.ply")],
                voxel_size=0.01,
                maximum_points=2,
                compute_backend="cpu",
                apply_denoise=True,
            )

        self.assertEqual(preview.input_points, 4)
        self.assertEqual(preview.after_denoise_points, 3)
        self.assertEqual(preview.before_limit_points, 3)
        self.assertEqual(len(preview.points), 2)


if __name__ == "__main__":
    unittest.main()
