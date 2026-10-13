import unittest

import numpy as np

from pointcloud_app.pipeline import build_surface_preview


class SurfacePreviewTests(unittest.TestCase):
    def test_build_surface_preview_creates_triangles_for_a_plane_grid(self) -> None:
        x_values, y_values = np.meshgrid(
            np.linspace(-1.0, 1.0, 24),
            np.linspace(-1.0, 1.0, 24),
        )
        points = np.column_stack((x_values.ravel(), y_values.ravel(), 0.15 * x_values.ravel()))

        surface = build_surface_preview(
            points,
            voxel_size=0.05,
            maximum_points=2_000,
            maximum_triangles=5_000,
            compute_backend="cpu",
        )

        self.assertGreater(len(surface.vertices), 100)
        self.assertGreater(len(surface.triangles), 100)
        self.assertLessEqual(len(surface.triangles), 5_000)


if __name__ == "__main__":
    unittest.main()
