import unittest
from pathlib import Path

import numpy as np

from app import _make_section_picker_figure, make_preview_figure
from pointcloud_app.models import ProcessingConfig
from pointcloud_app.pipeline import PreviewData


class PreviewProjectionTests(unittest.TestCase):
    @staticmethod
    def _preview_data() -> PreviewData:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 5.0, 2.5],
                [40.0, 10.0, 5.0],
            ],
            dtype=np.float32,
        )
        return PreviewData(
            points=points,
            colors=np.zeros_like(points),
            source_indices=np.arange(len(points)),
            input_points=len(points),
            after_denoise_points=len(points),
            before_limit_points=len(points),
        )

    def test_default_preview_is_orthographic_with_equal_coordinate_scale(self) -> None:
        figure = make_preview_figure(
            [Path("sample.ply")],
            ProcessingConfig(compute_backend="cpu"),
            preview_voxel_size=0.1,
            preview_maximum_points=100,
            preview_data=self._preview_data(),
            reference_file=Path("sample.ply"),
        )

        scene = figure.layout.scene
        self.assertEqual(scene.aspectmode, "manual")
        self.assertEqual(scene.camera.projection.type, "orthographic")
        self.assertAlmostEqual(scene.aspectratio.x, 1.0, places=6)
        self.assertAlmostEqual(scene.aspectratio.y, 0.25, places=6)
        self.assertAlmostEqual(scene.aspectratio.z, 0.125, places=6)

    def test_preview_can_switch_back_to_perspective_projection(self) -> None:
        figure = make_preview_figure(
            [Path("sample.ply")],
            ProcessingConfig(compute_backend="cpu"),
            preview_voxel_size=0.1,
            preview_maximum_points=100,
            preview_data=self._preview_data(),
            reference_file=Path("sample.ply"),
            use_orthographic_projection=False,
        )

        self.assertEqual(figure.layout.scene.camera.projection.type, "perspective")

    def test_section_picker_requires_explicit_selection_mode(self) -> None:
        preview_data = self._preview_data()
        config = ProcessingConfig(compute_backend="cpu").section

        browse_figure = _make_section_picker_figure(
            preview_data.points,
            config,
            selected_region=None,
            selection_enabled=False,
        )
        selection_figure = _make_section_picker_figure(
            preview_data.points,
            config,
            selected_region=None,
            selection_enabled=True,
        )

        self.assertEqual(browse_figure.layout.dragmode, "pan")
        self.assertEqual(selection_figure.layout.dragmode, "select")


if __name__ == "__main__":
    unittest.main()
