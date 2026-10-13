import unittest

import numpy as np
import open3d as o3d

from pointcloud_app.models import RegistrationConfig
from pointcloud_app.pipeline import _calibration_geometry_initial_correction


def _fixture_cloud() -> o3d.geometry.PointCloud:
    """Build a small asymmetric multi-plane calibration fixture."""
    grid = np.linspace(-2.0, 2.0, 51)
    narrow = np.linspace(-1.5, 1.5, 39)
    points = []

    # Horizontal faces.
    xx, yy = np.meshgrid(grid, narrow, indexing="xy")
    points.extend((np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 1.0))),))
    points.extend((np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, -1.0))),))

    # Side faces use the full height but deliberately different spans.
    yy, zz = np.meshgrid(narrow, np.linspace(-1.0, 1.0, 27), indexing="xy")
    points.extend((np.column_stack((np.full(yy.size, -2.0), yy.ravel(), zz.ravel())),))
    points.extend((np.column_stack((np.full(yy.size, 2.0), yy.ravel(), zz.ravel())),))

    xx, zz = np.meshgrid(grid, np.linspace(-1.0, 1.0, 27), indexing="xy")
    points.extend((np.column_stack((xx.ravel(), np.full(xx.size, -1.5), zz.ravel())),))

    coordinates = np.concatenate(points, axis=0)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coordinates))


def _rigid_transform(rotation_degrees: float, translation: tuple[float, float, float]) -> np.ndarray:
    radians = np.radians(rotation_degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transformation


def _transformed_copy(cloud: o3d.geometry.PointCloud, transformation: np.ndarray) -> o3d.geometry.PointCloud:
    copied = o3d.geometry.PointCloud(cloud)
    copied.transform(transformation)
    return copied


class CalibrationGeometryTests(unittest.TestCase):
    def _config(self, **overrides: object) -> RegistrationConfig:
        values = {
            "calibration_geometry_voxel_size": 0.08,
            "calibration_geometry_plane_distance": 0.035,
            "calibration_geometry_minimum_plane_points": 80,
            "calibration_geometry_minimum_planes": 3,
            "calibration_geometry_max_planes": 8,
            "calibration_geometry_max_normal_angle_degrees": 25.0,
            "calibration_geometry_max_centroid_distance": 4.0,
            "pose_prior_rigid_max_rotation_degrees": 2.0,
            "pose_prior_rigid_max_translation": 0.45,
        }
        values.update(overrides)
        return RegistrationConfig(**values)

    def test_geometry_recovers_small_rigid_correction(self) -> None:
        target = _fixture_cloud()
        expected = _rigid_transform(1.0, (0.12, -0.08, 0.06))
        source = _transformed_copy(target, np.linalg.inv(expected))

        correction, report = _calibration_geometry_initial_correction(
            source,
            target,
            self._config(),
        )

        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(int(report["matched_plane_count"]), 3)
        np.testing.assert_allclose(correction, expected, atol=0.08)
        self.assertLess(float(report["plane_rmse_after"]), float(report["plane_rmse_before"]))

    def test_plane_normal_sign_is_not_treated_as_a_flip(self) -> None:
        target = _fixture_cloud()
        expected = _rigid_transform(-0.8, (-0.10, 0.05, 0.03))
        source = _transformed_copy(target, np.linalg.inv(expected))

        correction, report = _calibration_geometry_initial_correction(
            source,
            target,
            self._config(),
        )

        self.assertEqual(report["status"], "ok")
        self.assertLess(float(report["correction_rotation_degrees"]), 2.0)
        np.testing.assert_allclose(correction[:3, 3], expected[:3, 3], atol=0.08)

    def test_insufficient_planes_falls_back_to_identity(self) -> None:
        coordinates = np.column_stack(
            (
                np.linspace(-2.0, 2.0, 400),
                np.linspace(-1.0, 1.0, 400),
                np.zeros(400),
            )
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coordinates))

        correction, report = _calibration_geometry_initial_correction(
            cloud,
            cloud,
            self._config(),
        )

        np.testing.assert_allclose(correction, np.eye(4), atol=1e-12)
        self.assertEqual(report["status"], "insufficient_planes")

    def test_large_correction_is_rejected(self) -> None:
        target = _fixture_cloud()
        expected = _rigid_transform(8.0, (0.8, 0.0, 0.0))
        source = _transformed_copy(target, np.linalg.inv(expected))

        correction, report = _calibration_geometry_initial_correction(
            source,
            target,
            self._config(
                pose_prior_rigid_max_rotation_degrees=1.0,
                pose_prior_rigid_max_translation=0.10,
            ),
        )

        np.testing.assert_allclose(correction, np.eye(4), atol=1e-12)
        self.assertIn(report["status"], {"rejected_rotation_bound", "rejected_translation_bound"})

    def test_disabled_geometry_returns_identity(self) -> None:
        cloud = _fixture_cloud()
        correction, report = _calibration_geometry_initial_correction(
            cloud,
            cloud,
            self._config(use_calibration_geometry_constraints=False),
        )

        np.testing.assert_allclose(correction, np.eye(4), atol=1e-12)
        self.assertEqual(report["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
