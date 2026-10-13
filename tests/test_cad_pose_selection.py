import unittest

import numpy as np
from scipy.spatial import cKDTree

from media_analysis_core import _select_profile_pose, inverse_transform, transform
from pointcloud_app.native import figures


class CadPoseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.anchor = np.column_stack((np.linspace(-1, 1, 100), np.zeros(100)))
        self.tree = cKDTree(self.anchor)

    def candidate(self, score, angle=0., y=0., source="global_search", mirror=1.):
        return score, mirror, np.array([angle, 0., y]), source

    def test_wrong_selected_path_pose_does_not_override_better_datum(self):
        correct = self.candidate(.2)
        wrong = self.candidate(1.4, y=.2, source="selected_registration_path")
        selected, metrics = _select_profile_pose([wrong, correct], self.anchor, self.tree)
        self.assertIs(selected, correct)
        self.assertFalse(metrics[0]["eligible"])

    def test_outline_breaks_symmetric_datum_tie(self):
        wrong = self.candidate(1.4, angle=np.pi, source="selected_registration_path", mirror=-1.)
        correct = self.candidate(.2)
        selected, metrics = _select_profile_pose([wrong, correct], self.anchor, self.tree)
        self.assertIs(selected, correct)
        self.assertTrue(all(item["eligible"] for item in metrics))

    def test_better_outline_cannot_override_bad_datum(self):
        wrong = self.candidate(.01, y=.2)
        correct = self.candidate(.5, source="selected_registration_path")
        selected, _ = _select_profile_pose([wrong, correct], self.anchor, self.tree)
        self.assertIs(selected, correct)

    def test_p95_gate_rejects_partial_datum_match(self):
        anchor = np.vstack((self.anchor, [[1., y] for y in np.linspace(0, 1, 20)]))
        tree = cKDTree(anchor)
        correct = self.candidate(.5)
        wrong = self.candidate(.01, mirror=-1.)
        selected, metrics = _select_profile_pose([wrong, correct], anchor, tree)
        self.assertIs(selected, correct)
        self.assertFalse(metrics[0]["eligible"])

    def test_without_datum_uses_outline(self):
        correct = self.candidate(.1)
        selected, metrics = _select_profile_pose([self.candidate(.5), correct], None, self.tree)
        self.assertIs(selected, correct)
        self.assertEqual(metrics, [])

    def test_cad_display_inverse_preserves_angles_and_source_points(self):
        points = np.array([[0., 0.], [2., 0.], [2., 1.]])
        original = points.copy()
        for mirror in (1., -1.):
            mapped = transform(points, np.array([.4, .7]), mirror, -.82, np.array([3., -2.]))
            restored = inverse_transform(mapped, np.array([.4, .7]), mirror, -.82, np.array([3., -2.]))
            np.testing.assert_allclose(restored, points, atol=1e-12)
        np.testing.assert_array_equal(points, original)

    def test_default_cloud_marker_diameter(self):
        figure = figures.cloud(np.array([[0., 0., 0.], [1., 2., 3.]]))
        self.assertEqual(figure.data[0].marker.size, 1.5)


if __name__ == "__main__":
    unittest.main()
