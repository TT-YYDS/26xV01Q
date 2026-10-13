from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app
from streamlit.testing.v1 import AppTest
from app import (
    _clear_pointcloud_page_state,
    _clear_section_region_picker,
    _normalize_section_offset_scan_rows,
    _pointcloud_data_signature,
    _selected_group_names_from_index,
    apply_family_profile,
    classify_group_family,
)
from pointcloud_app.models import Bounds2D, ProcessingConfig


class PointcloudPageStateTests(unittest.TestCase):
    def test_section_controls_disable_cleanup_in_simple_and_expert_modes(self) -> None:
        page = AppTest.from_string(
            "import streamlit as st\n"
            "import app\n"
            "st.session_state['config_under_test'] = app.build_processing_config()\n",
            default_timeout=30,
        ).run()
        for expert_mode in (False, True):
            with self.subTest(expert_mode=expert_mode):
                if expert_mode:
                    page.sidebar.checkbox(key="advanced_parameter_mode").set_value(True).run()
                self.assertEqual(len(page.exception), 0)
                config = page.session_state["config_under_test"]
                config.validate()
                self.assertFalse(config.section.use_statistical_filter)
                self.assertFalse(config.section.use_component_filter)
                self.assertEqual(config.section.component_cell_size, 0.30)
                self.assertTrue(config.denoise.use_statistical_filter)
                for control in (*page.checkbox, *page.number_input):
                    self.assertFalse(
                        "截面" in control.label
                        and any(word in control.label for word in ("去噪", "连通域")),
                        control.label,
                    )

    def test_group_family_classification_matches_media_layout(self) -> None:
        self.assertEqual(classify_group_family("1"), "c1")
        self.assertEqual(classify_group_family("4"), "c1")
        self.assertEqual(classify_group_family("木板"), "c2")
        self.assertEqual(classify_group_family("木板2"), "c2")
        self.assertEqual(classify_group_family("木板-2"), "c3")
        self.assertEqual(classify_group_family("木板-6"), "c3")
        self.assertEqual(classify_group_family("other"), "default")

    def test_family_profile_is_independent_and_preserves_section_geometry(self) -> None:
        config = ProcessingConfig(compute_backend="cpu")
        config.section.target_region = Bounds2D(-2.0, 2.0, -1.0, 1.0)

        numeric = apply_family_profile(config, "numeric")
        wood = apply_family_profile(config, "wood_base")

        self.assertEqual(config.denoise.voxel_size, 0.15)
        self.assertEqual(numeric.data_family, "c1")
        self.assertEqual(wood.data_family, "c2")
        self.assertEqual(numeric.profile_source, "auto")
        self.assertTrue(numeric.registration.use_section_alignment_validation)
        self.assertEqual(numeric.registration.minimum_calibration_icp_fitness, 0.65)
        self.assertEqual(numeric.registration.calibration_graph_maximum_loop_rmse, 0.16)
        self.assertEqual(numeric.registration.calibration_graph_max_correspondence_distance, 0.32)
        self.assertNotEqual(wood.registration, numeric.registration)
        self.assertNotEqual(wood.denoise, numeric.denoise)
        self.assertEqual(numeric.section.target_region.x_min, -2.0)
        self.assertIsNot(numeric.registration, wood.registration)

    def test_data_signature_changes_for_new_group_or_replaced_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_file = root / "1.ply"
            second_file = root / "2.ply"
            first_file.write_bytes(b"first")
            second_file.write_bytes(b"second")
            groups = {"1": [first_file], "2": [second_file]}

            first_signature = _pointcloud_data_signature(groups, ["1"])
            self.assertNotEqual(first_signature, _pointcloud_data_signature(groups, ["2"]))

            first_file.write_bytes(b"first-replaced")
            self.assertNotEqual(first_signature, _pointcloud_data_signature(groups, ["1"]))

    def test_single_group_selection_does_not_expand_to_other_groups(self) -> None:
        group_names = ["1", "2", "木板-2"]

        self.assertEqual(_selected_group_names_from_index(group_names, 1), ["1"])
        self.assertEqual(_selected_group_names_from_index(group_names, 3), ["木板-2"])
        self.assertEqual(_selected_group_names_from_index(group_names, 0), group_names)

    def test_invalid_group_selection_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _selected_group_names_from_index(["1", "2"], 4)

    def test_clearing_new_pointcloud_state_removes_old_page_results(self) -> None:
        state = {
            "pipeline_results": [object()],
            "pipeline_result_data_signature": "old-signature",
            "pipeline_completion_message": "done",
            "section_picker_group": "1",
            "section_picker_region": {"x_min": 0.0},
            "section_picker_locked": True,
            "section_picker_drawing": True,
            "section_picker_last_signature": (1, 2, 3),
            "section_picker_revision": 4,
            "pending_section_plane": {"angle_degrees": 10.0},
            "pending_section_offset": 13.0,
            "section_angle_degrees": 10.0,
            "section_tilt_degrees": 2.0,
            "section_plane_offset": 12.0,
            "cad-click-selection-old": {"x": [1], "y": [], "fit": []},
            "unrelated_setting": "keep",
        }

        _clear_pointcloud_page_state(state)

        self.assertNotIn("pipeline_results", state)
        self.assertNotIn("pipeline_result_data_signature", state)
        self.assertNotIn("pipeline_completion_message", state)
        self.assertNotIn("section_picker_region", state)
        self.assertNotIn("pending_section_plane", state)
        self.assertNotIn("pending_section_offset", state)
        self.assertNotIn("cad-click-selection-old", state)
        self.assertEqual(state["section_picker_revision"], 5)
        self.assertEqual(state["section_angle_degrees"], 0.0)
        self.assertEqual(state["section_tilt_degrees"], 0.0)
        self.assertEqual(state["section_plane_offset"], 9.0)
        self.assertEqual(state["unrelated_setting"], "keep")

    def test_clearing_section_region_removes_pending_offset(self) -> None:
        state = {
            "section_picker_revision": 2,
            "pending_section_plane": {"angle_degrees": 10.0},
            "pending_section_offset": -7.5,
        }

        with patch.object(app.st, "session_state", state):
            _clear_section_region_picker("1")

        self.assertNotIn("pending_section_plane", state)
        self.assertNotIn("pending_section_offset", state)
        self.assertEqual(state["section_plane_offset"], 9.0)

    def test_legacy_section_offset_scan_rows_gain_points_per_cell(self) -> None:
        legacy_rows = [
            {
                "plane_offset": -9.0,
                "point_count": 120,
                "component_count": 2,
                "cell_count": 30,
                "horizontal_span": 45.0,
                "vertical_span": 16.5,
                "largest_component_span": 42.0,
                "is_current": True,
            }
        ]

        rows = _normalize_section_offset_scan_rows(legacy_rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["plane_offset"], -9.0)
        self.assertEqual(rows[0]["point_count"], 120)
        self.assertEqual(rows[0]["cell_count"], 30)
        self.assertEqual(rows[0]["points_per_cell"], 4.0)
        self.assertEqual(rows[0]["largest_component_ratio"], 0.0)
        self.assertTrue(rows[0]["is_current"])

    def test_section_offset_scan_row_normalization_skips_invalid_rows(self) -> None:
        rows = _normalize_section_offset_scan_rows(
            [
                {"plane_offset": "not-a-number"},
                {"point_count": 10, "cell_count": 5},
                {
                    "plane_offset": 1.0,
                    "point_count": "invalid",
                    "cell_count": "invalid",
                    "points_per_cell": float("nan"),
                },
                "not-a-row",
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["point_count"], 0)
        self.assertEqual(rows[0]["cell_count"], 0)
        self.assertEqual(rows[0]["points_per_cell"], 0.0)


if __name__ == "__main__":
    unittest.main()
