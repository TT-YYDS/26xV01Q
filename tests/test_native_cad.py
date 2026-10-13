import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtWidgets import QMessageBox

from pointcloud_app.cad_geometry import CadEntity, normalize_cad_path_selection
from pointcloud_app.native.backend import default_config
from pointcloud_app.native.window import CAD_PAGE, STEPS, DesktopWindow, Group


class CadController:
    cad_mode = DesktopWindow.cad_mode
    cad_step = DesktopWindow.cad_step
    _apply_cad_state = DesktopWindow._apply_cad_state
    _cad_measurement_blocker = DesktopWindow._cad_measurement_blocker
    cad_mode_changed = DesktopWindow.cad_mode_changed
    cad_step_changed = DesktopWindow.cad_step_changed
    cad_single = DesktopWindow.cad_single
    cad_cancel = DesktopWindow.cad_cancel
    cad_undo = DesktopWindow.cad_undo
    cad_clear_current = DesktopWindow.cad_clear_current
    cad_clear = DesktopWindow.cad_clear
    cad_delete_path = DesktopWindow.cad_delete_path
    measure = DesktopWindow.measure
    open_cad = DesktopWindow.open_cad

    def __init__(self):
        self.state = Group([], default_config("1"))
        self.job = None
        self.log, self.error, self._draw = Mock(), Mock(), Mock()
        self.view_values = {}
        self.select_page = Mock()
        self.backend = Mock()
        self._start = lambda operation, callback: callback(operation(self.log))


class NativeCadTests(unittest.TestCase):
    def setUp(self):
        self.owner = CadController()
        self.state = self.owner.state
        self.state.dxf = Path("drawing.dxf")
        self.state.entities = [
            CadEntity(index, "LINE", np.array([[0., float(index)], [2., float(index)]]))
            for index in range(3)
        ]
        self.state.selection = normalize_cad_path_selection({
            "registration_paths": [[0]], "measurement_paths": [[1], [2]]})

    def test_import_completion_returns_to_existing_cad_page(self):
        self.state.selection = {}
        path = Path(__file__).resolve().parents[1] / "test母头.dxf"
        with patch("pointcloud_app.native.window.QFileDialog.getOpenFileName",
                   return_value=(str(path), "")):
            self.owner.open_cad()
        self.owner.select_page.assert_called_once_with(CAD_PAGE)
        self.assertLess(CAD_PAGE, len(STEPS))
        self.assertEqual(self.owner.view_values[CAD_PAGE], "paths")
        self.assertEqual(len(self.state.entities), 28)
        self.assertEqual(self.state.cad_mode, "registration")

    def test_failed_or_cancelled_import_preserves_existing_selection_and_report(self):
        self.state.report = {"result": "previous"}
        original = copy.deepcopy(self.state.selection)
        with patch("pointcloud_app.native.window.QFileDialog.getOpenFileName",
                   return_value=("", "")):
            self.owner.open_cad()
        with patch("pointcloud_app.native.window.QFileDialog.getOpenFileName",
                   return_value=("missing.dxf", "")), \
             patch("pointcloud_app.native.window.QMessageBox.question",
                   return_value=QMessageBox.StandardButton.Yes):
            with self.assertRaisesRegex(RuntimeError, "DXF"):
                self.owner.open_cad()
        self.assertEqual(self.state.selection, original)
        self.assertEqual(self.state.report, {"result": "previous"})
        self.assertEqual(self.state.dxf, Path("drawing.dxf"))
        self.owner.select_page.assert_not_called()

    def test_pending_endpoint_blocks_measurement_without_destroying_report(self):
        self.state.report = report = {"result": "previous"}
        self.owner._apply_cad_state(self.state.selection, 2)
        self.assertIn("未完成", self.owner._cad_measurement_blocker())
        self.assertIs(self.state.report, report)
        self.owner.cad_cancel()
        self.assertIsNone(self.state.pending)
        self.assertIs(self.state.report, report)

    def test_undo_restores_endpoint_role_and_group_modes_are_independent(self):
        self.state.pending = 0
        self.owner.cad_mode_changed("measurement")
        self.assertIsNone(self.state.pending)
        self.owner.cad_undo()
        self.assertEqual(self.state.pending, 0)
        self.assertEqual(self.owner.cad_mode, "registration")
        self.owner.state = Group([], default_config("2"), cad_mode="measurement")
        self.assertEqual(self.owner.cad_mode, "measurement")
        self.owner.state = self.state
        self.assertEqual(self.owner.cad_mode, "registration")

    def test_conflicting_single_path_has_no_undo_entry_or_result_invalidation(self):
        self.state.report = {"result": "previous"}
        self.state.cad_mode = "measurement"
        self.state.pending = 0
        self.owner.cad_single()
        self.assertEqual(self.state.pending, 0)
        self.assertEqual(self.state.undo, [])
        self.assertIsNotNone(self.state.report)

    def test_current_type_clear_and_single_path_delete_can_be_undone(self):
        original = copy.deepcopy(self.state.selection)
        self.state.report = report = {"result": "previous"}
        self.state.cad_mode = "measurement"
        self.owner.cad_clear_current()
        self.assertEqual(self.state.selection["registration"], [0])
        self.assertEqual(self.state.selection["measurement"], [])
        self.assertIsNone(self.state.report)
        self.owner.cad_undo()
        self.assertEqual(self.state.selection, original)
        self.assertIs(self.state.report, report)
        self.owner.cad_paths = SimpleNamespace(currentData=lambda: ("measurement_paths", 0))
        self.owner.cad_delete_path()
        self.assertEqual(self.state.selection["measurement_paths"], [[2]])
        self.owner.cad_undo()
        self.assertEqual(self.state.selection, original)

    def test_sampling_change_invalidates_report_and_undo_restores_parameters(self):
        self.state.report = report = {"result": "previous"}
        self.owner.cad_step_changed(.1)
        self.assertEqual(self.owner.cad_step, .1)
        self.assertIsNone(self.state.report)
        self.owner.cad_undo()
        self.assertEqual(self.owner.cad_step, .05)
        self.assertIs(self.state.report, report)

    def test_readiness_checks_path_continuity_and_actual_section_availability(self):
        self.assertIn("没有处理结果", self.owner._cad_measurement_blocker())
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "section.ply"
            path.touch()
            self.state.result = SimpleNamespace(section_cloud_path=path, section_points=100)
            self.assertEqual(self.owner._cad_measurement_blocker(), "")
            self.state.result.section_points = 0
            self.assertIn("不足", self.owner._cad_measurement_blocker())
            self.state.result.section_points = 100
            self.state.selection = normalize_cad_path_selection({
                "registration_paths": [[0, 2]], "measurement_paths": [[1]]})
            self.assertIn("不连续", self.owner._cad_measurement_blocker())

    def test_measurement_completion_uses_cad_page_and_selected_sampling(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "section.ply"
            path.touch()
            self.state.result = SimpleNamespace(section_cloud_path=path, section_points=100)
            self.state.cad_step = .125
            self.owner.backend.analyze.return_value = report = {
                "measurement_quality": {"status": "INSUFFICIENT_DATA"}}
            self.owner.measure()
            self.owner.select_page.assert_called_once_with(CAD_PAGE)
            self.assertEqual(self.owner.view_values[CAD_PAGE], "source_overlay")
            self.assertEqual(self.owner.backend.analyze.call_args.args[3], .125)
            self.assertIs(self.state.report, report)

    def test_source_overlay_routes_through_read_only_source_trace(self):
        report = {"measurement_quality": {"status": "PASS"}}
        result = object()
        self.state.report, self.state.result = report, result
        sources = [("1.ply", np.array([0]))]
        self.owner.backend.cad_sources.return_value = sources
        self.owner._figure_task = lambda create: create(self.owner.log)
        with patch("pointcloud_app.native.window.figures.cad_overlay") as overlay:
            DesktopWindow._draw_measurement(self.owner, "source_overlay")
        self.owner.backend.cad_sources.assert_called_once_with(result, report, self.owner.log)
        overlay.assert_called_once_with(report, sources, measurement_lines=True)
        self.assertIs(self.state.report, report)

    def test_combined_overlay_without_report_shows_empty_state(self):
        self.state.report = None
        self.owner._empty = Mock()
        self.owner._figure_task = Mock()
        DesktopWindow._draw_measurement(self.owner, "source_overlay")
        self.owner._empty.assert_called_once_with("暂无测量结果")
        self.owner._figure_task.assert_not_called()
        self.owner.backend.cad_sources.assert_not_called()
        self.owner.backend.analyze.assert_not_called()

    def test_measurement_overlay_preserves_station_gaps_and_cartesian_deviations(self):
        import csv
        import ezdxf
        from pointcloud_app.native import figures

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            section = root / "inputs" / "vertical_section" / "section.ply"
            section.parent.mkdir(parents=True)
            (section.parent.parent / "processing_report.json").write_text(json.dumps({
                "vertical_section": {"horizontal_xyz": [1, 0, 0], "vertical_xyz": [0, 1, 0]}}),
                encoding="utf-8")
            dxf = root / "drawing.dxf"
            document = ezdxf.new()
            document.modelspace().add_line((9, 18), (11, 20))
            document.saveas(dxf)
            path = root / "stations.csv"
            rows = [
                dict(station=.025, cad_x=9, cad_y=18, pointcloud_x=10, pointcloud_y=20,
                     signed_normal_deviation=.1, valid=True),
                dict(station=.075, cad_x=10, cad_y=19, pointcloud_x=99, pointcloud_y=99,
                     signed_normal_deviation=None, valid=False),
                dict(station=.125, cad_x=11, cad_y=20, pointcloud_x=13, pointcloud_y=24,
                     signed_normal_deviation=.2, valid=True),
            ]

            def save_rows():
                with path.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            save_rows()
            report = {"source_pointcloud": str(section), "source_dxf": str(dxf),
                      "fit": {"cad_transform": {"center": [1, 2], "mirror": -1,
                                                "angle_rad": .5, "translation": [2, 3]}},
                      "measurement_coordinate_system": "cad_original_xy",
                      "measurements": {"path": {"csv_path": str(path), "label": "测量路径 1"}}}
            original = copy.deepcopy(report)
            points = np.array([[1, 2, 0], [3, 4, 0]], dtype=float)
            with patch("pointcloud_app.pipeline.read_point_cloud",
                       return_value=SimpleNamespace(points=points)):
                plain = figures.cad_overlay(report)
                overlay = figures.cad_overlay(report, measurement_lines=True)
                np.testing.assert_array_equal(overlay.data[0].x, plain.data[0].x)
                np.testing.assert_array_equal(overlay.data[0].y, plain.data[0].y)
                self.assertEqual(overlay.layout.uirevision, plain.layout.uirevision)
                trace = overlay.data[-1]
                self.assertEqual(tuple(trace.x), (10., None, 13.))
                self.assertEqual(tuple(trace.y), (20., None, 24.))
                self.assertFalse(trace.connectgaps)
                np.testing.assert_array_equal(trace.customdata[0], [.025, .1, 1., 2.])
                np.testing.assert_array_equal(trace.customdata[2], [.125, .2, 2., 4.])
                self.assertIn("X 偏差", trace.hovertemplate)
                combined = figures.cad_overlay(report, [("1.ply", [0]), ("2.ply", [1])],
                                               measurement_lines=True)
                for index in range(2):
                    self.assertEqual(combined.data[index].x[0], plain.data[0].x[index])
                    self.assertEqual(combined.data[index].y[0], plain.data[0].y[index])
                    self.assertEqual(combined.data[index].legendgroup, f"source-{index}")
                    self.assertLess(combined.data[index].marker.opacity, 1)
                self.assertEqual(tuple(combined.data[-1].x), tuple(trace.x))
                self.assertEqual(tuple(combined.data[-1].y), tuple(trace.y))
                self.assertNotEqual(combined.data[-1].legendgroup, combined.data[0].legendgroup)
                self.assertEqual(combined.layout.title.text, "来源与测量代表线")
                self.assertEqual(report, original)

                old_report = copy.deepcopy(report)
                del old_report["measurement_coordinate_system"]
                with self.assertRaisesRegex(ValueError, "坐标系"):
                    figures.cad_overlay(old_report, measurement_lines=True)
                rows[0]["pointcloud_x"] = float("nan")
                save_rows()
                with self.assertRaisesRegex(ValueError, "非有限"):
                    figures.cad_overlay(report, measurement_lines=True)
                for row in rows:
                    row["valid"] = False
                save_rows()
                empty = figures.cad_overlay(report, measurement_lines=True)
                self.assertTrue(all(value is None for value in empty.data[-1].x))
                self.assertEqual(empty.layout.annotations[0].text, "无有效测量站位")
                path.unlink()
                with self.assertRaisesRegex(ValueError, "缺少站位 CSV"):
                    figures.cad_overlay(report, measurement_lines=True)

    def test_source_overlay_preserves_all_coordinates_and_rejects_partial_indices(self):
        import ezdxf
        from pointcloud_app.native import figures

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            section = root / "inputs" / "vertical_section" / "section.ply"
            section.parent.mkdir(parents=True)
            processing = {"vertical_section": {"horizontal_xyz": [1, 0, 0],
                                                "vertical_xyz": [0, 1, 0]}}
            (section.parent.parent / "processing_report.json").write_text(
                json.dumps(processing), encoding="utf-8")
            dxf = root / "drawing.dxf"
            document = ezdxf.new()
            document.modelspace().add_line((0, 0), (10, 0))
            document.saveas(dxf)
            report = {"source_pointcloud": str(section), "source_dxf": str(dxf),
                      "fit": {"cad_transform": {"center": [1, 2], "mirror": -1,
                                                "angle_rad": .5, "translation": [2, 3]}}}
            original_report = copy.deepcopy(report)
            points = np.array([[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0]], dtype=float)
            original_points = points.copy()
            sources = [("1.ply", [0, 2]), ("2.ply", [1]), ("未追溯", [3])]
            with patch("pointcloud_app.pipeline.read_point_cloud",
                       return_value=SimpleNamespace(points=points)):
                plain = figures.cad_overlay(report)
                colored = figures.cad_overlay(report, sources)
                for trace, (_, indices) in zip(colored.data, sources):
                    np.testing.assert_array_equal(trace.x, np.asarray(plain.data[0].x)[indices])
                    np.testing.assert_array_equal(trace.y, np.asarray(plain.data[0].y)[indices])
                    self.assertIn("%{x:.5f}", trace.hovertemplate)
                self.assertNotEqual(colored.data[0].marker.color, colored.data[1].marker.color)
                self.assertEqual(sum(len(trace.x) for trace in colored.data[:3]), len(points))
                self.assertEqual(plain.layout.uirevision, colored.layout.uirevision)
                with self.assertRaisesRegex(ValueError, "未覆盖"):
                    figures.cad_overlay(report, [("1.ply", [0, 1])])
                for bad in ([0, 0, 1, 2, 3], [0, 1, 2, 4], [-1, 0, 1, 2]):
                    with self.assertRaisesRegex(ValueError, "索引"):
                        figures.cad_overlay(report, [("1.ply", bad)])
            self.assertEqual(report, original_report)
            np.testing.assert_array_equal(points, original_points)


if __name__ == "__main__":
    unittest.main()
