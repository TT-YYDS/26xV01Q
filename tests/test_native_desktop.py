import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pointcloud_app.models import Bounds2D, Bounds3D, ProcessingConfig, SectionConfig
from pointcloud_app.desktop_runtime import load_station_mapping, save_station_mapping, STATIONS
from pointcloud_app.native.backend import (
    Backend, acquisition_signature, config_from_dict, default_config, fit_selection, processing_quality,
)
from pointcloud_app.native import figures
from pointcloud_app.native.window import CAD_PAGE, STEPS


class NativeBackendTests(unittest.TestCase):
    def test_region_figure_restores_editable_selection(self):
        section = SectionConfig(target_region=Bounds2D(-7, 9, -2.5, 3.5))
        points = np.array([[0., 0., 0.], [1., 1., 1.]])
        figure = figures.cloud(points, section, projection="XY")
        box = figure.layout.selections[0]
        self.assertEqual((box.x0, box.x1, box.y0, box.y1), (-7, 9, -2.5, 3.5))
        self.assertFalse(figure.layout.shapes)
        self.assertFalse(figures.cloud(points, section).layout.selections)

    def test_cad_page_index_matches_navigation_and_is_not_a_toolbar_index(self):
        self.assertEqual(CAD_PAGE, 4)
        self.assertEqual(STEPS[CAD_PAGE], "CAD")

    def test_processing_timing_records_stage_seconds(self):
        from pointcloud_app.pipeline import ProcessingTiming, _timed_progress_callback

        messages = []
        timing = ProcessingTiming()
        progress = _timed_progress_callback(messages.append, timing)
        progress("读取并去噪 1/2")
        progress("应用位姿并融合")
        progress("从已融合点云提取三维法向平面截面")
        progress("保存点云和截面图")
        summary = timing.finish()
        self.assertEqual(messages, [
            "读取并去噪 1/2",
            "应用位姿并融合",
            "从已融合点云提取三维法向平面截面",
            "保存点云和截面图",
        ])
        self.assertIn("single_view_denoise", summary["stages"])
        self.assertIn("fusion", summary["stages"])
        self.assertIn("section", summary["stages"])
        self.assertIn("output_rendering", summary["stages"])
        self.assertGreaterEqual(summary["total_seconds"], 0)

    def test_elapsed_time_format_is_stable(self):
        from pointcloud_app.native.window import DesktopWindow

        self.assertEqual(DesktopWindow._format_elapsed(0), "00:00:00")
        self.assertEqual(DesktopWindow._format_elapsed(3661.9), "01:01:01")

    def test_group_selector_refreshes_cloud_page_in_place(self):
        from unittest.mock import Mock
        from pointcloud_app.native.window import DesktopWindow

        owner = SimpleNamespace(
            groups={"1": object(), "2": object()}, group_name="1", page=1,
            group_combo=Mock(), nav=[Mock() for _ in STEPS], title=Mock(),
            view_epoch=0, canvas=Mock(), _toolbar=Mock(), _draw=Mock(),
            _showing_automatic_measurements=Mock(return_value=False), job=None,
        )
        owner.select_page = lambda page: DesktopWindow.select_page(owner, page)
        DesktopWindow.change_group(owner, "2")
        self.assertEqual(owner.group_name, "2")
        self.assertEqual(owner.page, 1)
        owner._draw.assert_called_once_with()
        owner.group_combo.setToolTip.assert_called_once_with("2")

        owner.page = 2
        DesktopWindow.change_group(owner, "1")
        self.assertEqual(owner.page, 0)

    def test_numeric_radius_rejects_sparse_returns_but_preserves_dense_faces(self):
        import open3d as o3d
        from pointcloud_app.pipeline import denoise_cloud

        x, y = np.meshgrid(np.arange(0, 4, .08), np.arange(0, 4, .08))
        face = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        x, y = np.meshgrid(np.arange(0, 3, .16), np.arange(0, 3, .16))
        noise = np.c_[x.ravel() + 10, y.ravel(), np.zeros(x.size)]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((face, noise))))
        config = default_config("1").denoise
        self.assertEqual(config.radius, .20)
        config.use_component_filter = False
        config.use_statistical_filter = False
        old = copy.deepcopy(config)
        old.radius = .30
        previous, _ = denoise_cloud(cloud, old, "synthetic")
        current, _ = denoise_cloud(cloud, config, "synthetic")
        before, after = np.asarray(previous.points), np.asarray(current.points)
        self.assertGreater(np.sum(before[:, 0] > 9), 100)
        self.assertEqual(np.sum(after[:, 0] > 9), 0)
        self.assertGreater(np.sum(after[:, 0] < 5), .90 * np.sum(before[:, 0] < 5))

    def test_section_mode_switch_after_cloud_toolbar_widgets_are_deleted(self):
        from unittest.mock import Mock
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication, QComboBox, QHBoxLayout
        from shiboken6 import isValid
        from pointcloud_app.native.window import DesktopWindow

        app = QApplication.instance() or QApplication([])
        old_selector = QComboBox()
        tools = QHBoxLayout()
        tools.addWidget(old_selector)
        owner = SimpleNamespace(
            tools=tools, single_view=old_selector, page=2, job=None, workflow_mode="manual",
            state=SimpleNamespace(files=[Path("scan.ply")]),
            view_values={}, view_epoch=0, canvas=Mock(), _draw=Mock(),
            _button=Mock(), _busy=Mock(), parameters=Mock(), process=Mock(),
        )
        owner._update_single_view_enabled = lambda: DesktopWindow._update_single_view_enabled(owner)
        owner.change_view = lambda: DesktopWindow.change_view(owner)
        owner._views = lambda options: DesktopWindow._views(owner, options)
        DesktopWindow._toolbar(owner)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertFalse(isValid(old_selector))
        self.assertIsNone(owner.single_view)
        for mode in ("region", "slice", "final", "raw", "neighbors", "plane"):
            owner.view.setCurrentIndex(owner.view.findData(mode))
            self.assertEqual(owner.view_values[2], mode)
        self.assertEqual(owner._draw.call_count, 6)
        self.assertEqual(owner.canvas.invalidate.call_count, 6)
        owner.view.deleteLater()
        app.processEvents()

    def test_fusion_removes_final_isolated_points_even_when_consensus_rejects_them(self):
        import open3d as o3d
        from pointcloud_app.pipeline import (
            _merge_clouds_with_section_consensus, _single_view_denoise_config,
        )

        config = _single_view_denoise_config(default_config("1"))
        points = np.array([[0., 0., 0.], [0., .2, 0.], [0., 4., 0.]])
        clouds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))]
        section = SectionConfig(
            angle_degrees=0., plane_offset=0., thickness=.1,
            horizontal_min=-10., horizontal_max=10.,
            vertical_min=-10., vertical_max=10.,
        )
        with patch("pointcloud_app.pipeline._local_consensus_keep_masks",
                   side_effect=lambda parts, *a, **kw: [
                       np.zeros(len(part.points), dtype=bool) for part in parts
                   ]):
            with patch("pointcloud_app.pipeline.denoise_cloud",
                       side_effect=AssertionError("fusion must not denoise")):
                merged, report = _merge_clouds_with_section_consensus(
                    [Path("unused.ply")], [np.eye(4)], [True], np.eye(4),
                    section, .01, "cpu", source_clouds=clouds,
                    coverage_preserving=config.registration.coverage_preserving_fusion,
                )
        self.assertEqual(len(merged.points), len(points) - 1)
        for point in points:
            present = np.any(np.all(np.asarray(merged.points) == point, axis=1))
            self.assertEqual(present, not np.array_equal(point, [0., 4., 0.]))
        self.assertFalse(report["filter_applied"])
        self.assertTrue(report["post_fusion_denoise_applied"])
        self.assertGreater(report["final_component_cleanup"]["removed_points"], 0)
        self.assertEqual(report["removed_section_points"], 0)
        self.assertGreater(report["diagnostic_removed_section_points"], 0)

    def test_processing_limits_denoise_to_single_views_without_mutating_settings(self):
        from pointcloud_app.pipeline import _single_view_denoise_config

        config = default_config("1")
        config.registration.coverage_preserving_fusion = False
        config.section.use_statistical_filter = True
        config.section.use_component_filter = True
        original = copy.deepcopy(config.as_dict())
        effective = _single_view_denoise_config(config)
        self.assertTrue(effective.registration.coverage_preserving_fusion)
        self.assertFalse(effective.section.use_statistical_filter)
        self.assertFalse(effective.section.use_component_filter)
        self.assertEqual(effective.denoise, config.denoise)
        self.assertEqual(config.as_dict(), original)

    def test_single_preview_filters_roi_and_separates_denoise_cache(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.ply"
            path.touch()
            config = default_config("1")
            original = copy.deepcopy(config.as_dict())
            backend = Backend()
            with patch("pointcloud_app.native.backend.build_preview", return_value=object()) as preview:
                first = backend.single_preview(path, config, 1, True, lambda m: None)
                self.assertIs(first, backend.single_preview(path, config, 1, True, lambda m: None))
                self.assertEqual(preview.call_count, 1)
                self.assertTrue(preview.call_args.args[5])
                self.assertIsNone(preview.call_args.kwargs.get("protected_bounds"))
                backend.single_preview(path, config, 1, False, lambda m: None)
                self.assertEqual(preview.call_count, 2)
                self.assertFalse(preview.call_args.args[5])
            self.assertEqual(config.as_dict(), original)

    def test_inspection_preview_separates_roi_cache_and_keeps_leveling(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.ply"
            path.touch()
            config = default_config("1")
            original = copy.deepcopy(config.as_dict())
            backend = Backend()
            leveling = object()
            with patch("pointcloud_app.native.backend.select_reference_file_with_leveling",
                       return_value=(path, leveling)) as select, \
                 patch("pointcloud_app.native.backend.build_preview",
                       side_effect=lambda *a, **kw: object()) as build:
                protected = backend.preview([path], config, 1, True, lambda m: None)
                self.assertEqual(build.call_args.kwargs["protected_bounds"],
                                 (config.calibration.bounds.min_bound, config.calibration.bounds.max_bound))
                inspection = backend.preview(
                    [path], config, 1, True, lambda m: None, protect_calibration=False)
                self.assertIsNot(protected[1], inspection[1])
                self.assertIsNone(build.call_args.kwargs["protected_bounds"])
                self.assertIs(build.call_args.args[6], leveling)
                self.assertIs(select.call_args.args[3], config.calibration)
                self.assertEqual(build.call_count, 2)
                self.assertIs(inspection, backend.preview(
                    [path], config, 1, True, lambda m: None, protect_calibration=False))
                self.assertIs(protected, backend.preview([path], config, 1, True, lambda m: None))
                self.assertEqual(build.call_count, 2)
                backend.preview([path], config, 1, False, lambda m: None, protect_calibration=False)
                self.assertFalse(build.call_args.args[5])
                self.assertIs(build.call_args.args[6], leveling)
                self.assertEqual(build.call_count, 3)
                config.calibration.use_roi = False
                backend.preview([path], config, 1, True, lambda m: None)
                self.assertIsNone(build.call_args.kwargs["protected_bounds"])
                config.calibration.use_roi = True
            self.assertEqual(config.as_dict(), original)

    def test_cloud_dropdown_filters_roi_but_section_preview_protects_it(self):
        from unittest.mock import Mock
        from pointcloud_app.native.window import DesktopWindow

        files = [Path("first.ply"), Path("selected.ply")]
        for page, mode in ((1, "single"), (2, "plane"), (2, "region")):
            with self.subTest(page=page, mode=mode):
                preview = SimpleNamespace(points=np.array([[0., 0., 0.], [1., 1., 1.]]))
                backend = Mock()
                backend.preview.return_value = (files[-1], preview)
                owner = SimpleNamespace(
                    job=None, state=SimpleNamespace(files=files, config=default_config("1"), result=None),
                    page=page, view=Mock(), single_view=Mock(), view_epoch=0,
                    level=1, apply_denoise=True, backend=backend, point_size=1.5,
                    _showing_automatic_measurements=lambda: False, _empty=Mock(), error=Mock(),
                )
                owner.view.currentData.return_value = mode
                owner.single_view.currentData.return_value = "view:1"
                owner._start = lambda create, done: create(lambda m: None)
                DesktopWindow._draw(owner)
                owner.error.assert_not_called()
                backend.preview.assert_called_once()
                self.assertEqual(backend.preview.call_args.args[0], [files[-1]] if page == 1 else files)
                self.assertEqual(backend.preview.call_args.kwargs["protect_calibration"], page != 1)

    def test_inspection_preview_removes_isolated_roi_cluster_and_keeps_faces(self):
        import open3d as o3d
        from pointcloud_app.pipeline import build_preview

        x, y = np.meshgrid(np.arange(0, 4, .05), np.arange(0, 4, .05))
        fixture = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        workpiece = fixture + [-10, 0, 0]
        rng = np.random.default_rng(81)
        noise = rng.uniform([1, 1, 2.5], [2, 2, 3.5], (90, 3))
        points = np.vstack((fixture, workpiece, noise))
        config = default_config("1")
        config.compute_backend = "cpu"
        config.calibration.bounds = Bounds3D(-.5, 4.5, -.5, 4.5, -1, 4)
        transform = np.eye(4)
        transform[2, 3] = 7
        leveling = SimpleNamespace(applied=True, transformation=transform)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.ply"
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
            clean_path = Path(directory) / "clean.ply"
            clean_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((fixture, workpiece))))
            self.assertTrue(o3d.io.write_point_cloud(str(clean_path), clean_cloud))
            clean_preview = build_preview([clean_path], .1, 100_000, "cpu", config.denoise, True, leveling)
            backend = Backend()
            with patch("pointcloud_app.native.backend.select_reference_file_with_leveling",
                       return_value=(path, leveling)):
                _, protected = backend.preview([path], config, 1, True, lambda m: None)
                _, inspection = backend.preview(
                    [path], config, 1, True, lambda m: None, protect_calibration=False)
                _, raw = backend.preview(
                    [path], config, 1, False, lambda m: None, protect_calibration=False)
            for data in (protected, raw):
                self.assertGreater(np.sum(data.points[:, 2] > 9), 50)
            self.assertEqual(np.sum(inspection.points[:, 2] > 9), 0)
            for mask in (lambda p: p[:, 0] > -.5, lambda p: p[:, 0] < -5):
                expected = int(mask(clean_preview.points).sum())
                self.assertGreater(expected, 700)
                self.assertGreater(mask(inspection.points).sum(), .9 * expected)
            np.testing.assert_allclose(inspection.points[:, 2], 7, atol=1e-6)
            self.assertIs(inspection.leveling, protected.leveling)

    def test_inspection_option_does_not_redenoise_fused_results(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "merged.ply"
            path.touch()
            result = SimpleNamespace(merged_cloud_path=path)
            with patch("pointcloud_app.native.backend.build_preview", return_value=object()) as build, \
                 patch("pointcloud_app.native.backend.select_reference_file_with_leveling") as select:
                Backend().preview([], default_config("1"), 1, True, lambda m: None,
                                  result, protect_calibration=False)
            select.assert_not_called()
            self.assertFalse(build.call_args.args[5])
            self.assertIsNone(build.call_args.args[6])

    def test_config_roundtrip_all_fields(self):
        for name in ("1", "木板", "木板-4", "木板2"):
            config = default_config(name)
            self.assertEqual(config.as_dict(), config_from_dict(config.as_dict()).as_dict())
            self.assertFalse(config.section.auto_offset_enabled)
            self.assertFalse(config.registration.use_six_station_pose_prior)
            self.assertTrue(config.registration.use_global_registration)

    def test_config_rejects_unknown_nonfinite_and_invalid_values(self):
        for update in ({"unknown": 1}, {"fusion_voxel_size": float("nan")},
                       {"fusion_voxel_size": -1}, {"compute_backend": "fake"}):
            payload = ProcessingConfig().as_dict()
            payload.update(update)
            with self.assertRaises(ValueError):
                config_from_dict(payload)

    def test_region_uses_xyz_normal_and_keeps_original_points(self):
        rng = np.random.default_rng(21)
        t = np.linspace(-12, 12, 2000)
        points = np.c_[t, t * .2 + rng.normal(0, .05, len(t)), t * .3]
        original = points.copy()
        config = SectionConfig()
        section, count, confidence = fit_selection(points, config, [-8, 8, -4, 4])
        self.assertGreater(count, 30)
        self.assertGreater(abs(section.tilt_degrees), 5)
        self.assertFalse(section.auto_offset_enabled)
        self.assertIsNone(config.target_region)
        np.testing.assert_array_equal(points, original)

    def test_region_rejects_empty_nonfinite_and_reversed_zero_span(self):
        points = np.ones((20, 3))
        for bounds in ([0, 2, 0, 2], [0, float("nan"), 0, 2], [1, 1, 0, 2], []):
            with self.assertRaises(ValueError):
                fit_selection(points, SectionConfig(), bounds)

    def test_manifest_requires_unique_station_and_exact_record_count(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"{i}.ply" for i in range(5)]
            for path in files:
                path.touch()
            mapping = dict(zip([p.name for p in files], STATIONS[:5]))
            manifest = save_station_mapping(files, mapping)
            self.assertEqual(load_station_mapping(files), mapping)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["views"].append(data["views"][0])
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_station_mapping(files)

    def test_history_cannot_claim_formal_pass_without_confirmation(self):
        base = dict(status="PASS", formal_measurement_allowed=True, section_points=3000,
                    failure_reasons=[], insufficient_data_reasons=[])
        with patch("pointcloud_app.downstream_bridge._derive_processing_quality", return_value=base):
            quality = processing_quality({})
            self.assertEqual(quality["status"], "INSUFFICIENT_DATA")
            self.assertFalse(quality["formal_measurement_allowed"])
            self.assertTrue(quality["insufficient_data_reasons"])
            confirmed = processing_quality({"desktop_acquisition": {"confirmed": True}})
            self.assertEqual(confirmed["status"], "PASS")
        self.assertEqual(base["status"], "PASS")

    def test_preview_cache_survives_section_edit_but_not_file_change(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "merged.ply"
            path.write_bytes(b"initial")
            backend = Backend()
            config = ProcessingConfig()
            result = SimpleNamespace(merged_cloud_path=path)
            with patch("pointcloud_app.native.backend.build_preview", return_value=object()) as preview:
                first = backend.preview([], config, 0, False, lambda m: None, result)
                config.section.plane_offset += 5
                second = backend.preview([], config, 0, False, lambda m: None, result)
                self.assertIs(first, second)
                self.assertEqual(preview.call_count, 1)
                path.write_bytes(b"modified input")
                backend.preview([], config, 0, False, lambda m: None, result)
                self.assertEqual(preview.call_count, 2)

    def test_alignment_matches_by_filename_and_levels_once(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            a, b = directory / "a.ply", directory / "b.ply"
            a.touch()
            b.touch()
            level = np.eye(4)
            level[2, 3] = 5
            transform = np.eye(4)
            transform[0, 3] = 10
            summary = {
                "input_files": [str(a), str(b)],
                "config": ProcessingConfig().as_dict(),
                "leveling": {"transformation": level.tolist()},
                "registration_metrics": [
                    {"file_name": b.name, "transformation": transform.tolist()},
                    {"file_name": a.name, "transformation": np.eye(4).tolist(),
                     "included_in_fusion": False}],
            }
            path = directory / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            with patch("pointcloud_app.native.backend.read_point_cloud",
                       side_effect=lambda p: SimpleNamespace(points=np.array([[1 if p == a else 2, 0, 0]]))):
                with patch("pointcloud_app.native.backend.denoise_cloud",
                           side_effect=lambda cloud, *args: (cloud, None)):
                    items = Backend().alignment(SimpleNamespace(summary_path=path), lambda m: None)
            np.testing.assert_array_equal(items[0][1], [[12, 0, 5]])
            np.testing.assert_array_equal(items[1][1], [[1, 0, 5]])
            self.assertFalse(items[1][2])

    def test_cad_sources_require_unchanged_inputs_and_unique_rigid_matches(self):
        from pointcloud_app.native.backend import sha256

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b, section = [root / name for name in ("a.ply", "b.ply", "section.ply")]
            for path in (a, b, section):
                path.write_bytes(path.name.encode())
            level = np.eye(4)
            level[2, 3] = 5
            transform = np.array([[0, -1, 0, 10], [1, 0, 0, 0],
                                  [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            summary = {
                "input_files": [str(a), str(b)],
                "leveling": {"transformation": level.tolist()},
                "desktop_acquisition": {"input_sha256": {a.name: sha256(a), b.name: sha256(b)}},
                "registration_metrics": [
                    {"file_name": b.name, "transformation": transform.tolist()},
                    {"file_name": a.name, "transformation": np.eye(4).tolist()}],
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            points = np.array([[10, 2, 5], [0, 0, 5], [10, 1, 5], [10, 2.001, 5]])
            original = points.copy()
            clouds = {section: points, a: np.array([[0, 0, 0], [10, 1, 0]]),
                      b: np.array([[1, 0, 0], [2, 0, 0]])}
            backend = Backend()
            result = SimpleNamespace(summary_path=summary_path)
            report = {"source_pointcloud": str(section)}
            with patch("pointcloud_app.native.backend.read_point_cloud",
                       side_effect=lambda path: SimpleNamespace(points=clouds[path])) as read:
                groups = backend.cad_sources(result, report, lambda message: None)
                self.assertEqual({label: indices.tolist() for label, indices in groups},
                                 {"b.ply": [0], "a.ply": [1], "多源重合": [2], "未追溯": [3]})
                self.assertIs(backend.cad_sources(result, report, lambda message: None), groups)
                self.assertEqual(read.call_count, 3)
                b.write_bytes(b"replaced source")
                groups = backend.cad_sources(result, report, lambda message: None)
                self.assertEqual({label: indices.tolist() for label, indices in groups},
                                 {"b.ply": [], "a.ply": [1, 2], "未追溯": [0, 3]})
                a.unlink()
                groups = backend.cad_sources(result, report, lambda message: None)
                self.assertEqual(groups[-1][0], "未追溯")
                np.testing.assert_array_equal(groups[-1][1], np.arange(len(points)))
            np.testing.assert_array_equal(points, original)

    def test_automatic_summary_layout_aligns_columns_and_keeps_multiline_batches(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QHeaderView
        from pointcloud_app.automatic_runtime import measurement_table
        from pointcloud_app.native.widgets import PagedTable
        from pointcloud_app.native.window import STYLE

        app = QApplication.instance() or QApplication([])
        columns, rows = measurement_table([{
            "name": "3", "batch_id": "20261008_100709_772538",
            "report": {"measurement_quality": {"status": "PASS"}, "measurements": {
                str(index): {"label": f"测量路径 {index}（实体 6→14，共 5 段）",
                             "mean_signed_x": -.0068342, "mean_signed_y": -.0039361,
                             "signed_angle_difference_deg": -1.892}
                for index in range(1, 17)}},
        }])
        original = copy.deepcopy(rows)
        widget = PagedTable()
        widget.setStyleSheet(STYLE)
        widget.resize(1100, 500)
        widget.set_data(columns, rows, measurement_summary=True)
        widget.show()
        try:
            app.processEvents()
            widget._fit_rows()
            app.processEvents()
            table = widget.table
            self.assertEqual(table.horizontalScrollBar().maximum(), 0)
            self.assertEqual(table.textElideMode(), Qt.TextElideMode.ElideNone)
            self.assertGreaterEqual(table.rowHeight(0), table.fontMetrics().lineSpacing() * 2 + 14)
            self.assertEqual(table.item(0, 0).text(), "3\n20261008_100709_772538")
            for column in range(6):
                self.assertEqual(table.item(0, column).textAlignment(),
                                 table.horizontalHeaderItem(column).textAlignment())
                self.assertEqual(table.horizontalHeader().sectionResizeMode(column),
                                 QHeaderView.ResizeMode.Fixed)
            self.assertTrue(table.item(0, 1).textAlignment() & Qt.AlignmentFlag.AlignHCenter)
            self.assertTrue(table.item(0, 2).textAlignment() & Qt.AlignmentFlag.AlignLeft)
            self.assertTrue(table.item(0, 3).textAlignment() & Qt.AlignmentFlag.AlignRight)
            for column in (0, 2):
                width = max(table.fontMetrics().horizontalAdvance(line)
                            for line in table.item(0, column).text().split("\n"))
                self.assertGreaterEqual(table.columnWidth(column), width + 28)
            widget.move(1)
            self.assertEqual(table.item(0, 2).text(), rows[widget.page * widget.page_size][2])
            widget.resize(650, 400)
            app.processEvents()
            widget._fit_rows()
            app.processEvents()
            self.assertGreater(table.horizontalScrollBar().maximum(), 0)
            self.assertLessEqual(table.rowCount() * table.rowHeight(0), table.viewport().height())
            for column in (0, 2):
                width = max(table.fontMetrics().horizontalAdvance(line)
                            for line in table.item(0, column).text().split("\n"))
                self.assertGreaterEqual(table.columnWidth(column), width + 28)
            self.assertEqual(rows, original)
        finally:
            widget.close()

    def test_automatic_summary_layout_does_not_leak_into_other_tables(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QHeaderView
        from pointcloud_app.automatic_runtime import measurement_table
        from pointcloud_app.native.widgets import PagedTable

        app = QApplication.instance() or QApplication([])
        widget = PagedTable()
        columns, rows = measurement_table([])
        widget.set_data(columns, rows, measurement_summary=True)
        self.assertEqual(widget.table.columnCount(), 6)
        widget.set_data(["文件", "大小"], [["file.ply", "1 MB"]])
        self.assertFalse(widget.measurement_summary)
        self.assertEqual(widget.table.verticalHeader().defaultSectionSize(), 44)
        self.assertEqual(widget.table.horizontalHeader().sectionResizeMode(0), QHeaderView.ResizeMode.Stretch)
        self.assertEqual(widget.table.textElideMode(), Qt.TextElideMode.ElideRight)
        widget.close()

    def test_audit_toolbar_defaults_to_3d_and_preserves_selected_mode(self):
        from unittest.mock import Mock
        from PySide6.QtWidgets import QApplication, QHBoxLayout
        from pointcloud_app.native.window import DesktopWindow

        app = QApplication.instance() or QApplication([])
        owner = SimpleNamespace(
            tools=QHBoxLayout(), state=None, page=3, job=None, view_values={},
            _views=Mock(), _point_size=Mock(), _busy=Mock(),
        )
        owner.audit_changed = lambda mode: owner.view_values.__setitem__("audit_mode", mode)
        DesktopWindow._toolbar(owner)
        self.assertEqual(owner.audit_mode.currentText(), "3D")
        points = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
        figure = figures.alignment([("1", points, True)], owner.audit_mode.currentText())
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].type, "scatter3d")

        for mode in ("XY", "四视图"):
            owner.audit_mode.setCurrentText(mode)
            DesktopWindow._toolbar(owner)
            self.assertEqual(owner.audit_mode.currentText(), mode)

    def test_figures_white_four_views_and_distinct_sources(self):
        points = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
        figure = figures.alignment([("1", points, True), ("2", points, False)])
        self.assertEqual(len(figure.data), 8)
        self.assertEqual(figure.layout.paper_bgcolor, "white")
        self.assertEqual(figure.layout.legend.itemsizing, "constant")
        self.assertNotEqual(figure.data[0].marker.color, figure.data[4].marker.color)
        self.assertEqual(figure.data[4].visible, "legendonly")

    def test_fractional_point_size_reaches_cloud_and_audit_markers(self):
        points = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
        for size in (0.5, 1.0, 1.5):
            cloud = figures.cloud(points, point_size=size)
            audit = figures.alignment([("1", points, True)], point_size=size)
            for trace in (*cloud.data, *audit.data):
                self.assertEqual(trace.marker.size, size)

    def test_point_size_control_supports_half_steps(self):
        from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QHBoxLayout
        from pointcloud_app.native.window import DesktopWindow

        app = QApplication.instance() or QApplication([])
        changes = []
        owner = SimpleNamespace(tools=QHBoxLayout(), point_size=1.0,
                                size_changed=changes.append)
        DesktopWindow._point_size(owner)
        control = owner.tools.itemAt(1).widget()
        self.assertIsInstance(control, QDoubleSpinBox)
        self.assertEqual(control.minimum(), 0.5)
        self.assertEqual(control.maximum(), 10.0)
        self.assertEqual(control.singleStep(), 0.5)
        control.stepDown()
        control.editingFinished.emit()
        self.assertEqual(changes, [0.5])
        control.stepDown()
        self.assertEqual(control.value(), 0.5)
        control.stepUp()
        self.assertEqual(control.value(), 1.0)
        for index in range(owner.tools.count()):
            owner.tools.itemAt(index).widget().deleteLater()
        app.processEvents()

    def test_process_requires_region_and_valid_file_count(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"{i}.ply" for i in range(7)]
            backend = Backend()
            with self.assertRaisesRegex(ValueError, "框选"):
                backend.process(files, ProcessingConfig(), lambda m: None, Path(directory))
            with self.assertRaisesRegex(ValueError, "5 至 8"):
                backend.process(files[:4], ProcessingConfig(), lambda m: None, Path(directory))

    def test_automatic_process_ignores_station_metadata_and_old_config(self):
        for count in (5, 6, 7, 8):
            with self.subTest(count=count), TemporaryDirectory() as directory:
                root = Path(directory)
                files = [root / f"{index}.ply" for index in range(count)]
                for path in files:
                    path.write_bytes(b"input")
                stamp = acquisition_signature(files)
                manifest = root / "station_order.json"
                manifest.write_text("invalid obsolete mapping", encoding="utf-8")
                self.assertEqual(acquisition_signature(files), stamp)
                config = ProcessingConfig()
                config.section.target_region = Bounds2D(0, 1, 0, 1)
                config.registration.use_global_registration = False
                config.registration.station_pose_corrections = {"reference": {"translation_x": 1.0}}
                original = copy.deepcopy(config.as_dict())
                result = SimpleNamespace(summary_path=root / "run_summary.json")

                def process(group, paths, effective, output, progress):
                    self.assertEqual(paths, files)
                    self.assertFalse(effective.registration.use_six_station_pose_prior)
                    self.assertTrue(effective.registration.use_global_registration)
                    self.assertEqual(effective.registration.station_pose_corrections, {})
                    result.summary_path.write_text(json.dumps({"config": effective.as_dict()}), encoding="utf-8")
                    return result

                with patch("pointcloud_app.native.backend.process_group", side_effect=process):
                    Backend().process(files, config, lambda message: None, root)
                saved = json.loads(result.summary_path.read_text(encoding="utf-8"))
                self.assertTrue(saved["desktop_acquisition"]["input_verified"])
                self.assertEqual(saved["desktop_acquisition"]["registration_mode"], "automatic_calibration")
                self.assertNotIn("confirmed", saved["desktop_acquisition"])
                self.assertEqual(len(saved["desktop_acquisition"]["input_sha256"]), count)
                self.assertIn("timing", saved["desktop_acquisition"])
                self.assertGreaterEqual(saved["desktop_acquisition"]["timing"]["total_seconds"], 0)
                self.assertEqual(config.as_dict(), original)
                self.assertEqual(manifest.read_text(encoding="utf-8"), "invalid obsolete mapping")

    def test_automatic_process_rejects_input_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / f"{index}.ply" for index in range(5)]
            for path in files:
                path.write_bytes(b"input")
            config = default_config("1")
            config.section.target_region = Bounds2D(0, 1, 0, 1)
            def process(*args):
                files[0].write_bytes(b"changed input")
                return SimpleNamespace(summary_path=root / "not-written.json")
            with patch("pointcloud_app.native.backend.process_group", side_effect=process):
                with self.assertRaisesRegex(RuntimeError, "输入文件发生变化"):
                    Backend().process(files, config, lambda message: None, root)

    def test_automatic_quality_preserves_failure_and_coverage_gates(self):
        for status in ("PASS", "FAIL", "INSUFFICIENT_DATA"):
            with self.subTest(status=status):
                base = dict(status=status, formal_measurement_allowed=status == "PASS",
                            failure_reasons=[], insufficient_data_reasons=[])
                summary = {"config": default_config("1").as_dict(), "desktop_acquisition": {
                    "registration_mode": "automatic_calibration", "input_verified": True}}
                with patch("pointcloud_app.downstream_bridge._derive_processing_quality", return_value=base):
                    self.assertEqual(processing_quality(summary)["status"], status)
                    self.assertEqual(processing_quality(summary)["formal_measurement_allowed"], status == "PASS")
                    summary["desktop_acquisition"]["input_verified"] = False
                    self.assertFalse(processing_quality(summary)["formal_measurement_allowed"])

    def test_five_six_view_pipeline_can_bypass_fixed_station_branch(self):
        from pointcloud_app.pipeline import process_group
        for count in (5, 6):
            with self.subTest(count=count):
                files = [Path(f"{i}.ply") for i in range(count)]
                config = default_config("1")
                with patch("pointcloud_app.pipeline.process_group_with_six_station_pose_prior") as fixed, \
                     patch("pointcloud_app.pipeline.resolve_compute_backend", side_effect=RuntimeError("generic path")):
                    with self.assertRaisesRegex(RuntimeError, "generic path"):
                        process_group("test", files, config, Path("output"))
                    fixed.assert_not_called()
                    config.registration.use_six_station_pose_prior = True
                    process_group("test", files, config, Path("output"))
                    fixed.assert_called_once()

    def test_reference_preview_uses_geometry_selector_for_five_six_views(self):
        for count in (5, 6):
            with self.subTest(count=count), TemporaryDirectory() as directory:
                files = [Path(directory) / f"{i}.ply" for i in range(count)]
                for path in files:
                    path.touch()
                with patch("pointcloud_app.native.backend.select_reference_file_with_leveling",
                           return_value=(files[-1], None)) as select, \
                     patch("pointcloud_app.native.backend.build_preview", return_value=object()):
                    reference, _ = Backend().preview(files, default_config("1"), 0, False, lambda message: None)
                self.assertEqual(reference, files[-1])
                select.assert_called_once()

    def test_export_formal_gate_before_writing(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text("{}", encoding="utf-8")
            result = SimpleNamespace(summary_path=path)
            with patch("pointcloud_app.native.backend.processing_quality",
                       return_value={"formal_measurement_allowed": False}):
                with self.assertRaises(ValueError):
                    Backend().export(result, {"measurement_quality": {"status": "PASS"}},
                                     Path(directory) / "out.zip", True, lambda m: None)
            self.assertFalse((Path(directory) / "out.zip").exists())


if __name__ == "__main__":
    unittest.main()
