"""Exercise section controls across real Qt toolbar deletion and WebChannel."""
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt, QTimer, QCoreApplication, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QDialogButtonBox

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
app = QApplication([])
from pointcloud_app.native.window import DesktopWindow, Group
from pointcloud_app.native.backend import default_config, Cancelled
from pointcloud_app.native.widgets import Parameters

errors = []
def report_exception(typ, value, tb):
    import traceback
    traceback.print_exception(typ, value, tb)
    errors.append(str(value))


sys.excepthook = report_exception
window = DesktopWindow(ROOT / "__no_autoload__")
from pointcloud_app.workflow_profile import save_profile
window._save_workflow_profile = lambda state=None: save_profile(
    {"schema_version": 1, "config": (state or window.state).config.as_dict()},
    ROOT / "tmp" / "section_controls_verification" / "isolated_profile.json")
window.error = lambda message: errors.append(str(message))
window.canvas.render_error.connect(window.error)
destination = ROOT / "tmp" / "section_controls_verification"
destination.mkdir(parents=True, exist_ok=True)


def wait(predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if errors:
            raise AssertionError(errors)
        if predicate():
            return
        QTest.qWait(20)
    raise TimeoutError(window.logs[-5:])


def ready():
    return window.job is None and (
        window.stack.currentWidget() is not window.canvas
        or window.canvas.rendered_generation == window.canvas.generation)


def mode(value):
    window.view.setCurrentIndex(window.view.findData(value))
    wait(ready)
    assert window.view_values[2] == value


def button(text):
    return next(widget for widget in window.toolbar.findChildren(QPushButton)
                if widget.text() == text)


def js(expression):
    result = []
    window.canvas.page().runJavaScript(expression, result.append)
    wait(lambda: bool(result))
    return result[0]


window.show()
try:
    wait(lambda: window.canvas.loaded)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        files = [root / f"{i}.ply" for i in range(6)]
        for path in files:
            path.touch()
        t = np.linspace(-10, 10, 2000)
        points = np.c_[t, .2 * t + .02 * np.sin(10 * t), .3 * t]
        state = Group(files, default_config("1"))
        preview = SimpleNamespace(points=points)
        window.groups["controls"] = state
        window.group_name = "controls"
        with patch.object(window.backend, "preview", return_value=(files[0], preview)):
            window.select_page(1)
            wait(ready)
            window.select_page(2)
            wait(ready)
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            mode("region")
            assert js("plot.layout.dragmode") == "select"
            js("plot.emit('plotly_selected', {range:{x:[-8,8],y:[-3,3]}})")
            wait(lambda: state.config.section.target_region is not None and ready())
            assert window.view.currentData() == "region"
            assert abs(state.config.section.tilt_degrees) > 5
            js("Plotly.relayout(plot, {'selections[0].x0': -7, 'selections[0].x1': 9, "
               "'selections[0].y0': -2.5, 'selections[0].y1': 3.5})")
            wait(lambda: state.config.section.target_region.x_max == 9 and ready())
            assert state.config.section.target_region.y_min == -2.5
            persisted = json.loads((destination / "isolated_profile.json").read_text(encoding="utf-8"))
            assert persisted["config"]["section"]["target_region"]["x_max"] == 9
            generation = window.canvas.generation
            js("Plotly.relayout(plot, {'xaxis.range': [-15,15]})")
            QTest.qWait(150)
            assert window.canvas.generation == generation
            js("Plotly.relayout(plot, {'selections[0].x0': -7, 'selections[0].x1': 7})")
            wait(lambda: state.config.section.target_region is not None and
                 abs(state.config.section.target_region.x_max - 7) < 1e-6 and ready())
            assert abs(state.config.section.target_region.x_max - 7) < 1e-6
            window.view.setCurrentIndex(window.view.findData("plane"))
            wait(ready)
            old_thickness = state.config.section.thickness

            def apply_parameters():
                dialog = app.activeModalWidget()
                assert isinstance(dialog, Parameters)
                assert dialog.category.currentData() == "section"
                control = next(control for path, control, kind in dialog.controls
                               if path[-1] == "thickness")
                control.setValue(old_thickness + .01)
                box = dialog.findChild(QDialogButtonBox)
                QTest.mouseClick(box.button(QDialogButtonBox.StandardButton.Ok),
                                 Qt.MouseButton.LeftButton)

            QTimer.singleShot(100, apply_parameters)
            QTest.mouseClick(button("\u622a\u9762\u53c2\u6570"), Qt.MouseButton.LeftButton)
            wait(ready)
            assert abs(state.config.section.thickness - old_thickness - .01) < 1e-6
            mode("slice")
            mode("neighbors")
            assert window.stack.currentWidget() is window.table
            mode("final")
            assert window.stack.currentWidget() is not window.canvas
            mode("raw")
            assert window.stack.currentWidget() is not window.canvas

            # Saved section modes use actual PLY and summary readers.
            section_path = root / "section.ply"
            o3d.io.write_point_cloud(str(section_path),
                                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
            summary = root / "run_summary.json"
            summary.write_text(json.dumps({"config": state.config.as_dict()}), encoding="utf-8")
            state.result = SimpleNamespace(
                section_cloud_path=section_path, raw_section_cloud_path=section_path,
                summary_path=summary)
            mode("final")
            assert window.stack.currentWidget() is window.canvas
            mode("raw")
            assert window.stack.currentWidget() is window.canvas
            mode("region")
            QTest.qWait(300)
            window.grab().save(str(destination / "region.png"))
            mode("plane")
            QTest.qWait(300)
            window.grab().save(str(destination / "plane.png"))
            calls = []

            def processing(files, config, progress, output):
                calls.append(config.section.target_region)
                raise Cancelled("UI verification only")

            with patch.object(window.backend, "process", side_effect=processing):
                QTest.mouseClick(button("\u5f00\u59cb\u5904\u7406"), Qt.MouseButton.LeftButton)
                wait(lambda: window.job is None)
            assert calls == [state.config.section.target_region]
            # Repeat navigation after all old controls have been destroyed.
            for _ in range(3):
                window.select_page(1)
                wait(ready)
                window.select_page(2)
                wait(ready)
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                mode("region")
                mode("plane")
    report = {"ok": True, "errors": errors, "section_modes": 6,
              "region_bridge": True, "region_move_resize_persist": True,
              "zoom_does_not_refit": True, "parameters_applied": True,
              "process_button_dispatch": True, "repeat_navigation": 3,
              "processing_backend_mocked": True}
    (destination / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
finally:
    if window.job:
        window.cancel()
        wait(lambda: window.job is None)
    window.close()
    app.processEvents()
