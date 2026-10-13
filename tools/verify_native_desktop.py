"""Exercise the actual Qt/WebGL window and capture repeatable visual checks."""

from __future__ import annotations

import json
import sys
import time
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
app = QApplication([])
app.setStyle("Fusion")

from pointcloud_app.native.window import DesktopWindow, Group
from pointcloud_app.native.backend import (
    Cancelled, default_config, config_from_dict, result_from_summary,
)
from pointcloud_app.native.widgets import Parameters
from pointcloud_app.cad_geometry import load_cad_entities


def wait(predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if errors:
            raise AssertionError(errors)
        if predicate():
            return
        time.sleep(0.03)
    raise TimeoutError(f"UI wait timed out; logs={window.logs[-8:]}; "
                       f"render={window.canvas.rendered_generation}/{window.canvas.generation}")


def js(expression):
    output = []
    window.canvas.page().runJavaScript(expression, lambda value: output.append(value))
    wait(lambda: bool(output), 10)
    return output[0]


def view(page, mode):
    rendered.clear()
    window.view_values[page] = mode
    window.select_page(page)
    wait(lambda: window.job is None and (bool(rendered) or window.stack.currentWidget() is not window.canvas))
    if errors:
        raise AssertionError(errors)
    if window.stack.currentWidget() is window.canvas:
        wait(lambda: bool(rendered))
        QTest.qWait(600)


def screenshot(name):
    QTest.qWait(1000)
    app.processEvents()
    path = destination / (name + ".png")
    window.grab().save(str(path))
    image = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    values = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
    colored = (values[:, :, :3].max(axis=2).astype(int) - values[:, :, :3].min(axis=2).astype(int)) > 45
    if window.stack.currentWidget() is window.canvas:
        print(name, js("JSON.stringify({traces:plot.data?.length, rect:plot.getBoundingClientRect().toJSON(),html:plot.innerHTML.length,canvas:document.querySelectorAll('canvas').length})"), flush=True)
        assert int(colored.sum()) > 150, f"Canvas is blank: {name}"
    screenshots.append({"name": name, "colored_pixels": int(colored.sum()),
                        "width": window.width(), "height": window.height()})


destination = ROOT / "tmp" / "native_ui_verification"
destination.mkdir(parents=True, exist_ok=True)
screenshots, errors, rendered = [], [], []
window = DesktopWindow(ROOT / "__no_autoload__")
window._save_workflow_profile = lambda state=None: None
window.error = lambda message: errors.append(str(message))
window.canvas.render_error.disconnect()
window.canvas.render_error.connect(window.error)
window.canvas.rendered.connect(lambda: rendered.append(True))
window.show()
try:
    wait(lambda: window.canvas.loaded)
    window.import_root(ROOT / "media")
    wait(lambda: window.job is None and bool(window.state))
    assert len(window.groups) >= 1
    screenshot("01_data")
    summary_path = max(
        (ROOT / "outputs" / "group_1").glob("*/run_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    window.load_result(summary_path)
    wait(lambda: window.job is None and window.state.result is not None and bool(rendered))
    screenshot("02_fused")
    assert js("document.body.scrollHeight === document.body.clientHeight")
    assert not window.findChildren(QScrollArea)
    camera_before = js("JSON.stringify(plot.layout.scene.camera || {})")
    js("Plotly.relayout(plot, {'scene.camera.eye': {x:0.4,y:2,z:0.9}});")
    QTest.qWait(500)
    camera_after = js("JSON.stringify(plot.layout.scene.camera || {})")
    assert camera_before != camera_after
    screenshot("03_rotated")
    view(3, "alignment")
    screenshot("04_alignment")
    assert js("plot.data.filter(t => t.type === 'scatter3d').length") >= 5
    window.resize(900, 620)
    QTest.qWait(800)
    screenshot("05_small_alignment")
    assert window.width() == 900
    view(2, "final")
    screenshot("06_final_section")
    window.resize(1360, 860)
    window.state.dxf = ROOT / "test母头.dxf"
    window.state.entities = load_cad_entities(window.state.dxf)
    view(4, "paths")
    screenshot("07_cad")
    entity = window.state.entities[0].index
    js(f"plot.emit('plotly_click', {{points:[{{customdata:{entity}}}]}})")
    wait(lambda: window.state.pending == entity)
    window.cad_single()
    assert window.state.selection["registration"] == [entity]
    window.cad_undo()
    window.resize(900, 620)
    QTest.qWait(600)
    screenshot("08_small_cad")
    for index in range(window.tools.count()):
        widget = window.tools.itemAt(index).widget()
        if widget:
            assert widget.geometry().right() <= window.toolbar.width(), widget.objectName()
    dialog = Parameters(window.state.config, window)
    dialog.show()
    QTest.qWait(300)
    dialog.grab().save(str(destination / "09_parameters.png"))
    dialog.reject()
    report_paths = sorted((ROOT / "tmp/native_measurement_verification").glob("*/native_cad/*/report.json"))
    if report_paths:
        report = json.loads(report_paths[-1].read_text(encoding="utf-8"))
        window.state.report = report
        view(4, "overlay")
        screenshot("12_cad_overlay")
        view(4, "curve")
        screenshot("13_deviation_curve")
        view(4, "measurements")
        assert window.table.table.rowCount() >= 2
        screenshot("14_measurement_table")
        view(4, "report")
        screenshot("15_measurement_quality")
        recorded = sorted((ROOT / "tmp/native_pipeline_verification").glob("*/recorded_acquisition"))
        if recorded:
            window.import_root(recorded[-1])
            wait(lambda: window.job is None)
            view(2, "region")
            screenshot("16_real_region")
            assert window.state.preview is not None
            assert len(window.state.preview.points) >= 30
    # Exercise Qt -> local JavaScript -> QWebChannel -> PCA -> process worker.
    with TemporaryDirectory() as temporary:
        files = [Path(temporary) / f"{i}.ply" for i in range(7)]
        for path in files:
            path.touch()
        state = Group(files, default_config("1"))
        t = np.linspace(-10, 10, 2000)
        points = np.c_[t, .2 * t + .02 * np.sin(10 * t), .3 * t]
        preview = SimpleNamespace(points=points)
        window.groups["ui-test"] = state
        window.group_name = "ui-test"
        with patch.object(window.backend, "preview", return_value=(files[0], preview)):
            view(2, "region")
            screenshot("10_region")
            rendered.clear()
            js("plot.emit('plotly_selected', {range:{x:[-8,8],y:[-3,3]}})")
            wait(lambda: state.config.section.target_region is not None and window.job is None and bool(rendered))
            view(2, "plane")
            screenshot("11_normal_plane")
            assert abs(state.config.section.tilt_degrees) > 5
            frozen_offset = state.config.section.plane_offset
            old_generation = window.canvas.generation - 1
            window.interaction({"type": "region", "generation": old_generation, "bounds": [0, 0, 0, 0]})
            assert state.config.section.plane_offset == frozen_offset
            received = []
            def process_spy(files, config, progress, output):
                received.append((config, threading.get_ident()))
                raise Cancelled("verification")
            with patch.object(window.backend, "process", side_effect=process_spy):
                window.process()
                wait(lambda: window.job is None)
            assert received and received[0][0].section.target_region == state.config.section.target_region
            assert received[0][1] != threading.get_ident()
        gate = threading.Event()
        calls = []
        window._start(lambda progress: (gate.wait(2), progress("done")), lambda result: calls.append(result))
        window.select_page(0)
        window.close()
        assert window.isVisible()
        gate.set()
        wait(lambda: window.job is None)
        assert not calls
    assert "streamlit" not in sys.modules
    assert not errors, errors
    (destination / "verification.json").write_text(json.dumps(
        {"screenshots": screenshots, "errors": errors, "no_streamlit": True,
         "camera_changed": True, "cad_click_bridge": True,
         "region_bridge": True, "process_receives_section": True,
         "close_waits_for_worker": True}, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "screenshots": screenshots}, indent=2), flush=True)
finally:
    if window.job:
        window.cancel()
        wait(lambda: window.job is None, 180)
    window.close()
    app.processEvents()
