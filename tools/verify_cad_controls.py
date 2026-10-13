"""Verify the native CAD import callback and path-selection workflow."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ezdxf

from pointcloud_app.cad_geometry import load_cad_entities
from pointcloud_app.native.backend import config_from_dict, result_from_summary
from pointcloud_app.native.window import CAD_PAGE, DesktopWindow, Group


def main() -> None:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    summary_path = sorted((ROOT / "outputs" / "group_1").glob("*/run_summary.json"))[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = result_from_summary(summary_path)
    state = Group(
        [Path(path) for path in summary["input_files"]],
        config_from_dict(summary),
        result=result,
    )

    with TemporaryDirectory() as temporary_directory:
        dxf_path = Path(temporary_directory) / "workflow.dxf"
        document = ezdxf.new()
        modelspace = document.modelspace()
        modelspace.add_line((0, 0), (2, 0))
        modelspace.add_point((50, 50))
        modelspace.add_line((0, 1), (2, 1))
        modelspace.add_line((2, 0), (3, 1))
        modelspace.add_line((3, 1), (4, 1))
        modelspace.add_line((4, 1), (5, 0))
        document.saveas(dxf_path)

        window = DesktopWindow(ROOT / "__no_autoload__")
        window._save_workflow_profile = lambda state=None: None
        window.groups["cad-test"] = state
        window.group_name = "cad-test"
        window.show()
        app.processEvents()

        def wait_render():
            deadline = time.monotonic() + 120
            while window.canvas.rendered_generation != window.canvas.generation and time.monotonic() < deadline:
                app.processEvents()
                QTest.qWait(50)
            assert window.canvas.rendered_generation == window.canvas.generation

        def synchronous_start(operation, callback):
            callback(operation(lambda message: None))
            return True

        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(dxf_path), "DXF (*.dxf)")), \
             patch.object(window, "_start", side_effect=synchronous_start):
            window.open_cad()
        QTest.qWait(250)
        app.processEvents()

        assert window.page == CAD_PAGE
        assert window.view.currentData() == "paths"
        assert state.dxf == dxf_path
        assert [entity.index for entity in state.entities] == [0, 2, 3, 4, 5]
        assert window.measure_button.isEnabled() is False

        generation = window.canvas.generation
        window.interaction({"type": "cad", "generation": generation, "entity": 2})
        wait_render()
        assert state.pending == 2
        assert "待确认起点" in window.cad_hint.text()
        window.cad_single()
        wait_render()
        assert state.selection["registration"] == [2]

        window.cad_role_buttons["measurement"].click()
        QTest.qWait(100)
        wait_render()
        window.interaction({"type": "cad", "generation": window.canvas.generation, "entity": 0})
        wait_render()
        assert state.pending == 0
        window.cad_single()
        wait_render()
        assert state.selection["measurement"] == [0]
        assert window.measure_button.isEnabled() is True

        for index in (3, 4, 5):
            window.interaction({"type": "cad", "generation": window.canvas.generation, "entity": index})
            wait_render()
        window.cad_single()
        wait_render()
        assert state.selection["measurement_paths"] == [[0], [3, 4], [5]]
        rendered = []
        window.canvas.page().runJavaScript(
            "JSON.stringify(document.getElementById('plot').data.map(t => t.line.color))",
            rendered.append,
        )
        deadline = time.monotonic() + 10
        while not rendered and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(50)
        assert rendered
        colors = json.loads(rendered[0])
        assert colors[1] == "#075cff"
        assert colors[2] == colors[3]
        assert len({colors[0], colors[2], colors[4]}) == 3

        current_width = window.toolbar.width()
        for index in range(window.tools.count()):
            widget = window.tools.itemAt(index).widget()
            if widget is not None:
                assert widget.geometry().right() <= current_width, widget.objectName()
        visible_controls = [
            widget.text()
            for widget in window.toolbar.findChildren(type(window.measure_button))
            if widget.isVisible()
        ]
        assert "导入 DXF" in visible_controls
        assert "配准并测量" in visible_controls

        locked_controls = [
            window.toolbar, window.stack, window.canvas, window.table,
            window.group_combo, window.params_button, window.history_button,
            window.export_button, window.log_button, *window.nav,
        ]
        window.workflow_mode = "automatic"
        window.auto_enabled = True
        window._refresh_auto_runtime()
        for busy in (False, True, False):
            window._busy(busy)
            assert all(not control.isEnabled() for control in locked_controls)
            assert window.workflow_mode_button.isEnabled()
            assert window.auto_stop_button.isEnabled()
        window.auto_enabled = False
        window.workflow_mode = "manual"
        window._busy(True)
        assert all(not control.isEnabled() for control in locked_controls)
        window._busy(False)
        window._refresh_auto_runtime()
        assert all(control.isEnabled() for control in locked_controls)

        output = ROOT / "tmp" / "cad_controls_verification"
        output.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(output / "cad_controls.png"))
        window.auto_records = {
            f"run-{index}": {
                "id": f"run-{index}", "name": f"group-{index}", "batch_id": "switch-test",
                "report": {"measurements": {"path": {
                    "label": "path-1", "mean_signed": index * 0.01,
                }}},
            }
            for index in (1, 2)
        }
        window.workflow_mode = "automatic"
        window.view_values[CAD_PAGE] = "measurements"
        window._toolbar()
        window._draw()
        before_rows = [list(row) for row in window.table.rows]
        before_columns = list(window.table.columns)
        before_group = window.group_name
        window.workflow_mode_changed("manual")
        assert window.view.currentData() == "automatic_measurements"
        assert window.table.rows == before_rows
        assert window.table.columns == before_columns
        assert window.group_name == before_group
        assert window.state is state
        assert window.stack.currentWidget() is window.table
        assert window.table.isEnabled()
        window.select_page(0)
        window.select_page(CAD_PAGE)
        assert window.table.rows == before_rows
        assert window.view.currentData() == "automatic_measurements"
        app.processEvents()
        window.grab().save(str(output / "automatic_summary_manual.png"))
        started_pages = []
        for page in range(CAD_PAGE + 1):
            window.view_values[page] = {
                0: None, 1: "view:0", 2: "plane", 3: "alignment", CAD_PAGE: "overlay",
            }[page]
            with patch.object(window, "_draw"):
                window.select_page(page)
            with patch.object(window, "_source_blocker", return_value=""), \
                 patch.object(window, "auto_poll") as poll, \
                 patch.object(window, "_start", side_effect=AssertionError("Unexpected preview task")):
                window.workflow_mode_button.click()
                assert window.workflow_mode == "automatic"
                assert window.auto_enabled
                assert window.auto_timer.isActive()
                assert window.page == CAD_PAGE
                assert window.view.currentData() == "automatic_measurements"
                assert window.stack.currentWidget() is window.table
                assert not window.toolbar.isEnabled()
                poll.assert_called_once()
                started_pages.append(page)
                window.workflow_mode_button.click()
                assert window.workflow_mode == "manual"
                assert not window.auto_enabled
                assert not window.auto_timer.isActive()
                assert window.toolbar.isEnabled()
        (output / "verification.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "cad_page": window.page,
                    "entity_indices": [entity.index for entity in state.entities],
                    "registration": state.selection["registration"],
                    "measurement": state.selection["measurement"],
                    "measurement_paths": state.selection["measurement_paths"],
                    "rendered_colors": colors,
                    "automatic_controls_lock": True,
                    "automatic_summary_preserved_in_manual": True,
                    "automatic_started_from_pages": started_pages,
                    "measure_enabled": window.measure_button.isEnabled(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        window.close()
        QCoreApplication.processEvents()
        print(json.dumps({"ok": True, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
