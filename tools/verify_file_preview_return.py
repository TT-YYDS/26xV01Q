"""Regression check for the data-page preview and return action."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
app = QApplication([])
from pointcloud_app.native.window import DesktopWindow

window = DesktopWindow(ROOT / "__no_autoload__")
errors = []
window.error = lambda text: errors.append(text)
window.show()


def wait(predicate):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        app.processEvents()
        if errors:
            raise AssertionError(errors)
        if predicate():
            return
        QTest.qWait(30)
    raise TimeoutError("Preview wait timed out")


try:
    wait(lambda: window.canvas.loaded)
    window.import_root(ROOT / "media")
    wait(lambda: window.job is None and window.state is not None)
    state = window.state
    for attempt in range(2):
        window.data_row_clicked(attempt, 0)
        wait(lambda: window.job is None and window.single_file_view
             and window.canvas.rendered_generation == window.canvas.generation)
        payload = json.loads(window.canvas.last_json)
        assert payload["data"][0]["marker"]["size"] == 1.5
        button = next(b for b in window.toolbar.findChildren(QPushButton)
                      if b.text() == "返回文件列表")
        QTest.qWait(1000)
        if attempt == 0:
            folder = ROOT / "tmp/native_ui_verification"
            folder.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(folder / "file_preview_small_points.png"))
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert window.stack.currentWidget() is window.table
        assert window.table.table.currentRow() == attempt
        assert state.selected_file == state.files[attempt]
        assert window.table.table.columnCount() == 2
        assert not window.single_file_view
    window.select_page(3)
    window.table.table.cellClicked.emit(0, 0)
    assert window.job is None
    assert window.page == 3
    print("PASS: preview size 1.5; return twice; selection preserved; no station controls; other tables unaffected")
finally:
    if window.job:
        window.cancel()
        wait(lambda: window.job is None)
    window.close()
    app.processEvents()
