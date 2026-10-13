"""Verify that acquisition no longer requires station selection."""
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

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
window.error = errors.append
window.resize(900, 620)
window.show()


def wait(predicate):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        app.processEvents()
        assert not errors, errors
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError(f"job={window.job is not None}; logs={window.logs[-8:]}; "
                       f"render={window.canvas.rendered_generation}/{window.canvas.generation}")


try:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for i in range(6):
            path = root / f"random_{i}.ply"
            path.touch()
            os.utime(path, ns=(1_700_000_000_000_000_000 + i * 1_000_000_000,) * 2)
        window.import_root(root)
        wait(lambda: window.job is None and window.state is not None)
        texts = {b.text() for b in window.toolbar.findChildren(QPushButton) if b.isVisible()}
        assert "预览选中文件" in texts
        assert not {"自动选择工位", "确认工位", "已确认"} & texts
        assert not (root / "station_order.json").exists()
        assert window.table.table.columnCount() == 2
        for i, path in enumerate(window.state.files):
            assert window.table.table.cellWidget(i, 1) is None
        from pointcloud_app.models import Bounds2D
        window.state.config.section.target_region = Bounds2D(0, 1, 0, 1)
        window._ready_to_process(window.state)
        for button in window.toolbar.findChildren(QPushButton):
            assert button.geometry().right() < window.toolbar.width()
        destination = ROOT / "tmp/native_ui_verification"
        destination.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(destination / "automatic_registration_data.png"))
        print("PASS: no station controls; preview retained; ready without confirmation; 900px layout")
    wait(lambda: window.canvas.loaded)
    window.import_root(ROOT / "media" / "1")
    wait(lambda: window.job is None)
    window.level = 0
    window.view_values[2] = "region"
    window.select_page(2)
    wait(lambda: window.job is None and window.state.preview is not None
         and window.canvas.rendered_generation == window.canvas.generation)
    assert window.state.config.section.target_region is None
    window.grab().save(str(destination / "automatic_reference_region.png"))
    points = window.state.preview.points
    bounds = [float(points[:, 0].min()), float(points[:, 0].max()),
              float(points[:, 1].min()), float(points[:, 1].max())]
    window.interaction({"type": "region", "generation": window.canvas.generation, "bounds": bounds})
    wait(lambda: window.job is None and window.state.config.section.target_region is not None)
    window._ready_to_process(window.state)
    print("PASS: real reference preview and region selection without station confirmation", flush=True)
finally:
    if window.job:
        window.cancel()
        wait(lambda: window.job is None)
    window.close()
    app.processEvents()
