"""Check actual Qt canvas rendering and point-diameter controls."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox
from pointcloud_app.native.window import CAD_PAGE, DesktopWindow, Group
from pointcloud_app.native.backend import default_config, result_from_summary
from pointcloud_app.native import figures


def main():
    destination = sorted((ROOT / "tmp/cad_pose_verification").glob("*/verification.json"))[-1].parent
    report = json.loads((destination / "group_1/report.json").read_text(encoding="utf-8"))
    verification = json.loads((destination / "verification.json").read_text(encoding="utf-8"))
    summary_path = Path(verification["groups"][0]["old_report"]).parents[2] / "run_summary.json"
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication([])
    window = DesktopWindow(ROOT / "__no_autoload__")
    window._save_workflow_profile = lambda state=None: None
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.canvas.render_error.connect(window.error)
    window.groups["pose-test"] = Group([], default_config("1"), report=report,
                                      result=result_from_summary(summary_path))
    window.group_name = "pose-test"
    window.show()

    def wait(predicate):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            app.processEvents()
            if errors:
                raise AssertionError(errors)
            if predicate():
                return
            QTest.qWait(30)
        raise TimeoutError("Canvas did not render")

    def js(expression):
        output = []
        window.canvas.page().runJavaScript(expression, output.append)
        wait(lambda: bool(output))
        return output[0]

    try:
        wait(lambda: window.canvas.loaded)
        assert window.point_size == 1.5
        window.view_values[CAD_PAGE] = "overlay"
        window.select_page(CAD_PAGE)
        wait(lambda: window.job is None and window.canvas.rendered_generation == window.canvas.generation)
        window._point_size()
        control = window.tools.itemAt(window.tools.count() - 1).widget()
        assert isinstance(control, QDoubleSpinBox)
        assert control.value() == 1.5
        control.setValue(2.0)
        control.editingFinished.emit()
        assert window.point_size == 2.0
        wait(lambda: window.job is None and window.canvas.rendered_generation == window.canvas.generation)
        window.point_size = 1.5
        window.view_values[CAD_PAGE] = "overlay"
        window.select_page(CAD_PAGE)
        wait(lambda: window.job is None and window.canvas.rendered_generation == window.canvas.generation)
        for width, height in ((1360, 860), (900, 620)):
            window.resize(width, height)
            QTest.qWait(1400)
            app.processEvents()
            assert js("plot.data[0].type") == "scattergl"
            assert js("plot._fullData[0].x.length") >= 30
            image = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            values = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
            orange = (values[:, :, 0] > 200) & (values[:, :, 1] > 50) & (values[:, :, 1] < 170) & (values[:, :, 2] < 90)
            assert int(orange.sum()) > 150
            window.grab().save(str(destination / f"overlay_{width}x{height}.png"))
        assert figures.cloud(np.array([[0., 0., 0.], [1., 1., 1.]])).data[0].marker.size == 1.5
        (destination / "ui_verification.json").write_text(
            json.dumps(dict(ok=True, default_point_size=1.5, size_control_editable=True,
                            viewports=[[1360, 860], [900, 620]], errors=errors), indent=2), encoding="utf-8")
        print(str(destination / "ui_verification.json"), flush=True)
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
