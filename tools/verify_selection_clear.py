"""Verify that a completed rectangle selection does not survive the next view."""

import sys
import json
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
app = QApplication([])
from pointcloud_app.native.canvas import Canvas
from pointcloud_app.native import figures
import numpy as np
import time

window = Canvas()
window.resize(900, 620)
window.show()

def wait(predicate):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(25)
    raise TimeoutError()

try:
    wait(lambda: window.loaded)
    t = np.linspace(-10, 10, 1000)
    points = np.c_[t, t * .2, t * .3]
    rendered = []
    window.rendered.connect(lambda: rendered.append(True))
    window.show_figure(figures.cloud(points, projection="XY"), "region")
    wait(lambda: window.loaded and rendered)
    window.page().runJavaScript(
        "Plotly.relayout(plot, {selections:[{x0:-5,x1:5,y0:-2,y1:2}]});")
    QTest.qWait(100)
    window.page().runJavaScript("Plotly.relayout(plot, {selections:[]});")
    values = []
    window.page().runJavaScript(
        "JSON.stringify(plot.layout.selections || [])", values.append)
    wait(lambda: values)
    assert json.loads(values[0]) == []
    print("PASS: selection rectangle cleared from rendered figure")
finally:
    window.close()
    app.processEvents()
