"""Check source tracing, unchanged measurement files, and the native CAD canvas."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pointcloud_app.native import figures
from pointcloud_app.cad_geometry import load_cad_entities, normalize_cad_path_selection
from pointcloud_app.native.backend import config_from_dict, result_from_summary, sha256
from pointcloud_app.native.window import CAD_PAGE, DesktopWindow, Group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path)
    args = parser.parse_args()
    batch_path = args.batch or sorted((ROOT / "outputs/automatic").glob("*/batch.json"))[-1]
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    records = [r for r in batch["records"] if r.get("status") == "completed" and r.get("report")]
    destination = ROOT / "tmp/cad_source_overlay" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    protected = [ROOT / "settings/workflow_profile.json", batch_path]
    for record in records:
        report, summary_path = record["report"], Path(record["summary_path"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        section = Path(report["source_pointcloud"])
        protected.extend([summary_path, section, section.parent.parent / "processing_report.json",
                          Path(report["source_dxf"]), Path(report["source_dxf"]).parent / "report.json"])
        protected.extend(Path(path) for path in summary["input_files"])
        protected.extend(Path(m["csv_path"]) for m in report["measurements"].values())
    hashes = {path: sha256(path) for path in set(protected) if path.is_file()}
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    window = DesktopWindow(ROOT / "__no_autoload__")
    window._save_workflow_profile = lambda state=None: None
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.canvas.render_error.connect(window.error)
    rows = []

    def wait(predicate):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            app.processEvents()
            if errors:
                raise AssertionError(errors)
            if predicate():
                return
            QTest.qWait(30)
        raise TimeoutError("Native canvas did not finish rendering")

    def js(expression):
        values = []
        window.canvas.page().runJavaScript(expression, values.append)
        wait(lambda: bool(values))
        return values[0]

    def wait_render():
        wait(lambda: window.job is None and window.canvas.rendered_generation == window.canvas.generation)

    try:
        for record in records:
            report = record["report"]
            summary_path = Path(record["summary_path"])
            result = result_from_summary(summary_path)
            sources = window.backend.cad_sources(result, report, lambda message: None)
            assert sum(len(indices) for _, indices in sources) == result.section_points
            plain = figures.cad_overlay(report)
            colored = figures.cad_overlay(report, sources, measurement_lines=True)
            for trace, (_, indices) in zip(colored.data, sources):
                np.testing.assert_array_equal(trace.x, np.asarray(plain.data[0].x)[indices])
                np.testing.assert_array_equal(trace.y, np.asarray(plain.data[0].y)[indices])
            counts = {label: len(indices) for label, indices in sources}
            rows.append({"group": record["name"], "points": result.section_points,
                         "source_counts": counts, "quality": report["measurement_quality"]["status"]})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

        record = next(r for r in records if r["name"] == "4")
        report = record["report"]
        summary_path = Path(record["summary_path"])
        window.groups["4"] = Group([], config_from_dict(
            json.loads(summary_path.read_text(encoding="utf-8"))),
            result=result_from_summary(summary_path), report=report,
            dxf=Path(report["source_dxf"]), entities=load_cad_entities(report["source_dxf"]),
            selection=normalize_cad_path_selection(report["desktop_input"]["selection"]))
        window.group_name = "4"
        window.group_combo.blockSignals(True)
        window.group_combo.addItem("4")
        window.group_combo.setCurrentText("4")
        window.group_combo.blockSignals(False)
        window.view_values[CAD_PAGE] = "source_overlay"
        window.show()
        window.select_page(CAD_PAGE)
        wait_render()
        assert window.view.currentData() == "source_overlay"
        assert window.view.findData("overlay") == -1
        assert window.view.findData("measurement_overlay") == -1
        for width, height in ((1360, 860), (900, 620)):
            window.resize(width, height)
            QTest.qWait(1200)
            app.processEvents()
            count = js("plot._fullData.filter(t => String(t.legendgroup).startsWith('source-')).reduce((n,t) => n+t.x.length,0)")
            assert count == window.state.result.section_points, count
            station_count = js("plot._fullData.filter(t => String(t.legendgroup).startsWith('measurement-'))"
                               ".reduce((n,t) => n+t.x.filter(x => x !== null && Number.isFinite(x)).length,0)")
            assert station_count == sum(m["valid_stations"] for m in report["measurements"].values())
            window.grab().save(str(destination / f"sources_{width}x{height}.png"))
            spacing = json.loads(js("JSON.stringify((() => { const a=plot.querySelector('.gtitle').getBoundingClientRect(), b=plot.querySelector('.legend').getBoundingClientRect(); return {title_bottom:a.bottom,legend_top:b.top}; })())"))
            assert spacing["title_bottom"] + 4 <= spacing["legend_top"], spacing
            image = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
            assert int((pixels[:, :, :3].max(axis=2) - pixels[:, :, :3].min(axis=2) > 60).sum()) > 200
        js("Plotly.relayout(plot, {'xaxis.range':[1144.15,1144.4], 'yaxis.range':[817.1,817.8]}); true")
        QTest.qWait(700)
        window.grab().save(str(destination / "local_layers.png"))
        js("Plotly.restyle(plot, {visible:'legendonly'}, [0]); true")
        assert js("plot.data[0].visible") == "legendonly"
        js("Plotly.restyle(plot, {visible:true}, [0]); true")
        assert js("plot.data[0].visible") is True
        window.view.setCurrentIndex(window.view.findData("paths"))
        wait_render()
        window.view.setCurrentIndex(window.view.findData("source_overlay"))
        wait_render()
        assert js("plot.layout.title.text") == "来源与测量代表线"
        assert all(sha256(path) == digest for path, digest in hashes.items())
        (destination / "verification.json").write_text(json.dumps(
            {"ok": True, "batch": str(batch_path), "groups": rows, "inputs_unchanged": True,
             "viewports": [[1360, 860], [900, 620]], "legend_toggle": True, "errors": errors},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(destination / "verification.json"), flush=True)
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
