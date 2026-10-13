"""Verify combined source and median-station contours without changing measurements."""
from __future__ import annotations

import argparse
import csv
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

from pointcloud_app.cad_geometry import load_cad_entities, normalize_cad_path_selection
from pointcloud_app.native import figures
from pointcloud_app.native.backend import Backend, config_from_dict, result_from_summary, sha256
from pointcloud_app.native.window import CAD_PAGE, DesktopWindow, Group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path)
    args = parser.parse_args()
    batch_path = args.batch or sorted((ROOT / "outputs/automatic").glob("*/batch.json"))[-1]
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    records = [r for r in batch["records"] if r.get("status") == "completed" and r.get("report")]
    assert records, "No completed measurement reports"
    original = json.dumps(records, sort_keys=True)
    destination = ROOT / "tmp/cad_measurement_overlay" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    protected = [batch_path, ROOT / "settings/workflow_profile.json"]
    for record in records:
        report, summary_path = record["report"], Path(record["summary_path"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        section = Path(report["source_pointcloud"])
        protected.extend([summary_path, section, section.parent.parent / "processing_report.json",
                          Path(report["source_dxf"]), Path(report["source_dxf"]).parent / "report.json"])
        protected.extend(Path(path) for path in summary["input_files"])
        protected.extend(Path(m["csv_path"]) for m in report["measurements"].values())
    hashes = {path: sha256(path) for path in set(protected) if path.is_file()}
    backend = Backend()
    checks = []
    for record in records:
        report = record["report"]
        result = result_from_summary(Path(record["summary_path"]))
        sources = backend.cad_sources(result, report, lambda message: None)
        plain = figures.cad_overlay(report)
        overlay = figures.cad_overlay(report, sources, measurement_lines=True)
        for trace, (_, indices) in zip(overlay.data, sources):
            np.testing.assert_array_equal(trace.x, np.asarray(plain.data[0].x)[indices])
            np.testing.assert_array_equal(trace.y, np.asarray(plain.data[0].y)[indices])
        assert sum(len(indices) for _, indices in sources) == result.section_points
        traces = {t.legendgroup.removeprefix("measurement-"): t for t in overlay.data
                  if t.legendgroup and t.legendgroup.startswith("measurement-")}
        counts = {}
        for key, measurement in report["measurements"].items():
            with Path(measurement["csv_path"]).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            trace = traces[key]
            assert len(trace.x) == len(rows) == measurement["total_stations"]
            valid_indices = []
            for index, row in enumerate(rows):
                if row["valid"] == "True":
                    assert trace.x[index] == float(row["pointcloud_x"])
                    assert trace.y[index] == float(row["pointcloud_y"])
                    valid_indices.append(index)
                else:
                    assert trace.x[index] is None and trace.y[index] is None
            assert len(valid_indices) == measurement["valid_stations"]
            assert trace.connectgaps is False
            if valid_indices:
                details = np.asarray([trace.customdata[index] for index in valid_indices], dtype=float)
                np.testing.assert_allclose(details[:, 1:].mean(axis=0),
                                           [measurement["mean_signed"], measurement["mean_signed_x"],
                                            measurement["mean_signed_y"]], atol=1e-12, rtol=0)
            counts[key] = len(valid_indices)
        checks.append({"group": record["name"], "source_counts": {label: len(indices) for label, indices in sources},
                       "valid_stations": counts, "statistics_match": True})
        print(json.dumps(checks[-1]), flush=True)

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    window = DesktopWindow(ROOT / "__no_autoload__")
    window.backend = backend
    window._save_workflow_profile = lambda state=None: None
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.canvas.render_error.connect(window.error)

    def wait(predicate):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            app.processEvents()
            assert not errors, errors
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
        record = next((r for r in records if r["name"] == "4"), records[0])
        report = record["report"]
        summary_path = Path(record["summary_path"])
        window.groups[record["name"]] = Group([], config_from_dict(
            json.loads(summary_path.read_text(encoding="utf-8"))),
            result=result_from_summary(summary_path), report=report,
            dxf=Path(report["source_dxf"]), entities=load_cad_entities(report["source_dxf"]),
            selection=normalize_cad_path_selection(report["desktop_input"]["selection"]))
        window.group_name = record["name"]
        window.group_combo.blockSignals(True)
        window.group_combo.addItem(record["name"])
        window.group_combo.setCurrentText(record["name"])
        window.group_combo.blockSignals(False)
        window.view_values[CAD_PAGE] = "measurement_overlay"
        window.show()
        window.select_page(CAD_PAGE)
        wait_render()
        assert window.view.currentData() == "source_overlay"
        assert window.view.currentText() == "来源与代表线"
        assert window.view_values[CAD_PAGE] == "source_overlay"
        assert window.view.findData("measurement_overlay") == -1
        assert window.view.findData("overlay") == -1
        expected = sum(m["valid_stations"] for m in report["measurements"].values())
        for width, height in ((1360, 860), (900, 620)):
            window.resize(width, height)
            QTest.qWait(1200)
            window.grab().save(str(destination / f"representative_{width}x{height}.png"))
            count = js("plot._fullData.filter(t => String(t.legendgroup).startsWith('measurement-'))"
                       ".reduce((n,t) => n+t.x.filter(x => x !== null && Number.isFinite(x)).length,0)")
            assert count == expected, (count, expected)
            source_count = js("plot._fullData.filter(t => String(t.legendgroup).startsWith('source-'))"
                              ".reduce((n,t) => n+t.x.length,0)")
            assert source_count == window.state.result.section_points
            spacing = json.loads(js("JSON.stringify((() => { const a=plot.querySelector('.gtitle').getBoundingClientRect(), b=plot.querySelector('.legend').getBoundingClientRect(); return {title_bottom:a.bottom,legend_top:b.top}; })())"))
            assert spacing["title_bottom"] + 4 <= spacing["legend_top"], spacing
            image = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
            assert int((pixels[:, :, :3].max(axis=2) - pixels[:, :, :3].min(axis=2) > 60).sum()) > 200
        window.resize(1360, 860)
        QTest.qWait(500)
        first = next(t for t in figures.cad_overlay(report, measurement_lines=True).data
                     if t.legendgroup and t.legendgroup.startswith("measurement-"))
        coordinates = np.asarray([[x, y] for x, y in zip(first.x, first.y) if x is not None])
        focus = coordinates[len(coordinates) // 4:len(coordinates) // 4 + 8]
        ranges = {f"{axis}axis.range": [float(focus[:, index].min() - .08),
                                      float(focus[:, index].max() + .08)]
                  for index, axis in enumerate(("x", "y"))}
        js(f"Plotly.relayout(plot, {json.dumps(ranges)}); true")
        QTest.qWait(700)
        window.canvas.grab().save(str(destination / "local_representative.png"))
        line_index = int(js("plot._fullData.findIndex(t => String(t.legendgroup).startsWith('measurement-'))"))
        js(f"Plotly.restyle(plot, {{visible:'legendonly'}}, [{line_index}]); true")
        assert js(f"plot.data[{line_index}].visible") == "legendonly"
        js(f"Plotly.restyle(plot, {{visible:true}}, [{line_index}]); true")
        assert js(f"plot.data[{line_index}].visible") is True
        js("Plotly.restyle(plot, {visible:'legendonly'}, [0]); true")
        assert js("plot.data[0].visible") == "legendonly"
        assert js(f"plot.data[{line_index}].visible") is True
        js("Plotly.restyle(plot, {visible:true}, [0]); true")
        window.view.setCurrentIndex(window.view.findData("paths"))
        wait_render()
        window.view_values[CAD_PAGE] = "overlay"
        window.select_page(CAD_PAGE)
        wait_render()
        assert window.view.currentData() == "source_overlay"
        assert window.view_values[CAD_PAGE] == "source_overlay"
        assert js("plot.layout.title.text") == "来源与测量代表线"
        assert json.dumps(records, sort_keys=True) == original
        assert all(sha256(path) == digest for path, digest in hashes.items())
        (destination / "verification.json").write_text(json.dumps(
            {"ok": True, "batch": str(batch_path), "groups": checks, "files_unchanged": True,
             "viewports": [[1360, 860], [900, 620]], "legend_toggle": True,
             "removed_views_migrated": True, "errors": errors},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(destination / "verification.json"), flush=True)
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
