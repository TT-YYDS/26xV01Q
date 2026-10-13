"""Verify the reported PLY through the desktop single-view dropdown."""
import argparse
import copy
import faulthandler
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox

from pointcloud_app.native.backend import config_from_dict, result_from_summary, sha256
from pointcloud_app.native.window import DesktopWindow, Group, CAD_PAGE
from pointcloud_app.cad_geometry import load_cad_entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--cad-report", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    source = Path(baseline["source_file"])
    profile = Path(baseline["profile"])
    files = sorted(source.parent.glob("*.ply"))
    selected = files.index(source)
    destination = args.baseline.parent / "ui" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    hashes = {str(p): sha256(p) for p in (*files, profile)}
    code_hashes = {str(p): sha256(p) for p in (
        ROOT / "pointcloud_app/native/backend.py", ROOT / "pointcloud_app/native/window.py",
        ROOT / "pointcloud_app/pipeline.py", ROOT / "pointcloud_app/automatic_runtime.py",
    )}
    assert hashes[str(profile)] == baseline["source_hashes"][str(profile)]
    config = config_from_dict(json.loads(profile.read_text(encoding="utf-8")))
    original_config = copy.deepcopy(config.as_dict())
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setStyle("Fusion")
    window = DesktopWindow(ROOT / "__no_autoload__")
    window._save_workflow_profile = lambda state=None: None
    state = Group(files, config)
    window.groups["inspection-test"] = state
    window.group_name = "inspection-test"
    window.group_combo.blockSignals(True)
    window.group_combo.addItem("inspection-test")
    window.group_combo.blockSignals(False)
    errors, renders, tables = [], [], []
    window.error = lambda message: errors.append(str(message))
    window.canvas.render_error.connect(window.error)
    original_log = window.log
    def log(message):
        print(str(message).encode("ascii", "backslashreplace").decode(), flush=True)
        original_log(message)
    window.log = log
    window.show()

    def wait(predicate):
        deadline = time.monotonic() + 150
        last_progress = time.monotonic()
        while time.monotonic() < deadline:
            app.processEvents()
            if errors:
                raise AssertionError(errors)
            if predicate():
                return
            if time.monotonic() - last_progress > 15:
                print(f"Waiting: job={window.job is not None}, loaded={window.canvas.loaded}, "
                      f"rendered={window.canvas.rendered_generation}/{window.canvas.generation}",
                      flush=True)
                last_progress = time.monotonic()
            # Let the Python worker acquire the GIL between Qt event pumps.
            time.sleep(.03)
        raise TimeoutError("Desktop inspection did not become ready")

    def rendered():
        wait(lambda: window.job is None and
             window.canvas.rendered_generation == window.canvas.generation)

    def js(expression):
        output = []
        window.canvas.page().runJavaScript(expression, output.append)
        wait(lambda: bool(output))
        return json.loads(output[0])

    def denoise_control():
        for index in range(window.tools.count()):
            widget = window.tools.itemAt(index).widget()
            if isinstance(widget, QCheckBox):
                return widget
        raise AssertionError("Denoise checkbox missing")

    def capture(name, *, expect_cluster=None):
        rendered()
        QTest.qWait(800)
        data = js("""JSON.stringify((() => {
            const t = plot._fullData[0];
            let cluster = 0;
            for (let i = 0; i < t.x.length; i++)
                if (t.x[i] >= 8 && t.x[i] <= 11 &&
                    t.y[i] >= 1.5 && t.y[i] <= 4 &&
                    t.z[i] >= 2.2 && t.z[i] <= 5) cluster++;
            return {type: t.type, points: t.x.length, circled_cluster: cluster,
                    camera: plot.layout.scene.camera};
        })())""")
        assert data["type"] == "scatter3d" and data["points"] > 20_000, data
        if expect_cluster is True:
            assert data["circled_cluster"] > 40, data
        elif expect_cluster is False:
            assert data["circled_cluster"] == 0, data
        image = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
        colored = int(np.count_nonzero(np.ptp(pixels[:, :, :3].astype(int), axis=2) > 45))
        assert colored > 500, colored
        assert window.grab().save(str(destination / (name + ".png")))
        data.update(name=name, width=window.width(), height=window.height(), colored_pixels=colored,
                    denoise=window.apply_denoise, selected=window.single_view.currentData())
        renders.append(data)
        print(json.dumps(data), flush=True)
        return data

    def capture_table(name):
        QTest.qWait(400)
        columns = [window.table.table.horizontalHeaderItem(i).text()
                   for i in range(window.table.table.columnCount())]
        assert all("CAD \u8def\u5f84\u89d2" not in column for column in columns), columns
        assert any("\u89d2\u5ea6\u5dee" in column for column in columns), columns
        assert window.table.rows
        assert window.stack.currentWidget() is window.table
        assert window.grab().save(str(destination / (name + ".png")))
        scroll = window.table.table.horizontalScrollBar()
        if scroll.maximum() > 0:
            scroll.setValue(scroll.maximum())
            QTest.qWait(200)
            assert window.grab().save(str(destination / (name + "_right.png")))
            scroll.setValue(0)
        tables.append(dict(name=name, columns=columns, rows=len(window.table.rows)))

    try:
        wait(lambda: window.canvas.loaded)
        for width, height in ((1360, 860), (900, 620)):
            window.resize(width, height)
            window.view_values[1] = f"view:{selected}"
            window.select_page(1)
            rendered()
            assert window.view.currentData() == "single"
            assert window.single_view.currentData() == f"view:{selected}"
            assert denoise_control().isChecked()
            first = state.preview
            on = capture(f"denoise_on_{width}x{height}", expect_cluster=False)
            assert on["points"] == baseline["after"]["displayed"]
            np.testing.assert_allclose(first.leveling.transformation,
                                       baseline["leveling"]["transformation"], atol=1e-9)
            denoise_control().setChecked(False)
            capture(f"denoise_off_{width}x{height}", expect_cluster=True)
            denoise_control().setChecked(True)
            capture(f"denoise_on_again_{width}x{height}", expect_cluster=False)
            assert state.preview is first
            window.single_view.setCurrentIndex(0 if selected != 0 else 1)
            rendered()
            assert state.preview is not first
            window.single_view.setCurrentIndex(selected)
            capture(f"switch_back_{width}x{height}", expect_cluster=False)
            assert state.preview is first
            window.select_page(0)
            window.select_page(1)
            capture(f"page_back_{width}x{height}", expect_cluster=False)
            assert state.preview is first

        # Exercise the protected section path with the same one-file reference.
        section_state = Group([source], copy.deepcopy(config))
        window.groups["section-test"] = section_state
        window.group_name = "section-test"
        window.view_values[2] = "region"
        window.select_page(2)
        rendered()
        np.testing.assert_allclose(section_state.preview.leveling.transformation,
                                   baseline["leveling"]["transformation"], atol=1e-9)
        assert len(section_state.preview.points) == baseline["before"]["displayed"]
        assert window.grab().save(str(destination / "section_roi_protected.png"))
        window.group_name = "inspection-test"

        if args.cad_report:
            report_path = args.cad_report.resolve()
            summary_path = report_path.parents[2] / "run_summary.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for path in (report_path, summary_path, Path(report["source_dxf"]), Path(report["source_pointcloud"])):
                hashes[str(path)] = sha256(path)
            state.result = result_from_summary(summary_path)
            state.report = report
            state.dxf = Path(report["source_dxf"])
            state.entities = load_cad_entities(state.dxf)
            state.selection = {"registration_paths": report["fit"]["registration_paths"],
                               "measurement_paths": report["measurement_paths"]}
            for width, height in ((1360, 860), (900, 620)):
                window.resize(width, height)
                window.workflow_mode = "manual"
                window._refresh_workflow_mode_button()
                window.view_values[CAD_PAGE] = "measurements"
                window.select_page(CAD_PAGE)
                capture_table(f"manual_results_{width}x{height}")
                window.workflow_mode = "automatic"
                window._refresh_workflow_mode_button()
                window.auto_records = {"verified": dict(name="2", report=report, elapsed_seconds=1)}
                window.select_page(CAD_PAGE)
                capture_table(f"automatic_results_{width}x{height}")
            window.workflow_mode = "manual"
        assert state.config.as_dict() == original_config
        assert all(sha256(Path(p)) == digest for p, digest in hashes.items())
        assert all(sha256(Path(p)) == digest for p, digest in code_hashes.items())
        result = dict(
            ok=True, source_file=str(source), baseline=str(args.baseline.resolve()),
            source_hashes=hashes, source_files_unchanged=True, config_unchanged=True,
            code_hashes=code_hashes,
            leveling_unchanged=True, cache_toggle_roundtrip=True, cache_view_roundtrip=True,
            section_roi_protection_preserved=True, errors=errors, rendered_views=renders, result_tables=tables,
        )
        (destination / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(destination, flush=True)
    except Exception as error:
        faulthandler.dump_traceback()
        failure = dict(ok=False, error=str(error), logs=window.logs, errors=errors,
                       job_active=window.job is not None, canvas_loaded=window.canvas.loaded,
                       generation=window.canvas.generation, rendered=window.canvas.rendered_generation,
                       pending=window.canvas.pending)
        (destination / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        window.grab().save(str(destination / "failure.png"))
        print(destination, flush=True)
        raise
    finally:
        if window.job:
            window.job.cancelled.set()
            window.job.wait()
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
