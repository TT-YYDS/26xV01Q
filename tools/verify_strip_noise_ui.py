"""Exercise the actual single-view, fused and section desktop views."""
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
    summary_path = Path(sys.argv[1]).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output = summary_path.parents[2] / "ui" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    files = sorted(map(Path, summary["input_files"]))
    profile = ROOT / "settings/workflow_profile.json"
    report_path = sorted(summary_path.parent.glob("native_cad/*/report.json"))[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hashes = {str(p): sha256(p) for p in (
        *files, profile, summary_path, Path(summary["output_files"]["merged_cloud"]),
        Path(summary["output_files"]["section_cloud"]), ROOT / "pointcloud_app/pipeline.py",
        report_path, Path(report["source_dxf"]), Path(report["source_pointcloud"]),
        ROOT / "pointcloud_app/native/window.py", ROOT / "pointcloud_app/native/backend.py",
        ROOT / "pointcloud_app/automatic_runtime.py",
    )}
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setStyle("Fusion")
    window = DesktopWindow(ROOT / "__no_autoload__")
    window._save_workflow_profile = lambda state=None: None
    state = Group(files, config_from_dict(summary), result=result_from_summary(summary_path),
                  report=report, dxf=Path(report["source_dxf"]),
                  entities=load_cad_entities(report["source_dxf"]))
    state.selection = report["desktop_input"]["selection"]
    window.groups["strip-test"] = state
    window.group_name = "strip-test"
    window.group_combo.blockSignals(True)
    window.group_combo.addItem("strip-test")
    window.group_combo.blockSignals(False)
    errors, screenshots, tables = [], [], []
    window.error = lambda message: errors.append(str(message))
    window.canvas.render_error.connect(window.error)
    window.show()

    def wait(predicate):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            app.processEvents()
            if errors:
                raise AssertionError(errors)
            if predicate():
                return
            time.sleep(.03)
        raise TimeoutError("Strip UI did not become ready")

    def ready():
        wait(lambda: window.job is None and window.canvas.generation == window.canvas.rendered_generation)

    def js(expression):
        values = []
        window.canvas.page().runJavaScript(expression, values.append)
        wait(lambda: bool(values))
        return json.loads(values[0])

    def toggle():
        for index in range(window.tools.count()):
            widget = window.tools.itemAt(index).widget()
            if isinstance(widget, QCheckBox):
                return widget
        raise AssertionError("Missing denoise toggle")

    toggle_points = {}

    def capture(name, *, strips=None, minimum=20_000, expected_type="scatter3d"):
        ready()
        QTest.qWait(500)
        info = js("""JSON.stringify((() => {
            const t = plot._fullData[0];
            let marked = 0;
            for (let i = 0; t.z && i < t.x.length; i++) {
                const x=t.x[i], y=t.y[i], z=t.z[i];
                if (x > -17 && x < -1.5 && y > 6 && y < 7 &&
                    z < -4 && z > -4.85-(x+17)*.06) marked++;
            }
            return {type: t.type, points:t.x.length, marked:marked};
        })())""")
        assert info["type"] == expected_type and info["points"] >= minimum, info
        if strips is False:
            previous = toggle_points.get("off")
            if previous is not None:
                assert info["points"] < previous * .95, (previous, info)
            toggle_points["on"] = info["points"]
        elif strips is True:
            previous = toggle_points.get("on")
            if previous is not None:
                assert info["points"] > previous * 1.15, (previous, info)
            toggle_points["off"] = info["points"]
        elif name.startswith("single_on_again") or name.startswith("single_back"):
            previous = toggle_points.get("off")
            if previous is not None:
                assert info["points"] < previous * .95, (previous, info)
            toggle_points["on"] = info["points"]
        image = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
        colored = int(np.count_nonzero(np.ptp(pixels[:, :, :3].astype(int), axis=2) > 45))
        assert colored > 300, colored
        assert window.grab().save(str(output / (name + ".png")))
        info.update(name=name, width=window.width(), height=window.height(), colored_pixels=colored)
        screenshots.append(info)
        print(json.dumps(info), flush=True)

    def capture_table(name):
        QTest.qWait(400)
        columns = [window.table.table.horizontalHeaderItem(i).text()
                   for i in range(window.table.table.columnCount())]
        assert all("CAD \u8def\u5f84\u89d2" not in column for column in columns), columns
        assert any("\u89d2\u5ea6\u5dee" in column for column in columns), columns
        assert window.table.rows and window.stack.currentWidget() is window.table
        assert window.grab().save(str(output / (name + ".png")))
        tables.append(dict(name=name, columns=columns, rows=len(window.table.rows)))

    try:
        wait(lambda: window.canvas.loaded)
        for width, height in ((1360, 860), (900, 620)):
            window.resize(width, height)
            window.view_values[1] = "view:0"
            window.select_page(1)
            ready()
            initial = state.preview
            assert toggle().isChecked()
            capture(f"single_on_{width}x{height}", strips=False)
            toggle().setChecked(False)
            capture(f"single_off_{width}x{height}", strips=True)
            toggle().setChecked(True)
            capture(f"single_on_again_{width}x{height}", strips=False)
            assert state.preview is initial
            window.single_view.setCurrentIndex(1)
            ready()
            window.single_view.setCurrentIndex(0)
            capture(f"single_back_{width}x{height}", strips=False)
            assert state.preview is initial
            window.view.setCurrentIndex(window.view.findData("merged"))
            capture(f"merged_{width}x{height}")
            window.view_values[2] = "final"
            window.select_page(2)
            capture(f"final_section_{width}x{height}", minimum=state.result.section_points)
            window.view_values[CAD_PAGE] = "overlay"
            window.select_page(CAD_PAGE)
            capture(f"cad_overlay_{width}x{height}", minimum=state.result.section_points,
                    expected_type="scattergl")
            window.view_values[CAD_PAGE] = "measurements"
            window.select_page(CAD_PAGE)
            capture_table(f"manual_results_{width}x{height}")
            window.workflow_mode = "automatic"
            window._refresh_workflow_mode_button()
            window.auto_records = {"verified": dict(name="2", report=report, elapsed_seconds=1)}
            window.select_page(CAD_PAGE)
            capture_table(f"automatic_results_{width}x{height}")
            window.workflow_mode = "manual"
            window._refresh_workflow_mode_button()
        assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
        (output / "verification.json").write_text(json.dumps(dict(
            ok=True, summary=str(summary_path), source_hashes=hashes, unchanged=True,
            errors=errors, screenshots=screenshots, toggle_cache_roundtrip=True,
            selected_view_cache_roundtrip=True, result_tables=tables,
        ), indent=2), encoding="utf-8")
        print(output, flush=True)
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(dict(
            ok=False, error=str(error), logs=window.logs, errors=errors,
        ), indent=2), encoding="utf-8")
        window.grab().save(str(output / "failure.png"))
        raise
    finally:
        if window.job:
            window.job.cancelled.set()
            window.job.wait()
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
