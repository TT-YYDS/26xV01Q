"""Check surface-filter controls and render the real processed result."""
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
from PySide6.QtWidgets import QApplication
from pointcloud_app.native.backend import config_from_dict, result_from_summary, sha256
from pointcloud_app.native.window import DesktopWindow, Group, CAD_PAGE
from pointcloud_app.native.widgets import Parameters
from pointcloud_app.cad_geometry import load_cad_entities


def main():
    report_path = Path(sys.argv[1]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary_path = report_path.parents[2] / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    destination = report_path.parents[4] / "ui" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    source_paths = [
        ROOT / "settings/workflow_profile.json", summary_path, report_path,
        Path(report["source_dxf"]), Path(report["source_pointcloud"]),
        *map(Path, summary["input_files"]),
    ]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication([])
    app.setStyle("Fusion")
    window = DesktopWindow(ROOT / "__no_autoload__")
    window._save_workflow_profile = lambda state=None: None
    config = config_from_dict(summary)
    state = Group([Path(p) for p in summary["input_files"]], config,
                  result=result_from_summary(summary_path), report=report,
                  dxf=Path(report["source_dxf"]), entities=load_cad_entities(report["source_dxf"]))
    state.selection = {"registration_paths": report["fit"]["registration_paths"],
                       "measurement_paths": report["measurement_paths"]}
    window.groups["surface-test"] = state
    window.group_name = "surface-test"
    window.group_combo.blockSignals(True)
    window.group_combo.addItem("surface-test")
    window.group_combo.blockSignals(False)
    window.show()
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.canvas.render_error.connect(window.error)
    rendered_views = []

    def wait(predicate):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            app.processEvents()
            if errors:
                raise AssertionError(errors)
            if predicate():
                return
            QTest.qWait(30)
        raise TimeoutError("UI did not become ready")

    def js(expression):
        output = []
        window.canvas.page().runJavaScript(expression, output.append)
        wait(lambda: bool(output))
        value = output[0]
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise AssertionError(f"JavaScript evaluation failed: {value}")
        return value

    def screenshot(name, minimum_points, expected_type):
        QTest.qWait(1000)
        app.processEvents()
        traces = js(
            "JSON.stringify(plot._fullData.map(t => "
            "({type:t.type, points:t.x ? t.x.length : 0})))"
        )
        assert isinstance(traces, list) and traces, traces
        assert traces[0]["type"] == expected_type, traces
        assert traces[0]["points"] >= minimum_points, traces
        image = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        p = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
        colored_pixels = int(np.count_nonzero(np.ptp(p[:, :, :3].astype(int), axis=2) > 45))
        assert colored_pixels > 150
        assert window.grab().save(str(destination / (name + ".png")))
        rendered_views.append(dict(
            name=name, width=window.width(), height=window.height(),
            colored_pixels=colored_pixels, traces=traces,
        ))

    try:
        wait(lambda: window.canvas.loaded)
        dialog = Parameters(config, window)
        dialog.page = next(i // 8 for i, (path, value) in enumerate(dialog.subset())
                           if path[-1] == "use_surface_support_filter")
        dialog.refresh()
        controls = {path[-1]: control for path, control, kind in dialog.controls}
        toggle = controls["use_surface_support_filter"]
        assert toggle.isChecked()
        toggle.setChecked(False)
        dialog.collect()
        assert not config_from_dict(dialog.payload).denoise.use_surface_support_filter
        toggle.setChecked(True)
        dialog.collect()
        assert config_from_dict(dialog.payload).denoise.use_surface_support_filter
        dialog.show()
        QTest.qWait(400)
        assert dialog.grab().save(str(destination / "parameters.png"))
        controls = {path[-1]: control for path, control, kind in dialog.controls}
        dialog.move(1)
        controls = {path[-1]: control for path, control, kind in dialog.controls}
        for name, value in (("surface_support_neighbors", 32),
                            ("surface_support_distance_factor", .35)):
            controls[name].setValue(value)
        dialog.collect()
        edited = config_from_dict(dialog.payload)
        assert edited.denoise.surface_support_neighbors == 32
        assert edited.denoise.surface_support_distance_factor == .35
        controls["surface_support_neighbors"].setValue(config.denoise.surface_support_neighbors)
        controls["surface_support_distance_factor"].setValue(config.denoise.surface_support_distance_factor)
        assert dialog.grab().save(str(destination / "surface_parameters.png"))
        dialog.apply()
        assert dialog.config.denoise == config.denoise
        parameters_path = destination / "parameters.json"
        parameters_path.write_text(json.dumps(dialog.config.as_dict(), indent=2), encoding="utf-8")
        restored = config_from_dict(json.loads(parameters_path.read_text(encoding="utf-8")))
        assert restored.denoise == config.denoise
        assert window.point_size == 1.5
        for width, height in ((1360, 860), (900, 620)):
            window.resize(width, height)
            for page, mode, minimum_points, expected_type in (
                (1, "merged", 1000, "scatter3d"),
                (2, "final", state.result.section_points, "scatter3d"),
                (CAD_PAGE, "overlay", state.result.section_points, "scattergl"),
            ):
                window.view_values[page] = mode
                window.select_page(page)
                wait(lambda: window.job is None and window.canvas.rendered_generation == window.canvas.generation)
                screenshot(f"{mode}_{width}x{height}", minimum_points, expected_type)
        assert all(sha256(path) == digest for path, digest in source_hashes.items())
        (destination / "verification.json").write_text(json.dumps(dict(
            ok=True, errors=errors, filter_toggle_roundtrip=True, default_point_size=window.point_size,
            numeric_controls_roundtrip=True, parameters_json_roundtrip=True,
            source_hashes=source_hashes, source_files_unchanged=True,
            source_report=str(report_path), viewports=[[1360, 860], [900, 620]],
            rendered_views=rendered_views,
        ), indent=2), encoding="utf-8")
        print(destination, flush=True)
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
