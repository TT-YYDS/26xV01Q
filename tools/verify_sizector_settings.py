"""Settings persistence and Qt layout; physical capture requires --serial explicitly."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pointcloud_app.native.camera_dialog import CameraDialog
from pointcloud_app.native.window import STYLE
from pointcloud_app.pipeline import read_point_cloud
from pointcloud_app.sizector_camera import DEFAULT_EXPOSURE
from pointcloud_app.workflow_profile import load_profile, save_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Opt in to settings write, capture, export/import and restore on this device")
    args = parser.parse_args()
    destination = ROOT / "tmp" / "sizector_settings_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    protected = [*ROOT.joinpath("settings").glob("*.json"), *ROOT.joinpath("media/c1").rglob("*.ply")]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    profile, error = load_profile()
    if error:
        raise RuntimeError(error)
    original_profile = copy.deepcopy(profile)
    target = destination / "profile.json"

    def persist(preferences):
        profile["camera"] = copy.deepcopy(preferences)
        return save_profile(profile, target)

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    report = {"physical_capture_tested": bool(args.serial), "destination": str(destination)}
    dialog = CameraDialog(output_root=destination / "input", save_preferences=persist)
    original_exposure = None
    camera = None

    def idle():
        deadline = time.monotonic() + 40
        while dialog.task is not None:
            if time.monotonic() > deadline:
                raise TimeoutError(dialog.status.text())
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()

    try:
        dialog.show()
        assert dialog.views.value() == 5
        dialog.name.setText("settings_test")
        dialog.timeout.setValue(30)
        dialog.views.setValue(7)
        dialog.exposure_enabled.setChecked(True)
        dialog._set_exposure(dict(DEFAULT_EXPOSURE, number=3, gray_intensity=21.25))
        dialog.save_settings()
        stored, error = load_profile(target)
        assert not error
        restored = CameraDialog(preferences=stored["camera"])
        assert restored.preferences() == dialog.preferences()
        restored.reject()
        assert {k: v for k, v in stored.items() if k != "camera"} == {
            k: v for k, v in original_profile.items() if k != "camera"}
        report["offline_save_reload_passed"] = True
        dialog.views.setValue(5)
        dialog.exposure_enabled.setChecked(False)
        if args.serial:
            dialog.saved_serial = args.serial
            dialog.enumerate()
            idle()
            assert dialog.devices.currentData() == args.serial, dialog.status.text()
            dialog.toggle_connection()
            idle()
            assert dialog.capture.isEnabled(), dialog.status.text()
            camera = dialog.camera
            original_exposure = camera.read_exposure()
            changed = dict(original_exposure, mode=0, number=1, intensity_1=10,
                           gray_intensity=min(100.0, original_exposure["gray_intensity"] + 2.0))
            if changed["gray_intensity"] == original_exposure["gray_intensity"]:
                changed["gray_intensity"] -= 2.0
            dialog.exposure_enabled.setChecked(True)
            dialog._set_exposure(changed)
            dialog.update_controls()
            assert dialog.capture.isEnabled()
            assert "应用并采集" in dialog.capture.text()
            dialog.save_settings()
            idle()
            assert dialog.capture.isEnabled(), dialog.status.text()
            actual = camera.read_exposure()
            np.testing.assert_allclose(list(actual.values()), list(changed.values()), atol=0.001, rtol=0)
            exported = camera.export_settings(destination / "camera_settings.mpset")
            camera.disconnect()
            camera.connect(args.serial, settings_file=exported, timeout_ms=30000)
            imported = camera.read_exposure()
            np.testing.assert_allclose(list(imported.values()), list(actual.values()), atol=0.001, rtol=0)
            report["original_exposure"] = original_exposure
            report["changed_exposure"] = actual
            report["settings_export_import_passed"] = True
            dialog.tabs.setCurrentIndex(1)
            QTest.qWait(300)
            assert dialog.grab().save(str(destination / "exposure_editable.png"))
            dialog.capture_view()
            idle()
            assert len(dialog.group.records) == 1, dialog.status.text()
            assert dialog._preview_pixmap is not None and not dialog._preview_pixmap.isNull()
            cloud = read_point_cloud(dialog.group.next_path.with_name("1.ply"))
            assert len(cloud.points) == dialog.group.records[0]["points"]
            assert np.isfinite(np.asarray(cloud.points)).all()
            assert not dialog.publish.isEnabled()
            assert dialog.exposure_enabled.isEnabled()
            assert dialog.exposure_apply.isEnabled()
            assert dialog.exposure_read.isEnabled()
            report["capture"] = dialog.group.records[0]
            report["staging"] = str(dialog.group.staging)
            report["partial_group_unpublished"] = True
            report["mid_group_exposure_editing_enabled"] = True
            first_record = copy.deepcopy(dialog.group.records[0])
            first_path = dialog.group.staging / "1.ply"
            first_hash = hashlib.sha256(first_path.read_bytes()).hexdigest()
            saved_profile = target.read_bytes()
            ticks = []
            heartbeat = QTimer(dialog)
            heartbeat.setInterval(20)
            heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
            heartbeat.start()
            report["capture_after_edit"] = []
            try:
                mode = dialog.exposure_controls["mode"]
                mode.setCurrentIndex(mode.findData(0))
                dialog.exposure_controls["number"].setValue(1)
                intensity = dialog.exposure_controls["intensity_1"]
                for value in (100, 10):
                    dialog.tabs.currentWidget().ensureWidgetVisible(intensity)
                    QTest.qWait(100)
                    QTest.mouseClick(intensity.lineEdit(), Qt.MouseButton.LeftButton)
                    intensity.lineEdit().selectAll()
                    QTest.keyClicks(intensity.lineEdit(), str(value))
                    QTest.keyClick(intensity.lineEdit(), Qt.Key.Key_Return)
                    assert intensity.value() == value and dialog.task is None
                    assert dialog._exposure_pending()
                    assert dialog.capture.isEnabled() and "应用并采集" in dialog.capture.text()
                    expected = dialog._exposure_values()
                    count, tick_count = len(dialog.group.records), len(ticks)
                    started = time.monotonic()
                    QTest.mouseClick(dialog.capture, Qt.MouseButton.LeftButton)
                    assert dialog.task is not None and not dialog.capture.isEnabled()
                    idle()
                    elapsed = time.monotonic() - started
                    assert len(dialog.group.records) == count + 1, dialog.status.text()
                    assert dialog.capture.isEnabled() and not dialog._exposure_pending()
                    readback = camera.read_exposure()
                    np.testing.assert_allclose([readback[key] for key in expected], list(expected.values()),
                                               atol=0.001, rtol=0)
                    record = dialog.group.records[-1]
                    # Single-exposure frame headers can clear inactive HDR/third-exposure fields.
                    active_keys = ("mode", "number", "intensity_1", "gray_intensity", "gain")
                    np.testing.assert_allclose([record["exposure_settings"][key] for key in active_keys],
                                               [expected[key] for key in active_keys], atol=0.001, rtol=0)
                    cloud = read_point_cloud(dialog.group.staging / record["file"])
                    assert len(cloud.points) == record["points"] and record["points"] > 0
                    assert np.isfinite(np.asarray(cloud.points)).all()
                    assert len(ticks) > tick_count, "GUI did not respond during acquisition"
                    assert dialog.group.records[0] == first_record
                    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first_hash
                    assert target.read_bytes() == saved_profile, "Capture unexpectedly saved page preferences"
                    report["capture_after_edit"].append({"intensity_1": value, "capture": record,
                                                       "apply_and_capture_seconds": elapsed,
                                                       "gui_timer_ticks": len(ticks) - tick_count})
                    assert dialog.grab().save(str(destination / f"real_camera_after_edit_{value}.png"))
            finally:
                heartbeat.stop()
            report["capture_auto_applies_exposure"] = True
            report["first_capture_unchanged"] = True
        else:
            report["physical_capture_tested"] = False
            dialog.exposure_enabled.setChecked(True)
        for size, name in (((720, 750), "desktop"), ((560, 630), "small")):
            dialog.resize(*size)
            for index, page in enumerate(("capture", "exposure")):
                dialog.tabs.setCurrentIndex(index)
                scroll = dialog.tabs.currentWidget()
                scroll.verticalScrollBar().setValue(0)
                QTest.qWait(300)
                for widget in (dialog.tabs, dialog.capture, dialog.publish, dialog.preview, dialog.records,
                               dialog.status, dialog.save_button, dialog.close_button):
                    origin = widget.mapTo(dialog, widget.rect().topLeft())
                    assert origin.x() >= 0 and origin.y() >= 0
                    assert origin.x() + widget.width() <= dialog.width()
                    assert origin.y() + widget.height() <= dialog.height()
                assert dialog.grab().save(str(destination / f"{page}_{name}.png"))
                scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
                QTest.qWait(150)
                assert dialog.grab().save(str(destination / f"{page}_{name}_bottom.png"))
        report["layout_bounds_passed"] = True
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        cleanup_error = None
        try:
            idle()
            if camera is not None and original_exposure is not None:
                camera.apply_exposure(original_exposure)
                np.testing.assert_allclose(list(camera.read_exposure().values()), list(original_exposure.values()), atol=0.001, rtol=0)
                report["original_exposure_restored"] = True
        except Exception as error:
            cleanup_error = error
            report["restoration_error"] = str(error)
            report["status"] = "failed"
        camera = dialog.camera
        try:
            dialog._closing = True
            dialog.reject()
            idle()
        except Exception as error:
            cleanup_error = cleanup_error or error
            report["close_error"] = str(error)
            report["status"] = "failed"
        report["sdk_closed"] = camera is None or (camera.handle is None and camera.device is None)
        report["protected_files_unchanged"] = before == {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        if not report["protected_files_unchanged"] or not report["sdk_closed"]:
            cleanup_error = cleanup_error or RuntimeError("Protected files changed or SDK not closed")
            report["status"] = "failed"
        (destination / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
        if cleanup_error is not None:
            raise cleanup_error


if __name__ == "__main__":
    main()
