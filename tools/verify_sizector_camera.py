"""Offline SDK and Qt acquisition regression. No physical camera is triggered."""

from __future__ import annotations

import ctypes as ct
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from pointcloud_app.sizector_camera import (
    SizectorCamera, DEFAULT_SDK, DEFAULT_EXPOSURE, Frame, CameraError, frame_points, native_path,
    validate_exposure, ExposureSettings,
)
from pointcloud_app.workflow_profile import empty_profile
from pointcloud_app.desktop_runtime import discover_stable_input_groups
from pointcloud_app.pipeline import read_point_cloud
from pointcloud_app.native.camera_dialog import CameraDialog
from pointcloud_app.native.window import DesktopWindow


def fingerprint(paths):
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main():
    destination = ROOT / "tmp" / "sizector_camera_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    source = DEFAULT_SDK / "06_Demo" / "TestData" / "float3DV1.5.mpdat"
    protected = [source, DEFAULT_SDK / "02_Binary/02_SizectorS_API/x64/MPSizectorS_API.dll",
                 *ROOT.glob("*.dxf"), *ROOT.joinpath("settings").glob("*.json"),
                 *ROOT.joinpath("media/c1").rglob("*.ply")]
    before = fingerprint(protected)
    probe = SizectorCamera()
    probe._load()
    frame = Frame()
    try:
        assert probe.api.MPSizectorS_Utils_Load(ct.byref(frame), native_path(source), None, None)
        xyz, gray, preview = frame_points(frame)
        device_info = frame.FrameInfo.DeviceInfo
        expected = len(xyz)
        fixture_exposure = ExposureSettings.from_buffer_copy(bytes(frame.FrameInfo.Settings[16:48])).values()
    finally:
        if frame.Data:
            probe.api.MPSizectorS_Utils_FreeDataFrame(frame)
    enumeration = {"devices": None, "error": None}
    try:
        enumeration["devices"] = len(probe.enumerate())
    except CameraError as error:
        enumeration["error"] = str(error)
    finally:
        probe.close()

    class OfflineCamera(SizectorCamera):
        def enumerate(self):
            self._load()
            return [device_info]

        def connect(self, serial, *args):
            assert serial == device_info.DeviceSerialNumber
            self.device = device_info
            return device_info

        def acquire(self, path, timeout_ms=10000, **kwargs):
            return super().acquire(path, timeout_ms, mpdat=source)

        def read_exposure(self):
            return dict(getattr(self, "exposure", DEFAULT_EXPOSURE))

        def apply_exposure(self, values):
            self.exposure = validate_exposure(values)
            return self.read_exposure()

        def disconnect(self):
            self.device = None

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")

    def wait(predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.02)
        raise TimeoutError("Qt camera operation did not finish")

    with patch("pointcloud_app.native.window.load_profile", return_value=(empty_profile(), None)), \
         patch("pointcloud_app.native.window.save_profile", return_value=destination / "unused_profile.json"):
        window = DesktopWindow(destination / "nonexistent")
        window.show()
        window.resize(1360, 860)
        QTest.qWait(300)
        assert window.camera_button.isEnabled()
        window.grab().save(str(destination / "data_desktop.png"))
        root = destination / "input"
        root.mkdir()
        dialog = CameraDialog(window, output_root=root,
                              preferences={"preferences_version": 2, "expected_views": 6},
                              save_preferences=window._camera_preferences_changed)
        dialog.camera = OfflineCamera()
        dialog.group_ready.connect(window._camera_group_ready)
        dialog.preferences_changed.connect(window._camera_preferences_changed)
        published = []
        dialog.group_ready.connect(published.append)
        dialog.show()
        dialog.enumerate()
        wait(lambda: dialog.task is None)
        assert dialog.devices.count() == 1
        dialog.toggle_connection()
        wait(lambda: dialog.task is None)
        assert dialog.capture.isEnabled()
        assert not dialog.mode.isEnabled()
        dialog.name.setText("offline_sdk_group")
        for index in range(6):
            if index == 1:
                first_record = json.loads(json.dumps(dialog.group.records[0]))
                first_ply = fingerprint([dialog.group.staging / "1.ply"])
                assert dialog.exposure_enabled.isEnabled()
                dialog.tabs.setCurrentIndex(1)
                QTest.qWait(100)
                QTest.mouseClick(dialog.exposure_enabled, Qt.MouseButton.LeftButton,
                                 pos=QPoint(8, dialog.exposure_enabled.height() // 2))
                assert dialog.exposure_enabled.isChecked()
                number = dialog.exposure_controls["number"]
                assert number.isEnabled()
                QTest.mouseClick(number.lineEdit(), Qt.MouseButton.LeftButton)
                number.lineEdit().selectAll()
                QTest.keyClicks(number.lineEdit(), "3")
                QTest.keyClick(number.lineEdit(), Qt.Key.Key_Return)
                assert number.value() == 3
                assert dialog.exposure_controls["intensity_2"].isEnabled()
                assert dialog.exposure_controls["intensity_3"].isEnabled()
                intensity = dialog.exposure_controls["intensity_1"]
                QTest.mouseClick(intensity.lineEdit(), Qt.MouseButton.LeftButton)
                intensity.lineEdit().selectAll()
                QTest.keyClicks(intensity.lineEdit(), "100")
                QTest.keyClick(intensity.lineEdit(), Qt.Key.Key_Return)
                assert intensity.value() == 100
                assert dialog.capture.isEnabled()
                assert "应用并采集" in dialog.capture.text()
                assert dialog.group.records[0] == first_record
                assert fingerprint([dialog.group.staging / "1.ply"]) == first_ply
                for size, name in (((720, 750), "desktop"), ((560, 630), "small")):
                    dialog.resize(*size)
                    dialog.tabs.currentWidget().verticalScrollBar().setValue(0)
                    QTest.qWait(250)
                    dialog.grab().save(str(destination / f"exposure_after_first_{name}.png"))
                    scroll = dialog.tabs.currentWidget()
                    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
                    QTest.qWait(150)
                    dialog.grab().save(str(destination / f"exposure_after_first_{name}_bottom.png"))
            QTest.mouseClick(dialog.capture, Qt.MouseButton.LeftButton)
            wait(lambda: dialog.task is None)
            assert len(dialog.group.records) == index + 1, dialog.status.text()
            if index == 1:
                assert dialog.camera.read_exposure()["intensity_1"] == 100
                assert not dialog._exposure_pending()
                assert dialog.group.records[0] == first_record
                assert fingerprint([dialog.group.staging / "1.ply"]) == first_ply
                dialog.grab().save(str(destination / "exposure_after_auto_capture.png"))
            assert dialog.group.records[-1]["exposure_settings"] == fixture_exposure
            assert dialog.records.rowCount() == index + 1
            assert discover_stable_input_groups(root, minimum_file_age_seconds=0) == {}
            assert dialog.publish.isEnabled() == (index == 5)
        assert not dialog.capture.isEnabled()
        dialog.save_settings()
        wait(lambda: dialog.task is None)
        assert dialog.status.text() == "相机采集设置已保存"
        assert window.workflow_profile["camera"]["exposure"]["intensity_1"] == 100
        dialog.tabs.setCurrentIndex(0)
        assert dialog._preview_pixmap is not None and not dialog._preview_pixmap.isNull()
        assert preview.max() > preview.min()
        for size, name in (((720, 750), "camera_desktop"), ((560, 630), "camera_small")):
            dialog.resize(*size)
            QTest.qWait(250)
            dialog.grab().save(str(destination / (name + ".png")))
            for widget in (dialog.capture, dialog.connection, dialog.publish, dialog.devices, dialog.preview,
                           dialog.records, dialog.status, dialog.close_button):
                origin = widget.mapTo(dialog, widget.rect().topLeft())
                assert origin.x() >= 0 and origin.y() >= 0
                assert origin.x() + widget.width() <= dialog.width(), widget.objectName()
                assert origin.y() + widget.height() <= dialog.height(), widget.objectName()
        dialog.publish_group()
        wait(lambda: dialog.task is None)
        assert len(published) == 1 and window.group_name == "offline_sdk_group"
        wait(lambda: window.job is None)
        discovered = discover_stable_input_groups(root, minimum_file_age_seconds=0)
        assert len(discovered["offline_sdk_group"]) == 6
        cloud = read_point_cloud(discovered["offline_sdk_group"][0])
        np.testing.assert_array_equal(np.asarray(cloud.points), xyz)
        np.testing.assert_allclose(np.asarray(cloud.colors)[:, 0], gray / 255, atol=1e-15)
        assert window.workflow_profile["camera"]["expected_views"] == 6
        hardware_dialog = CameraDialog(window, output_root=root)
        hardware_dialog.show()
        hardware_dialog.enumerate()
        wait(lambda: hardware_dialog.task is None)
        if hardware_dialog.devices.count() == 0:
            assert not hardware_dialog.capture.isEnabled() and not hardware_dialog.connection.isEnabled()
        hardware_dialog.grab().save(str(destination / "native_device_search.png"))
        hardware_dialog.reject()
        wait(lambda: not hardware_dialog.isVisible())
        window.resize(900, 620)
        window.grab().save(str(destination / "imported_small.png"))
        dialog.reject()
        wait(lambda: not dialog.isVisible())
        assert dialog.task is None
        window.apply_denoise = False
        window.level = 0
        window.data_row_clicked(0, 0)
        wait(lambda: window.job is None and window.single_file_view, 60)
        wait(lambda: window.canvas.rendered_generation == window.canvas.generation, 60)
        traces = []
        window.canvas.page().runJavaScript(
            "JSON.stringify((plot._fullData || []).map(t => ({type:t.type, points:t.x ? t.x.length : 0})))",
            traces.append)
        wait(lambda: bool(traces))
        traces = [json.loads(traces[0])]
        assert any(trace["type"] == "scatter3d" and trace.get("points", 0) > 1000
                   for trace in traces[0]), traces
        for size, name in (((1360, 860), "cloud_desktop"), ((900, 620), "cloud_small")):
            window.resize(*size)
            window.activateWindow()
            QTest.qWait(2000)
            dimensions = []
            window.canvas.page().runJavaScript(
                "JSON.stringify([plot.clientWidth, plot.clientHeight, plot._fullLayout.width, plot._fullLayout.height])",
                dimensions.append)
            wait(lambda: bool(dimensions))
            dimensions = json.loads(dimensions[0])
            assert dimensions[:2] == dimensions[2:], dimensions
            window.grab().save(str(destination / (name + ".png")))
            pixels = window.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            values = np.frombuffer(pixels.constBits(), dtype=np.uint8).reshape(pixels.height(), pixels.width(), 4)
            colored = values[:, :, :3].max(axis=2).astype(int) - values[:, :, :3].min(axis=2).astype(int)
            assert np.sum(colored > 45) > 500, "Blank point-cloud canvas"
            assert np.sum(colored[:, -4:] > 45) < 30, "Point-cloud scene clipped at right edge"
        window.close()
        wait(lambda: not window.isVisible())

        # Close during an in-flight operation: await completion, then release the camera.
        dialog = CameraDialog(output_root=root)
        camera = OfflineCamera()
        dialog.camera = camera
        dialog.show()
        dialog._run(lambda progress: time.sleep(0.2), lambda _: None)
        dialog.reject()
        assert dialog.isVisible()
        wait(lambda: not dialog.isVisible())
        assert dialog.task is None and camera.handle is None

    after = fingerprint(protected)
    assert before == after
    result = {"ok": True, "sdk_version": probe.version, "physical_capture_tested": False,
              "native_device_enumeration": enumeration, "source": str(source), "valid_points_per_view": expected,
              "resolution": [preview.shape[1], preview.shape[0]], "group": str(published[0]),
              "views": 6, "coordinates_and_colors_equal": True, "partial_group_invisible": True,
              "mid_group_exposure_editing_passed": True, "capture_auto_applies_exposure": True,
              "edited_intensity_1": 100, "per_view_exposure_from_fixture": fixture_exposure,
              "close_during_operation": True, "native_3d_preview_nonblank": True, "protected_sha256": after}
    (destination / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
    print(json.dumps({key: value for key, value in result.items() if key != "protected_sha256"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
