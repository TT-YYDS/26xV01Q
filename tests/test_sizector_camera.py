import ctypes as ct
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pointcloud_app.sizector_camera import (
    CameraError, CaptureGroup, DataInfo, DeviceInfo, FLOAT_POINT, Frame, HeadPack,
    SizectorCamera, frame_points, write_capture_ply, ExposureBasic, ExposureSettings,
    DEFAULT_EXPOSURE, validate_exposure,
)
from pointcloud_app.desktop_runtime import discover_stable_input_groups
from pointcloud_app.pipeline import read_point_cloud


def make_frame():
    data = np.zeros(6, dtype=FLOAT_POINT)
    data["x"] = [1, 2, 3, 4, np.nan, 6]
    data["y"], data["z"] = 2, -3
    data["gray"] = [10, 20, 30, 40, 50, 60]
    data["mask"] = [0, 8, 1, 3, 0, 0]
    frame = Frame()
    frame.FrameInfo.DataInfo.XPixResolution = 3
    frame.FrameInfo.DataInfo.YPixResolution = 2
    exposure = ExposureSettings()
    exposure.update_values(DEFAULT_EXPOSURE)
    frame.FrameInfo.Settings[16:48] = bytes(exposure)
    frame.Data = data.ctypes.data_as(ct.POINTER(ct.c_ubyte))
    return frame, data


class FakeAPI:
    def __init__(self):
        self.freed = 0
        self.closed = 0
        self.deleted = 0
        self.fail_setting = False
        self.fail_snap = False
        self.state = 2
        self.frame, self.data = make_frame()
        self.info = DeviceInfo()
        self.info.DeviceName = "Sizector"
        self.info.DeviceSerialNumber = "123"
        self.exposure = ExposureSettings()
        self.exposure.update_values(DEFAULT_EXPOSURE)
        self.exposure.MultiHeadExpoVarMode = 7
        self.exposure.MultiHeadExpoVarRatio = 2
        self.hold = False
        self.exposure_writes = 0
        self.reject_exposure = False
        self.ignore_exposure = False

    def MPSizectorS_Create(self, *_):
        return 42

    def MPSizectorS_Delete(self, *_):
        self.deleted += 1

    def MPSizectorS_SetAutoReconnect(self, *_):
        pass

    def MPSizectorS_UpdateDeviceList(self, *_):
        return True

    def MPSizectorS_GetDeviceCount(self, *_):
        return 1

    def MPSizectorS_GetDeviceInfo(self, handle, target, index):
        ct.memmove(target, ct.byref(self.info), ct.sizeof(self.info))

    def MPSizectorS_Open(self, *_):
        return True

    def MPSizectorS_Close(self, *_):
        self.closed += 1

    def MPSizectorS_GetDeviceState(self, *_):
        return self.state

    def MPSizectorS_GetHoldState(self, handle, target):
        ct.cast(target, ct.POINTER(ct.c_bool)).contents.value = self.hold
        return True

    def MPSizectorS_SetHoldState(self, handle, value):
        if self.fail_setting:
            return False
        self.hold = value
        return True

    def MPSizectorS_GetExposureSettings(self, handle, target):
        ct.memmove(target, ct.byref(self.exposure), ct.sizeof(self.exposure))
        return True

    def MPSizectorS_SetExposureSettings(self, handle, settings):
        self.exposure_writes += 1
        if self.reject_exposure and self.exposure_writes == 1:
            return False
        if not self.ignore_exposure:
            self.exposure = ExposureSettings.from_buffer_copy(settings)
        return True

    def MPSizectorS_SetWorkingMode(self, *_):
        return not self.fail_setting

    MPSizectorS_SetTriggerSource = MPSizectorS_SetWorkingMode
    MPSizectorS_SetDataOutMode = MPSizectorS_SetWorkingMode

    def MPSizectorS_Snap(self, handle, trigger, format_target, frame_target, timeout):
        self.snap_timeout = timeout
        self.frame.FrameInfo.Settings[16:48] = bytes(self.exposure)
        ct.cast(format_target, ct.POINTER(ct.c_int)).contents.value = 0
        ct.memmove(frame_target, ct.byref(self.frame), ct.sizeof(self.frame))
        return not self.fail_snap

    def MPSizectorS_Utils_Load(self, frame_target, *_):
        ct.memmove(frame_target, ct.byref(self.frame), ct.sizeof(self.frame))
        return not self.fail_snap

    def MPSizectorS_Utils_FreeDataFrame(self, *_):
        self.freed += 1


class SizectorTests(unittest.TestCase):
    def camera(self):
        camera = SizectorCamera()
        camera.api, camera.version = FakeAPI(), 2.776
        return camera

    def test_native_layout_matches_supplied_x64_header(self):
        self.assertEqual(ct.sizeof(DeviceInfo), 192)
        self.assertEqual(ct.sizeof(DataInfo), 32)
        self.assertEqual(ct.sizeof(HeadPack), 2048)
        self.assertEqual(ct.alignment(HeadPack), 8)
        self.assertEqual(ct.sizeof(Frame), 2056)
        self.assertEqual(Frame.Data.offset, 2048)
        self.assertEqual(FLOAT_POINT.itemsize, 16)
        self.assertEqual(ct.sizeof(ExposureBasic), 8)
        self.assertEqual(ct.sizeof(ExposureSettings), 32)
        self.assertEqual(ExposureSettings.UserGain.offset, 8)
        self.assertEqual(ExposureSettings.ExposureIntensity3D_1st.offset, 12)
        self.assertEqual(ExposureSettings.MultiHeadExpoVarMode.offset, 28)

    def test_mask_filters_invalid_but_keeps_hdr_and_negative_coordinates(self):
        frame, allocation = make_frame()
        xyz, gray, image = frame_points(frame)
        np.testing.assert_array_equal(xyz[:, 0], [1, 2, 6])
        np.testing.assert_array_equal(xyz[:, 2], [-3, -3, -3])
        np.testing.assert_array_equal(gray, [10, 20, 60])
        self.assertEqual(image.shape, (2, 3))
        allocation["gray"][:] = 0
        self.assertEqual(image[0, 1], 20)

    def test_rejects_2d_empty_and_invalid_resolution(self):
        frame, _ = make_frame()
        for kind in (1, 3, 5, 8):
            frame.FrameInfo.DataInfo.DataFormat = kind
            with self.assertRaises(CameraError):
                frame_points(frame)
        frame.FrameInfo.DataInfo.DataFormat = 0
        frame.FrameInfo.DataInfo.XPixResolution = 0
        with self.assertRaises(CameraError):
            frame_points(frame)

    def test_capture_ply_roundtrip_and_no_overwrite(self):
        frame, _ = make_frame()
        xyz, gray, _ = frame_points(frame)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "1.ply"
            write_capture_ply(path, xyz, gray)
            original = path.read_bytes()
            cloud = read_point_cloud(path)
            np.testing.assert_array_equal(np.asarray(cloud.points), xyz)
            np.testing.assert_allclose(np.asarray(cloud.colors)[:, 0], gray / 255)
            with self.assertRaises(CameraError):
                write_capture_ply(path, xyz + 1, gray)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.capture")), [])

    def test_capture_borrows_frame_and_closes_device_and_instance(self):
        camera = self.camera()
        camera.connect("123")
        with TemporaryDirectory() as directory:
            result = camera.acquire(Path(directory) / "1.ply", 17000)
        self.assertEqual(result["points"], 3)
        self.assertEqual(camera.api.snap_timeout, 17000)
        self.assertEqual(camera.api.freed, 0)
        camera.close()
        camera.close()
        self.assertEqual(camera.api.closed, 1)
        self.assertEqual(camera.api.deleted, 1)

    def test_failed_capture_does_not_free_camera_buffer_or_save(self):
        camera = self.camera()
        camera.connect("123")
        camera.api.fail_snap = True
        with TemporaryDirectory() as directory:
            path = Path(directory) / "1.ply"
            with self.assertRaises(CameraError):
                camera.acquire(path)
            self.assertFalse(path.exists())
        self.assertEqual(camera.api.freed, 0)
        camera.close()

    def test_write_failure_does_not_free_camera_buffer(self):
        camera = self.camera()
        camera.connect("123")
        with patch("pointcloud_app.sizector_camera.write_capture_ply", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                camera.acquire(Path("never_written.ply"))
        self.assertEqual(camera.api.freed, 0)
        camera.close()

    def test_offline_frame_is_freed_on_success_load_failure_and_write_failure(self):
        for failure in (None, "load", "write"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                camera = self.camera()
                source = Path(directory) / "sample.mpdat"
                source.touch()
                target = Path(directory) / "1.ply"
                if failure == "load":
                    camera.api.fail_snap = True
                    with self.assertRaises(CameraError):
                        camera.acquire(target, mpdat=source)
                elif failure == "write":
                    with patch("pointcloud_app.sizector_camera.write_capture_ply", side_effect=OSError("disk full")):
                        with self.assertRaises(OSError):
                            camera.acquire(target, mpdat=source)
                else:
                    result = camera.acquire(target, mpdat=source)
                    self.assertEqual(result["points"], 3)
                self.assertEqual(target.exists(), failure is None)
                self.assertEqual(camera.api.freed, 1)
                camera.close()

    def test_live_capture_copies_preview_and_points_before_buffer_reuse(self):
        camera = self.camera()
        camera.connect("123")
        with TemporaryDirectory() as directory:
            target = Path(directory) / "1.ply"
            result = camera.acquire(target)
            camera.api.data["gray"][:] = 0
            camera.api.data["x"][:] = 99
            self.assertEqual(result["image"][0, 1], 20)
            cloud = read_point_cloud(target)
            np.testing.assert_array_equal(np.asarray(cloud.points)[:, 0], [1, 2, 6])
        self.assertEqual(camera.api.freed, 0)
        camera.close()

    def test_setting_failure_disconnects_and_stale_serial_is_rejected(self):
        camera = self.camera()
        with self.assertRaises(CameraError):
            camera.connect("missing")
        camera.api.fail_setting = True
        with self.assertRaises(CameraError):
            camera.connect("123")
        self.assertIsNone(camera.device)
        self.assertEqual(camera.api.closed, 1)
        camera.close()

    def test_capture_without_ready_device_and_bad_timeout_are_rejected(self):
        camera = self.camera()
        with self.assertRaises(CameraError):
            camera.acquire(Path("no_device.ply"))
        camera.connect("123")
        camera.api.state = 0
        with self.assertRaises(CameraError):
            camera.acquire(Path("offline.ply"))
        with self.assertRaises(CameraError):
            camera.acquire(Path("bad_timeout.ply"), 0)
        self.assertEqual(camera.api.freed, 0)
        camera.close()

    def test_group_is_invisible_until_complete_then_reaches_existing_discovery(self):
        frame, _ = make_frame()
        xyz, gray, _ = frame_points(frame)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            root.mkdir()
            group = CaptureGroup(root, "camera_test", 6)
            for index in range(6):
                path = group.next_path
                write_capture_ply(path, xyz, gray)
                group.record({"file": path.name, "points": 3, "image": np.zeros((2, 3))})
                self.assertEqual(discover_stable_input_groups(root, minimum_file_age_seconds=0), {})
                if index < 5:
                    with self.assertRaises(CameraError):
                        group.publish()
            target = group.publish()
            found = discover_stable_input_groups(root, minimum_file_age_seconds=0)
            self.assertEqual(list(found), ["camera_test"])
            self.assertEqual([path.name for path in found["camera_test"]], [f"{i}.ply" for i in range(1, 7)])
            self.assertFalse(group.staging.exists())
            manifest = json.loads((target / "camera_capture.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["unit"], "mm")
            self.assertNotIn("image", manifest["views"][0])
            with self.assertRaises(CameraError):
                group.publish()

    def test_group_validation_rejects_unsafe_names_and_existing_groups(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            root.mkdir()
            for name in ("../x", "a/b", "a\\b", "CON", "com1.txt", "x:", "x.", ""):
                with self.subTest(name=name), self.assertRaises(CameraError):
                    CaptureGroup(root, name)
            (root / "existing").mkdir()
            with self.assertRaises(CameraError):
                CaptureGroup(root, "existing")
            with self.assertRaises(CameraError):
                CaptureGroup(root, "new", 4)

    def test_profile_rebuild_preserves_camera_settings_without_changing_template(self):
        from pointcloud_app.workflow_profile import empty_profile
        from pointcloud_app.native.window import DesktopWindow
        payload = empty_profile()
        payload["camera"] = {"sdk_root": "custom SDK", "expected_views": 6}
        owner = SimpleNamespace(workflow_mode="automatic", workflow_profile=payload, state=None,
                                source_kind="camera", source_path=Path("media"), _profile_config=lambda: None)
        with patch("pointcloud_app.native.window.save_profile") as save:
            DesktopWindow._save_workflow_profile(owner)
        self.assertEqual(save.call_args.args[0]["camera"], payload["camera"])

    def test_published_group_routes_to_automatic_intake_without_replacing_state(self):
        from pointcloud_app.native.window import DesktopWindow
        owner = SimpleNamespace(workflow_mode="automatic", auto_enabled=True, groups={"old": object()},
                                log=Mock(), _save_workflow_profile=Mock(), auto_poll=Mock())
        DesktopWindow._camera_group_ready(owner, Path("media/camera/new"))
        self.assertEqual(owner.source_kind, "camera")
        owner.auto_poll.assert_called_once()
        self.assertEqual(list(owner.groups), ["old"])

    def test_import_includes_camera_subgroups_alongside_old_direct_ply_group(self):
        from pointcloud_app.native.window import DesktopWindow
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1.ply").touch()
            child = root / "camera_new"
            child.mkdir()
            (child / "1.ply").touch()
            owner = SimpleNamespace(_start=Mock())
            DesktopWindow.import_root(owner, root)
            operation = owner._start.call_args.args[0]
            found = operation(Mock())
            self.assertEqual(set(found), {root.name, "camera_new"})

    def test_unknown_sdk_version_is_rejected_before_hardware_creation(self):
        with TemporaryDirectory() as directory:
            sdk = Path(directory)
            dll = sdk / "02_Binary/02_SizectorS_API/x64/MPSizectorS_API.dll"
            dll.parent.mkdir(parents=True)
            dll.touch()
            api = Mock()
            api.MPSizectorS_GetAPIVersion.return_value = 2.8
            entry = Mock()
            camera = SizectorCamera(sdk)
            with patch("pointcloud_app.sizector_camera.ct.CDLL", return_value=api), \
                 patch("pointcloud_app.sizector_camera.os.add_dll_directory", return_value=entry):
                with self.assertRaises(CameraError):
                    camera._load()
            self.assertIsNone(camera.api)
            self.assertTrue(entry.close.called)
            api.MPSizectorS_Create.assert_not_called()

    def test_invalid_saved_camera_preferences_fall_back_to_safe_controls(self):
        from PySide6.QtWidgets import QApplication
        from pointcloud_app.native.camera_dialog import CameraDialog
        app = QApplication.instance() or QApplication([])
        dialog = CameraDialog(preferences={"sdk_root": 42, "timeout_seconds": 10 ** 100,
                                           "expected_views": "bad", "working_mode": []})
        self.assertEqual(dialog.timeout.value(), 120)
        self.assertEqual(dialog.views.value(), 5)
        self.assertEqual(dialog.mode.currentData(), 2)
        self.assertFalse(dialog.capture.isEnabled())
        dialog.reject()
        dialog.deleteLater()
        app.processEvents()

    def test_exposure_validation_rejects_invalid_values_before_device_write(self):
        camera = self.camera()
        camera.connect("123")
        for key, bad in (("mode", 4), ("number", 0), ("number", 1.5), ("gain", 4),
                         ("hdr_priority", 8), ("phdr_quality", 101), ("intensity_1", float("nan")),
                         ("gray_intensity", float("inf")), ("intensity_2", 0), ("gain", True)):
            with self.subTest(key=key, bad=bad), self.assertRaises(CameraError):
                camera.apply_exposure(dict(DEFAULT_EXPOSURE, **{key: bad}))
        self.assertEqual(camera.api.exposure_writes, 0)
        with self.assertRaises(CameraError):
            validate_exposure({})
        camera.close()

    def test_apply_exposure_readback_preserves_unexposed_fields_and_hold(self):
        camera = self.camera()
        camera.connect("123")
        for held in (False, True):
            with self.subTest(held=held):
                camera.api.hold = held
                actual = camera.apply_exposure(dict(DEFAULT_EXPOSURE, gray_intensity=30.25, number=3))
                self.assertAlmostEqual(actual["gray_intensity"], 30.25)
                self.assertEqual(actual["number"], 3)
                self.assertEqual(camera.api.hold, held)
                self.assertEqual(camera.api.exposure.MultiHeadExpoVarMode, 7)
                self.assertEqual(camera.api.exposure.MultiHeadExpoVarRatio, 2)
        camera.close()

    def test_rejected_and_mismatched_exposure_restore_previous_settings(self):
        for reject, ignore in ((True, False), (False, True)):
            with self.subTest(reject=reject):
                camera = self.camera()
                camera.connect("123")
                original = camera.read_exposure()
                camera.api.reject_exposure, camera.api.ignore_exposure = reject, ignore
                with self.assertRaises(CameraError):
                    camera.apply_exposure(dict(DEFAULT_EXPOSURE, intensity_1=25.0))
                self.assertEqual(camera.read_exposure(), original)
                self.assertEqual(camera.api.exposure_writes, 2)
                self.assertFalse(camera.api.hold)
                camera.close()

    def test_exposure_restore_failure_disconnects_camera(self):
        camera = self.camera()
        camera.connect("123")
        camera.api.MPSizectorS_SetExposureSettings = Mock(return_value=False)
        with self.assertRaisesRegex(CameraError, "已断开"):
            camera.apply_exposure(DEFAULT_EXPOSURE)
        self.assertIsNone(camera.device)
        camera.close()

    def test_default_and_migrated_view_counts_and_offline_settings_save(self):
        from PySide6.QtWidgets import QApplication
        from pointcloud_app.native.camera_dialog import CameraDialog
        app = QApplication.instance() or QApplication([])
        for preferences, expected in (({}, 5), ({"expected_views": 6}, 5),
                                      ({"preferences_version": 2, "expected_views": 6}, 6),
                                      ({"expected_views": 8}, 8)):
            with self.subTest(preferences=preferences):
                save = Mock(return_value=True)
                dialog = CameraDialog(preferences=preferences, save_preferences=save)
                self.assertEqual(dialog.views.value(), expected)
                dialog.views.setValue(7)
                self.assertIn("7", str(dialog.preferences()["expected_views"]))
                dialog.name.setText("named_group")
                dialog.timeout.setValue(22)
                dialog.exposure_enabled.setChecked(True)
                dialog.exposure_controls["gray_intensity"].setValue(25.125)
                dialog.save_settings()
                stored = save.call_args.args[0]
                self.assertEqual(stored["group_name"], "named_group")
                self.assertEqual(stored["timeout_seconds"], 22)
                restored = CameraDialog(preferences=stored)
                self.assertEqual(restored.preferences(), stored)
                restored.reject()
                dialog.reject()
                app.processEvents()

    def test_camera_settings_save_only_changes_camera_section_and_reports_failure(self):
        from pointcloud_app.native.window import DesktopWindow
        from pointcloud_app.workflow_profile import empty_profile, save_profile, load_profile
        payload = empty_profile()
        payload["cad"]["selection"] = {"keep": [1, 2]}
        owner = SimpleNamespace(workflow_profile=payload, log=Mock())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            with patch("pointcloud_app.native.window.save_profile",
                       side_effect=lambda value: save_profile(value, path)):
                self.assertEqual(DesktopWindow._camera_preferences_changed(owner, {"expected_views": 5}), path)
            stored, error = load_profile(path)
            self.assertIsNone(error)
            self.assertEqual({k: v for k, v in stored.items() if k != "camera"}, payload)
            previous = owner.workflow_profile
            with patch("pointcloud_app.native.window.save_profile", side_effect=OSError("disk full")):
                self.assertFalse(DesktopWindow._camera_preferences_changed(owner, {"expected_views": 8}))
            self.assertIs(owner.workflow_profile, previous)

    def test_reopening_camera_preserves_saved_output_except_fixed_automatic_intake(self):
        from pointcloud_app.native.window import DesktopWindow
        for automatic, saved_output, expected in ((False, "saved/input", None),
                                                   (True, "saved/input", Path("watched/input")),
                                                   (False, "", Path("watched/input"))):
            with self.subTest(automatic=automatic, saved_output=saved_output):
                owner = SimpleNamespace(job=None, workflow_mode="automatic", auto_enabled=automatic,
                                        workflow_profile={"camera": {"output_root": saved_output}},
                                        source_kind="camera", source_path=Path("watched/input"),
                                        _camera_preferences_changed=Mock(), _camera_group_ready=Mock())
                with patch("pointcloud_app.native.window.CameraDialog") as dialog:
                    DesktopWindow.open_camera(owner)
                self.assertEqual(dialog.call_args.kwargs["output_root"], expected)
                self.assertEqual(dialog.call_args.kwargs["fixed_output"], automatic)
                dialog.return_value.exec.assert_called_once()

    def test_ui_exposure_pending_save_and_mid_group_editing(self):
        from PySide6.QtWidgets import QApplication
        from pointcloud_app.native.camera_dialog import CameraDialog
        app = QApplication.instance() or QApplication([])
        save = Mock(return_value=True)
        dialog = CameraDialog(save_preferences=save)
        camera = self.camera()
        camera.connect("123")
        dialog.camera = camera
        dialog._exposure_loaded(camera.read_exposure())
        dialog.exposure_enabled.setChecked(True)

        def run(operation, callback):
            callback(operation(Mock()))
            dialog.update_controls()

        dialog._run = Mock(side_effect=run)
        try:
            dialog.exposure_controls["gray_intensity"].setValue(27.25)
            self.assertTrue(dialog.capture.isEnabled())
            self.assertIn("应用并采集", dialog.capture.text())
            dialog.save_settings()
            self.assertTrue(dialog.capture.isEnabled())
            self.assertEqual(save.call_args.args[0]["exposure"]["gray_intensity"], 27.25)
            self.assertEqual(camera.read_exposure()["gray_intensity"], 27.25)
            mode = dialog.exposure_controls["mode"]
            mode.setCurrentIndex(mode.findData(3))
            self.assertTrue(dialog.exposure_controls["phdr_quality"].isEnabled())
            self.assertFalse(dialog.exposure_controls["number"].isEnabled())
            self.assertFalse(dialog.exposure_controls["intensity_1"].isEnabled())
            dialog.save_settings()
            dialog.group = SimpleNamespace(records=[{}], expected_views=5)
            dialog.update_controls()
            self.assertTrue(dialog.exposure_enabled.isEnabled())
            self.assertTrue(mode.isEnabled())
            self.assertTrue(dialog.exposure_controls["gray_intensity"].isEnabled())
            self.assertTrue(dialog.exposure_apply.isEnabled())
            self.assertTrue(dialog.exposure_read.isEnabled())
            before = camera.api.exposure_writes
            dialog.exposure_controls["gray_intensity"].setValue(28)
            self.assertTrue(dialog.capture.isEnabled())
            self.assertIn("应用并采集", dialog.capture.text())
            dialog.apply_exposure()
            save.reset_mock()
            dialog.save_settings()
            self.assertEqual(camera.api.exposure_writes, before + 1)
            self.assertEqual(camera.read_exposure()["gray_intensity"], 28)
            self.assertTrue(dialog.capture.isEnabled())
            self.assertEqual(save.call_args.args[0]["exposure"]["gray_intensity"], 28)
            dialog.exposure_controls["gray_intensity"].setValue(29)
            dialog.save_settings()
            self.assertEqual(camera.read_exposure()["gray_intensity"], 29)
            self.assertTrue(dialog.capture.isEnabled())
            dialog.task = Mock()
            dialog.update_controls()
            self.assertFalse(dialog.exposure_enabled.isEnabled())
            self.assertTrue(all(not control.isEnabled() for control in dialog.exposure_controls.values()))
            self.assertFalse(dialog.exposure_apply.isEnabled())
            self.assertFalse(dialog.exposure_read.isEnabled())
            self.assertFalse(dialog.save_button.isEnabled())
        finally:
            dialog.task = None
            dialog.group = None
            dialog.camera = None
            camera.close()
            dialog.reject()
            app.processEvents()

    def test_each_view_keeps_frame_exposure_settings_without_overwriting_previous_views(self):
        camera = self.camera()
        camera.connect("123")
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory) / "input"
                root.mkdir()
                group = CaptureGroup(root, "exposure_changes", 5, {"exposure": dict(DEFAULT_EXPOSURE)})
                expected = []
                for index in range(5):
                    actual = camera.apply_exposure(dict(DEFAULT_EXPOSURE, intensity_1=20 + index))
                    result = camera.acquire(group.next_path)
                    self.assertEqual(result["exposure_settings"], actual)
                    group.record(result)
                    expected.append(20 + index)
                    self.assertEqual([v["exposure_settings"]["intensity_1"] for v in group.records], expected)
                manifest = json.loads((group.publish() / "camera_capture.json").read_text(encoding="utf-8"))
                self.assertEqual([v["exposure_settings"]["intensity_1"] for v in manifest["views"]], expected)
                self.assertEqual(manifest["capture_settings"]["exposure"], DEFAULT_EXPOSURE)
                self.assertTrue(all("image" not in view for view in manifest["views"]))
        finally:
            camera.close()

    def test_mid_group_capture_applies_exposure_and_retries_failed_settings_safely(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from pointcloud_app.native.camera_dialog import CameraDialog
        app = QApplication.instance() or QApplication([])
        camera = self.camera()
        camera.connect("123")
        save = Mock(return_value=True)
        with TemporaryDirectory() as directory:
            dialog = CameraDialog(output_root=directory, save_preferences=save)
            dialog.camera = camera
            dialog._exposure_loaded(camera.read_exposure())
            dialog.exposure_enabled.setChecked(True)

            def idle():
                for _ in range(150):
                    app.processEvents()
                    if dialog.task is None:
                        return
                    QTest.qWait(20)
                self.fail("Camera task did not finish")

            try:
                dialog.capture_view()
                idle()
                first_path = dialog.group.staging / "1.ply"
                original_file = first_path.read_bytes()
                original_record = dict(dialog.group.records[0])
                self.assertTrue(dialog.exposure_controls["intensity_1"].isEnabled())
                dialog.exposure_controls["intensity_1"].setValue(25)
                self.assertTrue(dialog.capture.isEnabled())
                camera.api.reject_exposure = True
                with patch.object(camera, "acquire", wraps=camera.acquire) as acquire:
                    QTest.mouseClick(dialog.capture, Qt.MouseButton.LeftButton)
                    idle()
                    acquire.assert_not_called()
                self.assertIn("拒绝", dialog.status.text())
                save.assert_not_called()
                self.assertTrue(dialog.capture.isEnabled())
                self.assertIn("应用并采集", dialog.capture.text())
                self.assertEqual(camera.read_exposure(), DEFAULT_EXPOSURE)
                self.assertEqual(dialog.group.records, [original_record])
                self.assertEqual(first_path.read_bytes(), original_file)
                self.assertFalse(dialog.group.next_path.exists())
                camera.api.reject_exposure = False
                QTest.mouseClick(dialog.capture, Qt.MouseButton.LeftButton)
                idle()
                self.assertEqual(len(dialog.group.records), 2)
                self.assertEqual(dialog.group.records[0], original_record)
                self.assertEqual(dialog.group.records[1]["exposure_settings"]["intensity_1"], 25)
                self.assertEqual(first_path.read_bytes(), original_file)
                self.assertFalse(dialog._exposure_pending())
                self.assertEqual(dialog.capture.text(), "采集视角 3")
                self.assertTrue(dialog.capture.isEnabled())
                save.assert_not_called()
            finally:
                dialog._closing = True
                dialog.reject()
                idle()
                self.assertIsNone(camera.handle)

    def test_capture_failure_after_apply_keeps_readback_and_ui_responsive_for_retry(self):
        from threading import Event
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from pointcloud_app.native.camera_dialog import CameraDialog
        app = QApplication.instance() or QApplication([])
        camera = self.camera()
        camera.connect("123")
        started, release, heartbeat = Event(), Event(), Event()
        with TemporaryDirectory() as directory:
            dialog = CameraDialog(output_root=directory, save_preferences=Mock(return_value=True))
            dialog.camera = camera
            dialog._exposure_loaded(camera.read_exposure())
            dialog.exposure_enabled.setChecked(True)

            def idle():
                for _ in range(150):
                    app.processEvents()
                    if dialog.task is None:
                        return
                    QTest.qWait(20)
                self.fail("Camera task did not finish")

            acquire = camera.acquire

            def slow_acquire(path, timeout):
                started.set()
                if not release.wait(3):
                    raise RuntimeError("Test capture was not released")
                return acquire(path, timeout)

            try:
                dialog.capture_view()
                idle()
                first_path = dialog.group.staging / "1.ply"
                original_file = first_path.read_bytes()
                original_record = dict(dialog.group.records[0])
                dialog.exposure_controls["intensity_1"].setValue(100)
                camera.api.fail_snap = True
                with patch.object(camera, "acquire", side_effect=slow_acquire) as capture:
                    QTest.mouseClick(dialog.capture, Qt.MouseButton.LeftButton)
                    self.assertFalse(dialog.capture.isEnabled())
                    self.assertFalse(dialog.connection.isEnabled())
                    self.assertFalse(dialog.exposure_controls["intensity_1"].isEnabled())
                    QTimer.singleShot(0, heartbeat.set)
                    for _ in range(50):
                        QTest.qWait(10)
                        if (started.is_set() and heartbeat.is_set() and
                                dialog._applied_exposure.get("intensity_1") == 100):
                            break
                    self.assertTrue(started.is_set())
                    self.assertTrue(heartbeat.is_set())
                    self.assertIsNotNone(dialog.task)
                    self.assertEqual(dialog._applied_exposure["intensity_1"], 100)
                    dialog.capture_view()
                    self.assertEqual(capture.call_count, 1)
                    release.set()
                    idle()
                self.assertIn("采集失败", dialog.status.text())
                self.assertEqual(camera.read_exposure()["intensity_1"], 100)
                self.assertFalse(dialog._exposure_pending())
                self.assertTrue(dialog.capture.isEnabled())
                self.assertEqual(dialog.capture.text(), "采集视角 2")
                self.assertEqual(dialog.group.records, [original_record])
                self.assertEqual(first_path.read_bytes(), original_file)
                self.assertFalse(dialog.group.next_path.exists())
                writes = camera.api.exposure_writes
                camera.api.fail_snap = False
                QTest.mouseClick(dialog.capture, Qt.MouseButton.LeftButton)
                idle()
                self.assertEqual(camera.api.exposure_writes, writes)
                self.assertEqual(len(dialog.group.records), 2)
                self.assertEqual(dialog.group.records[1]["exposure_settings"]["intensity_1"], 100)
                self.assertEqual(dialog.group.records[0], original_record)
                self.assertEqual(first_path.read_bytes(), original_file)
                dialog._save_preferences.assert_not_called()
            finally:
                release.set()
                idle()
                dialog._closing = True
                dialog.reject()
                idle()
                self.assertIsNone(camera.handle)

    def test_enter_commits_exposure_without_triggering_camera_commands(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from pointcloud_app.native.camera_dialog import CameraDialog
        app = QApplication.instance() or QApplication([])
        dialog = CameraDialog(save_preferences=Mock(return_value=True))
        dialog.camera = self.camera()
        dialog.camera.connect("123")
        dialog._exposure_loaded(dialog.camera.read_exposure())
        dialog.exposure_enabled.setChecked(True)
        dialog.group = SimpleNamespace(records=[{}], expected_views=5)
        dialog.update_controls()
        dialog.tabs.setCurrentIndex(1)
        clicks = Mock()
        for button in (dialog.connection, dialog.capture, dialog.publish, dialog.exposure_apply,
                       dialog.save_button, dialog.close_button):
            button.clicked.connect(clicks)
        try:
            dialog.show()
            QTest.qWait(50)
            number = dialog.exposure_controls["number"]
            number.setFocus()
            number.lineEdit().selectAll()
            QTest.keyClicks(number.lineEdit(), "3")
            QTest.keyClick(number.lineEdit(), Qt.Key.Key_Return)
            app.processEvents()
            self.assertEqual(number.value(), 3)
            self.assertTrue(dialog.exposure_controls["intensity_2"].isEnabled())
            self.assertTrue(dialog.exposure_controls["intensity_3"].isEnabled())
            self.assertIsNone(dialog.task)
            self.assertIsNotNone(dialog.camera.device)
            self.assertEqual(dialog.camera.api.exposure_writes, 0)
            self.assertEqual(len(dialog.group.records), 1)
            self.assertTrue(dialog.isVisible())
            clicks.assert_not_called()
        finally:
            camera = dialog.camera
            dialog.group = None
            dialog.camera = None
            camera.close()
            dialog.reject()
            app.processEvents()

    def test_ui_save_failure_is_not_reported_as_saved(self):
        from PySide6.QtWidgets import QApplication
        from pointcloud_app.native.camera_dialog import CameraDialog
        app = QApplication.instance() or QApplication([])
        dialog = CameraDialog(save_preferences=Mock(return_value=False))
        dialog.save_settings()
        self.assertIn("保存失败", dialog.status.text())
        self.assertNotIn("已保存", dialog.status.text())
        dialog.reject()
        app.processEvents()

    def test_settings_export_rejects_overwrite_and_cleans_failed_temporary_files(self):
        camera = self.camera()
        camera.connect("123")
        try:
            with TemporaryDirectory() as directory:
                target = Path(directory) / "saved.mpset"
                target.write_bytes(b"original")
                writer = Mock()
                camera.api.MPSizectorS_WriteSettingsToFile = writer
                with self.assertRaises(CameraError):
                    camera.export_settings(target)
                writer.assert_not_called()
                self.assertEqual(target.read_bytes(), b"original")
                new = Path(directory) / "new.mpset"
                for success, content in ((False, b"partial"), (True, b"")):
                    with self.subTest(success=success):
                        def export(handle, path):
                            Path(path.decode("mbcs")).write_bytes(content)
                            return success
                        writer.side_effect = export
                        with self.assertRaises(CameraError):
                            camera.export_settings(new)
                        self.assertEqual(list(Path(directory).iterdir()), [target])
                writer.side_effect = lambda handle, path: Path(path.decode("mbcs")).write_bytes(b"sdk settings")
                self.assertEqual(camera.export_settings(new), new)
                self.assertEqual(new.read_bytes(), b"sdk settings")
                self.assertEqual(set(Path(directory).iterdir()), {target, new})
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
