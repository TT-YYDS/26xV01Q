"""Explicit, one-view-at-a-time Sizector acquisition for the native workflow."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QStyle, QMessageBox, QSizePolicy, QDoubleSpinBox, QCheckBox, QTabWidget, QScrollArea, QWidget, QFrame,
)

from ..sizector_camera import (
    SizectorCamera, CaptureGroup, DEFAULT_SDK, DEFAULT_VIEWS, WORKING_MODES, CameraError,
    EXPOSURE_MODES, EXPOSURE_LIMITS, DEFAULT_EXPOSURE, validate_exposure,
)
from .widgets import Task, icon_button


class CameraDialog(QDialog):
    group_ready = Signal(object)
    preferences_changed = Signal(object)

    def __init__(self, parent=None, *, preferences=None, output_root=None, fixed_output=False,
                 save_preferences=None):
        super().__init__(parent)
        saved = preferences if isinstance(preferences, dict) else {}
        numeric = {"working_mode", "expected_views", "timeout_seconds", "preferences_version"}
        try:
            exposure = validate_exposure(saved.get("exposure", DEFAULT_EXPOSURE))
            custom_exposure = saved.get("exposure_enabled") is True
        except CameraError:
            exposure, custom_exposure = dict(DEFAULT_EXPOSURE), False
        saved = {key: value for key, value in saved.items()
                 if isinstance(value, int if key in numeric else str) and not isinstance(value, bool)}
        if saved.get("preferences_version", 0) < 2 and saved.get("expected_views") == 6:
            saved["expected_views"] = DEFAULT_VIEWS
        self.saved_serial = saved.get("serial", "")
        self.setWindowTitle("盛相 3D 相机采集")
        self.resize(720, 750)
        self.setMinimumSize(560, 630)
        self.camera = None
        self.group = None
        self.task = None
        self._callback = None
        self._closing = False
        self._save_preferences = save_preferences
        self._applied_exposure = None
        self.fixed_output = fixed_output
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        self.sdk = QLineEdit(saved.get("sdk_root") or str(DEFAULT_SDK))
        self.sdk.setReadOnly(True)
        self.sdk_picker = icon_button(self, QStyle.StandardPixmap.SP_DirOpenIcon,
                                      "选择 SDK 目录", self.choose_sdk)
        form.addRow("SDK 目录", self._field(self.sdk, self.sdk_picker))
        self.devices = QComboBox()
        self.devices.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.search = icon_button(self, QStyle.StandardPixmap.SP_BrowserReload, "搜索相机", self.enumerate)
        form.addRow("设备", self._field(self.devices, self.search))
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(170)
        self.tabs.setMaximumHeight(330)
        layout.addWidget(self.tabs)
        form = self._form_tab("采集设置")
        self.mode = QComboBox()
        for key, title in WORKING_MODES.items():
            self.mode.addItem(title, key)
        mode = saved.get("working_mode", 2)
        index = self.mode.findData(mode if mode in WORKING_MODES else 2)
        self.mode.setCurrentIndex(index if index >= 0 else 2)
        form.addRow("拍摄模式", self.mode)
        self.settings = QLineEdit(saved.get("settings_file", ""))
        self.settings.setPlaceholderText("保留相机现有曝光参数")
        self.settings_picker = icon_button(self, QStyle.StandardPixmap.SP_DialogOpenButton,
                                           "选择相机参数文件", self.choose_settings)
        self.settings_clear = icon_button(self, QStyle.StandardPixmap.SP_DialogResetButton,
                                          "清除参数文件", self.settings.clear)
        self.settings_export = icon_button(self, QStyle.StandardPixmap.SP_DialogSaveButton,
                                           "导出相机参数文件", self.export_settings)
        form.addRow("参数文件", self._field(self.settings, self.settings_picker, self.settings_clear,
                                          self.settings_export))
        self.timeout = QSpinBox()
        self.timeout.setRange(1, 120)
        self.timeout.setValue(min(120, max(1, saved.get("timeout_seconds", 15))))
        self.timeout.setSuffix(" s")
        form.addRow("超时", self.timeout)
        self.output = QLineEdit(str(output_root or saved.get("output_root") or DEFAULT_SDK.parent / "media" / "camera"))
        self.output.setReadOnly(True)
        self.output_picker = icon_button(self, QStyle.StandardPixmap.SP_DirOpenIcon,
                                         "选择采集输入目录", self.choose_output)
        form.addRow("输入目录", self._field(self.output, self.output_picker))
        self.name = QLineEdit(saved.get("group_name") or self.new_name())
        form.addRow("组名", self.name)
        self.views = QSpinBox()
        self.views.setRange(5, 8)
        self.views.setValue(min(8, max(5, saved.get("expected_views", DEFAULT_VIEWS))))
        form.addRow("视角数", self.views)
        exposure_form = self._form_tab("曝光设置")
        self.exposure_enabled = QCheckBox("自定义曝光")
        self.exposure_enabled.setChecked(custom_exposure)
        exposure_form.addRow(self.exposure_enabled)
        self.exposure_controls = {}
        labels = {"mode": "曝光模式", "number": "曝光次数", "intensity_1": "3D 曝光强度 1",
                  "intensity_2": "3D 曝光强度 2", "intensity_3": "3D 曝光强度 3",
                  "gray_intensity": "灰度图曝光强度", "gain": "用户增益",
                  "hdr_priority": "HDR 优先级", "phdr_quality": "PHDR 质量"}
        for key, label in labels.items():
            if key == "mode":
                control = QComboBox()
                for value, title in EXPOSURE_MODES.items():
                    control.addItem(title, value)
                control.currentIndexChanged.connect(self.update_controls)
            else:
                low, high = EXPOSURE_LIMITS[key]
                control = QDoubleSpinBox() if isinstance(low, float) else QSpinBox()
                control.setRange(low, high)
                control.setKeyboardTracking(False)
                if isinstance(low, float):
                    control.setDecimals(3)
                    control.setSingleStep(0.1)
                    control.setToolTip("SDK 曝光强度，范围 0.1～100；不是曝光时间毫秒")
                control.valueChanged.connect(self.update_controls)
            self.exposure_controls[key] = control
            exposure_form.addRow(label, control)
        self.exposure_read = icon_button(self, QStyle.StandardPixmap.SP_BrowserReload,
                                         "读取相机当前曝光参数", self.read_exposure)
        self.exposure_apply = QPushButton("应用曝光")
        self.exposure_apply.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.exposure_apply.clicked.connect(self.apply_exposure)
        exposure_form.addRow(self._field(self.exposure_apply, self.exposure_read))
        self._set_exposure(exposure)
        actions = QHBoxLayout()
        layout.addLayout(actions)
        self.connection = QPushButton("连接相机")
        self.connection.clicked.connect(self.toggle_connection)
        actions.addWidget(self.connection)
        self.capture = QPushButton("采集视角 1")
        self.capture.setProperty("primary", True)
        self.capture.clicked.connect(self.capture_view)
        actions.addWidget(self.capture)
        self.publish = QPushButton("完成并导入")
        self.publish.clicked.connect(self.publish_group)
        actions.addWidget(self.publish)
        self.reset = icon_button(self, QStyle.StandardPixmap.SP_DialogResetButton,
                                 "结束当前组，保留暂存点云", self.reset_group)
        actions.addWidget(self.reset)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(0, 100)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.preview, 1)
        self._preview_pixmap = None
        self.records = QTableWidget(0, 4)
        self.records.setHorizontalHeaderLabels(["视角", "有效点数", "分辨率", "耗时 (s)"])
        self.records.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.records.verticalHeader().hide()
        self.records.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.records.setAlternatingRowColors(True)
        self.records.setMaximumHeight(190)
        layout.addWidget(self.records)
        self.status = QLabel("未连接")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.reject)
        footer = QHBoxLayout()
        self.save_button = QPushButton("保存设置")
        self.save_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_button.clicked.connect(self.save_settings)
        footer.addWidget(self.save_button)
        footer.addStretch()
        footer.addWidget(self.close_button)
        layout.addLayout(footer)
        # Enter commits a field, without triggering an unrelated camera command.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self.views.valueChanged.connect(self.update_controls)
        self.exposure_enabled.toggled.connect(self.update_controls)
        self.update_controls()

    def _form_tab(self, title):
        content = QWidget()
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, title)
        return form

    @staticmethod
    def new_name():
        return "camera_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _field(self, *widgets):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for index, widget in enumerate(widgets):
            row.addWidget(widget, 1 if index == 0 else 0)
        return row

    def preferences(self):
        return {"preferences_version": 2, "sdk_root": self.sdk.text(),
                "settings_file": self.settings.text().strip(), "group_name": self.name.text(),
                "working_mode": self.mode.currentData(), "expected_views": self.views.value(),
                "timeout_seconds": self.timeout.value(), "output_root": self.output.text(),
                "serial": self.devices.currentData() or self.saved_serial,
                "exposure_enabled": self.exposure_enabled.isChecked(), "exposure": self._exposure_values()}

    def _exposure_values(self):
        return {key: control.currentData() if key == "mode" else control.value()
                for key, control in self.exposure_controls.items()}

    def _set_exposure(self, values):
        values = validate_exposure(values)
        for key, control in self.exposure_controls.items():
            blocked = control.blockSignals(True)
            if key == "mode":
                control.setCurrentIndex(control.findData(values[key]))
            else:
                control.setValue(values[key])
            control.blockSignals(blocked)

    def _exposure_pending(self):
        return self.exposure_enabled.isChecked() and (self._applied_exposure is None or
                any(not math.isclose(value, self._applied_exposure[key], abs_tol=0.001, rel_tol=0)
                    for key, value in self._exposure_values().items()))

    @Slot(object)
    def _exposure_loaded(self, values):
        self._set_exposure(values)
        self._applied_exposure = dict(values)

    def read_exposure(self):
        def loaded(values):
            self._exposure_loaded(values)
            self.status.setText("相机曝光参数已读取")
        self._run(lambda progress: self.camera.read_exposure(), loaded)

    def apply_exposure(self):
        values = self._exposure_values()
        def applied(actual):
            self._exposure_loaded(actual)
            self.status.setText("曝光参数已应用并回读确认")
        self._run(lambda progress: self.camera.apply_exposure(values), applied)

    def save_settings(self):
        if self.task is not None:
            return
        def saved(actual=None):
            if actual is not None:
                self._exposure_loaded(actual)
            if self._save_preferences is None or self._save_preferences(self.preferences()) is False:
                raise CameraError("相机设置保存失败")
            self.status.setText("相机采集设置已保存")
        if self.camera is not None and self.camera.device is not None and self._exposure_pending():
            values = self._exposure_values()
            self._run(lambda progress: self.camera.apply_exposure(values), saved)
        else:
            try:
                saved()
            except Exception as error:
                self.status.setText(f"相机设置保存失败：{error}")

    def export_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出相机参数", "camera_settings.mpset",
                                             "相机参数 (*.mpset);;XML (*.xml)")
        if path:
            def exported(target):
                self.status.setText(f"相机参数已导出：{target.name}")
                self.status.setToolTip(str(target))
            self._run(lambda progress: self.camera.export_settings(path), exported)

    def choose_sdk(self):
        directory = QFileDialog.getExistingDirectory(self, "选择盛相 SDK 目录", self.sdk.text())
        if directory:
            self.sdk.setText(directory)
            self.devices.clear()

    def choose_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择相机参数文件", self.settings.text(), "相机参数 (*.mpset *.xml);;所有文件 (*)")
        if path:
            self.settings.setText(path)

    def choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择采集输入目录", self.output.text())
        if directory:
            self.output.setText(directory)

    def _service(self):
        if self.camera is None or self.camera.sdk_root != Path(self.sdk.text()).resolve():
            if self.camera is not None:
                self.camera.close()
            self.camera = SizectorCamera(self.sdk.text())
        return self.camera

    def _run(self, operation, callback, updated=None):
        if self.task:
            return
        self.task = Task(operation, self, streaming=updated is not None)
        self._callback = callback
        self.task.progress.connect(self.status.setText)
        if updated is not None:
            self.task.updated.connect(updated)
        self.task.finished.connect(self._finished)
        self.update_controls()
        self.task.start()

    def _finished(self):
        task, callback = self.task, self._callback
        self.task, self._callback = None, None
        if task.error:
            self.status.setText(str(task.error))
        else:
            try:
                callback(task.value)
            except Exception as error:
                self.status.setText(str(error))
        task.deleteLater()
        self.update_controls()
        if self._closing:
            self.reject()

    def update_controls(self):
        idle = self.task is None
        connected = self.camera is not None and self.camera.device is not None
        active = self.group is not None
        count = len(self.group.records) if active else 0
        expected = self.group.expected_views if active else self.views.value()
        for widget in (self.sdk_picker, self.search, self.devices, self.mode, self.settings,
                       self.settings_picker, self.settings_clear):
            widget.setEnabled(idle and not connected and not count)
        for widget in (self.name, self.views):
            widget.setEnabled(idle and not active)
        self.output_picker.setEnabled(idle and not active and not self.fixed_output)
        self.timeout.setEnabled(idle)
        self.exposure_enabled.setEnabled(idle)
        custom = idle and self.exposure_enabled.isChecked()
        exposure_mode = self.exposure_controls["mode"].currentData()
        exposure_number = self.exposure_controls["number"].value()
        for key, control in self.exposure_controls.items():
            enabled = custom
            if key.startswith("intensity_"):
                index = int(key[-1])
                enabled = enabled and ((exposure_mode == 0 and index <= exposure_number) or
                                       (exposure_mode == 1 and index == 1))
            elif key == "number":
                enabled = enabled and exposure_mode != 3
            elif key == "hdr_priority":
                enabled = enabled and exposure_mode in (2, 3)
            elif key == "phdr_quality":
                enabled = enabled and exposure_mode == 3
            control.setEnabled(enabled)
        self.exposure_read.setEnabled(idle and connected)
        self.exposure_apply.setEnabled(custom and connected)
        self.settings_export.setEnabled(idle and connected)
        self.save_button.setEnabled(idle and self._save_preferences is not None)
        self.connection.setText("断开相机" if connected else "连接相机")
        self.connection.setEnabled(idle and (connected or bool(self.devices.count())))
        pending = self._exposure_pending()
        prefix = "应用并采集视角" if pending else "采集视角"
        self.capture.setText(f"{prefix} {min(count + 1, expected)}")
        self.capture.setEnabled(idle and connected and count < expected)
        self.capture.setToolTip("先应用并回读曝光参数，再采集当前视角" if pending else "采集当前视角")
        self.publish.setEnabled(idle and active and count == expected)
        self.reset.setEnabled(idle and active)
        self.close_button.setText("等待操作结束后关闭" if self._closing else "关闭")

    def enumerate(self):
        camera = self._service()
        def operation(progress):
            progress("正在搜索相机")
            return camera.enumerate()
        def loaded(devices):
            previous = self.devices.currentData() or self.saved_serial
            self.devices.clear()
            for info in devices:
                self.devices.addItem(f"{info.DeviceName} · {info.DeviceSerialNumber}", info.DeviceSerialNumber)
                self.devices.setItemData(self.devices.count() - 1,
                                         f"{info.SensorWidth} × {info.SensorHeight} / 固件 {info.FWVersion:g}", Qt.ItemDataRole.ToolTipRole)
            index = self.devices.findData(previous)
            if index >= 0:
                self.devices.setCurrentIndex(index)
            self.status.setText(f"SDK {camera.version:.3f} · {len(devices)} 台设备" if devices else
                                f"SDK {camera.version:.3f} · 未发现相机，请检查 USB 连接及 SDK 的 04_Driver 驱动")
        self._run(operation, loaded)

    def toggle_connection(self):
        camera = self._service()
        if camera.device:
            self._run(lambda progress: camera.disconnect(), lambda _: self.status.setText("已断开"))
            return
        prefs = self.preferences()
        def operation(progress):
            progress("正在连接相机")
            try:
                info = camera.connect(prefs["serial"], prefs["working_mode"], prefs["settings_file"],
                                      prefs["timeout_seconds"] * 1000)
                exposure = (camera.apply_exposure(prefs["exposure"]) if prefs["exposure_enabled"]
                            else camera.read_exposure())
                validate_exposure(exposure)
                return info, exposure
            except Exception:
                camera.disconnect()
                raise
        def connected(result):
            info, exposure = result
            self._exposure_loaded(exposure)
            self.status.setText(f"已连接 {info.DeviceName} · {info.DeviceSerialNumber} · 坐标单位 mm")
            self.preferences_changed.emit(self.preferences())
        self._run(operation, connected)

    def capture_view(self):
        if self.task is not None:
            return
        if self.camera is None or self.camera.device is None:
            self.status.setText("请先连接相机")
            return
        pending = self._exposure_pending()
        values = self._exposure_values()
        camera = self.camera
        try:
            if self.group is None:
                root = Path(self.output.text()).resolve()
                root.mkdir(parents=True, exist_ok=True)
                self.group = CaptureGroup(root, self.name.text(), self.views.value(), self.preferences())
            path = self.group.next_path
        except (CameraError, OSError) as error:
            self.status.setText(str(error))
            return
        timeout = self.timeout.value() * 1000
        def operation(progress, updated, cancelled):
            if pending:
                progress("正在应用并确认曝光参数")
                actual = camera.apply_exposure(values)
                updated(actual)
            progress(f"正在采集 {path.name}")
            return camera.acquire(path, timeout)
        def captured(result):
            self.group.record(result)
            image = result["image"]
            self._preview_pixmap = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0],
                                                           image.strides[0], QImage.Format.Format_Grayscale8).copy())
            self._fit_preview()
            row = self.records.rowCount()
            self.records.insertRow(row)
            for index, value in enumerate((row + 1, f"{result['points']:,}",
                                           f"{result['width']} × {result['height']}", f"{result['elapsed_seconds']:.2f}")):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.records.setItem(row, index, item)
            self.status.setText(f"已采集 {len(self.group.records)}/{self.group.expected_views} · {path.name} · 坐标单位 mm")
        self._run(operation, captured, updated=self._exposure_loaded)

    def publish_group(self):
        def published(path):
            self.group_ready.emit(path)
            self.group = None
            self.name.setText(self.new_name())
            self.preferences_changed.emit(self.preferences())
            self.records.setRowCount(0)
            self.status.setText(f"整组已导入：{path.name}")
            self.status.setToolTip(str(path))
        self._run(lambda progress: self.group.publish(), published)

    def reset_group(self):
        if self.group is None:
            return
        answer = QMessageBox.question(self, "结束当前组", f"已采数据将保留在：\n{self.group.staging}\n开始新的采集组？")
        if answer == QMessageBox.StandardButton.Yes:
            self.status.setText("暂存点云已保留，未导入")
            self.status.setToolTip(str(self.group.staging))
            self.group = None
            self.records.setRowCount(0)
            self._preview_pixmap = None
            self.preview.clear()
            self.name.setText(self.new_name())
            self.update_controls()

    def _fit_preview(self):
        if getattr(self, "_preview_pixmap", None) is not None:
            self.preview.setPixmap(self._preview_pixmap.scaled(self.preview.size(),
                                   Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_preview()

    def reject(self):
        if self.task is not None:
            self._closing = True
            self.status.setText("等待当前 SDK 操作结束后关闭")
            self.update_controls()
            return
        if self.group is not None and not self._closing:
            answer = QMessageBox.question(self, "未发布的采集", f"本组尚未导入。已采数据将保留在：\n{self.group.staging}\n仍然关闭？")
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self.camera is not None:
            self._closing = True
            camera = self.camera
            self._run(lambda progress: camera.close(), lambda _: None)
            # The completion callback calls reject again after cleanup.
            self.camera = None
            return
        super().reject()

    def closeEvent(self, event):
        event.ignore()
        self.reject()
