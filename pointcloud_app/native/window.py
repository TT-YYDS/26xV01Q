"""A fixed workflow rail and one switchable desktop work area."""

from __future__ import annotations

import copy
import json
from time import perf_counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QStackedWidget, QButtonGroup, QFileDialog, QMessageBox,
    QStyle, QDoubleSpinBox, QCheckBox, QDialog, QPlainTextEdit,
    QProgressBar, QFrame, QMenu, QSizePolicy, QToolButton,
)

from ..cad_geometry import (
    load_cad_entities, normalize_cad_path_selection, select_cad_path_endpoint,
    select_cad_single_entity_path, validate_cad_path_selection,
)
from ..desktop_runtime import discover_stable_input_groups
from ..automatic_runtime import (
    estimate_batch_seconds, load_duration_history, measurement_table,
    prepare_batch, run_batch,
)
from ..gpu import resolve_compute_backend
from ..pipeline import discover_groups, read_point_cloud, scan_section_offsets
from ..workflow_profile import build_profile, load_profile, save_profile
from . import figures
from .backend import (
    Backend, Cancelled, ROOT, config_from_dict, default_config, acquisition_signature,
    fit_selection, processing_quality, result_from_summary,
)
from .canvas import Canvas
from .widgets import FitImage, PagedTable, Parameters, Task, icon_button
from .camera_dialog import CameraDialog

STYLE = """
QWidget { font-family: "Microsoft YaHei"; font-size: 13px; color: #29343a; }
QMainWindow, QDialog { background: #ffffff; }
QFrame#rail { background: #f6f8f9; border-right: 1px solid #dde3e6; }
QPushButton, QToolButton { background: #ffffff; border: 1px solid #d5dde1;
 border-radius: 4px; padding: 6px 12px; min-height: 20px; }
QToolButton { padding: 3px; }
QPushButton:hover, QToolButton:hover { background: #edf5f5; border-color: #8bb9b6; }
QPushButton:pressed, QToolButton:pressed { background: #d9edeb; }
QPushButton:disabled, QToolButton:disabled { color: #99a3a8; background: #f4f6f7; border-color: #e5e9eb; }
QPushButton[primary="true"] { background: #087f78; color: white; border-color: #087f78; }
QPushButton[primary="true"]:disabled { background: #c6d5d3; border-color: #c6d5d3; }
QPushButton#workflowMode { min-width: 138px; min-height: 36px; padding: 7px 18px;
 font-size: 15px; font-weight: 600; border-width: 2px; }
QPushButton#workflowMode[mode="manual"] { color: #075cff; border-color: #8eb2ff; background: #f2f6ff; }
QPushButton#workflowMode[mode="automatic"] { color: white; border-color: #087f78; background: #087f78; }
QPushButton#workflowMode:hover { border-color: #087f78; }
QPushButton[cadRole="registration"]:checked { background: #e8f0ff; color: #075cff; border-color: #075cff; }
QPushButton[cadRole="measurement"]:checked { background: #e7f5ec; color: #008844; border-color: #008844; }
QPushButton[nav="true"] { text-align: left; border: none; background: transparent;
 padding: 11px 16px; border-radius: 4px; font-size: 14px; }
QPushButton[nav="true"]:checked { background: #deeeeb; color: #086d65; font-weight: 600; }
QPushButton[nav="true"]:hover { background: #e9efef; }
QLabel#brand { font-size: 18px; font-weight: 600; }
QLabel#title { font-size: 17px; font-weight: 600; }
QLabel#status { color: #59676e; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: white; border: 1px solid #d5dde1;
 border-radius: 3px; padding: 5px 7px; min-height: 22px; selection-background-color: #087f78; }
QComboBox:disabled { color: #87949b; }
QComboBox QAbstractItemView { background: white; color: #29343a; selection-background-color: #deeeeb;
 selection-color: #086d65; border: 1px solid #d5dde1; }
QTableWidget { background: white; alternate-background-color: #f6f8f9;
 border: none; selection-background-color: #deeeeb; selection-color: #174943; }
QTableWidget::item { padding: 7px; border-bottom: 1px solid #edf0f2; }
QHeaderView::section { background: #f2f5f6; border: none; border-bottom: 1px solid #dce3e6;
 padding: 10px 6px; font-weight: 600; color: #4c5e67; }
QProgressBar { border: none; background: #e9efef; border-radius: 2px; max-height: 5px; }
QProgressBar::chunk { background: #087f78; }
QPlainTextEdit { background: white; color: #29343a; border: 1px solid #d5dde1; }
QToolTip { color: #29343a; background: #ffffff; border: 1px solid #c4cdd2; padding: 5px; }
QMenu { background: white; border: 1px solid #d5dde1; padding: 4px; }
QMenu::item { padding: 7px 20px; }
QMenu::item:selected { background: #deeeeb; }
"""

STEPS = ("数据", "点云", "截面", "审核", "CAD")
CAD_PAGE = STEPS.index("CAD")


@dataclass
class Group:
    files: list
    config: object
    result: object = None
    preview: object = None
    region_stamp: object = None
    dxf: object = None
    entities: list = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    pending: object = None
    undo: list = field(default_factory=list)
    report: object = None
    selected_file: object = None
    cad_mode: str = "registration"
    cad_step: float = .05
    automatic_result: bool = False


class DesktopWindow(QMainWindow):
    def __init__(self, data_root=None):
        super().__init__()
        self.setWindowTitle("FCMS | Floor Clip Measurement System")
        self.logo_path = Path(__file__).with_name("fcms_logo.svg")
        self.setWindowIcon(QIcon(str(self.logo_path)))
        self.resize(1360, 860)
        self.setMinimumSize(900, 620)
        self.setStyleSheet(STYLE)
        self.backend = Backend()
        self.workflow_profile, self.profile_error = load_profile()
        source = self.workflow_profile.get("source") or {}
        # The saved mode is a template detail only. Every application launch
        # starts in manual mode so automatic intake always requires an explicit click.
        self.workflow_mode = "manual"
        self.source_kind = source.get("kind", "folder")
        self.source_path = Path(source["path"]) if source.get("path") else None
        self.auto_enabled = False
        self.auto_seen = {}
        self.auto_pending = set()
        self.auto_records = {}
        self.auto_resources = None
        self.auto_batch_total = 0
        self.auto_batch_completed = 0
        self.auto_batch_started_at = None
        self.auto_batch_elapsed = 0.0
        self.auto_batch_expected_seconds = None
        self.auto_batch_remaining_seconds = None
        self.auto_batch_last_elapsed = None
        self.auto_batch_last_expected = None
        self.auto_duration_history = []
        self.auto_started_at = None
        self.auto_elapsed = 0.0
        self.auto_timer = QTimer(self)
        self.auto_timer.setInterval(2000)
        self.auto_timer.timeout.connect(self.auto_poll)
        self.auto_runtime_timer = QTimer(self)
        self.auto_runtime_timer.setInterval(1000)
        self.auto_runtime_timer.timeout.connect(self._refresh_auto_runtime)
        self.groups = {}
        self.group_name = ""
        self.page = 0
        self.single_file_view = False
        self.view_epoch = 0
        self.job = None
        self.job_callback = None
        self.pending_draw = False
        self.closing = False
        self.logs = []
        self.view_values = {}
        self.level = 1
        self.point_size = 1.5
        self.apply_denoise = True
        self._task_started_at = None
        self._task_elapsed = None
        self._task_message = "就绪"
        self.task_timer = QTimer(self)
        self.task_timer.setInterval(1000)
        self.task_timer.timeout.connect(self._refresh_task_status)
        self._build()
        root = (
            Path(data_root)
            if data_root
            else self.source_path
            if self.source_path and (
                self.source_path.is_dir() or self.workflow_mode == "automatic"
            )
            else ROOT / "media"
        )
        if root.is_dir():
            QTimer.singleShot(0, lambda: self.import_root(root, self.source_kind))

    @property
    def state(self):
        return self.groups.get(self.group_name)

    @property
    def cad_mode(self):
        return self.state.cad_mode if self.state else "registration"

    @property
    def cad_step(self):
        return self.state.cad_step if self.state else .05

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(164)
        left = QVBoxLayout(rail)
        left.setContentsMargins(14, 22, 14, 16)
        left.setSpacing(8)
        brand = QWidget()
        brand_row = QHBoxLayout(brand)
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(9)
        logo = QLabel()
        logo.setPixmap(QPixmap(str(self.logo_path)).scaled(
            32, 32, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        logo.setFixedSize(32, 32)
        brand_row.addWidget(logo)
        brand_text = QVBoxLayout()
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(0)
        name = QLabel("FCMS")
        name.setObjectName("brand")
        subtitle = QLabel("Floor Clip\nMeasurement\nSystem")
        subtitle.setStyleSheet("color: #68777e; font-size: 9px;")
        brand_text.addWidget(name)
        brand_text.addWidget(subtitle)
        brand_row.addLayout(brand_text)
        left.addWidget(brand)
        left.addSpacing(22)
        group = QButtonGroup(self)
        self.nav = []
        icons = (QStyle.StandardPixmap.SP_DirIcon, QStyle.StandardPixmap.SP_ComputerIcon,
                 QStyle.StandardPixmap.SP_FileDialogContentsView, QStyle.StandardPixmap.SP_MediaPlay,
                 QStyle.StandardPixmap.SP_FileDialogDetailedView, QStyle.StandardPixmap.SP_FileIcon)
        for i, (name, icon) in enumerate(zip(STEPS, icons)):
            button = QPushButton(name)
            button.setIcon(self.style().standardIcon(icon))
            button.setCheckable(True)
            button.setProperty("nav", True)
            group.addButton(button, i)
            left.addWidget(button)
            self.nav.append(button)
        group.idClicked.connect(self.select_page)
        self.nav[0].setChecked(True)
        left.addStretch()
        self.camera_button = QPushButton("相机采集")
        self.camera_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.camera_button.clicked.connect(self.open_camera)
        left.addWidget(self.camera_button)
        self.params_button = QPushButton("参数")
        self.params_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.params_button.clicked.connect(self.parameters)
        left.addWidget(self.params_button)
        self.log_button = QPushButton("日志")
        self.log_button.clicked.connect(self.show_logs)
        left.addWidget(self.log_button)
        self.export_button = QPushButton("导出")
        self.export_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.export_button.clicked.connect(self.export_menu)
        left.addWidget(self.export_button)
        outer.addWidget(rail)
        main = QWidget()
        right = QVBoxLayout(main)
        right.setContentsMargins(20, 16, 20, 10)
        right.setSpacing(12)
        header = QHBoxLayout()
        self.title = QLabel(STEPS[0])
        self.title.setObjectName("title")
        header.addWidget(self.title)
        header.addStretch()
        self.workflow_mode_button = QPushButton()
        self.workflow_mode_button.setObjectName("workflowMode")
        self.workflow_mode_button.setToolTip(
            "点击切换手动/自动；运行中切回手动会停止派发，并等待当前计算阶段退出")
        self.workflow_mode_button.clicked.connect(self.toggle_workflow_mode)
        header.addWidget(self.workflow_mode_button)
        self._refresh_workflow_mode_button()
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(120)
        self.group_combo.setMaximumWidth(250)
        self.group_combo.currentTextChanged.connect(self.change_group)
        header.addWidget(self.group_combo)
        self.history_button = icon_button(self, QStyle.StandardPixmap.SP_DirOpenIcon, "打开历史结果", self.open_result)
        header.addWidget(self.history_button)
        right.addLayout(header)
        self.auto_status_bar = QWidget()
        auto_status = QHBoxLayout(self.auto_status_bar)
        auto_status.setContentsMargins(0, 0, 0, 0)
        self.auto_status = QLabel()
        self.auto_status.setObjectName("status")
        self.auto_status.setWordWrap(True)
        auto_status.addWidget(self.auto_status, 1)
        self.auto_stop_button = icon_button(
            self, QStyle.StandardPixmap.SP_MediaStop, "停止自动流程", self.stop_automatic)
        auto_status.addWidget(self.auto_stop_button)
        right.addWidget(self.auto_status_bar)
        self._refresh_auto_runtime()
        self.toolbar = QWidget()
        self.tools = QHBoxLayout(self.toolbar)
        self.tools.setContentsMargins(0, 0, 0, 0)
        self.tools.setSpacing(8)
        right.addWidget(self.toolbar)
        self.stack = QStackedWidget()
        self.empty = QLabel("未导入点云")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setStyleSheet("color: #869299; font-size: 15px;")
        self.canvas = Canvas()
        self.canvas.interaction.connect(self.interaction)
        self.canvas.render_error.connect(self.error)
        self.table = PagedTable()
        self.table.table.cellClicked.connect(self.data_row_clicked)
        self.image = FitImage()
        for widget in (self.empty, self.canvas, self.table, self.image):
            self.stack.addWidget(widget)
        right.addWidget(self.stack, 1)
        bottom = QHBoxLayout()
        self.status = QLabel("就绪")
        self.status.setObjectName("status")
        self.status.setMinimumWidth(0)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        bottom.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(120)
        self.progress.setTextVisible(False)
        self.progress.hide()
        bottom.addWidget(self.progress)
        self.cancel_button = icon_button(self, QStyle.StandardPixmap.SP_MediaStop, "取消当前任务", self.cancel)
        self.cancel_button.hide()
        bottom.addWidget(self.cancel_button)
        right.addLayout(bottom)
        outer.addWidget(main, 1)
        self._toolbar()

    def log(self, text):
        self.logs.append(f"{datetime.now():%H:%M:%S}  {text}")
        self.logs = self.logs[-2000:]
        self._task_message = str(text).replace("\n", " ")
        if self._task_started_at is not None:
            self._refresh_task_status()
        else:
            self.status.setText(self._task_message[:180])
            self.status.setToolTip(self._task_message)

    @staticmethod
    def _format_elapsed(seconds):
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _refresh_task_status(self):
        if self._task_started_at is None:
            return
        elapsed = perf_counter() - self._task_started_at
        self._task_elapsed = elapsed
        message = self._task_message or "处理中"
        suffix = f" · 当前任务 {self._format_elapsed(elapsed)}"
        if self.auto_enabled:
            suffix += f" · 自动累计 {self._format_elapsed(self._current_auto_elapsed())}"
        text = f"{message[:180 - len(suffix)]}{suffix}"
        self.status.setText(text)
        self.status.setToolTip(text)

    def _current_auto_elapsed(self):
        if self.auto_started_at is None:
            return self.auto_elapsed
        return self.auto_elapsed + (perf_counter() - self.auto_started_at)

    def _refresh_auto_runtime(self):
        if not hasattr(self, "auto_status"):
            return
        self.auto_status_bar.setVisible(
            self.workflow_mode == "automatic" or self.auto_started_at is not None)
        self.auto_stop_button.setEnabled(self.auto_enabled)
        elapsed = self._current_auto_elapsed()
        status = "运行中" if self.auto_enabled else (
            "正在停止" if self.auto_started_at is not None else "已停止")
        text = f"自动{status} · 总用时 {self._format_elapsed(elapsed)}"
        if self.auto_batch_total:
            text += f" · 本批完成 {self.auto_batch_completed}/{self.auto_batch_total} 组"
        text += f" · 累计结果 {len(self.auto_records)} 组次"
        if self.auto_resources:
            text += (f" · {self.auto_resources['workers']} 进程"
                     f" × {self.auto_resources['threads_per_worker']} 线程")
        if self.auto_batch_total:
            estimate = self._current_batch_estimate()
            if estimate is None:
                text += " · 预计完成时间：首组完成后计算"
            else:
                self.auto_batch_expected_seconds = estimate["total_seconds"]
                self.auto_batch_remaining_seconds = estimate["remaining_seconds"]
                eta = datetime.now().astimezone() + timedelta(
                    seconds=estimate["remaining_seconds"])
                text += (
                    f" · 预计剩余 {self._format_elapsed(estimate['remaining_seconds'])}"
                    f" · 预计完成 {eta:%H:%M:%S}"
                )
        elif self.auto_batch_last_elapsed is not None:
            text += f" · 上批实际 {self._format_elapsed(self.auto_batch_last_elapsed)}"
            if self.auto_batch_last_expected is not None:
                text += f" · 预计 {self._format_elapsed(self.auto_batch_last_expected)}"
        self.auto_status.setText(text)
        self.auto_status.setToolTip(text)

    def _current_batch_elapsed(self):
        if self.auto_batch_started_at is None:
            return self.auto_batch_elapsed
        return self.auto_batch_elapsed + (perf_counter() - self.auto_batch_started_at)

    def _current_batch_estimate(self):
        if not self.auto_batch_total:
            return None
        workers = int((self.auto_resources or {}).get("workers", 1))
        samples = [
            *self.auto_duration_history,
            *[
                record.get("elapsed_seconds")
                for record in self.auto_records.values()
                if record.get("status") == "completed"
            ],
        ]
        return estimate_batch_seconds(
            self.auto_batch_total,
            self.auto_batch_completed,
            workers,
            samples,
            self._current_batch_elapsed(),
        )

    def _refresh_workflow_mode_button(self):
        button = getattr(self, "workflow_mode_button", None)
        if button is None:
            return
        automatic = self.workflow_mode == "automatic"
        button.setText("自动流程" if automatic else "手动流程")
        button.setProperty("mode", "automatic" if automatic else "manual")
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _finish_task_status(self, message=None):
        if message is not None:
            self._task_message = str(message).replace("\n", " ")
        elapsed = self._task_elapsed
        if elapsed is None:
            elapsed = (
                perf_counter() - self._task_started_at
                if self._task_started_at is not None
                else 0.0
            )
        self._task_elapsed = elapsed
        suffix = f" · 总耗时 {self._format_elapsed(elapsed)}"
        text = f"{(self._task_message or '任务完成')[:180 - len(suffix)]}{suffix}"
        self.status.setText(text)
        self.status.setToolTip(text)

    def _stop_task_timer(self):
        self.task_timer.stop()
        self._finish_task_status()
        self._task_started_at = None

    def error(self, text):
        self.log(text)
        QMessageBox.warning(self, "点云测量", str(text))

    def _start(self, operation, callback, *, streaming=False):
        if self.job is not None:
            return False
        self.job = Task(operation, self, streaming=streaming)
        self.job_callback = callback
        self.job.progress.connect(self.log)
        self.job.updated.connect(self._task_updated)
        self.job.finished.connect(self._finished)
        self._task_started_at = perf_counter()
        self._task_elapsed = 0.0
        self._task_message = "正在准备"
        self._refresh_task_status()
        self.task_timer.start()
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.show()
        self._busy(True)
        self.job.start()
        return True

    def _task_updated(self, value):
        if not isinstance(value, dict):
            return
        if value.get("type") == "automatic_record":
            self._automatic_record_update(value["record"])

    def _busy(self, busy):
        automatic = self.workflow_mode == "automatic"
        interactive = not busy and not automatic
        self.toolbar.setEnabled(interactive)
        self.stack.setEnabled(interactive)
        self.table.setEnabled(interactive)
        self.group_combo.setEnabled(interactive)
        for button in self.nav:
            button.setEnabled(interactive)
        self.log_button.setEnabled(interactive)
        if getattr(self, "camera_button", None) is not None:
            self.camera_button.setEnabled(not busy or automatic)
        if getattr(self, "workflow_mode_button", None) is not None:
            self.workflow_mode_button.setEnabled(not busy or automatic)
        self.params_button.setEnabled(interactive and bool(self.state))
        self.history_button.setEnabled(interactive)
        self.export_button.setEnabled(interactive and bool(self.state and self.state.result))

    def _finished(self):
        task, callback = self.job, self.job_callback
        self.job, self.job_callback = None, None
        self._task_elapsed = (
            perf_counter() - self._task_started_at
            if self._task_started_at is not None
            else 0.0
        )
        self.task_timer.stop()
        self.progress.hide()
        self.cancel_button.hide()
        self._busy(False)
        if self.closing:
            self._finish_task_status("正在关闭")
            self._task_started_at = None
            task.deleteLater()
            self.pending_draw = False
            self.close()
            return
        if task.error is None:
            try:
                callback(task.value)
            except Exception as error:
                import traceback
                self.logs.append(traceback.format_exc())
                self.error(str(error))
        elif isinstance(task.error, Cancelled):
            self.log("已取消")
        else:
            self.logs.append(task.detail)
            self.error(str(task.error))
        if self.job is None:
            self._finish_task_status()
            self._task_started_at = None
        if task.streaming and self.workflow_mode == "automatic":
            self.auto_pending.clear()
            if task.error is not None:
                self.stop_automatic()
                self._freeze_auto_batch()
        task.deleteLater()
        if self.closing:
            self.close()
        elif self.pending_draw and self.job is None:
            self.pending_draw = False
            self._draw()

    def cancel(self):
        if self.auto_enabled:
            self.stop_automatic()
        if hasattr(self, "batch"):
            self.batch.clear()
        if self.job:
            self.job.cancelled.set()
            self.log("正在取消，等待当前计算阶段结束")

    def toggle_workflow_mode(self):
        mode = "automatic" if self.workflow_mode == "manual" else "manual"
        self.workflow_mode_changed(mode)

    def workflow_mode_changed(self, mode=None):
        if mode is None:
            mode = "automatic" if self.workflow_mode == "manual" else "manual"
        if mode not in {"manual", "automatic"} or mode == self.workflow_mode:
            return
        if mode == "automatic" and self.job is not None:
            self.log("当前任务仍在运行，暂不能开启新的自动测量会话")
            return
        if mode == "manual" and self._showing_automatic_measurements():
            self.view_values[CAD_PAGE] = "automatic_measurements"
        self.workflow_mode = mode
        if mode == "manual":
            self.stop_automatic(log_message=False)
            self.log("已切换到手动流程")
        else:
            self._reset_automatic_session()
            self.log("已切换到自动流程，将使用已保存的自动模板")
        try:
            self._save_workflow_profile()
        except Exception as error:
            self.log(f"工作模式已切换，但自动模板保存失败：{error}")
        self._refresh_workflow_mode_button()
        self._refresh_auto_runtime()
        if mode == "automatic":
            # Leave preview pages before drawing: their background tasks would
            # otherwise occupy the worker slot needed to start automatic intake.
            self.view_values[CAD_PAGE] = "automatic_measurements"
            self.select_page(CAD_PAGE)
            self.start_automatic()
        else:
            self._toolbar()
            self._draw()

    def _reset_automatic_session(self):
        """Start a fresh in-memory automatic run without deleting past outputs."""
        if self.job is not None:
            return False
        self.auto_enabled = False
        self.auto_timer.stop()
        self.auto_runtime_timer.stop()
        self.auto_seen.clear()
        self.auto_pending.clear()
        self.auto_records.clear()
        self.auto_pose_reference = {}
        self.auto_resources = None
        self.auto_batch_total = 0
        self.auto_batch_completed = 0
        self.auto_batch_started_at = None
        self.auto_batch_elapsed = 0.0
        self.auto_batch_expected_seconds = None
        self.auto_batch_remaining_seconds = None
        self.auto_batch_last_elapsed = None
        self.auto_batch_last_expected = None
        self.auto_started_at = None
        self.auto_elapsed = 0.0

        # Automatic result states are disposable previews. Keep manually
        # imported/configured groups and all on-disk historical outputs.
        removed_names = [
            name for name, state in self.groups.items()
            if getattr(state, "automatic_result", False)
        ]
        for name in removed_names:
            self.groups.pop(name, None)
        if removed_names and hasattr(self, "group_combo"):
            self.group_combo.blockSignals(True)
            for name in removed_names:
                index = self.group_combo.findText(name)
                if index >= 0:
                    self.group_combo.removeItem(index)
            if self.group_name not in self.groups:
                self.group_name = next(iter(self.groups), "")
                if self.group_name:
                    self.group_combo.setCurrentText(self.group_name)
            self.group_combo.blockSignals(False)
        self._refresh_auto_runtime()
        return True

    def _profile_config(self):
        payload = self.workflow_profile.get("config")
        if not isinstance(payload, dict):
            return None
        try:
            return config_from_dict(payload)
        except (ValueError, TypeError, KeyError) as error:
            self.profile_error = f"自动流程模板中的点云参数无效：{error}"
            return None

    def _apply_profile_to_state(self, state):
        config = self._profile_config()
        if config is not None:
            state.config = config
            # A saved region is deliberately reusable across new captures.
            state.region_stamp = None
        cad = self.workflow_profile.get("cad") or {}
        dxf_text = cad.get("dxf")
        if dxf_text:
            dxf = Path(dxf_text)
            if dxf.is_file():
                try:
                    entities = load_cad_entities(dxf)
                    selection = normalize_cad_path_selection(cad.get("selection") or {})
                    validate_cad_path_selection(
                        {entity.index for entity in entities}, selection, entities
                    )
                    state.dxf = dxf
                    state.entities = entities
                    state.selection = selection
                    state.cad_step = float(cad.get("step", 0.05))
                except (OSError, ValueError, TypeError, RuntimeError):
                    state.dxf = None
                    state.entities = []
                    state.selection = {}
        return state

    def _new_group_state(self, name, files):
        state = Group(files, default_config(name))
        return self._apply_profile_to_state(state)

    def _save_workflow_profile(self, state=None):
        if self.workflow_mode == "automatic" or (
            state is None and self.state and self.state.automatic_result
        ):
            state = None
        else:
            state = state or self.state
        config = state.config if state else self._profile_config()
        dxf = state.dxf if state else (self.workflow_profile.get("cad") or {}).get("dxf")
        selection = (
            normalize_cad_path_selection(state.selection)
            if state
            else (self.workflow_profile.get("cad") or {}).get("selection", {})
        )
        step = state.cad_step if state else (self.workflow_profile.get("cad") or {}).get("step", 0.05)
        camera_preferences = copy.deepcopy(self.workflow_profile.get("camera", {}))
        self.workflow_profile = build_profile(
            workflow_mode=self.workflow_mode,
            source_kind=self.source_kind,
            source_path=self.source_path,
            config=config,
            dxf=dxf,
            selection=selection,
            cad_step=step,
        )
        if camera_preferences:
            self.workflow_profile["camera"] = camera_preferences
        path = save_profile(self.workflow_profile)
        self.profile_error = None
        return path

    def save_workflow_profile(self):
        try:
            path = self._save_workflow_profile()
            self.log(f"自动流程模板已保存：{path.name}")
            self._toolbar()
        except Exception as error:
            self.error(f"保存自动流程模板失败：{error}")

    def _source_blocker(self):
        if not self.source_path:
            return "请先选择自动流程输入目录"
        if not self.source_path.is_dir():
            return f"输入目录不存在：{self.source_path}"
        config = self._profile_config()
        if config is None or config.section.target_region is None:
            return "请在手动流程中框选主轴并保存自动流程模板"
        cad = self.workflow_profile.get("cad") or {}
        dxf_text = cad.get("dxf")
        if not dxf_text or not Path(dxf_text).is_file():
            return "请在手动流程中导入 DXF 并保存自动流程模板"
        try:
            entities = load_cad_entities(Path(dxf_text))
            selection = normalize_cad_path_selection(cad.get("selection") or {})
            validate_cad_path_selection(
                {entity.index for entity in entities}, selection, entities
            )
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            return f"自动模板中的 CAD 路径无效：{error}"
        return ""

    def choose_source(self):
        title = "选择 3D 相机传输目录" if self.source_kind == "camera" else "选择点云输入目录"
        directory = QFileDialog.getExistingDirectory(self, title, str(self.source_path or ROOT / "media"))
        if directory:
            self.source_path = Path(directory).resolve()
            try:
                self._save_workflow_profile()
            except Exception as error:
                self.error(f"保存输入源失败：{error}")
                return
            self.import_root(self.source_path, self.source_kind)

    def source_kind_changed(self):
        if getattr(self, "source_combo", None):
            self.source_kind = self.source_combo.currentData() or "folder"
            try:
                self._save_workflow_profile()
            except Exception as error:
                self.error(f"保存输入源类型失败：{error}")

    def open_camera(self):
        if self.job and self.workflow_mode != "automatic":
            return
        preferences = self.workflow_profile.get("camera")
        saved_output = preferences.get("output_root") if isinstance(preferences, dict) else None
        output_root = (self.source_path if self.auto_enabled or
                       (self.source_kind == "camera" and not saved_output) else None)
        dialog = CameraDialog(self, preferences=preferences, output_root=output_root,
                              fixed_output=self.auto_enabled, save_preferences=self._camera_preferences_changed)
        dialog.preferences_changed.connect(self._camera_preferences_changed)
        dialog.group_ready.connect(self._camera_group_ready)
        dialog.exec()
        dialog.deleteLater()

    def _camera_preferences_changed(self, preferences):
        payload = copy.deepcopy(self.workflow_profile)
        payload["camera"] = copy.deepcopy(preferences)
        try:
            path = save_profile(payload)
        except Exception as error:
            self.log(f"相机设置暂未保存：{error}")
            return False
        self.workflow_profile = payload
        return path

    def _camera_group_ready(self, path):
        path = Path(path).resolve()
        self.source_kind, self.source_path = "camera", path.parent
        self.log(f"相机完整组已发布：{path}")
        try:
            self._save_workflow_profile()
        except Exception as error:
            self.log(f"相机输入目录暂未保存：{error}")
        if self.workflow_mode == "automatic":
            if not self.auto_enabled:
                self.start_automatic()
            else:
                self.auto_poll()
            return
        self.groups[path.name] = self._new_group_state(path.name, sorted(path.glob("*.ply")))
        self.group_combo.blockSignals(True)
        if self.group_combo.findText(path.name) < 0:
            self.group_combo.addItem(path.name)
        self.group_combo.setCurrentText(path.name)
        self.group_combo.blockSignals(False)
        self.change_group(path.name)

    def _data_toolbar(self):
        panel = QWidget()
        rows = QVBoxLayout(panel)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)
        source_row = QHBoxLayout()
        action_row = QHBoxLayout()
        source_row.setSpacing(8)
        action_row.setSpacing(8)
        rows.addLayout(source_row)
        rows.addLayout(action_row)
        self.tools.addWidget(panel, 1)
        self.source_combo = QComboBox()
        self.source_combo.addItem("指定文件夹", "folder")
        self.source_combo.addItem("3D 相机传输目录", "camera")
        index = self.source_combo.findData(self.source_kind)
        self.source_combo.setCurrentIndex(index if index >= 0 else 0)
        self.source_combo.setToolTip("手动选择点云目录，或选择相机传输后落盘的目录")
        self.source_combo.currentIndexChanged.connect(self.source_kind_changed)
        self.source_combo.setEnabled(self.workflow_mode == "manual")
        source_row.addWidget(self.source_combo)
        self._button(
            "选择输入目录",
            self.choose_source,
            primary=self.workflow_mode == "manual",
            enabled=self.workflow_mode == "manual",
            layout=source_row,
        )
        source_text = str(self.source_path) if self.source_path else "未选择输入目录"
        self.source_label = QLabel(source_text)
        self.source_label.setObjectName("status")
        self.source_label.setToolTip(source_text)
        self.source_label.setMinimumWidth(0)
        self.source_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        source_row.addWidget(self.source_label, 1)
        self._button("保存自动模板", self.save_workflow_profile,
                     enabled=self.workflow_mode == "manual", layout=action_row)
        try:
            template_ready = not self._source_blocker()
        except Exception:
            template_ready = False
        template_label = QLabel("模板：已就绪" if template_ready else "模板：待手动流程设置")
        template_label.setObjectName("status")
        template_label.setToolTip("自动流程会复用手动流程保存的主轴、点云参数和 CAD 路径")
        action_row.addWidget(template_label)
        if self.workflow_mode == "automatic":
            self._button(
                "停止自动流程" if self.auto_enabled else "启动自动流程",
                (lambda: self.stop_automatic()) if self.auto_enabled else self.start_automatic,
                primary=self.auto_enabled,
                enabled=not bool(self.job),
                layout=action_row,
            )
            self._button("立即扫描", self.auto_poll, enabled=not bool(self.job), layout=action_row)
        else:
            self._button("导入当前目录", lambda: self.import_root(
                self.source_path, self.source_kind) if self.source_path else self.choose_source(),
                enabled=bool(self.source_path and self.source_path.is_dir()),
                layout=action_row,
            )
        if self.profile_error:
            hint = QLabel(self.profile_error)
            hint.setObjectName("status")
            hint.setWordWrap(True)
            action_row.addWidget(hint, 1)

    def start_automatic(self):
        if self.workflow_mode != "automatic" or self.job or self.auto_enabled:
            return
        blocker = self._source_blocker()
        if blocker:
            self.log(blocker)
            self._toolbar()
            return
        try:
            self._save_workflow_profile()
        except Exception as error:
            self.error(f"无法保存自动流程模板：{error}")
            return
        self.auto_enabled = True
        self.auto_started_at = perf_counter()
        self.auto_elapsed = 0.0
        self.auto_batch_last_elapsed = None
        self.auto_batch_last_expected = None
        self.auto_timer.start()
        self.auto_runtime_timer.start()
        self._refresh_auto_runtime()
        self.log("自动流程已启动，等待稳定的完整点云组")
        self._toolbar()
        self.auto_poll()

    def stop_automatic(self, _checked=False, *, log_message=True):
        was_running = self.auto_enabled or self.auto_started_at is not None
        self.auto_enabled = False
        self.auto_timer.stop()
        stopping = bool(self.job and self.job.streaming)
        if stopping:
            self.job.cancelled.set()
        else:
            self._freeze_auto_runtime()
            self.auto_pending.clear()
        if log_message:
            suffix = (
                f" · 总耗时 {self._format_elapsed(self.auto_elapsed)}"
                if was_running else ""
            )
            self.log("自动流程正在停止，等待当前计算阶段结束" if stopping
                     else f"自动流程已停止{suffix}")
        self._refresh_auto_runtime()
        self._toolbar()

    def _freeze_auto_runtime(self):
        if self.auto_started_at is not None:
            self.auto_elapsed = self._current_auto_elapsed()
        self.auto_started_at = None
        self.auto_runtime_timer.stop()
        self._refresh_auto_runtime()

    def _freeze_auto_batch(self):
        if self.auto_batch_started_at is not None:
            self.auto_batch_elapsed = self._current_batch_elapsed()
        self.auto_batch_started_at = None
        self.auto_batch_last_elapsed = self.auto_batch_elapsed
        self.auto_batch_last_expected = self.auto_batch_expected_seconds
        self.auto_batch_total = 0
        self.auto_batch_completed = 0
        self.auto_batch_remaining_seconds = None

    def _automatic_discovery(self, progress):
        progress("扫描自动流程输入目录")
        groups = discover_stable_input_groups(
            self.source_path,
            minimum_file_age_seconds=1.0,
        )
        return [
            (name, files, acquisition_signature(files))
            for name, files in sorted(groups.items())
        ]

    def auto_poll(self):
        if (
            self.workflow_mode != "automatic"
            or not self.auto_enabled
            or self.job
            or not self.source_path
        ):
            return
        try:
            if self._source_blocker():
                return
        except Exception:
            return
        self._start(self._automatic_discovery, self._automatic_discovered)

    def _automatic_discovered(self, groups):
        if not self.auto_enabled or self.workflow_mode != "automatic":
            return
        pending = []
        for name, files, signature in groups:
            if name in self.auto_pending or self.auto_seen.get(name) == signature:
                continue
            pending.append((name, files, signature))
            self.auto_pending.add(name)
        if not pending:
            return
        profile_config = self._profile_config()
        cad = self.workflow_profile.get("cad") or {}
        dxf = Path(cad["dxf"]).resolve()
        selection = normalize_cad_path_selection(cad.get("selection") or {})
        step = float(cad.get("step", 0.05))

        try:
            active_backend = resolve_compute_backend(profile_config.compute_backend)
            requests, plan, manifest = prepare_batch(
                pending,
                profile_config,
                {"dxf": dxf, "selection": selection, "step": step},
                ROOT / "outputs",
                gpu=active_backend == "gpu",
            )
        except Exception as error:
            for name, _files, _signature in pending:
                self.auto_pending.discard(name)
                self.auto_seen[name] = _signature
                self.auto_records[f"prepare:{datetime.now().isoformat()}:{name}"] = {
                    "name": name, "status": "failed", "error": str(error),
                    "report": None, "summary_path": None,
                }
            self.log(f"自动批次准备失败 · {error}")
            self._draw()
            return
        self.auto_resources = plan
        self.auto_batch_total = len(requests)
        self.auto_batch_completed = 0
        self.auto_batch_started_at = perf_counter()
        self.auto_batch_elapsed = 0.0
        self.auto_batch_expected_seconds = None
        self.auto_batch_remaining_seconds = None
        self.auto_duration_history = load_duration_history(ROOT / "outputs")
        self.log(
            f"自动批次已排队 · {len(requests)} 组 · "
            f"{plan['workers']} 个并行进程 · 每进程 {plan['threads_per_worker']} 线程"
        )

        if not hasattr(self, "auto_pose_reference"):
            self.auto_pose_reference = {}
        reference_state = self.auto_pose_reference

        def process_batch(progress, updated, cancelled):
            return run_batch(
                requests,
                plan,
                manifest,
                progress,
                lambda record: updated({"type": "automatic_record", "record": record}),
                cancelled,
                reference_state=reference_state,
            )

        self.view_values[CAD_PAGE] = "automatic_measurements"
        self.select_page(CAD_PAGE)
        self._start(process_batch, self._automatic_batch_finished, streaming=True)
        self._refresh_auto_runtime()

    def _automatic_record_update(self, record):
        name = record["name"]
        key = record["id"]
        if key in self.auto_records:
            return
        self.auto_batch_completed += 1
        self.auto_records[key] = record
        self.auto_pending.discard(name)
        if record.get("status") != "cancelled":
            self.auto_seen[name] = tuple(tuple(item) for item in record["signature"])
        self._refresh_auto_runtime()
        if record.get("status") != "completed" or not record.get("summary_path"):
            self.log(f"自动处理{record.get('status')} · {name} · {record.get('error', '未知错误')}")
            self._draw()
            return
        try:
            result = result_from_summary(Path(record["summary_path"]))
            report = record.get("report")
            dxf = Path(report["source_dxf"]) if report and report.get("source_dxf") else None
            saved = json.loads(Path(record["summary_path"]).read_text(encoding="utf-8"))
            state = Group(
                [Path(path) for path in record["files"]],
                config_from_dict(saved),
                result=result,
                dxf=dxf,
                report=report,
                cad_step=float((report or {}).get("desktop_input", {}).get("step", 0.05)),
                automatic_result=True,
            )
            if dxf:
                state.entities = load_cad_entities(dxf)
            state.selection = normalize_cad_path_selection(
                (report or {}).get("desktop_input", {}).get("selection") or {}
            )
            state.region_stamp = None
            self.groups[name] = state
            self.group_combo.blockSignals(True)
            if self.group_combo.findText(name) < 0:
                self.group_combo.addItem(name)
            self.group_combo.setCurrentText(name)
            self.group_combo.blockSignals(False)
            self.group_name = name
            self.log(
                f"自动处理完成 · {name} · {result.merged_points:,} 点 · "
                f"CAD {report.get('measurement_quality', {}).get('status', '未知')} · "
                f"用时 {self._format_elapsed(record.get('elapsed_seconds', 0))}"
            )
        except Exception as error:
            self.log(f"自动结果已保存，但预览载入失败 · {name} · {error}")
        self._toolbar()
        self._draw()

    def _automatic_batch_finished(self, records):
        for record in records:
            self._automatic_record_update(record)
        self.auto_pending.clear()
        if not self.auto_enabled:
            self._freeze_auto_runtime()
        self._freeze_auto_batch()
        if records:
            self.view_values[CAD_PAGE] = "automatic_measurements"
            self.select_page(CAD_PAGE)
        self._toolbar()
        self._draw()
        self._refresh_auto_runtime()

    def closeEvent(self, event):
        if self.job is not None:
            self.closing = True
            self.cancel()
            event.ignore()
            return
        self.auto_timer.stop()
        self.auto_runtime_timer.stop()
        self.canvas.invalidate()
        event.accept()

    def import_root(self, root, source_kind=None):
        root = Path(root).resolve()
        if source_kind in {"folder", "camera"}:
            self.source_kind = source_kind
        self.source_path = root
        def discover(progress):
            progress("读取数据目录")
            files = sorted(root.glob("*.ply"))
            groups = {root.name: files, **discover_groups(root)} if files else discover_groups(root)
            if not groups:
                raise ValueError("目录中没有 PLY 点云")
            return groups
        def loaded(groups):
            for name, files in groups.items():
                state = self._new_group_state(name, files)
                self.groups[name] = state
            self.group_combo.blockSignals(True)
            self.group_combo.clear()
            self.group_combo.addItems(list(self.groups))
            self.group_combo.setCurrentText(next(iter(groups)))
            self.group_combo.blockSignals(False)
            self.change_group(next(iter(groups)))
            self.log(f"已导入 {len(groups)} 组")
            try:
                self._save_workflow_profile()
            except Exception as error:
                self.log(f"自动模板暂未保存：{error}")
            if self.workflow_mode == "automatic":
                QTimer.singleShot(0, self.start_automatic)
        self._start(discover, loaded)

    def choose_root(self):
        self.source_kind = "folder"
        self.choose_source()

    def change_group(self, name):
        if name not in self.groups:
            return
        self.group_name = name
        self.group_combo.setToolTip(name)
        self.select_page(1 if self.page == 1 else 0)

    def select_page(self, index):
        if not 0 <= index < len(STEPS):
            self.error(f"页面索引无效：{index}")
            return
        self.page = index
        self.single_file_view = False
        self.nav[index].setChecked(True)
        self.title.setText(STEPS[index])
        self.view_epoch += 1
        self.canvas.invalidate()
        self._toolbar()
        if self._showing_automatic_measurements():
            self._draw()
        elif self.job:
            self._empty("正在处理")
            self.pending_draw = True
        else:
            self._draw()

    def _empty(self, text):
        self.empty.setText(text)
        self.stack.setCurrentWidget(self.empty)

    def _button(self, text, callback, primary=False, enabled=True, layout=None):
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.setEnabled(enabled)
        button.clicked.connect(callback)
        (layout if layout is not None else self.tools).addWidget(button)
        return button

    def _views(self, options, layout=None):
        self.view = QComboBox()
        for title, key in options:
            self.view.addItem(title, key)
        selected = self.view_values.get(self.page)
        index = self.view.findData(selected)
        if index >= 0:
            self.view.setCurrentIndex(index)
        self.view.currentIndexChanged.connect(self.change_view)
        (layout if layout is not None else self.tools).addWidget(self.view)

    def _cloud_views(self):
        self.view = QComboBox()
        self.view.addItem("单视角检查", "single")
        self.view.addItem("融合结果", "merged")
        self.view.addItem("表面预览", "surface")
        selected = self.view_values.get(self.page, "view:0")
        self.view.setCurrentIndex(
            1 if selected == "merged" else 2 if selected == "surface" else 0
        )
        self.view.currentIndexChanged.connect(self.change_view)
        self.tools.addWidget(self.view)

        self.single_view = QComboBox()
        self.single_view.setMinimumWidth(260)
        self.single_view.setToolTip("选择要检查的单个原始点云")
        if self.state:
            for index, path in enumerate(self.state.files):
                self.single_view.addItem(f"点云 {index + 1} · {path.name}", f"view:{index}")
        selected_index = 0
        selected = self.view_values.get(self.page, "view:0")
        if isinstance(selected, str) and selected.startswith("view:"):
            try:
                selected_index = max(0, int(selected.split(":", 1)[1]))
            except ValueError:
                selected_index = 0
        self.single_view.setCurrentIndex(
            min(selected_index, max(0, self.single_view.count() - 1))
        )
        self.single_view.currentIndexChanged.connect(self.single_view_changed)
        self._update_single_view_enabled()
        self.tools.addWidget(self.single_view)

    def _update_single_view_enabled(self):
        """Only a single-view inspection mode can use the view selector."""
        enabled = bool(
            getattr(self, "view", None)
            and self.view.currentData() == "single"
            and self.state
            and self.state.files
        )
        if getattr(self, "single_view", None) is not None:
            self.single_view.setEnabled(enabled)

    def _cad_measurement_blocker(self):
        state = self.state
        if not state or not state.entities or not state.dxf:
            return "未导入 DXF"
        if state.pending is not None:
            return "存在未完成路径，请选择终点、确认单段或取消起点"
        try:
            validate_cad_path_selection({e.index for e in state.entities}, state.selection, state.entities)
        except RuntimeError as error:
            return str(error)
        if not state.result:
            return "没有处理结果，请先处理点云或打开历史结果"
        path = state.result.section_cloud_path
        if not path or not Path(path).is_file() or state.result.section_points < 30:
            return "最终截面不足 30 点或文件缺失"
        return ""

    def _update_cad_controls(self):
        if self.page != CAD_PAGE or getattr(self, "cad_hint", None) is None:
            return
        state = self.state
        has_report = bool(state and state.report)
        for index in range(1, self.view.count()):
            view_mode = self.view.itemData(index)
            aggregate = (
                view_mode == "automatic_measurements"
                and (bool(self.auto_records) or self.workflow_mode == "automatic")
            ) or (self.workflow_mode == "automatic" and view_mode == "measurements")
            self.view.model().item(index).setEnabled(has_report or aggregate)
        if not has_report and self.view.currentData() != "paths" and not self._showing_automatic_measurements():
            self.view.blockSignals(True)
            self.view.setCurrentIndex(0)
            self.view.blockSignals(False)
            self.view_values[CAD_PAGE] = "paths"
        manual = self.workflow_mode == "manual"
        editing = bool(manual and state and state.entities and self.view.currentData() == "paths")
        pending = state.pending if state else None
        selection = normalize_cad_path_selection(state.selection if state else {})
        for mode, button in self.cad_role_buttons.items():
            button.setEnabled(editing)
            button.setChecked(mode == self.cad_mode)
        self.single_button.setEnabled(editing and pending is not None)
        self.cancel_cad_button.setEnabled(editing and pending is not None)
        self.undo_cad_button.setEnabled(editing and bool(state.undo))
        self.clear_cad_button.setEnabled(editing and bool(
            pending is not None or selection["registration"] or selection["measurement"]))
        self.cad_spacing.blockSignals(True)
        self.cad_spacing.setValue(self.cad_step)
        self.cad_spacing.blockSignals(False)
        self.cad_spacing.setEnabled(bool(manual and state and state.entities))
        blocker = self._cad_measurement_blocker()
        self.measure_button.setEnabled(manual and not blocker)
        self.measure_button.setToolTip(blocker or "将 CAD 拟合到固定最终截面并计算偏差与角度")
        previous = self.cad_paths.currentData()
        self.cad_paths.blockSignals(True)
        self.cad_paths.clear()
        for key, label in (("registration_paths", "配准"), ("measurement_paths", "测量")):
            for index, path in enumerate(selection[key]):
                text = f"{label} {index + 1} · {len(path)} 段"
                self.cad_paths.addItem(text, (key, index))
                self.cad_paths.setItemData(self.cad_paths.count() - 1,
                                          "实体：" + ", ".join(map(str, path)),
                                          Qt.ItemDataRole.ToolTipRole)
        selected_index = self.cad_paths.findData(previous)
        if selected_index >= 0:
            self.cad_paths.setCurrentIndex(selected_index)
        self.cad_paths.blockSignals(False)
        self.cad_paths.setEnabled(editing and bool(self.cad_paths.count()))
        self.delete_cad_button.setEnabled(editing and bool(self.cad_paths.count()))
        filename = Path(state.dxf).name if state and state.dxf else "未导入 DXF"
        section_count = state.result.section_points if state and state.result else 0
        status = (
            f"{filename} · 配准 {len(selection['registration'])} 段"
            f" · 测量 {len(selection['measurement_paths'])} 组"
            f" / {len(selection['measurement'])} 段 · 最终截面 {section_count:,} 点"
        )
        if pending is not None:
            status += f" · 待确认起点：实体 {pending}"
        if has_report:
            status += f" · 质量 {state.report.get('measurement_quality', {}).get('status', '未知')}"
        if self._showing_automatic_measurements():
            status = f"自动测量汇总 · {len(self.auto_records)} 组次"
        self.cad_hint.setText(status)
        self.cad_hint.setToolTip(str(state.dxf) if state and state.dxf else "")

    def _cad_toolbar(self):
        panel = QWidget()
        rows = QVBoxLayout(panel)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(8)
        top, edit = QHBoxLayout(), QHBoxLayout()
        top.setSpacing(8)
        edit.setSpacing(8)
        rows.addLayout(top)
        rows.addLayout(edit)
        self.tools.addWidget(panel, 1)
        self._button("导入 DXF", self.open_cad,
                     enabled=bool(self.state) and self.workflow_mode == "manual", layout=top)
        if self.view_values.get(CAD_PAGE) in {"overlay", "measurement_overlay"}:
            self.view_values[CAD_PAGE] = "source_overlay"
        self._views([("路径选择", "paths"), ("来源与代表线", "source_overlay"),
                     ("测量结果", "measurements"),
                     ("偏差曲线", "curve"), ("测量报告", "report"),
                     ("自动测量汇总", "automatic_measurements")], layout=top)
        top.addWidget(QLabel("采样间距"))
        self.cad_spacing = QDoubleSpinBox()
        self.cad_spacing.setDecimals(6)
        self.cad_spacing.setRange(.000001, 100)
        self.cad_spacing.setSingleStep(.005)
        self.cad_spacing.setFixedWidth(105)
        self.cad_spacing.setValue(self.cad_step)
        self.cad_spacing.setToolTip("偏差采样间距，单位与 DXF 和点云坐标一致")
        self.cad_spacing.valueChanged.connect(self.cad_step_changed)
        top.addWidget(self.cad_spacing)
        top.addStretch()
        self.measure_button = self._button("配准并测量", self.measure, primary=True, layout=top)
        self.cad_role_group = QButtonGroup(panel)
        self.cad_role_buttons = {}
        for mode, label in (("registration", "配准路径"), ("measurement", "测量路径")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("cadRole", mode)
            button.setToolTip("先点起始实体，再点结束实体以填充连续路径；单段可单独确认")
            self.cad_role_group.addButton(button)
            self.cad_role_buttons[mode] = button
            button.clicked.connect(lambda checked=False, role=mode: self.cad_mode_changed(role))
            edit.addWidget(button)
        self.single_button = self._button("确认单段", self.cad_single, layout=edit)
        self.single_button.setToolTip("将当前起点实体确认为一条单段路径")
        self.cancel_cad_button = icon_button(
            self, QStyle.StandardPixmap.SP_DialogCancelButton, "取消未完成起点", self.cad_cancel)
        self.undo_cad_button = icon_button(
            self, QStyle.StandardPixmap.SP_ArrowBack, "撤销上一步 CAD 操作", self.cad_undo)
        edit.addWidget(self.cancel_cad_button)
        edit.addWidget(self.undo_cad_button)
        edit.addStretch()
        self.cad_paths = QComboBox()
        self.cad_paths.setMinimumWidth(135)
        self.cad_paths.setMaximumWidth(190)
        self.cad_paths.setToolTip("已选路径，悬停查看实体编号")
        edit.addWidget(self.cad_paths)
        self.delete_cad_button = icon_button(
            self, QStyle.StandardPixmap.SP_DialogDiscardButton, "删除选中的一组路径", self.cad_delete_path)
        edit.addWidget(self.delete_cad_button)
        self.clear_cad_button = icon_button(
            self, QStyle.StandardPixmap.SP_TrashIcon, "清空路径", lambda: None)
        menu = QMenu(self.clear_cad_button)
        menu.addAction("清空当前类型路径", self.cad_clear_current)
        menu.addAction("清空全部路径", self.cad_clear)
        self.clear_cad_button.setMenu(menu)
        self.clear_cad_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        edit.addWidget(self.clear_cad_button)
        self.cad_hint = QLabel()
        self.cad_hint.setObjectName("status")
        self.cad_hint.setWordWrap(True)
        self.cad_hint.setMinimumWidth(0)
        self.cad_hint.setTextFormat(Qt.TextFormat.PlainText)
        rows.addWidget(self.cad_hint)
        self._update_cad_controls()

    def _select_cloud_view(self, mode):
        self.view_values[self.page] = mode
        self.view_epoch += 1
        self.canvas.invalidate()
        self._draw()

    def single_view_changed(self):
        if self.view.currentData() == "single" and self.single_view.isEnabled():
            self._select_cloud_view(self.single_view.currentData() or "view:0")

    def _toolbar(self):
        # Toolbar widgets are deleted asynchronously; discard their Python
        # references before another page can invoke a shared view callback.
        self.single_view = None
        self.cad_hint = None
        while self.tools.count():
            item = self.tools.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.deleteLater()
        self.view = None
        state = self.state
        if self.page == 0:
            if self.single_file_view:
                back = self._button("返回文件列表", self.return_to_files)
                back.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
                back.setToolTip("返回文件列表")
            self._data_toolbar()
            self._button("预览选中文件", self.preview_selected,
                         enabled=bool(state and state.selected_file))
        elif self.page == 1:
            self._cloud_views()
            self._cloud_controls()
        elif self.page == 2:
            self._views([("法向截面", "plane"), ("框选主轴", "region"), ("切片预览", "slice"),
                         ("最终截面", "final"), ("原始截面", "raw"), ("邻域诊断", "neighbors")])
            self._button("截面参数", lambda: self.parameters("section"),
                         enabled=bool(state) and self.workflow_mode == "manual")
            self._button("开始处理", self.process, primary=True,
                         enabled=bool(state) and self.workflow_mode == "manual")
        elif self.page == 3:
            self._views([("位姿叠加", "alignment"), ("配准指标", "metrics"), ("质量判定", "quality"),
                         ("截面叠加", "section_alignment_audit"), ("覆盖诊断", "section_coverage_audit"),
                         ("多截面诊断", "multi_section_audit")])
            self.audit_mode = QComboBox()
            self.audit_mode.addItems(["四视图", "XY", "XZ", "YZ", "3D"])
            self.audit_mode.setCurrentText(self.view_values.get("audit_mode", "3D"))
            self.audit_mode.currentTextChanged.connect(self.audit_changed)
            self.tools.addWidget(self.audit_mode)
            self._point_size()
        elif self.page == CAD_PAGE:
            self._cad_toolbar()
        self.tools.addStretch()
        self._busy(self.job is not None)

    def _point_size(self):
        self.tools.addWidget(QLabel("点径"))
        size = QDoubleSpinBox()
        size.setDecimals(1)
        size.setRange(0.5, 10)
        size.setSingleStep(0.5)
        size.setValue(self.point_size)
        size.setFixedWidth(65)
        size.editingFinished.connect(lambda: self.size_changed(size.value()))
        self.tools.addWidget(size)

    def _cloud_controls(self):
        density = QComboBox()
        density.addItems(["快速", "标准", "精细", "高密度"])
        density.setCurrentIndex(self.level)
        density.currentIndexChanged.connect(self.level_changed)
        self.tools.addWidget(density)
        denoise = QCheckBox("去噪")
        denoise.setChecked(self.apply_denoise)
        denoise.toggled.connect(self.denoise_changed)
        self.tools.addWidget(denoise)
        self._point_size()

    def change_view(self):
        selected = self.view.currentData()
        self._update_single_view_enabled()
        if selected == "single":
            current = self.view_values.get(self.page)
            if not isinstance(current, str) or not current.startswith("view:"):
                self.view_values[self.page] = "view:0"
        else:
            self.view_values[self.page] = selected
        self.view_epoch += 1
        self.canvas.invalidate()
        self._draw()

    def audit_changed(self, mode):
        self.view_values["audit_mode"] = mode
        self.change_view()

    def size_changed(self, value):
        if self.point_size != value:
            self.point_size = value
            self.change_view()

    def level_changed(self, value):
        self.level = value
        self.change_view()

    def denoise_changed(self, value):
        self.apply_denoise = value
        self.change_view()

    def _table(self, columns, rows, *, measurement_summary=False):
        self.table.set_data(columns, rows, measurement_summary=measurement_summary)
        self.stack.setCurrentWidget(self.table)

    def _data(self):
        state = self.state
        rows = [[path.name, f"{path.stat().st_size / 1048576:.1f} MB" if path.exists() else "文件缺失"]
                for path in state.files]
        self._table(["文件", "大小"], rows)
        if state.selected_file in state.files:
            self.table.table.selectRow(state.files.index(state.selected_file))

    def return_to_files(self):
        self.select_page(0)

    def data_row_clicked(self, row, column):
        state = self.state
        if self.page != 0 or self.job or not state or not (0 <= row < len(state.files)):
            return
        state.selected_file = state.files[row]
        self.log(f"已选文件：{state.selected_file.name}")
        self._toolbar()
        self.preview_selected()

    def preview_selected(self):
        state = self.state
        if not state or not state.selected_file or self.job:
            return
        selected = Path(state.selected_file)
        epoch = self.view_epoch
        self._empty("生成单文件预览")

        def create(progress):
            preview = self.backend.single_preview(
                selected, state.config, self.level, self.apply_denoise, progress,
                state.result)
            figure = figures.cloud(preview.points, projection="3D",
                                   point_size=1.5,
                                   key=f"single:{selected}")
            return preview, figure.to_json()

        def done(value):
            preview, figure = value
            if self.view_epoch == epoch:
                self.single_file_view = True
                self._toolbar()
                self._show(figure, "view")
                self.log(f"{selected.name} · {len(preview.points):,} 个预览点")
            else:
                self.pending_draw = True

        self._start(create, done)

    def _draw(self):
        if self._showing_automatic_measurements():
            self.pending_draw = False
            self._table(*measurement_table(self.auto_records.values()), measurement_summary=True)
            return
        if self.job:
            self.pending_draw = True
            return
        state = self.state
        if not state:
            self._empty("未导入点云")
            return
        self.pending_draw = False
        if self.page == CAD_PAGE:
            self._update_cad_controls()
        mode = self.view.currentData() if self.view else None
        if self.page == 1 and mode == "single" and self.single_view is not None:
            mode = self.single_view.currentData() or "view:0"
        try:
            if self.page == 0:
                self._data()
                return
            if self.page == CAD_PAGE and mode == "paths":
                if not state.entities:
                    self._empty("未导入 DXF")
                else:
                    self._show(figures.cad(state.entities, state.selection, self.cad_mode, state.pending,
                                           str(state.dxf)), "cad")
                return
            if (self.page == 3 or self.page == CAD_PAGE or mode in {"merged", "surface", "final", "raw"}) and not state.result:
                self._empty("暂无处理结果")
                return
            if self.page == 3:
                self._draw_audit(mode)
                return
            if self.page == CAD_PAGE:
                self._draw_measurement(mode)
                return
            if self.page == 2 and mode in {"final", "raw"}:
                path = state.result.section_cloud_path if mode == "final" else state.result.raw_section_cloud_path
                if not path:
                    self._empty("暂无截面点云")
                    return
                self._figure_task(lambda progress: figures.section_cloud(
                    np.asarray(read_point_cloud(path).points),
                    config_from_dict(json.loads(state.result.summary_path.read_text(encoding="utf-8"))).section,
                    self.point_size, str(path), frozen=True))
                return
            config = copy.deepcopy(state.config)
            source_result = state.result if mode in {"merged", "surface"} else None
            selected_view = None
            if isinstance(mode, str) and mode.startswith("view:"):
                selected_view = int(mode.split(":", 1)[1])
                if not 0 <= selected_view < len(state.files):
                    raise ValueError("点云视角索引无效")
            epoch, page = self.view_epoch, self.page
            self._empty("读取点云")
            def create(progress):
                preview_files = (
                    [state.files[selected_view]]
                    if selected_view is not None
                    else state.files
                )
                reference, preview = self.backend.preview(
                    preview_files, config, self.level, self.apply_denoise, progress,
                    source_result, protect_calibration=page != 1)
                points = preview.points
                key = str(reference) + ("result" if source_result else "reference")
                if mode == "neighbors":
                    progress("扫描相邻截面")
                    return preview, scan_section_offsets(
                        points, config.section, half_range=config.section.offset_scan_half_range,
                        step=config.section.offset_scan_step), "table"
                if mode == "slice":
                    figure = figures.section_cloud(points, config.section, self.point_size, key)
                else:
                    mesh = self.backend.surface(points, self.level, config.compute_backend) if mode == "surface" else None
                    figure = figures.cloud(points, config.section if page == 2 else None,
                                           "XY" if mode == "region" else "3D", self.point_size, key, mesh)
                    if mode == "region":
                        figure.update_layout(dragmode="select", selectdirection="any")
                return preview, figure.to_json(), "region" if mode == "region" else "view"
            def rendered(value):
                preview, figure, kind = value
                if not source_result:
                    state.preview = preview
                if epoch != self.view_epoch:
                    self.pending_draw = True
                    return
                if kind == "table":
                    self._json_table(figure)
                else:
                    self._show(figure, kind)
                self.log(f"{len(preview.points):,} 预览点" + (" · 非最终截面" if mode == "slice" else ""))
            self._start(create, rendered)
        except Exception as error:
            self.error(str(error))

    def _show(self, figure, mode="view"):
        self.stack.setCurrentWidget(self.canvas)
        self.canvas.show_figure(figure, mode)

    def _figure_task(self, create, mode="view"):
        epoch = self.view_epoch
        self._empty("生成视图")
        def done(figure):
            if self.view_epoch == epoch:
                self._show(figure, mode)
                report = self.state.report if self.state else None
                if self.page == CAD_PAGE and report:
                    self.log(f"测量质量 · {report['measurement_quality']['status']}")
                else:
                    self.log("就绪")
            else:
                self.pending_draw = True
        self._start(lambda progress: create(progress).to_json(), done)

    def _json_table(self, value):
        if isinstance(value, dict):
            for name in ("candidates", "samples", "offsets", "rows"):
                if isinstance(value.get(name), list):
                    return self._json_table(value[name])
            self._table(["项目", "值"], [(key, json.dumps(v, ensure_ascii=False)) for key, v in value.items()])
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            columns = list(value[0])
            self._table(columns[:6], [[self._format(row.get(k)) for k in columns[:6]] for row in value])
        else:
            self._table(["结果"], [[json.dumps(value, ensure_ascii=False)]])

    @staticmethod
    def _format(value):
        return f"{value:.5g}" if isinstance(value, float) else value

    def _draw_audit(self, mode):
        state = self.state
        summary = json.loads(state.result.summary_path.read_text(encoding="utf-8"))
        if mode == "alignment":
            audit_mode = self.audit_mode.currentText()
            self._figure_task(lambda progress: figures.alignment(
                self.backend.alignment(state.result, progress), audit_mode, self.point_size,
                str(state.result.summary_path)))
        elif mode == "metrics":
            self._table(["文件", "视角", "标定 Fitness", "标定 RMSE", "融合", "截面"],
                        [[row["file_name"], row.get("pose_prior_label") or f"视角 {index + 1}",
                          self._format(row.get("icp_fitness")), self._format(row.get("icp_rmse")),
                          "是" if row.get("included_in_fusion", True) else "否",
                          row.get("section_validation_status")] for index, row in enumerate(summary["registration_metrics"])])
        elif mode == "quality":
            quality = processing_quality(summary)
            self._table(["项目", "结果"], [
                ["质量", quality["status"]],
                ["正式测量", "允许" if quality.get("formal_measurement_allowed") else "禁止"],
                *[["原因", reason] for reason in quality.get("reasons", [])]])
        else:
            path = getattr(state.result, mode + "_path", None)
            if path and path.is_file():
                self.image.load(path)
                self.stack.setCurrentWidget(self.image)
            else:
                self._empty("暂无诊断图")

    def interaction(self, event):
        if self.job or self.workflow_mode != "manual" or event.get("generation") != self.canvas.generation:
            return
        state = self.state
        if not state:
            return
        if event.get("type") == "region" and self.page == 2 and self.view.currentData() == "region":
            if state.preview is None:
                return
            bounds = event.get("bounds", [])
            def selected(value):
                state.config.section, count, confidence = value
                state.region_stamp = acquisition_signature(state.files)
                state.result = None
                state.report = None
                try:
                    self._save_workflow_profile(state)
                except Exception as error:
                    self.log(f"主轴已应用，但自动模板保存失败：{error}")
                self.log(f"主轴框已更新 · {count:,} 点 · 主轴方向性 {confidence:.3f}")
                self._draw()
            self._start(lambda progress: fit_selection(state.preview.points, state.config.section, bounds), selected)
        elif event.get("type") == "cad" and self.page == CAD_PAGE and self.view.currentData() == "paths":
            index = event.get("entity")
            if index not in {e.index for e in state.entities}:
                return
            selection, pending, message = select_cad_path_endpoint(
                state.selection, self.cad_mode, index, state.entities, state.pending)
            self._apply_cad_state(selection, pending)
            self.log(message)
            self._draw()

    def parameters(self, category=None):
        if not self.state or self.job:
            return
        previous = self.state.config
        dialog = Parameters(previous, self)
        if category:
            dialog.category.setCurrentIndex(max(0, dialog.category.findData(category)))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated = dialog.config
            frame_changed = (
                updated.registration.use_six_station_pose_prior
                or previous.calibration != updated.calibration or previous.denoise != updated.denoise
                or previous.registration.reference_vertex_count != updated.registration.reference_vertex_count)
            updated.registration.use_six_station_pose_prior = False
            updated.registration.use_global_registration = True
            updated.registration.station_pose_corrections = {}
            if frame_changed:
                updated.section.target_region = None
                self.state.region_stamp = None
                self.state.preview = None
            self.state.config = updated
            try:
                self._save_workflow_profile(self.state)
            except Exception as error:
                self.error(f"参数已更新，但自动模板保存失败：{error}")
                return
            self.log("参数已更新" + ("，请重新框选主轴" if frame_changed else ""))
            self._draw()

    def _ready_to_process(self, state):
        if len(state.files) not in (5, 6, 7, 8):
            raise ValueError("每组应包含 5 至 8 个 PLY")
        stamp = acquisition_signature(state.files)
        if state.config.section.target_region is None:
            raise ValueError("请在截面页框选主轴区域")
        if state.region_stamp is not None and state.region_stamp != stamp:
            raise ValueError("输入点云已变化，请重新预览并框选")
        state.config.validate()

    def process(self):
        state = self.state
        if not state:
            return
        try:
            self._ready_to_process(state)
        except Exception as error:
            self.error(str(error))
            return
        config = copy.deepcopy(state.config)
        def done(result):
            state.result = result
            state.report = None
            self.log(f"处理完成 · {result.merged_points:,} 点")
            self.view_values[3] = "alignment"
            self.select_page(3)
        self._start(lambda progress: self.backend.process(state.files, config,
                                                          progress, ROOT / "outputs"), done)

    def process_ready(self):
        self.error("多组并行处理已归入自动流程，请切换到自动流程后由系统自动调度")

    def open_result(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开处理结果", str(ROOT / "outputs"),
                                               "处理结果 (run_summary.json);;JSON (*.json)")
        if path:
            self.load_result(Path(path))

    def load_result(self, path):
        def load(progress):
            progress("载入历史结果")
            result = result_from_summary(path)
            summary = json.loads(path.read_text(encoding="utf-8"))
            state = Group([Path(p) for p in summary["input_files"]], config_from_dict(summary), result=result)
            # Old fixed-station sections belong to a different preview reference frame.
            if state.config.registration.use_six_station_pose_prior:
                state.config.section.target_region = None
            state.config.registration.use_six_station_pose_prior = False
            state.config.registration.use_global_registration = True
            state.config.registration.station_pose_corrections = {}
            reports = sorted(result.run_directory.glob("native_cad/*/report.json"))
            if reports:
                state.report = json.loads(reports[-1].read_text(encoding="utf-8"))
                state.dxf = Path(state.report["source_dxf"])
                state.entities = load_cad_entities(state.dxf)
                state.selection = normalize_cad_path_selection({
                    "registration_paths": state.report["fit"]["registration_paths"],
                    "measurement_paths": state.report["measurement_paths"]})
                state.cad_step = float(state.report.get("desktop_input", {}).get("step", .05))
            return state
        def done(state):
            name = f"{state.result.group_name} [{state.result.run_directory.name}]"
            self.groups[name] = state
            self.group_combo.blockSignals(True)
            self.group_combo.addItem(name)
            self.group_combo.setCurrentText(name)
            self.group_combo.blockSignals(False)
            self.group_name = name
            self.view_values[1] = "merged"
            self.select_page(1)
            self.log(f"历史结果 · {state.result.run_directory.name}")
        self._start(load, done)

    def open_cad(self):
        if not self.state or self.job:
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入 CAD", str(ROOT), "DXF (*.dxf)")
        if not path:
            return
        state = self.state
        if (state.selection or state.pending is not None or state.report) and QMessageBox.question(
            self, "替换 DXF", "替换图纸将清空当前 CAD 路径和测量结果，是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return

        def load(progress):
            progress(f"读取 DXF · {Path(path).name}")
            try:
                return load_cad_entities(Path(path))
            except Exception as error:
                raise RuntimeError(f"无法导入 DXF「{Path(path).name}」：{error}") from error

        def done(entities):
            state.dxf, state.entities = Path(path), entities
            state.selection, state.pending, state.undo = {}, None, []
            state.report = None
            state.cad_mode = "registration"
            try:
                self._save_workflow_profile(state)
            except Exception as error:
                self.log(f"DXF 已导入，但自动模板保存失败：{error}")
            self.view_values[CAD_PAGE] = "paths"
            self.select_page(CAD_PAGE)
            self.log(f"已导入 {Path(path).name} · {len(entities)} 个可选实体")
        self._start(load, done)

    def _apply_cad_state(self, selection, pending, mode=None, step=None):
        state = self.state
        before = (copy.deepcopy(normalize_cad_path_selection(state.selection)),
                  state.pending, state.cad_mode, state.cad_step, state.report)
        selection = normalize_cad_path_selection(selection)
        mode = state.cad_mode if mode is None else mode
        step = state.cad_step if step is None else step
        if (selection, pending, mode, step) == before[:4]:
            return False
        state.undo.append(before)
        if selection != before[0] or step != before[3]:
            state.report = None
        state.selection, state.pending, state.cad_mode, state.cad_step = selection, pending, mode, step
        persist = getattr(self, "_save_workflow_profile", None)
        if callable(persist):
            try:
                persist(state)
            except Exception as error:
                self.log(f"CAD 路径已更新，但自动模板保存失败：{error}")
        return True

    def cad_mode_changed(self, mode):
        state = self.state
        if not state or self.job or mode == state.cad_mode:
            return
        if state.pending is not None:
            self._apply_cad_state(state.selection, None, mode=mode)
            self.log("已取消未完成的 CAD 起点选择，可撤销恢复")
        else:
            state.cad_mode = mode
            try:
                self._save_workflow_profile(state)
            except Exception as error:
                self.log(f"CAD 模式已更新，但自动模板保存失败：{error}")
        self._draw()

    def cad_step_changed(self, value):
        if not self.state or self.job or value == self.cad_step:
            return
        self._apply_cad_state(self.state.selection, self.state.pending, step=value)
        self.log("采样间距已修改，测量结果待重新计算")
        self._draw()

    def cad_single(self):
        state = self.state
        if not state or self.job or state.pending is None:
            return
        selection, committed, message = select_cad_single_entity_path(
            state.selection, self.cad_mode, state.pending)
        if committed:
            self._apply_cad_state(selection, None)
        self.log(message)
        self._draw()

    def cad_cancel(self):
        state = self.state
        if not state or self.job or state.pending is None:
            return
        self._apply_cad_state(state.selection, None)
        self.log("已取消未完成起点")
        self._draw()

    def cad_undo(self):
        state = self.state
        if state and state.undo and not self.job:
            state.selection, state.pending, state.cad_mode, state.cad_step, state.report = state.undo.pop()
            self.log("已撤销上一步 CAD 操作")
            self._draw()

    def cad_delete_path(self):
        state = self.state
        selected = self.cad_paths.currentData()
        if not state or self.job or selected is None:
            return
        key, index = selected
        selection = normalize_cad_path_selection(state.selection)
        del selection[key][index]
        self._apply_cad_state(selection, None)
        self.log("已删除选中的一组路径")
        self._draw()

    def cad_clear_current(self):
        state = self.state
        if not state or self.job:
            return
        selection = normalize_cad_path_selection(state.selection)
        if self.cad_mode == "registration":
            selection["registration_paths"] = []
            selection["registration"] = []
            message = "已清空 CAD 配准路径"
        else:
            selection["measurement_paths"] = []
            selection["measurement"] = []
            message = "已清空 CAD 测量路径"
        self._apply_cad_state(selection, None)
        self.log(message)
        self._draw()

    def cad_clear(self):
        state = self.state
        if state and not self.job:
            self._apply_cad_state({}, None, mode="registration")
            self.log("已清空全部 CAD 路径")
            self._draw()

    def measure(self):
        state = self.state
        if not state or self.job:
            return
        blocker = self._cad_measurement_blocker()
        if blocker:
            self.error(blocker)
            return
        selection = normalize_cad_path_selection(state.selection)
        result, dxf, step = state.result, state.dxf, self.cad_step
        def done(report):
            state.report = report
            self.view_values[CAD_PAGE] = "source_overlay"
            self.select_page(CAD_PAGE)
            self.log(f"测量完成 · {report['measurement_quality']['status']}")
        self._start(lambda progress: self.backend.analyze(result, dxf, selection, step, progress), done)

    def _showing_automatic_measurements(self):
        return (
            self.page == CAD_PAGE
            and (
                self.view_values.get(CAD_PAGE) == "automatic_measurements"
                or (
                    self.workflow_mode == "automatic"
                    and self.view_values.get(CAD_PAGE) == "measurements"
                )
            )
        )

    def _draw_measurement(self, mode):
        report = self.state.report
        if mode == "automatic_measurements" or (
            mode == "measurements" and self.workflow_mode == "automatic"
        ):
            self._table(*measurement_table(self.auto_records.values()), measurement_summary=True)
            return
        if not report:
            self._empty("暂无测量结果")
        elif mode == "source_overlay":
            result = self.state.result
            self._figure_task(lambda progress: figures.cad_overlay(
                report, self.backend.cad_sources(result, report, progress), measurement_lines=True))
        elif mode == "measurements":
            rows = [["质量", report["measurement_quality"]["status"], "", "", ""]]
            rows.extend([[
                r.get("label"),
                self._format(r.get("mean_signed")),
                self._format(r.get("mean_signed_x")),
                self._format(r.get("mean_signed_y")),
                self._format(r.get("signed_angle_difference_deg")),
            ] for r in report.get("measurements", {}).values()])
            self._table(
                ["路径", "平均法向偏差", "平均 X 偏差", "平均 Y 偏差", "角度差"],
                rows,
            )
        elif mode == "curve":
            self._figure_task(lambda progress: figures.measurement_curves(report))
        else:
            quality = report["measurement_quality"]
            self._table(["项目", "结果"], [
                ["质量", quality["status"]],
                ["CAD 旋转角", self._format(report["fit"].get("cad_to_pointcloud_angle_deg"))],
                ["基准残差", self._format(report["fit"].get("mean_residual"))],
                ["测量坐标系", report.get("measurement_coordinate_system", "cad_original_xy")],
                ["点云变换", "未变换"],
                *[["原因", reason] for reason in [
                    *quality.get("failure_reasons", []), *quality.get("insufficient_data_reasons", [])]]])

    def show_logs(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("运行日志")
        dialog.resize(900, 520)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit("\n".join(self.logs))
        text.setReadOnly(True)
        layout.addWidget(text)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(lambda: text.setPlainText("\n".join(self.logs)))
        layout.addWidget(refresh)
        dialog.exec()

    def export_menu(self):
        if not self.state or not self.state.result:
            return
        menu = QMenu(self)
        menu.addAction("诊断包 ZIP", lambda: self.export(False))
        menu.addAction("正式测量包 ZIP", lambda: self.export(True))
        menu.addAction("当前视图 PNG", self.export_view)
        menu.addAction("结果目录", lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.state.result.run_directory))))
        menu.exec(self.export_button.mapToGlobal(self.export_button.rect().topRight()))

    def export(self, formal):
        state = self.state
        path, _ = QFileDialog.getSaveFileName(self, "导出结果", str(ROOT / (
            f"{state.result.group_name}_{'measurement' if formal else 'diagnostic'}.zip")), "ZIP (*.zip)")
        if path:
            self._start(lambda progress: self.backend.export(
                state.result, state.report, Path(path), formal, progress),
                lambda destination: self.log(f"已导出 {destination.name}"))

    def export_view(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出视图", str(ROOT / "view.png"), "PNG (*.png)")
        if path:
            if not self.stack.currentWidget().grab().save(path):
                self.error("无法写入图像")
            else:
                self.log("视图已导出")
