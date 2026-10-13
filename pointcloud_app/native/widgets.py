"""Native controls with bounded layouts and paginated data."""

from __future__ import annotations

import copy
import json
import math
import threading
import traceback

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QToolButton, QStyle, QSizePolicy, QDialog, QDialogButtonBox,
    QComboBox, QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QMessageBox, QFileDialog, QPushButton,
)

from .backend import Cancelled, config_from_dict


def icon_button(parent, icon, tooltip, callback):
    button = QToolButton(parent)
    button.setIcon(parent.style().standardIcon(icon))
    button.setToolTip(tooltip)
    button.setFixedSize(32, 32)
    button.clicked.connect(callback)
    return button


class Task(QThread):
    progress = Signal(str)
    updated = Signal(object)

    def __init__(self, operation, parent=None, streaming=False):
        super().__init__(parent)
        self.operation = operation
        self.cancelled = threading.Event()
        self.value = None
        self.error = None
        self.detail = ""
        self.streaming = streaming

    def checkpoint(self, message):
        if self.cancelled.is_set():
            raise Cancelled("已取消")
        self.progress.emit(str(message))

    def run(self):
        try:
            self.checkpoint("正在准备")
            if self.streaming:
                self.value = self.operation(self.checkpoint, self.updated.emit, self.cancelled)
            else:
                self.value = self.operation(self.checkpoint)
            if self.cancelled.is_set() and not self.streaming:
                raise Cancelled("已取消")
        except Exception as error:
            self.error = error
            self.detail = traceback.format_exc()


class PagedTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns, self.rows = [], []
        self.measurement_summary = False
        self._summary_widths = []
        self.page = 0
        self.page_size = 8
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(self)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(44)
        layout.addWidget(self.table, 1)
        footer = QHBoxLayout()
        footer.addStretch()
        self.previous = icon_button(self, QStyle.StandardPixmap.SP_ArrowLeft, "上一页", lambda: self.move(-1))
        self.next = icon_button(self, QStyle.StandardPixmap.SP_ArrowRight, "下一页", lambda: self.move(1))
        self.counter = QLabel()
        footer.addWidget(self.previous)
        footer.addWidget(self.counter)
        footer.addWidget(self.next)
        layout.addLayout(footer)

    def set_data(self, columns, rows, *, measurement_summary=False):
        same_columns = columns == self.columns
        self.columns, self.rows = columns, rows
        self.measurement_summary = bool(measurement_summary and len(columns) == 6)
        if not same_columns:
            self.page = 0
        self.page = min(self.page, max(0, math.ceil(len(rows) / self.page_size) - 1))
        self.refresh()
        header = self.table.horizontalHeader()
        wide = len(columns) > 6
        self.table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded if wide or self.measurement_summary
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header.setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed if self.measurement_summary else
            QHeaderView.ResizeMode.Interactive if wide else QHeaderView.ResizeMode.Stretch)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone if self.measurement_summary
                                    else Qt.TextElideMode.ElideRight)
        if self.measurement_summary:
            self.table.ensurePolished()
        row_height = max(62, self.table.fontMetrics().lineSpacing() * 2 + 18)
        self.table.verticalHeader().setDefaultSectionSize(
            row_height if self.measurement_summary else 58 if wide else 44)
        if self.measurement_summary:
            metrics = self.table.fontMetrics()
            self._summary_widths = [180, 96, 300, 128, 128, 128]
            for column in range(len(columns)):
                texts = [str(columns[column]), *[
                    "" if row[column] is None else str(row[column]) for row in rows]]
                texts = ["数据不足" if text == "INSUFFICIENT_DATA" else text for text in texts]
                width = max(metrics.horizontalAdvance(line) for text in texts for line in text.split("\n"))
                self._summary_widths[column] = max(self._summary_widths[column], width + 28)
            self._fit_columns()
        if wide and not same_columns:
            widths = (166, 100, 210, 114, 114, 114, 120, 100, 105)
            for index in range(len(columns)):
                self.table.setColumnWidth(index, widths[index] if index < len(widths) else 120)
        QTimer.singleShot(0, self._fit_rows)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_rows)

    def _fit_rows(self):
        self._fit_columns()
        size = max(1, self.table.viewport().height() // self.table.verticalHeader().defaultSectionSize())
        if size != self.page_size:
            first = self.page * self.page_size
            self.page_size = size
            self.page = first // size
            self.refresh()

    def _fit_columns(self):
        if not self.measurement_summary:
            return
        widths = self._summary_widths.copy()
        extra = max(0, self.table.viewport().width() - sum(widths))
        weights = (1, 0, 3, 1, 1, 1)
        additions = [extra * weight // sum(weights) for weight in weights]
        additions[2] += extra - sum(additions)
        for column, (width, addition) in enumerate(zip(widths, additions)):
            self.table.setColumnWidth(column, width + addition)

    def move(self, amount):
        self.page = max(0, min(math.ceil(len(self.rows) / self.page_size) - 1, self.page + amount))
        self.refresh()

    def refresh(self):
        rows = self.rows[self.page * self.page_size:(self.page + 1) * self.page_size]
        self.table.clear()
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        self.table.setRowCount(len(rows))
        alignments = (Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignCenter,
                      Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignRight,
                      Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignRight)
        if self.measurement_summary:
            for column, alignment in enumerate(alignments):
                self.table.horizontalHeaderItem(column).setTextAlignment(alignment | Qt.AlignmentFlag.AlignVCenter)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                text = "" if value is None else str(value)
                label = "数据不足" if text == "INSUFFICIENT_DATA" else text
                item = QTableWidgetItem(label)
                item.setToolTip(text)
                if self.measurement_summary:
                    item.setTextAlignment(alignments[j] | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        pages = max(1, math.ceil(len(self.rows) / self.page_size))
        self.counter.setText(f"{self.page + 1} / {pages}")
        self.previous.setEnabled(self.page > 0)
        self.next.setEnabled(self.page + 1 < pages)


class FitImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)

    def load(self, path):
        self.original = QPixmap(str(path))
        if self.original.isNull():
            raise ValueError("图像不存在或无法读取")
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        if not self.original.isNull():
            self.setPixmap(self.original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))


LABELS = {
    "use_surface_support_filter": "局部曲面支持去噪",
    "surface_support_neighbors": "曲面支持邻居数",
    "surface_support_distance_factor": "曲面支持距离 / 去噪体素",
    "voxel_size": "去噪体素", "use_statistical_filter": "统计滤波",
    "statistical_neighbors": "邻居数", "statistical_std_ratio": "标准差倍率",
    "use_radius_filter": "半径滤波", "radius": "滤波半径", "radius_min_points": "半径最少点数",
    "use_roi": "标定块 ROI", "minimum_points": "标定块最少点数",
    "exclude_from_fusion": "融合排除标定块", "exclusion_margin": "排除余量",
    "level_reference_plane": "自动调平", "leveling_plane_distance": "调平平面距离",
    "leveling_minimum_points": "调平最少点数", "angle_degrees": "法向方位角",
    "tilt_degrees": "法向俯仰角", "plane_offset": "平面偏移", "thickness": "截面厚度",
    "horizontal_min": "横向下限", "horizontal_max": "横向上限",
    "vertical_min": "竖向下限", "vertical_max": "竖向上限",
    "point_size": "导出点大小", "target_region": "主轴选区",
    "auto_offset_enabled": "自动偏移（手动启用）", "offset_scan_half_range": "邻域扫描半径",
    "offset_scan_step": "邻域扫描步长", "offset_min_quality_gain": "偏移最小增益",
    "use_component_filter": "连通域滤波", "component_cell_size": "连通域网格",
    "component_min_points": "连通域最少点数", "compute_backend": "计算后端",
    "fusion_voxel_size": "融合体素", "reference_vertex_count": "参考视角目标点数",
    "registration_voxel_size": "配准体素", "use_global_registration": "全局配准",
    "global_max_iterations": "全局迭代次数", "global_attempts": "全局尝试次数",
    "global_random_seed": "随机种子", "use_six_station_pose_prior": "工位先验",
    "station_pose_corrections": "工位微调", "use_multiscale_icp": "多尺度 ICP",
    "icp_max_iterations": "ICP 迭代次数", "icp_max_correspondence_distance": "ICP 对应距离",
    "use_point_to_plane_icp": "点到面 ICP", "coverage_preserving_fusion": "保留互补覆盖",
    "data_family": "数据类型", "profile_name": "参数方案", "profile_source": "方案来源",
    "x_min": "X 下限", "x_max": "X 上限", "y_min": "Y 下限", "y_max": "Y 上限",
    "z_min": "Z 下限", "z_max": "Z 上限",
    "normal_radius_factor": "法线半径倍率", "feature_radius_factor": "特征半径倍率",
    "global_distance_factor": "全局对应距离倍率", "fallback_icp_fitness_threshold": "回退 ICP 阈值",
    "skip_low_quality_views": "跳过低质量视角", "minimum_full_cloud_fitness": "全云最低 Fitness",
    "minimum_calibration_icp_fitness": "标定块最低 Fitness",
    "maximum_calibration_icp_rmse": "标定块最大 RMSE",
    "skip_low_quality_calibration_views": "跳过低质量标定块视角",
    "use_section_alignment_validation": "截面对齐验证", "section_alignment_thickness": "验证截面厚度",
    "section_alignment_overlap_distance": "截面重叠距离",
    "section_alignment_max_p90_distance": "截面最大 P90 距离",
    "section_alignment_minimum_overlap_points": "截面最少重叠点数",
    "section_alignment_probe_count": "验证截面数量", "section_alignment_probe_spacing": "验证截面间距",
    "section_alignment_minimum_views": "验证最少视角数",
    "pose_prior_calibration_voxel_size": "先验标定块体素",
    "pose_prior_translation_max_correspondence_distance": "平移微调最大对应距离",
    "pose_prior_rigid_max_rotation_degrees": "先验最大旋转修正",
    "pose_prior_rigid_max_translation": "先验最大平移修正",
    "calibration_graph_voxel_size": "标定块位姿图体素",
    "calibration_graph_max_correspondence_distance": "标定块位姿图对应距离",
    "calibration_graph_icp_max_iterations": "标定块位姿图迭代次数",
    "calibration_graph_max_rotation_degrees": "标定块位姿图旋转上限",
    "calibration_graph_max_translation": "标定块位姿图平移上限",
    "calibration_graph_minimum_adjacent_fitness": "标定块相邻边最低 Fitness",
    "calibration_graph_minimum_loop_fitness": "标定块闭环边最低 Fitness",
    "calibration_graph_maximum_adjacent_rmse": "标定块相邻边最大 RMSE",
    "calibration_graph_maximum_loop_rmse": "标定块闭环边最大 RMSE",
    "workpiece_graph_voxel_size": "工件诊断体素",
    "workpiece_graph_max_correspondence_distance": "工件诊断对应距离",
    "workpiece_graph_icp_max_iterations": "工件诊断迭代次数",
    "workpiece_graph_max_rotation_degrees": "工件诊断旋转上限",
    "workpiece_graph_max_translation": "工件诊断平移上限",
    "workpiece_graph_minimum_adjacent_fitness": "工件相邻边最低 Fitness",
    "workpiece_graph_minimum_loop_fitness": "工件闭环边最低 Fitness",
    "use_calibration_geometry_constraints": "标定块几何约束",
    "calibration_geometry_voxel_size": "几何约束体素",
    "calibration_geometry_plane_distance": "几何约束平面距离",
    "calibration_geometry_minimum_plane_points": "几何约束单平面最少点数",
    "calibration_geometry_minimum_planes": "几何约束最少平面数",
    "calibration_geometry_max_planes": "几何约束最多平面数",
    "calibration_geometry_max_normal_angle_degrees": "几何约束最大法向角",
    "calibration_geometry_max_centroid_distance": "几何约束最大中心距离",
}


def flatten(data, prefix=()):
    for key, value in data.items():
        path = (*prefix, key)
        if isinstance(value, dict) and key not in {"target_region", "station_pose_corrections"}:
            yield from flatten(value, path)
        else:
            yield path, value


class Parameters(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("参数")
        self.resize(650, 590)
        self.payload = copy.deepcopy(config.as_dict())
        self.config = config
        self.entries = list(flatten(self.payload))
        self.controls = []
        self.page = 0
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.category = QComboBox()
        for text, name in (("去噪", "denoise"), ("标定", "calibration"), ("拼接", "registration"),
                           ("截面", "section"), ("其他", "other")):
            self.category.addItem(text, name)
        top.addWidget(self.category)
        top.addStretch()
        load = icon_button(self, QStyle.StandardPixmap.SP_DialogOpenButton, "导入参数 JSON", self.load_config)
        save = icon_button(self, QStyle.StandardPixmap.SP_DialogSaveButton, "导出参数 JSON", self.save_config)
        top.addWidget(load)
        top.addWidget(save)
        layout.addLayout(top)
        self.form = QFormLayout()
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(self.form)
        layout.addStretch()
        footer = QHBoxLayout()
        self.previous = icon_button(self, QStyle.StandardPixmap.SP_ArrowLeft, "上一页", lambda: self.move(-1))
        self.next = icon_button(self, QStyle.StandardPixmap.SP_ArrowRight, "下一页", lambda: self.move(1))
        self.counter = QLabel()
        footer.addWidget(self.previous)
        footer.addWidget(self.counter)
        footer.addWidget(self.next)
        footer.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("应用")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        layout.addLayout(footer)
        self.category.currentIndexChanged.connect(self.change_category)
        self.refresh()

    def collect(self):
        for path, control, kind in self.controls:
            if kind == "bool":
                value = control.isChecked()
            elif kind == "number":
                value = control.value()
            elif kind == "backend":
                value = control.currentText()
            else:
                value = json.loads(control.text())
            target = self.payload
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        self.entries = list(flatten(self.payload))

    def subset(self):
        category = self.category.currentData()
        automatic_settings = {"use_six_station_pose_prior", "station_pose_corrections", "use_global_registration"}
        return [(p, v) for p, v in self.entries
                if (p[0] == category or (category == "other" and len(p) == 1))
                and not (p[0] == "registration" and any(key in automatic_settings for key in p[1:]))]

    def refresh(self):
        self.controls = []
        while self.form.rowCount():
            self.form.removeRow(0)
        rows = self.subset()
        for path, value in rows[self.page * 8:self.page * 8 + 8]:
            key = path[-1]
            if isinstance(value, bool):
                control, kind = QCheckBox(), "bool"
                control.setChecked(value)
            elif isinstance(value, (int, float)):
                control = QSpinBox() if isinstance(value, int) else QDoubleSpinBox()
                control.setRange(-1_000_000_000, 1_000_000_000)
                if isinstance(control, QDoubleSpinBox):
                    control.setDecimals(6)
                    control.setSingleStep(.01)
                control.setValue(value)
                kind = "number"
            elif key == "compute_backend":
                control, kind = QComboBox(), "backend"
                control.addItems(["auto", "cpu", "gpu"])
                control.setCurrentText(value)
            else:
                control, kind = QLineEdit(json.dumps(value, ensure_ascii=False)), "json"
            label = QLabel(LABELS.get(key, key.replace("_", " ")))
            label.setFixedWidth(315)
            label.setWordWrap(True)
            label.setToolTip(".".join(path))
            control.setToolTip(".".join(path))
            control.setMinimumHeight(30)
            self.form.addRow(label, control)
            self.controls.append((path, control, kind))
        pages = max(1, math.ceil(len(rows) / 8))
        self.counter.setText(f"{self.page + 1} / {pages}")
        self.previous.setEnabled(self.page > 0)
        self.next.setEnabled(self.page + 1 < pages)

    def change_category(self):
        try:
            self.collect()
            self.page = 0
            self.refresh()
        except ValueError as error:
            QMessageBox.warning(self, "参数", str(error))

    def move(self, direction):
        try:
            self.collect()
            self.page += direction
            self.refresh()
        except ValueError as error:
            QMessageBox.warning(self, "参数", str(error))

    def apply(self):
        try:
            self.collect()
            self.config = config_from_dict(self.payload)
            self.accept()
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, "参数", str(error))

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入参数", "", "JSON (*.json)")
        if path:
            try:
                from pathlib import Path
                self.payload = config_from_dict(json.loads(Path(path).read_text(encoding="utf-8"))).as_dict()
                self.entries = list(flatten(self.payload))
                self.page = 0
                self.refresh()
            except Exception as error:
                QMessageBox.warning(self, "参数", str(error))

    def save_config(self):
        try:
            from pathlib import Path
            self.collect()
            config = config_from_dict(self.payload)
            path, _ = QFileDialog.getSaveFileName(self, "导出参数", "parameters.json", "JSON (*.json)")
            if path:
                Path(path).write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as error:
            QMessageBox.warning(self, "参数", str(error))
