from __future__ import annotations

from pointcloud_app.cad_geometry import (
    CadEntity,
    load_cad_entities,
    _cad_entity_geometry,
    _cad_entity_catalog,
    cad_entity_connection_tolerance,
    build_cad_entity_topology,
    cad_contour_path_indices,
    validate_cad_selection,
    _normalize_cad_entity_indices,
    _normalize_cad_path_collection,
    normalize_cad_path_selection,
    _empty_cad_path_selection,
    normalize_cad_selection,
    select_cad_entity,
    select_cad_path_endpoint,
    select_cad_single_entity_path,
    validate_cad_path_selection,
    cad_entity_from_plotly_event,
    build_cad_selection_figure,
)
import hashlib
import heapq
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import ezdxf
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from media_analysis_core import analyze, create_height_colored_cad_overlay_image, sample_entity
from pointcloud_app.downstream_bridge import prepare_final_section_for_cad
from pointcloud_app.pipeline import PipelineResult








































def _prepare_section_state(result: PipelineResult) -> dict[str, Any]:
    state_key = f"cad-prepared-section-{result.run_directory.resolve()}"
    state = st.session_state.get(state_key)
    if isinstance(state, dict):
        section_path = Path(str(state.get("section_cloud_path", "")))
        report_path = Path(str(state.get("processing_report_path", "")))
        if section_path.is_file() and report_path.is_file():
            try:
                processing_report = json.loads(report_path.read_text(encoding="utf-8"))
                section_contract = processing_report.get("vertical_section", {})
            except (OSError, ValueError, TypeError):
                section_contract = None
            if (
                isinstance(section_contract, dict)
                and section_contract.get("coordinate_system") == "26xV01Q_final_section_xyz"
                and section_contract.get("pointcloud_transform_applied") is False
                and section_contract.get("coordinates_preserved_from_source") is True
                and section_contract.get("point_order_preserved") is True
            ):
                return state
    handoff = prepare_final_section_for_cad(result)
    state = {
        "package_directory": str(handoff.package_directory),
        "section_cloud_path": str(handoff.section_cloud_path),
        "processing_report_path": str(handoff.processing_report_path),
        "section_points": handoff.section_points,
    }
    st.session_state[state_key] = state
    return state


def _save_uploaded_dxf(upload: Any, output_directory: Path) -> tuple[Path, str]:
    content = upload.getvalue()
    if not content:
        raise RuntimeError("DXF 图纸为空")
    digest = hashlib.sha256(content).hexdigest()[:24]
    input_directory = output_directory / "cad_inputs"
    input_directory.mkdir(parents=True, exist_ok=True)
    dxf_path = input_directory / f"drawing_{digest}.dxf"
    if not dxf_path.exists():
        dxf_path.write_bytes(content)
    return dxf_path, digest


def _ensure_height_colored_cad_overlay(report: dict[str, Any], report_path: Path) -> Path:
    outputs = report.setdefault("outputs", {})
    image_path = Path(str(outputs.get("height_colored_cad_overlay_image", "")))
    overlay_version = int(outputs.get("height_colored_cad_overlay_version", 0) or 0)
    if image_path.is_file() and overlay_version >= 10:
        return image_path

    source_pointcloud = Path(str(report.get("source_pointcloud", "")))
    source_dxf = Path(str(report.get("source_dxf", "")))
    if not source_pointcloud.is_file() or not source_dxf.is_file():
        raise RuntimeError("缺少点云或 DXF 源文件，无法生成高度彩色对照图")
    image_path = report_path.parent / "height_colored_pointcloud_cad_overlay.png"
    registration_entities = report.get("fit", {}).get("entities", [])
    cad_transform = report.get("fit", {}).get("cad_transform")
    cad_selection = report.get("cad_selection", {})
    if not cad_selection:
        desktop_input = report.get("desktop_input", {})
        if isinstance(desktop_input, dict):
            cad_selection = desktop_input.get("selection", {})
    measurement_entities = []
    if isinstance(cad_selection, dict):
        measurement_paths = cad_selection.get("measurement_paths", [])
        if isinstance(measurement_paths, list):
            measurement_entities = [
                int(index)
                for path in measurement_paths
                if isinstance(path, list)
                for index in path
            ]
        if not measurement_entities:
            measurement_entities = cad_selection.get("measurement", [])
        if not measurement_entities:
            measurement_entities = [
                *cad_selection.get("x", []),
                *cad_selection.get("y", []),
            ]
    create_height_colored_cad_overlay_image(
        source_pointcloud,
        source_dxf,
        image_path,
        registration_entities=registration_entities if isinstance(registration_entities, list) else None,
        measurement_entities=measurement_entities if isinstance(measurement_entities, list) else None,
        cad_transform=cad_transform if isinstance(cad_transform, dict) else None,
    )
    outputs["height_colored_cad_overlay_image"] = str(image_path)
    outputs["height_colored_cad_overlay_version"] = 10
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path


def _render_height_colored_cad_overlay(report: dict[str, Any], report_path: Path) -> None:
    try:
        image_path = _ensure_height_colored_cad_overlay(report, report_path)
    except Exception as error:
        st.warning(f"高度彩色点云 / CAD 对照图生成失败：{error}")
        return

    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 18px;margin:10px 0 0;border:1px solid #dbe3ec;border-radius:12px 12px 0 0;background:#f8fafc;">
          <div style="font-size:1.35rem;font-weight:750;color:#0f172a;">最终清噪截面 / CAD 选中轮廓拟合图</div>
          <div style="padding:6px 14px;border-radius:999px;background:#e8f7ef;color:#07865f;font-weight:700;white-space:nowrap;">点云固定</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image(str(image_path), use_container_width=True)
    st.caption("彩色点云与上方清噪后截面使用同一批点和同一截面坐标，点云不旋转、不重排；仅将 CAD 拟合到固定点云上。蓝色为 CAD 配准路径，绿色为偏差与角度测量路径，浅灰线仅作图纸位置参考。")


def _render_cad_report(report: dict[str, Any], report_path: Path) -> None:
    fit = report.get("fit", {})
    raw_measurements = report.get("measurements", report.get("green_measurements", {}))
    measurement_entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_measurements, Mapping):
        for key, value in raw_measurements.items():
            if isinstance(value, Mapping):
                measurement_entries.append((str(key), dict(value)))

    def path_label(value: Mapping[str, Any], fallback: str = "测量线") -> str:
        entity_indices = value.get("entity_indices", [])
        if isinstance(entity_indices, (list, tuple)):
            normalized = []
            for item in entity_indices:
                try:
                    normalized.append(int(item))
                except (TypeError, ValueError):
                    continue
            if len(normalized) > 1:
                path_index = value.get("path_index", "?")
                return (
                    f"路径 {path_index}：实体 {normalized[0]}→{normalized[-1]}"
                    f"（{len(normalized)} 段）"
                )
            if normalized:
                return f"实体 {normalized[0]}"
        entity_index = value.get("entity_index")
        if entity_index not in (None, "?"):
            return f"实体 {entity_index}"
        return fallback

    def measurement_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        try:
            return int(item[1].get("path_index", 10**9)), item[0]
        except (TypeError, ValueError):
            return 10**9, item[0]

    measurement_entries.sort(key=measurement_sort_key)
    cad_to_pointcloud_angle = fit.get("cad_to_pointcloud_angle_deg")
    if cad_to_pointcloud_angle is None:
        cad_to_pointcloud_angle = -float(fit.get("pointcloud_to_cad_angle_deg", 0.0))
    fit_left, fit_center, fit_right = st.columns(3)
    fit_left.metric("CAD 配准基准线平均残差", f"{float(fit.get('mean_residual', 0.0)):.4f}")
    fit_center.metric("CAD 配准基准线 P95 残差", f"{float(fit.get('p95_residual', 0.0)):.4f}")
    fit_right.metric("CAD 拟合到点云", f"{float(cad_to_pointcloud_angle):.3f}°")
    measurement_quality = report.get("measurement_quality", {})
    if isinstance(measurement_quality, Mapping):
        quality_status = str(measurement_quality.get("status", "INSUFFICIENT_DATA")).upper()
        failure_reasons = measurement_quality.get("failure_reasons", [])
        insufficient_reasons = measurement_quality.get("insufficient_data_reasons", [])
        reasons = failure_reasons if quality_status == "FAIL" else insufficient_reasons
        reason_text = "；".join(str(item) for item in reasons if item)
        if quality_status == "PASS":
            st.success("正式测量结论：PASS。当前点云拼接、截面和 CAD 测量证据满足正式输出条件。")
        elif quality_status == "FAIL":
            st.error(
                "正式测量结论：FAIL。数值仍可查看，但当前结果不能作为合格判定。"
                + (f"原因：{reason_text}" if reason_text else "")
            )
        else:
            st.warning(
                "正式测量结论：INSUFFICIENT_DATA。当前数值仅供参考，不能输出正式合格结论。"
                + (f"原因：{reason_text}" if reason_text else "")
            )
        coverage_warnings = measurement_quality.get("coverage_diagnostic_warnings", [])
        if not isinstance(coverage_warnings, (list, tuple)):
            coverage_warnings = []
        if not coverage_warnings:
            processing_quality = report.get("processing_quality", {})
            if isinstance(processing_quality, Mapping):
                candidate_warnings = processing_quality.get("coverage_diagnostic_warnings", [])
                if isinstance(candidate_warnings, (list, tuple)):
                    coverage_warnings = candidate_warnings
        normalized_coverage_warnings = list(
            dict.fromkeys(str(item) for item in coverage_warnings if item)
        )
        if normalized_coverage_warnings:
            st.warning(
                "覆盖诊断提示：覆盖优先融合已保留互补视角；以下重叠区或截面差异仍需人工复核，"
                "正式 PASS 不等同于覆盖风险已经消失。"
                + "；".join(normalized_coverage_warnings)
            )
        with st.expander("查看正式测量质量闸门详情", expanded=False):
            st.json(dict(measurement_quality))
    registration_paths = fit.get("registration_paths", [])
    if isinstance(registration_paths, list) and registration_paths:
        st.caption(
            "CAD 配准路径："
            + path_label(
                {
                    "entity_indices": registration_paths[0],
                    "path_index": 1,
                },
                "未命名路径",
            )
        )

    if measurement_entries:
        st.markdown("**测量线结果（法向偏差 / 角度差）**")
        measurement_columns = st.columns(min(3, len(measurement_entries)))

        def measurement_number(value: object, format_spec: str, suffix: str = "") -> str:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return "不可用"
            return f"{format(number, format_spec)}{suffix}" if np.isfinite(number) else "不可用"

        for position, (_, measurement) in enumerate(measurement_entries):
            measurement_name = path_label(measurement)
            column = measurement_columns[position % len(measurement_columns)]
            column.metric(
                f"{measurement_name} 平均法向偏差",
                measurement_number(measurement.get("mean_signed"), "+.4f"),
            )
            column.caption(
                f"绝对值 / RMS：{measurement_number(measurement.get('mean_absolute'), '.4f')} / "
                f"{measurement_number(measurement.get('rms'), '.4f')}；"
                f"点云路径角：{measurement_number(measurement.get('pointcloud_path_angle_deg'), '+.3f', '°')}；"
                f"角度差：{measurement_number(measurement.get('signed_angle_difference_deg'), '+.3f', '°')}；"
                f"平均法向 X/Y：{measurement_number(measurement.get('mean_signed_x'), '+.4f')} / "
                f"{measurement_number(measurement.get('mean_signed_y'), '+.4f')}；"
                f"覆盖率：{float(measurement.get('coverage_ratio', 0.0)) * 100:.1f}%"
            )
    else:
        st.info("本次 CAD 分析没有可展示的测量线结果。")

    outputs = report.get("outputs", {})
    output_tabs = st.tabs(["CAD 对齐", "偏差曲线", "测量报告"])
    for tab, key, title in zip(
        output_tabs,
        ("cad_alignment_image", "curve_image", "report_image"),
        ("固定点云与 CAD 拟合", "测量线偏差曲线", "CAD 偏差测量报告"),
    ):
        image_path = Path(str(outputs.get(key, "")))
        with tab:
            if image_path.is_file():
                st.image(str(image_path), caption=title, use_container_width=True)
            else:
                st.warning(f"未找到{title}图片")

    st.markdown("**下载 CAD 分析结果**")
    downloads = [
        ("下载 CAD 分析 JSON", report_path, "application/json"),
        (
            "下载高度彩色 CAD 对照图",
            Path(str(outputs.get("height_colored_cad_overlay_image", ""))),
            "image/png",
        ),
        ("下载测量线角度 CSV", Path(str(outputs.get("measurement_angle_csv", outputs.get("angle_csv", "")))), "text/csv"),
    ]
    measurement_csvs = outputs.get("measurement_csvs", [])
    if isinstance(measurement_csvs, list):
        for csv_info in measurement_csvs:
            if not isinstance(csv_info, Mapping):
                continue
            measurement_name = path_label(csv_info)
            downloads.append(
                (
                    f"下载{measurement_name}偏差 CSV",
                    Path(str(csv_info.get("path", ""))),
                    "text/csv",
                )
            )
    elif outputs.get("x_csv") or outputs.get("y_csv"):
        downloads.extend(
            [
                ("下载 X 边偏差 CSV", Path(str(outputs.get("x_csv", ""))), "text/csv"),
                ("下载 Y 边偏差 CSV", Path(str(outputs.get("y_csv", ""))), "text/csv"),
            ]
        )
    for label, path, mime in downloads:
        if path.is_file():
            st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime, key=f"cad-download-{label}-{path}")


def render_cad_analysis_panel(result: PipelineResult) -> None:
    st.divider()
    st.header("8. CAD 偏差分析")
    st.caption("CAD 只在本步骤导入。当前页面直接使用第 7 步最终融合点云导出的清噪侧面截面；无需跳转或重新上传点云，CAD 只能拟合固定点云。")
    try:
        section_state = _prepare_section_state(result)
    except Exception as error:
        st.error(f"准备 CAD 分析截面失败：{error}")
        return

    package_directory = Path(str(section_state["package_directory"]))
    section_path = Path(str(section_state["section_cloud_path"]))
    preview_path = package_directory / "pointcloud_side_preview.png"
    preview_column, details_column = st.columns((1.1, 1.0))
    with preview_column:
        if preview_path.is_file():
            st.image(str(preview_path), caption="本次处理得到的最终侧面截面", use_container_width=True)
    with details_column:
        st.success(f"已准备 {int(section_state['section_points']):,} 个最终截面点用于 CAD 分析。")
        st.markdown(
            "1. 上传 DXF。  \n"
            "2. 点选 1 条 CAD 配准基准线。  \n"
            "3. 再点选 1 条或多条线段，进行偏差与角度测量。"
        )

    run_key = hashlib.sha1(str(result.run_directory.resolve()).encode("utf-8")).hexdigest()[:12]
    upload = st.file_uploader("上传 DXF 图纸", type=["dxf"], key=f"cad-dxf-upload-{run_key}")
    if upload is None:
        st.info("上传 DXF 后将在本页面显示 CAD 图纸和实体选择控件。")
        return

    output_directory = result.run_directory / "cad_analysis"
    try:
        dxf_path, dxf_digest = _save_uploaded_dxf(upload, output_directory)
        entities = load_cad_entities(dxf_path)
    except Exception as error:
        st.error(f"读取 DXF 失败：{error}")
        return

    available_indices = {entity.index for entity in entities}
    trace_entity_indices = [entity.index for entity in entities]
    selection_state_key = f"cad-click-selection-{run_key}-{dxf_digest}"
    click_signature_key = f"cad-click-signature-{run_key}-{dxf_digest}"
    chart_revision_state_key = f"cad-chart-revision-{run_key}-{dxf_digest}"
    registration_start_key = f"cad-registration-start-{run_key}-{dxf_digest}"
    measurement_start_key = f"cad-measurement-start-{run_key}-{dxf_digest}"
    selection_status_key = f"cad-selection-status-{run_key}-{dxf_digest}"
    raw_selection = normalize_cad_path_selection(st.session_state.get(selection_state_key))
    st.session_state[selection_state_key] = raw_selection
    chart_revision = int(st.session_state.get(chart_revision_state_key, 0))

    st.subheader("点击选择 CAD 线段")
    st.caption("在同一工具下先点击起始线段，再点击结束线段，程序会自动填充两条线段之间的连续 CAD 轮廓；只需一条线段时，可在起点提示处将其设为单段路径。点云始终保持固定，只有 CAD 按配准路径拟合。")
    mode_labels = {
        "CAD 配准（起始线段 → 结束线段）": "registration",
        "偏差与角度测量（起始线段 → 结束线段）": "measurement",
    }
    mode_label = st.radio(
        "当前选择工具",
        list(mode_labels),
        horizontal=True,
        key=f"cad-selection-mode-{run_key}-{dxf_digest}",
    )
    selection_mode = mode_labels[mode_label]

    clear_columns = st.columns(3)
    selection_changed = False
    if clear_columns[0].button("清除 CAD 配准路径", key=f"cad-clear-registration-{run_key}-{dxf_digest}"):
        raw_selection["registration"] = []
        raw_selection["registration_paths"] = []
        st.session_state.pop(registration_start_key, None)
        selection_changed = True
    if clear_columns[1].button("清除测量路径", key=f"cad-clear-measurement-{run_key}-{dxf_digest}"):
        raw_selection["measurement"] = []
        raw_selection["measurement_paths"] = []
        st.session_state.pop(measurement_start_key, None)
        selection_changed = True
    if clear_columns[2].button("清除全部", key=f"cad-clear-all-{run_key}-{dxf_digest}"):
        raw_selection = _empty_cad_path_selection()
        st.session_state.pop(registration_start_key, None)
        st.session_state.pop(measurement_start_key, None)
        selection_changed = True
    if selection_changed:
        chart_revision += 1
        st.session_state[chart_revision_state_key] = chart_revision
    st.session_state[selection_state_key] = raw_selection

    summary_columns = st.columns(2)
    registration_paths = cast(list[list[int]], raw_selection["registration_paths"])
    measurement_paths = cast(list[list[int]], raw_selection["measurement_paths"])
    registration_summary = "未选择"
    if registration_paths:
        registration_summary = (
            f"实体 {registration_paths[0][0]}→{registration_paths[0][-1]}"
            f"（{len(registration_paths[0])} 段）"
        )
    summary_columns[0].metric(
        "CAD 配准路径",
        registration_summary,
    )
    summary_columns[1].metric(
        "偏差与角度测量路径",
        f"已选 {len(measurement_paths)} 组 / {len(raw_selection['measurement'])} 段",
    )

    current_start_key = registration_start_key if selection_mode == "registration" else measurement_start_key
    pending_index = st.session_state.get(current_start_key)
    if pending_index is not None:
        st.info(f"已选择起始线段 {pending_index}，请继续点击结束线段；再次点击同一线段可取消。")
        if st.button(
            "将起始线段设为单段路径",
            key=f"cad-commit-single-{run_key}-{dxf_digest}-{selection_mode}",
            use_container_width=True,
        ):
            (
                updated_selection,
                committed,
                status_message,
            ) = select_cad_single_entity_path(
                raw_selection,
                selection_mode,
                int(pending_index),
            )
            st.session_state[selection_state_key] = updated_selection
            st.session_state[selection_status_key] = status_message
            if committed:
                st.session_state.pop(current_start_key, None)
                st.session_state[chart_revision_state_key] = chart_revision + 1
            st.rerun()
    saved_status = st.session_state.get(selection_status_key)
    if saved_status:
        st.caption(str(saved_status))

    selection_event = st.plotly_chart(
        build_cad_selection_figure(
            entities,
            raw_selection,
            pending_mode=selection_mode,
            pending_index=int(pending_index) if pending_index is not None else None,
        ),
        width="stretch",
        key=f"cad-drawing-{run_key}-{dxf_digest}-{chart_revision}",
        on_select="rerun",
        selection_mode="points",
        config={"scrollZoom": True, "displaylogo": False},
    )
    clicked_entity, click_signature = cad_entity_from_plotly_event(
        selection_event,
        available_indices,
        trace_entity_indices,
    )
    if click_signature is None:
        st.session_state.pop(click_signature_key, None)
    elif click_signature != st.session_state.get(click_signature_key):
        st.session_state[click_signature_key] = click_signature
        if clicked_entity is not None:
            updated_selection, updated_start, status_message = select_cad_path_endpoint(
                raw_selection,
                selection_mode,
                clicked_entity,
                entities,
                start_index=pending_index,
            )
            st.session_state[selection_state_key] = updated_selection
            st.session_state[current_start_key] = updated_start
            st.session_state[selection_status_key] = status_message
        st.session_state[chart_revision_state_key] = chart_revision + 1
        st.rerun()

    selection: dict[str, list[int] | list[list[int]]] | None = None
    try:
        selection = validate_cad_path_selection(
            available_indices,
            raw_selection,
            entities,
        )
    except RuntimeError as error:
        st.info(f"待完成：{error}")

    st.caption("蓝色为 CAD 配准路径，绿色为偏差与角度测量路径；每组路径都由起始线段和结束线段自动填充。滚轮缩放，拖拽平移。")
    with st.expander("显示 CAD 实体编号", expanded=False):
        st.dataframe(
            [{"实体": entity.index, "类型": entity.entity_type, "采样点": len(entity.points)} for entity in entities],
            hide_index=True,
            use_container_width=True,
        )

    step = st.number_input(
        "偏差采样间隔",
        min_value=0.005,
        max_value=2.0,
        value=0.05,
        step=0.005,
        format="%.4f",
        key=f"cad-sampling-step-{run_key}-{dxf_digest}",
    )
    signature = json.dumps(
        {
            "dxf": dxf_digest,
            "section": str(section_path),
            "selection": selection,
            "step": float(step),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    report_state_key = f"cad-analysis-report-{run_key}"
    if st.button(
        "执行 CAD 配准与偏差分析",
        type="primary",
        use_container_width=True,
        disabled=selection is None,
        key=f"cad-run-analysis-{run_key}-{dxf_digest}",
    ):
        try:
            with st.spinner("正在将 CAD 按配准基准线拟合到固定点云，并计算所选测量线的偏差和角度..."):
                report = analyze(
                    section_path,
                    dxf_path,
                    output_directory,
                    step=float(step),
                    registration_entities=selection["registration"],
                    measurement_entities=selection["measurement"],
                    registration_paths=selection["registration_paths"],
                    measurement_paths=selection["measurement_paths"],
                )
                report["cad_selection"] = selection
                report["prepared_section"] = section_state
                report_path = output_directory / "unified_cad_report.json"
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state[report_state_key] = {
                "signature": signature,
                "report": report,
                "report_path": str(report_path),
            }
        except Exception as error:
            st.error(f"CAD 偏差分析失败：{error}")

    report_state = st.session_state.get(report_state_key)
    if isinstance(report_state, dict) and report_state.get("signature") == signature:
        cad_report = dict(report_state["report"])
        cad_report_path = Path(str(report_state["report_path"]))
        _render_height_colored_cad_overlay(
            cad_report,
            cad_report_path,
        )
        st.subheader("CAD 偏差分析结果")
        _render_cad_report(
            cad_report,
            cad_report_path,
        )
