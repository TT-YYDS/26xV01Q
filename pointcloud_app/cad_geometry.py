"""Shared CAD entity selection and topology, independent of any UI."""
from __future__ import annotations
import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import ezdxf
import numpy as np
import plotly.graph_objects as go
from media_analysis_core import sample_entity

MEASUREMENT_PATH_COLORS = (
    "#00a651", "#d13c75", "#8b5cf6", "#008c95",
    "#c95716", "#a38b00", "#c43c39", "#687d32",
)

@dataclass(frozen=True)
class CadEntity:
    index: int
    entity_type: str
    points: np.ndarray

    @property
    def label(self) -> str:
        return f"实体 {self.index} · {self.entity_type} · {len(self.points)} 个采样点"


def load_cad_entities(dxf_path: Path) -> list[CadEntity]:
    document = ezdxf.readfile(str(dxf_path))
    catalog: list[CadEntity] = []
    for index, entity in enumerate(document.modelspace()):
        try:
            points = np.asarray(sample_entity(entity), dtype=np.float64)
        except Exception:
            continue
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            continue
        finite_points = points[np.isfinite(points).all(axis=1)]
        if not len(finite_points):
            continue
        catalog.append(CadEntity(index=index, entity_type=entity.dxftype(), points=finite_points))
    if not catalog:
        raise RuntimeError("DXF 中没有可用于配准的线或圆弧实体")
    return catalog


def _cad_entity_geometry(entity: object) -> np.ndarray:
    if isinstance(entity, CadEntity):
        points = entity.points
    else:
        points = np.asarray(entity, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    return points


def _cad_entity_catalog(entities: object) -> dict[int, np.ndarray]:
    if isinstance(entities, Mapping):
        items = entities.items()
    else:
        try:
            items = enumerate(cast(Iterable[object], entities))
        except TypeError:
            return {}
    catalog: dict[int, np.ndarray] = {}
    for position, entity in items:
        try:
            index = int(entity.index) if isinstance(entity, CadEntity) else int(position)
        except (AttributeError, TypeError, ValueError):
            continue
        geometry = _cad_entity_geometry(entity)
        if len(geometry):
            catalog[index] = geometry
    return catalog


def cad_entity_connection_tolerance(
    entities: object,
    tolerance: float | None = None,
) -> float:
    if tolerance is not None:
        normalized = float(tolerance)
        if not np.isfinite(normalized) or normalized <= 0:
            raise ValueError("CAD 线段连接容差必须大于零")
        return normalized
    catalog = _cad_entity_catalog(entities)
    endpoints = [geometry[[0, -1]] for geometry in catalog.values() if len(geometry) >= 2]
    if not endpoints:
        return 1e-6
    all_endpoints = np.vstack(endpoints)
    span = float(np.ptp(all_endpoints, axis=0).max()) if len(all_endpoints) else 0.0
    return max(1e-6, span * 1e-7)


def build_cad_entity_topology(
    entities: object,
    tolerance: float | None = None,
) -> dict[int, set[int]]:
    catalog = _cad_entity_catalog(entities)
    topology: dict[int, set[int]] = {index: set() for index in catalog}
    if len(catalog) < 2:
        return topology
    connection_tolerance = cad_entity_connection_tolerance(catalog, tolerance)
    for left_position, (left_index, left_geometry) in enumerate(catalog.items()):
        left_endpoints = left_geometry[[0, -1]]
        for right_index, right_geometry in list(catalog.items())[left_position + 1 :]:
            right_endpoints = right_geometry[[0, -1]]
            distances = np.linalg.norm(
                left_endpoints[:, None, :] - right_endpoints[None, :, :],
                axis=2,
            )
            if float(np.min(distances)) <= connection_tolerance:
                topology[left_index].add(right_index)
                topology[right_index].add(left_index)
    return topology


def cad_contour_path_indices(
    entities: object,
    start_index: int,
    end_index: int,
    topology: Mapping[int, object] | None = None,
) -> list[int]:
    catalog = _cad_entity_catalog(entities)
    start_index = int(start_index)
    end_index = int(end_index)
    if start_index not in catalog or end_index not in catalog:
        return []
    if start_index == end_index:
        return [start_index]
    graph = topology if topology is not None else build_cad_entity_topology(catalog)
    lengths = {
        index: float(np.linalg.norm(np.diff(geometry, axis=0), axis=1).sum())
        if len(geometry) > 1
        else 1e-9
        for index, geometry in catalog.items()
    }
    distances = {index: float("inf") for index in catalog}
    previous: dict[int, int] = {}
    distances[start_index] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start_index)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance > distances[current] + 1e-12:
            continue
        if current == end_index:
            break
        neighbors = graph.get(current, ())
        if isinstance(neighbors, Mapping):
            neighbor_indices = neighbors.keys()
        else:
            neighbor_indices = neighbors
        for neighbor_value in sorted(neighbor_indices, key=lambda value: int(value)):
            neighbor = int(neighbor_value)
            if neighbor not in catalog:
                continue
            edge_cost = max(1e-9, (lengths[current] + lengths[neighbor]) * 0.5)
            candidate_distance = current_distance + edge_cost
            if candidate_distance + 1e-12 < distances[neighbor]:
                distances[neighbor] = candidate_distance
                previous[neighbor] = current
                heapq.heappush(queue, (candidate_distance, neighbor))
    if end_index not in previous:
        return []
    path = [end_index]
    while path[-1] != start_index:
        prior = previous.get(path[-1])
        if prior is None:
            return []
        path.append(prior)
    path.reverse()
    return path


def validate_cad_selection(
    available_indices: set[int],
    registration_index: int | None,
    measurement_indices: list[int],
) -> dict[str, list[int]]:
    if registration_index is None or not measurement_indices:
        raise RuntimeError("请先选择 1 条 CAD 配准基准线，以及至少 1 条偏差与角度测量线")
    normalized_measurements = _normalize_cad_entity_indices(measurement_indices)
    selected = [int(registration_index), *normalized_measurements]
    unknown = sorted(set(selected) - available_indices)
    if unknown:
        raise RuntimeError("选择的 CAD 实体不存在：" + ", ".join(map(str, unknown)))
    if int(registration_index) in normalized_measurements:
        raise RuntimeError("同一条 CAD 线段不能同时作为配准基准线和测量线")
    return {
        "registration": [int(registration_index)],
        "measurement": normalized_measurements,
    }


def _normalize_cad_entity_indices(values: object) -> list[int]:
    if not isinstance(values, (list, tuple, set)):
        return []

    indices: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in indices:
            indices.append(index)
    return indices


def _normalize_cad_path_collection(values: object) -> list[list[int]]:
    if not isinstance(values, (list, tuple, set)):
        return []
    values_list = list(values)
    if not values_list:
        return []
    if all(not isinstance(value, (list, tuple, set)) for value in values_list):
        path = _normalize_cad_entity_indices(values_list)
        return [path] if path else []
    paths: list[list[int]] = []
    for value in values_list:
        path = _normalize_cad_entity_indices(value)
        if path and path not in paths and path[::-1] not in paths:
            paths.append(path)
    return paths


def normalize_cad_path_selection(selection: object) -> dict[str, list[int] | list[list[int]]]:
    normalized: dict[str, list[int] | list[list[int]]] = {
        "registration": [],
        "measurement": [],
        "registration_paths": [],
        "measurement_paths": [],
    }
    if not isinstance(selection, Mapping):
        return normalized

    if "registration_paths" in selection:
        registration_paths = _normalize_cad_path_collection(selection.get("registration_paths"))
    else:
        registration_values = selection.get("registration")
        if registration_values is None:
            registration_values = selection.get("fit", [])
        registration_paths = _normalize_cad_path_collection(registration_values)

    if "measurement_paths" in selection:
        measurement_paths = _normalize_cad_path_collection(selection.get("measurement_paths"))
    else:
        measurement_values = selection.get("measurement")
        if measurement_values is None:
            measurement_values = [
                *_normalize_cad_entity_indices(selection.get("x", [])),
                *_normalize_cad_entity_indices(selection.get("y", [])),
            ]
        measurement_paths = [
            [index]
            for index in _normalize_cad_entity_indices(measurement_values)
        ]

    registration_indices = [index for path in registration_paths for index in path]
    measurement_indices = [index for path in measurement_paths for index in path]
    normalized["registration_paths"] = registration_paths
    normalized["measurement_paths"] = measurement_paths
    normalized["registration"] = registration_indices
    normalized["measurement"] = measurement_indices
    return normalized


def _empty_cad_path_selection() -> dict[str, list[int] | list[list[int]]]:
    return {
        "registration": [],
        "measurement": [],
        "registration_paths": [],
        "measurement_paths": [],
    }


def normalize_cad_selection(selection: object) -> dict[str, list[int]]:
    if isinstance(selection, Mapping) and (
        "registration_paths" in selection or "measurement_paths" in selection
    ):
        path_selection = normalize_cad_path_selection(selection)
        return {
            "registration": list(cast(list[int], path_selection["registration"])),
            "measurement": list(cast(list[int], path_selection["measurement"])),
        }
    normalized = {"registration": [], "measurement": []}
    if not isinstance(selection, Mapping):
        return normalized

    registration_values = selection.get("registration")
    measurement_values = selection.get("measurement")
    if registration_values is None and measurement_values is None:
        registration_values = selection.get("fit", [])
        measurement_values = [
            *_normalize_cad_entity_indices(selection.get("x", [])),
            *_normalize_cad_entity_indices(selection.get("y", [])),
        ]

    normalized["registration"] = _normalize_cad_entity_indices(registration_values)[:1]
    normalized["measurement"] = _normalize_cad_entity_indices(measurement_values)
    if normalized["registration"]:
        normalized["measurement"] = [
            index
            for index in normalized["measurement"]
            if index != normalized["registration"][0]
        ]
    return normalized


def select_cad_entity(
    selection: object,
    mode: str,
    entity_index: int,
) -> dict[str, list[int]]:
    if mode not in {"registration", "measurement"}:
        raise ValueError(f"Unsupported CAD selection mode: {mode}")

    updated = normalize_cad_selection(selection)
    entity_index = int(entity_index)
    if mode == "measurement":
        if entity_index in updated["measurement"]:
            updated["measurement"].remove(entity_index)
        else:
            updated["measurement"].append(entity_index)
        updated["registration"] = [
            index for index in updated["registration"] if index != entity_index
        ]
        return updated

    updated["registration"] = [entity_index]
    updated["measurement"] = [
        index for index in updated["measurement"] if index != entity_index
    ]
    return updated


def select_cad_path_endpoint(
    selection: object,
    mode: str,
    entity_index: int,
    entities: object,
    start_index: int | None = None,
) -> tuple[dict[str, list[int] | list[list[int]]], int | None, str]:
    if mode not in {"registration", "measurement"}:
        raise ValueError(f"Unsupported CAD path selection mode: {mode}")

    updated = normalize_cad_path_selection(selection)
    entity_index = int(entity_index)
    if start_index is None:
        return updated, entity_index, f"已选择起始线段 {entity_index}，请点击结束线段"
    start_index = int(start_index)
    if start_index == entity_index:
        return updated, None, f"已取消起始线段 {entity_index}，请重新点击起始线段"

    path = cad_contour_path_indices(entities, start_index, entity_index)
    if not path:
        return updated, None, "起始线段和结束线段不在同一条连续 CAD 轮廓上，请重新选择"

    registration_paths = [
        list(path)
        for path in cast(list[list[int]], updated["registration_paths"])
    ]
    measurement_paths = [
        list(path)
        for path in cast(list[list[int]], updated["measurement_paths"])
    ]
    registration_indices = set(cast(list[int], updated["registration"]))
    measurement_indices = set(cast(list[int], updated["measurement"]))
    path_set = set(path)

    if mode == "registration":
        if path_set & measurement_indices:
            return updated, None, "配准路径不能包含已经选择为测量路径的线段，请重新选择"
        registration_paths = [path]
        registration_indices = path_set
        message = (
            f"CAD 配准路径已自动填充 {len(path)} 条连续线段"
            f"（实体 {start_index} → {entity_index}）"
        )
    else:
        if path_set & registration_indices:
            return updated, None, "测量路径不能包含 CAD 配准路径中的线段，请重新选择"
        if not any(path == existing or path == existing[::-1] for existing in measurement_paths):
            measurement_paths.append(path)
            measurement_indices.update(path_set)
        message = (
            f"偏差与角度测量路径已自动填充 {len(path)} 条连续线段"
            f"（实体 {start_index} → {entity_index}）"
        )

    updated["registration_paths"] = registration_paths
    updated["measurement_paths"] = measurement_paths
    updated["registration"] = [index for item in registration_paths for index in item]
    updated["measurement"] = [index for item in measurement_paths for index in item]
    return updated, None, message


def select_cad_single_entity_path(
    selection: object,
    mode: str,
    entity_index: int,
) -> tuple[dict[str, list[int] | list[list[int]]], bool, str]:
    """Commit the pending entity as a one-segment CAD path.

    A repeated click still cancels a pending endpoint in the chart. This
    explicit action keeps that cancellation gesture while making it possible
    to select a single line or arc as a measurement path.
    """
    if mode not in {"registration", "measurement"}:
        raise ValueError(f"Unsupported CAD selection mode: {mode}")

    updated = normalize_cad_path_selection(selection)
    entity_index = int(entity_index)
    registration_paths = [
        list(path)
        for path in cast(list[list[int]], updated["registration_paths"])
    ]
    measurement_paths = [
        list(path)
        for path in cast(list[list[int]], updated["measurement_paths"])
    ]
    registration_indices = set(cast(list[int], updated["registration"]))
    measurement_indices = set(cast(list[int], updated["measurement"]))

    if mode == "registration":
        if entity_index in measurement_indices:
            return (
                updated,
                False,
                "配准路径不能包含已经选择为测量路径的线段，请重新选择",
            )
        registration_paths = [[entity_index]]
        message = f"CAD 配准路径已选择单条线段（实体 {entity_index}）"
    else:
        if entity_index in registration_indices:
            return (
                updated,
                False,
                "测量路径不能包含 CAD 配准路径中的线段，请重新选择",
            )
        single_path = [entity_index]
        if not any(
            path == single_path or path == single_path[::-1]
            for path in measurement_paths
        ):
            measurement_paths.append(single_path)
            message = f"偏差与角度测量路径已选择单条线段（实体 {entity_index}）"
        else:
            message = f"实体 {entity_index} 已在测量路径中"

    updated["registration_paths"] = registration_paths
    updated["measurement_paths"] = measurement_paths
    updated["registration"] = [index for path in registration_paths for index in path]
    updated["measurement"] = [index for path in measurement_paths for index in path]
    return updated, True, message


def validate_cad_path_selection(
    available_indices: set[int],
    selection: object,
    entities: object | None = None,
) -> dict[str, list[int] | list[list[int]]]:
    normalized = normalize_cad_path_selection(selection)
    registration_paths = [
        list(path)
        for path in cast(list[list[int]], normalized["registration_paths"])
    ]
    measurement_paths = [
        list(path)
        for path in cast(list[list[int]], normalized["measurement_paths"])
    ]
    if len(registration_paths) != 1 or not registration_paths[0]:
        raise RuntimeError("请先选择 CAD 配准的起始/结束线段，或将起始线段设为单段路径")
    if not measurement_paths or any(not path for path in measurement_paths):
        raise RuntimeError("请至少选择 1 组偏差与角度测量路径（可为单段路径）")

    all_indices = [
        *cast(list[int], normalized["registration"]),
        *cast(list[int], normalized["measurement"]),
    ]
    unknown = sorted(set(all_indices) - set(available_indices))
    if unknown:
        raise RuntimeError("选择的 CAD 实体不存在：" + ", ".join(map(str, unknown)))
    overlap = sorted(
        set(cast(list[int], normalized["registration"]))
        & set(cast(list[int], normalized["measurement"]))
    )
    if overlap:
        raise RuntimeError(
            "同一条 CAD 线段不能同时作为配准路径和测量路径："
            + ", ".join(map(str, overlap))
        )

    if entities is not None:
        topology = build_cad_entity_topology(entities)
        for path in [*registration_paths, *measurement_paths]:
            for left, right in zip(path, path[1:]):
                if int(right) not in topology.get(int(left), set()):
                    raise RuntimeError(
                        "CAD 路径中存在不连续线段，请重新点击起始线段和结束线段"
                    )
    return normalized


def cad_entity_from_plotly_event(
    event: object,
    available_indices: set[int],
    trace_entity_indices: list[int],
) -> tuple[int | None, str | None]:
    if not isinstance(event, Mapping):
        return None, None
    event_selection = event.get("selection", {})
    if not isinstance(event_selection, Mapping):
        return None, None
    points = event_selection.get("points", [])
    if not isinstance(points, list) or not points:
        return None, None
    point = points[-1]
    if not isinstance(point, Mapping):
        return None, None

    customdata = point.get("customdata")
    if isinstance(customdata, (list, tuple)) and customdata:
        customdata = customdata[0]
    try:
        entity_index = int(customdata)
    except (TypeError, ValueError):
        entity_index = -1

    curve_number = point.get("curve_number")
    try:
        curve_number = int(curve_number)
    except (TypeError, ValueError):
        curve_number = -1
    if entity_index not in available_indices and 0 <= curve_number < len(trace_entity_indices):
        entity_index = trace_entity_indices[curve_number]
    if entity_index not in available_indices:
        return None, None

    point_number = point.get("point_number", point.get("point_index", -1))
    return entity_index, f"{entity_index}:{curve_number}:{point_number}"


def build_cad_selection_figure(
    entities: list[CadEntity],
    selection: object = None,
    pending_mode: str | None = None,
    pending_index: int | None = None,
) -> go.Figure:
    path_selection = normalize_cad_path_selection(selection)
    registration_entities = set(cast(list[int], path_selection["registration"]))
    measurement_groups = {
        entity_index: group_index
        for group_index, path in enumerate(
            cast(list[list[int]], path_selection["measurement_paths"])
        )
        for entity_index in path
    }
    figure = go.Figure()
    for entity in entities:
        if pending_index == entity.index and pending_mode in {"registration", "measurement"}:
            color, width, opacity, role = "#f59e0b", 5.0, 1.0, "当前起始线段"
        elif entity.index in registration_entities:
            color, width, opacity, role = "#075cff", 4.0, 1.0, "CAD 配准基准线"
        elif entity.index in measurement_groups:
            group_index = measurement_groups[entity.index]
            color = MEASUREMENT_PATH_COLORS[group_index % len(MEASUREMENT_PATH_COLORS)]
            width, opacity, role = 3.5, 1.0, f"测量 {group_index + 1} · 偏差与角度测量线"
        else:
            color, width, opacity, role = "#64748b", 1.25, 0.72, "未选择"
        figure.add_trace(
            go.Scatter(
                x=entity.points[:, 0],
                y=entity.points[:, 1],
                mode="lines+markers",
                line={"color": color, "width": width},
                marker={"color": color, "opacity": 0.05, "size": 10},
                opacity=opacity,
                customdata=np.full(len(entity.points), entity.index, dtype=np.int64),
                hovertemplate=(
                    f"实体 {entity.index}<br>{entity.entity_type}<br>"
                    f"角色：{role}<br>"
                    f"采样点：{len(entity.points)}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_layout(
        height=620,
        margin={"l": 12, "r": 12, "t": 18, "b": 12},
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        dragmode="pan",
        hovermode="closest",
        uirevision="cad-entity-selection",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#334155", zeroline=False, color="#dbeafe", title="CAD X")
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#334155",
        zeroline=False,
        color="#dbeafe",
        title="CAD Y",
        scaleanchor="x",
        scaleratio=1,
    )
    return figure

