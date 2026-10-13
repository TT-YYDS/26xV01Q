"""Compact, white-background figures for the single desktop canvas."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..cad_geometry import build_cad_selection_figure
from ..pipeline import select_vertical_section_indices, vertical_plane_basis

COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc3377", "#8c6bb1", "#b69500", "#222222", "#56b4e9"]
AXIS = dict(showbackground=True, backgroundcolor="white", gridcolor="#e2e6e9",
            zerolinecolor="#bbc3c9", color="#48515a", showspikes=False)


def light(figure, key="view"):
    figure.update_layout(
        template="plotly_white", autosize=True, height=None, width=None,
        margin=dict(l=45, r=24, t=24, b=42), paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Microsoft YaHei, sans-serif", size=12, color="#29343a"),
        legend=dict(orientation="h", y=1.02, x=0, itemsizing="constant",
                    font=dict(size=13), groupclick="togglegroup"),
        uirevision=key,
    )
    figure.update_xaxes(gridcolor="#e2e6e9", zeroline=False, color="#48515a")
    figure.update_yaxes(gridcolor="#e2e6e9", zeroline=False, color="#48515a")
    figure.update_scenes(xaxis={**AXIS, "title": "X"}, yaxis={**AXIS, "title": "Y"},
                         zaxis={**AXIS, "title": "Z"}, aspectmode="data",
                         bgcolor="white", camera=dict(projection=dict(type="orthographic"),
                                                      eye=dict(x=1.65, y=1.65, z=1.25)))
    return figure


def cloud(points, section=None, projection="3D", point_size=1.5, key="cloud", surface=None):
    points = np.asarray(points, dtype=np.float32)
    figure = go.Figure()
    figure.update_layout(selections=[])
    if surface is not None:
        v, t = surface.vertices, surface.triangles
        figure.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                  i=t[:, 0], j=t[:, 1], k=t[:, 2],
                                  intensity=v[:, 2], colorscale="Turbo",
                                  showscale=False, name="表面预览", hoverinfo="skip"))
    elif projection == "3D":
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker=dict(size=point_size, color=points[:, 2], colorscale="Turbo", opacity=1),
            hoverinfo="skip", name="点云", showlegend=False))
    else:
        first, second = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[projection]
        figure.add_trace(go.Scattergl(
            x=points[:, first], y=points[:, second], mode="markers",
            marker=dict(size=point_size, color=points[:, 2], colorscale="Turbo"),
            hoverinfo="skip", name="点云", showlegend=False))
        figure.update_xaxes(title="XYZ"[first])
        figure.update_yaxes(title="XYZ"[second], scaleanchor="x", scaleratio=1)
    if section is not None and len(points):
        n, h, v = vertical_plane_basis(section.angle_degrees, section.tilt_degrees)
        if projection == "3D":
            corners = np.array([
                n * section.plane_offset + h * x + v * y
                for x, y in ((section.horizontal_min, section.vertical_min),
                             (section.horizontal_max, section.vertical_min),
                             (section.horizontal_max, section.vertical_max),
                             (section.horizontal_min, section.vertical_max))
            ])
            figure.add_trace(go.Mesh3d(
                x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                color="#d33b51", opacity=.22, name="法向截面", hoverinfo="skip", showlegend=True))
        if projection == "XY" and section.target_region is not None:
            r = section.target_region
            figure.update_layout(selections=[dict(
                type="rect", x0=r.x_min, x1=r.x_max, y0=r.y_min, y1=r.y_max,
                line=dict(color="#d33b51", width=2),
            )])
            figure.update_layout(activeselection=dict(fillcolor="rgba(211,59,81,.06)"))
    return light(figure, key)


def section_cloud(points, section, point_size=4, key="section", frozen=False):
    points = np.asarray(points)
    if not frozen:
        points = points[select_vertical_section_indices(points, section)]
    return cloud(points, projection="3D", point_size=point_size, key=key)


def alignment(items, mode="四视图", point_size=1.5, key="alignment"):
    if mode != "四视图":
        figure = go.Figure()
    else:
        figure = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {"type": "scene"}]],
                               subplot_titles=("XY", "XZ", "YZ", "3D"),
                               vertical_spacing=.13, horizontal_spacing=.08)
    for index, (label, points, included) in enumerate(items):
        points = np.asarray(points, dtype=np.float32)
        color = COLORS[index % len(COLORS)]
        common = dict(name=label, legendgroup=str(index), visible=True if included else "legendonly",
                      hoverinfo="skip", mode="markers", marker=dict(size=point_size, color=color))
        if mode == "四视图":
            for (a, b), (row, col) in zip(((0, 1), (0, 2), (1, 2)), ((1, 1), (1, 2), (2, 1))):
                figure.add_trace(go.Scattergl(x=points[:, a], y=points[:, b],
                                              showlegend=row == col == 1, **common), row=row, col=col)
            figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                         showlegend=False, **common), row=2, col=2)
        elif mode == "3D":
            figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], **common))
        else:
            a, b = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[mode]
            figure.add_trace(go.Scattergl(x=points[:, a], y=points[:, b], **common))
    if mode == "四视图":
        for number in range(1, 4):
            figure.update_layout(**{f"yaxis{number if number > 1 else ''}":
                                     dict(scaleanchor=f"x{number if number > 1 else ''}", scaleratio=1)})
    elif mode != "3D":
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
    light(figure, key)
    figure.update_layout(margin=dict(l=42, r=24, t=112, b=48),
                         legend=dict(y=1.2, yanchor="bottom", entrywidth=130))
    if mode == "四视图":
        figure.update_scenes(camera=dict(eye=dict(x=2, y=2, z=1.5)),
                             xaxis=dict(nticks=4), yaxis=dict(nticks=4), zaxis=dict(nticks=4))
    return figure


def cad(entities, selection, mode, pending, key="cad"):
    return light(build_cad_selection_figure(entities, selection, mode, pending), key)


def cad_overlay(report, sources=None, *, measurement_lines=False):
    import ezdxf
    from media_analysis_core import inverse_transform, sample_entity, section_profile_from_points
    from ..pipeline import read_point_cloud
    import json
    from pathlib import Path

    path = Path(report["source_pointcloud"])
    processing = json.loads((path.parent.parent / "processing_report.json").read_text(encoding="utf-8"))
    points = np.asarray(read_point_cloud(path).points)
    profile = section_profile_from_points(points, processing)
    if profile is None:
        raise ValueError("结果缺少冻结截面坐标系")
    cad_transform = report["fit"]["cad_transform"]
    profile = inverse_transform(
        profile,
        np.asarray(cad_transform["center"]),
        float(cad_transform["mirror"]),
        float(cad_transform["angle_rad"]),
        np.asarray(cad_transform["translation"]),
    )
    figure = go.Figure()
    if sources is None:
        figure.add_trace(go.Scattergl(x=profile[:, 0], y=profile[:, 1], mode="markers",
                                     marker=dict(size=3 if measurement_lines else 5,
                                                 color="#84939a" if measurement_lines else "#f97316",
                                                 opacity=.25 if measurement_lines else .92),
                                     name="最终截面" if measurement_lines else "最终截面（CAD 原图坐标）",
                                     hoverinfo="skip"))
    else:
        covered = np.zeros(len(profile), dtype=bool)
        for index, (label, indices) in enumerate(sources):
            indices = np.asarray(indices, dtype=np.int64)
            if (indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(profile))
                    or len(np.unique(indices)) != len(indices) or np.any(covered[indices])):
                raise ValueError("截面来源索引无效或重复")
            covered[indices] = True
            color = "#707b86" if label in {"未追溯", "多源重合"} else COLORS[index % len(COLORS)]
            figure.add_trace(go.Scattergl(
                x=profile[indices, 0], y=profile[indices, 1], mode="markers",
                marker=dict(size=2.5 if measurement_lines else 3, color=color,
                            opacity=.4 if measurement_lines else .92),
                name=f"{label}（{len(indices):,} 点）", legendgroup=f"source-{index}",
                hovertemplate="%{fullData.name}<br>X %{x:.5f}<br>Y %{y:.5f}<extra></extra>",
            ))
        if not covered.all():
            raise ValueError("截面来源索引未覆盖全部测量点")
    for index, entity in enumerate(ezdxf.readfile(report["source_dxf"]).modelspace()):
        sample = sample_entity(entity)
        if not len(sample):
            continue
        figure.add_trace(go.Scattergl(x=sample[:, 0], y=sample[:, 1], mode="lines",
                                      line=dict(color="#cbd5e1", width=1.2),
                                      name="CAD 原图" if measurement_lines else "CAD 原图（淡化参考）",
                                      showlegend=index == 0, legendgroup="cad", hoverinfo="skip"))
    if measurement_lines:
        import csv

        valid_count = 0
        for index, (key, measurement) in enumerate(report.get("measurements", {}).items()):
            if (report.get("measurement_coordinate_system") != "cad_original_xy"
                    and measurement.get("coordinate_system") != "cad_original_xy"):
                raise ValueError("该报告的测量站位不在 CAD 原图坐标系，请重新测量")
            csv_path = Path(measurement.get("csv_path", ""))
            if not csv_path.is_file():
                raise ValueError(f"测量代表线缺少站位 CSV：{csv_path}")
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            x, y, details = [], [], []
            for row in rows:
                if str(row.get("valid", "")).lower() != "true":
                    x.append(None)
                    y.append(None)
                    details.append([None] * 4)
                    continue
                try:
                    station, cad_x, cad_y, point_x, point_y, deviation = [float(row[field]) for field in (
                        "station", "cad_x", "cad_y", "pointcloud_x", "pointcloud_y", "signed_normal_deviation")]
                except (KeyError, ValueError, TypeError) as error:
                    raise ValueError(f"测量代表线的有效站位数据无效：{csv_path}") from error
                if not np.isfinite([station, cad_x, cad_y, point_x, point_y, deviation]).all():
                    raise ValueError(f"测量代表线的有效站位包含非有限坐标：{csv_path}")
                x.append(point_x)
                y.append(point_y)
                details.append([station, deviation, point_x - cad_x, point_y - cad_y])
                valid_count += 1
            figure.add_trace(go.Scattergl(
                x=x, y=y, customdata=details, mode="lines+markers", connectgaps=False,
                line=dict(color=COLORS[index % len(COLORS)], width=3),
                marker=dict(color=COLORS[index % len(COLORS)], size=4),
                name=f"代表线 {index + 1}", legendgroup=f"measurement-{key}",
                meta=measurement.get("label", key),
                hovertemplate="%{meta}<br>站位 %{customdata[0]:.4f}<br>X %{x:.5f}<br>Y %{y:.5f}"
                              "<br>法向偏差 %{customdata[1]:+.5f}<br>X 偏差 %{customdata[2]:+.5f}"
                              "<br>Y 偏差 %{customdata[3]:+.5f}<extra></extra>",
            ))
        if not valid_count:
            figure.add_annotation(text="无有效测量站位", x=.5, y=.5, xref="paper", yref="paper",
                                  showarrow=False, font=dict(color="#68777e", size=15))
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figure.update_xaxes(title="CAD 原图 X")
    figure.update_yaxes(title="CAD 原图 Y")
    figure.update_layout(title=("来源与测量代表线" if measurement_lines and sources is not None else
                                "测量代表线 / CAD 叠加" if measurement_lines else
                                "最终截面 / CAD 来源叠加" if sources is not None
                                else "最终截面 / CAD 叠加（按 CAD 原图角度显示）"))
    light(figure, str(path))
    if sources is not None or measurement_lines:
        figure.update_layout(title=dict(x=.02, y=.99, yanchor="top", font=dict(size=15)),
                             margin=dict(l=45, r=24, t=168 if sources is not None and measurement_lines else 144, b=42),
                             legend=dict(y=1.02, yanchor="bottom", entrywidth=130))
    return figure


def measurement_curves(report):
    import csv
    from pathlib import Path
    figure = go.Figure()
    for index, (name, measurement) in enumerate(report.get("measurements", {}).items()):
        path = Path(measurement["csv_path"])
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        x, y = [], []
        for row in rows:
            try:
                station = float(row["station"])
                value = float(row.get("signed_normal_deviation", row.get("signed_axis_deviation", "")))
                if not np.isfinite(value):
                    value = None
            except (ValueError, TypeError):
                station, value = float(row.get("station", 0)), None
            x.append(station)
            y.append(value)
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", connectgaps=False,
                                   line=dict(color=COLORS[index % len(COLORS)], width=2),
                                   name=measurement.get("label", name)))
    figure.update_xaxes(title="路径距离")
    figure.update_yaxes(title="法向偏差")
    return light(figure, report["source_dxf"])
