import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ezdxf
from matplotlib.axes import Axes
import numpy as np
import open3d as o3d

from cad_analysis_panel import (
    CadEntity,
    build_cad_selection_figure,
    cad_contour_path_indices,
    cad_entity_from_plotly_event,
    load_cad_entities,
    select_cad_path_endpoint,
    select_cad_single_entity_path,
    select_cad_entity,
    validate_cad_selection,
)
from media_analysis_core import (
    _blue_fit_anchor,
    analyze,
    cad_registration_frame,
    evaluate_measurement_quality,
    fit_polyline_angle,
    line_angle,
    measure_line,
    save_height_colored_cad_overlay_image,
    upright_display_rotation_degrees,
)


class CadAnalysisPanelTests(unittest.TestCase):
    def test_clicking_path_endpoints_fills_registration_and_measurement_paths(self) -> None:
        entities = [
            CadEntity(0, "LINE", np.asarray([[0.0, 0.0], [1.0, 0.0]])),
            CadEntity(1, "LINE", np.asarray([[1.0, 0.0], [2.0, 0.0]])),
            CadEntity(2, "LINE", np.asarray([[2.0, 0.0], [3.0, 0.0]])),
            CadEntity(3, "LINE", np.asarray([[3.0, 0.0], [4.0, 0.0]])),
            CadEntity(4, "LINE", np.asarray([[4.0, 0.0], [5.0, 0.0]])),
            CadEntity(5, "LINE", np.asarray([[5.0, 0.0], [6.0, 0.0]])),
        ]

        selection, start, _ = select_cad_path_endpoint(
            {}, "registration", 0, entities
        )
        selection, start, _ = select_cad_path_endpoint(
            selection, "registration", 2, entities, start
        )
        self.assertIsNone(start)
        self.assertEqual(selection["registration_paths"], [[0, 1, 2]])

        selection, start, _ = select_cad_path_endpoint(
            selection, "measurement", 3, entities
        )
        selection, start, _ = select_cad_path_endpoint(
            selection, "measurement", 5, entities, start
        )
        self.assertIsNone(start)
        self.assertEqual(selection["measurement_paths"], [[3, 4, 5]])

    def test_single_entity_path_can_be_committed_without_relaxing_overlap_rules(self) -> None:
        entities = [
            CadEntity(0, "LINE", np.asarray([[0.0, 0.0], [1.0, 0.0]])),
            CadEntity(1, "LINE", np.asarray([[1.0, 0.0], [2.0, 0.0]])),
        ]

        selection, start, _ = select_cad_path_endpoint(
            {}, "registration", 0, entities
        )
        selection, committed, _ = select_cad_single_entity_path(
            selection, "registration", start
        )
        self.assertTrue(committed)
        self.assertEqual(selection["registration_paths"], [[0]])

        selection, start, _ = select_cad_path_endpoint(
            selection, "measurement", 1, entities
        )
        selection, committed, _ = select_cad_single_entity_path(
            selection, "measurement", start
        )
        self.assertTrue(committed)
        self.assertEqual(selection["measurement_paths"], [[1]])

        _, committed, message = select_cad_single_entity_path(
            selection, "measurement", 0
        )
        self.assertFalse(committed)
        self.assertIn("不能包含", message)

    def test_registration_anchor_uses_only_the_selected_entity(self) -> None:
        sampled_entities = [
            np.asarray([[100.0, 100.0], [101.0, 100.0]]),
            np.asarray([[100.0, 101.0], [100.0, 102.0]]),
            np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        ]

        anchor = _blue_fit_anchor(sampled_entities, [2])

        np.testing.assert_array_equal(anchor, sampled_entities[2])
        self.assertFalse(np.any(np.isin(anchor, sampled_entities[0])))
        self.assertFalse(np.any(np.isin(anchor, sampled_entities[1])))

    def test_catalog_selection_and_figure_use_cad_entity_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dxf_path = Path(temporary_directory) / "drawing.dxf"
            document = ezdxf.new()
            modelspace = document.modelspace()
            modelspace.add_line((0, 0), (10, 0))
            modelspace.add_line((0, 0), (0, 10))
            modelspace.add_line((0, 10), (10, 10))
            document.saveas(dxf_path)

            entities = load_cad_entities(dxf_path)
            self.assertEqual([entity.index for entity in entities], [0, 1, 2])
            selection = validate_cad_selection({0, 1, 2}, 0, [1, 2])
            self.assertEqual(selection, {"registration": [0], "measurement": [1, 2]})
            figure = build_cad_selection_figure(entities, selection)
            self.assertEqual(len(figure.data), 3)
            self.assertEqual(figure.data[0].line.color, "#075cff")
            self.assertEqual(figure.data[1].line.color, "#00a651")
            self.assertNotEqual(figure.data[2].line.color, figure.data[1].line.color)
            self.assertEqual(figure.data[0].mode, "lines+markers")

    def test_real_dxf_inner_registration_path_uses_the_short_continuous_branch(self) -> None:
        dxf_path = Path(__file__).resolve().parents[1] / "test母头.dxf"
        entities = load_cad_entities(dxf_path)

        path = cad_contour_path_indices(entities, 16, 13)

        self.assertEqual(
            path,
            [16, 18, 20, 22, 24, 26, 27, 25, 23, 21, 19, 17, 15, 13],
        )

    def test_non_line_dxf_entities_do_not_shift_selected_entity_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dxf_path = root / "drawing.dxf"
            section_directory = root / "section"
            section_directory.mkdir()
            pointcloud_path = section_directory / "vertical_section.ply"
            output_directory = root / "cad_output"
            (root / "processing_report.json").write_text(
                json.dumps({"vertical_section": {"coordinate_system": "26xV01Q_section_local"}}),
                encoding="utf-8",
            )

            document = ezdxf.new()
            modelspace = document.modelspace()
            modelspace.add_line((0, 0), (2, 0))
            modelspace.add_point((50, 50))
            modelspace.add_line((0, 1), (2, 1))
            document.saveas(dxf_path)

            entities = load_cad_entities(dxf_path)
            self.assertEqual([entity.index for entity in entities], [0, 2])

            stations = np.linspace(0.0, 2.0, 201)
            registration_points = np.column_stack(
                (stations, np.zeros_like(stations), np.zeros_like(stations))
            )
            measurement_points = np.column_stack(
                (stations, np.zeros_like(stations), np.full_like(stations, 1.05))
            )
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(
                np.vstack((registration_points, measurement_points))
            )
            self.assertTrue(o3d.io.write_point_cloud(str(pointcloud_path), pointcloud))

            def identity_fit(profile, cad_outline, selected_anchor):
                return (
                    np.zeros(2),
                    1.0,
                    np.zeros(3),
                    np.zeros(len(selected_anchor)),
                    {
                        "method": "test_identity",
                        "global_score": 0.0,
                        "global_cad_to_pointcloud_mean": 0.0,
                        "global_cad_to_pointcloud_p95": 0.0,
                        "global_pointcloud_to_cad_mean": 0.0,
                        "global_pointcloud_to_cad_p95": 0.0,
                    },
                )

            with patch("media_analysis_core.fit_profile_global", side_effect=identity_fit):
                report = analyze(
                    pointcloud_path,
                    dxf_path,
                    output_directory,
                    step=0.1,
                    registration_entities=[0],
                    measurement_entities=[2],
                )

            self.assertEqual(report["fit"]["registration_entity_index"], 0)
            self.assertEqual(report["measurements"]["measurement_entity_2"]["entity_index"], 2)
            self.assertAlmostEqual(report["measurements"]["measurement_entity_2"]["mean_signed"], .05)

    def test_selection_rejects_overlap_between_registration_and_measurement(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "不能同时"):
            validate_cad_selection({0, 1, 2}, 0, [0, 2])

    def test_click_selection_replaces_registration_and_toggles_measurement_lines(self) -> None:
        selection = select_cad_entity({}, "registration", 0)
        selection = select_cad_entity(selection, "measurement", 1)
        selection = select_cad_entity(selection, "measurement", 2)
        self.assertEqual(selection, {"registration": [0], "measurement": [1, 2]})

        selection = select_cad_entity(selection, "measurement", 1)
        self.assertEqual(selection, {"registration": [0], "measurement": [2]})

        selection = select_cad_entity(selection, "registration", 2)
        self.assertEqual(selection, {"registration": [2], "measurement": []})

        selection = select_cad_entity(selection, "measurement", 2)
        self.assertEqual(selection, {"registration": [], "measurement": [2]})

    def test_selection_figure_shows_registration_and_measurement_roles(self) -> None:
        entities = [
            CadEntity(0, "LINE", np.asarray([[0.0, 0.0], [1.0, 0.0]])),
            CadEntity(1, "LINE", np.asarray([[1.0, 0.0], [2.0, 0.0]])),
        ]

        figure = build_cad_selection_figure(
            entities,
            {"registration": [0], "measurement": [1]},
        )

        self.assertEqual(figure.data[0].line.color, "#075cff")
        self.assertEqual(figure.data[1].line.color, "#00a651")
        self.assertEqual(figure.data[1].line.width, 3.5)

    def test_measurement_path_groups_have_distinct_consistent_colors(self) -> None:
        entities = [
            CadEntity(index, "LINE", np.asarray([[index, 0.0], [index + 1, 0.0]]))
            for index in range(8)
        ]
        selection = {
            "registration_paths": [[0]],
            "measurement_paths": [[1, 2], [3, 4], [5, 6]],
        }
        figure = build_cad_selection_figure(entities, selection)
        colors = [trace.line.color for trace in figure.data]
        self.assertEqual(len({colors[1], colors[3], colors[5]}), 3)
        for group, indices in enumerate(selection["measurement_paths"], start=1):
            self.assertEqual(colors[indices[0]], colors[indices[1]])
            for index in indices:
                self.assertEqual(figure.data[index].marker.color, colors[index])
                self.assertIn(f"测量 {group}", figure.data[index].hovertemplate)
                self.assertTrue(np.all(figure.data[index].customdata == index))
        self.assertEqual(colors[0], "#075cff")
        self.assertEqual(colors[7], "#64748b")
        pending = build_cad_selection_figure(entities, selection, "measurement", 3)
        self.assertEqual(pending.data[3].line.color, "#f59e0b")
        restored = build_cad_selection_figure(entities, selection)
        self.assertEqual([trace.line.color for trace in restored.data], colors)
        appended = build_cad_selection_figure(
            entities, {**selection, "measurement_paths": [*selection["measurement_paths"], [7]]}
        )
        self.assertEqual([trace.line.color for trace in appended.data[:7]], colors[:7])
        self.assertNotIn(appended.data[7].line.color, {colors[0], colors[1], colors[3], colors[5]})

    def test_measure_line_uses_each_cad_line_normal_for_diagonal_geometry(self) -> None:
        tangent = np.asarray([1.0, 1.0], dtype=float) / np.sqrt(2.0)
        normal = np.asarray([-tangent[1], tangent[0]])
        start = np.asarray([0.0, 0.0])
        end = 2.0 * tangent
        stations = np.linspace(0.01, 1.99, 401)
        pointcloud = start + stations[:, None] * tangent + 0.04 * normal

        rows, statistics = measure_line(pointcloud, start, end, step=0.1)
        angle_difference, _, angle_count = line_angle(pointcloud, start, end)

        self.assertTrue(all(row["valid"] for row in rows))
        self.assertAlmostEqual(statistics["mean_signed"], 0.04, places=6)
        self.assertEqual(statistics["measurement_axis"], "CAD 左法线")
        self.assertAlmostEqual(angle_difference, 0.0, places=6)
        self.assertGreaterEqual(angle_count, 20)
        self.assertAlmostEqual(statistics["mean_signed_x"], -0.02 * np.sqrt(2.0), places=6)
        self.assertAlmostEqual(statistics["mean_signed_y"], 0.02 * np.sqrt(2.0), places=6)

    def test_registration_frame_reports_when_a_single_line_cannot_fix_all_pose_degrees(self) -> None:
        identity = (np.zeros(2), 1.0, 0.0, np.zeros(2))
        straight = cad_registration_frame(
            identity,
            np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        )
        corner = cad_registration_frame(
            identity,
            np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]]),
        )

        self.assertFalse(straight["fully_constrained"])
        self.assertIsNotNone(straight["warning"])
        self.assertTrue(corner["fully_constrained"])
        self.assertEqual(corner["id"], "cad_original_xy")

    def test_sparse_measurement_stations_expand_only_along_the_selected_line(self) -> None:
        stations = np.arange(0.0, 2.0, 0.04)
        pointcloud = np.column_stack((stations, np.full_like(stations, 0.03)))

        rows, statistics = measure_line(
            pointcloud,
            np.asarray([0.0, 0.0]),
            np.asarray([2.0, 0.0]),
            step=0.05,
        )

        self.assertGreater(statistics["coverage_ratio"], 0.90)
        self.assertGreater(statistics["adaptive_station_count"], 0)
        self.assertLessEqual(statistics["maximum_longitudinal_half_width"], 0.075 + 1e-12)
        self.assertTrue(all(row["search_half_width"] == 0.15 for row in rows))

    def test_sampled_nineteen_segment_path_uses_bounded_local_angle_windows(self) -> None:
        waypoint_x = np.linspace(0.0, 4.0, 20)
        waypoints = np.column_stack((waypoint_x, 0.15 * np.sin(waypoint_x)))
        parts = []
        for start, end in zip(waypoints, waypoints[1:]):
            count = max(3, int(np.linalg.norm(end - start) / 0.025) + 1)
            part = np.linspace(start, end, count)
            parts.append(part if not parts else part[1:])
        cad_path = np.vstack(parts)
        expected_angle = 1.25
        angle = np.radians(expected_angle)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        sparse_pointcloud = cad_path[::2] @ rotation.T + np.asarray([0.005, -0.01])

        result = fit_polyline_angle(sparse_pointcloud, cad_path)

        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(result["angle_difference_deg"], expected_angle, delta=0.35)
        self.assertGreaterEqual(result["point_count"], 8)
        self.assertTrue(result["segments"])
        self.assertTrue(
            all(segment["length"] <= 0.60 + 1e-12 for segment in result["segments"])
        )
        self.assertTrue(
            all(segment["nearest_path_p95"] <= 0.15 for segment in result["segments"])
        )

    def test_polyline_angle_does_not_pull_points_from_outside_its_corridor(self) -> None:
        cad_path = np.column_stack((np.linspace(0.0, 2.0, 101), np.zeros(101)))
        nearby_points = cad_path[:7]
        unrelated_points = np.column_stack(
            (np.linspace(0.0, 2.0, 401), np.full(401, 0.31))
        )

        result = fit_polyline_angle(
            np.vstack((nearby_points, unrelated_points)),
            cad_path,
        )

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["candidate_point_count"], 7)
        self.assertEqual(result["point_count"], 0)
        self.assertIsNone(result["angle_difference_deg"])

    def test_measurement_quality_gate_blocks_formal_pass_when_stitching_is_not_ready(self) -> None:
        result = evaluate_measurement_quality(
            {
                "selected_cad_to_pointcloud_mean": 0.01,
                "selected_cad_to_pointcloud_p95": 0.02,
            },
            {
                "line": {
                    "valid_stations": 20,
                    "total_stations": 20,
                    "coverage_ratio": 1.0,
                    "rms": 0.01,
                    "p95_absolute": 0.02,
                    "signed_angle_difference_deg": 0.1,
                    "angle_fit_point_count": 20,
                }
            },
            {
                "status": "INSUFFICIENT_DATA",
                "section_points": 1000,
                "insufficient_data_reasons": ["截面覆盖不足"],
            },
            section_points=1000,
        )

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertFalse(result["formal_measurement_allowed"])
        self.assertIn("截面覆盖不足", result["insufficient_data_reasons"])

    def test_measurement_quality_gate_passes_when_all_evidence_is_ready(self) -> None:
        result = evaluate_measurement_quality(
            {
                "selected_cad_to_pointcloud_mean": 0.01,
                "selected_cad_to_pointcloud_p95": 0.02,
            },
            {
                "line": {
                    "valid_stations": 20,
                    "total_stations": 20,
                    "coverage_ratio": 1.0,
                    "rms": 0.01,
                    "p95_absolute": 0.02,
                    "signed_angle_difference_deg": 0.1,
                    "angle_fit_point_count": 20,
                }
            },
            {"status": "PASS", "section_points": 1000},
            section_points=1000,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["formal_measurement_allowed"])

    def test_measurement_quality_requires_angle_evidence(self) -> None:
        result = evaluate_measurement_quality(
            {
                "selected_cad_to_pointcloud_mean": 0.01,
                "selected_cad_to_pointcloud_p95": 0.02,
            },
            {
                "line": {
                    "valid_stations": 20,
                    "total_stations": 20,
                    "coverage_ratio": 1.0,
                    "rms": 0.01,
                    "p95_absolute": 0.02,
                    "signed_angle_difference_deg": None,
                    "angle_fit_point_count": 0,
                }
            },
            {"status": "PASS", "section_points": 1000},
            section_points=1000,
        )

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIn("角度拟合有效点不足", result["insufficient_data_reasons"][0])

    def test_analyze_outputs_every_selected_measurement_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            section_directory = root / "section_package"
            section_directory.mkdir()
            pointcloud_path = section_directory / "vertical_section.ply"
            dxf_path = root / "drawing.dxf"
            output_directory = root / "cad_output"
            (root / "processing_report.json").write_text(
                json.dumps({"vertical_section": {"coordinate_system": "26xV01Q_section_local"}}),
                encoding="utf-8",
            )

            document = ezdxf.new()
            modelspace = document.modelspace()
            modelspace.add_line((0, 0), (2, 0))
            modelspace.add_line((0, 1), (2, 1))
            modelspace.add_line((2, 0), (2, 2))
            document.saveas(dxf_path)

            stations = np.linspace(0.0, 2.0, 401)
            registration_points = np.column_stack((stations, np.zeros_like(stations), np.zeros_like(stations)))
            horizontal_measurement_points = np.column_stack(
                (stations, np.zeros_like(stations), np.full_like(stations, 1.05))
            )
            vertical_measurement_points = np.column_stack(
                (np.full_like(stations, 1.97), np.zeros_like(stations), stations)
            )
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(
                np.vstack((registration_points, horizontal_measurement_points, vertical_measurement_points))
            )
            self.assertTrue(o3d.io.write_point_cloud(str(pointcloud_path), pointcloud))

            def identity_fit(profile, cad_outline, selected_anchor):
                return (
                    np.zeros(2),
                    1.0,
                    np.zeros(3),
                    np.zeros(len(selected_anchor)),
                    {
                        "method": "test_identity",
                        "global_score": 0.0,
                        "global_cad_to_pointcloud_mean": 0.0,
                        "global_cad_to_pointcloud_p95": 0.0,
                        "global_pointcloud_to_cad_mean": 0.0,
                        "global_pointcloud_to_cad_p95": 0.0,
                    },
                )

            legend_labels: list[list[str]] = []
            original_legend = Axes.legend

            def capture_legend(axis, *args, **kwargs):
                legend = original_legend(axis, *args, **kwargs)
                legend_labels.append([text.get_text() for text in legend.get_texts()])
                return legend

            with patch.object(Axes, "legend", new=capture_legend), patch(
                "media_analysis_core.fit_profile_global", side_effect=identity_fit
            ):
                report = analyze(
                    pointcloud_path,
                    dxf_path,
                    output_directory,
                    step=0.1,
                    registration_entities=[0],
                    measurement_entities=[1, 2],
                )

            self.assertEqual(report["fit"]["registration_entity_index"], 0)
            self.assertEqual(
                list(report["measurements"]),
                ["measurement_entity_1", "measurement_entity_2"],
            )
            self.assertEqual(
                [entry["entity_index"] for entry in report["outputs"]["measurement_csvs"]],
                [1, 2],
            )
            self.assertAlmostEqual(
                report["measurements"]["measurement_entity_1"]["mean_signed"],
                0.05,
                places=6,
            )
            self.assertAlmostEqual(
                report["measurements"]["measurement_entity_2"]["mean_signed"],
                0.03,
                places=6,
            )
            self.assertTrue(Path(report["outputs"]["measurement_angle_csv"]).is_file())
            self.assertTrue(Path(report["outputs"]["curve_image"]).is_file())
            cad_legends = [
                labels
                for labels in legend_labels
                if "CAD 配准基准线" in labels and "CAD 测量线" in labels
            ]
            self.assertGreaterEqual(len(cad_legends), 2)
            for labels in cad_legends:
                self.assertEqual(labels.count("CAD 配准基准线"), 1)
                self.assertEqual(labels.count("CAD 测量线"), 1)

    def test_nineteen_segment_sparse_path_completes_as_insufficient_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            section_directory = root / "section_package"
            section_directory.mkdir()
            pointcloud_path = section_directory / "vertical_section.ply"
            dxf_path = root / "drawing.dxf"
            output_directory = root / "cad_output"
            (root / "processing_report.json").write_text(
                json.dumps(
                    {"vertical_section": {"coordinate_system": "26xV01Q_section_local"}}
                ),
                encoding="utf-8",
            )

            document = ezdxf.new()
            modelspace = document.modelspace()
            modelspace.add_line((0.0, -1.0), (1.0, -1.0))
            modelspace.add_line((1.0, -1.0), (2.0, -1.0))
            measurement_indices = list(range(2, 21))
            for index in range(19):
                modelspace.add_line((index * 0.1, 0.0), ((index + 1) * 0.1, 0.0))
            document.saveas(dxf_path)

            registration_stations = np.linspace(0.0, 2.0, 101)
            registration_points = np.column_stack(
                (
                    registration_stations,
                    np.zeros_like(registration_stations),
                    np.full_like(registration_stations, -1.0),
                )
            )
            sparse_measurement_points = np.asarray(
                [[0.02, 0.0, 0.03], [0.08, 0.0, 0.03]],
                dtype=float,
            )
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(
                np.vstack((registration_points, sparse_measurement_points))
            )
            self.assertTrue(o3d.io.write_point_cloud(str(pointcloud_path), pointcloud))
            source_bytes = pointcloud_path.read_bytes()

            def identity_fit(profile, cad_outline, selected_anchor):
                return (
                    np.zeros(2),
                    1.0,
                    np.zeros(3),
                    np.zeros(len(selected_anchor)),
                    {
                        "method": "test_identity",
                        "global_score": 0.0,
                        "global_cad_to_pointcloud_mean": 0.0,
                        "global_cad_to_pointcloud_p95": 0.0,
                        "global_pointcloud_to_cad_mean": 0.0,
                        "global_pointcloud_to_cad_p95": 0.0,
                    },
                )

            with patch("media_analysis_core.fit_profile_global", side_effect=identity_fit):
                report = analyze(
                    pointcloud_path,
                    dxf_path,
                    output_directory,
                    step=0.05,
                    registration_paths=[[0, 1]],
                    measurement_paths=[measurement_indices],
                )

            measurement = report["measurements"]["measurement_path_1"]
            self.assertEqual(measurement["entity_indices"], measurement_indices)
            self.assertEqual(measurement["angle_fit_status"], "INSUFFICIENT_DATA")
            self.assertIsNone(measurement["signed_angle_difference_deg"])
            self.assertEqual(measurement["angle_fit_candidate_point_count"], 2)
            self.assertEqual(report["measurement_quality"]["status"], "INSUFFICIENT_DATA")
            self.assertFalse(report["measurement_quality"]["formal_measurement_allowed"])
            self.assertTrue((output_directory / "report.json").is_file())
            self.assertEqual(pointcloud_path.read_bytes(), source_bytes)

    def test_plotly_click_event_returns_the_clicked_entity(self) -> None:
        entity_index, signature = cad_entity_from_plotly_event(
            {"selection": {"points": [{"customdata": [2], "curve_number": 2, "point_number": 5}]}},
            {0, 1, 2},
            [0, 1, 2],
        )
        self.assertEqual(entity_index, 2)
        self.assertEqual(signature, "2:2:5")

        fallback_index, fallback_signature = cad_entity_from_plotly_event(
            {"selection": {"points": [{"curve_number": 1, "point_index": 7}]}},
            {0, 1, 2},
            [0, 1, 2],
        )
        self.assertEqual(fallback_index, 1)
        self.assertEqual(fallback_signature, "1:1:7")

    def test_height_colored_cad_overlay_image_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "height_colored_overlay.png"
            document = ezdxf.new()
            modelspace = document.modelspace()
            modelspace.add_line((0, 0), (10, 0))
            modelspace.add_arc((10, 5), radius=5, start_angle=-90, end_angle=90)
            modelspace.add_line((10, 10), (0, 10))
            entities = list(modelspace)
            pointcloud_cad = np.asarray(
                [
                    [0.0, 0.1],
                    [2.5, -0.1],
                    [5.0, 0.05],
                    [7.5, 0.0],
                    [10.0, 0.2],
                    [14.8, 5.0],
                    [10.0, 9.8],
                    [5.0, 10.1],
                    [0.0, 9.9],
                ],
                dtype=float,
            )
            heights = np.linspace(-3.0, 3.0, len(pointcloud_cad))
            original_pointcloud_cad = pointcloud_cad.copy()
            original_heights = heights.copy()

            result_path = save_height_colored_cad_overlay_image(
                image_path,
                pointcloud_cad,
                heights,
                entities,
                fit_indices=[1],
            )

            self.assertEqual(result_path, image_path)
            self.assertTrue(image_path.is_file())
            self.assertGreater(image_path.stat().st_size, 1000)
            np.testing.assert_array_equal(pointcloud_cad, original_pointcloud_cad)
            np.testing.assert_array_equal(heights, original_heights)

    def test_upright_display_rotation_makes_reference_edge_horizontal(self) -> None:
        reference_edge = np.asarray([[0.0, 0.0], [2.0, -2.0]], dtype=float)

        rotation_degrees = upright_display_rotation_degrees(reference_edge)
        rotated_angle = np.degrees(
            np.arctan2(
                reference_edge[-1, 1] - reference_edge[0, 1],
                reference_edge[-1, 0] - reference_edge[0, 0],
            )
        ) + rotation_degrees

        self.assertAlmostEqual(rotation_degrees, 45.0)
        self.assertAlmostEqual(rotated_angle, 0.0)


if __name__ == "__main__":
    unittest.main()
