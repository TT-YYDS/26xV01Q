import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import open3d as o3d

from pointcloud_app.downstream_bridge import (
    EXTERNAL_SECTION_MODE,
    _derive_processing_quality,
    export_final_section,
)
from media_analysis_core import section_profile_from_points, validate_section_input_contract
from pointcloud_app.pipeline import vertical_plane_basis


class DownstreamBridgeTests(unittest.TestCase):
    def test_missing_final_section_error_does_not_imply_section_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            for source_path in (None, Path(temporary_directory) / "missing.ply"):
                with self.subTest(source_path=source_path):
                    result = SimpleNamespace(section_cloud_path=source_path)
                    with self.assertRaisesRegex(RuntimeError, "没有可用的最终截面点云"):
                        export_final_section(result)

    @staticmethod
    def _ready_summary(section_offset_status=None):
        summary = {
            "input_files": ["a.ply", "b.ply", "c.ply"],
            "fused_view_count": 3,
            "registration_metrics": [
                {"file_name": "a.ply", "included_in_fusion": True},
                {"file_name": "b.ply", "included_in_fusion": True},
                {"file_name": "c.ply", "included_in_fusion": True},
            ],
            "config": {
                "registration": {
                    "section_alignment_minimum_views": 3,
                }
            },
            "section_points": 1000,
            "section_integrity": {"status": "stable_after_cleanup"},
            "section_consensus": {
                "fusion_confidence": {"confidence_score": 0.8},
            },
        }
        if section_offset_status is not None:
            summary["section_consensus"]["section_offset_diagnostic"] = {
                "status": section_offset_status,
                "recommended_offset": -10.0,
            }
        return summary

    def test_neighbor_only_multi_section_diagnostic_blocks_formal_measurement(self) -> None:
        summary = self._ready_summary()
        summary["multi_section_diagnostic"] = {
            "diagnostic_status": "neighbor_only_stable",
            "probe_count": 5,
        }

        quality = _derive_processing_quality(summary)

        self.assertEqual(quality["status"], "INSUFFICIENT_DATA")
        self.assertFalse(quality["formal_measurement_allowed"])
        self.assertEqual(
            quality["multi_section_diagnostic_status"], "neighbor_only_stable"
        )

    def test_coverage_first_neighbor_only_diagnostic_blocks_formal_measurement(self) -> None:
        summary = self._ready_summary()
        summary["config"]["registration"]["coverage_preserving_fusion"] = True
        summary["multi_section_diagnostic"] = {
            "diagnostic_status": "neighbor_only_stable",
            "probe_count": 5,
        }

        quality = _derive_processing_quality(summary)

        self.assertEqual(quality["status"], "INSUFFICIENT_DATA")
        self.assertFalse(quality["formal_measurement_allowed"])
        self.assertTrue(
            any("中心截面存在缺口" in reason for reason in quality["insufficient_data_reasons"])
        )

    def test_stable_multi_section_diagnostic_does_not_block_formal_measurement(self) -> None:
        summary = self._ready_summary()
        summary["multi_section_diagnostic"] = {
            "diagnostic_status": "stable_neighbor_sections",
            "probe_count": 5,
        }

        quality = _derive_processing_quality(summary)

        self.assertEqual(quality["status"], "PASS")
        self.assertTrue(quality["formal_measurement_allowed"])

    def test_source_coverage_gap_blocks_formal_measurement(self) -> None:
        quality = _derive_processing_quality(
            self._ready_summary("source_coverage_gap")
        )

        self.assertEqual(quality["status"], "INSUFFICIENT_DATA")
        self.assertFalse(quality["formal_measurement_allowed"])
        self.assertEqual(
            quality["section_offset_diagnostic_status"], "source_coverage_gap"
        )
        self.assertTrue(
            any("多工位共同支持" in reason for reason in quality["insufficient_data_reasons"])
        )

    def test_manual_review_and_unconfirmed_recommendation_block_formal_measurement(self) -> None:
        for status in ("manual_review", "auto_recommended"):
            with self.subTest(status=status):
                quality = _derive_processing_quality(self._ready_summary(status))
                self.assertEqual(quality["status"], "INSUFFICIENT_DATA")
                self.assertFalse(quality["formal_measurement_allowed"])

    def test_current_is_best_diagnostic_does_not_block_formal_measurement(self) -> None:
        quality = _derive_processing_quality(self._ready_summary("current_is_best"))

        self.assertEqual(quality["status"], "PASS")
        self.assertTrue(quality["formal_measurement_allowed"])

    def test_legacy_summary_without_offset_diagnostic_remains_compatible(self) -> None:
        quality = _derive_processing_quality(self._ready_summary())

        self.assertEqual(quality["status"], "PASS")
        self.assertIsNone(quality["section_offset_diagnostic_status"])
        self.assertTrue(quality["formal_measurement_allowed"])

    def test_export_preserves_final_section_coordinates_and_point_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "source_run"
            downstream_root = root / "26xV01"
            run_directory.mkdir()
            downstream_root.mkdir()
            (downstream_root / "media_web_app.py").write_text("# test", encoding="utf-8")

            angle_degrees = 27.0
            tilt_degrees = -8.0
            plane_offset = 4.25
            normal, horizontal, vertical = vertical_plane_basis(angle_degrees, tilt_degrees)
            expected_profile = np.array([[-2.0, -1.0], [0.5, 1.5], [3.0, 4.0]], dtype=np.float64)
            signed_distances = np.array([-0.02, 0.0, 0.03], dtype=np.float64)
            points = (
                expected_profile[:, :1] * horizontal
                + expected_profile[:, 1:] * vertical
                + (plane_offset + signed_distances)[:, None] * normal
            )
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            section_path = run_directory / "vertical_section.ply"
            self.assertTrue(o3d.io.write_point_cloud(str(section_path), cloud, write_ascii=False))
            source_bytes = section_path.read_bytes()
            section_image_path = run_directory / "vertical_section.png"
            section_image_path.write_bytes(b"test-image")
            summary_path = run_directory / "run_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "group_name": "test",
                        "input_files": ["a.ply", "b.ply"],
                        "config": {
                            "section": {
                                "angle_degrees": angle_degrees,
                                "tilt_degrees": tilt_degrees,
                                "plane_offset": plane_offset,
                                "thickness": 0.15,
                            }
                        },
                        "compute": {"active_backend": "cpu", "gpu_capabilities": {}},
                    }
                ),
                encoding="utf-8",
            )
            result = SimpleNamespace(
                group_name="test",
                run_directory=run_directory,
                section_cloud_path=section_path,
                section_image_path=section_image_path,
                summary_path=summary_path,
                compute_backend="cpu",
                gpu_device_name=None,
            )

            handoff = export_final_section(
                result,
                downstream_root=downstream_root,
                downstream_url="http://127.0.0.1:5000",
            )

            exported_points = np.asarray(o3d.io.read_point_cloud(str(handoff.section_cloud_path)).points)
            np.testing.assert_allclose(exported_points, points, atol=1e-7)
            self.assertEqual(handoff.section_cloud_path.read_bytes(), source_bytes)
            manifest = json.loads((handoff.package_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], EXTERNAL_SECTION_MODE)
            self.assertEqual(manifest["source_file_count"], 2)
            report = json.loads(handoff.processing_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["vertical_section"]["coordinate_system"], "26xV01Q_final_section_xyz")
            self.assertFalse(report["vertical_section"]["pointcloud_transform_applied"])
            self.assertTrue(report["vertical_section"]["point_order_preserved"])
            self.assertTrue(report["vertical_section"]["coordinates_preserved_from_source"])
            self.assertEqual(report["vertical_section"]["section_source"], "merged_fused_cloud")
            expected_hash = hashlib.sha256(handoff.section_cloud_path.read_bytes()).hexdigest()
            self.assertEqual(expected_hash, hashlib.sha256(section_path.read_bytes()).hexdigest())
            self.assertEqual(report["vertical_section"]["analysis_section_sha256"], expected_hash)
            self.assertEqual(manifest["analysis_pointcloud_sha256"], expected_hash)
            self.assertEqual(report["vertical_section"]["section_points"], 3)
            profile = section_profile_from_points(exported_points, report)
            np.testing.assert_allclose(profile, expected_profile, atol=1e-7)

    def test_external_section_contract_rejects_a_changed_cad_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            section_path = Path(temporary_directory) / "vertical_section_points.ply"
            section_path.write_bytes(b"original")
            expected_hash = hashlib.sha256(section_path.read_bytes()).hexdigest()
            section_path.write_bytes(b"changed")

            with self.assertRaises(RuntimeError):
                validate_section_input_contract(
                    section_path,
                    {
                        "processing": {"mode": EXTERNAL_SECTION_MODE},
                        "vertical_section": {
                            "section_source": "merged_fused_cloud",
                            "analysis_section_sha256": expected_hash,
                        },
                    },
                )

    def test_final_section_contract_rejects_a_transformed_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            section_path = Path(temporary_directory) / "vertical_section_points.ply"
            section_path.write_bytes(b"unchanged")
            expected_hash = hashlib.sha256(section_path.read_bytes()).hexdigest()

            with self.assertRaisesRegex(RuntimeError, "原始最终点云"):
                validate_section_input_contract(
                    section_path,
                    {
                        "processing": {"mode": EXTERNAL_SECTION_MODE},
                        "vertical_section": {
                            "coordinate_system": "26xV01Q_final_section_xyz",
                            "section_source": "merged_fused_cloud",
                            "analysis_section_sha256": expected_hash,
                            "pointcloud_transform_applied": True,
                            "coordinates_preserved_from_source": False,
                            "point_order_preserved": False,
                        },
                    },
                )


if __name__ == "__main__":
    unittest.main()
