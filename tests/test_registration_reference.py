import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import open3d as o3d

from pointcloud_app.models import ProcessingConfig
from pointcloud_app.registration_reference import (
    make_reference, match_reference, station_ids, valid_rigid, registration_verified,
)
from pointcloud_app.pipeline import _cached_registration_cloud, _refine_pose_seed


def cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


class RegistrationReferenceTests(TestCase):
    def setUp(self):
        self.config = ProcessingConfig()
        rng = np.random.default_rng(431)
        self.clouds = [cloud(rng.uniform(-1, 1, (300, 3)) + [i * 6, 0, 0])
                       for i in range(3)]
        self.transforms = [np.eye(4) for _ in self.clouds]
        for i, matrix in enumerate(self.transforms):
            matrix[0, 3] = -i * 6
        self.metrics = [SimpleNamespace(calibration_quality_passed=True,
                                       included_in_fusion=True) for _ in self.clouds]

    def reference(self, files):
        return make_reference(files, self.clouds, self.transforms, self.config, self.metrics)

    def test_unknown_names_use_geometry_not_file_order_and_rebase_reference(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"random_{i}.ply" for i in range(3)]
            reference = self.reference(files)
            order = [2, 0, 1]
            seeds, report = match_reference(
                reference, [files[i] for i in order], [self.clouds[i] for i in order],
                self.config)
            self.assertEqual(report["mapping"], "fixture_geometry")
            self.assertEqual(report["station_indices"], order)
            np.testing.assert_allclose(seeds[0], np.eye(4))
            for seed, i in zip(seeds, order):
                np.testing.assert_allclose(seed, np.linalg.inv(self.transforms[2]) @ self.transforms[i])

    def test_identical_fixture_views_are_ambiguous(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"random_{i}.ply" for i in range(3)]
            self.clouds = [self.clouds[0]] * 3
            seeds, report = match_reference(self.reference(files), files, self.clouds, self.config)
            self.assertIsNone(seeds)
            self.assertEqual(report["reason"], "ambiguous_fixture_station_mapping")

    def test_one_unmatched_view_does_not_disable_other_confident_matches(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"random_{i}.ply" for i in range(3)]
            reference = self.reference(files)
            changed = [self.clouds[0], self.clouds[1],
                       cloud(np.asarray(self.clouds[2].points) + 50)]
            seeds, report = match_reference(reference, files, changed, self.config)
            self.assertEqual(report["station_indices"], [0, 1, None])
            self.assertIsNone(seeds[2])
            self.assertIsNotNone(seeds[1])

    def test_numbered_stations_and_metadata_do_not_depend_on_mtime(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"view_{i + 1}.ply" for i in range(3)]
            reference = self.reference(files)
            seeds, report = match_reference(
                reference, files[::-1], self.clouds[::-1], self.config)
            self.assertEqual(report["mapping"], "station_id")
            self.assertEqual(report["station_indices"], [2, 1, 0])
            (Path(directory) / "station_order.json").write_text(json.dumps({
                "views": [{"file": path.name, "station": str(i)} for i, path in enumerate(files)]
            }))
            self.assertEqual(station_ids(files), ["0", "1", "2"])
            self.assertTrue(valid_rigid(seeds[1]))

    def test_changed_configuration_and_invalid_matrix_fall_back(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"view_{i}.ply" for i in range(3)]
            reference = self.reference(files)
            reference["transforms"][0][0][0] = 2
            self.assertIsNone(match_reference(reference, files, self.clouds, self.config)[0])
            reference = self.reference(files)
            self.config.registration.normal_radius_factor += 1
            seeds, report = match_reference(reference, files, self.clouds, self.config)
            self.assertIsNone(seeds)
            self.assertEqual(report["reason"], "configuration_changed")

    def test_bad_or_partial_station_ids_do_not_fall_through_to_geometry(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / name for name in ("view_1.ply", "view_1_copy.ply", "other.ply")]
            with self.assertRaises(ValueError):
                station_ids(files)
            self.assertIsNone(self.reference(files))

    def test_failed_fixture_never_produces_reference(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"view_{i}.ply" for i in range(3)]
            self.metrics[1].calibration_quality_passed = False
            self.assertIsNone(self.reference(files))

    def test_prepared_cache_isolates_voxel_normal_and_backend(self):
        cache = {}
        c = self.config.registration
        source = self.clouds[0]
        with patch("pointcloud_app.pipeline.prepare_registration_cloud", side_effect=lambda *a: object()) as prepare:
            a = _cached_registration_cloud(source, c, "cpu", cache)
            self.assertIs(a, _cached_registration_cloud(source, c, "cpu", cache))
            c.registration_voxel_size /= 2
            self.assertIsNot(a, _cached_registration_cloud(source, c, "cpu", cache))
            c.normal_radius_factor += 1
            _cached_registration_cloud(source, c, "cpu", cache)
            _cached_registration_cloud(source, c, "gpu", cache)
            self.assertEqual(prepare.call_count, 4)

    def test_far_geometry_and_missing_views_fall_back(self):
        with TemporaryDirectory() as directory:
            files = [Path(directory) / f"random_{i}.ply" for i in range(3)]
            reference = self.reference(files)
            shifted = [cloud(np.asarray(c.points) + 100) for c in self.clouds]
            self.assertIsNone(match_reference(reference, files, shifted, self.config)[0])
            self.assertIsNone(match_reference(reference, files[:-1], self.clouds[:-1], self.config)[0])

    def test_seed_is_only_initialization_and_fixture_rejection_falls_back(self):
        self.config.calibration.minimum_points = 30
        correction = np.eye(4)
        correction[0, 3] = .02
        args = (self.clouds[0], self.clouds[0], self.clouds[1], self.clouds[1],
                np.eye(4), self.config, self.config.registration, "cpu", {}, {})
        with patch("pointcloud_app.pipeline._refine_with_icp_diagnostics",
                   return_value=(correction, .95, .01, {})) as refine, \
                patch("pointcloud_app.pipeline.evaluate_full_cloud_alignment",
                      return_value=(.95, .01)):
            candidate, diagnostic = _refine_pose_seed(*args)
            self.assertTrue(diagnostic["reused"])
            self.assertAlmostEqual(candidate[0][0, 3], .02)
            self.assertEqual(refine.call_count, 1)
            correction[0, 3] = 50
            self.assertIsNone(_refine_pose_seed(*args)[0])
            refine.return_value = (np.eye(4), .01, 2, {})
            self.assertIsNone(_refine_pose_seed(*args)[0])
            refine.side_effect = ValueError("insufficient current fixture points")
            self.assertIsNone(_refine_pose_seed(*args)[0])

    def test_seed_localizes_current_fixture_but_keeps_full_scene_gate(self):
        localized = cloud(np.tile([5., 5., 0.], (1000, 1)))
        before = np.asarray(self.clouds[0].points).copy()
        args = (self.clouds[0], self.clouds[0], self.clouds[1], self.clouds[1],
                np.eye(4), self.config, self.config.registration, "cpu", {}, {})
        with patch("pointcloud_app.pipeline.crop_calibration_cloud", return_value=localized), patch(
                "pointcloud_app.pipeline._refine_with_icp_diagnostics",
                return_value=(np.eye(4), .95, .01, {})) as refine, patch(
                "pointcloud_app.pipeline.evaluate_full_cloud_alignment",
                return_value=(.95, .01)) as evaluate:
            candidate, diagnostic = _refine_pose_seed(*args)
            self.assertTrue(diagnostic["localized_roi"])
            self.assertTrue(diagnostic["reused"])
            self.assertIs(refine.call_args.args[0], localized)
            self.assertEqual(candidate[5], 1000)
            evaluate.return_value = (.2, .01)
            self.assertIsNone(_refine_pose_seed(*args)[0])
        np.testing.assert_array_equal(np.asarray(self.clouds[0].points), before)

    def test_history_loader_rejects_unverified_changed_or_tampered_reference(self):
        from pointcloud_app.automatic_runtime import load_verified_pose_reference
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / "capture/group_1" / f"view_{i}.ply" for i in range(3)]
            reference = self.reference(files)
            source = root / "group_1/run_summary.json"
            source.parent.mkdir()
            summary = {"input_files": list(map(str, files)),
                "registration_metrics": [dict(calibration_quality_passed=True,
                    included_in_fusion=True, transformation=t.tolist()) for t in self.transforms],
                "pose_graph": {"calibration_nodes": [dict(accepted=True) for _ in files],
                    "calibration_edges": [dict(source_index=0, target_index=i, accepted=True)
                                          for i in (1, 2)]}}
            source.write_text(json.dumps(summary))
            reference["source_summary"] = str(source)
            history = root / "automatic/20260929/pose_reference.json"
            history.parent.mkdir(parents=True)
            history.write_text(json.dumps(reference))
            newer = root / "automatic/20260930/pose_reference.json"
            newer.parent.mkdir()
            newer.write_text("invalid json")
            with patch("pointcloud_app.downstream_bridge._derive_processing_quality",
                       return_value={"status": "PASS"}):
                loaded, path = load_verified_pose_reference(root, self.config, files)
                self.assertEqual(path, history)
                self.assertEqual(loaded, reference)
                self.config.registration.global_random_seed += 1
                self.assertEqual(load_verified_pose_reference(root, self.config, files), (None, None))
                self.config.registration.global_random_seed -= 1
                self.assertEqual(load_verified_pose_reference(root, self.config,
                    [root / "another_family/1/a.ply"]), (None, None))
                summary["registration_metrics"][1]["included_in_fusion"] = False
                source.write_text(json.dumps(summary))
                self.assertEqual(load_verified_pose_reference(root, self.config, files), (None, None))
                summary["registration_metrics"][1]["included_in_fusion"] = True
                source.write_text(json.dumps(summary))
                reference["transforms"][1][0][3] += .1
                history.write_text(json.dumps(reference))
                self.assertEqual(load_verified_pose_reference(root, self.config, files), (None, None))

    def test_worker_final_quality_failure_reprocesses_without_seed(self):
        import threading
        from pointcloud_app import automatic_runtime as runtime
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = []
            for i in range(2):
                path = root / str(i) / "run_summary.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"pose_reference": {"views": [{"reused": i == 0}]}}))
                summaries.append(SimpleNamespace(summary_path=path))
            request = dict(id="x", batch_id="b", name="g", files=[], signature=[],
                           resources={}, config={}, output_root=directory,
                           dxf=str(root / "test.dxf"), selection={}, step=.05,
                           use_pose_reference=True, pose_reference={"version": 1})
            reports = [{"measurement_quality": {"status": status},
                        "source_dxf": str(root / str(i) / "test.dxf")}
                       for i, status in enumerate(("INSUFFICIENT_DATA", "PASS"))]
            with patch("pointcloud_app.native.backend.Backend") as factory, \
                    patch("pointcloud_app.native.backend.config_from_dict"), \
                    patch("pointcloud_app.native.backend.acquisition_signature", return_value=()), \
                    patch.object(runtime, "_stop", threading.Event(), create=True), \
                    patch.object(runtime, "_events", SimpleNamespace(put=lambda _: None), create=True):
                factory.return_value.process.side_effect = summaries
                factory.return_value.analyze.side_effect = reports
                record = runtime._run_group(request)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["report"]["measurement_quality"]["status"], "PASS")
            self.assertIn("pose_reference_attempt", record)
            self.assertEqual(factory.return_value.process.call_count, 2)
            self.assertNotIn("pose_reference", factory.return_value.process.call_args.kwargs)
            self.assertNotIn("learned_pose_reference", record)

    def test_scheduler_learns_once_then_reuses_across_batches_and_invalidates_config(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from pointcloud_app.automatic_runtime import run_batch
        state, executed = {}, []
        template = {"version": 1, "source_summary": "first"}
        requests = [dict(id=str(i), name=str(i), files=[f"D:/capture/{i}/view_1.ply"],
                         config={"a": 1}, use_pose_reference=True) for i in range(4)]

        def worker(request):
            executed.append(request)
            result = dict(id=request["id"], name=request["name"], status="completed")
            if "pose_reference" not in request:
                result["learned_pose_reference"] = dict(template)
            return result

        with patch("pointcloud_app.automatic_runtime.ProcessPoolExecutor",
                   side_effect=lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"])), \
                patch("pointcloud_app.automatic_runtime._run_group", side_effect=worker):
            records = run_batch(requests, {"workers": 2, "threads_per_worker": 1}, None,
                                lambda _: None, lambda _: None, threading.Event(),
                                reference_state=state)
            by_id = {r["id"]: r for r in executed}
            self.assertTrue(all("pose_reference" not in by_id[str(i)] for i in (0, 1)))
            self.assertTrue(all(by_id[str(i)]["pose_reference"] == template for i in (2, 3)))
            self.assertTrue(all("learned_pose_reference" not in r for r in records))
            run_batch(requests[:1], {"workers": 1, "threads_per_worker": 1}, None,
                      lambda _: None, lambda _: None, threading.Event(), reference_state=state)
            self.assertEqual(executed[-1]["pose_reference"], template)
            requests[0]["config"] = {"a": 2}
            run_batch(requests[:1], {"workers": 1, "threads_per_worker": 1}, None,
                      lambda _: None, lambda _: None, threading.Event(), reference_state=state)
            self.assertNotIn("pose_reference", executed[-1])

    def test_new_batch_initial_workers_receive_verified_history(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from pointcloud_app.automatic_runtime import run_batch
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "automatic/current/batch.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"records": []}))
            requests = [dict(id=str(i), name=str(i), files=[str(root / f"capture/{i}/1.ply")],
                config=self.config.as_dict(), output_root=str(root), use_pose_reference=True)
                for i in range(2)]
            template = {"version": 1, "source_summary": "verified_history"}
            executed = []
            def worker(request):
                executed.append(request)
                return dict(id=request["id"], name=request["name"], status="completed")
            with patch("pointcloud_app.automatic_runtime.load_verified_pose_reference",
                    return_value=(template, root / "automatic/older/pose_reference.json")), patch(
                    "pointcloud_app.automatic_runtime.ProcessPoolExecutor",
                    side_effect=lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"])), patch(
                    "pointcloud_app.automatic_runtime._run_group", side_effect=worker):
                run_batch(requests, {"workers": 2, "threads_per_worker": 1}, manifest,
                    lambda _: None, lambda _: None, threading.Event(), reference_state={})
            self.assertEqual(len(executed), 2)
            self.assertTrue(all(r["pose_reference"] == template for r in executed))
            stored = json.loads(manifest.read_text())
            self.assertIn("pose_reference_bootstrap_source", stored)
            self.assertEqual(json.loads(Path(stored["pose_reference_path"]).read_text()), template)

    def test_reference_requires_connected_accepted_graph_and_processing_quality(self):
        summary = {
            "registration_metrics": [
                dict(calibration_quality_passed=True, included_in_fusion=True,
                     transformation=np.eye(4).tolist()) for _ in range(3)],
            "pose_graph": {
                "calibration_nodes": [dict(accepted=True) for _ in range(3)],
                "calibration_edges": [
                    dict(source_index=0, target_index=i, accepted=True) for i in (1, 2)],
            },
        }
        with patch("pointcloud_app.downstream_bridge._derive_processing_quality",
                   return_value={"status": "PASS"}) as quality:
            self.assertTrue(registration_verified(summary))
            quality.return_value = {"status": "INSUFFICIENT_DATA"}
            self.assertFalse(registration_verified(summary))
            quality.return_value = {"status": "PASS"}
            summary["pose_graph"]["calibration_edges"].pop()
            self.assertFalse(registration_verified(summary))

    def test_cad_coverage_does_not_override_current_fixture_registration(self):
        import threading
        from pointcloud_app import automatic_runtime as runtime
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "run_summary.json"
            path.write_text(json.dumps({"pose_reference": {"views": [{"reused": True}]}}))
            request = dict(id="x", batch_id="b", name="g", files=[], signature=[],
                           resources={}, config={}, output_root=directory,
                           dxf=str(root / "test.dxf"), selection={}, step=.05,
                           use_pose_reference=True, pose_reference={"version": 1})
            with patch("pointcloud_app.native.backend.Backend") as factory, \
                    patch("pointcloud_app.native.backend.config_from_dict"), \
                    patch("pointcloud_app.native.backend.acquisition_signature", return_value=()), \
                    patch("pointcloud_app.registration_reference.registration_verified", return_value=True), \
                    patch.object(runtime, "_stop", threading.Event(), create=True), \
                    patch.object(runtime, "_events", SimpleNamespace(put=lambda _: None), create=True):
                factory.return_value.process.return_value = SimpleNamespace(summary_path=path)
                factory.return_value.analyze.return_value = {
                    "measurement_quality": {"status": "INSUFFICIENT_DATA", "processing_status": "PASS"},
                    "source_dxf": str(root / "test.dxf")}
                record = runtime._run_group(request)
            self.assertEqual(factory.return_value.process.call_count, 1)
            self.assertEqual(record["report"]["measurement_quality"]["status"], "INSUFFICIENT_DATA")

    def test_station_metadata_changed_in_queue_is_rejected(self):
        import threading
        from pointcloud_app import automatic_runtime as runtime
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "station_order.json"
            metadata.write_text('{"views":[]}')
            request = dict(id="x", batch_id="b", name="g",
                           files=[str(root / "view_1.ply")], signature=[], resources={},
                           station_metadata_signature="old")
            with patch("pointcloud_app.native.backend.acquisition_signature", return_value=()), \
                    patch("pointcloud_app.native.backend.Backend") as backend, \
                    patch.object(runtime, "_stop", threading.Event(), create=True), \
                    patch.object(runtime, "_events", SimpleNamespace(put=lambda _: None), create=True):
                record = runtime._run_group(request)
            self.assertEqual(record["status"], "failed")
            backend.assert_not_called()
