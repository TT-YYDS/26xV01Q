import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pointcloud_app.desktop_runtime import discover_stable_input_groups
from pointcloud_app.models import Bounds2D, ProcessingConfig
from pointcloud_app.native.backend import default_config
from pointcloud_app.native.window import CAD_PAGE, DesktopWindow
from pointcloud_app.automatic_runtime import (
    estimate_batch_seconds, load_duration_history, measurement_table,
    resource_plan, prepare_batch, run_batch,
)
from pointcloud_app.workflow_profile import (
    build_profile,
    empty_profile,
    load_profile,
    save_profile,
)


class AutomaticWorkflowTests(unittest.TestCase):
    def test_automatic_start_from_every_page_does_not_launch_old_preview(self):
        for page in range(CAD_PAGE + 1):
            with self.subTest(page=page):
                calls = []
                owner = SimpleNamespace(
                    workflow_mode="manual", job=None, page=page, view_values={},
                    _reset_automatic_session=Mock(), log=Mock(),
                    _save_workflow_profile=Mock(), _refresh_workflow_mode_button=Mock(),
                    _refresh_auto_runtime=Mock(), _toolbar=Mock(), _draw=Mock(),
                )
                def select_page(index):
                    self.assertEqual(index, CAD_PAGE)
                    self.assertEqual(owner.view_values[CAD_PAGE], "automatic_measurements")
                    owner.page = index
                    calls.append("summary")
                owner.select_page = select_page
                owner.start_automatic = lambda: calls.append("start")
                DesktopWindow.workflow_mode_changed(owner, "automatic")
                self.assertEqual(calls, ["summary", "start"])
                self.assertEqual(owner.workflow_mode, "automatic")
                owner._draw.assert_not_called()
                owner._reset_automatic_session.assert_called_once()

    def test_switch_to_manual_preserves_automatic_table_and_selected_data(self):
        state = object()
        records = {"run-1": {"id": "run-1", "name": "automatic-group"}}
        owner = SimpleNamespace(
            workflow_mode="automatic", job=None, page=CAD_PAGE,
            view_values={CAD_PAGE: "measurements"},
            groups={"automatic-group": state}, group_name="automatic-group",
            auto_records=records, stop_automatic=Mock(), log=Mock(),
            _save_workflow_profile=Mock(), _refresh_workflow_mode_button=Mock(),
            _refresh_auto_runtime=Mock(), _toolbar=Mock(), _draw=Mock(),
        )
        owner._showing_automatic_measurements = lambda: DesktopWindow._showing_automatic_measurements(owner)
        DesktopWindow.workflow_mode_changed(owner, "manual")
        self.assertEqual(owner.workflow_mode, "manual")
        self.assertEqual(owner.view_values[CAD_PAGE], "automatic_measurements")
        self.assertTrue(owner._showing_automatic_measurements())
        self.assertIs(owner.auto_records, records)
        self.assertIs(owner.groups["automatic-group"], state)
        self.assertEqual(owner.group_name, "automatic-group")
        owner._table = Mock()
        with patch("pointcloud_app.native.window.measurement_table", return_value=(["batch"], [["run-1"]])) as table:
            DesktopWindow._draw(owner)
        self.assertEqual(list(table.call_args.args[0]), list(records.values()))
        owner._table.assert_called_once_with(["batch"], [["run-1"]], measurement_summary=True)
        owner.view_values[CAD_PAGE] = "measurements"
        self.assertFalse(owner._showing_automatic_measurements())

    def test_batch_finishing_after_manual_switch_keeps_automatic_summary(self):
        owner = SimpleNamespace(
            workflow_mode="manual", auto_enabled=False, auto_pending=set(),
            view_values={}, _automatic_record_update=Mock(), _freeze_auto_runtime=Mock(),
            _freeze_auto_batch=Mock(), select_page=Mock(), _toolbar=Mock(),
            _draw=Mock(), _refresh_auto_runtime=Mock(),
        )
        DesktopWindow._automatic_batch_finished(owner, [{"id": "last-result"}])
        self.assertEqual(owner.view_values[CAD_PAGE], "automatic_measurements")
        owner.select_page.assert_called_once_with(CAD_PAGE)

    def _files(self, directory: Path, count: int, prefix: str = "scan") -> list[Path]:
        files = []
        for index in range(count):
            path = directory / f"{prefix}_{index}.ply"
            path.write_bytes(b"ply\n")
            files.append(path)
        old_time = max(0.0, time.time() - 5.0)
        for path in files:
            os.utime(path, (old_time, old_time))
        return files

    def test_profile_roundtrip_preserves_source_region_and_cad_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = ProcessingConfig()
            config.section.target_region = Bounds2D(-1.0, 2.0, -3.0, 4.0)
            payload = build_profile(
                workflow_mode="automatic",
                source_kind="camera",
                source_path=root / "camera_inbox",
                config=config,
                dxf=root / "drawing.dxf",
                selection={
                    "registration_paths": [[4, 2]],
                    "measurement_paths": [[6, 8]],
                },
                cad_step=0.025,
            )
            path = root / "workflow_profile.json"
            save_profile(payload, path)
            loaded, error = load_profile(path)

            self.assertIsNone(error)
            self.assertEqual(loaded["workflow_mode"], "automatic")
            self.assertEqual(loaded["source"]["kind"], "camera")
            self.assertEqual(Path(loaded["source"]["path"]), (root / "camera_inbox").resolve())
            self.assertEqual(loaded["config"]["section"]["target_region"]["x_min"], -1.0)
            self.assertEqual(loaded["cad"]["selection"]["registration_paths"], [[4, 2]])
            self.assertEqual(loaded["cad"]["step"], 0.025)

    def test_stable_discovery_accepts_complete_direct_group(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root, 6)
            groups = discover_stable_input_groups(root, minimum_file_age_seconds=1.0)

            self.assertEqual(list(groups), [root.name])
            self.assertEqual(groups[root.name], [path.resolve() for path in files])

    def test_stable_discovery_ignores_incomplete_and_recent_groups(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "complete"
            complete.mkdir()
            self._files(complete, 5)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            self._files(incomplete, 4)
            recent = root / "recent"
            recent.mkdir()
            recent_files = self._files(recent, 5)
            now = time.time()
            for path in recent_files:
                os.utime(path, (now, now))

            groups = discover_stable_input_groups(root, minimum_file_age_seconds=1.0)

            self.assertEqual(set(groups), {"complete"})

    def test_stable_discovery_keeps_nested_groups_when_direct_transfer_is_incomplete(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            direct = self._files(root, 2, prefix="partial")
            nested = root / "batch_001"
            nested.mkdir()
            nested_files = self._files(nested, 5)

            groups = discover_stable_input_groups(root, minimum_file_age_seconds=1.0)

            self.assertEqual(set(groups), {"batch_001"})
            self.assertEqual(groups["batch_001"], [path.resolve() for path in nested_files])

    def test_empty_profile_is_safe_and_json_serializable(self) -> None:
        payload = empty_profile()
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.assertEqual(payload["source"]["kind"], "folder")
        self.assertIsNone(payload["config"])

    def test_automatic_batch_marks_failed_signature_and_does_not_retry_it(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root, 5)
            owner = SimpleNamespace(
                auto_enabled=True,
                workflow_mode="automatic",
                auto_pending={"batch"},
                auto_seen={},
                auto_records={},
                auto_batch_completed=0,
                _refresh_auto_runtime=Mock(),
                log=Mock(),
                _toolbar=Mock(),
                _draw=Mock(),
            )
            signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files)
            record = dict(id="run-1", name="batch", status="failed", error="synthetic failure",
                          signature=signature, summary_path=None, report=None)

            DesktopWindow._automatic_record_update(owner, record)
            DesktopWindow._automatic_record_update(owner, record)
            DesktopWindow._automatic_discovered(owner, [("batch", files, signature)])

            self.assertEqual(owner.auto_seen["batch"], signature)
            self.assertEqual(owner.auto_pending, set())
            self.assertEqual(owner.auto_batch_completed, 1)
            self.assertTrue(any("synthetic failure" in call.args[0] for call in owner.log.call_args_list))

    def test_cancelled_record_can_be_retried_and_repeated_group_names_keep_history(self):
        owner = SimpleNamespace(
            auto_pending={"batch"}, auto_seen={}, auto_records={}, auto_batch_completed=0,
            _refresh_auto_runtime=Mock(), log=Mock(), _draw=Mock(),
        )
        for index, status in enumerate(("failed", "cancelled")):
            record = dict(id=str(index), name="batch", status=status, error=status,
                          signature=(("scan", index, index),), summary_path=None, report=None)
            DesktopWindow._automatic_record_update(owner, record)
        self.assertEqual(len(owner.auto_records), 2)
        self.assertEqual(owner.auto_seen["batch"], (("scan", 0, 0),))

    def test_resource_plan_respects_memory_cpu_and_shared_gpu(self):
        with TemporaryDirectory() as directory:
            files = self._files(Path(directory), 5)
            groups = [files] * 8
            plan = resource_plan(groups, cpu_count=16, available_bytes=32 * 1024**3)
            self.assertEqual(plan["workers"], 4)
            self.assertLessEqual(plan["workers"] * plan["threads_per_worker"], 15)
            limited = resource_plan(groups, cpu_count=16, available_bytes=3 * 1024**3)
            self.assertEqual(limited["workers"], 1)
            gpu = resource_plan(groups, cpu_count=16, available_bytes=32 * 1024**3, gpu=True)
            self.assertEqual(gpu["workers"], 1)

    def test_batch_freezes_template_and_assigns_distinct_run_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root, 5)
            drawing = root / "drawing.dxf"
            drawing.write_text("fixture", encoding="utf-8")
            config = default_config("1")
            selection = {"registration_paths": [[1]], "measurement_paths": [[2]]}
            requests, plan, manifest = prepare_batch(
                [("a", files, ()), ("b", files, ())], config,
                {"dxf": str(drawing), "selection": selection, "step": 0.025}, root / "outputs")
            drawing.write_text("modified original", encoding="utf-8")
            self.assertEqual(Path(requests[0]["dxf"]).read_text(encoding="utf-8"), "fixture")
            self.assertNotEqual(requests[0]["id"], requests[1]["id"])
            self.assertEqual(requests[0]["config"], config.as_dict())
            self.assertEqual(requests[0]["selection"], selection)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["resources"], plan)

    def test_measurement_table_hides_normal_deviation_and_time_without_changing_records(self):
        measurement = dict(label="path", mean_signed=-0.01, mean_signed_x=-0.02,
                           mean_signed_y=0.03, cad_path_angle_deg=45,
                           signed_angle_difference_deg=-0.5)
        records = [
            dict(name="a", batch_id="capture1", elapsed_seconds=12.5, report={
                "measurement_quality": {"status": "PASS"},
                "measurements": {"one": measurement, "two": measurement},
            }),
            dict(name="a", batch_id="capture2", elapsed_seconds=11, report={
                "measurement_quality": {"status": "INSUFFICIENT_DATA"}, "measurements": {},
            }),
            dict(name="b", status="failed", error="missing data", elapsed_seconds=3),
            dict(name="c", status="cancelled", error="cancelled"),
        ]
        original_records = json.dumps(records, sort_keys=True)
        columns, rows = measurement_table(records)
        self.assertEqual(columns, ["组 / 批次", "质量", "测量路径", "平均 X 偏差",
                                   "平均 Y 偏差", "角度差 (°)"])
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(len(row) == len(columns) for row in rows))
        self.assertEqual(rows[0][3:], ["-0.02", "0.03", "-0.5"])
        self.assertEqual(json.dumps(records, sort_keys=True), original_records)
        self.assertFalse(any("CAD 路径角" in column for column in columns))
        self.assertEqual(measurement["cad_path_angle_deg"], 45)
        self.assertIn("capture2", rows[2][0])
        self.assertEqual(rows[2][1], "INSUFFICIENT_DATA")
        self.assertEqual(rows[3][1], "失败")
        self.assertEqual(rows[4][1], "已取消")

    def test_auto_clock_is_independent_of_task_clock(self):
        owner = SimpleNamespace(auto_started_at=100.0, auto_elapsed=2.0,
                                _task_started_at=119.0)
        with patch("pointcloud_app.native.window.perf_counter", return_value=125.0):
            self.assertEqual(DesktopWindow._current_auto_elapsed(owner), 27.0)
        owner.auto_started_at = None
        self.assertEqual(DesktopWindow._current_auto_elapsed(owner), 2.0)

    def test_startup_mode_is_manual_even_when_saved_template_mode_is_automatic(self):
        from PySide6.QtWidgets import QApplication
        from unittest.mock import patch
        app = QApplication.instance() or QApplication([])
        with patch("pointcloud_app.native.window.load_profile",
                   return_value=({"workflow_mode": "automatic", "source": {}}, None)):
            window = DesktopWindow(Path("__no_startup_import__"))
            self.assertEqual(window.workflow_mode, "manual")
            self.assertFalse(window.auto_enabled)
            self.assertFalse(window.auto_timer.isActive())
            window.close()
            app.processEvents()

    def test_automatic_mode_locks_other_controls_between_jobs_and_during_stop(self):
        names = (
            "toolbar", "stack", "table", "group_combo", "history_button",
            "params_button", "export_button", "log_button",
        )
        owner = SimpleNamespace(
            **{name: Mock() for name in names},
            nav=[Mock() for _ in range(5)], workflow_mode_button=Mock(),
            workflow_mode="automatic", state=SimpleNamespace(result=object()),
        )
        controls = [getattr(owner, name) for name in names] + owner.nav
        for enabled, busy in ((True, False), (True, True), (False, True), (False, False)):
            with self.subTest(auto_enabled=enabled, busy=busy):
                owner.auto_enabled = enabled
                DesktopWindow._busy(owner, busy)
                for control in controls:
                    control.setEnabled.assert_called_with(False)
                owner.workflow_mode_button.setEnabled.assert_called_with(True)
        owner.workflow_mode = "manual"
        DesktopWindow._busy(owner, True)
        for control in controls:
            control.setEnabled.assert_called_with(False)
        DesktopWindow._busy(owner, False)
        for control in controls:
            control.setEnabled.assert_called_with(True)
        owner.state = None
        DesktopWindow._busy(owner, False)
        owner.params_button.setEnabled.assert_called_with(False)
        owner.export_button.setEnabled.assert_called_with(False)

    def test_new_automatic_session_clears_previous_results_and_signatures(self):
        owner = SimpleNamespace(
            job=None,
            auto_enabled=True,
            auto_timer=Mock(),
            auto_runtime_timer=Mock(),
            auto_seen={"batch": (("scan.ply", 1, 2),)},
            auto_pending={"batch"},
            auto_records={"run-1": {"name": "batch", "status": "completed"}},
            auto_resources={"workers": 2, "threads_per_worker": 4},
            auto_batch_total=1,
            auto_batch_completed=1,
            auto_batch_started_at=10.0,
            auto_batch_elapsed=12.0,
            auto_batch_expected_seconds=15.0,
            auto_batch_remaining_seconds=0.0,
            auto_batch_last_elapsed=12.0,
            auto_batch_last_expected=15.0,
            auto_started_at=10.0,
            auto_elapsed=12.0,
            groups={
                "batch": SimpleNamespace(automatic_result=True),
                "manual": SimpleNamespace(automatic_result=False),
            },
            group_name="batch",
            _refresh_auto_runtime=Mock(),
        )

        self.assertTrue(DesktopWindow._reset_automatic_session(owner))
        self.assertEqual(owner.auto_seen, {})
        self.assertEqual(owner.auto_pending, set())
        self.assertEqual(owner.auto_records, {})
        self.assertIsNone(owner.auto_resources)
        self.assertEqual(owner.auto_batch_total, 0)
        self.assertEqual(owner.auto_batch_completed, 0)
        self.assertEqual(owner.auto_batch_elapsed, 0.0)
        self.assertIsNone(owner.auto_batch_started_at)
        self.assertIsNone(owner.auto_batch_expected_seconds)
        self.assertIsNone(owner.auto_batch_remaining_seconds)
        self.assertIsNone(owner.auto_batch_last_elapsed)
        self.assertIsNone(owner.auto_batch_last_expected)
        self.assertIsNone(owner.auto_started_at)
        self.assertEqual(owner.auto_elapsed, 0.0)
        self.assertEqual(set(owner.groups), {"manual"})
        owner.auto_timer.stop.assert_called_once_with()
        owner.auto_runtime_timer.stop.assert_called_once_with()

    def test_new_session_allows_same_signature_to_be_discovered_again(self):
        signature = (("scan.ply", 1, 2),)
        owner = SimpleNamespace(
            job=None,
            auto_enabled=True,
            workflow_mode="automatic",
            auto_timer=Mock(),
            auto_runtime_timer=Mock(),
            auto_seen={"batch": signature},
            auto_pending=set(),
            auto_records={"run-1": {"name": "batch", "status": "completed"}},
            auto_resources=None,
            auto_batch_total=0,
            auto_batch_completed=0,
            auto_batch_started_at=None,
            auto_batch_elapsed=0.0,
            auto_batch_expected_seconds=None,
            auto_batch_remaining_seconds=None,
            auto_batch_last_elapsed=None,
            auto_batch_last_expected=None,
            auto_started_at=None,
            auto_elapsed=0.0,
            groups={},
            group_name="",
            workflow_profile={"cad": {"dxf": "drawing.dxf", "selection": {}, "step": 0.05}},
            view_values={},
            _refresh_auto_runtime=Mock(),
            _source_blocker=Mock(return_value=""),
            _profile_config=Mock(return_value=default_config("batch")),
            _save_workflow_profile=Mock(),
            _draw=Mock(),
            select_page=Mock(),
            _start=Mock(return_value=True),
            _automatic_batch_finished=Mock(),
            log=Mock(),
        )
        DesktopWindow._reset_automatic_session(owner)
        owner.auto_enabled = True
        requests = [{"id": "run-2", "batch_id": "new-batch", "name": "batch"}]
        with patch("pointcloud_app.native.window.resolve_compute_backend", return_value="cpu"), \
                patch("pointcloud_app.native.window.prepare_batch",
                      return_value=(requests, {"workers": 1, "threads_per_worker": 1},
                                    "manifest.json")):
            DesktopWindow._automatic_discovered(owner, [("batch", [], signature)])

        self.assertEqual(owner.auto_records, {})
        self.assertEqual(owner.auto_pending, {"batch"})
        self.assertEqual(owner.auto_batch_total, 1)
        self.assertEqual(owner._start.call_count, 1)

    def test_new_session_does_not_clear_a_running_task(self):
        owner = SimpleNamespace(job=object())
        self.assertFalse(DesktopWindow._reset_automatic_session(owner))

    def test_batch_eta_uses_history_and_completed_groups(self):
        estimate = estimate_batch_seconds(5, 1, 2, [100, 120], 55)
        self.assertEqual(estimate["per_group_seconds"], 110)
        self.assertEqual(estimate["remaining_seconds"], 220)
        self.assertEqual(estimate["total_seconds"], 275)
        self.assertIsNone(estimate_batch_seconds(2, 0, 1, [], 0))

    def test_duration_history_reads_automatic_and_desktop_timing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "run_summary.json"
            second = root / "b" / "run_summary.json"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text(json.dumps({
                "automatic_workflow": {"elapsed_seconds": 12.5}}, ensure_ascii=False)
            )
            second.write_text(json.dumps({
                "desktop_acquisition": {"timing": {"total_seconds": 8.0}}}),
                encoding="utf-8")
            self.assertEqual(load_duration_history(root), [8.0, 12.5])

    def test_streaming_task_keeps_completed_values_when_cancelled(self):
        from pointcloud_app.native.widgets import Task
        def operation(progress, notify, cancelled):
            notify({"id": "finished"})
            cancelled.set()
            return ["finished"]
        task = Task(operation, streaming=True)
        task.run()
        self.assertIsNone(task.error)
        self.assertEqual(task.value, ["finished"])

    def test_eta_updates_and_freezes_after_cancel(self):
        from unittest.mock import patch
        owner = SimpleNamespace(
            auto_batch_total=4, auto_batch_completed=1, auto_batch_started_at=100.0,
            auto_batch_elapsed=0.0, auto_resources={"workers": 2},
            auto_duration_history=[60.0], auto_records={},
            _current_batch_elapsed=lambda: 10.0,
        )
        with patch("pointcloud_app.native.window.perf_counter", return_value=110.0):
            estimate = DesktopWindow._current_batch_estimate(owner)
        self.assertEqual(estimate["remaining_seconds"], 120.0)
        self.assertEqual(estimate["total_seconds"], 130.0)

    def test_scheduler_publishes_before_batch_end_and_stops_dispatch_on_cancel(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "batch.json"
            manifest.write_text('{"records":[]}', encoding="utf-8")
            requests = [dict(id=str(index), name=str(index)) for index in range(3)]
            cancelled = threading.Event()
            notified = []
            executed = []
            def work(request):
                executed.append(request["id"])
                return dict(request, status="completed")
            def notify(record):
                notified.append(record)
                cancelled.set()
            with patch("pointcloud_app.automatic_runtime.ProcessPoolExecutor",
                       side_effect=lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"])), \
                    patch("pointcloud_app.automatic_runtime._run_group", side_effect=work):
                result = run_batch(requests, {"workers": 1, "threads_per_worker": 1},
                                   manifest, Mock(), notify, cancelled)
            self.assertEqual(executed, ["0"])
            self.assertEqual(result, notified)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["records"], result)

    def test_worker_rejects_queued_input_changes_without_processing(self):
        from pointcloud_app.automatic_runtime import _run_group
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = self._files(root, 5)
            signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files)
            files[0].write_bytes(b"changed")
            request = dict(id="run", batch_id="batch", name="group", files=list(map(str, files)),
                           signature=signature, resources={}, config={}, output_root=directory)
            with patch("pointcloud_app.automatic_runtime._stop", threading.Event(), create=True), \
                    patch("pointcloud_app.automatic_runtime._events", Mock(), create=True), \
                    patch("pointcloud_app.native.backend.Backend") as backend:
                result = _run_group(request)
            backend.assert_not_called()
            self.assertEqual(result["status"], "failed")
            self.assertIn("排队期间", result["error"])

    def test_automatic_profile_save_keeps_operator_template_not_frozen_outputs(self):
        config = default_config("1")
        payload = build_profile(
            workflow_mode="manual", source_kind="folder", source_path=Path("media"),
            config=config, dxf=Path("original.dxf"),
            selection={"registration_paths": [[1]], "measurement_paths": [[2]]}, cad_step=0.025,
        )
        owner = SimpleNamespace(
            workflow_mode="automatic", workflow_profile=payload, source_kind="folder",
            source_path=Path("media"), _profile_config=lambda: config,
            state=SimpleNamespace(automatic_result=True, dxf=Path("outputs/frozen.dxf")),
        )
        with patch("pointcloud_app.native.window.save_profile") as save:
            DesktopWindow._save_workflow_profile(owner)
        saved = save.call_args.args[0]
        self.assertEqual(saved["cad"]["dxf"], str(Path("original.dxf").resolve()))
        self.assertEqual(saved["cad"]["step"], 0.025)


if __name__ == "__main__":
    unittest.main()
