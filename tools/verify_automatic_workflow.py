"""Check automatic workflow UI without touching the operator's saved template."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_real_automatic_ui():
    import hashlib
    from unittest.mock import patch

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from pointcloud_app.native.backend import config_from_dict
    from pointcloud_app.native.window import CAD_PAGE, DesktopWindow
    from pointcloud_app.workflow_profile import build_profile, load_profile, save_profile

    manifests = sorted((ROOT / "tmp/automatic_runtime_verification").glob(
        "*/outputs/automatic/*/batch.json"))
    if not manifests:
        raise AssertionError("没有可复用的真实自动批次输入")
    source_manifest = manifests[-1]
    saved = json.loads(source_manifest.read_text(encoding="utf-8"))
    completed = [record for record in saved["records"] if record.get("status") == "completed"]
    if not completed:
        raise AssertionError("真实自动批次没有完成组")
    request = saved["requests"][0]
    source_dir = Path(request["files"][0]).parent
    source_hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in request["files"]
    }
    config = config_from_dict(request["config"])
    profile = build_profile(
        workflow_mode="automatic", source_kind="folder", source_path=source_dir,
        config=config, dxf=Path(request["dxf"]),
        selection=request["selection"], cad_step=request["step"],
    )
    output = ROOT / "tmp" / "automatic_real_ui_verification"
    output.mkdir(parents=True, exist_ok=True)
    template = output / "isolated_workflow_profile.json"
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    errors = []
    with patch("pointcloud_app.native.window.load_profile", return_value=(profile, None)), \
            patch("pointcloud_app.native.window.save_profile",
                  side_effect=lambda payload: save_profile(payload, template)):
        window = DesktopWindow(ROOT / "__real_ui_no_autoload__")
        window.error = errors.append
        window.show()
        assert window.workflow_mode == "manual"
        assert not window.auto_enabled
        assert not window.auto_timer.isActive()

        def wait(predicate, timeout=600):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                app.processEvents()
                if errors:
                    raise AssertionError(errors)
                if predicate():
                    return
                time.sleep(0.05)
            raise TimeoutError(f"真实自动 UI 超时：{window.logs[-8:]}")

        try:
            window.import_root(source_dir)
            wait(lambda: window.job is None and bool(window.groups))
            window.resize(1360, 860)
            QTest.mouseClick(window.workflow_mode_button, Qt.MouseButton.LeftButton)
            wait(lambda: window.workflow_mode == "automatic" and window.auto_enabled)
            wait(lambda: len(window.auto_records) >= 1 and window.job is None)
            window.stop_automatic()
            window.view_values[CAD_PAGE] = "measurements"
            window.select_page(CAD_PAGE)
            wait(lambda: window.job is None and bool(window.table.rows), timeout=10)
            record = next(iter(window.auto_records.values()))
            assert record["status"] == "completed", record
            assert window.table.rows
            assert window.auto_elapsed > 0
            assert window.state.report is not None
            assert window.state.result is not None
            assert window.state.report["fit"]["pointcloud_transform_applied"] is False
            assert source_hashes == {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in request["files"]
            }
            window.grab().save(str(output / "real_automatic_results.png"))
            result = {
                "ok": True, "group": record["name"], "status": record["status"],
                "elapsed_seconds": record["elapsed_seconds"],
                "automatic_total_seconds": window.auto_elapsed,
                "rows": len(window.table.rows),
                "pointcloud_transform_applied": False,
                "source_inputs_unchanged": True,
                "output": str(output / "real_automatic_results.png"),
                "template_isolated": str(template),
            }
            (output / "verification.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            window.close()
            wait(lambda: window.job is None, timeout=30)
    return 0


def main():
    if "--real-ui" in sys.argv:
        return verify_real_automatic_ui()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QComboBox, QPushButton
    from pointcloud_app.workflow_profile import empty_profile, save_profile
    from pointcloud_app.native.window import DesktopWindow, CAD_PAGE

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    output = ROOT / "tmp" / "automatic_ui_verification"
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with TemporaryDirectory() as temporary:
        template = Path(temporary) / "workflow_profile.json"
        with patch("pointcloud_app.native.window.load_profile", return_value=(empty_profile(), None)), \
                patch("pointcloud_app.native.window.save_profile",
                      side_effect=lambda payload: save_profile(payload, template)):
            window = DesktopWindow(ROOT / "__no_autoload__")
            window.error = errors.append
            window.show()

            def wait(predicate, timeout=15):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    app.processEvents()
                    if predicate():
                        return
                    time.sleep(0.02)
                raise AssertionError(f"UI timed out: {window.logs[-4:]}")

            try:
                window.resize(900, 620)
                QTest.qWait(100)
                assert window.workflow_mode == "manual"
                assert not isinstance(window.workflow_mode_button, QComboBox)
                assert window.workflow_mode_button.height() >= 48
                assert not any(
                    button.text() == "并行处理全部有效组"
                    for button in window.findChildren(QPushButton)
                )
                QTest.mouseClick(window.workflow_mode_button, Qt.MouseButton.LeftButton)
                wait(lambda: window.workflow_mode == "automatic")
                assert window.workflow_mode_button.text() == "自动流程"
                assert not window.auto_enabled  # An incomplete template cannot run.
                window.auto_seen = {"stale": (("scan.ply", 1, 2),)}
                window.auto_pending = {"stale"}
                window.auto_records = {
                    "old-run": {"name": "stale", "status": "completed"}
                }
                window.groups["stale"] = SimpleNamespace(
                    automatic_result=True, selected_file=None, result=None, files=[]
                )
                window.group_combo.addItem("stale")
                window.group_name = "stale"
                QTest.mouseClick(window.workflow_mode_button, Qt.MouseButton.LeftButton)
                wait(lambda: window.workflow_mode == "manual")
                QTest.mouseClick(window.workflow_mode_button, Qt.MouseButton.LeftButton)
                wait(lambda: window.workflow_mode == "automatic")
                assert window.auto_seen == {}
                assert window.auto_pending == set()
                assert window.auto_records == {}
                assert "stale" not in window.groups
                assert window.group_combo.findText("stale") < 0
                window.view_values[CAD_PAGE] = "measurements"
                window.select_page(CAD_PAGE)
                assert window.stack.currentWidget() is window.table
                records = []
                for index in range(12):
                    status = ["PASS", "INSUFFICIENT_DATA", "FAIL"][index % 3]
                    records.append({
                        "id": str(index), "name": f"group_{index + 1}",
                        "batch_id": f"20260914_1200{index:02d}", "status": "completed",
                        "elapsed_seconds": 90.5 + index,
                        "report": {"measurement_quality": {"status": status},
                                   "measurements": {"path1": {
                                       "label": "测量路径 1 (实体 6 - 13)",
                                       "mean_signed": -0.027108, "mean_signed_x": -0.013133,
                                       "mean_signed_y": -0.012329, "cad_path_angle_deg": -146,
                                       "signed_angle_difference_deg": -0.49583,
                                   }}},
                    })
                records.append({"id": "failure", "name": "group_13", "status": "failed",
                                "error": "输入点云在排队期间发生变化", "elapsed_seconds": 0.2})
                window.auto_records = {row["id"]: row for row in records}
                window.auto_resources = {"workers": 3, "threads_per_worker": 5}
                window.auto_batch_total = 5
                window.auto_batch_completed = 1
                window.auto_batch_started_at = time.perf_counter() - 10
                window.auto_duration_history = [90.0]
                window.auto_enabled = True
                window.auto_started_at = time.perf_counter() - 65
                window.auto_runtime_timer.start()
                window._refresh_auto_runtime()
                assert "预计剩余" in window.auto_status.text()
                assert "预计完成" in window.auto_status.text()
                window._toolbar()
                window._draw()
                QTest.qWait(200)
                before = window.auto_status.text()
                wait(lambda: window.auto_status.text() != before, 4)
                assert len(window.table.rows) == 13
                assert window.table.table.horizontalScrollBar().maximum() > 0
                assert window.width() == 900
                window.grab().save(str(output / "automatic_900x620.png"))
                window.table.move(1)
                assert window.table.page == 1
                window.table.table.horizontalScrollBar().setValue(
                    window.table.table.horizontalScrollBar().maximum())
                QTest.qWait(100)
                window.grab().save(str(output / "automatic_small_metrics.png"))
                for size in ((1360, 860), (1920, 1080)):
                    window.resize(*size)
                    QTest.qWait(200)
                    window.grab().save(str(output / f"automatic_{size[0]}x{size[1]}.png"))
                    button = window.workflow_mode_button
                    assert button.geometry().right() < window.centralWidget().width()
                    assert button.fontMetrics().horizontalAdvance(button.text()) < button.width() - 20

                manifests = sorted((ROOT / "tmp/automatic_runtime_verification").glob(
                    "*/outputs/automatic/*/batch.json"))
                if manifests:
                    real = json.loads(manifests[-1].read_text(encoding="utf-8"))["records"]
                    if real and all(record.get("status") == "completed" for record in real):
                        window.auto_records = {}
                        window.auto_batch_completed = 0
                        for record in real:
                            window._automatic_record_update(record)
                        assert len(window.auto_records) == len(real)
                        assert len(window.table.rows) >= len(real)
                        assert window.state.report is not None
                        assert window.state.files[0].is_file()
                        window.auto_resources = json.loads(
                            manifests[-1].read_text(encoding="utf-8"))["resources"]
                        window.resize(1360, 860)
                        window.table.table.horizontalScrollBar().setValue(0)
                        QTest.qWait(200)
                        window.grab().save(str(output / "automatic_real_results.png"))

                # A discovery completion starts another job; the old completion must
                # not clear the replacement job's timer.
                gate = threading.Event()
                def next_job(_value):
                    window._start(lambda progress: gate.wait(3), lambda _value: None)
                window._start(lambda progress: None, next_job)
                wait(lambda: window.job is not None and window.job_callback is not next_job)
                assert window._task_started_at is not None
                assert window.task_timer.isActive()
                gate.set()
                wait(lambda: window.job is None)

                # Stop remains accessible while batch controls are busy.
                def streaming(progress, updated, cancelled):
                    while not cancelled.wait(0.02):
                        pass
                    return []
                window._start(streaming, window._automatic_batch_finished, streaming=True)
                assert window.auto_stop_button.isEnabled()
                QTest.mouseClick(window.auto_stop_button, Qt.MouseButton.LeftButton)
                wait(lambda: window.job is None)
                assert not window.auto_enabled
                assert window.auto_started_at is None
                elapsed = window.auto_elapsed
                QTest.qWait(100)
                assert window._current_auto_elapsed() == elapsed
                QTest.mouseClick(window.workflow_mode_button, Qt.MouseButton.LeftButton)
                wait(lambda: window.workflow_mode == "manual")
                assert not window.auto_status_bar.isVisible()
                window.workflow_mode = "automatic"
                window.auto_enabled = True
                window.auto_started_at = time.perf_counter()
                window._refresh_workflow_mode_button()
                window._start(streaming, window._automatic_batch_finished, streaming=True)
                assert window.workflow_mode_button.isEnabled()
                QTest.mouseClick(window.workflow_mode_button, Qt.MouseButton.LeftButton)
                assert window.workflow_mode == "manual"
                wait(lambda: window.job is None)
                assert not window.auto_enabled
                assert not errors, errors
                result = {
                    "ok": True, "rows": len(window.auto_records), "synthetic_rows": 13,
                    "errors": errors,
                    "viewports": [[900, 620], [1360, 860], [1920, 1080]],
                    "mode_button": True, "independent_timer": True,
                    "stop_during_batch": True, "nested_job_timer": True,
                    "template_isolated": str(template),
                }
                (output / "verification.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            finally:
                window.close()
                wait(lambda: window.job is None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
