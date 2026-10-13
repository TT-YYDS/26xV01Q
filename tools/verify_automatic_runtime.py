"""Run two isolated copies of a real acquisition through automatic processing."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parallel_probe(request):
    import os
    from pointcloud_app.automatic_runtime import _events
    directory = Path(request["directory"])
    (directory / request["name"]).touch()
    deadline = time.monotonic() + 10
    while len(list(directory.glob("probe_*"))) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError("The second worker did not start concurrently")
        time.sleep(0.02)
    _events.put({"name": request["name"], "message": "parallel worker ready"})
    time.sleep(0.3)
    return dict(request, status="completed", worker_pid=os.getpid(),
                omp_threads=os.environ.get("OMP_NUM_THREADS"))


def verify_parallel_scheduler():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from pointcloud_app.automatic_runtime import run_batch, write_json
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        manifest = directory / "batch.json"
        write_json(manifest, {"records": []})
        requests = [dict(id=str(index), name=f"probe_{index}", directory=temporary)
                    for index in range(2)]
        progress = []
        with patch("pointcloud_app.automatic_runtime._run_group", _parallel_probe):
            records = run_batch(
                requests, {"workers": 2, "threads_per_worker": 1}, manifest,
                progress.append, lambda _record: None, threading.Event())
        assert len(records) == 2
        assert len({record["worker_pid"] for record in records}) == 2
        assert all(record["omp_threads"] == "1" for record in records)
        assert len(progress) == 2
        result = dict(ok=True, independent_processes=2, progress_events=len(progress),
                      wall_seconds=json.loads(manifest.read_text())["wall_seconds"],
                      worker_pids=[record["worker_pid"] for record in records])
        output = ROOT / "tmp" / "automatic_parallel_verification"
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "verification.json", result)
        print(json.dumps(result, indent=2))
    return 0


def main():
    if "--probe" in sys.argv:
        return verify_parallel_scheduler()
    import shutil
    from pointcloud_app.automatic_runtime import prepare_batch, run_batch, measurement_table
    from pointcloud_app.native.backend import acquisition_signature, config_from_dict, sha256
    from pointcloud_app.workflow_profile import load_profile

    profile, error = load_profile()
    assert error is None, error
    source_summary = ROOT / "outputs/group_1/20260911_152711_067936/run_summary.json"
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    config = config_from_dict(summary)
    source_files = [Path(path) for path in summary["input_files"]]
    assert all(path.is_file() for path in source_files)
    cad = profile["cad"]
    assert cad["selection"], "Configure the manual CAD paths before the real verification."
    original_hashes = {str(path): sha256(path) for path in source_files}
    if len(sys.argv) == 3 and sys.argv[1] == "--review":
        manifest = Path(sys.argv[2])
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        requests, plan, records = saved["requests"], saved["resources"], saved["records"]
        output = manifest.parents[3]
        elapsed = saved.get("wall_seconds")
    else:
        output = ROOT / "tmp" / "automatic_runtime_verification" / time.strftime("%Y%m%d_%H%M%S")
        groups = []
        for name in ("automatic_A", "automatic_B"):
            directory = output / "inputs" / name
            directory.mkdir(parents=True)
            files = []
            for source in source_files:
                path = directory / source.name
                shutil.copy2(source, path)
                files.append(path)
            groups.append((name, files, acquisition_signature(files)))

        requests, plan, manifest = prepare_batch(groups, config, cad, output / "outputs")
        records = []
        started = time.perf_counter()
        def completed(record):
            records.append(record)
            print(f"RESULT {record['name']}: {record['status']} {record['elapsed_seconds']}", flush=True)
        run_batch(requests, plan, manifest, lambda message: print(message, flush=True),
                  completed, threading.Event())
        elapsed = time.perf_counter() - started
    assert len(records) == 2
    for record in records:
        assert record["status"] == "completed", record.get("error")
        result = json.loads(Path(record["summary_path"]).read_text(encoding="utf-8"))
        assert result["desktop_acquisition"]["input_verified"] is True
        cleanup = result["section_consensus"].get("final_component_cleanup")
        assert cleanup and cleanup["removed_points"] >= 0
        assert result["automatic_workflow"]["resources"] == plan
        assert record["report"]["fit"]["pointcloud_transform_applied"] is False
        assert record["report"]["measurement_quality"]["status"] in ("PASS", "FAIL", "INSUFFICIENT_DATA")
        assert result["desktop_acquisition"]["input_sha256"] == {
            path.name: original_hashes[str(path)] for path in source_files}
    assert original_hashes == {str(path): sha256(path) for path in source_files}
    columns, rows = measurement_table(records)
    assert len(rows) >= 2
    assert all(len(row) == len(columns) for row in rows)
    result = {
        "ok": True, "manifest": str(manifest), "resources": plan,
        "wall_seconds": elapsed,
        "sum_group_seconds": sum(record["elapsed_seconds"] for record in records),
        "records": [{key: record[key] for key in
                     ("name", "status", "elapsed_seconds", "summary_path")} for record in records],
        "rows": rows, "original_inputs_unchanged": True,
    }
    (output / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
