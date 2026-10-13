"""Resource-bounded automatic batches, isolated from the desktop and its cache."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import shutil
from hashlib import sha256
from statistics import median
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def resource_plan(groups, *, gpu=False, cpu_count=None, available_bytes=None):
    if cpu_count is None or available_bytes is None:
        try:
            import psutil
            if cpu_count is None:
                cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 1
            if available_bytes is None:
                available_bytes = psutil.virtual_memory().available
        except ImportError:
            cpu_count = cpu_count or os.cpu_count() or 1
            available_bytes = available_bytes or 0
    cpu_budget = max(1, cpu_count - 1)
    largest_input = max(
        (sum(Path(path).stat().st_size for path in files) for files in groups), default=0
    )
    # Raw arrays, neighbor trees, denoised copies and rendering coexist in a worker.
    # Keep headroom for the desktop and OS, but do not collapse ordinary
    # multi-group CPU batches to one worker merely because the machine has
    # less than 8 GB free. The worker itself still has an input-size-based cap.
    bytes_per_worker = max(1536 * 1024**2, largest_input * 12 + 384 * 1024**2)
    memory_workers = max(1, int(available_bytes * 0.8) // bytes_per_worker)
    workers = max(1, min(len(groups), 4, max(1, cpu_budget // 2), memory_workers))
    if gpu:
        workers = 1
    return {
        "workers": workers,
        "threads_per_worker": max(1, cpu_budget // workers),
        "cpu_budget": cpu_budget,
        "available_bytes": available_bytes,
        "estimated_bytes_per_worker": bytes_per_worker,
        "backend": "gpu" if gpu else "cpu",
    }


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def measurement_table(records):
    columns = ["组 / 批次", "质量", "测量路径", "平均 X 偏差",
               "平均 Y 偏差", "角度差 (°)"]
    rows = []

    def number(value):
        return f"{value:.5g}" if isinstance(value, (float, int)) else ""

    for record in records:
        name = record["name"]
        if record.get("batch_id"):
            name += f"\n{record['batch_id']}"
        if record.get("error"):
            status = "已取消" if record.get("status") == "cancelled" else "失败"
            rows.append([name, status, record["error"], "", "", ""])
            continue
        report = record.get("report") or {}
        quality = report.get("measurement_quality", {}).get("status", "INSUFFICIENT_DATA")
        measurements = report.get("measurements", {})
        if not measurements:
            rows.append([name, quality, "无测量路径结果", "", "", ""])
        for measurement in measurements.values():
            rows.append([
                name, quality, measurement.get("label"),
                *[number(measurement.get(key)) for key in (
                    "mean_signed_x", "mean_signed_y",
                    "signed_angle_difference_deg")],
            ])
    return columns, rows


def load_duration_history(output_root, *, limit=40):
    """Read recent completed group durations for a conservative ETA baseline."""
    root = Path(output_root)
    if not root.is_dir():
        return []
    candidates = sorted(
        root.rglob("run_summary.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    durations = []
    for path in candidates[:limit]:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        automatic = summary.get("automatic_workflow") or {}
        timing = (summary.get("desktop_acquisition") or {}).get("timing") or {}
        value = automatic.get("elapsed_seconds", timing.get("total_seconds"))
        if isinstance(value, (int, float)) and value > 0:
            durations.append(float(value))
    return durations


def estimate_batch_seconds(total, completed, workers, samples, elapsed_seconds=0):
    """Return a median-based total and remaining estimate, or None without evidence."""
    if total <= 0 or completed < 0 or completed > total or workers <= 0:
        raise ValueError("批次估算参数无效")
    valid = [float(value) for value in samples if isinstance(value, (int, float)) and value > 0]
    if not valid:
        return None
    per_group = median(valid)
    remaining_groups = total - completed
    remaining_seconds = (remaining_groups + workers - 1) // workers * per_group
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    return {
        "per_group_seconds": per_group,
        "remaining_seconds": float(remaining_seconds),
        "total_seconds": float(elapsed_seconds + remaining_seconds),
    }


def _station_metadata_signature(files):
    if not files:
        return None
    path = Path(files[0]).parent / "station_order.json"
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_verified_pose_reference(output_root, config, files):
    """Only verified, same-family history may initialize a new batch."""
    if not files:
        return None, None
    import numpy as np
    from .registration_reference import config_signature, registration_verified

    family = Path(files[0]).resolve().parent.parent
    signature = config_signature(config)
    root = Path(output_root).resolve()
    for path in sorted((root / "automatic").glob("*/pose_reference.json"), reverse=True):
        try:
            reference = json.loads(path.read_text(encoding="utf-8"))
            if (reference["version"] != 1 or reference["config_signature"] != signature
                    or Path(reference["files"][0]).resolve().parent.parent != family):
                continue
            source = Path(reference["source_summary"]).resolve()
            if not source.is_relative_to(root):
                continue
            summary = json.loads(source.read_text(encoding="utf-8"))
            if (reference["files"] != summary["input_files"]
                    or not registration_verified(summary)):
                continue
            matrices = [m["transformation"] for m in summary["registration_metrics"]]
            if not np.allclose(reference["transforms"], matrices, atol=1e-9, rtol=0):
                continue
            return reference, path
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return None, None


def prepare_batch(groups, config, cad, output_root, *, gpu=False):
    plan = resource_plan([files for _name, files, _signature in groups], gpu=gpu)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = Path(output_root) / "automatic" / batch_id
    directory.mkdir(parents=True)
    dxf = Path(cad["dxf"])
    stamp = (dxf.stat().st_size, dxf.stat().st_mtime_ns)
    frozen_dxf = directory / dxf.name
    shutil.copy2(dxf, frozen_dxf)
    if stamp != (dxf.stat().st_size, dxf.stat().st_mtime_ns):
        raise RuntimeError("DXF 在冻结模板时发生变化，请重新保存模板")
    requests = [
        {
            "id": uuid4().hex,
            "batch_id": batch_id,
            "name": name,
            "files": [str(path) for path in files],
            "signature": signature,
            "config": config.as_dict(),
            "dxf": str(frozen_dxf),
            "selection": cad["selection"],
            "step": cad.get("step", 0.05),
            "output_root": str(output_root),
            "resources": plan,
            "use_pose_reference": True,
            "station_metadata_signature": _station_metadata_signature(files),
        }
        for name, files, signature in groups
    ]
    manifest = directory / "batch.json"
    write_json(manifest, {"batch_id": batch_id, "resources": plan, "requests": requests, "records": []})
    return requests, plan, manifest


def _initialize_worker(events, stop, threads):
    global _events, _stop
    _events, _stop = events, stop
    # The spawn entry point imports the numeric stack only after these limits.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MPLBACKEND"] = "Agg"
    # Batch-level process parallelism already runs several groups at once.
    # Prevent nested per-view pools from oversubscribing each worker.
    os.environ["POINTCLOUD_BATCH_WORKER"] = "1"


def _run_group(request):
    from .native.backend import Backend, Cancelled, acquisition_signature, config_from_dict
    from .registration_reference import registration_verified

    started = perf_counter()
    record = {
        key: request[key] for key in ("id", "batch_id", "name", "files", "signature", "resources")
    }
    record.update(status="completed", summary_path=None, report=None, error=None)
    record["worker_pid"] = os.getpid()

    def progress(message):
        if _stop.is_set():
            raise Cancelled("已取消")
        _events.put({"type": "progress", "name": request["name"], "message": str(message)})

    try:
        progress("开始点云处理")
        files = [Path(path) for path in request["files"]]
        expected = tuple(tuple(item) for item in request["signature"])
        if acquisition_signature(files) != expected:
            raise RuntimeError("输入文件在排队期间发生变化，等待下一批次")
        if ("station_metadata_signature" in request
                and _station_metadata_signature(files) != request["station_metadata_signature"]):
            raise RuntimeError("工位编号在排队期间发生变化，等待下一批次")
        backend = Backend()
        options = {}
        if request.get("use_pose_reference"):
            options = dict(pose_reference=request.get("pose_reference"),
                           capture_pose_reference=True)
        result = backend.process(files, config_from_dict(request["config"]),
                                 progress, Path(request["output_root"]), **options)
        record["summary_path"] = str(result.summary_path)
        record["report"] = backend.analyze(
            result, Path(request["dxf"]), request["selection"], request["step"], progress
        )
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        reference_report = summary.get("pose_reference") or {}
        reused = any(view.get("reused") for view in reference_report.get("views", []))
        if reused and not registration_verified(summary):
            record["pose_reference_attempt"] = {
                "summary_path": str(result.summary_path),
                "seconds": perf_counter() - started,
                "quality": record["report"].get("measurement_quality"),
            }
            progress("位姿参考结果未通过拼接质量检查，回退完整标定块配准")
            result = backend.process(files, config_from_dict(request["config"]),
                                     progress, Path(request["output_root"]))
            record["summary_path"] = str(result.summary_path)
            record["report"] = backend.analyze(
                result, Path(request["dxf"]), request["selection"], request["step"], progress)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        record["pose_reference"] = summary.get("pose_reference")
        if (request.get("use_pose_reference") and not request.get("pose_reference")
                and registration_verified(summary)):
            candidate = summary.get("pose_reference_candidate")
            if candidate:
                candidate["source_summary"] = str(result.summary_path)
                record["learned_pose_reference"] = candidate
    except Cancelled as error:
        record.update(status="cancelled", error=str(error))
    except Exception as error:
        record.update(status="failed", error=str(error))
    record["elapsed_seconds"] = perf_counter() - started
    record["finished_at"] = datetime.now().astimezone().isoformat()
    if record["summary_path"]:
        metadata = {key: value for key, value in record.items()
                    if key not in ("report", "learned_pose_reference")}
        summary_path = Path(record["summary_path"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["automatic_workflow"] = metadata
        write_json(summary_path, summary)
        if record["report"]:
            report = record["report"]
            report["automatic_workflow"] = metadata
            write_json(Path(report["source_dxf"]).parent / "report.json", report)
    return record


def _run_pool(requests, plan, manifest, progress, notify, cancelled, worker,
              reference_state=None):
    """Publish each result immediately; keep completed groups when stopping."""
    context = multiprocessing.get_context("spawn")
    events, stop = context.Queue(), context.Event()
    records = []
    pending_requests = iter(requests)
    futures = {}
    reference_state = reference_state if reference_state is not None else {}
    use_reference = bool(requests and requests[0].get("use_pose_reference"))
    if use_reference:
        identity = json.dumps(
            {"config": requests[0]["config"],
             "source": str(Path(requests[0]["files"][0]).resolve().parent.parent)},
            sort_keys=True,
        )
        if reference_state.get("identity") != identity:
            reference_state.clear()
            reference_state["identity"] = identity
    manifest_data = (
        json.loads(Path(manifest).read_text(encoding="utf-8"))
        if manifest is not None else None
    )
    if use_reference and not reference_state.get("template") and manifest_data is not None:
        from .native.backend import config_from_dict
        template, source = load_verified_pose_reference(
            requests[0]["output_root"], config_from_dict(requests[0]["config"]),
            requests[0]["files"])
        if template:
            reference_state["template"] = template
            reference_path = Path(manifest).with_name("pose_reference.json")
            write_json(reference_path, template)
            manifest_data["pose_reference_path"] = str(reference_path)
            manifest_data["pose_reference_bootstrap_source"] = str(source)
    started = perf_counter()
    if manifest_data is not None:
        manifest_data["started_at"] = datetime.now().astimezone().isoformat()

    def submit_next(executor):
        if cancelled.is_set():
            return
        request = next(pending_requests, None)
        if request is not None:
            if use_reference and reference_state.get("template"):
                request = dict(request, pose_reference=reference_state["template"])
            futures[executor.submit(worker, request)] = request

    def flush_progress():
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                break
            if not cancelled.is_set():
                try:
                    progress(f"自动处理 {event['name']} · {event['message']}")
                except Exception:
                    cancelled.set()
                    stop.set()

    try:
        with ProcessPoolExecutor(
            max_workers=plan["workers"], mp_context=context,
            initializer=_initialize_worker,
            initargs=(events, stop, plan["threads_per_worker"]),
        ) as executor:
            # Learn from the first verified completion without stalling other
            # workers. Queued groups and subsequent batches receive its seed.
            for _ in range(plan["workers"]):
                submit_next(executor)
            while futures:
                if cancelled.is_set():
                    stop.set()
                flush_progress()
                done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    request = futures.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        record = {
                            key: request[key] for key in
                            ("id", "name", "files", "signature", "resources")
                        }
                        if "batch_id" in request:
                            record["batch_id"] = request["batch_id"]
                        record.update(status="failed", error=str(error), report=None,
                                      summary_path=None, elapsed_seconds=None,
                                      finished_at=datetime.now().astimezone().isoformat())
                    learned = record.pop("learned_pose_reference", None)
                    if use_reference and learned and not reference_state.get("template"):
                        reference_state["template"] = learned
                        if manifest_data is not None:
                            reference_path = Path(manifest).with_name("pose_reference.json")
                            write_json(reference_path, learned)
                            manifest_data["pose_reference_path"] = str(reference_path)
                    records.append(record)
                    if manifest_data is not None:
                        manifest_data["records"] = records
                        write_json(manifest, manifest_data)
                    notify(record)
                    # An unsuitable first group must not serialize the whole batch.
                    slots = plan["workers"]
                    for _ in range(max(0, slots - len(futures))):
                        submit_next(executor)
                flush_progress()
    finally:
        stop.set()
        events.close()
        events.join_thread()
        if manifest_data is not None:
            manifest_data["wall_seconds"] = perf_counter() - started
            manifest_data["finished_at"] = datetime.now().astimezone().isoformat()
            manifest_data["cancelled"] = cancelled.is_set()
            write_json(manifest, manifest_data)
    return records


def run_batch(requests, plan, manifest, progress, notify, cancelled, *, reference_state=None):
    return _run_pool(requests, plan, manifest, progress, notify, cancelled, _run_group,
                     reference_state)
