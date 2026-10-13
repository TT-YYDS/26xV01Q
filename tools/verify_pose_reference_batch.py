"""Compare full, cold-reference and warm-reference spawned automatic batches."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=["1", "2"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["MPLBACKEND"] = "Agg"
    from pointcloud_app.automatic_runtime import prepare_batch, run_batch, write_json
    from pointcloud_app.native.backend import acquisition_signature, config_from_dict, sha256

    profile = ROOT / "settings/workflow_profile.json"
    payload = json.loads(profile.read_text(encoding="utf-8"))
    groups = []
    tracked = [profile, Path(payload["cad"]["dxf"])]
    for name in args.groups:
        files = sorted((Path(payload["source"]["path"]) / name).glob("*.ply"))
        assert 5 <= len(files) <= 8
        groups.append((name, files, acquisition_signature(files)))
        tracked.extend(files)
        metadata = files[0].parent / "station_order.json"
        if metadata.exists():
            tracked.append(metadata)
    hashes = {str(p): sha256(p) for p in tracked}
    code_hashes = {name: sha256(ROOT / name) for name in (
        "pointcloud_app/pipeline.py", "pointcloud_app/registration_reference.py",
        "pointcloud_app/automatic_runtime.py", "pointcloud_app/native/backend.py",
        "pointcloud_app/native/window.py",
    )}
    output = ROOT / "tmp/pose_reference_batch_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print(f"OUTPUT {output}", flush=True)
    state, results = {}, []
    for mode in ("full", "cold", "warm"):
        requests, plan, manifest = prepare_batch(
            groups, config_from_dict(payload), payload["cad"], output / mode)
        plan["workers"] = min(plan["workers"], args.workers)
        plan["threads_per_worker"] = min(plan["threads_per_worker"], args.threads)
        for request in requests:
            request["resources"] = plan
            request["use_pose_reference"] = mode != "full"
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        saved.update(requests=requests, resources=plan)
        write_json(manifest, saved)
        started = perf_counter()
        records = run_batch(requests, plan, manifest,
                            lambda message: print(f"{mode}: {message}", flush=True),
                            lambda record: print(f"DONE {mode} {record['name']} {record['status']}", flush=True),
                            threading.Event(), reference_state=state)
        results.append({
            "mode": mode, "wall_seconds": perf_counter() - started,
            "resources": plan, "manifest": str(manifest),
            "records": [{key: value for key, value in r.items() if key != "report"}
                        for r in records],
            "quality": {r["name"]: (r.get("report") or {}).get("measurement_quality") for r in records},
        })
        unchanged = hashes == {str(p): sha256(p) for p in tracked}
        write_json(output / "verification.json", {
            "results": results, "input_sha256": hashes, "inputs_unchanged": unchanged,
            "code_sha256": code_hashes,
        })
        assert unchanged
        assert len(records) == len(groups) and all(r["status"] == "completed" for r in records)
        print(f"RESULT {mode} {results[-1]['wall_seconds']:.3f}s", flush=True)
    assert code_hashes == {name: sha256(ROOT / name) for name in code_hashes}
    print(f"REPORT {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
