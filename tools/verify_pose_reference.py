"""Paired, isolated real-data checks of the automatic worker's pose shortcut."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["POINTCLOUD_BATCH_WORKER"] = "1"

    from pointcloud_app import automatic_runtime as runtime
    from pointcloud_app.native.backend import acquisition_signature, sha256
    from pointcloud_app.registration_reference import registration_verified
    profile = ROOT / "settings/workflow_profile.json"
    payload = json.loads(profile.read_text(encoding="utf-8"))
    files_by_group = {name: sorted((Path(payload["source"]["path"]) / name).glob("*.ply"))
                      for name in args.groups}
    assert all(5 <= len(files) <= 8 for files in files_by_group.values())
    tracked = [profile, Path(payload["cad"]["dxf"])]
    for files in files_by_group.values():
        tracked.extend(files)
        metadata = files[0].parent / "station_order.json"
        if metadata.exists():
            tracked.append(metadata)
    original_hashes = {str(path): sha256(path) for path in tracked}
    code_hashes = {name: sha256(ROOT / name) for name in (
        "pointcloud_app/pipeline.py", "pointcloud_app/registration_reference.py",
        "pointcloud_app/automatic_runtime.py", "pointcloud_app/native/backend.py",
        "pointcloud_app/native/window.py", "media_analysis_core.py",
    )}
    output = ROOT / "tmp/pose_reference_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print(f"OUTPUT {output}", flush=True)
    runtime._initialize_worker(SimpleNamespace(put=lambda event: print(
        f"{event['name']}: {event['message']}", flush=True)), threading.Event(), args.threads)
    results, template = [], None
    if args.reference:
        template = json.loads(args.reference.read_text(encoding="utf-8"))
        source = json.loads(Path(template["source_summary"]).read_text(encoding="utf-8"))
        assert registration_verified(source), "The supplied seed failed stitching verification."
        assert source["desktop_acquisition"]["input_sha256"] == {
            Path(path).name: sha256(Path(path)) for path in source["input_files"]}
        runtime.write_json(output / "pose_reference.json", template)
    start = perf_counter()
    for repeat in range(args.rounds):
        for name, files in files_by_group.items():
            modes = ("full", "reference") if repeat % 2 == 0 else ("reference", "full")
            for mode in modes:
                request = {
                    "id": f"{repeat}_{name}_{mode}", "batch_id": output.name, "name": name,
                    "files": list(map(str, files)), "signature": acquisition_signature(files),
                    "config": payload["config"], "dxf": payload["cad"]["dxf"],
                    "selection": payload["cad"]["selection"], "step": payload["cad"]["step"],
                    "output_root": str(output / f"{repeat}_{name}_{mode}"),
                    "resources": {"workers": 1, "threads_per_worker": args.threads},
                    "use_pose_reference": True,
                    "station_metadata_signature": runtime._station_metadata_signature(files),
                }
                if mode == "reference" and template:
                    request["pose_reference"] = template
                record = runtime._run_group(request)
                learned = record.pop("learned_pose_reference", None)
                if template is None and learned:
                    template = learned
                    runtime.write_json(output / "pose_reference.json", template)
                summary = (json.loads(Path(record["summary_path"]).read_text(encoding="utf-8"))
                           if record.get("summary_path") else {})
                report = record.pop("report", None) or {}
                record.update(mode=mode, repeat=repeat,
                              processing_timing=summary.get("timing"),
                              quality=report.get("measurement_quality"),
                              measurements=report.get("measurements"),
                              template_available=template is not None)
                results.append(record)
                runtime.write_json(output / "verification.json", {
                    "threads": args.threads, "results": results,
                    "input_sha256": original_hashes, "code_sha256": code_hashes,
                    "wall_seconds": perf_counter() - start,
                    "inputs_unchanged": original_hashes == {str(p): sha256(p) for p in tracked},
                })
                print(f"RESULT {repeat} {name} {mode} {record['status']} "
                      f"{record['elapsed_seconds']:.3f}s "
                      f"{(record['quality'] or {}).get('status')} "
                      f"reuse={sum(v.get('reused', False) for v in (record.get('pose_reference') or {}).get('views', []))}",
                      flush=True)
                assert record["status"] == "completed", record.get("error")
                assert original_hashes == {str(p): sha256(p) for p in tracked}
    assert code_hashes == {name: sha256(ROOT / name) for name in code_hashes}
    print(f"REPORT {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
