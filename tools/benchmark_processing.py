"""Measure real processing in isolated output folders without editing inputs."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--group", default="2")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--capture-reference", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MPLBACKEND"] = "Agg"

    from pointcloud_app.native.backend import Backend, config_from_dict, sha256
    from pointcloud_app.registration_reference import registration_verified

    profile = ROOT / "settings/workflow_profile.json"
    payload = json.loads(profile.read_text(encoding="utf-8"))
    files = sorted((Path(payload["source"]["path"]) / args.group).glob("*.ply"))
    assert len(files) in (5, 6, 7, 8), files
    dxf = Path(payload["cad"]["dxf"])
    tracked = [profile, dxf, *files]
    hashes = {str(path): sha256(path) for path in tracked}
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (
        ROOT / "pointcloud_app/pipeline.py", ROOT / "pointcloud_app/native/backend.py",
        ROOT / "pointcloud_app/automatic_runtime.py", ROOT / "media_analysis_core.py",
        ROOT / "pointcloud_app/registration_reference.py",
    )}
    destination = ROOT / "tmp/performance_verification" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.label
    )
    destination.mkdir(parents=True)
    print(destination, flush=True)
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    backend = Backend()
    start = perf_counter()
    result = backend.process(files, config_from_dict(payload), lambda text: print(text, flush=True),
                             destination,
                             pose_reference=(json.loads(args.reference.read_text(encoding="utf-8"))
                                             if args.reference else None),
                             capture_pose_reference=args.capture_reference)
    processing_seconds = perf_counter() - start
    start = perf_counter()
    report = backend.analyze(result, dxf, payload["cad"]["selection"],
                             payload["cad"]["step"], lambda text: print(text, flush=True))
    cad_seconds = perf_counter() - start
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(destination / "profile.pstats"))
        with (destination / "profile.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(85)
    assert hashes == {str(path): sha256(path) for path in tracked}
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    candidate = summary.get("pose_reference_candidate")
    if candidate and registration_verified(summary):
        candidate["source_summary"] = str(result.summary_path)
        (destination / "pose_reference.json").write_text(
            json.dumps(candidate, ensure_ascii=False), encoding="utf-8")
    record = {
        "label": args.label, "group": args.group, "threads": args.threads,
        "profiled": args.profile, "processing_seconds": processing_seconds,
        "cad_seconds": cad_seconds, "total_seconds": processing_seconds + cad_seconds,
        "summary_path": str(result.summary_path), "timing": summary["timing"],
        "merged_points": result.merged_points, "section_points": result.section_points,
        "quality": report["measurement_quality"], "fit": report["fit"],
        "measurements": report["measurements"], "input_sha256": hashes,
        "code_sha256": code_hashes,
        "pose_reference": summary.get("pose_reference"),
    }
    (destination / "benchmark.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "processing_seconds": processing_seconds,
                      "cad_seconds": cad_seconds, "points": result.section_points}), flush=True)


if __name__ == "__main__":
    main()
