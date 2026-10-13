"""Compare registration implementations on frozen inputs without changing settings."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import importlib.util
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import open3d as o3d
from pointcloud_app import pipeline
from pointcloud_app.native.backend import Backend, config_from_dict, sha256, acquisition_signature
from pointcloud_app.registration_reference import registration_verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*", default=["1"])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    batch_path = ROOT / "outputs/automatic/20260929_160826_206460/batch.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if args.batch:
        from pointcloud_app.automatic_runtime import prepare_batch, run_batch
        groups = [(r["name"], [Path(f) for f in r["files"]],
                   acquisition_signature([Path(f) for f in r["files"]])) for r in batch["requests"]]
        cad = dict(dxf=ROOT / "test母头.dxf", selection=batch["requests"][0]["selection"], step=.05)
        requests, plan, manifest = prepare_batch(groups,
            config_from_dict(batch["requests"][0]["config"]), cad, ROOT / "outputs")
        plan["workers"] = min(3, plan["workers"])
        plan["threads_per_worker"] = max(1, plan["cpu_budget"] // plan["workers"])
        for request in requests:
            request["resources"] = plan
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.update(resources=plan, requests=requests)
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        protected = [ROOT / "settings/workflow_profile.json", ROOT / "test母头.dxf",
                     *[p for _, files, _ in groups for p in files]]
        hashes = {str(p): sha256(p) for p in protected}
        records = run_batch(requests, plan, manifest, lambda _: None,
            lambda r: print("GROUP", r["name"], r["status"], round(r["elapsed_seconds"], 3), flush=True),
            threading.Event(), reference_state={})
        assert all(sha256(Path(p)) == value for p, value in hashes.items())
        for record in records:
            assert record["status"] == "completed", record.get("error")
            summary = json.loads(Path(record["summary_path"]).read_text(encoding="utf-8"))
            assert summary["fused_view_count"] == len(record["files"])
            assert record["report"]["measurement_quality"]["status"] == "PASS"
            assert all(m["coverage_ratio"] == 1. for m in
                       record["report"]["measurement_quality"]["measurements"])
            print("TIMING", record["name"], summary["timing"], flush=True)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        print("BATCH", manifest, "WALL", payload["wall_seconds"],
              "BOOTSTRAP", payload.get("pose_reference_bootstrap_source"), flush=True)
        return
    destination = ROOT / "tmp/registration_optimization" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    modules = [("optimized", pipeline)]
    if args.baseline:
        path = ROOT / "tmp/optimization_baseline/pipeline_v35.py"
        spec = importlib.util.spec_from_file_location("pointcloud_app._pipeline_baseline", path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        modules.insert(0, ("baseline", baseline))
    reference = (json.loads((batch_path.parent / "pose_reference.json").read_text(encoding="utf-8"))
                 if args.reference else None)
    rows = []
    for group in args.groups:
        record = next(r for r in batch["records"] if r["name"] == group)
        summary = json.loads(Path(record["summary_path"]).read_text(encoding="utf-8"))
        files = [Path(f) for f in summary["input_files"]]
        protected = [ROOT / "settings/workflow_profile.json", ROOT / "test母头.dxf", *files]
        hashes = {str(p): sha256(p) for p in protected}
        for name, module in modules:
            events, gicp = [], {"seconds": 0., "calls": 0}
            native_gicp = o3d.pipelines.registration.registration_generalized_icp

            def measure_gicp(*positional, **keywords):
                started = perf_counter()
                try:
                    return native_gicp(*positional, **keywords)
                finally:
                    gicp["seconds"] += perf_counter() - started
                    gicp["calls"] += 1

            backend = Backend()
            started = perf_counter()
            with patch("pointcloud_app.native.backend.process_group", module.process_group), patch(
                    "open3d.pipelines.registration.registration_generalized_icp", measure_gicp):
                result = backend.process(files, config_from_dict(summary),
                    lambda message: events.append((perf_counter(), str(message))),
                    destination / name, pose_reference=reference)
            processing_seconds = perf_counter() - started
            actual = json.loads(result.summary_path.read_text(encoding="utf-8"))
            phases = {}
            for (begin, message), (end, _) in zip(events, events[1:]):
                if module._processing_stage_for_message(message) != "registration":
                    continue
                phase = ("recovery" if "低重叠位姿" in message else
                         "pose_disambiguation" if "整体朝向" in message else
                         "coarse" if "FPFH + RANSAC" in message else
                         "fixture_icp_and_evaluation" if "ICP 精配准" in message else
                         "pose_graph" if "位姿图" in message else
                         "overlap" if "三维重叠配准" in message else "other")
                phases[phase] = phases.get(phase, 0.) + end - begin
            cad_started = perf_counter()
            report = backend.analyze(result, ROOT / "test母头.dxf",
                record["report"]["desktop_input"]["selection"], .05, lambda _: None)
            assert all(sha256(Path(p)) == digest for p, digest in hashes.items())
            row = dict(group=group, implementation=name,
                implementation_sha256=sha256(Path(module.__file__)), summary=str(result.summary_path),
                processing_seconds=round(processing_seconds, 3),
                cad_seconds=round(perf_counter() - cad_started, 3), timing=actual["timing"],
                registration_phases={k: round(v, 3) for k, v in phases.items()},
                gicp=gicp, fused_views=actual["fused_view_count"],
                pose_reference=actual["pose_reference"],
                recoveries=actual["overlap_refinement"]["multiview_pose_recovery"],
                registration_verified=registration_verified(actual),
                cad_quality=report["measurement_quality"]["status"],
                coverage=[m["coverage_ratio"] for m in report["measurement_quality"]["measurements"]],
                input_hashes=hashes)
            rows.append(row)
            (destination / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: row[k] for k in ("group", "implementation", "processing_seconds",
                "cad_seconds", "timing", "registration_phases", "gicp", "fused_views",
                "cad_quality", "coverage")}), flush=True)
            if name == "optimized":
                assert row["fused_views"] == len(files), row
                assert row["cad_quality"] == "PASS", row
                assert all(v == 1. for v in row["coverage"]), row
    print("COMPARISON", destination / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
