"""Reprocess current real groups and trace the reported pre-fusion strips."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from pointcloud_app.native.backend import Backend, config_from_dict, sha256
from pointcloud_app.pipeline import (
    _surface_support_inlier_mask, read_point_cloud, vertical_plane_basis,
)


def transformed(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def black_region(points):
    # Measured in the reported single-view frame, solely for test assertions.
    return (
        (points[:, 0] > -17) & (points[:, 0] < -1.5)
        & (points[:, 1] > 6.) & (points[:, 1] < 7.)
        & (points[:, 2] < -4.)
        & (points[:, 2] > -4.85 - (points[:, 0] + 17.) * .06)
    )


def trace_strips(cache, before_summary, after_summary, output):
    baseline = json.loads((cache / "baseline.json").read_text(encoding="utf-8"))
    local_level = np.load(cache / "leveling.npy")
    source = baseline["views"][0]["file"]
    name = Path(source).name
    data = np.load(cache / "view_0.npz")
    points, before = data["points"], data["before"]
    denoise = config_from_dict(before_summary).denoise
    after = _surface_support_inlier_mask(
        points, denoise.voxel_size, denoise.surface_support_neighbors,
        denoise.surface_support_distance_factor,
    )
    local = transformed(points, local_level)
    marked = black_region(local) & before
    assert marked.sum() > 150
    assert not np.any(after & marked)
    panels = []
    reports = []
    normal, h, v = vertical_plane_basis(
        before_summary["config"]["section"]["angle_degrees"],
        before_summary["config"]["section"]["tilt_degrees"],
    )
    noise = points[marked]
    for summary, label in ((before_summary, "Fused cloud before update"), (after_summary, "Fused cloud after update")):
        metric = next(row for row in summary["registration_metrics"] if row["file_name"] == name)
        matrix = np.array(summary["leveling"]["transformation"]) @ np.array(metric["transformation"])
        world_noise = transformed(noise, matrix)
        merged = np.asarray(read_point_cloud(Path(summary["output_files"]["merged_cloud"])).points)
        nearest, _ = cKDTree(merged).query(world_noise)
        report = dict(label=label, source_noise_points=len(noise),
                      merged_matches_005=int((nearest <= .05).sum()),
                      nearest_min=float(nearest.min()), nearest_median=float(np.median(nearest)))
        reports.append(report)
        panels.append((merged, world_noise, label))
    assert reports[0]["merged_matches_005"] >= len(noise) * .95
    assert reports[1]["merged_matches_005"] == 0
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, mask, label in zip(axes[0], (before, after), ("Single view before", "Single view after")):
        subset = mask & (local[:, 0] > -17) & (local[:, 0] < -1.5) & (local[:, 1] > 5.8) & (local[:, 1] < 7.)
        p = local[subset]
        ax.scatter(p[:, 0], p[:, 2], c=p[:, 2], cmap="turbo", s=2, vmin=-7, vmax=-3.5)
        ax.set(xlim=(-17, -1.5), ylim=(-6.8, -3.8), xlabel="X", ylabel="Z",
               title=f"{label}: {int((mask & marked).sum())} marked strip points")
        ax.grid(alpha=.2)
    limits = None
    for ax, (merged, world_noise, label) in zip(axes[1], panels):
        profile = np.column_stack((merged @ h, merged @ v))
        noise_profile = np.column_stack((world_noise @ h, world_noise @ v))
        if limits is None:
            limits = (noise_profile.min(axis=0) - [.65, .65], noise_profile.max(axis=0) + [.65, .65])
        ax.scatter(profile[:, 0], profile[:, 1], c=merged[:, 2], cmap="turbo", s=.4)
        ax.set(xlim=(limits[0][0], limits[1][0]), ylim=(limits[0][1], limits[1][1]),
               xlabel="Section horizontal", ylabel="Section vertical", title=label)
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    fig.savefig(output / "reported_strips_comparison.png", dpi=155)
    plt.close(fig)
    result = dict(file=source, single_view_before=int(marked.sum()), single_view_after=int((after & marked).sum()),
                  trace=reports, baseline_cache=str(cache.resolve()))
    (output / "strip_trace.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*", default=["2"])
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/strip_noise_verification/20260915_173400_414148")
    args = parser.parse_args()
    destination = ROOT / "tmp/strip_pipeline_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    profile = ROOT / "settings/workflow_profile.json"
    payload = json.loads(profile.read_text(encoding="utf-8"))
    dxf = Path(payload["cad"]["dxf"])
    source_hashes = {str(p): sha256(p) for p in (profile, dxf)}
    code_hashes = {str(p): sha256(p) for p in (
        ROOT / "pointcloud_app/pipeline.py", ROOT / "pointcloud_app/native/backend.py",
        ROOT / "pointcloud_app/models.py", ROOT / "media_analysis_core.py",
    )}
    rows = []
    print(destination, flush=True)
    for group in args.groups:
        old_path = sorted((ROOT / f"outputs/group_{group}").glob("20260915_172*/run_summary.json"))[-1]
        old = json.loads(old_path.read_text(encoding="utf-8"))
        assert old["config"] == payload["config"], "Baseline/config mismatch"
        config = config_from_dict(payload)
        files = [Path(p) for p in old["input_files"]]
        old_report_path = sorted(old_path.parent.glob("native_cad/*/report.json"))[-1]
        old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
        assert old_report["desktop_input"] == dict(
            dxf_sha256=sha256(dxf), step=payload["cad"]["step"],
            selection=payload["cad"]["selection"],
        ), "Baseline/CAD input mismatch"
        for path in (*files, old_path, old_report_path, Path(old_report["source_dxf"]),
                     Path(old["output_files"]["merged_cloud"]), Path(old["output_files"]["section_cloud"])):
            source_hashes[str(path)] = sha256(path)
        backend = Backend()
        result = backend.process(files, config, lambda message: print(f"[{group}] {message}", flush=True), destination)
        frozen = sha256(result.section_cloud_path)
        selection = payload["cad"]["selection"]
        report = backend.analyze(result, Path(payload["cad"]["dxf"]), selection, payload["cad"]["step"],
                                 lambda message: print(f"[{group}] {message}", flush=True))
        assert sha256(result.section_cloud_path) == frozen
        current = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert current["config"] == config.as_dict()
        final_cleanup = current["section_consensus"]["final_component_cleanup"]
        assert final_cleanup["removed_points"] >= 0
        assert not current["config"]["section"]["use_statistical_filter"]
        assert not current["config"]["section"]["use_component_filter"]
        row = dict(group=group, baseline=str(old_path), summary=str(result.summary_path),
                   report=str(Path(report["source_dxf"]).parent / "report.json"),
                   before_merged=old["merged_points"], after_merged=result.merged_points,
                   before_section=old["section_points"], after_section=result.section_points,
                   before_p95=old_report["fit"]["p95_residual"], after_p95=report["fit"]["p95_residual"],
                   before_quality=old_report["measurement_quality"]["status"],
                   after_quality=report["measurement_quality"]["status"],
                   before_coverage=old_report["measurements"]["measurement_path_1"]["coverage_ratio"],
                   after_coverage=report["measurements"]["measurement_path_1"]["coverage_ratio"],
                   quality_reasons=report["measurement_quality"]["insufficient_data_reasons"],
                   section_integrity=current["section_integrity"],
                   before_denoise=old["denoise_stats"], after_denoise=current["denoise_stats"],
                   frozen_section_unchanged=True)
        if group == "2":
            row["reported_strips"] = trace_strips(args.cache, old, current, destination)
        rows.append(row)
        assert all(sha256(Path(path)) == digest for path, digest in source_hashes.items())
        assert all(sha256(Path(path)) == digest for path, digest in code_hashes.items())
        (destination / "verification.json").write_text(json.dumps(dict(
            groups=rows, source_hashes=source_hashes, code_hashes=code_hashes,
            source_files_unchanged=True,
            final_fusion_component_cleanup=True,
        ), indent=2), encoding="utf-8")
        print(json.dumps({k: value for k, value in row.items() if k not in
                          {"section_integrity", "before_denoise", "after_denoise"}}), flush=True)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
