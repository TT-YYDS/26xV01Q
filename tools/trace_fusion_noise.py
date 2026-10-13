"""Trace each denoised source in the saved fusion coordinate frame."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app.native.backend import config_from_dict
from pointcloud_app.pipeline import read_point_cloud, denoise_cloud, remove_calibration_region


def main():
    summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    config = config_from_dict(summary)
    variant = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    if variant == "tight":
        config.denoise.radius = .20
    elif variant == "local":
        config.denoise.radius = .18
        config.denoise.radius_min_points = 10
        config.denoise.secondary_min_points = 16
    output = ROOT / "tmp" / "fusion_noise_trace" / variant
    output.mkdir(parents=True, exist_ok=True)
    level = np.asarray(summary["leveling"]["transformation"])
    metrics = {row["file_name"]: row for row in summary["registration_metrics"]}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    arrays = {}
    for index, file in enumerate(summary["input_files"]):
        path = Path(file)
        cloud, stats = denoise_cloud(read_point_cloud(path), config.denoise, path.name)
        cloud.transform(level @ np.asarray(metrics[path.name]["transformation"]))
        cloud = remove_calibration_region(
            cloud, config.calibration, coordinate_transformation=level,
            component_voxel_size=max(config.fusion_voxel_size * 4, .24))
        points = np.asarray(cloud.points).copy()
        arrays[f"view{index + 1}"] = points
        ax = axes.flat[index]
        ax.scatter(points[:, 1], points[:, 2], s=.25)
        ax.set_title(f"View {index + 1}: {path.stem[10:18]} ({len(points)} points)")
        ax.set_xlim(-2, 14)
        ax.set_ylim(-8, 6)
        ax.set_aspect("equal")
        ax.grid(alpha=.25)
        print(index + 1, stats, flush=True)
    fig.tight_layout()
    fig.savefig(output / "sources.png", dpi=160)
    np.savez_compressed(output / "sources.npz", **arrays)
    baseline_path = output.parent / "baseline" / "sources.npz"
    if not baseline_path.exists():
        baseline_path = output.parent / "sources.npz"
    if variant != "baseline" and baseline_path.exists():
        baseline = np.load(baseline_path)
        report = []
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for index, (key, current) in enumerate(arrays.items()):
            previous = baseline[key]
            distances = cKDTree(current).query(previous)[0]
            removed = distances > 1e-7
            row = {"view": index + 1, "before": len(previous), "after": len(current),
                   "removed": int(removed.sum()),
                   "retained_fraction": float((~removed).mean()),
                   "removed_nearest_retained_p95": float(np.quantile(distances[removed], .95))
                   if removed.any() else 0.}
            report.append(row)
            for ax, points in zip(axes, (previous, current)):
                ax.scatter(points[:, 1], points[:, 2], s=.2, label=f"View {index + 1}")
        for ax, title in zip(axes, ("Before", "After (same poses)")):
            ax.set_title(title)
            ax.set_xlim(3.5, 7.2)
            ax.set_ylim(-2.5, 2.8)
            ax.set_aspect("equal")
            ax.grid(alpha=.25)
        axes[1].legend(markerscale=8)
        fig.tight_layout()
        fig.savefig(output / "focus.png", dpi=160)
        (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
