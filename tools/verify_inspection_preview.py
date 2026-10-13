"""Reproduce the selected-view ROI exemption using the reported PLY."""
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree

from pointcloud_app.native.backend import Backend, config_from_dict, sha256
from pointcloud_app.pipeline import build_preview


def main():
    path = ROOT / "media/c1/2/5.ply"
    profile = ROOT / "settings/workflow_profile.json"
    destination = ROOT / "tmp/inspection_preview_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    hashes = {str(p): sha256(p) for p in (path, profile)}
    config = config_from_dict(json.loads(profile.read_text(encoding="utf-8")))
    _, before = Backend().preview([path], config, 1, True, lambda text: print(text, flush=True),
                                  protect_calibration=True)
    print("ROI-protected preview:", len(before.points), flush=True)
    after = build_preview([path], .1, 100_000, config.compute_backend, config.denoise, True, before.leveling)
    distance, _ = cKDTree(after.points).query(before.points)
    removed = before.points[distance > .2]
    roi = config.calibration.bounds
    inverse = np.linalg.inv(before.leveling.transformation)
    raw_removed = removed @ inverse[:3, :3].T + inverse[:3, 3]
    in_roi = np.all((raw_removed >= roi.min_bound) & (raw_removed <= roi.max_bound), axis=1)
    np.savez_compressed(destination / "preview.npz", before=before.points, after=after.points,
                        removed=removed, raw_removed=raw_removed, leveling=before.leveling.transformation)
    print("Removal candidates:", len(removed), "in ROI:", int(in_roi.sum()), flush=True)
    print("Before bounds:", before.points.min(axis=0), before.points.max(axis=0), flush=True)
    print("After bounds:", after.points.min(axis=0), after.points.max(axis=0), flush=True)
    print("Leveling:", before.leveling.transformation.tolist(), flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    lo, hi = before.points[:, 2].min(), before.points[:, 2].max()
    for row, (name, points) in enumerate((("ROI exemption (before)", before.points), ("Inspection denoise (after)", after.points))):
        for column, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = axes[row, column]
            ax.scatter(points[:, a], points[:, b], c=points[:, 2], cmap="turbo", vmin=lo, vmax=hi, s=.5)
            ax.set(xlabel="XYZ"[a], ylabel="XYZ"[b], title=name)
            ax.set_aspect("equal")
            ax.set_xlim(before.points[:, a].min() - 1, before.points[:, a].max() + 1)
            ax.set_ylim(before.points[:, b].min() - 1, before.points[:, b].max() + 1)
            ax.grid(alpha=.2)
    fig.savefig(destination / "comparison.png", dpi=150)
    plt.close(fig)
    report = dict(
        source_file=str(path), profile=str(profile), config=config.as_dict(),
        source_hashes=hashes, source_files_unchanged=all(sha256(Path(p)) == h for p, h in hashes.items()),
        before=dict(input=before.input_points, after_denoise=before.after_denoise_points, displayed=len(before.points)),
        after=dict(input=after.input_points, after_denoise=after.after_denoise_points, displayed=len(after.points)),
        leveling=asdict(before.leveling), removed_farther_than_02=len(removed), removed_in_roi=int(in_roi.sum()),
    )
    cluster_min, cluster_max = np.array([8, 1.5, 2.2]), np.array([11, 4, 5])
    def cluster(points):
        return np.all((points >= cluster_min) & (points <= cluster_max), axis=1)
    report["circled_cluster"] = dict(
        min_bound=cluster_min.tolist(), max_bound=cluster_max.tolist(),
        before=int(cluster(before.points).sum()), after=int(cluster(after.points).sum()),
    )
    assert report["circled_cluster"]["before"] > 50
    assert report["circled_cluster"]["after"] == 0
    assert report["source_files_unchanged"]
    # This box measures the reported artifact only; it is never a filter rule.
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    for ax, label, data in zip(axes, ("Before", "After"), (before, after)):
        points = data.points
        points = points[(points[:, 0] >= 8) & (points[:, 0] <= 11)]
        ax.scatter(points[:, 1], points[:, 2], c=points[:, 2], cmap="turbo", vmin=lo, vmax=hi, s=3)
        ax.add_patch(Rectangle((1.5, 2.2), 2.5, 2.8, fill=False, edgecolor="#d55e00", linewidth=1.5))
        count = int(cluster(data.points).sum())
        ax.set(xlim=(.5, 5.5), ylim=(-1.5, 5.5), xlabel="Y", ylabel="Z",
               title=f"{label}: {count} isolated preview points")
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    fig.savefig(destination / "circled_cluster.png", dpi=160)
    plt.close(fig)
    (destination / "baseline.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.tolist()), encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
