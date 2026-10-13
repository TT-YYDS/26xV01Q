"""Render raw/current denoising evidence for an individual capture."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app.native.backend import default_config
from pointcloud_app.pipeline import read_point_cloud, denoise_cloud


def main():
    path = Path(sys.argv[1])
    config = default_config(path.parent.name)
    raw = read_point_cloud(path)
    previous, previous_stats = denoise_cloud(
        raw, config.denoise, path.name, "cpu",
        (config.calibration.bounds.min_bound, config.calibration.bounds.max_bound))
    cleaned, stats = denoise_cloud(raw, config.denoise, path.name, "cpu")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "tmp" / "denoise_frame_review"
    output.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(raw.points), np.asarray(previous.points), np.asarray(cleaned.points)]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for row, points in enumerate(arrays):
        points = points[::max(1, len(points) // 100000)]
        for col, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = axes[row, col]
            ax.scatter(points[:, a], points[:, b], c=points[:, 2], s=.15)
            ax.set_aspect("equal")
            ax.set_title(f"{('Raw', 'Previous (ROI bypass)', 'Updated')[row]} {'XYZ'[a]}{'XYZ'[b]}")
            ax.set_xlim(arrays[0][:, a].min(), arrays[0][:, a].max())
            ax.set_ylim(arrays[0][:, b].min(), arrays[0][:, b].max())
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    (output / "stats.json").write_text(json.dumps({
        "previous": previous_stats.__dict__, "updated": stats.__dict__,
    }, indent=2), encoding="utf-8")
    print(stats, flush=True)


if __name__ == "__main__":
    main()
