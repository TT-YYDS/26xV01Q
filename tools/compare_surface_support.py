"""Compare single-view filtering at identical saved poses, without CAD."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app.native.backend import config_from_dict
from pointcloud_app.pipeline import _surface_support_inlier_mask, vertical_plane_basis, select_vertical_section_indices


def main():
    summary_path = ROOT / "outputs/group_1/20260915_100143_452856/run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = config_from_dict(summary)
    directory = ROOT / "tmp/surface_noise_diagnostic" / summary_path.parent.name
    level = np.array(summary["leveling"]["transformation"])
    metrics = {m["file_name"]: m for m in summary["registration_metrics"]}
    _, h, v = vertical_plane_basis(config.section.angle_degrees, config.section.tilt_degrees)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    cloud_fig, cloud_axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    rows = []
    for index, name in enumerate(summary["input_files"]):
        points = np.load(directory / f"view_{index}.npz")["points"]
        keep = _surface_support_inlier_mask(points, config.denoise.voxel_size)
        t = level @ np.array(metrics[Path(name).name]["transformation"])
        world = points @ t[:3, :3].T + t[:3, 3]
        indices = select_vertical_section_indices(world, config.section)
        section_keep = keep[indices]
        section = world[indices]
        projected = np.column_stack((section @ h, section @ v))
        for col, p in enumerate((projected, projected[section_keep])):
            axes[0, col].scatter(*p.T, s=3, label=f"View {index + 1}")
            axes[1, col].scatter(*p.T, s=5)
        broad = np.column_stack((world @ h, world @ v))
        for ax, p in zip(cloud_axes, (broad, broad[keep])):
            ax.scatter(*p[::3].T, s=.1)
        row = dict(view=index + 1, before=len(points), after=int(keep.sum()),
                   section_before=len(indices), section_after=int(section_keep.sum()))
        rows.append(row)
        print(row, flush=True)
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.set_xlim(1.7, 6.5)
        ax.set_ylim(-2.5, 1.8)
    for col, name in enumerate(("Before", "Surface support")):
        axes[0, col].set_title(name)
        axes[1, col].set_title(name + ": detail")
    axes[0, 1].legend(markerscale=3)
    for ax, name in zip(cloud_axes, ("Before (all depths)", "After (all depths)")):
        ax.set_aspect("equal")
        ax.set_xlim(1.7, 6.5)
        ax.set_ylim(-2.5, 1.8)
        ax.set_title(name)
        ax.grid(alpha=.2)
    fig.savefig(directory / "surface_comparison.png", dpi=140)
    cloud_fig.savefig(directory / "surface_cloud_comparison.png", dpi=140)
    plt.close("all")
    (directory / "surface_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
