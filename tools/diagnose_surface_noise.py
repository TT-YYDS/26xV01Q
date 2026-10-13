"""Inspect single-view density and geometry in a saved section frame."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from pointcloud_app.native.backend import config_from_dict
from pointcloud_app.pipeline import denoise_cloud, read_point_cloud, vertical_plane_basis, select_vertical_section_indices


def main():
    summary_path = Path(sys.argv[1])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = config_from_dict(summary)
    output = ROOT / "tmp/surface_noise_diagnostic" / summary_path.parent.name
    output.mkdir(parents=True, exist_ok=True)
    level = np.array(summary["leveling"]["transformation"])
    metrics = {m["file_name"]: m for m in summary["registration_metrics"]}
    _, horizontal, vertical = vertical_plane_basis(config.section.angle_degrees, config.section.tilt_degrees)
    fig, axes = plt.subplots(6, 3, figsize=(18, 24), constrained_layout=True)
    density_fig, density_axes = plt.subplots(6, 2, figsize=(14, 24), constrained_layout=True)
    for index, name in enumerate(summary["input_files"]):
        path = Path(name)
        cached = output / f"view_{index}.npz"
        if cached.exists():
            data = np.load(cached)
            points = data["points"]
        else:
            cloud, _ = denoise_cloud(read_point_cloud(path), config.denoise, path.name)
            points = np.array(cloud.points)
            np.savez_compressed(cached, points=points)
        tree = cKDTree(points)
        density_path = output / f"density_{index}.npz"
        if density_path.exists():
            density = np.load(density_path)
            support, ratio = density["support"], density["ratio"]
        else:
            raw = np.asarray(read_point_cloud(path).points)
            support = cKDTree(raw).query_ball_point(points, .12, return_length=True, workers=4)
            _, near = tree.query(points, k=32, workers=4)
            ratio = support / np.maximum(1., np.median(support[near], axis=1))
            np.savez_compressed(density_path, support=support, ratio=ratio)
        transform = level @ np.array(metrics[path.name]["transformation"])
        world = points @ transform[:3, :3].T + transform[:3, 3]
        section_indices = select_vertical_section_indices(world, config.section)
        section = world[section_indices]
        distances, neighbors = tree.query(points[section_indices], k=25)
        local = points[neighbors]
        centered = local - local.mean(axis=1, keepdims=True)
        eigenvalues = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", centered, centered) / 25)
        count = tree.query_ball_point(points[section_indices], .2, return_length=True)
        projected = np.column_stack((section @ horizontal, section @ vertical))
        for ax, values, title in zip(density_axes[index], (support[section_indices], ratio[section_indices]),
                                     ("Raw support r=.12", "Relative raw support")):
            image = ax.scatter(*projected.T, c=values, s=7, cmap="turbo",
                               vmin=0, vmax=60 if title.startswith("Raw") else 1.2)
            density_fig.colorbar(image, ax=ax, fraction=.03)
            ax.set_aspect("equal")
            ax.set_title(f"View {index + 1}: {title}")
            ax.grid(alpha=.2)
        for ax, values, title in zip(axes[index], (count, eigenvalues[:, 1] / eigenvalues[:, 2],
                                                     eigenvalues[:, 0] / eigenvalues.sum(axis=1)),
                                     ("Neighbors r=.2", "Surface width", "Scatter")):
            image = ax.scatter(*projected.T, c=values, s=5, cmap="turbo")
            fig.colorbar(image, ax=ax, fraction=.03)
            ax.set_aspect("equal")
            ax.set_title(f"View {index + 1}: {title}")
            ax.grid(alpha=.2)
        print(index, len(points), len(section), flush=True)
    fig.savefig(output / "sections.png", dpi=120)
    plt.close(fig)
    density_fig.savefig(output / "density.png", dpi=120)
    plt.close(density_fig)
    print(output, flush=True)


if __name__ == "__main__":
    main()
