"""Probe cross-view support around the reported group-4 single-view noise."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from pointcloud_app.native.backend import config_from_dict
from pointcloud_app.pipeline import denoise_cloud, read_point_cloud


def main():
    summary_path = Path(sys.argv[1]).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = config_from_dict(summary)
    files = [Path(path) for path in summary["input_files"]]
    level = np.asarray(summary["leveling"]["transformation"], dtype=float)
    transforms = [np.asarray(row["transformation"], dtype=float)
                  for row in summary["registration_metrics"]]
    aligned = []
    for path, transform in zip(files, transforms, strict=True):
        cloud, _ = denoise_cloud(read_point_cloud(path), config.denoise, path.name, "cpu")
        matrix = level @ transform
        points = np.asarray(cloud.points)
        aligned.append(points @ matrix[:3, :3].T + matrix[:3, 3])

    target_index = 0
    others = [points for index, points in enumerate(aligned) if index != target_index]
    other_tree = cKDTree(np.concatenate(others))
    all_other_distances, _ = other_tree.query(aligned[target_index], k=1)
    rows = []
    for threshold in (.08, .12, .16, .20, .28, .40):
        rows.append(dict(threshold=threshold,
                         unsupported=int(np.count_nonzero(all_other_distances > threshold)),
                         supported=int(np.count_nonzero(all_other_distances <= threshold))))
    output = summary_path.parent / "multiview_support_probe"
    output.mkdir(exist_ok=True)
    point_sets = [aligned[index] for index in range(len(aligned))]
    labels = np.zeros(len(aligned[target_index]), dtype=np.int8)
    labels[all_other_distances <= .12] = 1
    labels[(all_other_distances > .12) & (all_other_distances <= .28)] = 2
    labels[all_other_distances > .28] = 3
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), constrained_layout=True)
    for ax, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        ax.scatter(aligned[target_index][:, a], aligned[target_index][:, b],
                   c=labels, cmap="viridis", s=.7, vmin=0, vmax=3)
        ax.set_aspect("equal")
        ax.set(xlabel="XYZ"[a], ylabel="XYZ"[b])
        ax.grid(alpha=.2)
    fig.savefig(output / "view0_cross_support.png", dpi=150)
    plt.close(fig)

    # Report how much nearby other-view coverage exists around unsupported points.
    all_other = np.concatenate(others)
    context_tree = cKDTree(all_other)
    context_counts = np.asarray(context_tree.query_ball_point(
        aligned[target_index], .45, return_length=True,
    ))
    for radius in (.12, .20, .28):
        candidate = all_other_distances > radius
        rows.append(dict(threshold=radius, unsupported_with_context=int(
            np.count_nonzero(candidate & (context_counts >= 8)))))
    result = dict(
        target_file=str(files[target_index]),
        counts=[int(len(points)) for points in aligned],
        nearest_other_quantiles=np.quantile(all_other_distances, [0, .1, .25, .5, .75, .9, .99, 1]).tolist(),
        thresholds=rows,
        context_quantiles=np.quantile(context_counts, [0, .1, .5, .9, .99, 1]).tolist(),
        output=str(output),
    )
    (output / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
