"""Render actual saved fusion outputs and summarize same-parameter verification."""
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "tmp" / "overlap_fusion_verification"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def points(summary):
    cloud = o3d.io.read_point_cloud(summary["output_files"]["merged_cloud"])
    cloud = cloud.voxel_down_sample(.08)
    cloud.transform(np.linalg.inv(np.asarray(summary["leveling"]["transformation"])))
    return np.asarray(cloud.points)


def main():
    rows = []
    for path in sorted(REPORTS.glob("*/verification.json")):
        data = read(path)
        summary = read(Path(data["summary"])) if data["status"] == "ok" else {}
        rows.append({
            "group": data["group"], "execution_status": data["status"],
            "created_at": summary.get("created_at"),
            "summary": data.get("summary"),
            "included_views": summary.get("fused_view_count"),
            "input_views": len(summary.get("input_files", [])),
            "local_overlap_only": data.get("shared_overlap", {}),
            "multiview_pose_recovery": summary.get("overlap_refinement", {}).get(
                "multiview_pose_recovery", []),
            "parameters": {key: data["config"][key]
                           for key in ("denoise", "registration", "fusion_voxel_size")},
        })
    common = bool(rows) and all(row["parameters"] == rows[0]["parameters"] for row in rows)
    for row in rows:
        del row["parameters"]
    report = {
        "identical_algorithm_parameters": common,
        "note": "Execution success and local-overlap RMSE are not global geometric acceptance.",
        "groups": rows,
    }
    (REPORTS / "aggregate_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    old_path = ROOT / "outputs/group_1/20260909_175927_914691/run_summary.json"
    new_path = Path(read(REPORTS / "1/verification.json")["summary"])
    old, new = read(old_path), read(new_path)
    if old["input_files"][0] != new["input_files"][0]:
        raise ValueError("Reference frames differ; an explicit frame transform is required.")
    arrays = [points(old), points(new)]
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for row, (x, y) in enumerate(((0, 1), (0, 2), (1, 2))):
        extent = np.vstack(arrays)[:, [x, y]]
        lower, upper = extent.min(axis=0), extent.max(axis=0)
        padding = np.maximum((upper - lower) * .04, .1)
        for col, (label, cloud) in enumerate(zip(("Original", "Revised"), arrays)):
            ax = axes[row, col]
            ax.scatter(cloud[:, x], cloud[:, y], s=.25, c=cloud[:, 0],
                       cmap="turbo", vmin=-25, vmax=20, rasterized=True)
            ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label}: {'XYZ'[x]} / {'XYZ'[y]}")
            ax.grid(alpha=.2)
    fig.suptitle("第 1 组：实际融合输出，同一参考坐标系与显示范围", fontproperties=font)
    fig.tight_layout()
    destination = REPORTS / "group1_original_vs_revised.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, row in zip(axes.flat, rows):
        if row["execution_status"] != "ok":
            ax.set_title(row["group"], fontproperties=font)
            continue
        sample = points(read(Path(row["summary"])))
        ax.scatter(sample[:, 1], sample[:, 2], s=.2, c=sample[:, 0], cmap="turbo")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(row["group"], fontproperties=font)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    fig.tight_layout()
    gallery = REPORTS / "all_groups_side_review.png"
    fig.savefig(gallery, dpi=150)
    plt.close(fig)
    print(json.dumps({"identical_parameters": common, "groups": len(rows),
                      "comparison": str(destination), "gallery": str(gallery)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
