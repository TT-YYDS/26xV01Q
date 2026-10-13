"""Run identical settings on real groups and save 3D before/after evidence."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import copy
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app import pipeline
from pointcloud_app.native.backend import default_config
from pointcloud_app.profiles import classify_group_family
from pointcloud_app.overlap_registration import fixture_component_mask


def shared_error(before, after, gate):
    old, new = [], []
    for i in range(len(before)):
        for j in range(i):
            for a, b in ((i, j), (j, i)):
                if not len(before[a]) or not len(before[b]):
                    continue
                d0 = cKDTree(before[b]).query(before[a])[0]
                mask = d0 < gate
                if mask.sum() < 100:
                    continue
                d1 = cKDTree(after[b]).query(after[a])[0]
                old.extend(d0[mask])
                new.extend(np.minimum(d1[mask], gate))
    if not old:
        return {"points": 0}
    return {"points": len(old), "rmse_before": float(np.sqrt(np.mean(np.square(old)))),
            "rmse_after": float(np.sqrt(np.mean(np.square(new)))),
            "p90_before": float(np.quantile(old, .9)),
            "p90_after": float(np.quantile(new, .9))}


def verify(group):
    group_path = next((p for p in (ROOT / "media").rglob(group)
                       if p.is_dir() and list(p.glob("*.ply"))),
                      ROOT / "media" / group)
    root = ROOT / "tmp/overlap_fusion_verification" / group
    root.mkdir(parents=True, exist_ok=True)
    config = default_config(group)
    config.compute_backend = "cpu"
    captured = {}
    original = pipeline._refine_group_overlap

    def capture(clouds, transforms, metrics, cfg, callback, name):
        updated, report = original(clouds, transforms, metrics, cfg, callback, name)
        before, after, legacy = [], [], []
        for cloud, old, new, metric in zip(clouds, transforms, updated, metrics):
            if not metric.included_in_fusion:
                continue
            sample = cloud.voxel_down_sample(cfg.registration.workpiece_graph_voxel_size)
            old_cloud = copy.deepcopy(sample).transform(old)
            points = np.asarray(old_cloud.points)
            mask = fixture_component_mask(points, cfg.calibration,
                                          cfg.registration.workpiece_graph_voxel_size * 2)
            before.append(points[~mask].copy())
            delta = new @ np.linalg.inv(old)
            after.append(before[-1] @ delta[:3, :3].T + delta[:3, 3])
            old_fusion = pipeline.remove_calibration_region(sample, cfg.calibration)
            old_fusion.transform(old)
            legacy.append(np.asarray(old_fusion.points).copy())
        captured.update(before=before, after=after, legacy=legacy, refinement=report)
        return updated, report

    start = time.monotonic()
    try:
        with patch.object(pipeline, "_refine_group_overlap", side_effect=capture):
            result = pipeline.process_group(
                group, sorted(group_path.glob("*.ply")), config, root,
                lambda message: print(message.encode("ascii", "backslashreplace").decode(), flush=True))
        error = shared_error(captured["before"], captured["after"],
                             config.registration.workpiece_graph_max_correspondence_distance)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        for row, (label, data) in enumerate((
                ("Before fixture cleanup / local refinement", captured["legacy"]),
                ("Fixture removed, before refinement", captured["before"]),
                ("Fixture removed, after refinement", captured["after"]))):
            for col, (x, y) in enumerate(((0, 1), (0, 2), (1, 2))):
                ax = axes[row, col]
                for index, points in enumerate(data):
                    ax.scatter(points[:, x], points[:, y], s=.3, rasterized=True,
                               color=plt.get_cmap("tab10")(index))
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"{label}\n{'XYZ'[x]} / {'XYZ'[y]}", fontsize=10)
                ax.grid(alpha=.2)
        # Use identical limits for the two geometry-only comparisons.
        for col in range(3):
            for row in (1, 2):
                axes[row, col].set_xlim(axes[0, col].get_xlim())
                axes[row, col].set_ylim(axes[0, col].get_ylim())
        fig.tight_layout()
        image = root / "comparison.png"
        fig.savefig(image, dpi=120)
        plt.close(fig)
        report = {"group": group, "status": "ok", "seconds": time.monotonic() - start,
                  "shared_overlap": error, "refinement": captured["refinement"],
                  "summary": str(result.summary_path), "comparison": str(image),
                  "config": config.as_dict()}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report = {"group": group, "status": "failed", "error": str(exc),
                  "seconds": time.monotonic() - start, "config": config.as_dict()}
    (root / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("VERIFIED " + json.dumps({k:v for k,v in report.items()
                                   if k not in ("config", "refinement")}, ensure_ascii=True), flush=True)
    return report


if __name__ == "__main__":
    groups = sys.argv[1:] or [
        path.name for family in sorted((ROOT / "media").iterdir())
        if family.is_dir() for path in sorted(family.iterdir())
        if path.is_dir() and list(path.glob("*.ply"))
    ]
    rows = [verify(group) for group in groups]
    sys.exit(int(any(row["status"] != "ok" for row in rows)))
