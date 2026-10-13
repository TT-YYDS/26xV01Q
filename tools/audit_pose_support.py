"""Measure leave-one-view-out workpiece support of saved poses."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app import pipeline as p
from pointcloud_app.native.backend import config_from_dict


def main():
    summary_path = Path(sys.argv[1])
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    print("GROUP", saved["group_name"].encode("unicode_escape").decode(), flush=True)
    config = config_from_dict(saved)
    voxel = config.registration.registration_voxel_size
    aligned, native_bodies = {}, {}
    for index, path in enumerate(saved["input_files"]):
        metric = saved["registration_metrics"][index]
        if not metric["included_in_fusion"]:
            continue
        c, _ = p.denoise_cloud(
            p.read_point_cloud(Path(path)), config.denoise, Path(path).name,
            protected_bounds=(config.calibration.bounds.min_bound,
                              config.calibration.bounds.max_bound))
        pts = np.asarray(c.voxel_down_sample(voxel).points)
        mask = p.fixture_component_mask(pts, config.calibration, voxel * 2)
        transform = np.asarray(metric["transformation"])
        pts = pts @ transform[:3, :3].T + transform[:3, 3]
        if not mask.any():
            mask = p.fixture_component_mask(pts, config.calibration, voxel * 2)
        native_bodies[index] = pts[~mask]
        aligned[index] = pts
    rows = []
    for index, source in native_bodies.items():
        target = np.vstack([pts for peer, pts in aligned.items() if peer != index])
        target = np.asarray(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(target)).voxel_down_sample(voxel).points)
        target = target[~p.fixture_component_mask(target, config.calibration, voxel * 2)]
        forward = cKDTree(target).query(source)[0]
        reverse = cKDTree(source).query(target)[0]
        gate = config.registration.icp_max_correspondence_distance
        f, r = float(np.mean(forward < gate)), float(np.mean(reverse < gate))
        print(index, saved["registration_metrics"][index]["file_name"], len(source),
              "forward", f, "reverse", r, "sqrt", np.sqrt(f * r),
              "median", np.median(forward), flush=True)
        rows.append({"view_index": index, "forward": f, "reverse": r,
                     "symmetric_support": float(np.sqrt(f * r)),
                     "median_distance": float(np.median(forward))})
    summary_path.with_name("pose_support_audit.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
