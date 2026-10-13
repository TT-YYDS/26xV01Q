"""Compare rigid 3D overlap refinement against recorded poses."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import json
from pathlib import Path
import copy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app.native.backend import config_from_dict
from pointcloud_app.pipeline import read_point_cloud, denoise_cloud, remove_calibration_region
from pointcloud_app.overlap_registration import fixture_component_mask, refine_overlap_pair


def main():
    summary_path = Path(sys.argv[1])
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    config = config_from_dict(saved)
    clouds = []
    for path, metric in zip(saved["input_files"], saved["registration_metrics"]):
        c, _ = denoise_cloud(read_point_cloud(Path(path)), config.denoise, Path(path).name,
                             protected_bounds=(config.calibration.bounds.min_bound,
                                               config.calibration.bounds.max_bound))
        c.transform(np.array(metric["transformation"]))
        fixture = fixture_component_mask(np.asarray(c.points), config.calibration, .24)
        c = c.select_by_index(np.flatnonzero(~fixture))
        clouds.append(c.voxel_down_sample(.15))
    before = copy.deepcopy(clouds)
    target = copy.deepcopy(clouds[0])
    for i in range(1, len(clouds)):
        if not saved["registration_metrics"][i]["included_in_fusion"]:
            continue
        source = clouds[i]
        transform, report = refine_overlap_pair(source, target, config.registration)
        print(json.dumps(report), flush=True)
        old = cKDTree(np.asarray(target.points)).query(np.asarray(source.points))[0]
        source.transform(transform)
        new = cKDTree(np.asarray(target.points)).query(np.asarray(source.points))[0]
        mask = old < .6
        print(i, "overlap", mask.mean(), "median/p90", np.quantile(old[mask], [.5,.9]),
              np.quantile(new[mask], [.5,.9]), "transform", transform.tolist(), flush=True)
        target += source
        target = target.voxel_down_sample(.15)
    output = ROOT / "tmp/overlap_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, data in enumerate((before, clouds)):
        for col, (x,y) in enumerate(((0,1),(0,2),(1,2))):
            for i, c in enumerate(data):
                if saved["registration_metrics"][i]["included_in_fusion"]:
                    p=np.asarray(c.points)
                    axes[row,col].scatter(p[:,x], p[:,y], s=.25, label=str(i))
            axes[row,col].axis("equal")
            axes[row,col].set_title(f"{'Before' if row == 0 else 'After'}: {'XYZ'[x]} / {'XYZ'[y]}")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=130)


if __name__ == "__main__":
    main()
