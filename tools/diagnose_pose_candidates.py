"""Inspect global-pose hypotheses for a recorded view without changing its files."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import json
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app import pipeline as p
from pointcloud_app.native.backend import config_from_dict


def main():
    saved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    index = int(sys.argv[2])
    config = config_from_dict(saved)
    clouds = []
    for i in (0, index):
        path = Path(saved["input_files"][i])
        c, _ = p.denoise_cloud(p.read_point_cloud(path), config.denoise, path.name,
                               protected_bounds=(config.calibration.bounds.min_bound,
                                                 config.calibration.bounds.max_bound))
        clouds.append(c)
    target = p.crop_calibration_cloud(clouds[0], config.calibration, "reference")
    if "--multiview" in sys.argv:
        for i, path in enumerate(saved["input_files"]):
            if i in (0, index) or not saved["registration_metrics"][i]["included_in_fusion"]:
                continue
            c, _ = p.denoise_cloud(
                p.read_point_cloud(Path(path)), config.denoise, Path(path).name,
                protected_bounds=(config.calibration.bounds.min_bound,
                                  config.calibration.bounds.max_bound))
            c.transform(np.array(saved["registration_metrics"][i]["transformation"]))
            clouds[0] += c
        clouds[0] = clouds[0].voxel_down_sample(config.registration.calibration_graph_voxel_size)
        target = p.crop_calibration_cloud(clouds[0], config.calibration, "reference")
    metric = saved["registration_metrics"][index]
    best = (np.array(metric["transformation"]), metric["global_fitness"],
            metric["global_rmse"], metric["icp_fitness"], metric["icp_rmse"],
            metric["calibration_points"], metric["full_cloud_fitness"],
            metric["full_cloud_rmse"], {})
    refinement = copy.deepcopy(config.registration)
    refinement.registration_voxel_size = config.registration.calibration_graph_voxel_size
    refinement.icp_max_correspondence_distance = config.registration.calibration_graph_max_correspondence_distance
    original_icp = p._refine_with_icp_diagnostics
    original_global = p.coarse_global_register
    original_gicp = o3d.pipelines.registration.registration_generalized_icp
    body_candidates = []
    scores = []

    sample = np.asarray(clouds[0].voxel_down_sample(config.registration.registration_voxel_size).points)
    sample = sample[~p.fixture_component_mask(sample, config.calibration,
                                              config.registration.registration_voxel_size * 2)]
    values = np.linalg.eigvalsh(np.cov(sample.T))
    print("PRINCIPAL VARIANCES", values.tolist(), "ratio", values[-1] / values[-2], flush=True)

    def inspect_global(*args, **kwargs):
        scorer = kwargs.get("candidate_scorer")
        result = original_global(*args, **kwargs)
        if scorer:
            scores.append(scorer)
            print("BODY SCORE baseline", scorer(best[0]), "candidate", scorer(result[0]), flush=True)
        return result

    def inspect_gicp(*args, **kwargs):
        result = original_gicp(*args, **kwargs)
        body_candidates.append(np.array(result.transformation))
        print("GICP BODY SCORE", scores[0](result.transformation), flush=True)
        return result

    def inspect(source, target, cfg, backend):
        a = p.prepare_registration_cloud(source, cfg, backend)
        b = p.prepare_registration_cloud(target, cfg, backend)
        old = o3d.pipelines.registration.evaluate_registration(
            a, b, cfg.icp_max_correspondence_distance, np.eye(4))
        result = original_icp(source, target, cfg, backend)
        print("FIXTURE before", old.fitness, old.inlier_rmse, "after", result[1:3],
              "correction", result[0].tolist(), flush=True)
        return result

    with patch.object(p, "_refine_with_icp_diagnostics", side_effect=inspect), patch.object(
            p, "coarse_global_register", side_effect=inspect_global), patch.object(
                o3d.pipelines.registration, "registration_generalized_icp",
                side_effect=inspect_gicp):
        result, localized = p._disambiguate_global_pose(
            clouds[1], clouds[0], target, best, config, refinement, "source", "cpu")
    print("CHANGED", localized is not None, result[8], flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    candidates = sorted(body_candidates, key=scores[0], reverse=True)
    selected = [best[0], candidates[0], result[0]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row in range(2):
        for ax, transform, label in zip(axes[row], selected, ("Baseline", "Best Body", "Selected")):
            for c, t, color in ((clouds[0], np.eye(4), "blue"), (clouds[1], transform, "orange")):
                pts = np.asarray(c.voxel_down_sample(.12).points)
                if row:
                    pts = pts[~p.fixture_component_mask(pts, config.calibration, .24)]
                pts = pts @ t[:3, :3].T + t[:3, 3]
                ax.scatter(pts[:, 1], pts[:, 2], s=.2, c=color)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label}: {scores[0](transform):.3f}")
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")
    np.savez(ROOT / "tmp" / "pose_candidates.npz", transforms=np.array(selected))
    destination = ROOT / "tmp" / "pose_candidates.png"
    fig.savefig(destination, dpi=160)
    print("PLOT", destination, flush=True)


if __name__ == "__main__":
    main()
