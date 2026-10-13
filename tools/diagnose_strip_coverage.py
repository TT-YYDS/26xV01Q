"""Separate strip cleanup, registration and CAD fitting effects on coverage."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import media_analysis_core as cad
from pointcloud_app.native.backend import config_from_dict, sha256
from pointcloud_app.pipeline import (
    _merge_clouds_with_section_consensus, _section_profile_coordinates,
    _surface_support_inlier_mask, _vertical_section_mask,
    extract_section_from_merged_cloud,
)


def main():
    old_path = ROOT / "outputs/group_2/20260915_172226_494440/run_summary.json"
    new_path = Path(sys.argv[1]).resolve()
    cache = ROOT / "tmp/strip_noise_verification/20260915_173400_414148"
    destination = new_path.parents[2] / "coverage_diagnostic" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    old, new = [json.loads(p.read_text(encoding="utf-8")) for p in (old_path, new_path)]
    old_report_path = sorted(old_path.parent.glob("native_cad/*/report.json"))[-1]
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    config = config_from_dict(old)
    source_hashes = {str(p): sha256(p) for p in (
        old_path, new_path, old_report_path, ROOT / "pointcloud_app/pipeline.py",
    )}
    document = ezdxf.readfile(old_report["source_dxf"])
    sampled = [cad.sample_entity(entity) for entity in document.modelspace()]
    outline = np.vstack([p for p in sampled if len(p)])
    anchor = cad._stack_sampled_path(sampled, old_report["fit"]["entities"])
    measurement = cad._stack_sampled_path(sampled, old_report["measurement_paths"][0])
    transform = old_report["fit"]["cad_transform"]

    def to_cad(profile, fit=None):
        if fit is None:
            return cad.inverse_transform(profile, np.asarray(transform["center"]), transform["mirror"],
                                         transform["angle_rad"], np.asarray(transform["translation"]))
        center, mirror, parameters, _, _ = fit
        return cad.inverse_transform(profile, center, mirror, parameters[0], parameters[1:])

    sources = json.loads((cache / "baseline.json").read_text(encoding="utf-8"))["views"]
    cloud_sets = {"before": {}, "after": {}}
    removed_rows, removed_profiles = [], []
    for index, source in enumerate(sources):
        data = np.load(cache / f"view_{index}.npz")
        points, before = data["points"], data["before"]
        after = _surface_support_inlier_mask(points, config.denoise.voxel_size,
                                             config.denoise.surface_support_neighbors,
                                             config.denoise.surface_support_distance_factor)
        name = Path(source["file"]).name
        for label, keep in (("before", before), ("after", after)):
            cloud_sets[label][name] = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep]))
        removed = points[before & ~after]
        metric = next(row for row in old["registration_metrics"] if row["file_name"] == name)
        matrix = np.array(old["leveling"]["transformation"]) @ np.array(metric["transformation"])
        world = removed @ matrix[:3, :3].T + matrix[:3, 3]
        section = world[_vertical_section_mask(world, config.section, "cpu")]
        section_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(section))
        profile = to_cad(_section_profile_coordinates(section_cloud, config.section))
        removed_profiles.append(profile)
        distance, _ = cKDTree(measurement).query(profile) if len(profile) else (np.array([]), None)
        lower, upper = config.calibration.bounds.min_bound, config.calibration.bounds.max_bound
        roi = np.all((removed >= lower) & (removed <= upper), axis=1)
        removed_rows.append(dict(file=name, removed=len(removed), calibration_removed=int(roi.sum()),
                                 section_removed=len(section), within_measurement_015=int((distance < .15).sum()),
                                 within_measurement_004=int((distance < .04).sum())))
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row_index, (pose_label, summary) in enumerate((("old", old), ("new", new))):
        files = [Path(p) for p in summary["input_files"]]
        matrices = [np.asarray(row["transformation"]) for row in summary["registration_metrics"]]
        included = [row["included_in_fusion"] for row in summary["registration_metrics"]]
        for column, data_label in enumerate(("before", "after")):
            clouds = [cloud_sets[data_label][p.name] for p in files]
            merged, _ = _merge_clouds_with_section_consensus(
                files, matrices, included, np.array(summary["leveling"]["transformation"]),
                config.section, config.fusion_voxel_size, "cpu",
                calibration_config=config.calibration, source_clouds=clouds,
            )
            section, _ = extract_section_from_merged_cloud(merged, config.section, "cpu")
            profile = _section_profile_coordinates(section, config.section)
            fixed = to_cad(cad._grid_reduce(profile, .025))
            fixed_rows, fixed_stats = cad.measure_polyline(fixed, measurement, .05, allow_empty=True)
            fit = cad.fit_profile_global(profile, outline, anchor)
            refitted = to_cad(cad._grid_reduce(profile, .025), fit)
            _, refitted_stats = cad.measure_polyline(refitted, measurement, .05, allow_empty=True)
            item = dict(pose=pose_label, data=data_label, merged=len(merged.points), section=len(section.points),
                        fixed_cad_coverage=fixed_stats["coverage_ratio"],
                        refitted_cad_coverage=refitted_stats["coverage_ratio"])
            rows.append(item)
            print(json.dumps(item), flush=True)
            ax = axes[row_index, column]
            ax.scatter(fixed[:, 0], fixed[:, 1], s=6, color="#007a8c", label="Section")
            ax.plot(measurement[:, 0], measurement[:, 1], lw=1, color="#303030", label="Measurement path")
            if data_label == "after":
                removed = np.vstack(removed_profiles)
                ax.scatter(removed[:, 0], removed[:, 1], s=20, color="#d33d48", marker="x",
                           label="Removed at old poses")
            ax.set(xlim=(measurement[:, 0].min() - .25, measurement[:, 0].max() + .25),
                   ylim=(measurement[:, 1].min() - .25, measurement[:, 1].max() + .25),
                   title=f"{pose_label} poses / {data_label} data: {fixed_stats['coverage_ratio']:.1%} fixed CAD")
            ax.set_aspect("equal")
            ax.grid(alpha=.2)
            ax.legend()
    fig.savefig(destination / "coverage_comparison.png", dpi=150)
    assert all(sha256(Path(p)) == digest for p, digest in source_hashes.items())
    report = dict(comparisons=rows, removed=removed_rows, source_hashes=source_hashes, unchanged=True)
    (destination / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(removed_rows), flush=True)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
