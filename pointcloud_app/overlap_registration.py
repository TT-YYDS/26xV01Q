"""Conservative, reference-anchored rigid overlap registration."""
from __future__ import annotations

import copy
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .models import CalibrationConfig, RegistrationConfig


def fixture_component_mask(points: np.ndarray, config: CalibrationConfig,
                           voxel_size: float) -> np.ndarray:
    """Extend fixture seeds only to separate, predominantly fixture components."""
    points = np.asarray(points)
    inside = np.zeros(len(points), dtype=bool)
    if not config.use_roi or not len(points):
        return inside
    lower = np.asarray(config.bounds.min_bound) - config.exclusion_margin
    upper = np.asarray(config.bounds.max_bound) + config.exclusion_margin
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    if not inside.any() or inside.all():
        return inside
    keys, inverse = np.unique(np.floor(points / voxel_size).astype(np.int64),
                              axis=0, return_inverse=True)
    pairs = cKDTree(keys).query_pairs(1, p=np.inf, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(keys), len(keys))).tocsr()
    count, labels = connected_components(graph, directed=False)
    # Count occupied cells, not raw points: dense fixture faces must not
    # outweigh a connected, more sparsely sampled workpiece.
    seeded = np.zeros(len(keys), dtype=bool)
    seeded[inverse[inside]] = True
    sizes = np.bincount(labels, minlength=count)
    seeds = np.bincount(labels[seeded], minlength=count)
    component_lower = np.full((count, 3), np.inf)
    component_upper = np.full((count, 3), -np.inf)
    np.minimum.at(component_lower, labels, keys * voxel_size)
    np.maximum.at(component_upper, labels, (keys + 1) * voxel_size)
    padding = (upper - lower) * .5
    bounded = np.all((component_lower >= lower - padding - voxel_size - 1e-12)
                     & (component_upper <= upper + padding + voxel_size), axis=1)
    extend = (seeds >= 30) & (seeds >= .2 * sizes) & bounded
    return inside | extend[labels[inverse]]


def _points(cloud):
    return np.asarray(cloud.points)


def refine_overlap_pair(source, target, config: RegistrationConfig):
    """Fit with Open3D, reject weak-axis drift and independently score the pose."""
    identity = np.eye(4)
    report = {"accepted": False, "reason": "insufficient_overlap"}
    voxel = config.workpiece_graph_voxel_size
    gate = config.workpiece_graph_max_correspondence_distance
    if min(len(source.points), len(target.points)) < 100:
        return identity, report
    source = copy.deepcopy(source)
    target = copy.deepcopy(target)
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=40)
    target.estimate_normals(search)
    source.estimate_normals(search)
    source_points, target_points = _points(source), _points(target)
    tree = cKDTree(target_points)
    before, nearest = tree.query(source_points)
    reverse_before, _ = cKDTree(source_points).query(target_points)
    mask, reverse_mask = before < gate, reverse_before < gate
    minimum = max(100, int(.1 * min(len(source_points), len(target_points))))
    if min(mask.sum(), reverse_mask.sum()) < minimum:
        return identity, report

    # Leave a deterministic spatially distributed subset out of ICP fitting.
    training = np.flatnonzero(np.arange(len(source_points)) % 5 != 0)
    fitting = source.select_by_index(training)
    transform = identity.copy()
    for distance in (2 * gate, gate, max(.5 * gate, voxel)):
        result = o3d.pipelines.registration.registration_icp(
            fitting, target, distance, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=distance)),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max(20, config.workpiece_graph_icp_max_iterations // 3)))
        transform = np.array(result.transformation, copy=True)
    if not np.isfinite(transform).all():
        report["reason"] = "nonfinite_correction"
        return identity, report
    normals = np.asarray(target.normals)[nearest[mask]]
    values, vectors = np.linalg.eigh(normals.T @ normals)
    strong = vectors[:, values > max(values[-1] * .02, 1e-12)]
    center = source_points.mean(axis=0)
    movement = transform[:3, :3] @ center + transform[:3, 3] - center
    movement = strong @ strong.T @ movement
    transform[:3, 3] = center - transform[:3, :3] @ center + movement
    rotation = float(np.degrees(np.arccos(np.clip(
        (np.trace(transform[:3, :3]) - 1) / 2, -1, 1))))
    report.update(rotation_degrees=rotation, center_displacement=float(np.linalg.norm(movement)),
                  observable_translation_axes=int(strong.shape[1]))
    if (rotation > config.workpiece_graph_max_rotation_degrees
            or np.linalg.norm(movement) > config.workpiece_graph_max_translation):
        report["reason"] = "correction_limit"
        return identity, report
    moved = source_points @ transform[:3, :3].T + transform[:3, 3]
    after, _ = tree.query(moved)
    reverse_after, _ = cKDTree(moved).query(target_points)
    held_out = mask & (np.arange(len(source_points)) % 5 == 0)
    if held_out.sum() < 20:
        return identity, report
    # The masks and distance gate stay fixed between before and after.
    old = np.concatenate((before[held_out], reverse_before[reverse_mask]))
    new = np.concatenate((after[held_out], reverse_after[reverse_mask]))
    old_rmse = float(np.sqrt(np.mean(np.minimum(old, gate) ** 2)))
    new_rmse = float(np.sqrt(np.mean(np.minimum(new, gate) ** 2)))
    old_p90, new_p90 = float(np.quantile(old, .9)), float(np.quantile(new, .9))
    report.update(rmse_before=old_rmse, rmse_after=new_rmse,
                  p90_before=old_p90, p90_after=new_p90,
                  validation_points=len(old), overlap_points=int(mask.sum()))
    if (old_rmse < voxel * 1e-6 or new_rmse > old_rmse * .98
            or new_p90 > old_p90 + voxel * .05
            or np.mean(after[mask] < gate) < .97
            or np.mean(reverse_after[reverse_mask] < gate) < .97):
        report["reason"] = "validation_not_improved"
        return identity, report
    report.update(accepted=True, reason="validated_overlap",
                  correction=transform.tolist())
    return transform, report


def refine_multiview_overlap(clouds, transforms, included, calibration, registration,
                             *, reference_frame_roi=True):
    """Keep the first view fixed and retain all unique coverage and point samples."""
    if not (len(clouds) == len(transforms) == len(included)):
        raise ValueError("Overlap registration input counts must match.")
    result = [np.array(t, copy=True) for t in transforms]
    if not registration.use_overlap_refinement:
        return result, {"enabled": False, "views": []}
    active = [i for i, use in enumerate(included) if use]
    if len(active) < 2:
        return result, {"enabled": True, "views": []}
    aligned = []
    for cloud, transform in zip(clouds, transforms):
        c = copy.deepcopy(cloud)
        if not reference_frame_roi:
            mask = fixture_component_mask(_points(c), calibration,
                                          registration.workpiece_graph_voxel_size * 2)
            c = c.select_by_index(np.flatnonzero(~mask))
        c.transform(transform)
        if reference_frame_roi:
            mask = fixture_component_mask(_points(c), calibration,
                                          registration.workpiece_graph_voxel_size * 2)
            c = c.select_by_index(np.flatnonzero(~mask))
        aligned.append(c.voxel_down_sample(registration.workpiece_graph_voxel_size))
    target = copy.deepcopy(aligned[active[0]])
    rows = []
    for index in active[1:]:
        correction, report = refine_overlap_pair(aligned[index], target, registration)
        result[index] = correction @ result[index]
        aligned[index].transform(correction)
        rows.append({"view_index": index, **report})
        target += aligned[index]
        target = target.voxel_down_sample(registration.workpiece_graph_voxel_size)
    return result, {"enabled": True, "backend": "cpu", "reference_index": active[0], "views": rows}
