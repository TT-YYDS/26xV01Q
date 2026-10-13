"""Session-local pose seeds; current fixture geometry remains authoritative."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def config_signature(config):
    return hashlib.sha256(json.dumps(
        config.as_dict(), sort_keys=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def station_ids(files):
    """Never infer robot stations from file order or modification time."""
    parents = {Path(path).resolve().parent for path in files}
    if len(parents) != 1:
        raise ValueError("station files must share one directory")
    metadata = next(iter(parents)) / "station_order.json"
    if metadata.exists():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        views = payload["views"]
        if len(views) != len(files):
            raise ValueError("station metadata does not cover this acquisition")
        mapping = {item["file"]: str(item["station"]).strip() for item in views}
        if (len(mapping) != len(files) or set(mapping) != {Path(p).name for p in files}
                or len(set(mapping.values())) != len(files) or "" in mapping.values()):
            raise ValueError("station metadata must be complete and unique")
        return [mapping[Path(path).name] for path in files]
    matches = [re.search(r"(?:view|station|scan|camera)[_-]?(\d+)(?!\d)",
                         Path(path).stem, re.I) for path in files]
    if not any(matches):
        return None
    if not all(matches):
        raise ValueError("only some files have station numbers")
    labels = [str(int(match[1])) for match in matches]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate station numbers")
    return labels


def fixture_sample(cloud, voxel):
    points = np.asarray(cloud.voxel_down_sample(voxel).points)
    if len(points) > 4000:
        points = points[np.linspace(0, len(points) - 1, 4000, dtype=int)]
    return points.copy()


def valid_rigid(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return (matrix.shape == (4, 4) and np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            and abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-6)


def registration_verified(summary):
    """Gate pose reuse on stitching evidence, not whether the workpiece is good."""
    from .downstream_bridge import _derive_processing_quality

    metrics = summary.get("registration_metrics") or []
    if len(metrics) < 2 or not all(
            m.get("calibration_quality_passed") and m.get("included_in_fusion")
            and valid_rigid(m.get("transformation")) for m in metrics):
        return False
    graph = summary.get("pose_graph") or {}
    nodes = graph.get("calibration_nodes") or []
    if len(nodes) != len(metrics) or not all(n.get("accepted") for n in nodes):
        return False
    reached = {0}
    edges = [e for e in graph.get("calibration_edges", []) if e.get("accepted")]
    for _ in metrics:
        for edge in edges:
            a, b = edge["source_index"], edge["target_index"]
            if a in reached or b in reached:
                reached.update((a, b))
    return (reached == set(range(len(metrics)))
            and _derive_processing_quality(summary)["status"] == "PASS")


def make_reference(files, calibration_clouds, transforms, config, metrics):
    if not all(m.calibration_quality_passed and m.included_in_fusion for m in metrics):
        return None
    if not all(valid_rigid(matrix) for matrix in transforms):
        return None
    try:
        labels = station_ids(files)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return {
        "version": 1, "config_signature": config_signature(config),
        "stations": labels,
        "files": [str(Path(path).resolve()) for path in files],
        # These matrices precede leveling and map into this group's reference view.
        "transforms": [np.asarray(matrix).tolist() for matrix in transforms],
        "fixture_samples": [fixture_sample(cloud, config.registration.registration_voxel_size).tolist()
                            for cloud in calibration_clouds],
    }


def match_reference(reference, files, calibration_clouds, config):
    """Return seeds in CURRENT reference coordinates, or a conservative fallback."""
    diagnostic = {"status": "full_search", "reason": "no_reference"}
    if not reference:
        return None, diagnostic
    diagnostic["source_summary"] = reference.get("source_summary")
    try:
        if reference["version"] != 1 or reference["config_signature"] != config_signature(config):
            raise ValueError("configuration_changed")
        matrices = [np.asarray(t, dtype=float) for t in reference["transforms"]]
        if len(matrices) != len(files) or not all(valid_rigid(t) for t in matrices):
            raise ValueError("invalid_transform_or_view_count")
        labels = station_ids(files)
        stored_labels = reference["stations"]
        if labels is not None and stored_labels is not None:
            if set(labels) != set(stored_labels):
                raise ValueError("station_ids_changed")
            indices = [stored_labels.index(label) for label in labels]
            diagnostic["mapping"] = "station_id"
        else:
            current = [fixture_sample(c, config.registration.registration_voxel_size)
                       for c in calibration_clouds]
            previous = [np.asarray(p, dtype=float) for p in reference["fixture_samples"]]
            if len(previous) != len(files) or any(
                    p.ndim != 2 or p.shape[1] != 3 or len(p) < 30 or not np.isfinite(p).all()
                    for p in [*current, *previous]):
                raise ValueError("insufficient_fixture_mapping_evidence")
            old_trees = [cKDTree(p) for p in previous]
            new_trees = [cKDTree(p) for p in current]
            costs = np.array([
                [max(np.quantile(tree.query(points)[0], .8),
                     np.quantile(new_trees[i].query(previous[j])[0], .8))
                 for j, tree in enumerate(old_trees)]
                for i, points in enumerate(current)
            ])
            gate = config.registration.icp_max_correspondence_distance
            nearest = np.argmin(costs, axis=1).tolist()
            indices = [
                j if (costs[i, j] <= gate
                      and np.delete(costs[i], j).min() - costs[i, j] >= gate * .25
                      and nearest.count(j) == 1) else None
                for i, j in enumerate(nearest)
            ]
            diagnostic.update(mapping="fixture_geometry", mapping_costs=costs.tolist())
        if indices[0] is None:
            raise ValueError("ambiguous_fixture_station_mapping")
        origin = np.linalg.inv(matrices[indices[0]])
        seeds = [origin @ matrices[index] if index is not None else None for index in indices]
        diagnostic.pop("reason", None)
        diagnostic.update(status="matched", station_indices=indices)
        return seeds, diagnostic
    except (ValueError, KeyError, TypeError, OSError, IndexError, np.linalg.LinAlgError) as error:
        diagnostic["reason"] = str(error)
        return None, diagnostic
