"""Compare local features against peer support for diagnosis only."""
import sys
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pointcloud_app.native.backend import config_from_dict
from pointcloud_app.pipeline import vertical_plane_basis, select_vertical_section_indices


def main():
    summary = json.loads((ROOT / "outputs/group_1/20260915_100143_452856/run_summary.json").read_text(encoding="utf-8"))
    config = config_from_dict(summary)
    _, h, v = vertical_plane_basis(config.section.angle_degrees, config.section.tilt_degrees)
    directory = ROOT / "tmp/surface_noise_diagnostic/20260915_100143_452856"
    level = np.array(summary["leveling"]["transformation"])
    metrics = {r["file_name"]: r for r in summary["registration_metrics"]}
    locals_, worlds = [], []
    for index, file in enumerate(summary["input_files"]):
        p = np.load(directory / f"view_{index}.npz")["points"]
        t = level @ np.array(metrics[Path(file).name]["transformation"])
        locals_.append(p)
        worlds.append(p @ t[:3, :3].T + t[:3, 3])
    for index, (points, world) in enumerate(zip(locals_, worlds)):
        indices = select_vertical_section_indices(world, config.section)
        q = np.column_stack((world[indices] @ h, world[indices] @ v))
        selected = (q[:, 0] > 2) & (q[:, 0] < 6.5) & (q[:, 1] < 1.5)
        indices, q = indices[selected], q[selected]
        peer = cKDTree(np.vstack([w for i, w in enumerate(worlds) if i != index])).query(world[indices])[0]
        tree = cKDTree(points)
        features = {}
        for k in (24, 64, 128):
            distances, ids = tree.query(points[indices], k=k)
            local = points[ids]
            delta = local - local.mean(axis=1, keepdims=True)
            eig, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", delta, delta) / k)
            residual = np.abs(np.einsum("ni,ni->n", points[indices] - local.mean(axis=1), vec[:, :, 0]))
            features[str(k)] = np.column_stack((distances[:, -1], eig[:, 0] / eig.sum(axis=1), eig[:, 1] / eig[:, 2], residual))
        for j in np.flatnonzero(peer > .15):
            print(json.dumps(dict(view=index + 1, xy=np.round(q[j], 3).tolist(), peer=round(float(peer[j]), 3),
                                  features={k: np.round(value[j], 4).tolist() for k, value in features.items()})))


if __name__ == "__main__":
    main()
