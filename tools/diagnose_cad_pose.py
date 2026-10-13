"""Inspect real frozen-section pose candidates without changing inputs."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ezdxf
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import media_analysis_core as core


def main():
    reports = sorted(ROOT.glob("outputs/group_*/20260915_093*/native_cad/*/report.json"))
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        source = Path(report["source_pointcloud"])
        processing = json.loads((source.parent.parent / "processing_report.json").read_text(encoding="utf-8"))
        profile = core.section_profile_from_points(np.asarray(o3d.io.read_point_cloud(str(source)).points), processing)
        parts = [core.sample_entity(e) for e in ezdxf.readfile(report["source_dxf"]).modelspace()]
        outline = np.vstack([p for p in parts if len(p)])
        anchor = np.vstack([parts[i] for i in report["fit"]["entities"]])
        original = core.differential_evolution
        calls = []

        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            return result

        with patch.object(core, "differential_evolution", side_effect=record):
            center, mirror, parameters, distances, diagnostics = core.fit_profile_global(profile, outline, anchor)
        tree = core.cKDTree(core._grid_reduce(profile, .025))
        candidates = []
        for m, result in zip((1., -1.), calls[:2]):
            d = tree.query(core.transform(anchor, center, m, result.x[0], result.x[1:]))[0]
            candidates.append(dict(mirror=m, angle=float(np.degrees(result.x[0])), score=float(result.fun),
                                   anchor_mean=float(d.mean()), anchor_p95=float(np.quantile(d, .95))))
        print(json.dumps(dict(report=str(path), candidates=candidates, selected=diagnostics,
                              angle=float(np.degrees(parameters[0])), mirror=mirror)), flush=True)


if __name__ == "__main__":
    main()
