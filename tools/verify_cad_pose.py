"""Replay the four reported frozen sections, preserving profiles and inputs."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from media_analysis_core import analyze, inverse_transform, sample_entity, section_profile_from_points


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(ax, profile, parts, report, title):
    fit = report["fit"]["cad_transform"]
    points = inverse_transform(profile, np.array(fit["center"]), fit["mirror"],
                               fit["angle_rad"], np.array(fit["translation"]))
    origin = np.vstack(parts).mean(axis=0)
    for part in parts:
        ax.plot(*(part - origin).T, color="#94a3b8", linewidth=1)
    ax.scatter(*(points - origin).T, s=2, color="#f97316")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(alpha=.2)


def main():
    destination = ROOT / "tmp/cad_pose_verification" / datetime.now().strftime("%Y%m%d_%H%M%S")
    destination.mkdir(parents=True)
    settings = ROOT / "settings/workflow_profile.json"
    settings_hash = digest(settings)
    paths = sorted(ROOT.glob("outputs/group_*/20260915_093*/native_cad/*/report.json"))
    assert len(paths) == 4
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    results = []
    for index, path in enumerate(paths):
        old = json.loads(path.read_text(encoding="utf-8"))
        source = Path(old["source_pointcloud"])
        source_hash = digest(source)
        dxf_hash = digest(Path(old["source_dxf"]))
        new = analyze(source, old["source_dxf"], destination / f"group_{index + 1}",
                      step=.05, registration_paths=old["fit"]["registration_paths"],
                      measurement_paths=old["measurement_paths"])
        assert digest(source) == source_hash
        assert digest(Path(old["source_dxf"])) == dxf_hash
        assert new["fit"]["mean_residual"] < old["fit"]["mean_residual"]
        assert new["fit"]["p95_residual"] < old["fit"]["p95_residual"]
        assert new["fit"]["mirror_x"] == 1.
        assert -50 < new["fit"]["angle_deg"] < -44
        processing = json.loads((source.parent.parent / "processing_report.json").read_text(encoding="utf-8"))
        profile = section_profile_from_points(np.asarray(o3d.io.read_point_cloud(str(source)).points), processing)
        parts = [sample_entity(e) for e in ezdxf.readfile(old["source_dxf"]).modelspace()]
        parts = [p for p in parts if len(p)]
        draw(axes[index, 0], profile, parts, old, f"Group {index + 1}: before")
        draw(axes[index, 1], profile, parts, new, f"Group {index + 1}: after")
        row = dict(group=index + 1, old_report=str(path),
                   report=str(destination / f"group_{index + 1}/report.json"),
                   before_angle=old["fit"]["angle_deg"], angle=new["fit"]["angle_deg"],
                   before_mean=old["fit"]["mean_residual"], mean=new["fit"]["mean_residual"],
                   before_p95=old["fit"]["p95_residual"], p95=new["fit"]["p95_residual"],
                   quality=new["measurement_quality"]["status"], source_unchanged=True)
        results.append(row)
        print(json.dumps(row), flush=True)
    fig.savefig(destination / "comparison.png", dpi=130)
    plt.close(fig)
    assert digest(settings) == settings_hash
    (destination / "verification.json").write_text(
        json.dumps(dict(ok=True, settings_unchanged=True, groups=results), indent=2), encoding="utf-8")
    print(str(destination), flush=True)


if __name__ == "__main__":
    main()
