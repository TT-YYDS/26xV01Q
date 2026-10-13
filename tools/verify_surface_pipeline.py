"""Run real single-view denoising through fusion, section and CAD in isolation."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pointcloud_app.native.backend import Backend, config_from_dict, sha256
from pointcloud_app.pipeline import read_point_cloud, vertical_plane_basis
from media_analysis_core import inverse_transform, section_profile_from_points, sample_entity
import ezdxf


def digest(path):
    return sha256(path)


def overlay(ax, report, label):
    source = Path(report["source_pointcloud"])
    processing = json.loads((source.parent.parent / "processing_report.json").read_text(encoding="utf-8"))
    profile = section_profile_from_points(np.asarray(read_point_cloud(source).points), processing)
    t = report["fit"]["cad_transform"]
    points = inverse_transform(profile, np.array(t["center"]), t["mirror"],
                               t["angle_rad"], np.array(t["translation"]))
    parts = [sample_entity(e) for e in ezdxf.readfile(report["source_dxf"]).modelspace()]
    parts = [p for p in parts if len(p)]
    origin = np.vstack(parts).mean(axis=0)
    for part in parts:
        ax.plot(*(part - origin).T, color="#94a3b8", linewidth=1)
    ax.scatter(*(points - origin).T, s=3, color="#f97316")
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.set_title(label)


def main():
    destination = ROOT / "tmp/surface_pipeline_verification" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    settings = ROOT / "settings/workflow_profile.json"
    settings_hash = digest(settings)
    code_hashes = {name: digest(ROOT / name) for name in (
        "pointcloud_app/pipeline.py", "pointcloud_app/models.py",
        "pointcloud_app/native/backend.py", "media_analysis_core.py",
    )}
    all_rows = []
    for group in sys.argv[1:] or ["1"]:
        candidates = sorted((ROOT / f"outputs/group_{group}").glob("20260915_100*/run_summary.json"))
        if not candidates:
            candidates = sorted((ROOT / f"outputs/group_{group}").glob("20260915_093*/run_summary.json"))
        summary_path = candidates[-1]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline_report = sorted(summary_path.parent.glob("native_cad/*/report.json"))[-1]
        old_report = json.loads(baseline_report.read_text(encoding="utf-8"))
        config = config_from_dict(summary)
        config.compute_backend = "cpu"
        assert config.denoise.use_surface_support_filter
        files = [Path(p) for p in summary["input_files"]]
        protected_sources = [
            settings, summary_path, baseline_report, Path(old_report["source_dxf"]),
            Path(old_report["source_pointcloud"]), *files,
        ]
        source_hashes = {str(path): digest(path) for path in protected_sources}
        backend = Backend()
        result = backend.process(files, config, lambda text: print(f"[{group}] {text}", flush=True), destination)
        frozen_hash = digest(result.section_cloud_path)
        selection = dict(registration_paths=old_report["fit"]["registration_paths"],
                         measurement_paths=old_report["measurement_paths"])
        report = backend.analyze(result, Path(old_report["source_dxf"]), selection, .05,
                                 lambda text: print(f"[{group}] {text}", flush=True))
        assert digest(result.section_cloud_path) == frozen_hash
        actual = json.loads(result.summary_path.read_text(encoding="utf-8"))
        assert actual["config"]["denoise"] == config.as_dict()["denoise"]
        assert not actual["config"]["section"]["use_statistical_filter"]
        assert not actual["config"]["section"]["use_component_filter"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        _, horizontal, vertical = vertical_plane_basis(config.section.angle_degrees, config.section.tilt_degrees)
        for ax, path, label in zip(axes[0],
                                  (summary_path.parent / "vertical_section.ply", result.section_cloud_path),
                                  ("Before: frozen section", "After: frozen section")):
            points = np.asarray(read_point_cloud(path).points)
            ax.scatter(points @ horizontal, points @ vertical, c=points[:, 2], cmap="turbo", s=3)
            ax.set_aspect("equal")
            ax.set_title(label)
            ax.grid(alpha=.2)
        overlay(axes[1, 0], old_report, "Before: CAD reference only")
        overlay(axes[1, 1], report, "After: CAD reference only")
        fig.savefig(destination / f"group_{group}_comparison.png", dpi=140)
        plt.close(fig)
        assert all(digest(path) == value for path, value in source_hashes.items())
        row = dict(group=group, baseline=str(summary_path), summary=str(result.summary_path),
                   report=str(Path(report["source_dxf"]).parent / "report.json"),
                   section_points=result.section_points, merged_points=result.merged_points,
                   quality=report["measurement_quality"]["status"],
                   fit_mean=report["fit"]["mean_residual"], fit_p95=report["fit"]["p95_residual"],
                   measurements=report["measurement_quality"]["measurements"],
                   denoise_stats=actual.get("denoise_stats"), frozen_section_unchanged=True,
                   source_hashes=source_hashes, source_files_unchanged=True,
                   frozen_section_sha256=frozen_hash, code_hashes=code_hashes,
                   settings_unchanged=digest(settings) == settings_hash,
                   post_section_filters_disabled=True)
        all_rows.append(row)
        (destination / "verification.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    assert digest(settings) == settings_hash
    assert all(row["quality"] == "PASS" for row in all_rows)
    print(str(destination), flush=True)


if __name__ == "__main__":
    main()
