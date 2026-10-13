"""Run adjusted-region fitting, persistence, processing and CAD on real inputs."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointcloud_app.native.backend import Backend, config_from_dict, fit_selection
from pointcloud_app.workflow_profile import build_profile, load_profile, save_profile


def main():
    summary_path = ROOT / "outputs/group_1/20260914_173720_329542/run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = config_from_dict(summary)
    files = [Path(value) for value in summary["input_files"]]
    directory = ROOT / "tmp/region_pipeline_verification" / datetime.now().strftime("%Y%m%d_%H%M%S")
    directory.mkdir(parents=True)
    backend = Backend()
    _, preview = backend.preview(files, config, 2, True, lambda message: None)
    # Move the same-size test rectangle onto the elongated workpiece.
    old_bounds = [-8, 8, -3, 3]
    new_bounds = [-20, -4, -1, 5]
    config.section, count, confidence = fit_selection(preview.points, config.section, new_bounds)
    previous = json.loads(next(summary_path.parent.glob("native_cad/20260914_173722_*/report.json")).read_text(encoding="utf-8"))
    selection = {
        "registration_paths": previous["fit"]["registration_paths"],
        "measurement_paths": previous["measurement_paths"],
    }
    payload = build_profile(workflow_mode="manual", source_kind="folder",
                            source_path=files[0].parent, config=config,
                            dxf=previous["source_dxf"], selection=selection, cad_step=.05)
    saved_path = save_profile(payload, directory / "profile.json")
    reloaded, error = load_profile(saved_path)
    assert error is None
    config = config_from_dict(reloaded["config"])
    result = backend.process(files, config, lambda message: None, directory)
    assert result.section_points >= 30
    before = hashlib.sha256(result.section_cloud_path.read_bytes()).hexdigest()
    report = backend.analyze(result, Path(reloaded["cad"]["dxf"]), selection, .05, lambda message: None)
    assert before == hashlib.sha256(result.section_cloud_path.read_bytes()).hexdigest()
    actual = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert actual["config"]["section"] == config.as_dict()["section"]
    verification = dict(
        ok=True, old_bounds=old_bounds, adjusted_bounds=new_bounds,
        selected_points=count, confidence=confidence,
        section_points=result.section_points, merged_points=result.merged_points,
        summary=str(result.summary_path), report=str(Path(report["source_dxf"]).parent / "report.json"),
        quality=report["measurement_quality"]["status"],
        fit_mean=report["fit"]["mean_residual"], fit_p95=report["fit"]["p95_residual"],
        frozen_section_unchanged=True, profile=str(saved_path),
    )
    (directory / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
