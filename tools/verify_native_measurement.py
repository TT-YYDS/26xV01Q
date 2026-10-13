"""Run the native CAD adapter against an existing real frozen section."""

from pathlib import Path
import json
import shutil
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointcloud_app.native.backend import Backend, result_from_summary
from pointcloud_app.native.figures import cad_overlay, measurement_curves

original = ROOT / "outputs/group_1/20260904_100702_076333"
destination = ROOT / "tmp/native_measurement_verification" / datetime.now().strftime("%Y%m%d_%H%M%S")
destination.mkdir(parents=True)
shutil.copy2(original / "run_summary.json", destination / "run_summary.json")
result = result_from_summary(destination / "run_summary.json")
previous = json.loads((original / "cad_analysis/report.json").read_text(encoding="utf-8"))
selection = {
    "registration_paths": previous["fit"]["registration_paths"],
    "measurement_paths": previous["measurement_paths"],
}
report = Backend().analyze(result, Path(previous["source_dxf"]), selection, .05,
                           lambda message: print(message.encode("ascii", "backslashreplace").decode(), flush=True))
assert report["fit"]["pointcloud_transform_applied"] is False
assert report["measurement_quality"]["status"] != "PASS"
assert report["processing_quality"]["formal_measurement_allowed"] is False
assert "inputs" in str(report["source_pointcloud"])
assert cad_overlay(report).data
curve = measurement_curves(report)
assert any(any(v is not None for v in trace.y) for trace in curve.data)
report_path = Path(report["source_dxf"]).parent / "report.json"
print(json.dumps({"ok": True, "status": report["measurement_quality"]["status"],
                  "report": str(report_path), "curves": len(curve.data)}, ensure_ascii=True), flush=True)
