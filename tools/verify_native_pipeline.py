"""Real-data desktop regression in an isolated copy of a recorded acquisition."""

from datetime import datetime
from pathlib import Path
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointcloud_app.native.backend import Backend, config_from_dict, processing_quality

original = ROOT / "outputs/group_1/20260908_101655_059680/run_summary.json"
summary = json.loads(original.read_text(encoding="utf-8"))
directory = ROOT / "tmp/native_pipeline_verification" / datetime.now().strftime("%Y%m%d_%H%M%S")
inputs = directory / "recorded_acquisition"
inputs.mkdir(parents=True)
files = []
for name in summary["input_files"]:
    source = Path(name)
    target = inputs / source.name
    shutil.copy2(source, target)
    files.append(target)
config = config_from_dict(summary)
config.section.auto_offset_enabled = False
start = time.monotonic()
result = Backend().process(files, config,
                           lambda text: print(text.encode("ascii", "backslashreplace").decode(), flush=True),
                           directory / "results")
saved = json.loads(result.summary_path.read_text(encoding="utf-8"))
assert saved["desktop_acquisition"]["input_verified"]
assert saved["desktop_acquisition"]["registration_mode"] == "automatic_calibration"
assert not saved["config"]["registration"]["use_six_station_pose_prior"]
assert all(not row.get("pose_prior_label") for row in saved["registration_metrics"])
assert not (inputs / "station_order.json").exists()
assert len(saved["desktop_acquisition"]["input_sha256"]) == len(files)
assert saved["config"]["section"] == config.as_dict()["section"]
assert result.merged_cloud_path.is_file()
assert result.section_cloud_path and result.section_cloud_path.is_file()
assert len(saved["registration_metrics"]) == len(files)
print(json.dumps({"ok": True, "seconds": round(time.monotonic() - start, 2),
                  "merged_points": result.merged_points, "section_points": result.section_points,
                  "quality": processing_quality(saved)["status"], "summary": str(result.summary_path)},
                 ensure_ascii=True), flush=True)
