"""Run a real desktop processing pass and verify timing metadata."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointcloud_app.native.backend import Backend, config_from_dict


def _latest_summary_with_region() -> Path:
    candidates = sorted((ROOT / "outputs" / "group_1").glob("*/run_summary.json"), reverse=True)
    for path in candidates:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        region = (((summary.get("config") or {}).get("section") or {}).get("target_region"))
        if region:
            return path
    raise RuntimeError("没有找到带主轴框选 target_region 的 group_1 历史结果")


def main() -> int:
    source_summary = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_summary_with_region()
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    files = [Path(value) for value in summary["input_files"]]
    config = config_from_dict(summary)
    output_root = ROOT / "tmp" / "processing_timing_verification"
    output_root.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    started = time.perf_counter()
    result = Backend().process(files, config, messages.append, output_root)
    elapsed = time.perf_counter() - started
    saved = json.loads(result.summary_path.read_text(encoding="utf-8"))

    timing = saved.get("timing") or {}
    stages = timing.get("stages") or {}
    desktop_timing = ((saved.get("desktop_acquisition") or {}).get("timing") or {})
    required_stages = {"single_view_denoise", "registration", "fusion", "section", "output_rendering"}
    missing = sorted(required_stages - set(stages))
    if missing:
        raise AssertionError(f"缺少处理阶段耗时: {missing}")
    if not desktop_timing.get("input_hashing_seconds", 0) >= 0:
        raise AssertionError("缺少输入指纹耗时")
    cleanup = saved["section_consensus"].get("final_component_cleanup")
    if not cleanup or cleanup.get("removed_points", -1) < 0:
        raise AssertionError("缺少融合后孤立组件清理记录")
    if saved["section_consensus"].get("fusion_input_denoised") is not True:
        raise AssertionError("融合必须复用单视角去噪点云")

    report = {
        "ok": True,
        "source_summary": str(source_summary),
        "summary": str(result.summary_path),
        "wall_seconds": round(elapsed, 3),
        "pipeline_timing": timing,
        "desktop_timing": desktop_timing,
        "merged_points": saved.get("merged_points"),
        "section_points": saved.get("section_points"),
        "source_views": saved.get("section_consensus", {}).get("source_views"),
        "post_fusion_denoise_applied": saved.get("section_consensus", {}).get("post_fusion_denoise_applied"),
        "removed_section_points": saved.get("section_consensus", {}).get("removed_section_points"),
        "progress_messages": len(messages),
    }
    (output_root / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
