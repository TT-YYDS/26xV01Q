"""Persistent manual settings used as the automatic-workflow template."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = 1
PROFILE_FILENAME = "workflow_profile.json"


def profile_path(project_root: Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    return root / "settings" / PROFILE_FILENAME


def empty_profile() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "workflow_mode": "manual",
        "source": {"kind": "folder", "path": ""},
        "config": None,
        "cad": {"dxf": "", "selection": {}, "step": 0.05},
        "updated_at": None,
    }


def load_profile(path: Path | None = None) -> tuple[dict[str, Any], str | None]:
    target = profile_path() if path is None else Path(path)
    if not target.exists():
        return empty_profile(), None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("模板根节点必须是对象")
        if int(payload.get("schema_version", 0)) != PROFILE_SCHEMA_VERSION:
            raise ValueError("模板版本不兼容")
        merged = empty_profile()
        merged.update(copy.deepcopy(payload))
        if not isinstance(merged.get("source"), dict):
            raise ValueError("source 必须是对象")
        if not isinstance(merged.get("cad"), dict):
            raise ValueError("cad 必须是对象")
        return merged, None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return empty_profile(), f"自动流程模板读取失败：{error}"


def build_profile(
    *,
    workflow_mode: str,
    source_kind: str,
    source_path: Path | str | None,
    config: Any | None,
    dxf: Path | str | None,
    selection: dict[str, Any] | None,
    cad_step: float,
) -> dict[str, Any]:
    if workflow_mode not in {"manual", "automatic"}:
        raise ValueError("工作模式必须是 manual 或 automatic")
    if source_kind not in {"folder", "camera"}:
        raise ValueError("输入源必须是指定文件夹或 3D 相机传输目录")
    if not isinstance(cad_step, (int, float)) or cad_step <= 0:
        raise ValueError("CAD 采样间距必须大于 0")
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "workflow_mode": workflow_mode,
        "source": {
            "kind": source_kind,
            "path": str(Path(source_path).resolve()) if source_path else "",
        },
        "config": copy.deepcopy(config.as_dict()) if config is not None else None,
        "cad": {
            "dxf": str(Path(dxf).resolve()) if dxf else "",
            "selection": copy.deepcopy(selection or {}),
            "step": float(cad_step),
        },
        "updated_at": datetime.now().astimezone().isoformat(),
    }


def save_profile(payload: dict[str, Any], path: Path | None = None) -> Path:
    target = profile_path() if path is None else Path(path)
    if not isinstance(payload, dict):
        raise ValueError("自动流程模板必须是对象")
    if int(payload.get("schema_version", 0)) != PROFILE_SCHEMA_VERSION:
        raise ValueError("自动流程模板版本不兼容")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
