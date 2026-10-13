"""Runtime helpers shared by the local desktop host."""

from __future__ import annotations

import json
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

STATIONS = ("reference", "minus10", "minus28", "plus37", "minus50", "plus47")


def choose_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def build_server_arguments(project_root: Path, port: int) -> list[str]:
    return [
        "-u", "-m", "streamlit", "run", str(project_root / "app.py"),
        "--server.address=127.0.0.1", f"--server.port={port}",
        "--server.headless=true", "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none", "--client.toolbarMode=minimal",
    ]


def is_workspace_url(url: str, port: int) -> bool:
    try:
        parts = urlsplit(url)
        return (
            parts.scheme == "http"
            and parts.hostname == "127.0.0.1"
            and parts.port == port
            and parts.username is None
            and parts.password is None
        )
    except ValueError:
        return False


def station_labels(count: int) -> tuple[str, ...]:
    if count not in (5, 6):
        raise ValueError("固定工位要求 5 或 6 个点云文件。")
    return STATIONS[:count]


def validate_station_mapping(files: list[Path], mapping: dict[str, str]) -> None:
    expected = set(station_labels(len(files)))
    if len({path.name for path in files}) != len(files):
        raise ValueError("点云文件名不能重复。")
    if len({path.resolve().parent for path in files}) != 1:
        raise ValueError("一组点云必须来自同一目录。")
    if set(mapping) != {path.name for path in files}:
        raise ValueError("工位映射必须覆盖当前目录的全部点云文件。")
    if set(mapping.values()) != expected:
        raise ValueError("工位必须唯一，且必须包含全部固定工位标签。")


def load_station_mapping(files: list[Path]) -> dict[str, str]:
    station_labels(len(files))
    manifest = files[0].parent / "station_order.json"
    if manifest.exists():
        try:
            views = json.loads(manifest.read_text(encoding="utf-8"))["views"]
            if not isinstance(views, list) or len(views) != len(files):
                raise ValueError("工位记录数量与点云数量不一致")
            mapping = {item["file"]: item["station"] for item in views}
            validate_station_mapping(files, mapping)
            return mapping
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"现有 station_order.json 无效，请重新确认：{error}") from error
    mapping = {}
    for path in files:
        matches = [label for label in station_labels(len(files)) if label in path.stem.lower()]
        if len(matches) == 1:
            mapping[path.name] = matches[0]
    return mapping


def suggest_station_mapping(files: list[Path]) -> tuple[dict[str, str], str]:
    """Return an unconfirmed suggestion without modifying acquisition files."""
    from .pipeline import (
        _STATION_ORDER_NUMBER_PATTERN, order_files_with_station_pose_prior_with_source,
    )

    expected = set(station_labels(len(files)))
    if len({p.resolve().parent for p in files}) != 1 or len({p.name for p in files}) != len(files):
        raise ValueError("点云必须来自同一目录，且文件名不能重复。")
    if (files[0].parent / "station_order.json").exists():
        return load_station_mapping(files), "station_order.json"
    labels = [[label for label in expected if label in p.stem.lower()] for p in files]
    if any(labels):
        if any(len(matches) != 1 for matches in labels):
            raise ValueError("文件中的工位标签不完整或有歧义，请手动选择工位。")
        mapping = {p.name: matches[0] for p, matches in zip(files, labels)}
        validate_station_mapping(files, mapping)
        return mapping, "filename_station_label"
    numbers = [_STATION_ORDER_NUMBER_PATTERN.search(p.stem) for p in files]
    if any(numbers):
        if not all(numbers):
            raise ValueError("只有部分文件包含采集编号，请手动选择工位。")
        values = {int(match.group(1)) for match in numbers}
        if values not in (set(range(len(files))), set(range(1, len(files) + 1))):
            raise ValueError("采集编号必须连续且唯一（0 起始或 1 起始）。")
    elif len({p.stat().st_mtime_ns for p in files}) != len(files):
        raise ValueError("文件修改时间相同，无法可靠恢复先后顺序，请手动选择工位。")
    ordered, stations, source = order_files_with_station_pose_prior_with_source(files)
    mapping = dict(zip([p.name for p in ordered], stations))
    validate_station_mapping(files, mapping)
    return mapping, source


def save_station_mapping(files: list[Path], mapping: dict[str, str], *, suggestion_source=None) -> Path:
    validate_station_mapping(files, mapping)
    directory = files[0].resolve().parent
    now = datetime.now().astimezone()
    target = directory / "station_order.json"
    if target.exists():
        shutil.copy2(target, directory / f"station_order.{now:%Y%m%d_%H%M%S_%f}.bak.json")
    by_station = {station: name for name, station in mapping.items()}
    payload = {
        "schema_version": 1,
        "confirmed": True,
        "confirmed_at": now.isoformat(),
        "source": "desktop_operator_confirmation",
        "views": [
            {"station": station, "file": by_station[station]}
            for station in station_labels(len(files))
        ],
    }
    if suggestion_source:
        payload["suggestion_source"] = suggestion_source
    temporary = directory / f".station_order.{now:%Y%m%d_%H%M%S_%f}.tmp"
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def discover_stable_input_groups(
    data_root: Path,
    *,
    minimum_file_age_seconds: float = 1.0,
) -> dict[str, list[Path]]:
    """Discover complete, unchanged-on-disk PLY groups for automatic intake.

    A direct folder of PLY files is one group.  Otherwise the existing
    one-level group discovery convention is used.  Files younger than the
    stability window are ignored so a camera transfer cannot be processed
    while it is still writing.
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        return {}
    if minimum_file_age_seconds < 0:
        raise ValueError("文件稳定时间不能为负数")

    direct = sorted(root.glob("*.ply"), key=lambda item: item.name.lower())
    if direct:
        from .pipeline import discover_groups

        candidates = {root.name: direct, **discover_groups(root)}
    else:
        from .pipeline import discover_groups

        candidates = discover_groups(root)

    now = time.time()
    stable: dict[str, list[Path]] = {}
    for name, files in candidates.items():
        if len(files) not in (5, 6, 7, 8):
            continue
        try:
            if any(
                not path.is_file()
                or now - path.stat().st_mtime < minimum_file_age_seconds
                for path in files
            ):
                continue
        except OSError:
            continue
        stable[name] = [Path(path).resolve() for path in files]
    return stable
