"""PyCharm 右键启动 Streamlit 点云拼接网页。"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_FILE = PROJECT_ROOT / "app.py"
URL = "http://127.0.0.1:8501"
HEALTH_URL = f"{URL}/_stcore/health"


def server_is_ready() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def open_application_browser() -> None:
    if os.environ.get("POINTCLOUD_NO_BROWSER") != "1":
        webbrowser.open_new_tab(URL)


def open_browser_when_ready(server_process: subprocess.Popen[object]) -> None:
    """等待 Streamlit 就绪后打开浏览器，避免首次启动时打开空白页面。"""
    for _ in range(120):
        if server_process.poll() is not None:
            return
        if server_is_ready():
            open_application_browser()
            return
        time.sleep(0.25)


def main() -> None:
    if not APP_FILE.is_file():
        raise FileNotFoundError(f"未找到网页入口文件：{APP_FILE}")

    if server_is_ready():
        print(f"检测到已有点云拼接网页，直接复用：{URL}", flush=True)
        return

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("正在启动点云拼接网页，请稍候...", flush=True)
    print(f"浏览器地址：{URL}", flush=True)
    server_process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    threading.Thread(target=open_browser_when_ready, args=(server_process,), daemon=True).start()

    try:
        raise SystemExit(server_process.wait())
    except KeyboardInterrupt:
        print("\n正在停止网页服务...", flush=True)
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
