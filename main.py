"""PyCharm 入口：启动本地点云拼接与 CAD 测量客户端。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RUNTIME_PYTHON = Path(r"D:\dev\anacondalenvs\26xv01q\python.exe")
if not RUNTIME_PYTHON.is_file():
    RUNTIME_PYTHON = Path(sys.executable)

def main() -> None:
    if Path(sys.executable).resolve() != RUNTIME_PYTHON.resolve():
        print(f"Switching to application interpreter: {RUNTIME_PYTHON}", flush=True)
        os.execv(
            str(RUNTIME_PYTHON),
            [str(RUNTIME_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    from desktop_client import main as start_application

    start_application()


if __name__ == "__main__":
    main()
