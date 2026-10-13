"""Run the desktop client directly from PyCharm."""

from __future__ import annotations

import os
from pathlib import Path

from main import main


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
