"""Native Python desktop entry. The legacy web app is not started."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from pointcloud_app.native.window import DesktopWindow

    parser = argparse.ArgumentParser(description="Point cloud desktop client")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    window = DesktopWindow(args.data_root)
    window.show()
    if args.result:
        from PySide6.QtCore import QTimer
        def load_when_ready():
            if window.job is None:
                window.load_result(args.result)
            else:
                QTimer.singleShot(200, load_when_ready)
        QTimer.singleShot(300, load_when_ready)
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
