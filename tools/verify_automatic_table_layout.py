"""Render the six-column automatic summary without running acquisition or measurement."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pointcloud_app.native.backend import sha256
from pointcloud_app.native.window import CAD_PAGE, DesktopWindow


def main():
    batch_path = sorted((ROOT / "outputs/automatic").glob("*/batch.json"))[-1]
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    records = batch["records"]
    original = json.dumps(records, sort_keys=True)
    protected = [batch_path, ROOT / "settings/workflow_profile.json"]
    hashes = {path: sha256(path) for path in protected if path.is_file()}
    destination = ROOT / "tmp/automatic_table_layout" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window = DesktopWindow(ROOT / "__no_autoload__")
    window._save_workflow_profile = lambda state=None: None
    window.auto_records = {record["id"]: record for record in records}
    window.workflow_mode = "automatic"
    window._refresh_workflow_mode_button()
    window.view_values[CAD_PAGE] = "automatic_measurements"
    window.select_page(CAD_PAGE)
    window.show()
    checks = []
    try:
        for width, height in ((1360, 860), (900, 620)):
            window.resize(width, height)
            QTest.qWait(400)
            app.processEvents()
            widget, table = window.table, window.table.table
            window.grab().save(str(destination / f"summary_{width}x{height}.png"))
            assert widget.measurement_summary
            assert table.columnCount() == 6
            assert table.textElideMode() == Qt.TextElideMode.ElideNone
            assert table.rowCount() * table.rowHeight(0) <= table.viewport().height()
            for column in range(6):
                assert table.item(0, column).textAlignment() == table.horizontalHeaderItem(column).textAlignment()
                texts = ["数据不足" if str(row[column]) == "INSUFFICIENT_DATA"
                         else "" if row[column] is None else str(row[column]) for row in widget.rows]
                required = max(
                    table.fontMetrics().horizontalAdvance(line)
                    for text in texts for line in text.split("\n")) + 28
                assert table.columnWidth(column) >= required, (column, table.columnWidth(column), required, texts)
            scroll = table.horizontalScrollBar()
            assert scroll.maximum() == 0 if width == 1360 else scroll.maximum() > 0
            scroll.setValue(scroll.maximum())
            QTest.qWait(150)
            window.grab().save(str(destination / f"summary_right_{width}x{height}.png"))
            scroll.setValue(0)
            checks.append({"viewport": [width, height], "columns": widget.columns,
                           "widths": [table.columnWidth(index) for index in range(6)],
                           "row_height": table.rowHeight(0), "page_size": widget.page_size,
                           "horizontal_scroll": scroll.maximum()})
        widget.move(1)
        app.processEvents()
        assert table.item(0, 0).text() == widget.rows[widget.page * widget.page_size][0]
        assert table.horizontalHeaderItem(3).textAlignment() == table.item(0, 3).textAlignment()
        window.grab().save(str(destination / "summary_next_page.png"))
        assert json.dumps(records, sort_keys=True) == original
        assert all(sha256(path) == digest for path, digest in hashes.items())
        (destination / "verification.json").write_text(json.dumps(
            {"ok": True, "batch": str(batch_path), "checks": checks,
             "records_unchanged": True, "inputs_unchanged": True, "pagination": True},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(destination / "verification.json"), flush=True)
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
