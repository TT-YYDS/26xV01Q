"""Offline Plotly canvas. No web server or external content is loaded."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from plotly.offline import get_plotlyjs
from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView


class LocalPage(QWebEnginePage):
    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        return url.scheme() in {"file", "qrc", "about"}


class Bridge(QObject):
    received = Signal(dict)
    initialized = Signal()

    @Slot()
    def ready(self):
        self.initialized.emit()

    @Slot(str)
    def event(self, payload):
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                self.received.emit(data)
        except (ValueError, TypeError):
            pass


class Canvas(QWebEngineView):
    interaction = Signal(dict)
    render_error = Signal(str)
    rendered = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.storage = TemporaryDirectory(prefix="pointcloud_canvas_")
        self.directory = Path(self.storage.name)
        (self.directory / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
        shutil.copy2(Path(__file__).with_name("canvas.html"), self.directory / "index.html")
        self.setPage(LocalPage(self))
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        self.channel = QWebChannel(self.page())
        self.bridge = Bridge(self.channel)
        self.channel.registerObject("desktop", self.bridge)
        self.page().setWebChannel(self.channel)
        self.bridge.received.connect(self._event)
        self.bridge.initialized.connect(self._ready)
        self.loaded = False
        self.generation = 0
        self.rendered_generation = -1
        self.pending = None
        self.last_json = None
        self.load(QUrl.fromLocalFile(str(self.directory / "index.html")))

    def _ready(self):
        self.loaded = True
        if self.pending:
            self._dispatch()

    def show_figure(self, figure, mode="view"):
        self.generation += 1
        self.last_json = figure if isinstance(figure, str) else figure.to_json()
        path = self.directory / f"figure_{self.generation}.json"
        path.write_text(self.last_json, encoding="utf-8")
        self.pending = (path.name, self.generation, mode)
        if self.loaded:
            self._dispatch()

    def invalidate(self):
        self.generation += 1
        self.pending = None

    def _dispatch(self):
        name, generation, mode = self.pending
        self.page().runJavaScript(
            f"window.renderFile({json.dumps(name)}, {generation}, {json.dumps(mode)});")

    def _event(self, data):
        if data.get("generation") != self.generation:
            return
        if data.get("type") == "error":
            self.render_error.emit(data.get("message", "绘图失败"))
        elif data.get("type") == "rendered":
            self.rendered_generation = self.generation
            # Only the last render still needs its payload on disk.
            for path in self.directory.glob("figure_*.json"):
                if path.name != f"figure_{self.generation}.json":
                    path.unlink(missing_ok=True)
            self.rendered.emit()
        elif self.rendered_generation == self.generation:
            self.interaction.emit(data)
