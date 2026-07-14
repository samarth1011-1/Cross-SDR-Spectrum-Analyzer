"""Dedicated time-domain carrier activity and burst-edge window."""

from __future__ import annotations

from collections import deque

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CarrierActivityWindow(QWidget):
    def __init__(self, parent=None, history_length: int = 600):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Carrier Activity - Rising/Falling Edges")
        self.resize(850, 420)
        self._times = deque(maxlen=history_length)
        self._margins = deque(maxlen=history_length)
        self._rises = deque(maxlen=history_length)
        self._falls = deque(maxlen=history_length)
        self._start = None

        layout = QVBoxLayout(self)
        self.status = QLabel("Waiting for spectrum frames")
        layout.addWidget(self.status)
        self.plot = pg.PlotWidget(background="#000000")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Elapsed time", units="s")
        self.plot.setLabel("left", "Carrier above noise", units="dB")
        self.margin_curve = self.plot.plot(pen=pg.mkPen("#00F0FF", width=2))
        self.rise_points = pg.ScatterPlotItem(
            symbol="t1", size=13, brush=pg.mkBrush("#00FF66"), pen=None
        )
        self.fall_points = pg.ScatterPlotItem(
            symbol="t", size=13, brush=pg.mkBrush("#FF4444"), pen=None
        )
        self.plot.addItem(self.rise_points)
        self.plot.addItem(self.fall_points)
        layout.addWidget(self.plot)

    def update_frame(self, frame):
        carrier = getattr(frame, "carrier", None)
        if carrier is None:
            return
        if self._start is None:
            self._start = frame.timestamp
        elapsed = frame.timestamp - self._start
        self._times.append(elapsed)
        self._margins.append(carrier.margin_db)
        if carrier.event == "rising":
            self._rises.append((elapsed, carrier.margin_db))
        elif carrier.event == "falling":
            self._falls.append((elapsed, carrier.margin_db))

        self.margin_curve.setData(list(self._times), list(self._margins))
        self.rise_points.setData(
            [point[0] for point in self._rises], [point[1] for point in self._rises]
        )
        self.fall_points.setData(
            [point[0] for point in self._falls], [point[1] for point in self._falls]
        )
        state = "ACTIVE" if carrier.detected else "IDLE"
        event = f" | {carrier.event.upper()} EDGE" if carrier.event else ""
        self.status.setText(
            f"{state}{event} | {carrier.peak_frequency/1e6:.6f} MHz | "
            f"{carrier.margin_db:.1f} dB above noise"
        )
