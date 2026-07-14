import numpy as np

from .dsp import SPECTRUM_FLOOR_DBFS
from .models import TraceData


MIN_HOLD_FLOOR_GUARD_DB = 1.0


class TraceEngine:
    def __init__(self):
        self.clear()

    def update(self, spectrum):
        live = np.asarray(spectrum.amplitude, dtype=np.float32).copy()
        if self.max_hold is None or self.max_hold.shape != live.shape:
            self.max_hold = live.copy()
            self.average = live.copy()
            self.frames = 1
        else:
            np.maximum(self.max_hold, live, out=self.max_hold)
            self.average += (live - self.average) / (self.frames + 1)
            self.frames += 1

        # Values at the DSP clamp are numerical underflow/cancellation, not a
        # measurable receiver level. Letting one such value into a persistent
        # minimum pins that bin to -140 dBFS forever.
        valid_min = np.isfinite(live) & (
            live > SPECTRUM_FLOOR_DBFS + MIN_HOLD_FLOOR_GUARD_DB
        )
        if self.min_hold is None or self.min_hold.shape != live.shape:
            if np.any(valid_min):
                fallback = float(np.median(live[valid_min]))
                self.min_hold = np.where(valid_min, live, fallback).astype(np.float32)
            else:
                self.min_hold = live.copy()
        else:
            candidates = np.where(valid_min, live, np.inf)
            np.minimum(self.min_hold, candidates, out=self.min_hold)

        return TraceData(
            frequency=spectrum.frequency,
            live=live,
            max_hold=self.max_hold.copy(),
            min_hold=self.min_hold.copy(),
            average=self.average.copy(),
            frame_count=self.frames,
        )

    def clear(self):
        self.max_hold = None
        self.min_hold = None
        self.average = None
        self.frames = 0

    def reset_min_hold(self):
        """Start a new minimum acquisition without resetting other traces."""
        self.min_hold = None
