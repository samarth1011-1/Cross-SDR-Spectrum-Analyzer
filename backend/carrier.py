"""Stateful carrier/burst detection with hysteresis."""

from __future__ import annotations

import numpy as np

from .models import CarrierDetection


class CarrierDetector:
    """Detect temporal rising/falling carrier edges from live spectrum frames."""

    def __init__(
        self,
        rise_threshold_db: float = 12.0,
        fall_threshold_db: float = 8.0,
        confirmation_frames: int = 2,
    ):
        self.rise_threshold_db = rise_threshold_db
        self.fall_threshold_db = fall_threshold_db
        self.confirmation_frames = confirmation_frames
        self.active = False
        self._rise_count = 0
        self._fall_count = 0

    def update(self, frequency, amplitude) -> CarrierDetection:
        frequency = np.asarray(frequency, dtype=np.float64)
        amplitude = np.asarray(amplitude, dtype=np.float64)
        peak_index = int(np.argmax(amplitude))
        peak_level = float(amplitude[peak_index])
        noise_floor = float(np.median(amplitude))
        margin = peak_level - noise_floor
        event = ""

        if self.active:
            self._rise_count = 0
            self._fall_count = self._fall_count + 1 if margin <= self.fall_threshold_db else 0
            if self._fall_count >= self.confirmation_frames:
                self.active = False
                self._fall_count = 0
                event = "falling"
        else:
            self._fall_count = 0
            self._rise_count = self._rise_count + 1 if margin >= self.rise_threshold_db else 0
            if self._rise_count >= self.confirmation_frames:
                self.active = True
                self._rise_count = 0
                event = "rising"

        region_threshold = noise_floor + self.fall_threshold_db
        left = peak_index
        right = peak_index
        while left > 0 and amplitude[left - 1] >= region_threshold:
            left -= 1
        while right + 1 < amplitude.size and amplitude[right + 1] >= region_threshold:
            right += 1

        return CarrierDetection(
            detected=self.active,
            event=event,
            peak_frequency=float(frequency[peak_index]),
            peak_level=peak_level,
            noise_floor=noise_floor,
            margin_db=margin,
            lower_frequency=float(frequency[left]),
            upper_frequency=float(frequency[right]),
        )
