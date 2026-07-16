
"""
carrier_detection.py

Professional SATCOM Carrier Detection Engine
===========================================

Phase 1:
- Smooth FFT spectrum
- Estimate noise floor
- Detect occupied carrier envelopes
- Hysteresis edge tracking
- Reject tiny regions

Future phases:
- Carrier IDs
- Occupied Bandwidth
- Channel Power
- Tracking
"""

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class CarrierRegion:
    left_bin: int
    right_bin: int
    center_bin: int
    peak_bin: int
    peak_power: float
    noise_floor: float
    confidence: float = 1.0


class CarrierDetectionEngine:
    def __init__(
        self,
        enter_threshold_db: float = 6.0,
        exit_threshold_db: float = 3.5,
        smoothing_window: int = 11,
        minimum_width_bins: int = 40,
        merge_gap_bins: int = 25,
    ):
        self.enter_threshold_db = enter_threshold_db
        self.exit_threshold_db = exit_threshold_db
        self.smoothing_window = smoothing_window
        self.minimum_width_bins = minimum_width_bins
        self.merge_gap_bins = merge_gap_bins

    def smooth(self, spectrum: np.ndarray) -> np.ndarray:
        kernel = np.ones(self.smoothing_window, dtype=float)
        kernel /= kernel.sum()
        return np.convolve(spectrum, kernel, mode="same")

    def estimate_noise(self, spectrum: np.ndarray) -> float:
        return float(np.percentile(spectrum, 20))

    def detect(self, amplitude: np.ndarray):
        smoothed = self.smooth(amplitude)

        noise = self.estimate_noise(smoothed)

        enter_level = noise + self.enter_threshold_db
        exit_level = noise + self.exit_threshold_db

        carriers = []

        inside = False
        start = 0

        for i, value in enumerate(smoothed):

            if not inside:
                if value >= enter_level:
                    inside = True
                    start = i
            else:
                if value <= exit_level:

                    end = i

                    if (end - start) >= self.minimum_width_bins:

                        segment = smoothed[start:end]

                        if len(segment) == 0:
                            inside = False
                            continue

                        peak_rel = int(np.argmax(segment))
                        peak_bin = start + peak_rel

                        carriers.append(
                            CarrierRegion(
                                left_bin=start,
                                right_bin=end,
                                center_bin=(start + end) // 2,
                                peak_bin=peak_bin,
                                peak_power=float(smoothed[peak_bin]),
                                noise_floor=noise,
                            )
                        )

                    inside = False

        if inside:

            end = len(smoothed) - 1

            if (end - start) >= self.minimum_width_bins:

                segment = smoothed[start:end + 1]

                if len(segment):

                    peak_rel = int(np.argmax(segment))
                    peak_bin = start + peak_rel

                    carriers.append(
                        CarrierRegion(
                            left_bin=start,
                            right_bin=end,
                            center_bin=(start + end) // 2,
                            peak_bin=peak_bin,
                            peak_power=float(smoothed[peak_bin]),
                            noise_floor=noise,
                        )
                    )

        return carriers
