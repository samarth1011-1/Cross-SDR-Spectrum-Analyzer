"""Measurements derived from each live spectrum."""

import numpy as np

from .models import MeasurementData


class MeasurementEngine:
    def update(self, traces) -> MeasurementData:
        amplitude = np.asarray(traces.live, dtype=np.float64)
        frequency = np.asarray(traces.frequency, dtype=np.float64)
        if amplitude.size == 0:
            raise ValueError("Cannot measure an empty spectrum")

        noise_floor = float(np.median(amplitude))
        power = np.power(10.0, amplitude / 10.0)
        total_power = float(np.sum(power))
        channel_power = float(10.0 * np.log10(max(total_power, 1e-24)))

        cumulative = np.cumsum(power)
        lower = min(int(np.searchsorted(cumulative, total_power * 0.005)), len(frequency) - 1)
        upper = min(int(np.searchsorted(cumulative, total_power * 0.995)), len(frequency) - 1)
        occupied_bandwidth = float(max(0.0, frequency[upper] - frequency[lower]))

        return MeasurementData(
            peak_frequency=float(frequency[int(np.argmax(amplitude))]),
            peak_amplitude=float(np.max(amplitude)),
            noise_floor=noise_floor,
            occupied_bandwidth=occupied_bandwidth,
            channel_power=channel_power,
        )
