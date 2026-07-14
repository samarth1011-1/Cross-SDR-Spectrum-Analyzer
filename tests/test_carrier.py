import unittest

import numpy as np

from backend.carrier import CarrierDetector


class CarrierDetectorTests(unittest.TestCase):
    def test_rising_and_falling_edges_use_confirmation_and_hysteresis(self):
        detector = CarrierDetector(
            rise_threshold_db=12.0,
            fall_threshold_db=8.0,
            confirmation_frames=2,
        )
        frequency = np.linspace(99e6, 101e6, 101)
        burst = np.full(101, -80.0)
        burst[49:52] = (-45.0, -40.0, -45.0)
        idle = np.full(101, -80.0)
        idle[50] = -75.0

        first = detector.update(frequency, burst)
        rising = detector.update(frequency, burst)
        first_idle = detector.update(frequency, idle)
        falling = detector.update(frequency, idle)

        self.assertFalse(first.detected)
        self.assertEqual(rising.event, "rising")
        self.assertTrue(rising.detected)
        self.assertLess(rising.lower_frequency, rising.upper_frequency)
        self.assertTrue(first_idle.detected)
        self.assertEqual(falling.event, "falling")
        self.assertFalse(falling.detected)

    def test_hysteresis_prevents_chatter_near_rising_threshold(self):
        detector = CarrierDetector(12.0, 8.0, confirmation_frames=1)
        frequency = np.arange(8, dtype=float)
        amplitude = np.full(8, -80.0)
        amplitude[3] = -67.0
        active = detector.update(frequency, amplitude)
        amplitude[3] = -70.0
        held = detector.update(frequency, amplitude)

        self.assertTrue(active.detected)
        self.assertTrue(held.detected)
        self.assertEqual(held.event, "")


if __name__ == "__main__":
    unittest.main()
