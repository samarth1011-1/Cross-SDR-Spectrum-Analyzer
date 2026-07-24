"""Device frequency/power calibration loading.

Schema (per device serial, falling back to the device 'default'):
    frequency_fixed_error_hz : float   constant offset, Hz
    frequency_ppm_error      : float   proportional error, parts per million
    power_offset_db          : float | None

Correction applied downstream (see acquisition.py):
    frequency_error_hz = fixed + driver_freq * ppm * 1e-6
    calibrated_freq    = driver_freq - frequency_error_hz

Sign convention (from the calibration procedure doc):
    offset_hz = observed_frequency_hz - reference_frequency_hz
    A POSITIVE error means the project displays too HIGH, so we SUBTRACT.
"""

from __future__ import annotations

import json
import os
from typing import Any


_CALIBRATION_FILENAME = "calibration.json"

# Every key the rest of the app may read, with safe zero/None defaults so a
# missing file, missing device, or missing key can never raise or mis-correct.
_RESULT_DEFAULTS: dict[str, Any] = {
    "frequency_fixed_error_hz": 0.0,
    "frequency_ppm_error": 0.0,
    "power_offset_db": None,
}


def _calibration_path() -> str:
    # calibration.json sits at the project root, one level above backend/.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), _CALIBRATION_FILENAME)


def _load_file() -> dict[str, Any]:
    path = _calibration_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # No calibration on disk (or corrupt) -> everyone gets zero correction.
        return {}


def load_device_calibration(device_type: str, serial: str) -> dict[str, Any]:
    """Return merged calibration for one device.

    Resolution order, later winning:
        1. hard-coded zero defaults (never mis-correct an unknown device)
        2. devices[TYPE]['default']
        3. devices[TYPE]['serials'][serial]   (only if serial is known)

    A serial that is absent from the file therefore inherits the device
    default (normally zeros), and an entirely unknown device type inherits
    the hard-coded zero defaults. Missing individual keys are back-filled
    from the zero defaults so callers can read any key unconditionally.
    """
    result: dict[str, Any] = dict(_RESULT_DEFAULTS)

    data = _load_file()
    devices = data.get("devices", {}) if isinstance(data, dict) else {}
    device_block = devices.get(str(device_type).upper(), {})
    if not isinstance(device_block, dict):
        return result

    # (2) device-level default
    default_cal = device_block.get("default", {})
    if isinstance(default_cal, dict):
        for key in _RESULT_DEFAULTS:
            if key in default_cal and default_cal[key] is not None:
                result[key] = default_cal[key]
            elif key in default_cal:
                result[key] = default_cal[key]  # explicit null (e.g. power)

    # (3) serial-specific override (only when the serial actually matches)
    serials = device_block.get("serials", {})
    if serial and isinstance(serials, dict):
        serial_cal = serials.get(str(serial))
        if isinstance(serial_cal, dict):
            for key in _RESULT_DEFAULTS:
                if key in serial_cal:
                    result[key] = serial_cal[key]

    return result