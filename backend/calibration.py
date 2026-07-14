"""Load optional, device-specific measurement calibration values."""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "calibration.json"


def load_device_calibration(device_type: str, serial: str = "") -> dict[str, float | None]:
    """Return wildcard calibration merged with a serial-specific override."""
    path = Path(os.environ.get("FREQANALYZER_CALIBRATION", DEFAULT_PATH))
    result: dict[str, float | None] = {
        "frequency_axis_offset_hz": 0.0,
        "power_offset_db": None,
    }
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        device = document.get("devices", {}).get(device_type.upper(), {})
        result.update(device.get("default", {}))
        if serial:
            result.update(device.get("serials", {}).get(serial, {}))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, AttributeError):
        return result
    return result
