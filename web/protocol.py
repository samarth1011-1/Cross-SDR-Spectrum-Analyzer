"""
Binary wire format for SpectrumFrame.

Why binary and not JSON:
  A frame carries up to 5 float32 arrays of `fft_size` bins (default 4096).
  json.dumps of ~20k floats is CPU-heavy on a Raspberry Pi and inflates the
  payload 5-10x. We instead send a small JSON header (scalars + peaks, which
  are few) followed by raw little-endian float32 array buffers. The browser
  decodes the buffers straight into Float32Array with zero parsing cost.

Wire layout (one WebSocket *binary* message per frame):

    [ 4 bytes  ] uint32  header_len (little-endian)
    [ header_len bytes ] UTF-8 JSON header (see below)
    [ N*4 bytes ] float32 frequency
    [ N*4 bytes ] float32 amplitude   (live / clear-write)
    [ N*4 bytes ] float32 max_hold
    [ N*4 bytes ] float32 min_hold
    [ N*4 bytes ] float32 average

The header names the arrays, their length N, and byte order so the client can
slice the trailing buffer deterministically. Scalars (center freq, span, rbw,
noise floor, etc.), the peak list, and detected carriers ride in the header
because they are small and benefit from being human-readable during debugging.
"""

from __future__ import annotations

import json
import struct

import numpy as np

# Traces are sent in this fixed order after the header.
_TRACE_ORDER = ("frequency", "amplitude", "max_hold", "min_hold", "average")


def _decimate_max(freq: np.ndarray, arrays: dict, columns: int) -> tuple[np.ndarray, dict]:
    """Peak-preserving decimation to `columns` screen bins.

    A monitoring display rarely has more than ~1500 horizontal pixels, so
    shipping 4096 bins over a constrained link wastes bandwidth. We reduce by
    taking the *max* within each output bin rather than subsampling, so a
    narrowband carrier that falls between kept samples is never dropped. This
    is off by default; enable it only when the link is the bottleneck.
    """
    n = freq.size
    if columns >= n or columns <= 0:
        return freq, arrays
    # Bin edges over the index range; np.add.reduceat groups contiguous slices.
    edges = np.linspace(0, n, columns + 1).astype(np.int64)
    starts = edges[:-1]
    # Guard against empty groups when columns is close to n.
    starts = np.clip(starts, 0, n - 1)

    out_freq = freq[starts]
    out = {}
    for name, a in arrays.items():
        # Max within each group preserves peaks; reduceat handles the grouping.
        out[name] = np.maximum.reduceat(a, starts).astype(np.float32, copy=False)
    return out_freq.astype(np.float32, copy=False), out


def encode_frame(frame, decimate_columns: int | None = None) -> bytes:
    """Serialize a SpectrumFrame to the binary wire format.

    `decimate_columns`: if set, peak-decimate every trace to this many columns
    before sending. None (default) sends full resolution.
    """
    arrays = {
        "frequency": np.asarray(frame.frequency, dtype=np.float32),
        "amplitude": np.asarray(frame.amplitude, dtype=np.float32),
        "max_hold": np.asarray(frame.max_hold, dtype=np.float32),
        "min_hold": np.asarray(frame.min_hold, dtype=np.float32),
        "average": np.asarray(frame.average, dtype=np.float32),
    }
    freq = arrays.pop("frequency")

    if decimate_columns:
        freq, arrays = _decimate_max(freq, arrays, decimate_columns)

    arrays = {"frequency": freq, **arrays}
    n = int(freq.size)

    # Peaks and carriers are small; carry them in the JSON header.
    peaks = [
        {"id": p.id, "frequency": float(p.frequency),
         "amplitude": float(p.amplitude), "bin_index": int(p.bin_index)}
        for p in (frame.peaks or [])
    ]
    carriers = []
    for c in (frame.carriers or []):
        # CarrierDetectionEngine emits objects with left_bin/right_bin.
        left = getattr(c, "left_bin", None)
        right = getattr(c, "right_bin", None)
        if left is not None and right is not None:
            carriers.append({"left_bin": int(left), "right_bin": int(right)})

    header = {
        "type": "frame",
        "n": n,
        "dtype": "float32",
        "order": list(_TRACE_ORDER),
        "center_frequency": float(frame.center_frequency),
        "sample_rate": float(frame.sample_rate),
        "span": float(frame.span),
        "rbw": float(frame.rbw),
        "fft_size": int(frame.fft_size),
        "frame_count": int(frame.frame_count),
        "timestamp": float(frame.timestamp),
        "noise_floor": float(frame.noise_floor),
        "channel_power": float(frame.channel_power),
        "bandwidth": float(frame.bandwidth),
        "device_name": str(frame.device_name),
        "peaks": peaks,
        "carriers": carriers,
    }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    parts = [struct.pack("<I", len(header_bytes)), header_bytes]
    for name in _TRACE_ORDER:
        # Ensure contiguous little-endian float32 for a clean .tobytes().
        parts.append(np.ascontiguousarray(arrays[name], dtype="<f4").tobytes())
    return b"".join(parts)


def encode_status(text: str) -> str:
    """Status/error messages go as a small JSON *text* message."""
    return json.dumps({"type": "status", "status": text})


def encode_error(text: str) -> str:
    return json.dumps({"type": "error", "error": text})
