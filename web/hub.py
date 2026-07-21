"""
Hub — fan-out of frames/status to all connected browsers.

Back-pressure is the whole point of this file. If a browser or its network link
is slower than the SDR frame rate, we must NOT let its queue grow (latency
climbs without bound) and must NOT slow the SDR thread (SoapySDR overflows).

Solution: each subscriber holds a single-slot mailbox (asyncio.Queue maxsize=1).
publish_frame overwrites the slot with the newest encoded frame, discarding any
undelivered previous frame. Every client therefore always renders the most
recent spectrum and simply skips frames it was too slow to receive. This is the
correct behaviour for a live monitor: freshness beats completeness.
"""

from __future__ import annotations

import asyncio

from .protocol import encode_frame, encode_status, encode_error


class Subscriber:
    def __init__(self) -> None:
        # maxsize=1 => latest-frame-wins.
        self.mailbox: asyncio.Queue = asyncio.Queue(maxsize=1)

    def offer(self, payload) -> None:
        """Replace any pending payload with the newest one (never blocks)."""
        q = self.mailbox
        if q.full():
            try:
                q.get_nowait()          # drop the stale frame
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass                         # racing producer; newest wins next tick


class Hub:
    def __init__(self) -> None:
        self._subs: set[Subscriber] = set()
        self._last_status: str | None = None
        # Per-connection decimation is set by the client; None = full res.
        self.decimate_columns: int | None = None

    def subscribe(self) -> Subscriber:
        sub = Subscriber()
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        self._subs.discard(sub)

    # -- called on the event loop via call_soon_threadsafe -----------------
    def publish_frame(self, frame) -> None:
        if not self._subs:
            return
        # Encode once for all subscribers (they share the same decimation).
        try:
            payload = encode_frame(frame, self.decimate_columns)
        except Exception as exc:
            payload = encode_error(f"encode failed: {exc}")
        for sub in self._subs:
            sub.offer(payload)

    def publish_status(self, status: str) -> None:
        self._last_status = status
        if status.startswith("error:"):
            msg = encode_error(status[len("error:"):])
        else:
            msg = encode_status(status)
        for sub in self._subs:
            sub.offer(msg)

    @property
    def last_status(self) -> str | None:
        return self._last_status
