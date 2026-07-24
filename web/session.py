"""
AcquisitionSession — the web-side equivalent of the Qt BackendBridge.

The GUI's BackendBridge runs create_acquisition(config).run(frame_cb, status_cb)
on a daemon thread and forwards results via Qt signals. We do the exact same
lifecycle here, but forward frames to an asyncio broadcaster instead. The
acquisition backends (SyntheticAcquisition / HackRFAcquisition / USRPAcquisition)
are untouched and unaware of which frontend consumes them.

Threading model:
  - SoapySDR streaming is blocking and lives on its own OS thread (as in the Qt
    app). That thread MUST NOT block on network I/O, or SoapySDR overflows.
  - The frame callback fires on the acquisition thread. We hand the frame to the
    event loop via loop.call_soon_threadsafe and return immediately. Slow
    clients can never back-pressure the SDR thread; each client keeps only the
    latest frame (see hub.py).
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

# Import the *unchanged* backend from the analyzer repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acquisition import create_acquisition          # noqa: E402
from backend.models import AcquisitionConfig                # noqa: E402


class AcquisitionSession:
    """Start/stop a single live acquisition and broadcast its frames.

    This mirrors BackendBridge.start()/stop(), including the controlled
    restart-on-reconfigure behaviour: calling start() while running stops the
    current stream and queues the new config to start once teardown completes.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, hub):
        self._loop = loop
        self._hub = hub                      # web.hub.Hub, thread-safe fan-out
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._acquisition = None
        self._pending_config: AcquisitionConfig | None = None
        self._running = False

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, config: AcquisitionConfig):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                # Controlled restart: stop current, queue new config.
                self._pending_config = config
                if self._acquisition is not None:
                    self._acquisition.stop()
                return
            acquisition = create_acquisition(config)
            self._acquisition = acquisition
            self._running = True
            self._thread = threading.Thread(
                target=self._run, args=(acquisition,),
                name="sdr-acquisition-web", daemon=True,
            )
            self._thread.start()

    def _run(self, acquisition):
        try:
            acquisition.run(self._on_frame, self._on_status)
        except Exception as exc:  # matches BackendBridge.error path
            self._emit_status(f"error:{exc}")
        finally:
            with self._lock:
                pending = self._pending_config
                self._pending_config = None
                self._thread = None
                self._acquisition = None
                self._running = pending is not None
            if pending is not None:
                self.start(pending)

    def stop(self):
        with self._lock:
            self._pending_config = None
            acquisition = self._acquisition
            self._running = False
        if acquisition is not None:
            acquisition.stop()

    # -- callbacks fire on the acquisition thread --------------------------
    def _on_frame(self, frame):
        # Hand off to the event loop without blocking the SDR thread.
        self._loop.call_soon_threadsafe(self._hub.publish_frame, frame)

    def _on_status(self, status: str):
        self._emit_status(status)

    def _emit_status(self, status: str):
        self._loop.call_soon_threadsafe(self._hub.publish_status, status)
