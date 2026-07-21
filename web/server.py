"""
FastAPI server exposing the live spectrum over a WebSocket at localhost:8000.

Endpoints:
  GET  /            -> static single-page monitor (web/static/index.html)
  WS   /ws          -> bidirectional:
                         server -> client: binary frames + JSON status
                         client -> server: JSON control commands (start/stop/etc.)

Run:
  cd <repo root>
  python -m web.server
  # then open http://localhost:8000

One shared AcquisitionSession drives all connected browsers (a spectrum
analyzer has one RF front end; every viewer sees the same live stream). The
first client to press Start opens the SDR; Stop closes it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models import AcquisitionConfig            # noqa: E402
from web.hub import Hub                                  # noqa: E402
from web.session import AcquisitionSession              # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="SDR Frequency Analyzer — Live Web")

_hub: Hub | None = None
_session: AcquisitionSession | None = None


@app.on_event("startup")
async def _startup():
    global _hub, _session
    loop = asyncio.get_running_loop()
    _hub = Hub()
    _session = AcquisitionSession(loop, _hub)


def _build_config(cmd: dict) -> AcquisitionConfig:
    """Translate a client 'start' command into an AcquisitionConfig.

    The client sends the same fields the Qt controls expose. We validate span
    <= sample_rate here, matching the GUI/backend rule, so a bad request fails
    fast with a clear status instead of deep in the DSP.
    """
    sample_rate = float(cmd["sample_rate"])
    span = float(cmd.get("span", sample_rate))
    span = min(span, sample_rate)
    return AcquisitionConfig(
        device_type=str(cmd["device_type"]).upper(),
        center_frequency=float(cmd["center_frequency"]),
        sample_rate=sample_rate,
        span=span,
        gain=float(cmd.get("gain", 0.0)),
        fft_size=int(cmd.get("fft_size", 4096)),
        channel=int(cmd.get("channel", 0)),
    )


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    assert _hub is not None and _session is not None
    await ws.accept()
    sub = _hub.subscribe()

    # Replay last known status so a fresh page shows the right state.
    if _hub.last_status:
        await ws.send_text(f'{{"type":"status","status":"{_hub.last_status}"}}')

    async def pump():
        """Drain this client's single-slot mailbox to the socket."""
        try:
            while True:
                payload = await sub.mailbox.get()
                if isinstance(payload, (bytes, bytearray)):
                    await ws.send_bytes(payload)
                else:
                    await ws.send_text(payload)
        except (WebSocketDisconnect, RuntimeError):
            pass

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            cmd = await ws.receive_json()
            action = cmd.get("action")
            if action == "start":
                try:
                    _session.start(_build_config(cmd))
                except Exception as exc:
                    await ws.send_text(
                        f'{{"type":"error","error":"bad start: {exc}"}}'
                    )
            elif action == "stop":
                _session.stop()
            elif action == "set_decimation":
                cols = cmd.get("columns")
                _hub.decimate_columns = int(cols) if cols else None
            elif action == "reset_min_hold":
                acq = getattr(_session, "_acquisition", None)
                if acq is not None and hasattr(acq, "reset_min_hold"):
                    acq.reset_min_hold()
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        _hub.unsubscribe(sub)


# Serve remaining static assets (uPlot vendored file, etc.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    # host=0.0.0.0 lets other machines on the LAN reach a headless Pi; use
    # 127.0.0.1 to restrict to localhost only.
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
