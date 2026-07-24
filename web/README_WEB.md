# Live web frontend

Serves the analyzer's live spectrum in a browser at `http://localhost:8000`.
Reuses `backend/` unchanged; only adds a WebSocket transport and a uPlot page.

## Install (in the same env as the analyzer)

```bash
pip install fastapi "uvicorn[standard]" websockets
```

`numpy` is already required by the backend.

## Run

From the repository root (so `backend` and `web` are both importable):

```bash
python -m web.server
```

Open `http://localhost:8000`, pick a device, set center/span/sample-rate/gain,
and press **Start acquisition**. With a HackRF or USRP connected and its drivers
on PATH (see INSTALL.md), select that device instead of Simulator.

- `host=0.0.0.0` in `server.py` lets other machines on the LAN reach a headless
  Pi at `http://<pi-ip>:8000`. Change to `127.0.0.1` to restrict to localhost.
- One RF front end = one shared stream: all open browsers view the same live
  spectrum. First Start opens the SDR; Stop closes it.

## Files

| File | Role |
|---|---|
| `web/server.py` | FastAPI app: `/` page, `/ws` WebSocket, control channel |
| `web/session.py` | Owns the acquisition thread; mirrors the Qt BackendBridge lifecycle |
| `web/hub.py` | Latest-frame-wins fan-out (prevents SDR back-pressure) |
| `web/protocol.py` | Binary frame codec + optional peak-preserving decimation |
| `web/static/index.html` | uPlot renderer + controls, instrument palette |
| `web/static/uPlot.*` | Vendored plotter (no runtime CDN dependency) |

## Notes for Pi deployment

- The SDR streaming thread never touches the network; a slow client only causes
  dropped frames, never SoapySDR overflow.
- If the waterfall/plot lags over a constrained link, use the **Link** dropdown
  to decimate server-side to 1500 or 800 columns (max within each bin, so
  narrowband carriers are preserved).
- uPlot draws on canvas and holds 30–40 fps of 4096-bin traces on Pi-class
  hardware; Plotly does not and was deliberately avoided.
