"""Continuous HackRF One RX acquisition through SoapySDR.

Scope is deliberately narrow: SIMULATOR + HackRF One, receive only.
USRP and ADALM-Pluto support has been removed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from .controller import AnalyzerPipeline
from .models import AcquisitionConfig, DeviceInfo


HACKRF_DRIVER = "hackrf"

# HackRF One hardware limits.
HACKRF_MIN_FREQ = 1e6
HACKRF_MAX_FREQ = 6e9
HACKRF_MIN_RATE = 2e6
HACKRF_MAX_RATE = 20e6

# Front-end amp. Keep OFF for signal-generator work: +14 dB into a HackRF that is
# already fed a strong CW tone will compress the ADC and can damage the LNA.
HACKRF_AMP_DB = 0.0
HACKRF_LNA_DB = 24.0        # 0..40 dB, 8 dB steps
HACKRF_VGA_MAX = 62.0       # 0..62 dB, 2 dB steps

# libhackrf's device list is not reentrant. Two acquisition threads calling
# enumerate()/make() concurrently will race hackrf_device_list_open() and both
# fail. Serialize the entire bring-up sequence process-wide.
_OPEN_LOCK = threading.Lock()

# Set True while bringing up hardware. Makes SoapyHackRF print
#   [INFO]  Opening HackRF One #0 ...
#   [DEBUG] setGain VGA RX, channel 0, gain 20
#   [DEBUG] Start RX
#   [ERROR] hackrf_start_rx() failed -- ...  /  Activate RX Stream Failed.
SOAPY_DEBUG_LOG = True


class AcquisitionError(RuntimeError):
    pass


def create_acquisition(config: AcquisitionConfig):
    device_type = config.device_type.upper()
    if device_type == "SIMULATOR":
        return SyntheticAcquisition(config)
    if device_type == "HACKRF":
        return HackRFAcquisition(config)
    raise AcquisitionError(
        f"Unsupported device type '{config.device_type}'. This build supports "
        "SIMULATOR and HACKRF only."
    )


# ---------------------------------------------------------------------------
# Simulator (unchanged)
# ---------------------------------------------------------------------------
class SyntheticAcquisition:
    """Phase-continuous IQ generator for testing the complete live pipeline."""

    def __init__(self, config: AcquisitionConfig, frame_rate: float = 30.0):
        self.config = config
        self.frame_rate = frame_rate
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(
        self,
        frame_callback: Callable[[object], None],
        status_callback: Callable[[str], None] | None = None,
    ):
        if status_callback:
            status_callback("Connected: Built-in IQ Simulator")
        pipeline = AnalyzerPipeline(self.config, "Built-in IQ Simulator")
        rng = np.random.default_rng(20260711)
        sample_index = 0
        started = time.monotonic()
        deadline = started
        visible_span = min(self.config.span, self.config.sample_rate)
        offset_1 = 0.17 * visible_span
        offset_2 = -0.28 * visible_span

        try:
            while not self._stop_event.is_set():
                elapsed = time.monotonic() - started
                indices = sample_index + np.arange(self.config.fft_size, dtype=np.float64)
                amplitude_1 = 0.30 + 0.16 * np.sin(2.0 * np.pi * elapsed / 3.0)
                amplitude_2 = 0.16 if int(elapsed * 2.0) % 2 == 0 else 0.045
                tone_1 = amplitude_1 * np.exp(
                    2j * np.pi * offset_1 * indices / self.config.sample_rate
                )
                tone_2 = amplitude_2 * np.exp(
                    2j * np.pi * offset_2 * indices / self.config.sample_rate
                )
                noise = 0.006 * (
                    rng.standard_normal(self.config.fft_size)
                    + 1j * rng.standard_normal(self.config.fft_size)
                )
                samples = (tone_1 + tone_2 + noise).astype(np.complex64)
                frame_callback(pipeline.process(samples))
                sample_index += self.config.fft_size

                deadline += 1.0 / self.frame_rate
                remaining = deadline - time.monotonic()
                if remaining < 0:
                    deadline = time.monotonic()
                    remaining = 0
                self._stop_event.wait(remaining)
        finally:
            if status_callback:
                status_callback("Idle")


# ---------------------------------------------------------------------------
# SoapySDR loading / discovery
# ---------------------------------------------------------------------------
def _load_soapy():
    try:
        import SoapySDR
    except (ImportError, OSError) as exc:
        raise AcquisitionError(
            "SoapySDR Python bindings could not be loaded. Launch from the "
            "Radioconda Prompt (see INSTALL.md)."
        ) from exc
    return SoapySDR


def enumerate_hackrf() -> list[dict[str, str]]:
    soapy = _load_soapy()
    devices = []
    for entry in soapy.Device.enumerate(dict(driver=HACKRF_DRIVER)):
        item = {str(k): str(entry[k]) for k in entry.keys()}
        if item.get("driver", "").lower() == HACKRF_DRIVER:
            devices.append(item)
    return devices


# Back-compat shim for backend/main.py and the tests.
def enumerate_devices(device_type: str = "HACKRF") -> list[dict[str, str]]:
    if device_type.upper() != "HACKRF":
        raise AcquisitionError(f"Unsupported SDR type: {device_type}")
    return enumerate_hackrf()


# ---------------------------------------------------------------------------
# HackRF One, receive only
# ---------------------------------------------------------------------------
class HackRFAcquisition:
    def __init__(self, config: AcquisitionConfig):
        self.config = self._validate(config)
        self._stop_event = threading.Event()
        self._soapy = None
        self._device = None
        self._stream = None

    def stop(self):
        self._stop_event.set()

    # -- configuration -----------------------------------------------------
    @staticmethod
    def _validate(config: AcquisitionConfig) -> AcquisitionConfig:
        if not (HACKRF_MIN_FREQ <= config.center_frequency <= HACKRF_MAX_FREQ):
            raise AcquisitionError(
                f"Center frequency {config.center_frequency/1e6:.3f} MHz is outside the "
                f"HackRF One range ({HACKRF_MIN_FREQ/1e6:.0f} MHz - "
                f"{HACKRF_MAX_FREQ/1e9:.0f} GHz)."
            )
        if not (HACKRF_MIN_RATE <= config.sample_rate <= HACKRF_MAX_RATE):
            raise AcquisitionError(
                f"Sample rate {config.sample_rate/1e6:.3f} Msps is outside the HackRF One "
                f"range ({HACKRF_MIN_RATE/1e6:.0f} - {HACKRF_MAX_RATE/1e6:.0f} Msps)."
            )
        return config

    def _apply_gains(self, direction, channel, requested_gain: float) -> float:
        """Set AMP / LNA / VGA explicitly.

        Never use the aggregate Device::setGain(dir, chan, value): the SoapySDR
        base class distributes the value across AMP, LNA and VGA in registry
        order, which silently switches on the 14 dB front-end amp.
        """
        vga = 2.0 * round(max(0.0, min(HACKRF_VGA_MAX, requested_gain)) / 2.0)
        self._device.setGain(direction, channel, "AMP", HACKRF_AMP_DB)
        self._device.setGain(direction, channel, "LNA", HACKRF_LNA_DB)
        self._device.setGain(direction, channel, "VGA", vga)
        return vga

    # -- bring-up ----------------------------------------------------------
    def _open(self):
        with _OPEN_LOCK:
            return self._open_locked()

    def _open_locked(self):
        soapy = _load_soapy()
        self._soapy = soapy
        if SOAPY_DEBUG_LOG:
            try:
                soapy.setLogLevel(soapy.SOAPY_SDR_DEBUG)
            except AttributeError:
                pass

        matches = enumerate_hackrf()
        if not matches:
            raise AcquisitionError(
                "No HackRF One found by SoapySDR. Check the USB cable, then run "
                "'hackrf_info' and 'SoapySDRUtil --find=\"driver=hackrf\"'."
            )
        if len(matches) > 1:
            raise AcquisitionError(
                f"{len(matches)} HackRFs found. This build expects exactly one."
            )

        # Use the KEYWORD form. SoapySDR.py:1833 Device.__new__ only marshals args
        # into a C++ Kwargs map on the `if kwargs:` branch; a positional dict falls
        # through to Device_make(dict), SWIG fails to convert it, the map arrives
        # empty, and Factory.cpp:183 throws "Device::make() no match".
        self._device = soapy.Device(f"driver={HACKRF_DRIVER}")

        direction = soapy.SOAPY_SDR_RX          # receive only; TX is never set up
        channel = 0

        try:
            # SoapyHackRF latches rate / freq / bandwidth / gains into its
            # _rx_stream at setupStream() time, so all of this must precede it.
            self._device.setSampleRate(direction, channel, self.config.sample_rate)
            self._device.setFrequency(direction, channel, self.config.center_frequency)
            try:
                self._device.setBandwidth(direction, channel, self.config.sample_rate)
            except (AttributeError, RuntimeError):
                pass

            applied_gain = self._apply_gains(direction, channel, float(self.config.gain))

            actual_rate = float(self._device.getSampleRate(direction, channel))
            actual_freq = float(self._device.getFrequency(direction, channel))
            if abs(actual_rate - self.config.sample_rate) / self.config.sample_rate > 0.01:
                raise AcquisitionError(
                    f"HackRF selected {actual_rate/1e6:.3f} Msps instead of the requested "
                    f"{self.config.sample_rate/1e6:.3f} Msps."
                )

            self.config = AcquisitionConfig(
                device_type="HACKRF",
                center_frequency=actual_freq,
                sample_rate=actual_rate,
                span=min(self.config.span, actual_rate),
                gain=applied_gain,
                fft_size=self.config.fft_size,
                channel=channel,
            )

            self._stream = self._device.setupStream(
                direction, soapy.SOAPY_SDR_CF32, [channel]
            )

            # SoapyHackRF::activateStream() reports failure with a RETURN CODE
            # (SOAPY_SDR_STREAM_ERROR = -2), never with a C++ exception. If this is
            # ignored, _current_mode stays HACKRF_TRANSCEIVER_MODE_OFF, the RX LED
            # never lights, and the app streams nothing while reporting "Connected".
            rc = self._device.activateStream(self._stream)
            if rc != 0:
                detail = soapy.errToStr(rc) if hasattr(soapy, "errToStr") else str(rc)
                raise AcquisitionError(
                    f"activateStream failed ({rc}: {detail}). The HackRF opened but never "
                    "entered RX mode (hackrf_start_rx did not latch; the RX LED stays "
                    "off). Close anything else holding the board, unplug/replug it, and "
                    "try a lower sample rate."
                )

        except Exception as exc:
            self._close()
            if isinstance(exc, AcquisitionError):
                raise
            raise AcquisitionError(f"Could not configure the HackRF One: {exc}") from exc

        return soapy, DeviceInfo(
            connected=True,
            device_name=matches[0].get("label", "HackRF One"),
            driver=HACKRF_DRIVER,
            serial_number=matches[0].get("serial", "Unknown"),
            hardware_key=matches[0].get("device", "HackRF One"),
            details=matches[0],
        )

    # -- streaming ---------------------------------------------------------
    def run(
        self,
        frame_callback: Callable[[object], None],
        status_callback: Callable[[str], None] | None = None,
    ):
        soapy = None
        try:
            soapy, info = self._open()
            if status_callback:
                status_callback(f"Connected: {info.device_name}")
            pipeline = AnalyzerPipeline(self.config, info.device_name)

            block = np.empty(self.config.fft_size, dtype=np.complex64)
            filled = 0
            last_emit = 0.0
            last_sample = time.monotonic()

            timeout_code = getattr(soapy, "SOAPY_SDR_TIMEOUT", -1)
            overflow_code = getattr(soapy, "SOAPY_SDR_OVERFLOW", -4)

            while not self._stop_event.is_set():
                result = self._device.readStream(
                    self._stream,
                    [block[filled:]],
                    self.config.fft_size - filled,
                    timeoutUs=200_000,
                )

                if result.ret > 0:
                    last_sample = time.monotonic()
                    filled += result.ret
                    if filled == self.config.fft_size:
                        frame = pipeline.process(block.copy())
                        now = time.monotonic()
                        if now - last_emit >= 1.0 / 30.0:
                            frame_callback(frame)
                            last_emit = now
                        filled = 0

                # ret == 0 is a legal short read. Overflow means we fell behind:
                # the samples are gone but the stream is still alive.
                elif result.ret == 0 or result.ret in (timeout_code, overflow_code):
                    if time.monotonic() - last_sample > 5.0:
                        raise AcquisitionError(
                            "HackRF is open but has delivered no IQ for 5 s. The RX path "
                            "is not running (check the RX LED). Confirm with: "
                            "hackrf_transfer -r NUL -f "
                            f"{int(self.config.center_frequency)} -s "
                            f"{int(self.config.sample_rate)} -n 20000000"
                        )
                    continue

                else:
                    detail = (
                        soapy.errToStr(result.ret)
                        if hasattr(soapy, "errToStr")
                        else str(result.ret)
                    )
                    raise AcquisitionError(f"HackRF stream read failed: {detail}")

        finally:
            self._close()
            if status_callback:
                status_callback("Idle")

    # -- teardown ----------------------------------------------------------
    def _close(self):
        """Fully release the device.

        Device::make() is refcount-cached in the SoapySDR factory registry and the
        SWIG proxy has no destructor bound to unmake(). Dropping the Python
        reference alone leaves the HackRF USB handle open, so the next open()
        receives the same half-torn-down C++ instance and hackrf_start_rx() fails.
        unmake() is mandatory.
        """
        device, stream, soapy = self._device, self._stream, self._soapy

        if device is not None and stream is not None:
            try:
                device.deactivateStream(stream)   # hackrf_stop_rx -> RX LED off
            except Exception:
                pass
            try:
                device.closeStream(stream)
            except Exception:
                pass

        if device is not None:
            try:
                if soapy is None:
                    soapy = _load_soapy()
                soapy.Device.unmake(device)       # hackrf_close
            except Exception:
                pass

        self._stream = None
        self._device = None
        self._soapy = None


# Legacy alias so existing imports of SoapyAcquisition keep working.
SoapyAcquisition = HackRFAcquisition