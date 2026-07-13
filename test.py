import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import numpy as np

SoapySDR.setLogLevel(SoapySDR.SOAPY_SDR_DEBUG)

print("--- modules ---")
for m in SoapySDR.listModules():
    print(" ", m)

print("--- enumerate() ---")
results = [dict(r) for r in SoapySDR.Device.enumerate()]
for r in results:
    print(" ", r)

hackrfs = [r for r in results if r.get("driver") == "hackrf"]
if not hackrfs:
    raise SystemExit("HackRF NOT enumerated by SoapySDR -> module/driver/claim problem")

print("--- make() ---", flush=True)
sdr = SoapySDR.Device("driver=hackrf")    # no serial, no label
print("make() OK", flush=True)

sdr.setSampleRate(SOAPY_SDR_RX, 0, 10e6)
sdr.setFrequency(SOAPY_SDR_RX, 0, 2440e6)
sdr.setGain(SOAPY_SDR_RX, 0, "AMP", 0)
sdr.setGain(SOAPY_SDR_RX, 0, "LNA", 24)
sdr.setGain(SOAPY_SDR_RX, 0, "VGA", 20)

rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
rc = sdr.activateStream(rx)
print("activateStream ->", rc)          # 0 = RX LED must now be ON

buf = np.empty(4096, np.complex64)
for i in range(20):
    sr = sdr.readStream(rx, [buf], len(buf), timeoutUs=500000)
    print(i, "ret=", sr.ret, "rms=", float(np.sqrt(np.mean(np.abs(buf)**2))) if sr.ret > 0 else "-")

sdr.deactivateStream(rx)
sdr.closeStream(rx)
SoapySDR.Device.unmake(sdr)