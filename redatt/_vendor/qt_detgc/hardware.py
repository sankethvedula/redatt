"""Fixed properties of the Cephasonics CSK9130-A2-L5 platform used by the QT scanner.

All values are taken from the driver escrow (paths relative to QT-Escrow-Data-2016-09-17/):
  cquest-sw/ip/ss_us/ss_us_sw/src/ral/PlatformImplA2.cpp  _setupCapabilities / _setupParams
  cquest-sw/ip/ip_beamformer/ip_beamformer_mdl/CC/src/USConst.h
"""
import math

FS_RX = 40e6                 # RX sample clock, Hz (rxClkCtrl: INP_CLK1 = 40 MHz, divider 1)
RX_BUFFER_DEFAULT = 512      # <rxBufferSz>
SYSTEM_DELAY_OFFSET = 10     # decimated samples cropped in addition to the RX buffer (USConst.h)

# TGC capabilities (PlatformImplA2.cpp:651-667)
TGC_CONTROL_POINTS = 32      # MAX_TGC_RAMPS
DTGC_UNITY = 32              # DTGC_GAIN_FOR_UNITY: digital gain = code / 32
ATGC_BITS = 10
DTGC_BITS = 10
ATGC_MIN_DB = 10.0           # analog code 0
ATGC_MAX_DB = 42.8           # analog code 1024 (the software assumes a linear-in-dB VGA law)
DTGC_MIN_DB = -math.inf      # 20*log10(1/32) with *integer* 1/32 == 0 in the driver
DTGC_MAX_DB = 20 * math.log10(32)
TGC_MIN_DB, TGC_MAX_DB = -20.0, 70.0
SLOPE_SCALE = 2.0 ** 16      # ramp slopes are 16.16 fixed point

# Output word: 24-bit S24Q7 I and Q, sign-extended into int32 by the host
IQ_FULL_SCALE = 2 ** 23 - 1


def delta_rx(c):
    """Depth (m) of one raw 40 MHz sample, round trip."""
    return c / (2 * FS_RX)
