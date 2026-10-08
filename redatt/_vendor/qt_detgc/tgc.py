"""Time-gain compensation exactly as the Cephasonics driver programs it and the FPGA plays it back.

settgc_tables   line-by-line port of SetTGC::_updateFrameDefConcrete
                (cquest-sw/ip/ss_us/ss_us_sw/src/fal/cmd/SetTGC.cpp), including its C++/x86 edge-case
                behaviour (division by zero, NaN casts, raw-Q16 slope seed of ramp 1)
play_ramps      TGCCtrl3.sv behaviour (cquest-cicada-hw/ip/ip_tgcctrl/ip_tgcctrl_hw/rtl/TGCCtrl3.sv)
"""
import dataclasses
import math
from typing import List

import numpy as np

from . import hardware as hw


@dataclasses.dataclass(frozen=True)
class Ramp:
    duration: int      # 40 MHz clocks
    gain: int          # start code (10 bit)
    slope: int         # 16.16 fixed point, as the uint32 the driver writes


def _round(v):                                     # SetTGC::_round
    return math.floor(v) if v - math.floor(v) < 0.5 else math.ceil(v)


def _u32(v):                                       # static_cast<uint32>(double), x86-64
    return int(v) & 0xFFFFFFFF if math.isfinite(v) else 0


def _u16(v):                                       # static_cast<uint16>(double), x86
    return int(v) & 0xFFFF if math.isfinite(v) else 0


def _s32(x):
    return x - (1 << 32) if x & 0x80000000 else x


def _div(a, b):                                    # IEEE double division, no exception on /0
    if b != 0:
        return a / b
    return math.nan if a == 0 else math.copysign(math.inf, a)


def settgc_tables(cfg):
    """(analog ramps, digital ramps) exactly as written to the FPGA TGC controller."""
    end = cfg.end_depth
    a_start, d_start, dfrac = cfg.analog_start_depth, cfg.digital_start_depth, cfg.digital_gain_fraction
    d2c = hw.FS_RX / (0.5 * cfg.c)
    start = min(a_start, d_start)
    ncp = hw.TGC_CONTROL_POINTS

    # _check_params: keep sliders inside [start, endDepth], clamp gains
    zero_found, prof = False, []
    for i, (d, g) in enumerate(cfg.tgc_profile):
        if d == 0.0 or (i == 0 and d <= 1e-6):
            zero_found = True
        if d < start or d > end:
            continue
        prof.append((d, min(max(g, hw.TGC_MIN_DB), hw.TGC_MAX_DB)))
    if not prof:
        raise ValueError('no TGC slider inside [startDepth, endDepth]')

    # _setup_controlPointDepths: distribute the 32 control points, largest gaps first
    ns = len(prof)
    spi = math.floor((ncp - ns - 1) / (ns + 1))
    spaces = [prof[0][0] - start] + [prof[i + 1][0] - prof[i][0] for i in range(ns - 1)] + [end - prof[-1][0]]
    alloc = [spi] * len(spaces)
    extra = ncp - (spi * (ns + 1) + ns) - 1
    while extra > 1e-6:
        cur = [spaces[i] / (alloc[i] + 1) for i in range(len(spaces))]
        alloc[cur.index(max(cur))] += 1
        extra -= 1
    cp = []
    for i, e in enumerate([start] + [p[0] for p in prof]):
        cp.append(e)
        for k in range(int(alloc[i])):
            cp.append(e + (k + 1) * spaces[i] / (alloc[i] + 1))

    # _divideGain: analog/digital split (dFrac rule; analog floor 10 dB)
    a_lim = lambda x: min(max(x, hw.ATGC_MIN_DB), hw.ATGC_MAX_DB)
    d_lim = lambda x: min(max(x, hw.DTGC_MIN_DB), hw.DTGC_MAX_DB)
    ag, dg, dhold = [], [], None
    for d, g in prof:
        if d < a_start:
            ag.append(hw.ATGC_MIN_DB)
            dg.append(d_lim(g - hw.ATGC_MIN_DB))
            continue
        if d < d_start:
            if g >= hw.ATGC_MIN_DB:
                ag.append(a_lim(g))
                dg.append(0.0)
            else:
                ag.append(hw.ATGC_MIN_DB)
                dhold = d_lim(g - hw.ATGC_MIN_DB) if dhold is None else dhold
                dg.append(dhold)
            continue
        af = (1 - dfrac) * g
        a = a_lim(af)
        ag.append(a)
        dg.append(d_lim(g - a) if (af < hw.ATGC_MIN_DB or af > hw.ATGC_MAX_DB) else d_lim(dfrac * g))

    # _interpToGrid: linear in dB, padded at 0 and 1.01*endDepth
    xd = ([0.0] if not zero_found else []) + [p[0] for p in prof] + [end * 1.01]
    xa = ([ag[0]] if not zero_found else []) + ag + [ag[-1]]
    xg = ([dg[0]] if not zero_found else []) + dg + [dg[-1]]
    agg, dgg = np.interp(cp, xd, xa), np.interp(cp, xd, xg)

    # _makeRamps
    durs = [_round((cp[i + 1] - cp[i]) * d2c) for i in range(len(cp) - 1)] + [_round((end - cp[-1]) * d2c)]
    acode = [min(max(_round((x - hw.ATGC_MIN_DB) / (hw.ATGC_MAX_DB - hw.ATGC_MIN_DB) * (1 << hw.ATGC_BITS)), 0),
                 (1 << hw.ATGC_BITS) - 1) for x in agg]
    dcode = [min(max(_round(10 ** (x / 20.0) * hw.DTGC_UNITY), 0), (1 << hw.DTGC_BITS) - 1) for x in dgg]

    def slope_hw(v, mx):                           # _slopeToHW (NaN passes the comparisons)
        p = v * hw.SLOPE_SCALE
        return mx if p > mx else (-mx if p < -mx else p)

    def ramps(code, mx) -> List[Ramp]:
        out = [Ramp(durs[0], _u16(code[0] + 0.5), _u32(slope_hw(_div(code[1] - code[0], durs[0]), mx) + 0.5))]
        gp, sp = code[0], out[0].slope             # driver seeds ramp 1 with the *raw* Q16 slope (bug)
        for i in range(1, ncp):
            g = _u16(gp + sp * durs[i - 1] + 0.5)
            des = _div(code[i + 1] - g, durs[i]) if i < ncp - 1 else 0.0
            s = _u32(slope_hw(des, mx) + 0.5)
            out.append(Ramp(durs[i], g, s))
            gp, sp = g, s / hw.SLOPE_SCALE
        return out

    return ramps(acode, cfg.max_gradient_analog), ramps(dcode, cfg.max_gradient_digital)


def play_ramps(ramps, n_clocks):
    """Gain code per 40 MHz clock: each segment restarts from its own start code and adds its 16.16 slope
    per clock, saturating to 0..1023; output rounded to an integer code; the last value is held."""
    out = np.empty(n_clocks)
    n = 0
    for r in ramps:
        seg = np.clip(r.gain + _s32(r.slope) * np.arange(r.duration) / hw.SLOPE_SCALE, 0, 1023)
        m = min(len(seg), n_clocks - n)
        out[n:n + m] = np.floor(seg[:m] + 0.5)
        n += m
        if n >= n_clocks:
            break
    out[n:] = out[n - 1] if n else 0
    return out


def analog_code_to_db(code):
    """The driver's assumed VGA law: 10 dB + code * 32.8/1024 (SetTGC::_dB2AnaGain)."""
    return hw.ATGC_MIN_DB + np.asarray(code) * (hw.ATGC_MAX_DB - hw.ATGC_MIN_DB) / (1 << hw.ATGC_BITS)
