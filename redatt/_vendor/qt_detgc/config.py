"""Readers for the configuration files that determine the gains.

AcquisitionConfig   the scanner XML actually used for the acquisition (conf/unit{0,1,2}.xml of a scan)
QTReconParams       QT's reconstruction parameters: conf/reflection_params.txt (C++ pipeline, libconfig)
                    or refl_flip_params.txt (Fortran pipeline, namelist), plus da/refl/temperature.dat
"""
import dataclasses
import math
import os
import re
from typing import List, Optional, Tuple

from . import hardware as hw


# ---------------------------------------------------------------------------------- acquisition XML
@dataclasses.dataclass(frozen=True)
class AcquisitionConfig:
    path: str
    c: float                                   # spSound (m/s) used by the driver for all depth<->time maps
    rx_buffer: int
    start_depth: float                         # m
    end_depth: float                           # m
    num_beams: int
    num_parallel_beams: int
    decimation: int
    tgc_profile: Tuple[Tuple[float, float], ...]   # (depth m, gain dB) sliders
    analog_start_depth: float
    digital_start_depth: float
    digital_gain_fraction: float
    max_gradient_analog: float
    max_gradient_digital: float
    fir_rows: Tuple[Tuple[int, ...], ...]      # 32 banks x 64 coefficients (13-bit, Q11, folded symmetric)
    tfc_depths: Tuple[float, ...]              # SetDemodDecimFiltIntervals (m)
    demod_freqs: Tuple[float, ...]             # SetDemodFreqs (Hz)
    log_enabled: Optional[str] = None

    @property
    def tgc_is_flat(self):
        return len({g for _, g in self.tgc_profile}) == 1

    def samples_per_beam(self):
        """IQ samples per beam in the dumped file, incl. the final alignment-padding sample
        (USEngine::calcBeamInfo)."""
        d = hw.delta_rx(self.c)
        extra = self.rx_buffer + hw.SYSTEM_DELAY_OFFSET * self.decimation
        raw = int(self.end_depth / d + extra)
        crop = int(self.start_depth / d + extra)
        n = int((raw - crop) / self.decimation + 0.5)
        bytes_per_packet = -(-(n * self.num_parallel_beams * 6) // 24) * 24
        return bytes_per_packet // 6 // self.num_parallel_beams

    def bytes_per_frame(self):
        return self.num_beams * self.samples_per_beam() * 8


def _num(tag, text, cast=float, default=None):
    m = re.search(r'<%s>\s*([^<]+?)\s*</%s>' % (tag, tag), text)
    return cast(m.group(1)) if m else default


def read_acquisition_xml(path, frame=0) -> AcquisitionConfig:
    """Parse a Cephasonics scan XML.  `frame` selects the FrameDef (0 = forward sweep; both are identical
    in the QT XMLs)."""
    t = open(path).read()
    fr = re.findall(r'<FrameDef>(.*?)</FrameDef>', t, re.S)[frame]
    tgc = re.search(r'<SetTGC>(.*?)</SetTGC>', fr, re.S).group(1)
    prof = tuple((float(d), float(g)) for d, g in
                 re.findall(r'<tgcProfile\s+depth="([^"]+)"\s+gain="([^"]+)"', tgc))
    coeffs = re.search(r'<SetDecimFiltCoeffs>(.*?)</SetDecimFiltCoeffs>', t, re.S).group(1)
    rows = tuple(tuple(int(x) for x in re.findall(r'<coeff>\s*([-0-9]+)\s*</coeff>', line))
                 for line in coeffs.split('\n') if '<coeff>' in line)
    ints = re.search(r'<SetDemodDecimFiltIntervals>(.*?)</SetDemodDecimFiltIntervals>', t, re.S).group(1)
    return AcquisitionConfig(
        path=os.path.abspath(path),
        c=_num('spSound', t), rx_buffer=_num('rxBufferSz', t, int, hw.RX_BUFFER_DEFAULT),
        start_depth=_num('startDepth', fr), end_depth=_num('endDepth', fr),
        num_beams=_num('numBeams', fr, int), num_parallel_beams=_num('numParallelBeams', fr, int),
        decimation=_num('SetDecimFiltDecimation', t, int),
        tgc_profile=prof,
        analog_start_depth=_num('analogStartDepth', tgc), digital_start_depth=_num('digitalStartDepth', tgc),
        digital_gain_fraction=_num('digitalGainFraction', tgc),
        max_gradient_analog=_num('maxGradientA', tgc), max_gradient_digital=_num('maxGradientD', tgc),
        fir_rows=rows,
        tfc_depths=tuple(float(x) for x in re.findall(r'<depth>([^<]+)</depth>', ints)),
        demod_freqs=tuple(float(x) for x in re.findall(r'<freq>([^<]+)</freq>', t)),
        log_enabled=_num('SetLogarithm', t, str))


# ---------------------------------------------------------------------------- QT reconstruction
@dataclasses.dataclass(frozen=True)
class QTReconParams:
    """The subset of QT's reconstruction parameters that determines its gain removal.

    Only `remove_analog_gain` is modelled.  QT's `apply_digital_gain` (a ramp multiplied back in along
    each ray inside the breast support during ray tracing) is recorded for information but is never
    applied by this package."""
    source: str
    nt_shift: Tuple[int, ...]                  # per probe: file sample m -> envelope index m + 1 + nt_shift
    nt_probe: Tuple[int, ...]
    time_shift: float                          # envelope samples
    sample_rate_mhz: float                     # IQ sample rate (5 MHz)
    hray_mm: float = 0.1                       # ray step (hard-coded in refl_flip_image.f)
    remove_analog_gain: str = 'y'
    analog_gain_start_mm: float = 0.0
    analog_gain_slope_db_per_mm: float = 0.3
    analog_gain_max_db: float = 33.0
    apply_digital_gain: str = 'n'              # informational only -- see class docstring
    range_min_mm: Tuple[float, ...] = (0.0, 0.0, 0.0)
    range_max_mm: Tuple[float, ...] = (1000.0, 1000.0, 1000.0)
    temperature_c: Optional[float] = None
    c0_mm_per_us: Optional[float] = None       # QT's water speed (from temperature)

    @property
    def dt_us(self):
        return 1.0 / self.sample_rate_mhz

    def with_temperature(self, temp_c):
        return dataclasses.replace(self, temperature_c=temp_c, c0_mm_per_us=water_speed_mm_per_us(temp_c))


def water_speed_mm_per_us(temp_c):
    """QT's speed_of_sound() (refl_flip_image.f:791-795, interFuncRefl.cpp:57-63)."""
    t = temp_c
    return (1402.38754 + 5.03711129 * t - 5.80852166e-2 * t ** 2 + 3.34198834e-4 * t ** 3
            - 1.47800417e-6 * t ** 4 + 3.14643091e-9 * t ** 5) * 1e-3


def read_temperature(path):
    """First value of temperature.dat -- exactly what both QT pipelines read (fscanf / read(1,*))."""
    return float(open(path).read().split()[0])


def _libconfig_array(text, key):
    m = re.search(key + r'\s*=\s*\[([^\]]*)\]', text)
    return tuple(float(x) for x in m.group(1).split(',')) if m else None


def _libconfig_value(text, key, cast=float):
    m = re.search(r'\b' + key + r'\s*=\s*"?([^";\s]+)"?\s*;', text)
    return cast(m.group(1)) if m else None


def read_reflection_params(path, temperature_c=None) -> QTReconParams:
    """conf/reflection_params.txt of a scan (libconfig, C++ pipeline qt-recon-shift)."""
    t = open(path).read()
    p = QTReconParams(
        source=os.path.abspath(path),
        nt_shift=tuple(int(x) for x in _libconfig_array(t, 'nt_iq_shift')),
        nt_probe=tuple(int(x) for x in _libconfig_array(t, 'nt_iq_probe')),
        time_shift=_libconfig_value(t, 'time_shift'),
        sample_rate_mhz=_libconfig_value(t, 'sample_rate'),
        hray_mm=_libconfig_value(t, 'hray') or 0.1,
        remove_analog_gain=_libconfig_value(t, 'remove_analog_gain', str) or 'y',
        analog_gain_start_mm=_libconfig_value(t, 'analog_gain_start'),
        analog_gain_slope_db_per_mm=_libconfig_value(t, 'analog_gain_slope'),
        analog_gain_max_db=_libconfig_value(t, 'analog_gain_max'),
        apply_digital_gain=_libconfig_value(t, 'apply_digital_gain', str) or 'n',
        range_min_mm=_libconfig_array(t, 'range_min') or (0.0,) * 3,
        range_max_mm=_libconfig_array(t, 'range_max') or (1000.0,) * 3)
    return p.with_temperature(temperature_c) if temperature_c is not None else p


def _namelist_values(text):
    """key -> value(s) from a Fortran namelist file; indexed keys like nt_shift(1) collected into dicts."""
    out = {}
    for key, idx, val in re.findall(r"^\s*([A-Za-z_]\w*)\s*(?:\((\d+)\))?\s*=\s*('[^']*'|[-+0-9.eEdD]+)",
                                    text, re.M):
        v = val.strip("'") if val.startswith("'") else float(val.replace('d', 'e').replace('D', 'e'))
        if idx:
            out.setdefault(key, {})[int(idx)] = v
        else:
            out[key] = v
    return out


def read_fortran_params(path, temperature_c=None) -> QTReconParams:
    """refl_flip_params.txt used by qti_code/fortran_src/refl_flip_image.f (namelist)."""
    v = _namelist_values(open(path).read())
    idx = lambda k, n=3, cast=float: tuple(cast(v[k][i]) for i in range(n))
    p = QTReconParams(
        source=os.path.abspath(path),
        nt_shift=idx('nt_shift', cast=int), nt_probe=idx('nt_probe', cast=int),
        time_shift=v['time_shift'], sample_rate_mhz=v['sample_rate'], hray_mm=0.1,
        remove_analog_gain=v.get('remove_analog_gain', 'y'),
        analog_gain_start_mm=v['analog_gain_start'], analog_gain_slope_db_per_mm=v['analog_gain_slope'],
        analog_gain_max_db=v['analog_gain_max'], apply_digital_gain=v.get('apply_digital_gain', 'n'),
        range_min_mm=idx('range_min'), range_max_mm=idx('range_max'))
    return p.with_temperature(temperature_c) if temperature_c is not None else p
