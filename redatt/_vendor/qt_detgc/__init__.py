"""qt_detgc -- undo the receive gain of QT-Ultrasound (Cephasonics) reflection data.

    from qt_detgc import Scan
    scan = Scan('/path/to/QT007-S1-BL')
    iq   = scan.read(unit=1, view=1)                 # [levels, beams, samples] raw file values
    ours = scan.exact(1).apply(iq)                   # exact inversion (TGC + FIR bank), reverse order
    qt   = scan.qt(1).apply(iq)                      # QT's remove_analog_gain only (no ray-traced digital gain)
    ours.iq, ours.saturated, ours.gain_db, ours.depth_mm

Background and derivations: ~/postdoc/cephasonics/QT-Escrow-Data-2016-09-17/QT_RX_Gain_TGC_Report.md;
usage: QT_DETGC.md at the repository root.
"""
from .config import (AcquisitionConfig, QTReconParams, read_acquisition_xml, read_fortran_params,
                     read_reflection_params, read_temperature, water_speed_mm_per_us)
from .curve import GainCurve, gain_curve
from .fir import bank_gains as fir_bank_gains
from .inverse import Corrected, ExactInverse, GainInverse, QTInverse, refraction_gain_error_db
from .iq import clip_levels, read_iq, saturation_mask
from .scan import UNIT_NAMES, Scan
from .tgc import play_ramps, settgc_tables

__version__ = '0.1.0'

__all__ = ['AcquisitionConfig', 'QTReconParams', 'read_acquisition_xml', 'read_fortran_params',
           'read_reflection_params', 'read_temperature', 'water_speed_mm_per_us', 'GainCurve', 'gain_curve',
           'fir_bank_gains', 'Corrected', 'ExactInverse', 'GainInverse', 'QTInverse', 'refraction_gain_error_db',
           'clip_levels', 'read_iq', 'saturation_mask', 'UNIT_NAMES', 'Scan', 'play_ramps', 'settgc_tables']
