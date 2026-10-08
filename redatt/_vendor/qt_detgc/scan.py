"""A QT scan directory:  <scan>/conf/unit{0,1,2}.xml, conf/reflection_params.txt, da/refl/unit<k>_<view>,
da/refl/temperature.dat.  Everything here is read-only."""
import os
import re

from .config import read_acquisition_xml, read_fortran_params, read_reflection_params, read_temperature
from .inverse import ExactInverse, QTInverse
from .iq import read_iq

UNIT_NAMES = {0: 'unit0 (near)', 2: 'unit2 (mid)', 1: 'unit1 (far)'}


class Scan:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.conf = os.path.join(self.path, 'conf')
        self.data = os.path.join(self.path, 'da', 'refl')
        if not os.path.isdir(self.data):
            raise FileNotFoundError(f'{self.data} not found')
        self._cfg = {}

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def units(self):
        return sorted(int(m.group(1)) for f in os.listdir(self.conf)
                      if (m := re.fullmatch(r'unit(\d)\.xml', f)))

    def config(self, unit):
        if unit not in self._cfg:
            self._cfg[unit] = read_acquisition_xml(os.path.join(self.conf, f'unit{unit}.xml'))
        return self._cfg[unit]

    def views(self, unit):
        return sorted(int(m.group(1)) for f in os.listdir(self.data)
                      if (m := re.fullmatch(rf'unit{unit}_(\d+)', f)))

    def view_path(self, unit, view):
        return os.path.join(self.data, f'unit{unit}_{view}')

    def read(self, unit, view):
        """complex [levels, beams, samples] raw file values"""
        return read_iq(self.view_path(unit, view), self.config(unit))

    @property
    def temperature_c(self):
        return read_temperature(os.path.join(self.data, 'temperature.dat'))

    def qt_params(self, params_file=None):
        """QT reconstruction parameters: the scan's conf/reflection_params.txt (C++ pipeline) by default, or a
        Fortran namelist (refl_flip_params.txt) if given.  Temperature from da/refl/temperature.dat."""
        t = self.temperature_c
        if params_file is None:
            return read_reflection_params(os.path.join(self.conf, 'reflection_params.txt'), t)
        if params_file.endswith('reflection_params.txt'):
            return read_reflection_params(params_file, t)
        return read_fortran_params(params_file, t)

    def exact(self, unit, **kw):
        return ExactInverse(self.config(unit), **kw)

    def qt(self, unit, params_file=None):
        return QTInverse(self.qt_params(params_file), unit, self.config(unit))

    def inverse(self, unit, method='exact', **kw):
        if method == 'exact':
            return self.exact(unit, **kw)
        if method == 'qt':
            return self.qt(unit, kw.get('params_file'))
        raise ValueError("method must be 'exact' or 'qt'")
