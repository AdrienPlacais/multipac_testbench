#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define field probe to measure electric field."""
from functools import partial
from multipac_testbench.src.instruments.electric_field.i_electric_field import\
    IElectricField
from multipac_testbench.src.util.post_treaters import (v_acquisition_to_v_coax,
                                                       v_coax_to_v_acquisition)


class FieldProbe(IElectricField):
    """A probe to measure electric field."""

    def __init__(self,
                 *args,
                 g_probe: float | None = None,
                 a_rack: float | None = None,
                 b_rack: float | None = None,
                 patch: bool = False,
                 **kwargs) -> None:
        r"""Instantiate with some specific arguments.

        Parameters
        ----------
        g_probe : float | None, optional
            Total attenuation. Probe specific, also depends on frequency. The
            default is None.
        a_rack : float | None, optional
            Rack calibration slope in :math:`\mathrm{V/dBm}`. The default is
            None.
        b_rack : float | None, optional
            Rack calibration constant in :math:`\mathrm{dBm}`. The default is
            None.

        """
        super().__init__(*args, **kwargs)
        self._g_probe = g_probe
        self._a_rack, self._b_rack = a_rack, b_rack
        if patch:
            self._patch_data()

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Measured voltage [V]"

    def _patch_data(self, g_probe_in_labview: float = 1.) -> None:
        """Correct when ``g_probe`` in LabVIEWER is wrong."""
        assert self._a_rack is not None
        assert self._b_rack is not None
        assert self._g_probe is not None
        fun1 = partial(v_coax_to_v_acquisition,
                       g_probe=g_probe_in_labview,
                       a_rack=self._a_rack,
                       b_rack=self._b_rack,
                       z_0=50.)
        fun2 = partial(v_acquisition_to_v_coax,
                       g_probe=self._g_probe,
                       a_rack=self._a_rack,
                       b_rack=self._b_rack,
                       z_0=50.)
        self.add_post_treater(fun1)
        self.add_post_treater(fun2)
