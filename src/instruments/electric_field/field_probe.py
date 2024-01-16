#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define field probe to measure electric field."""
from multipac_testbench.src.instruments.electric_field.i_electric_field import\
    IElectricField


class FieldProbe(IElectricField):
    """A probe to measure electric field."""

    def __init__(self,
                 *args,
                 g_probe: float | None = None,
                 a_rack: float | None = None,
                 b_rack: float | None = None,
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
