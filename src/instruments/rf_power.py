#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define power probe to measure power."""
from typing import Self

import numpy as np

from multipac_testbench.src.instruments.e_field_probe import ElectricFieldProbe
from multipac_testbench.src.instruments.instrument import Instrument


class RfPower(Instrument):
    """An instrument to measure injected, reflected or probe rf power."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"

    @classmethod
    def from_electric_field_probe(cls,
                                  e_field_probe: ElectricFieldProbe,
                                  swr: float | np.ndarray,
                                  impedance: float = 50.,
                                  name: str | None = None,
                                  **kwargs,
                                  ) -> Self:
        r"""Instantiate the power probe from an electric field probe.

        It is expected that the ``e_field_probe`` holds the rf voltage, not the
        :math:`dBm` power nor the aquisition voltage in :math:`[0V, 10V]`. The
        transformation is the following:

        .. math::
            P_W = \frac{V_{coax}^2}{2 Z_0 SWR}

        where :math:`V_{coax}` is the ``ydata`` from ``e_field_probe`` and
        :math:`Z_0` the impedance.

        Parameters
        ----------
        e_field_probe : ElectricFieldProbe
            Electric field probe.
        swr : float | np.ndarray
            Voltage Signal Wave Ratio.
        impedance : float, optional
            Waveguide impedance in :math:`\Omega`. The default is
            :math:`50\Omega`.
        name : str | None, optional
            Name of the probe, optional. The default is inferred from the name
            of ``e_field_probe``.
        kwargs :
            Other keyword arguments passed to initializer.

        Returns
        -------
        RfPower
            Instantiated object.

        """
        if name is None:
            name = f"Power from {e_field_probe.name}"

        ydata = e_field_probe.ydata**2 / (2. * impedance * swr)

        return cls.from_array(name, ydata, **kwargs)
