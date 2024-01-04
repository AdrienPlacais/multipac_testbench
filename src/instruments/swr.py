#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a Signal Wave Ratio probe."""
from typing import Self

import numpy as np

from multipac_testbench.src.instruments.rf_power import RfPower
from multipac_testbench.src.instruments.virtual_instrument import \
    VirtualInstrument


class SWR(VirtualInstrument):
    """An object to study the SWR.

    Here, SWR stands for Standing Wave Ratio and is the Voltage Standing Wave
    Ratio.

    """

    @classmethod
    def from_rf_power_probes(cls,
                             power_forward: RfPower,
                             power_reflected: RfPower,
                             name: str = 'SWR (calculated from power probes)',
                             **kwargs
                             ) -> Self:
        r"""Instantiate the object by manual calculation.

        First, we compute the reflection coefficient, defined as:

        .. math::
            \Gamma = \frac{V_r}{V_f} = \sqrt{\frac{P_r}{P_f}}

        the SWR is then defined as:

        .. math::
            SWR = \frac{1 + |\Gamma|}{1 - |\Gamma|}

        In our case, :math:`P_r` and :math:`P_f` are real and positive.

        Parameters
        ----------
        power_forward : RfPower
            Holds injected rf power :math:`P_f`.
        power_reflected : RfPower
            Holds reflected power :math:`P_r`.
        name : str
            Name of the virtual instrument.
        kwargs :
            Other keyword arguments.

        Returns
        -------
        SWR
            A :class:`SWR` :class:`.VirtualInstrument`.

        """
        gamma = np.sqrt(power_reflected.ydata / power_forward.ydata)
        ydata = (1. + gamma) / (1. - gamma)
        return cls.from_array(name, ydata, **kwargs)
