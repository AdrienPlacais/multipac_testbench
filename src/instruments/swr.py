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

    .. deprecated:: v1.1.0

    """

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        print("SWR.__init__ warning! Do not use this class anymore, will "
              "be deprecated in the future. Use Powers instead.")
        super().__init__(*args, **kwargs)

    @classmethod
    def from_rf_power_probes(cls,
                             power_forward: RfPower,
                             power_reflected: RfPower,
                             name: str = 'SWR (calculated from power probes)',
                             tol: float = 5e-2,
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
        name : str, optional
            Name of the virtual instrument.
        tol : float, optional
            To remove :math:`\Gamma` values that are too close to unity.
        kwargs :
            Other keyword arguments.

        Returns
        -------
        SWR
            A :class:`SWR` :class:`.VirtualInstrument`.

        """
        gamma = np.sqrt(power_reflected.ydata / power_forward.ydata)
        gamma = np.abs(gamma)

        invalid_indexes = np.where(gamma > 1.)[0]
        n_invalid = len(invalid_indexes)
        if n_invalid > 0:
            gamma[invalid_indexes] = np.NaN
            print(f"Warning! {n_invalid} points where removed in SWR "
                  "calculation, where reflected power was higher than to "
                  f"forward power. See instruments.SWR for more info.")

        invalid_indexes = np.where(np.abs(gamma - 1.) < tol)[0]
        n_invalid = len(invalid_indexes)
        if n_invalid > 0:
            gamma[invalid_indexes] = np.NaN
            print(f"Warning! {n_invalid} points where removed in SWR "
                  "calculation, where reflected power was too close to "
                  f"forward power. Tolerance over Gamma was: {tol = }. See "
                  "instruments.SWR for more info.")

        ydata = (1. + gamma) / (1. - gamma)
        return cls.from_array(name, ydata, **kwargs)
