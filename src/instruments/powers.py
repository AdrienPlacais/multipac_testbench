#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Define power probe to measure forward and reflected power.

Also include calculation of reflection coefficient :math:`\Gamma` and (voltage)
SWR :math:`SWR`.

"""
import numpy as np

from multipac_testbench.src.instruments.instrument import Instrument


class Powers(Instrument):
    """An instrument to measure forward and reflected powers."""

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate the instrument, declare other specific attributes."""
        super().__init__(*args, **kwargs)

        self._gamma: np.ndarray | None = None
        self._swr: np.ndarray | None = None

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"

    @property
    def gamma(self) -> np.ndarray:
        r"""Return the reflection coefficient.

        Reflection coefficient ``gamma`` is defined as:

        .. math::
            \Gamma = \frac{V_r}{V_f} = \sqrt{\frac{P_r}{P_f}}

        where :math:`_f` means forward and :math:`_r` reflected.

        """
        if self._gamma is None:
            self._gamma = self._compute_gamma()
        return self._gamma

    def _compute_gamma(self,
                       warn_reflected_higher_than_forward: bool = True,
                       warn_gamma_too_close_to_unity: bool = True,
                       tol: float = 5e-2,
                       ) -> np.ndarray:
        r"""Compute the reflection coefficient :math:`\Gamma`."""
        assert self.is_2d, "Forward and Reflected power must be provided to "\
            "compute reflection coefficient."

        gamma = np.abs(np.sqrt(self.ydata[:, 1] / self.ydata[:, 0]))

        invalid_indexes = np.where(gamma > 1.)[0]
        n_invalid = len(invalid_indexes)
        if n_invalid > 0:
            gamma[invalid_indexes] = np.NaN
            if warn_reflected_higher_than_forward:
                print(f"Warning! {n_invalid} points where removed in SWR "
                      "calculation, where reflected power was higher than to "
                      f"forward power. See instruments.powers for more info.")

        invalid_indexes = np.where(np.abs(gamma - 1.) < tol)[0]
        n_invalid = len(invalid_indexes)
        if n_invalid > 0:
            gamma[invalid_indexes] = np.NaN
            if warn_gamma_too_close_to_unity:
                print(f"Warning! {n_invalid} points where removed in SWR "
                      "calculation, where reflected power was too close to "
                      f"forward power. Tolerance over Gamma was: {tol = }. See"
                      " instruments.powers for more info.")
        return gamma

    @property
    def swr(self) -> np.ndarray:
        r"""Give the voltage Standing Wave Ratio.

        .. math::
            SWR = \frac{1 + |\Gamma|}{1 - |\Gamma|}

        """
        if self._swr is None:
            self._swr = (1. + self.gamma) / (1. - self.gamma)
        return self._swr
