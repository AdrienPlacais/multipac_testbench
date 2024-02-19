#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define power probes to measure forward and reflected power."""
import numpy as np

from multipac_testbench.src.instruments.instrument import Instrument


class Power(Instrument):
    """An instrument to measure power."""

    def __init__(self, *args, **kwargs) -> None:
        """Instantiate the instrument, declare other specific attributes."""
        super().__init__(*args, **kwargs)
        self.position = np.NaN

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Power [W]"

    def where_is_growing(self, **kwargs) -> list[bool | float]:
        """Determine where power is growing (``True``) and where it is not."""
        n_points = self.raw_data.index[-1]
        is_growing = [_array_is_growing(self.data, i, **kwargs)
                      for i in range(n_points)]
        return is_growing


class ForwardPower(Power):
    """Store the forward power."""

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Forward power $P_f$ [W]"


class ReflectedPower(Power):
    """Store the reflected power."""

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Reflected power $P_r$ [W]"


def _array_is_growing(array: np.ndarray,
                      index: int,
                      width: int = 10,
                      tol: float = 1e-5) -> bool | float:
    """Tell if ``array`` is locally increasing at ``index``.

    Parameters
    ----------
    array : np.ndarray
        Array under study.
    index : int
        Where you want to know if we increase.
    width : int, optional
        Width of the sample to determine increase. The default is ``10``.
    tol : float, optional
        If absolute value of variation between ``array[idx-width/2]`` and
        ``array[idx+width/2]`` is lower than ``tol``, we return a ``NaN``. The
        default is ``1e-5``.

    Returns
    -------
    bool | float
        If the array is locally increasing, ``NaN`` if undetermined.

    """
    semi_width = width // 2
    if index <= semi_width:
        return np.NaN
    if index >= len(array) - semi_width:
        return np.NaN

    local_diff = array[index + semi_width] - array[index - semi_width]
    if abs(local_diff) < tol:
        return np.NaN
    if local_diff < 0.:
        return False
    return True
