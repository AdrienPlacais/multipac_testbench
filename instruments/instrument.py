#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define object to keep a single instrument measurements."""
from typing import Any
from dataclasses import dataclass
from abc import ABC

import numpy as np
from matplotlib.axes._axes import Axes

from multipac_testbench.filters import smooth
from multipac_testbench.util.multipactor_detectors import detect_multipactor


@dataclass
class Instrument(ABC):
    """
    Hold measurements of a single instrument.

    Attributes
    ----------
    name : str
        Name or nature of the current instrument.
    x_data : np.ndarray
        Measurement index.
    raw_y_data : np.ndarray
        Data as measured by the instrument.
    y_label : str
        Markdown name of ``raw_y_data``.
    x_label : str, optional
        Markdown name of ``x_data``. The default is ``'Measurement index'``.
    to_smooth : bool, optional
        If calling ``y_data`` should return smoothed ``raw_y_data`` or raw
        ``raw_y_data``. The default is False.
    _smooth_kw : dict[str, Any] | None, optional
        Keyword arguments given to the smoothing function. The default is None,
        in which case you shall not call any smoothing function.

    """

    name: str
    x_data: np.ndarray
    raw_y_data: np.ndarray[np.float64]
    y_label: str
    x_label: str = "Measurement index"
    to_smooth: bool = False
    _smooth_kw: dict[str, Any] | None = None
    _mp_detection_kw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Check inputs validity, declare attributes."""
        assert self.x_data.shape == self.y_data.shape
        self._smoothed: np.ndarray
        self._mp_indexes: np.ndarray[np.int64]

    @property
    def smooth_kw(self) -> dict[str, Any]:
        """Get the keyword arguments transmitted to the smoothing func.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for the smoothing function.

        """
        if self._smooth_kw is None:
            print("Warning!! smooth_kw not defined. Calling a smooth function "
                  "will probably lead to error. Returning empty dict...")
            return {}
        return self._smooth_kw

    @property
    def mp_detection_kw(self) -> dict[str, Any]:
        """Get the keyword arguments transmitted to the mp detection func.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for the multipactor detection function.

        """
        if self._mp_detection_kw is None:
            print("Warning!! mp_detection_kw not defined. Calling a "
                  "mp_detection function will probably lead to error. "
                  "Returning empty dict...")
            return {}
        return self._mp_detection_kw

    @property
    def y_data(self) -> np.ndarray:
        """Give raw or smoothed ``raw_y_data`` according to ``to_smooth``."""
        if self.to_smooth:
            return self.smoothed
        return self.raw_y_data

    @property
    def smoothed(self) -> np.ndarray[np.float64]:
        """Give smoothed data, compute it if necessary."""
        if not hasattr(self, '_smoothed'):
            self._smoothed = smooth(
                input_data=self.raw_y_data, **self.smooth_kw)
        return self._smoothed

    @property
    def mp_indexes(self) -> np.ndarray[np.int64]:
        """Determine index of measurements where MP was detected."""
        if not hasattr(self, '__mp_indexes'):
            mp_detection_kw = self.mp_detection_kw
            self._mp_indexes = detect_multipactor(quantity=self.y_data,
                                                  **mp_detection_kw)
        return self._mp_indexes

    def plot(self, axe: Axes, **kwargs) -> None:
        """Plot what the instrument measured."""
        axe.plot(self.x_data,
                 self.y_data,
                 label=self.name,
                 **kwargs)
