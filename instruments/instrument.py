#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define object to keep a single instrument measurements.

.. todo::
    Be more generic than "to smooth". Should be able to apply any number of
    transformations/transfer functions.
    PostTreater object???

"""
from typing import Any, Callable
from dataclasses import dataclass
import pandas as pd
from abc import ABC

import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D


class Instrument(ABC):
    """Hold measurements of a single instrument."""

    def __init__(self,
                 name: str,
                 raw_data: pd.Series,
                 **kwargs,
                 ) -> None:
        """Instantiate the class."""
        self.name = name
        self.raw_data = raw_data

        self._ydata: np.ndarray | None = None
        self._post_treaters: list[Callable[[np.ndarray], np.ndarray]] = []

        self._multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]]
        self._multipactor: np.ndarray[np.bool_] | None = None

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "default ylabel"

    @property
    def instrument_class_name(self) -> str:
        """Name of the instrument class."""
        return self.__class__.__name__

    @property
    def ydata(self) -> np.ndarray:
        """
        Get the treated data.

        Note that in order to save time, ``_ydata`` is not re-calculated
        from ``raw_data`` every time. Hence, it is primordial to re-set
        ``_y_data`` to ``None`` every time a change is made to
        ``_post_treaters``.

        """
        if self._ydata is None:
            self._ydata = self._post_treat(self.raw_data.to_numpy())
        return self._ydata

    @ydata.setter
    def ydata(self, value: np.ndarray | None) -> None:
        if self._multipactor is not None:
            print("Warning! Modifying the ydata (post-treated) makes "
                  "previously calculated multipactor zones obsolete.")
            self._multipactor = None
        self._ydata = value

    @property
    def post_treaters(self) -> list[Callable[[np.ndarray], np.ndarray]]:
        """Get the list of the post-treating functions."""
        return self._post_treaters

    @post_treaters.setter
    def post_treaters(self,
                      post_treaters: list[Callable[[np.ndarray], np.ndarray]]
                      ) -> None:
        """Set the full list of post-treating functions at once.

        Parameters
        ----------
        post_treaters : list[Callable[[np.ndarray], np.ndarray]]
            Post-treating functions.

        """
        if self.ydata is not None:
            print("Warning! Modifying the post treaters makes "
                  "previously post-treated data obsolete.")
            self.ydata = None

        self._post_treaters = post_treaters

    def add_post_treater(self, post_treater: Callable[[np.ndarray], np.ndarray]
                         ) -> None:
        """Append a single post-treating function.

        Parameters
        ----------
        post_treater : Callable[[np.ndarray], np.ndarray]
            Post-treating function to add.

        """
        self._post_treaters.append(post_treater)
        if self.ydata is not None:
            print("Warning! Modifying the post treaters makes "
                  "previously post-treated data obsolete.")
            self.ydata = None

    @property
    def multipac_detector(self
                          ) -> Callable[[np.ndarray], np.ndarray[np.bool_]]:
        """Get access to the function that determines where is multipactor."""
        return self._multipac_detector

    @multipac_detector.setter
    def multipac_detector(self,
                          value: Callable[[np.ndarray], np.ndarray[np.bool_]]
                          ) -> None:
        """Set the function that determines where there is multipactor."""
        if self._multipactor is not None:
            print("Warning! Modifying the multipactor detector makes "
                  "previously calculated multipactor zones obsolete.")
            self._multipactor = None
        self._multipac_detector = value

    @property
    def multipactor(self) -> np.ndarray[np.bool_]:
        """Use ``multipac_detector`` to determine where multipac happens."""
        if self._multipactor is None:
            self._multipactor = self.multipac_detector(self.ydata)
        return self._multipactor

    def _post_treat(self, data: np.ndarray) -> np.ndarray:
        """Apply all post-treatment functions."""
        original_data_shape = data.shape
        for post_treater in self.post_treaters:
            data = post_treater(data)
            assert original_data_shape == data.shape, "The post treater "\
                f"{post_treater} modified the shape of the array."
        return data

    def _plot_raw(self,
                  axe: Axes,
                  color: tuple[float, float, float] | None = None,
                  **subplot_kw) -> Line2D:
        """Plot the raw data as it is in the ``.csv``."""
        line1, = axe.plot(self.raw_data,
                          label=f"{self.name} (raw)",
                          color=color,
                          **subplot_kw)
        return line1

    def plot(self,
             axe: Axes,
             raw: bool = False,
             color: tuple[float, float, float] | None = None,
             **subplot_kw
             ) -> Line2D:
        """Plot what the instrument measured."""
        if raw:
            return self._plot_raw(axe, color=color, **subplot_kw)

        line1, = axe.plot(self.raw_data.index,
                          self.ydata,
                          label=f"{self.name} (post-treated)",
                          color=color,
                          **subplot_kw)
        return line1


@dataclass
class OldInstrument(ABC):
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
