#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define object to keep a single instrument measurements.

.. todo::
    Be more generic than "to smooth". Should be able to apply any number of
    transformations/transfer functions.
    PostTreater object???

"""
from typing import Callable
from abc import ABC
import pandas as pd

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
