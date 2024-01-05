#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define object to keep a single instrument measurements."""
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
        """Instantiate the class.

        Parameters
        ----------
        name : str
            Name of the instrument. It must correspond to the name of a column
            in the ``.csv`` file.
        raw_data : pd.Series
            ``x`` and ``y`` data as saved in the ``.csv`` produced by LabVIEW.
        kwargs :
            Additional keyword arguments coming from the ``.toml``
            configuration file.

        """
        self.name = name
        self.raw_data = raw_data

        self._ydata: np.ndarray | None = None
        self._post_treaters: list[Callable[[np.ndarray], np.ndarray]] = []

        self._multipac_detector: Callable[[np.ndarray], np.ndarray]
        self._multipactor: np.ndarray | None = None

    def __str__(self) -> str:
        """Give concise information on instrument."""
        out = f"{self.class_name} ({self.name})"
        return out

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "default ylabel"

    @property
    def class_name(self) -> str:
        """Shortcut to the name of the instrument class."""
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
            Post-treating function to add. It must take an array as input, and
            return an array with the same size as output.

        """
        self._post_treaters.append(post_treater)
        if self.ydata is not None:
            print("Warning! Modifying the post treaters makes "
                  "previously post-treated data obsolete.")
            self.ydata = None

    @property
    def multipac_detector(self
                          ) -> Callable[[np.ndarray], np.ndarray]:
        """Get access to the function that determines where is multipactor.

        .. note::
            It is not mandatory to define a multipactor detector for every
            :class:`Instrument`, as it may be unrelatable with some types of
            instrument.

        """
        return self._multipac_detector

    @multipac_detector.setter
    def multipac_detector(self,
                          value: Callable[[np.ndarray], np.ndarray]
                          ) -> None:
        """Set the function determining where/when there is multipactor.

        Parameters
        ----------
        value : Callable[[np.ndarray], np.ndarray]
            Function taking in the array of :attr:`~ydata`, and returning an
            array of boolean with the same shape. It contains ``True`` where
            there is multipactor, and ``False`` where multipactor does not
            happen.

        """
        if self._multipactor is not None:
            print("Warning! Modifying the multipactor detector makes "
                  "previously calculated multipactor zones obsolete.")
            self._multipactor = None
        self._multipac_detector = value

    @property
    def multipactor(self) -> np.ndarray:
        """Use ``multipac_detector`` to determine where multipac happens.

        Returns
        -------
        _multipactor : np.ndarray
            Array with the same shape as :attr:`~ydata`. It is ``True`` where
            there is multipactor and ``False`` elsewhere.

        """
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

        if len(self.post_treaters) == 0:
            return self._plot_raw(axe, color=color, **subplot_kw)

        line1, = axe.plot(self.raw_data.index,
                          self.ydata,
                          label=f"{self.name} (post-treated)",
                          color=color,
                          **subplot_kw)
        return line1
