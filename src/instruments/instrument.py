#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define object to keep a single instrument measurements."""
from abc import ABC
from collections.abc import Iterable
from typing import Callable, Self


import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from matplotlib.container import StemContainer
from matplotlib.lines import Line2D


class Instrument(ABC):
    """Hold measurements of a single instrument."""

    def __init__(self,
                 name: str,
                 raw_data: pd.Series,
                 position: np.ndarray | float | None = None,
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
        position : np.ndarray | float | None, optional
            The position of the instrument. The default is None, in which case
            :attr:`._position` is not set (case of :class:`.GlobalDiagnostic`).
        kwargs :
            Additional keyword arguments coming from the ``.toml``
            configuration file.

        """
        self.name = name
        self.raw_data = raw_data

        self._position: np.ndarray | float
        if position is not None:
            self._position = position

        self._ydata: np.ndarray | None = None
        self._post_treaters: list[Callable[[np.ndarray], np.ndarray]] = []

        self._multipac_detector: Callable[[np.ndarray], np.ndarray]
        self._multipactor: np.ndarray | None = None

        self.plot_vs_position = self._stem_vs_position

    def __str__(self) -> str:
        """Give concise information on instrument."""
        out = f"{self.class_name} ({self.name})"
        return out

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "default ylabel"

    @classmethod
    def from_array(cls,
                   name: str,
                   ydata: np.ndarray,
                   xdata: Iterable | None = None,
                   **kwargs) -> Self:
        """Instantiate from numpy array."""
        if xdata is None:
            n_points = len(ydata)
            xdata = range(1, n_points + 1)

        raw_data = pd.Series(data=ydata,
                             index=xdata,
                             name=name)

        return cls(name, raw_data, **kwargs)
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

    def plot_vs_time(self,
                     axe: Axes,
                     raw: bool = False,
                     color: tuple[float, float, float] | None = None,
                     **subplot_kw
                     ) -> Line2D:
        """Plot what the instrument measured."""
        ydata = self.ydata
        label = f"{self.name} (post-treated)"

        if raw or len(self.post_treaters) == 0:
            ydata = self.raw_data
            label = f"{self.name} (raw)"

        line1, = axe.plot(self.raw_data.index,
                          ydata,
                          color=color,
                          label=label,
                          **subplot_kw)
        return line1

    def _stem_vs_position(self,
                          sample_index: int,
                          raw: bool = False,
                          color: tuple[float, float, float] | None = None,
                          artist: StemContainer | None = None,
                          axe: Axes | None = None,
                          ) -> StemContainer:
        """
        Plot what instrument measured at its position, at a given time step.

        Adapted to Pick-Up instruments.

        Parameters
        ----------
        sample_index : int
            Index of the measurements.
        raw : bool
            If the raw data should be plotted. The default is False.
        color : tuple[float, float, float] | None, optional
            Color of the plot. The default is None (default color).
        artist : StemContainer | None, optional
            If provided, the stem Artist object is updated rather than
            overwritten. It is mandatory for matplotlib animation to work. The
            default is None.
        axe : Axes | None, optional
            Axe where the artist should be created. It must be provided if
            ``artist`` is not given. The default is None.

        Returns
        -------
        artist : StemContainer
            The plotted stem.

        """
        position = getattr(self, '_position', -1.)
        assert isinstance(position, float)

        ydata = self.ydata[sample_index]
        if raw or len(self.post_treaters) == 0:
            ydata = self.raw_data[sample_index]

        if artist is not None:
            artist[0].set_ydata(ydata)
            new_path = np.array([[position, 0.],
                                 [position, ydata]])
            artist[1].set_paths([new_path])
            return artist

        assert axe is not None
        artist = axe.stem(position, ydata)
        return artist

    def _plot_vs_position(self,
                          sample_index: int,
                          raw: bool = False,
                          color: tuple[float, float, float] | None = None,
                          axe: Axes | None = None,
                          artist: Line2D | None = None,
                          ) -> Line2D:
        """
        Plot what instrument measured at all positions, at a given time step.

        Adapted to instruments with several positions, such as
        VirtualInstrument reproducing electric field envelope at all positions.

        Parameters
        ----------
        sample_index : int
            Index of the measurements.
        raw : bool
            If the raw data should be plotted. The default is False.
        color : tuple[float, float, float] | None, optional
            Color of the plot. The default is None (default color).
        artist : Line2D | None, optional
            If provided, the Line2D Artist object is updated rather than
            overwritten. It is mandatory for matplotlib animation to work. The
            default is None.
        axe : Axes | None, optional
            Axe where the artist should be created. It must be provided if
            ``artist`` is not given. The default is None.

        Returns
        -------
        artist : Line2D
            The plotted line.

        """
        assert hasattr(self, '_position')
        assert isinstance(self._position, np.ndarray)

        ydata = self.ydata[sample_index, :]
        assert isinstance(ydata, np.ndarray)
        assert ydata.shape == self._position.shape

        if artist is not None:
            raise NotImplementedError

        assert axe is not None
        artist, = axe.plot(self._position, ydata, color=color)
        return artist
