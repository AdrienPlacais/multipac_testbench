#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define object to keep a single instrument measurements.

.. todo::
    Ensure that all instruments have a position (float, can be np.NaN)

"""
from abc import ABC
from collections.abc import Iterable
from typing import Callable, Self

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import StemContainer
from matplotlib.lines import Line2D

from multipac_testbench.src.multipactor_band.multipactor_bands import \
    MultipactorBands


class Instrument(ABC):
    """Hold measurements of a single instrument."""

    def __init__(self,
                 name: str,
                 data: pd.Series,
                 position: np.ndarray | float,
                 is_2d: bool = False,
                 **kwargs,
                 ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        name : str
            Name of the instrument.
        data : pd.Series
            ``x`` and ``y`` data as saved in the ``.csv`` produced by LabVIEW.
        position : np.ndarray | float
            The position of the instrument. If irrelevant (global diagnostic),
            must be set to np.NaN.
        is_2d : bool, optional
            To make the difference between instruments holding a single array
            of data (e.g. current vs time) and those holding several columns
            (eg forward and reflected power).
        kwargs :
            Additional keyword arguments coming from the ``.toml``
            configuration file.

        """
        self.name = name

        self.position = position

        self.is_2d = is_2d
        plotters = self._get_plot_methods(is_2d)
        self.plot_vs_time, self.plot_vs_position, self.scatter_data = plotters

        self._raw_data: pd.Series = data
        self._data: np.ndarray
        self._data_as_pd: pd.Series

        self._post_treaters: list[Callable[[np.ndarray], np.ndarray]] = []

    def __str__(self) -> str:
        """Give concise information on instrument."""
        out = f"{self.class_name} ({self.name})"
        return out

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "default ylabel"

    @property
    def label(self) -> str | None:
        """Label used for legends in plots vs position."""
        return

    @classmethod
    def from_array(cls,
                   name: str,
                   data: np.ndarray,
                   xdata: Iterable,
                   **kwargs) -> Self:
        """Instantiate :class:`Instrument` from a numpy array.

        Parameters
        ----------
        name : str
            Name of the instrument.
        data : np.ndarray
            The data measured by the instrument.
        xdata : Iterable | None, optional
            The data representing the measuring points. The default is None, in
            which case it is a list of integers starting from 1, which is the
            same as all data from the ``.csv``.
        kwargs :
            Other keyword arguments passed to the :class:`.Instrument`.

        Returns
        -------
        instrument : Instrument
            A regular instrument.

        """
        raw_data = pd.Series(data=data,
                             index=xdata,
                             name=name)
        return cls(name, raw_data, **kwargs)

    @classmethod
    def from_pd_dataframe(cls,
                          name: str,
                          raw_data: pd.DataFrame,
                          **kwargs,
                          ) -> Self:
        """Instantiate the object from several ``.csv`` file columns.

        Parameters
        ----------
        name : str
            Name of the instrument.
        raw_data : pd.DataFrame
            Object holding several columns of the ``.csv``.
        kwargs :
            Other keyword arguments passed to the :class:`.Instrument`.

        Returns
        -------
        instrument : Instrument
            An instrument. Note that its ``data`` attribute will be a 2D
            array.

        """
        is_2d = True
        return cls(name, raw_data, is_2d=is_2d, **kwargs)

    @property
    def class_name(self) -> str:
        """Shortcut to the name of the instrument class."""
        return self.__class__.__name__

    @property
    def data(self) -> np.ndarray:
        """
        Get the treated data.

        Note that in order to save time, ``_data`` is not re-calculated
        from ``raw_data`` every time. Hence, it is primordial to re-set
        ``_y_data`` to ``None`` every time a change is made to
        ``_post_treaters``.

        """
        if not hasattr(self, '_data'):
            self._data = self._post_treat(self._raw_data.to_numpy())
        return self._data

    @property
    def data_as_pd(self) -> pd.Series | pd.DataFrame:
        """Get the treated data as a pandas object."""
        if hasattr(self, '_data_as_pd'):
            return self._data_as_pd

        index = self._raw_data.index
        if self.is_2d:
            assert isinstance(self._raw_data, pd.DataFrame)
            columns = self._raw_data.columns
            self._data_as_pd = pd.DataFrame(self.data,
                                            columns=columns,
                                            index=index)
            return self._data_as_pd

        self._data_as_pd = pd.Series(self.data, index=index, name=self.name)
        return self._data_as_pd

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        """Set ``data``, clean previous ``_data_as_pd``."""
        self._data = new_data
        if hasattr(self, '_data_as_pd'):
            delattr(self, '_data_as_pd')

    def _get_plot_methods(self, is_2d: bool
                          ) -> tuple[Callable, Callable, Callable]:
        """Give the proper plotting functions according to ``is_2d``."""
        plotters = (self._plot_vs_time_for_1d,
                    self._plot_vs_position_for_1d,
                    self._scatter_data_1d)
        if is_2d:
            plotters = (self._plot_vs_time_for_2d,
                        self._plot_vs_position_for_2d,
                        self._scatter_data_2d)
        return plotters

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
        delattr(self, '_data')
        delattr(self, '_data_as_pd')
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
        if hasattr(self, '_data'):
            delattr(self, '_data')
        if hasattr(self, '_data_as_pd'):
            delattr(self, '_data_as_pd')
        self._post_treaters.append(post_treater)

    def values_at_barriers(
            self,
            multipactor_bands: MultipactorBands,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get measured data at lower and upper multipactor barriers.

        .. todo::
            Dirty patch to select 1d/2d data

        Parameters
        ----------
        multipactor_bands : MultipactorBands
            Object holding the multipacting barriers.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Holds measured data at lower and upper multipacting barriers.

        """
        barriers_idx = multipactor_bands.barriers
        lower_barrier_idx, upper_barrier_idx = barriers_idx
        assert isinstance(lower_barrier_idx, list)
        assert isinstance(upper_barrier_idx, list)
        name_of_detector = multipactor_bands.instrument_name

        match (self._raw_data):
            case pd.Series():
                label = f"{self} according to {name_of_detector}"

                lower_dict = {
                    f"Lower barrier {label}": self.data[lower_barrier_idx]}
                lower_values = pd.DataFrame(lower_dict,
                                            index=lower_barrier_idx)

                upper_dict = {
                    f"Lower barrier {label}": self.data[upper_barrier_idx]}
                upper_values = pd.DataFrame(upper_dict,
                                            index=upper_barrier_idx)

            case pd.DataFrame() as df:
                label = df.columns + f" according to {name_of_detector}"
                lower_values = pd.DataFrame(
                    data=self.data[lower_barrier_idx],
                    index=lower_barrier_idx,
                    columns="Lower barrier " + label,
                )

                upper_values = pd.DataFrame(
                    data=self.data[upper_barrier_idx],
                    index=upper_barrier_idx,
                    columns="Upper barrier " + label,
                )
            case _:
                raise TypeError
        return lower_values, upper_values

    def values_at_barriers_fully_conditioned(
            self,
            multipactor_bands: MultipactorBands,
    ) -> tuple[float, float]:
        """Get measured data at last mp limits."""
        barriers_idx = multipactor_bands.barriers
        last_low = barriers_idx[0][-1]
        last_upp = barriers_idx[1][-1]

        if isinstance(last_low, (list, np.ndarray)):
            last_low = last_low[0]
        if isinstance(last_upp, (list, np.ndarray)):
            last_upp = last_upp[0]

        return self.data[last_low], self.data[last_upp]

    def _post_treat(self, data: np.ndarray) -> np.ndarray:
        """Apply all post-treatment functions."""
        original_data_shape = data.shape
        for post_treater in self.post_treaters:
            data = post_treater(data)
            assert original_data_shape == data.shape, "The post treater "\
                f"{post_treater} modified the shape of the array."
        return data

    def _plot_vs_time_for_1d(self,
                             axe: Axes,
                             raw: bool = False,
                             color: tuple[float, float, float] | None = None,
                             xdata: np.ndarray | pd.Index | None = None,
                             **subplot_kw
                             ) -> Line2D:
        """Plot what the instrument measured."""
        data = self.data
        label = f"{self.name} (post-treated)"

        if raw or len(self.post_treaters) == 0:
            data = self._raw_data
            label = f"{self.name} (raw)"

        if xdata is None:
            xdata = self._raw_data.index

        line1, = axe.plot(xdata,
                          data,
                          color=color,
                          label=label,
                          **subplot_kw)
        return line1

    def _plot_vs_time_for_2d(self,
                             axe: Axes,
                             raw: bool = False,
                             color: tuple[float, float, float] | None = None,
                             xdata: np.ndarray | pd.Index | None = None,
                             **subplot_kw
                             ) -> Line2D:
        """Plot what the instrument measured."""
        data = self.data
        label = f"{self.name} (post-treated)"

        if raw or len(self.post_treaters) == 0:
            data = self._raw_data.to_numpy()
            label = f"{self.name} (raw)"
        if xdata is None:
            xdata = self._raw_data.index

        n_cols = data.shape[1]
        line1 = None
        for i in range(n_cols):
            line1, = axe.plot(xdata,
                              data[:, i],
                              color=color,
                              label=label + f" (column {i})",
                              **subplot_kw)
        assert line1 is not None
        return line1

    def _plot_vs_position_for_1d(self,
                                 sample_index: int,
                                 raw: bool = False,
                                 color: tuple[float, float,
                                              float] | None = None,
                                 artist: StemContainer | None = None,
                                 axe: Axes | None = None,
                                 **kwargs
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
        position = getattr(self, 'position', -1.)
        assert isinstance(position, float)

        data = self.data[sample_index]
        if raw or len(self.post_treaters) == 0:
            data = self._raw_data[sample_index]

        if artist is not None:
            artist[0].set_ydata(data)
            new_path = np.array([[position, 0.],
                                 [position, data]])
            artist[1].set_paths([new_path])
            return artist

        assert axe is not None
        artist = axe.stem(position,
                          data,
                          label=self.label,
                          **kwargs)
        return artist

    def _plot_vs_position_for_2d(self,
                                 sample_index: int,
                                 raw: bool = False,
                                 color: tuple[float, float,
                                              float] | None = None,
                                 axe: Axes | None = None,
                                 artist: Line2D | None = None,
                                 **kwargs,
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
        assert hasattr(self, 'position')
        assert isinstance(self.position, np.ndarray)

        data = self.data[sample_index, :]
        assert isinstance(data, np.ndarray)
        assert data.shape == self.position.shape

        if artist is not None:
            artist.set_data(self.position, data)
            return artist

        assert axe is not None
        artist, = axe.plot(self.position,
                           data,
                           color=color,
                           label=self.label,
                           **kwargs)
        axe.legend()
        return artist

    def _scatter_data_1d(self,
                         axes: Axes,
                         multipactor: np.ndarray,
                         xdata: float | np.ndarray | None = None,
                         ) -> None:
        """Plot ``data``, discriminating where there is multipactor or not.

        Parameters
        ----------
        axes : Axes
            Where to plot.
        multipactor : np.ndarray
            True where there is multipactor, False elsewhere.
        xdata : float | np.ndarray | None, optional
            x position of the data. The default is None, in which case we take
            :attr:`self._position`.

        """
        data = self.data

        if xdata is None:
            xdata = self.position
        if isinstance(xdata, float):
            xdata = np.full(len(data), xdata)
        assert isinstance(xdata, np.ndarray)

        mp_kwargs = {'c': 'r',
                     'marker': 's',
                     'alpha': 0.1,
                     }
        no_mp_kwargs = {'c': 'k',
                        'alpha': 0.1,
                        'marker': 'x',
                        }
        if axes.get_legend_handles_labels() == ([], []):
            mp_kwargs['label'] = 'MP'
            no_mp_kwargs['label'] = 'No MP'

        axes.scatter(xdata[multipactor] - 0.1,
                     data[multipactor],
                     **mp_kwargs)
        axes.scatter(xdata[~multipactor] + 0.1,
                     data[~multipactor],
                     **no_mp_kwargs)
        return

    def _scatter_data_2d(self, *args, **kwargs) -> None:
        """Hold place."""
        raise NotImplementedError()
