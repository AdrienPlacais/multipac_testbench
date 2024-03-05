#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define object to keep a single instrument measurements."""
from abc import ABC
from collections.abc import Iterable
import logging
from typing import Callable, Literal, Self, overload

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import StemContainer
from matplotlib.lines import Line2D

from multipac_testbench.src.multipactor_band.instrument_multipactor_bands \
    import InstrumentMultipactorBands
from multipac_testbench.src.multipactor_band.test_multipactor_bands \
    import TestMultipactorBands


class Instrument(ABC):
    """Hold measurements of a single instrument."""

    def __init__(self,
                 name: str,
                 data: pd.Series,
                 position: np.ndarray | float,
                 is_2d: bool = False,
                 color: tuple[float, float, float] | None = None,
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
        color : tuple[float, float, float] | None, optional
            Color for the plots; all instruments from a same :class:`.PickUp`
            have the same. The default is None, which happens for instruments
            defined in a :class:`.GlobalDiagnostics`.
        kwargs :
            Additional keyword arguments coming from the ``.toml``
            configuration file.

        """
        self.name = name

        self.position = position

        self.is_2d = is_2d
        self.color = color
        plotters = self._get_plot_methods(is_2d)
        self.plot_vs_position, self.scatter_data = plotters

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
                          ) -> tuple[Callable, Callable]:
        """Give the proper plotting functions according to ``is_2d``."""
        plotters = (self._plot_vs_position_for_1d,
                    self._scatter_data_1d)
        if is_2d:
            plotters = (self._plot_vs_position_for_2d,
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

    def at_thresholds(self,
                      instrument_multipactor_bands: InstrumentMultipactorBands
                      ) -> pd.DataFrame:
        """Get what was measured by instrument at thresholds."""
        lower = [self.data[i] if i is not None else np.NaN
                 for i in instrument_multipactor_bands.lower_indexes()]
        upper = [self.data[i] if i is not None else np.NaN
                 for i in instrument_multipactor_bands.upper_indexes()]
        label = f" threshold {self} "
        label += f"according to {instrument_multipactor_bands.instrument_name}"
        df_at_thresholds = pd.DataFrame({'Lower' + label: lower,
                                         'Upper' + label: upper})
        return df_at_thresholds

    @overload
    def multipactor_band_at_same_position(
        self,
        multipactor_bands: TestMultipactorBands,
        raise_no_match_error: Literal[True],
        global_diagnostics: bool = False,
        tol: float = 1e-10,
        **kwargs
    ) -> InstrumentMultipactorBands: ...

    @overload
    def multipactor_band_at_same_position(
        self,
        multipactor_bands: TestMultipactorBands,
        raise_no_match_error: Literal[False],
        global_diagnostics: bool = False,
        tol: float = 1e-10,
        **kwargs
    ) -> InstrumentMultipactorBands | None: ...

    @overload
    def multipactor_band_at_same_position(
        self,
        multipactor_bands: TestMultipactorBands,
        raise_no_match_error: bool,
        global_diagnostics: bool = False,
        tol: float = 1e-10,
        **kwargs
    ) -> InstrumentMultipactorBands | None: ...

    @overload
    def multipactor_band_at_same_position(
        self,
        multipactor_bands: InstrumentMultipactorBands,
        raise_no_match_error: bool,
        global_diagnostics: bool = False,
        tol: float = 1e-10,
        **kwargs
    ) -> InstrumentMultipactorBands: ...

    def multipactor_band_at_same_position(
        self,
        multipactor_bands: TestMultipactorBands | InstrumentMultipactorBands,
        raise_no_match_error: bool = False,
        global_diagnostics: bool = False,
        tol: float = 1e-10,
        **kwargs
    ) -> InstrumentMultipactorBands | None:
        """Get the multipactor that was measured at the same position.

        This is useful to easily match the data from a field probe to the
        multipactor bands measured by the curret probe at the same position.

        Parameters
        ----------
        multipactor_bands : TestMultipactorBands | InstrumentMultipactorBands
            List of :class:`.InstrumentMultipactorBands` among which you want
            to find the match. If a :class:`.InstrumentMultipactorBands` is
            given, return it back without further checking.
        tol : float, optional
            Mismatch allowed between positions. The default is ``1e-10``.
        global_diagnostics : bool, optional
            If multipactor detected by a global instrument should be returned.
            The default is False.
        raise_no_match_error : bool, optional
            If True, method always return an object. The default is False.
        kwargs :
            Other keyword arguments.

        Returns
        -------
        InstrumentMultipactorBands

        """
        if isinstance(multipactor_bands, InstrumentMultipactorBands):
            return multipactor_bands

        assert isinstance(self.position, float)
        matching_multipactor_bands = [
            band for band in multipactor_bands
            if band is not None
            and (abs(band.position - self.position) < tol
                 or np.isnan(self.position)
                 or (global_diagnostics and np.isnan(band.position)))
        ]
        n_found = len(matching_multipactor_bands)
        if n_found == 0:
            if not raise_no_match_error:
                return

            raise ValueError(f"No MultipactorBand among {multipactor_bands} "
                             f"with a position matching {self} was found.")

        if n_found > 1:
            logging.warning("There are several multipactor bands that were "
                            f"measured for the same instrument {self}:"
                            f"{matching_multipactor_bands}")

        return matching_multipactor_bands[0]

    def _post_treat(self, data: np.ndarray) -> np.ndarray:
        """Apply all post-treatment functions."""
        original_data_shape = data.shape
        for post_treater in self.post_treaters:
            data = post_treater(data)
            assert original_data_shape == data.shape, "The post treater "\
                f"{post_treater} modified the shape of the array."
        return data

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
