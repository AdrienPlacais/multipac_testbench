#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep several related measurements."""
from abc import ABC, ABCMeta
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes

from multipac_testbench.src.instruments.factory import InstrumentFactory
from multipac_testbench.src.instruments.instrument import Instrument
from multipac_testbench.src.multipactor_band.multipactor_bands import \
    MultipactorBands


class IMeasurementPoint(ABC):
    """Hold several related measurements.

    In particular, gather :class:`Instrument` which have the same position.

    """

    def __init__(self,
                 name: str,
                 df_data: pd.DataFrame,
                 instrument_factory: InstrumentFactory,
                 instruments_kw: dict[str, dict[str, Any]],
                 ) -> None:
        """Create the all the global instruments.

        Parameters
        ----------
        df_data : pd.DataFrame
            df_data
        instrument_factory : InstrumentFactory
            An object that creates :class:`.Instrument`.
        instruments_kw : dict[str, dict[str, Any]]
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.

        """
        self.name = name
        self.instruments = [
            instrument_factory.run(instr_name, df_data, **instr_kw)
            for instr_name, instr_kw in instruments_kw.items()
        ]
        self._color: tuple[float, float, float] | None = None
        self.multipactor_bands: MultipactorBands

    def add_instrument(self, instrument: Instrument) -> None:
        """Add a new instrument :attr:`.instruments`.

        A priori, useful only for :class:`.VirtualInstrument`, when they rely
        on other :class:`.Instrument` objects to be fully initialized.

        """
        self.instruments.append(instrument)

    def get_instruments(self,
                        instrument_class: ABCMeta,
                        instruments_to_ignore: Sequence[Instrument | str] = (),
                        ) -> list[Instrument]:
        """
        Get instruments which are (sub) classes of ``instrument_class``.

        An empty list is returned when current pick-up has no instrument of the
        desired instrument class.

        """
        instrument_names_to_ignore = [x if isinstance(x, str)
                                      else x.name
                                      for x in instruments_to_ignore]
        instruments = [
            instrument for instrument in self.instruments
            if isinstance(instrument, instrument_class)
            and instrument.name not in instrument_names_to_ignore]
        return instruments

    def get_instrument(self, *args, **kwargs) -> Instrument | None:
        """Get instrument which is (sub) class of ``instrument_class``.

        Raise an error if several instruments match the condition.

        """
        instruments = self.get_instruments(*args, **kwargs)
        if len(instruments) == 0:
            return
        if len(instruments) == 1:
            return instruments[0]
        raise IOError(f"More than one instrument found with {args = } and "
                      f"{kwargs = }.")

    def get_data(self, instrument_class: ABCMeta) -> np.ndarray:
        """Get the ``ydata`` from first ``instrument_class`` instrument."""
        instrument = self.get_instrument(instrument_class)
        assert instrument is not None
        return instrument.ydata

    def add_post_treater(self,
                         post_treater: Callable[[np.ndarray], np.ndarray],
                         instrument_class: ABCMeta = Instrument,
                         verbose: bool = False,
                         ) -> None:
        """Add post-treatment functions to instruments."""
        instruments = self.get_instruments(instrument_class)
        for instrument in instruments:
            instrument.add_post_treater(post_treater)

            if verbose:
                print(f"A post-treater was added to {str(instrument)}.")

    def detect_multipactor(
            self,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_class: ABCMeta = Instrument,
    ) -> None:
        """Detect multipactor with ``multipac_detector``."""
        instrument = self.get_instrument(instrument_class)
        if instrument is None:
            return
        self.multipactor_bands = MultipactorBands.from_ydata(
            multipac_detector,
            instrument.ydata,
            instrument.name,
        )

    def plot_instrument_vs_time(self,
                                instrument_class_axes: dict[ABCMeta, Axes],
                                instruments_to_plot: tuple[ABCMeta, ...] = (),
                                raw: bool = False,
                                **subplot_kw,
                                ) -> None:
        """Plot the signal of the ``instruments_to_plot`` of this object.

        Parameters
        ----------
        instrument_class_axes : dict[ABCMeta, Axes]
            Dictionary linking the class of the instruments to plot with the
            associated axes.
        instruments_to_plot : tuple[ABCMeta, ...]
            Class of the instruments to be plotted.
        raw : bool
            If the raw of the post-treated signal should be plotted.
        subplot_kw :
            Other keyword arguments passed to the ``plot_vs_time`` methods.

        """
        for instrument_class in instruments_to_plot:
            instruments = self.get_instruments(instrument_class)
            axe = instrument_class_axes[instrument_class]

            for instrument in instruments:
                line1 = instrument.plot_vs_time(axe,
                                                raw,
                                                color=self._color,
                                                **subplot_kw)
                if self._color is None:
                    self._color = line1.get_color()

    def _add_multipactor_vs_time(self,
                                 axe: Axes,
                                 plotted_instrument_class: ABCMeta,
                                 ) -> None:
        """Add arrows to display multipactor.

        .. todo::
            To refactor, should rely on MultipactorBand(s) methods

        Parameters
        ----------
        axe : Axes
            Matplotlib object on which multipacting zones should be added.
        plotted_instrument_class : ABCMeta
            The nature of the instrument which ``ydata`` is already plotted.

        """
        plotted_instrument = self.get_instrument(plotted_instrument_class)
        if plotted_instrument is None:
            return

        y_pos_of_multipactor_zone = 1.05 * np.nanmax(plotted_instrument.ydata)
        vline_kw = self._typical_vline_keywords()
        arrow_kw = self._typical_arrow_keywords(plotted_instrument)

        for multipactor_band in self.multipactor_bands:
            delta_x = multipactor_band[-1] - multipactor_band[0]
            axe.arrow(multipactor_band[0],
                      y_pos_of_multipactor_zone,
                      delta_x,
                      0.,
                      **arrow_kw)
            axe.arrow(multipactor_band[-1],
                      y_pos_of_multipactor_zone,
                      -delta_x,
                      0.,
                      **arrow_kw)
            axe.axvline(multipactor_band[0], **vline_kw)
            axe.axvline(multipactor_band[-1], **vline_kw)

    def scatter_instruments_data(self,
                                 instrument_class_axes: dict[ABCMeta, Axes],
                                 xdata: float,
                                 ) -> None:
        """Scatter data measured by desired instruments."""
        for instrument_class, axes in instrument_class_axes.items():
            instrument = self.get_instrument(instrument_class)
            if instrument is None:
                continue

            instrument.scatter_data(axes,
                                    self.multipactor_bands.multipactor,
                                    xdata)

    def _typical_vline_keywords(self) -> dict[str, Any]:
        """Set consistent plot properties."""
        vline_kw = {
            'color': self._color,
            'lw': 0.2,
        }
        return vline_kw

    def _typical_arrow_keywords(self,
                                instrument: Instrument) -> dict[str, Any]:
        """Set consistent plot properties."""
        typical_width = np.nanmean(instrument.ydata) * 1e-3
        typical_length = instrument.ydata.shape[0] / 70
        arrow_kw = {
            'color': self._color,
            'length_includes_head': True,
            'width': typical_width,
            'head_length': typical_length,
            'head_width': typical_width * 100.,
        }
        return arrow_kw
