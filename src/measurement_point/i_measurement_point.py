#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep several related measurements."""
from abc import ABC, ABCMeta
from typing import Any, Callable, Sequence

from matplotlib.axes._axes import Axes
import numpy as np
import pandas as pd

from multipac_testbench.src.instruments.factory import InstrumentFactory
from multipac_testbench.src.instruments.instrument import Instrument
from multipac_testbench.src.util.multipactor_detectors import \
    start_and_end_of_contiguous_true_zones


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

    def add_instrument(self, instrument: Instrument) -> None:
        """Add a new instrument :attr:`.instruments`.

        A priori, useful only for :class:`.VirtualInstrument`, when they rely
        on other :class:`.Instrument` objects to be fully initialized.

        """
        self.instruments.append(instrument)

    def add_post_treater(self,
                         post_treater: Callable[[np.ndarray], np.ndarray],
                         instrument_class: ABCMeta = Instrument,
                         ) -> None:
        """Add post-treatment functions to instruments."""
        instruments = self.get_instruments(instrument_class)
        for instrument in instruments:
            instrument.add_post_treater(post_treater)

    def set_multipac_detector(
            self,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_class: ABCMeta = Instrument,
    ) -> None:
        """Add multipactor detection function to instruments."""
        instruments = self.get_instruments(instrument_class)
        for instrument in instruments:
            instrument.multipac_detector = multipac_detector

    def _where_is_multipactor(self,
                              detector_instrument: Instrument
                              ) -> Sequence[tuple[int, int]]:
        """Get the list of multipacting zones (indexes).

        Need to pass in a ``referee_instrument_class`` to tell which type of
        :class:`.Instrument` we should trust to detect multipactor.

        """
        assert hasattr(detector_instrument, 'multipac_detector'), "No " \
            "multipacting detector defined for instrument under study."

        multipactor = detector_instrument.multipactor
        zones = start_and_end_of_contiguous_true_zones(multipactor)
        return zones

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
        affected_instruments = [
            instrument for instrument in self.instruments
            if isinstance(instrument, instrument_class)
            and instrument.name not in instrument_names_to_ignore]
        return affected_instruments

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
                                 detector_instrument_class: ABCMeta,
                                 ) -> None:
        """Add multipacting zone on a ``plot_instruments`` plot.

        Parameters
        ----------
        axe : Axes
            Matplotlib object on which multipacting zones should be added.
        plotted_instrument_class : ABCMeta
            The nature of the instrument which ``ydata`` is already plotted.
        detector_instrument_class : ABCMeta
            The nature of the instrument that determines where there is
            multipactor. It can be the same than ``plotted_instrument_class``,
            or it can be different.

        """
        instruments = self.get_instruments(detector_instrument_class)
        if len(instruments) == 0:
            return

        if len(instruments) > 1:
            print(f"Warning! More than one {detector_instrument_class} "
                  "instrument at current GlobalDiagnostics/PickUp. So I am not"
                  " sure which one should be used to determine when "
                  "multipactor appeared. I will take the first one.")
        detector_instrument = instruments[0]
        zones = self._where_is_multipactor(detector_instrument)

        plotted_instruments = self.get_instruments(
            plotted_instrument_class)
        if len(plotted_instruments) > 1:
            print("Warning! More than one instrument to be plotted with "
                  "multipactor. Only taking first one.")
        plotted_instrument = plotted_instruments[0]
        y_position_of_multipactor_zone = np.nanmax(plotted_instrument.ydata)
        y_position_of_multipactor_zone *= 1.05

        vline_kw = self._typical_vline_keywords()
        arrow_kw = self._typical_arrow_keywords(plotted_instrument)

        for zone in zones:
            delta_x = zone[1] - zone[0]
            axe.arrow(zone[0], y_position_of_multipactor_zone,
                      delta_x, 0., **arrow_kw)
            axe.arrow(zone[1], y_position_of_multipactor_zone,
                      -delta_x, 0., **arrow_kw)
            axe.axvline(zone[0], **vline_kw)
            axe.axvline(zone[1], **vline_kw)

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
