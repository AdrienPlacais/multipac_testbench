#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep measurements at a pick-up."""
from abc import ABCMeta
from typing import Any, Callable, Sequence
from dataclasses import dataclass
import pandas as pd

import numpy as np
from matplotlib.axes._axes import Axes

from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.factory import InstrumentFactory
from multipac_testbench.instruments.current_probe import CurrentProbe

from multipac_testbench.util.multipactor_detectors import \
    start_and_end_of_contiguous_true_zones


class PickUp:
    """Hold information on a single pick-up."""

    def __init__(self,
                 name: str,
                 df_data: pd.DataFrame,
                 instrument_factory: InstrumentFactory,
                 position: float,
                 instruments_kw: dict,
                 ) -> None:
        """Create the pick-up with all its instruments.

        Parameters
        ----------
        name : str
            Name of the pick-up.
        df_data : pd.DataFrame
            df_data
        instrument_factory : InstrumentFactory
            An object that creates :class:`.Instrument`.
        position : float
            position
        instruments_kw : dict[str, dict]
            Dictionary which keys are name of the column where the data from
            the instrument is. Values are dictionaries with keyword arguments
            passed to the proper :class:`.Instrument`.

        Returns
        -------
        None

        """
        self.name = name
        self.position = position
        self.instruments = [
            instrument_factory.run(instr_name, df_data, **instr_kw)
            for instr_name, instr_kw in instruments_kw.items()
        ]

        self._color: tuple[float, float, float] | None = None
        # doubt
        self.idx_of_mp_zones: list[tuple[int, int]]

    def add_post_treater(self,
                         post_treater: Callable[[np.ndarray], np.ndarray],
                         instrument_class: ABCMeta = Instrument,
                         ) -> None:
        """Add post-treatment functions to instruments."""
        affected_instruments = self._get_affected_instruments(
            instrument_class)

        for instrument in affected_instruments:
            instrument.add_post_treater(post_treater)

    def set_multipac_detector(
            self,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_class: ABCMeta = Instrument,
    ) -> None:
        """Add multipactor detection function to instruments."""
        affected_instruments = self._get_affected_instruments(
            instrument_class)

        for instrument in affected_instruments:
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

    def _get_affected_instruments(self, instrument_class: ABCMeta
                                  ) -> list[Instrument]:
        """Get instruments which are (sub) classes of ``instrument_class``."""
        affected_instruments = [instrument for instrument in self.instruments
                                if isinstance(instrument, instrument_class)]
        return affected_instruments

    def get_instrument_data(self, instrument_class: ABCMeta) -> np.ndarray:
        """Get the ``ydata`` from first ``instrument_class`` instrument."""
        instruments = self._get_affected_instruments(instrument_class)
        assert len(instruments) < 2
        instrument = instruments[0]
        return instrument.ydata

    def plot_instruments(self,
                         axes: dict[ABCMeta, Axes],
                         instruments_to_plot: tuple[ABCMeta, ...] = (),
                         raw: bool = False,
                         **subplot_kw,
                         ) -> None:
        """Plot the signal of every instrument at this pick-up."""
        for instrument_class in instruments_to_plot:
            affected_instruments = self._get_affected_instruments(
                instrument_class)
            axe = axes[instrument_class]

            for instrument in affected_instruments:
                line1 = instrument.plot(axe,
                                        raw,
                                        color=self._color,
                                        **subplot_kw)
                if self._color is None:
                    self._color = line1.get_color()

    def add_multipacting_zone(self,
                              axe: Axes,
                              plotted_instr: ABCMeta,
                              detector_instr: ABCMeta,
                              ) -> None:
        """Add multipacting zone on a ``plot_instruments`` plot."""
        instruments = self._get_affected_instruments(detector_instr)
        if len(instruments) > 1:
            print("Warning! More than one instrument to detect multipactor. "
                  "Taking the first one.")
        detector_instrument = instruments[0]
        zones = self._where_is_multipactor(detector_instrument)

        plotted_instruments = self._get_affected_instruments(plotted_instr)
        if len(plotted_instruments) > 1:
            print("Warning! More than one instrument to be plotted with "
                  "multipactor. Only taking first one.")
        plotted_instrument = plotted_instruments[0]
        y_position_of_multipactor_zone = np.nanmax(plotted_instrument.ydata)
        y_position_of_multipactor_zone *= 1.05

        arrow_kw = {
            'color': self._color,
            'length_includes_head': True,
            # 'head_width': 0.5,
        }
        vline_kw = {
            'color': self._color,
            'lw': .2,
        }
        for zone in zones:
            delta_x = zone[1] - zone[0]
            axe.arrow(zone[0], y_position_of_multipactor_zone,
                      delta_x, 0., **arrow_kw)
            axe.arrow(zone[1], y_position_of_multipactor_zone,
                      -delta_x, 0., **arrow_kw)
            axe.axvline(zone[0], **vline_kw)
            axe.axvline(zone[1], **vline_kw)
