#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to keep measurements at a pick-up."""
from abc import ABCMeta
from typing import Any, Callable
from dataclasses import dataclass
import pandas as pd

import numpy as np
from matplotlib.axes._axes import Axes

from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.factory import InstrumentFactory

from multipac_testbench.instruments.current_probe import CurrentProbe


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

    def _get_affected_instruments(self, instrument_class: ABCMeta
                                  ) -> list[Instrument]:
        """Get instruments which are (sub) classes of ``instrument_class``."""
        affected_instruments = [instrument for instrument in self.instruments
                                if isinstance(instrument, instrument_class)]
        return affected_instruments

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

        self._add_mp_zone(axes)

    def add_multipacting_zone(self, axe: Axes, **subplot_kw) -> None:
        """Add multipacting zone on a ``plot_instruments`` plot."""
        pass

    def _add_mp_zone(self, axes: dict[ABCMeta, Axes]):
        instruments = self.instruments
        if isinstance(instruments[0], CurrentProbe):
            current_probe = instruments[0]
            electric_field_probe = instruments[1]
        else:
            current_probe = instruments[1]
            electric_field_probe = instruments[0]
        multipactor = ~current_probe.multipactor
        xdata = np.ma.masked_array(electric_field_probe.raw_data.index,
                                   mask=multipactor)
        ydata = np.ma.masked_array(electric_field_probe.ydata,
                                   mask=multipactor)
        axes[type(electric_field_probe)].plot(xdata, ydata,
                                              alpha=.3, lw=6,
                                              color=self._color)


@dataclass
class OldPickUp:
    """Hold information on a single pick-up."""

    name: str
    position: float
    _sample_index: np.ndarray
    e_rf_probe: np.ndarray
    i_mp_probe: np.ndarray
    _color: tuple[float, float, float] | None = None
    _smooth_kw: dict[str, Any] | None = None

    def __post_init__(self):
        """Declare multipactor limits."""
        self.idx_of_mp_zones: list[tuple[int, int]]
        self._smoothed_e_rf_probe: np.ndarray
        self._smoothed_i_mp_probe: np.ndarray

    def __str__(self) -> str:
        """Print the voltage of MP zones if defined."""
        if not hasattr(self, 'idx_of_mp_zones'):
            return self.__repr__()
        voltages = self.mp_voltages
        out = self.name
        if len(voltages) == 0:
            return '\n|\t'.join((out, "no MP"))
        str_voltages = (f"Zone {i}: {volt[0]} V to {volt[1]} V"
                        for i, volt in enumerate(voltages, start=1))
        return '\n|\t'.join((out, *str_voltages))

    def __hash__(self) -> int:
        """Make this class hashable."""
        return hash(repr(self))

    def _get_multipactor_voltage(self, mp_limits_idx: tuple[int, int]
                                 ) -> tuple[float, float]:
        """Compute multipactor voltage start and end."""
        start = self.e_rf_probe[mp_limits_idx[0]]
        end = self.e_rf_probe[mp_limits_idx[1]]
        return (start, end)

    def plot_e_rf(self,
                  axx: Axes,
                  draw_mp_zones: bool = False,
                  smoothed: bool = False) -> None:
        """Plot the electric field as a function of sample index."""
        data_to_plot = self.e_rf_probe
        if smoothed:
            data_to_plot = self.smoothed_e_rf_probe
        line, = axx.plot(self._sample_index,
                         data_to_plot,
                         label=self.name,
                         color=self._color)
        if self._color is None:
            self._color = line.get_color()

        if draw_mp_zones:
            self.add_mp_zone(axx, y_zone=1.05 * np.nanmax(
                self.e_rf_probe))

    def plot_i_mp(self,
                  axx: Axes,
                  draw_mp_zones: bool = True,
                  smoothed: bool = True) -> None:
        """Plot the electron pick-up current vs sample index."""
        data_to_plot = self.i_mp_probe
        if smoothed:
            data_to_plot = self.smoothed_i_mp_probe
        line, = axx.plot(self._sample_index,
                         data_to_plot,
                         label=self.name,
                         color=self._color)
        if self._color is None:
            self._color = line.get_color()

        if draw_mp_zones:
            self.add_mp_zone(axx, y_zone=1.05 * np.nanmax(
                self.i_mp_probe))

    def add_mp_zone(self, axx: Axes, y_zone: float) -> None:
        """Plot the MP zones on the given axe.

        Parameters
        ----------
        axx : Axes
            Axe to plot on.
        y_zone : float
            Height at which the MP arrow will be drawn.

        """
        arrow_kw = {
            'color': self._color,
            'length_includes_head': True,
            'head_width': 2.,
        }
        vline_kw = {
            'color': self._color,
            'lw': .2,
        }
        for idx_zone in self.idx_of_mp_zones:
            delta_x = idx_zone[1] - idx_zone[0]
            axx.arrow(idx_zone[0], y_zone, delta_x, 0., **arrow_kw)
            axx.arrow(idx_zone[1], y_zone, -delta_x, 0., **arrow_kw)
            axx.axvline(idx_zone[0], **vline_kw)
            axx.axvline(idx_zone[1], **vline_kw)

    def determine_multipactor_zones(self,
                                    current_threshold: float,
                                    consecutive_criterion: int = 1,
                                    minimum_number_of_points: int = 1) -> None:
        """Filter and calculate where multipactor happens.

        Parameters
        ----------
        current_threshold : float
            Current above which multipactor is detected.
        consecutive_criterion : int, optional
            Maximum number of measure points between two consecutive
            multipactor zones. Useful for treating measure points that did not
            reach the multipactor current criterion but are in the middle of a
            multipacting zone. The default is 1.
        minimum_number_of_points : int, optional
            Minimum number of consecutive points to consider that there is
            multipactor. Useful for treating isolated measure points that did
            reach the multipactor current criterion. The default is 1.

        """
        mp_indexes = self._measure_points_with_multipactor(
            current_threshold)
        mp_zones = self._isolated_measure_points_to_mp_zones(
            mp_indexes,
            consecutive_criterion,
            minimum_number_of_points,
            print_info=True)
        self.idx_of_mp_zones = mp_zones

    def _measure_points_with_multipactor(self,
                                         current_threshold: float,
                                         print_info: bool = False
                                         ) -> np.ndarray[np.int64]:
        """Determine where multipactor happened.

        Parameters
        ----------
        exceeds_by_in_percent : float
            Current above which multipactor is detected.
        print_info : bool, optional
            To tell if the mp indexes should be printed. The default is False.

        Returns
        -------
        mp_indexes : np.ndarray[np.int64]
            Holds the index of every measure point where measured current is
            high enough to detect multipactor.

        """
        mp_indexes = np.where(self.i_mp_probe > current_threshold)[0]
        if print_info:
            print(f"{self.name}: detected {len(mp_indexes)} points with MP.")
        return mp_indexes

    def _isolated_measure_points_to_mp_zones(
            self,
            mp_indexes: np.ndarray[np.int64],
            consecutive_criterion: int,
            minimum_number_of_points: int,
            print_info: bool = False
    ) -> list[tuple[int, int]]:
        """Calculate the different multipacting zones.

        Parameters
        ----------
        mp_indexes : np.ndarray[np.int64]
            Holds the index of every measure point where measured current is
            high enough to detect multipactor.
        consecutive_criterion : int
            Maximum number of measure points between two consecutive
            multipactor zones. Useful for treating measure points that did not
            reach the multipactor current criterion but are in the middle of a
            multipacting zone.
        minimum_number_of_points : int
            Minimum number of consecutive points to consider that there is
            multipactor. Useful for treating isolated measure points that did
            reach the multipactor current criterion.
        print_info : bool, optional
            To tell if the mp indexes should be printed. The default is False.

        Returns
        -------
        list[tuple[int, int]]
            First and last measure point of every multipacting zone.

        .. todo::
            Check if ``minimum_number_of_points`` correctly handled.

        """
        if len(mp_indexes) == 0:
            return []
        assert consecutive_criterion >= 1, "Should be at least 1 to make any "\
            "sense."
        assert minimum_number_of_points >= 1, "Should be at least 1 to make "\
            "any sense."
        zones = []
        start = mp_indexes[0]
        end = 0
        for i, diff in enumerate(np.diff(mp_indexes)):
            if diff <= consecutive_criterion:
                continue

            end = mp_indexes[i]
            if end - start >= minimum_number_of_points - 1:
                zones.append((start, end))
            start = mp_indexes[i + 1]
        end = mp_indexes[-1]
        zones.append((start, end))

        if print_info:
            print(f"{self.name}: detected {len(zones)} multipactor zones.")
        return zones

    @property
    def mp_voltages(self) -> list[tuple[float, float]]:
        """Compute start and end voltage for each MP zone."""
        voltages = [(self.e_rf_probe[idx[0]],
                     self.e_rf_probe[idx[1]])
                    for idx in self.idx_of_mp_zones]
        return voltages

    @property
    def smoothed_e_rf_probe(self):
        """Measured voltage on electric field probe, but with less noise."""
        if not hasattr(self, '_smoothed_e_rf_probe'):
            self._smoothed_e_rf_probe = smooth(
                input_data=self.e_rf_probe,
                **self.smooth_kw)
        return self._smoothed_e_rf_probe

    @property
    def smoothed_i_mp_probe(self):
        """Measured MP current, but with less noise."""
        if not hasattr(self, '_smoothed_i_mp_probe'):
            self._smoothed_i_mp_probe = smooth(
                input_data=self.i_mp_probe,
                **self.smooth_kw)
        return self._smoothed_i_mp_probe

    @property
    def smooth_kw(self) -> dict[str, Any]:
        """Get the keyword arguments transmitted to the smoothing func.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for the smoothing funuction.

        """
        if self._smooth_kw is None:
            print("Warning!! smooth_kw not defined. Calling a smooth function "
                  "will probably lead to error. Returning empty dict...")
            return {}
        return self._smooth_kw
