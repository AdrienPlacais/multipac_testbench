#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store and treat data from pick-ups.

.. todo::
    Allow to trim data (remove noisy useless data at end of exp)

.. todo::
    name of pick ups in animation

.. todo::
    histograms for mp voltages? Maybe then add a gaussian fit, then we can
    determine the 3sigma multipactor limits?

"""
from abc import ABCMeta
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from multipac_testbench.src.instruments.electric_field.field_probe import \
    FieldProbe
from multipac_testbench.src.instruments.electric_field.reconstructed import \
    Reconstructed
from multipac_testbench.src.instruments.instrument import Instrument
from multipac_testbench.src.instruments.powers import Powers
from multipac_testbench.src.measurement_point.factory import \
    IMeasurementPointFactory
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint
from multipac_testbench.src.util import plot


class MultipactorTest:
    """Holds a mp test with several probes."""

    def __init__(self,
                 filepath: Path,
                 config: dict,
                 freq_mhz: float | None = None,
                 swr: float | None = None,
                 sep: str = ';') -> None:
        """Create all the pick-ups."""
        df_data = pd.read_csv(filepath, sep=sep, index_col="Sample index")
        self._n_points = len(df_data)

        imeasurement_point_factory = IMeasurementPointFactory()
        imeasurement_points = imeasurement_point_factory.run(config, df_data)
        self.global_diagnostics, self.pick_ups = imeasurement_points

        if freq_mhz is None:
            print("MultipactorTest.__init__ warning! Providing frequency will "
                  "soon be mandatory! Setting default 120MHz...")
            freq_mhz = 120.
        self.freq_mhz = freq_mhz

        if swr is None:
            print("MultipactorTest.__init__ warning! Providing SWR will soon "
                  "be mandatory! Setting default to np.NaN...")
            swr = np.NaN
        self.swr = swr

    def add_post_treater(self,
                         *args,
                         only_pick_up_which_name_is: tuple[str, ...] = (),
                         **kwargs) -> None:
        """Add post-treatment functions to instruments."""
        pick_ups = self.pick_ups
        if len(only_pick_up_which_name_is) > 0:
            pick_ups = [pick_up for pick_up in self.pick_ups
                        if pick_up.name in only_pick_up_which_name_is]

        for pick_up in pick_ups:
            pick_up.add_post_treater(*args, **kwargs)

    def detect_multipactor(
            self,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_class: ABCMeta,
            power_is_growing_kw: dict[str, int | float] | None = None,
    ) -> None:
        """Create the :class:`.MultipactorBands` objects.

        Parameters
        ----------
        multipac_detector : Callable[[np.ndarray], np.ndarray[np.bool_]]
            Function that takes in the ``ydata`` of an :class:`.Instrument` and
            returns an array, where True means multipactor and False no
            multipactor.
        instrument_class : ABCMeta
            Type of instrument on which ``multipac_detector`` should be
            applied.
        power_is_growing_kw : dict[str, int | float] | None, optional
            Keyword arguments passed to the function that determines when power
            is increasing, when it is decreasing. The default is None.

        """
        powers = self.get_instrument(Powers)

        for measurement_point in self.get_measurement_points():
            measurement_point.detect_multipactor(multipac_detector,
                                                 instrument_class)
            if not hasattr(measurement_point, 'multipactor_bands'):
                continue
            if not isinstance(powers, Powers):
                continue

            if power_is_growing_kw is None:
                power_is_growing_kw = {}
            power_is_growing = powers.where_is_growing(**power_is_growing_kw)
            measurement_point.multipactor_bands.power_is_growing = \
                power_is_growing

    def plot_instruments_vs_time(
        self,
        instruments_to_plot: tuple[ABCMeta, ...],
        measurement_points_to_exclude: tuple[str, ...] = (),
        png_path: Path | None = None,
        raw: bool = False,
        plot_multipactor: bool = False,
        **fig_kw,
    ) -> tuple[Figure, Axes]:
        """Plot signals measured by ``instruments_to_plot``.

        .. todo::
            Add a ``instruments_to_exclude`` argument. Could replace
            ``measurement_points_to_exclude``.

        Parameters
        ----------
        instruments_to_plot : tuple[ABCMeta, ...]
            Subclass of the :class:`.Instrument` to plot.
        measurement_points_to_exclude : tuple[str, ...], optional
            Name of the measurement points that should not be plotted. The
            default is an empty tuple.
        png_path : Path | None, optional
            If provided, the resulting figure is saved at this path. The
            default is None.
        raw : bool, optional
            If the data that should be plotted is the raw data before
            post-treatment. The default is False. Note that when the
            :attr:`.Instrument.post_treaters` list is empty, raw data is
            plotted even if ``raw==True``.
        multipactor_plots : bool, optional
            To add arrows to detect multipactor. The default is False.
        fig_kw :
            Keyword arguments passed to the ``Figure``.

        Returns
        -------
        fig : Figure
            The created figure.
        axes : Axes
            The created axes.

        """
        fig, instrument_class_axes = plot.create_fig(
            self.freq_mhz,
            self.swr,
            instruments_to_plot,
            xlabel='Measurement index',
            **fig_kw)

        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)

        for measurement_point in measurement_points:
            measurement_point.plot_instrument_vs_time(instrument_class_axes,
                                                      instruments_to_plot,
                                                      raw=raw)

            if plot_multipactor:
                self._add_multipactor_vs_time(measurement_point,
                                              instrument_class_axes)

        plot.finish_fig(fig, instrument_class_axes.values(), png_path)
        return fig, [axes for axes in instrument_class_axes.values()]

    def _add_multipactor_vs_time(self,
                                 measurement_point: IMeasurementPoint,
                                 instrument_class_axes: dict[ABCMeta, Axes],
                                 ) -> None:
        """Show with arrows when multipactor happens.

        Parameters
        ----------
        measurement_point : IMeasurementPoint
            :class:`.PickUp` or :class:`.GlobalDiagnostic` under study.
        instrument_class_axes : dict[ABCMeta, Axes]
            Links instrument class with the axes.

        """
        if not hasattr(measurement_point, 'multipactor_bands'):
            return
        for plotted_instrument_class, axe in instrument_class_axes.items():
            measurement_point._add_multipactor_vs_time(
                axe,
                plotted_instrument_class)

    def animate_instruments_vs_position(
            self,
            instruments_to_plot: Sequence[ABCMeta],
            gif_path: Path | None = None,
            fps: int = 50,
            keep_one_frame_over: int = 1,
            interval: int | None = None,
            **fig_kw,
    ) -> animation.FuncAnimation:
        """Represent measured signals with probe position."""
        fig, axes_instruments = self._prepare_animation_fig(
            instruments_to_plot,
            **fig_kw
        )

        frames = self._n_points - 1
        artists = self._plot_instruments_single_time_step(
            0,
            keep_one_frame_over=keep_one_frame_over,
            axes_instruments=axes_instruments,
            artists=None,
        )

        def update(step_idx: int) -> Sequence[Artist]:
            """Update the ``artists`` defined in outer scope.

            Parameters
            ----------
            step_idx : int
                Step that shall be plotted.

            Returns
            -------
            artists : Sequence[Artist]
                Updated artists.

            """
            self._plot_instruments_single_time_step(
                step_idx,
                keep_one_frame_over=keep_one_frame_over,
                axes_instruments=axes_instruments,
                artists=artists,
            )
            assert artists is not None
            return artists

        if interval is None:
            interval = int(200 / keep_one_frame_over)

        ani = animation.FuncAnimation(fig,
                                      update,
                                      frames=frames,
                                      interval=interval,
                                      repeat=True)

        if gif_path is not None:
            writergif = animation.PillowWriter(fps=fps)
            ani.save(gif_path, writer=writergif)
        return ani

    def _plot_instruments_single_time_step(
            self,
            step_idx: int,
            keep_one_frame_over: int,
            axes_instruments: dict[Axes, list[Instrument]],
            artists: Sequence[Artist] | None = None,
    ) -> Sequence[Artist] | None:
        """Plot all instruments signal at proper axe and time step."""
        if step_idx % keep_one_frame_over != 0:
            return

        sample_index = step_idx + 1

        if artists is None:
            artists = [instrument.plot_vs_position(sample_index, axe=axe)
                       for axe, instruments in axes_instruments.items()
                       for instrument in instruments]
            return artists

        i = 0
        for instruments in axes_instruments.values():
            for instrument in instruments:
                instrument.plot_vs_position(sample_index, artist=artists[i])
                i += 1
        return artists

    def scatter_instruments_data(
        self,
        instruments_to_plot: Sequence[ABCMeta],
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        png_path: Path | None = None,
        **fig_kw,
    ) -> tuple[Figure, Axes]:
        """Plot the data measured by instruments.

        This plot results in important amount of points. It becomes interesting
        when setting different colors for multipactor/no multipactor points and
        can help see trends.

        .. todo::
            Also show from global diagnostic

        .. todo::
            User should be able to select: reconstructed or measured electric
            field.

        """
        if fig_kw is None:
            fig_kw = {}
        fig, instrument_class_axes = plot.create_fig(self.freq_mhz,
                                                     self.swr,
                                                     instruments_to_plot,
                                                     xlabel='Probe index',
                                                     **fig_kw)
        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)
        for i, measurement_point in enumerate(measurement_points):
            measurement_point.scatter_instruments_data(instrument_class_axes,
                                                       xdata=float(i),
                                                       )

        fig, axes = plot.finish_fig(fig,
                                    instrument_class_axes.values(),
                                    png_path)
        return fig, axes

    def filter_measurement_points(
            self,
            to_exclude: Sequence[str | IMeasurementPoint] = (),
    ) -> Sequence[IMeasurementPoint]:
        """Get measurement points (Pick-Ups and GlobalDiagnostic)."""
        print("MultipactorTest.filter_measurement_points is deprecated. "
              "Use MultipactorTest.get_measurement_points instead.")
        return self.get_measurement_points(to_exclude=to_exclude)

    def filter_instruments(
            self,
            instrument_class: ABCMeta,
            measurement_points: Sequence[IMeasurementPoint] | None = None,
            instruments_to_ignore: Sequence[Instrument | str] = (),
    ) -> list[Instrument]:
        """Get all instruments of desired class from ``measurement_points``.

        But remove the instruments to ignore.

        Parameters
        ----------
        instrument_class : ABCMeta
            Class of the desired instruments.
        measurement_points : Sequence[IMeasurementPoint] | None, optional
            The measurement points from which you want the instruments. The
            default is None, in which case we look into every
            :class:`IMeasurementPoint` attribute of self.
        instruments_to_ignore : Sequence[Instrument | str], optional
            The :class:`.Instrument` or instrument names you do not want. The
            default is an empty tuple, in which case no instrument is ignored.

        Returns
        -------
        instruments : list[Instrument]
            All the instruments matching the required conditions.

        """
        if measurement_points is None:
            measurement_points = self.get_measurement_points()

        instruments_2d = [
            measurement_point.get_instruments(
                instrument_class,
                instruments_to_ignore=instruments_to_ignore,
            )
            for measurement_point in measurement_points
        ]
        instruments = [instrument
                       for instrument_1d in instruments_2d
                       for instrument in instrument_1d]
        return instruments

    def get_measurement_points(
        self,
        names: Sequence[str] | None = None,
        to_exclude: Sequence[str | IMeasurementPoint] = (),
    ) -> Sequence[IMeasurementPoint]:
        """Get all or some measurement points.

        Parameters
        ----------
        names : Sequence[str], optional
            If given, only the :class:`.IMeasurementPoint` which name is in
            ``names`` will be returned.
        to_exclude : Sequence[str | IMeasurementPoint], optional
            List of objects or objects names to exclude from returned list.

        Returns
        -------
        i_measurement_points : Sequence[IMeasurementPoint]
            The desired objects.

        """
        names_to_exclude = [x if isinstance(x, str) else x.name
                            for x in to_exclude]

        measurement_points = [
            x for x in self.pick_ups + [self.global_diagnostics]
            if x is not None and x.name not in names_to_exclude
        ]

        if names is not None and len(names) > 0:
            return [x for x in measurement_points if x.name in names]
        return measurement_points

    def get_measurement_point(
        self,
        name: str | None = None,
        to_exclude: Sequence[str | IMeasurementPoint] = (),
    ) -> IMeasurementPoint:
        """Get all or some measurement points. Ensure there is only one.

        Parameters
        ----------
        name : str | None, optional
            If given, only the :class:`.IMeasurementPoint` which name is in
            ``names`` will be returned.
        to_exclude : Sequence[str | IMeasurementPoint], optional
            List of objects or objects names to exclude from returned list.

        Returns
        -------
        measurement_point : IMeasurementPoint
            The desired object.

        """
        if name is not None:
            name = name,
        measurement_points = self.get_measurement_points(name, to_exclude)
        assert len(measurement_points) == 1, ("Only one IMeasurementPoint "
                                              "should match.")
        return measurement_points[0]

    def get_instruments(
            self,
            instrument_class: ABCMeta,
            measurement_points_to_exclude: Sequence[IMeasurementPoint
                                                    | str] = (),
            instruments_to_ignore: Sequence[Instrument | str] = (),
    ) -> list[Instrument]:
        """Get all instruments of type ``instrument_class``."""
        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)
        instruments = self.filter_instruments(
            instrument_class,
            measurement_points,
            instruments_to_ignore=instruments_to_ignore)
        return instruments

    def get_instrument(
            self,
            instrument_class: ABCMeta,
            measurement_points_to_exclude: Sequence[IMeasurementPoint
                                                    | str] = (),
            instruments_to_ignore: Sequence[Instrument | str] = (),
    ) -> Instrument | None:
        """Get a single instrument of type ``instrument_class``."""
        instruments = self.get_instruments(instrument_class,
                                           measurement_points_to_exclude,
                                           instruments_to_ignore)
        if len(instruments) == 0:
            print("multipactor_test.get_instrument warning! No instrument "
                  "found.")
            return
        if len(instruments) > 1:
            print("multipactor_test.get_instrument warning! Several "
                  "instruments found. Returning first one.")
        return instruments[0]

    def _get_limits(
            self,
            axes_instruments: dict[Axes, Sequence[Instrument]],
            instruments_to_ignore_for_limits: Sequence[Instrument | str] = (),
    ) -> dict[Axes, tuple[float, float]]:
        """Set limits for the plots."""
        names_to_ignore = [x if isinstance(x, str) else x.name
                           for x in instruments_to_ignore_for_limits]
        limits = {}
        for axe, instruments in axes_instruments.items():
            all_ydata = [instrument.ydata for instrument in instruments
                         if instrument.name not in names_to_ignore]

            lowers = [np.nanmin(ydata) for ydata in all_ydata]
            lower = min(lowers)

            uppers = [np.nanmax(ydata) for ydata in all_ydata]
            upper = max(uppers)
            amplitude = abs(upper - lower)

            limits[axe] = (lower - .1 * amplitude, upper + .1 * amplitude)
        return limits

    def _prepare_animation_fig(
        self,
        instruments_to_plot: tuple[ABCMeta, ...],
        measurement_points_to_exclude: tuple[str, ...] = (),
        instruments_to_ignore_for_limits: tuple[str, ...] = (),
        instruments_to_ignore: Sequence[Instrument | str] = (),
        **fig_kw,
    ) -> tuple[Figure, dict[Axes, list[Instrument]]]:
        """Prepare the figure and axes for the animation.

        Parameters
        ----------
        instruments_to_plot : tuple[ABCMeta, ...]
            Classes of instruments you want to see.
        measurement_points_to_exclude : tuple[str, ...]
            Measurement points that should not appear.
        instruments_to_ignore_for_limits : tuple[str, ...]
            Instruments to plot, but that can go off limits.
        instruments_to_ignore : Sequence[Instrument | str]
            Instruments that will not even be plotted.
        fig_kw :
            Other keyword arguments for Figure.

        Returns
        -------
        fig : Figure
         Figure holding the axes.
        axes_instruments : dict[Axes, list[Instrument]]
            Links the instruments to plot with the Axes they should be plotted
            on.

        """
        fig, instrument_class_axes = plot.create_fig(self.freq_mhz,
                                                     self.swr,
                                                     instruments_to_plot,
                                                     xlabel='Position [m]',
                                                     **fig_kw)

        for instrument_class, axe in instrument_class_axes.items():
            axe.set_ylabel(instrument_class.ylabel())

        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)

        axes_instruments = {
            axe: self.filter_instruments(
                instrument_class,
                measurement_points,
                instruments_to_ignore=instruments_to_ignore)
            for instrument_class, axe in instrument_class_axes.items()
        }

        y_limits = self._get_limits(
            axes_instruments,
            instruments_to_ignore_for_limits=instruments_to_ignore_for_limits)

        axe = None
        for axe, y_lim in y_limits.items():
            axe.set_ylim(y_lim)

        return fig, axes_instruments

    def reconstruct_voltage_along_line(
            self,
            name: str,
            probes_to_ignore: Sequence[str | FieldProbe],
    ) -> Reconstructed:
        """Reconstruct the voltage profile from the e field probes."""
        e_field_probes = self.filter_instruments(FieldProbe,
                                                 self.pick_ups,
                                                 probes_to_ignore)
        assert self.global_diagnostics is not None
        powers = self.get_instrument(Powers)

        reconstructed = Reconstructed(
            name=name,
            raw_data=None,
            e_field_probes=e_field_probes,
            powers=powers,
            freq_mhz=self.freq_mhz,
        )
        reconstructed.fit_voltage()

        self.global_diagnostics.add_instrument(reconstructed)

        return reconstructed

    def plot_multipactor_limits(
            self,
            instrument_class_to_plot: ABCMeta,
            measurement_points_to_exclude: Sequence[str
                                                    | IMeasurementPoint] = (),
            png_path: Path | None = None,
            multipactor_measured_at: IMeasurementPoint | str | None = None,
            **fig_kw,
    ) -> tuple[Figure, Axes]:
        """Plot lower and upper multipacting limits evolution.

        As for now, only one instrument should be plotted, and only one
        detector instrument should be defined.

        .. note::
            In order to discriminate lower multipacting barrier from upper
            multipacting barrier, we need to determine when the power is
            growing and when it is increasing. This can be non-trivial. Check
            :meth:`.Powers.where_is_growing` and ``power_is_growing_kw``.

        Parameters
        ----------
        instrument_class_to_plot : {Powers, FieldProbe, Reconstructed}
            The instrument which data will be plotted. As the goal of this
            method is to plot multipacting limits, it is expected that the
            instrument class is related to power/electric field/voltage.
        measurement_points_to_exclude : Sequence[str | IMeasurementPoint]
            Some measurement points to exclude from plot.
        png_path : Path | None
            If provided, will save the Figure. The default is None.
        multipactor_measured_at: IMeasurementPoint | str | None = None
            If you want to plot the multipactor bands from an instrument that
            is not at the same position as the instrument data to plot.
        fig_kw :
            Other keyword arguments passed to the ``Figure``.

        Returns
        -------
        tuple[Figure, Axes]
            Created fig and axes.

        """
        if instrument_class_to_plot not in (Powers, FieldProbe, Reconstructed):
            print("multipactor_test.plot_multipactor_limits warning: you want "
                  f"to plot the values measured by {instrument_class_to_plot} "
                  "at entry and exit of multipactor zones. Does it have any "
                  "sense?")

        fig, instrument_class_axes = plot.create_fig(
            self.freq_mhz,
            self.swr,
            (instrument_class_to_plot, ),
            xlabel='Measurement index',
            **fig_kw
        )

        instrument_to_plot = self.get_instrument(instrument_class_to_plot,
                                                 measurement_points_to_exclude,
                                                 )
        assert instrument_to_plot is not None, (
            f"No {instrument_class_to_plot} instrument was found.")

        if not isinstance(multipactor_measured_at, IMeasurementPoint):
            multipactor_measured_at = self.get_measurement_point(
                name=multipactor_measured_at,
                to_exclude=measurement_points_to_exclude)

        multipactor_bands = multipactor_measured_at.multipactor_bands

        lower_values, upper_values = instrument_to_plot.values_at_barriers(
            multipactor_bands
        )

        axe = instrument_class_axes[instrument_class_to_plot]
        lower_values.plot(ax=axe, kind='line', drawstyle='steps-post')
        upper_values.plot(ax=axe, kind='line', drawstyle='steps-post')
        axe.grid(True)
        plot.finish_fig(fig, instrument_class_axes.values(), png_path)
        return fig, [axes for axes in instrument_class_axes.values()]
