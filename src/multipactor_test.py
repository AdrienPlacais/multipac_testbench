#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store and treat data from pick-ups.

.. todo::
    Allow to trim data (remove noisy useless data at end of exp)

.. todo::
    name of pick ups in animation

"""
from abc import ABCMeta
from functools import partial
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
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
from multipac_testbench.src.measurement_point.pick_up import PickUp


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

    def set_multipac_detector(self,
                              *args,
                              only_pick_up_which_name_is: tuple[str, ...] = (),
                              **kwargs) -> None:
        """Set multipactor detection functions to instruments."""
        pick_ups = self.pick_ups
        if len(only_pick_up_which_name_is) > 0:
            pick_ups = [pick_up for pick_up in self.pick_ups
                        if pick_up.name in only_pick_up_which_name_is]

        for pick_up in pick_ups:
            pick_up.set_multipac_detector(*args, **kwargs)

    def plot_instruments_vs_time(
        self,
        instruments_to_plot: tuple[ABCMeta, ...],
        measurement_points_to_exclude: tuple[str, ...] = (),
        png_path: Path | None = None,
        raw: bool = False,
        multipactor_plots: dict[ABCMeta, ABCMeta] | None = None,
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
        multipactor_plots : dict[ABCMeta, ABCMeta] | None, optional
            Keys are the Instrument subclass for which you want to see the
            multipactor zones. Values are the Instrument subclass that detect
            the multipactor. The default is None, in which case no multipacting
            zone is drawn.
        fig_kw :
            Keyword arguments passed to the ``Figure``.

        Returns
        -------
        fig : Figure
            The created figure.
        axes : Axes
            The created axes.

        """
        fig, instrument_class_axes = self._create_fig(instruments_to_plot,
                                                      **fig_kw)

        measurement_points = self._filter_measurement_points(
            to_exclude=measurement_points_to_exclude)

        for measurement_point in measurement_points:
            measurement_point.plot_instrument_vs_time(instrument_class_axes,
                                                      instruments_to_plot,
                                                      raw=raw)

            if multipactor_plots is not None:
                self._add_multipactor_vs_time(measurement_point,
                                              instrument_class_axes,
                                              multipactor_plots)

        for axe in instrument_class_axes.values():
            axe.legend()

        if png_path is not None:
            fig.savefig(png_path)

        return fig, [axes for axes in instrument_class_axes.values()]

    def _add_multipactor_vs_time(self,
                                 measurement_point: IMeasurementPoint,
                                 instrument_class_axes: dict[ABCMeta, Axes],
                                 multipactor_plots: dict[ABCMeta, ABCMeta]
                                 ) -> None:
        """Show with arrows when multipactor happens.

        Parameters
        ----------
        measurement_point : IMeasurementPoint
            :class:`.PickUp` or :class:`.GlobalDiagnostic` under study.
        instrument_class_axes : dict[ABCMeta, Axes]
            Links instrument class with the axes.
        multipactor_plots : dict[ABCMeta, ABCMeta]
            Links the instrument plot where multipactor should appear (keys)
            with the instruments that actually detect the multipactor (values).

        """
        for plotted_instr, detector_instr in multipactor_plots.items():
            measurement_point._add_multipactor_vs_time(
                instrument_class_axes[plotted_instr],
                plotted_instr,
                detector_instr,
            )

    def animate_instruments_vs_position(
            self,
            instruments_to_plot: tuple[ABCMeta, ...],
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

    def scatter_instruments_data(self,
                                 instruments_to_plot: Sequence[ABCMeta],
                                 mp_detector_instrument: ABCMeta,
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
        fig, instrument_class_axes = self._create_fig(instruments_to_plot,
                                                      **fig_kw)
        for i, pick_up in enumerate(self.pick_ups):
            if i == 0:
                continue
            pick_up.scatter_instruments_data(instrument_class_axes,
                                             mp_detector_instrument,
                                             xdata=float(i),
                                             )
        if png_path is not None:
            fig.savefig(png_path)
        axes = [axes for axes in instrument_class_axes.values()]
        axes[0].legend()

        return fig, axes

    def _create_fig(self,
                    instruments_to_plot: Sequence[ABCMeta] = (),
                    **fig_kw,
                    ) -> tuple[Figure, dict[ABCMeta, Axes]]:
        """Create the figure and axes.

        Parameters
        ----------
        instruments_to_plot : tuple[ABCMeta, ...]
            Class of the instruments to be plotted.
        fig_kw :
            Keyword arguments passsed to the Figure constructor.

        Returns
        -------
        fig : Figure
            Figure holding the axes.
        instrument_class_axes : dict[ABCMeta, Axes]
            Dictionary linking the class of the instruments to plot with the
            associated axes.

        """
        nrows = len(instruments_to_plot)
        fig, instrument_class_axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=True,
            **fig_kw
        )

        # ensure that axes is an iterable
        if nrows == 1:
            instrument_class_axes = [instrument_class_axes, ]

        instrument_class_axes = dict(zip(instruments_to_plot,
                                         instrument_class_axes))

        axe = None
        for instrument_class, axe in instrument_class_axes.items():
            axe.grid(True)
            axe.set_ylabel(instrument_class.ylabel())
        assert isinstance(axe, Axes)
        if axe is not None:
            axe.set_xlabel("Measurement index")

        fig.suptitle(f"f = {self.freq_mhz}MHz; SWR = {self.swr}")

        return fig, instrument_class_axes

    def _filter_measurement_points(
            self,
            to_exclude: tuple[str | IMeasurementPoint, ...] = (),
    ) -> list[IMeasurementPoint]:
        """Get measurement points (Pick-Ups and GlobalDiagnostic)."""
        names_to_exclude = [x if isinstance(x, str) else x.name
                            for x in to_exclude]

        measurement_points = [x for x in self.pick_ups
                              if x.name not in names_to_exclude]
        if self.global_diagnostics is None:
            return measurement_points
        if self.global_diagnostics.name in names_to_exclude:
            return measurement_points
        measurement_points.append(self.global_diagnostics)
        return measurement_points

    def _filter_instruments(self,
                            instrument_class: ABCMeta,
                            measurement_points: Sequence[IMeasurementPoint],
                            instruments_to_ignore: Sequence[Instrument | str],
                            ) -> list[Instrument]:
        """Get all instruments of desired class from ``measurement_points``.

        But remove the instruments to ignore.

        Parameters
        ----------
        instrument_class : ABCMeta
            Class of the desired instruments.
        measurement_points : Sequence[IMeasurementPoint]
            The measurement points from which you want the instruments.
        instruments_to_ignore : Sequence[Instrument | str]
            The :class:`.Instrument` or instrument names you do not want.

        Returns
        -------
        instruments : list[Instrument]
            All the instruments matching the required conditions.

        """
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
        fig, instrument_class_axes = self._create_fig(instruments_to_plot,
                                                      **fig_kw)
        for instrument_class, axe in instrument_class_axes.items():
            axe.set_ylabel(instrument_class.ylabel())

        measurement_points = self._filter_measurement_points(
            to_exclude=measurement_points_to_exclude)

        axes_instruments = {
            axe: self._filter_instruments(
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
            axe.grid(True)
        if axe is not None:
            axe.set_xlabel('Position [m]')

        return fig, axes_instruments

    def reconstruct_voltage_along_line(
            self,
            name: str,
            probes_to_ignore: Sequence[str | FieldProbe],
    ) -> Reconstructed:
        """Reconstruct the voltage profile from the e field probes."""
        e_field_probes = self._filter_instruments(FieldProbe,
                                                  self.pick_ups,
                                                  probes_to_ignore)
        assert self.global_diagnostics is not None
        powers = self._filter_instruments(Powers,
                                          [self.global_diagnostics],
                                          probes_to_ignore)
        assert len(powers) == 1
        powers = powers[0]

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
