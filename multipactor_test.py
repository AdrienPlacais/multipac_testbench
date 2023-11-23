#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store and treat data from pick-ups."""
from abc import ABCMeta
from typing import Any, Sequence
from pathlib import Path
import os.path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.container import StemContainer
import matplotlib.animation as animation

from multipac_testbench.pick_up.pick_up import PickUp
from multipac_testbench.file_configuration import FileConfiguration

from multipac_testbench.instruments.factory import InstrumentFactory


class MultipactorTest:
    """Holds a mp test with several probes."""

    def __init__(self,
                 filepath: Path,
                 config: dict,
                 sep: str = ';') -> None:
        """Create all the pick-ups."""
        df_data = pd.read_csv(filepath, sep=sep, index_col="Sample index")
        instrument_factory = InstrumentFactory()
        self.pick_ups = [PickUp(key, df_data, instrument_factory, **value)
                         for key, value in config.items()]

        self._n_points = len(df_data)

    def add_post_treater(self,
                         *args,
                         only_pick_up_which_name_is: tuple[str, ...] = (),
                         **kwargs) -> None:
        """Add post-treatment functions to instruments."""
        affected_pick_ups = self.pick_ups
        if len(only_pick_up_which_name_is) > 0:
            affected_pick_ups = [pick_up for pick_up in self.pick_ups
                                 if pick_up.name in only_pick_up_which_name_is]

        for pick_up in affected_pick_ups:
            pick_up.add_post_treater(*args, **kwargs)

    def set_multipac_detector(self,
                              *args,
                              only_pick_up_which_name_is: tuple[str, ...] = (),
                              **kwargs) -> None:
        """Set multipactor detection functions to instruments."""
        affected_pick_ups = self.pick_ups
        if len(only_pick_up_which_name_is) > 0:
            affected_pick_ups = [pick_up for pick_up in self.pick_ups
                                 if pick_up.name in only_pick_up_which_name_is]

        for pick_up in affected_pick_ups:
            pick_up.set_multipac_detector(*args, **kwargs)

    def plot_pick_ups(
            self,
            pick_up_to_exclude: tuple[str, ...] = (),
            instruments_to_plot: tuple[ABCMeta, ...] = (),
            png_path: Path | None = None,
            raw: bool = False,
            multipactor_plots: dict[ABCMeta, ABCMeta] | None = None,
            **fig_kw,
    ) -> None:
        """Plot the different signals at the different pick-ups.

        Parameters
        ----------
        pick_up_to_exclude : tuple[str, ...], optional
            Name of the pick-ups that should not be plotted. The default is an
            empty tuple.
        instruments_to_plot : tuple[ABCMeta, ...]
            Subclass of the :class:`.Instrument` to plot. The default is an
            empty tuple, in which case nothing is plotted.
        png_path : Path | None, optional
            If provided, the resulting figure is saved at this path. The
            default is None.
        raw : bool, optional
            If the data that should be plotted is the raw data before
            post-treatment. The default is False.
        multipactor_plots : dict[ABCMeta, ABCMeta] | None, optional
            Keys are the Instrument subclass for which you want to see the
            multipactor zones. Values are the Instrument subclass that detect
            the multipactor. The default is None, in which case no multipacting
            zone is drawn.
        fig_kw :
            Keyword arguments passed to the ``Figure``.

        """
        fig, axes = self._create_fig(instruments_to_plot, **fig_kw)

        for pick_up in self.pick_ups:
            if pick_up.name in pick_up_to_exclude:
                continue

            pick_up.plot_instruments(axes, instruments_to_plot, raw=raw)

            if multipactor_plots is not None:
                self.add_multipacting_zone(pick_up,
                                           axes,
                                           multipactor_plots)

        for axe in axes.values():
            axe.legend()

        if png_path is not None:
            fig.savefig(png_path)

    def add_multipacting_zone(
            self,
            pick_up: PickUp,
            axes: dict[ABCMeta, Axes],
            multipactor_plots: dict[ABCMeta, ABCMeta]
    ) -> None:
        """Add multipacting zones on pick-up plot.

        Parameters
        ----------
        pick_up : PickUp
            Pick-up which detected multipactor.
        axes : dict[ABCMeta, Axes]
            Dictionary holding the plots and the associated instrument
            subclass.
        multipactor_plots : dict[ABCMeta, ABCMeta] | None, optional
            Keys are the Instrument subclass for which you want to see the
            multipactor zones. Values are the Instrument subclass that detect
            the multipactor.

        """
        for plotted_instr, detector_instr in multipactor_plots.items():
            pick_up.add_multipacting_zone(axes[plotted_instr],
                                          plotted_instr,
                                          detector_instr,
                                          )

    def _create_fig(self,
                    instruments_to_plot: tuple[ABCMeta, ...] = (),
                    **fig_kw,
                    ) -> tuple[Figure, dict[ABCMeta, Axes]]:
        """Create the figure."""
        fig, axes = plt.subplots(
            nrows=len(instruments_to_plot),
            ncols=1,
            sharex=True,
            **fig_kw
        )
        axes = {instrument_class: axe
                for instrument_class, axe in zip(instruments_to_plot, axes)}

        for instrument_class, axe in axes.items():
            assert isinstance(axe, Axes)
            axe.grid(True)
            axe.set_ylabel(instrument_class.ylabel())
        axe.set_xlabel("Measurement index")
        return fig, axes

    def animate_pick_ups(self,
                         instruments_to_plot: tuple[ABCMeta, ...],
                         pick_ups_to_exclude: tuple[str, ...] = (),
                         pick_ups_to_ignore_for_limits: tuple[str, ...] = (),
                         gif_path: Path | None = None,
                         fps: int = 50,
                         keep_one_frame_over: int = 1,
                         **fig_kw,
                         ) -> None:
        """Animate the pick-up measurements.

        Parameters
        ----------
        pick_up_to_exclude : tuple[str, ...], optional
            The pick-ups that should not be plotted. The default is an empty
            tuple, in which case all the pick-ups are represented.
        pick_ups_to_ignore_for_limits : tuple[str, ...], optional
            The pick-ups that should be plotted, but are not considered for the
            limits of the plots.
        gif_path : Path | None, optional
            The path where the resulting ``.gif`` will be saved. The optional
            is None, in which case the animation is not saved.
        fps : int, optional
            Number of frame per seconds in the save ``.gif``. The default is
            50.
        keep_one_frame_over : int, optional
           To reduce memory consumption of the animation, plot only one frame
           over ``keep_one_frame_over``. The default is 1, in which case all
           frames are plotted.
        fig_kw :
            Keyword arguments given to the matplotlib ``Figure``.

        .. todo::
            Name of pick-ups in x axis.
            Optimize. Surely I do not need to redraw everything at every
            iteration.

        """
        subplot_kw = {'xlabel': r'Probe position [m]'}
        fig, axes = self._create_fig(instruments_to_plot, **fig_kw)

        to_ignore = pick_ups_to_exclude + pick_ups_to_ignore_for_limits
        y_limits = {instrument: self._get_limits(instrument,
                                                 to_ignore=to_ignore)
                    for instrument in instruments_to_plot}

        def _redraw() -> None:
            """Redraw what does not change between two frames."""
            for instrument_class in instruments_to_plot:
                axe = axes[instrument_class]
                axe.set_ylabel(instrument_class.ylabel())
                axe.set_ylim(y_limits[instrument_class])
                axe.grid(True)

        locs = [pick_up.position
                for pick_up in self.pick_ups
                if pick_up.name not in pick_ups_to_exclude
                ]

        def _plot_pick_ups_single_time_step(
                step_idx: int
        ) -> Sequence[StemContainer] | None:
            """Plot as stem the instrument signals.

            Parameters
            ----------
            step_idx : int
                Current measurement point.

            """
            if step_idx % keep_one_frame_over != 0:
                return

            for axe in axes.values():
                axe.clear()
            _redraw()

            lines = []
            for instrument_class, axe in axes.items():
                heads = [
                    pick_up.get_instrument_data(instrument_class)[step_idx]
                    for pick_up in self.pick_ups
                    if pick_up.name not in pick_ups_to_exclude
                    ]
                line = axe.stem(locs, heads)
                lines.append(line)
            return lines

        frames = self._n_points
        ani = animation.FuncAnimation(
            fig,
            _plot_pick_ups_single_time_step,
            frames=frames,
            repeat=True,
        )
        plt.show()

        if gif_path is not None:
            writergif = animation.PillowWriter(fps=fps)
            ani.save(gif_path, writer=writergif)

    def _get_limits(self,
                    instrument_class: ABCMeta,
                    to_ignore: tuple[str, ...] = (),
                    ) -> tuple[float, float]:
        """Set limits for the plots.

        Parameters
        ----------
        instrument_class : ABCMeta
            Instrument subclass.
        to_ignore : tuple[str, ...]
            Name of the pick-ups that should not be considered for the limits.

        Returns
        -------
        tuple[float, float]
            Lower and upper limits that should allow to visualize all data but
            the one in pick-ups to ignore.

        """
        all_ydata = [pick_up.get_instrument_data(instrument_class)
                     for pick_up in self.pick_ups
                     if pick_up.name not in to_ignore]
        lowers = [np.nanmin(ydata) for ydata in all_ydata]
        lower = min(lowers)

        uppers = [np.nanmax(ydata) for ydata in all_ydata]
        upper = max(uppers)
        amplitude = abs(upper - lower)
        return lower - .1 * amplitude, upper + .1 * amplitude


class OldMultipactorTest:
    """Holds a mp test with several probes."""

    def __init__(self,
                 filepath: str,
                 file_config: FileConfiguration,
                 skiprows: int = 1,
                 delimiter: str | None = None,
                 smooth_kw: dict[str, Any] | None = None,
                 ) -> None:
        """
        Load the file and create the pick-ups.

        filepath : str
            The path to the file containing pick-up data. The first ``skiprow``
            rows are ignored (header). You must ensure that there is no empty
            column at the end of each line. Use ``.csv`` or ``.txt`` for file
            format.
        file_config : FileConfiguration
            An object holding names and position of data in ``filepath`` for
            every pick-up.
        skiprows : int, optional
            Number of header lines. The default is 1.
        delimiter : str | None, optional
            Column delimiter. The default is None, which corresponds to a
            space.
        smooth_kw : dict[str, Any] | None, optional
            Keyword passed to the smoothed function in :mod:`.filters`. The
            default is None, in which case you shall not try to smooth your
            data.

        .. todo::
            Proper data treatment to accept default file.

        """
        self._filepath = filepath
        data = np.loadtxt(filepath, skiprows=skiprows, delimiter=delimiter)

        self.pick_ups = self._instantiate_pick_ups(data,
                                                   file_config,
                                                   smooth_kw)
        self._data = data

    def __str__(self) -> str:
        """Print the voltage of the MP zone of every pick-up."""
        out = (pick_up.__str__() for pick_up in self.pick_ups)
        return '\n'.join(out)

    def _instantiate_pick_ups(self,
                              data: np.ndarray,
                              file_config: FileConfiguration,
                              smooth_kw: dict[str, Any] | None,
                              ) -> list[PickUp]:
        """Create the pick-ups."""
        pick_ups = []
        for i, pick_up_name in enumerate(file_config.names):
            position = file_config.positions[i]
            e_rf_idx = file_config.e_rf_idx[i]
            i_mp_idx = file_config.i_mp_idx[i]
            pick_ups.append(PickUp(pick_up_name,
                                   position,
                                   data[:, 0],
                                   data[:, e_rf_idx],
                                   data[:, i_mp_idx],
                                   _smooth_kw=smooth_kw,
                                   )
                            )
        return pick_ups

    def where_multipactor(self,
                          current_threshold: float,
                          consecutive_criterion: int,
                          minimum_number_of_points: int,
                          ) -> None:
        """Determine for every pick-up where there is multipactor.

        Parameters
        ----------
        current_threshold : float
            Current above which multipactor is detected.
        consecutive_criterion : int
            Maximum number of measure points between two consecutive
            multipactor zones. Useful for treating measure points that did not
            reach the multipactor current criterion but are in the middle of a
            multipacting zone.
        minimum_number_of_points : int
            Minimum number of consecutive points to consider that there is
            multipactor. Useful for treating isolated measure points that did
            reach the multipactor current criterion.

        Returns
        -------
        None

        """
        for pick_up in self.pick_ups:
            pick_up.determine_multipactor_zones(current_threshold,
                                                consecutive_criterion,
                                                minimum_number_of_points)

    def plot_pick_ups(self,
                      to_exclude: tuple[str, ...] = (),
                      png_path: str = '',
                      smoothed: tuple[bool, bool] = (False, True),
                      **fig_kw
                      ) -> None:
        """Plot the measured current and voltage @ every pick-up.

        Parameters
        ----------
        to_exclude : tuple[str, ...], optional
            Name of the pick-ups to exclude from the plot. The default is an
            empty tuple.
        png_path : str, optional
            Filepath where plot will be saved. The default is an empty string,
            in which case the name will be the one of the data file but with
            `.png` instead of `.csv`.
        smoothed : tuple[bool, bool]
            If the data should be smoothed for the electric field probe (1st
            element) and for the MP current (second element). The default is
            (False, True).
        fig_kw :
            Keyword arguments passed to the ``plt.Figure``.

        """
        subplot_kw = {'xlabel': 'Measurement index'}
        fig, field_ax, current_ax = self._e_rf_and_i_mp_plots(
            subplot_kw,
            **fig_kw)

        for pick_up in self.pick_ups:
            if pick_up.name in to_exclude:
                continue

            pick_up.plot_e_rf(field_ax,
                              draw_mp_zones=False,
                              smoothed=smoothed[0])
            pick_up.plot_i_mp(current_ax,
                              draw_mp_zones=True,
                              smoothed=smoothed[1])

        field_ax.grid(True)
        current_ax.grid(True)
        field_ax.legend()

        if png_path == '':
            png_path = os.path.splitext(self._filepath)[0] + '.png'
        fig.savefig(png_path)

    def animate_pick_ups(self,
                         to_exclude: tuple[str, ...] = (),
                         to_ignore_for_limits: tuple[str, ...] = (),
                         gif_path: str = '',
                         fps: int = 50,
                         keep_one_frame_over: int = 1,
                         **fig_kw,
                         ) -> None:
        """Animate the pick-up measurements.

        Parameters
        ----------
        to_exclude : tuple[str, ...]
            The pick-ups that should not be measured.
        gif_path : str, optional
            The path where the resulting ``.gif`` will be saved. The optional
            is an empty string, in which case the animation is not saved.
        fps : int, optional
            Number of frame per seconds in the save ``.gif``. The default is
            50.
        keep_one_frame_over : int, optional
           To reduce memory consumption of the animation, plot only one frame
           over ``keep_one_frame_over``. The default is 1, in which case all
           frames are plotted.
        fig_kw :
            Keyword arguments given to the matplotlib ``Figure``.

        .. todo::
            Name of pick-ups in x axis.
            Optimize. Surely I do not need to redraw everything at every
            iteration.

        """
        subplot_kw = {'xlabel': r'Probe position $[m]$'}
        fig, field_ax, current_ax = self._e_rf_and_i_mp_plots(
            subplot_kw,
            **fig_kw)

        to_ignore = to_exclude + to_ignore_for_limits
        x_lim = self._get_limits('position', to_ignore)
        y_lim1 = self._get_limits('e_rf_probe', to_ignore)
        y_lim2 = self._get_limits('i_mp_probe', to_ignore)

        def _redraw() -> None:
            """Redraw what does not change between two frames."""
            field_ax.set_ylabel(r'Field probe $[V]$')
            current_ax.set_ylabel(r'MP current $[\mu A]$')
            current_ax.set_xlim(x_lim)
            field_ax.set_ylim(y_lim1)
            current_ax.set_ylim(y_lim2)
            field_ax.grid(True)
            current_ax.grid(True)

        locs = [pick_up.position
                for pick_up in self.pick_ups
                if pick_up.name not in to_exclude
                ]

        def _plot_pick_ups_single_time_step(
                step_idx: int
        ) -> tuple[StemContainer, StemContainer] | None:
            """Plot as stem the current and voltage of pick ups at position.

            Parameters
            ----------
            step_idx : int
                Current measurement point.

            """
            if step_idx % keep_one_frame_over != 0:
                return

            field_ax.clear()
            current_ax.clear()
            _redraw()

            heads_field = [pick_up.e_rf_probe[step_idx]
                           for pick_up in self.pick_ups
                           if pick_up.name not in to_exclude
                           ]
            field_line = field_ax.stem(locs, heads_field)
            heads_current = [pick_up.i_mp_probe[step_idx]
                             for pick_up in self.pick_ups
                             if pick_up.name not in to_exclude
                             ]
            current_line = current_ax.stem(locs, heads_current)
            return field_line, current_line

        frames = len(self.pick_ups[0].e_rf_probe)
        ani = animation.FuncAnimation(
            fig,
            _plot_pick_ups_single_time_step,
            frames=frames,
            repeat=True,
        )
        plt.show()

        if gif_path == '':
            gif_path = os.path.splitext(self._filepath)[0] + '.gif'
        writergif = animation.PillowWriter(fps=fps)
        ani.save(gif_path, writer=writergif)

    def _get_limits(self,
                    attribute: str,
                    to_ignore: tuple[str, ...] = (),
                    ) -> tuple[float, float]:
        """Set limits for the plots.

        Parameters
        ----------
        attribute : str
            Name of the attribute. Must be in :class:`.PickUp` attributes.
        to_ignore : tuple[str, ...]
            Name of the pick-ups that should not be considered for the limits.

        Returns
        -------
        tuple[float, float]
            Lower and upper limits that should allow to visualize all data but
            the one in pick-ups to ignore.

        """
        lower, upper = None, None
        for pick_up in self.pick_ups:
            if pick_up.name in to_ignore:
                continue

            if lower is None:
                assert upper is None
                lower = np.nanmin(getattr(pick_up, attribute))
                upper = np.nanmax(getattr(pick_up, attribute))
                continue

            this_lower = np.nanmin(getattr(pick_up, attribute))
            if this_lower < lower:
                lower = this_lower
            this_upper = np.nanmax(getattr(pick_up, attribute))
            if this_upper > upper:
                upper = this_upper

        if lower is None:
            lower = 0.
        if upper is None:
            upper = 1.
        amplitude = abs(upper - lower)
        return lower - .1 * amplitude, upper + .1 * amplitude

    def _e_rf_and_i_mp_plots(self,
                             subplot_kw: dict[str, str],
                             **fig_kw
                             ) -> tuple[Figure, Axes, Axes]:
        """Set a figure with two Axes for electric field and MP current."""
        fig, (field_ax, current_ax) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            subplot_kw=subplot_kw,
            **fig_kw,
        )

        field_ax.set_ylabel(r'Field probe $[V]$')
        current_ax.set_ylabel(r'MP current $[\mu A]$')

        for ax in (field_ax, current_ax):
            ax.grid(True)
        return fig, field_ax, current_ax
