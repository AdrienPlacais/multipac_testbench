#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store and treat data from pick-ups.

.. todo::
    Allow to trim data (remove noisy useless data at end of exp)

"""
from abc import ABCMeta
from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes._axes import Axes
from matplotlib.container import StemContainer
from matplotlib.figure import Figure

from multipac_testbench.src.instruments.factory import InstrumentFactory
from multipac_testbench.src.pick_up import PickUp


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
            instruments_to_plot: tuple[ABCMeta, ...],
            pick_up_to_exclude: tuple[str, ...] = (),
            png_path: Path | None = None,
            raw: bool = False,
            multipactor_plots: dict[ABCMeta, ABCMeta] | None = None,
            **fig_kw,
    ) -> tuple[Figure, Axes]:
        """Plot the different signals at the different pick-ups.

        Parameters
        ----------
        instruments_to_plot : tuple[ABCMeta, ...]
            Subclass of the :class:`.Instrument` to plot.
        pick_up_to_exclude : tuple[str, ...], optional
            Name of the pick-ups that should not be plotted. The default is an
            empty tuple.
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

        Returns
        -------
        fig : Figure
            The created figure.
        axes : Axes
            The created axes.

        """
        fig, axes = self._create_fig(instruments_to_plot, **fig_kw)

        for pick_up in self.pick_ups:
            if pick_up.name in pick_up_to_exclude:
                continue

            pick_up.plot_instruments(axes, instruments_to_plot, raw=raw)

            if multipactor_plots is not None:
                self.add_multipacting_zones(pick_up, axes, multipactor_plots)

        for axe in axes.values():
            axe.legend()

        if png_path is not None:
            fig.savefig(png_path)

        return fig, axes

    def add_multipacting_zones(
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
            Keys are the :class:`Instrument` subclass for which you want to see
            the multipactor zones. Values are the :class:`Instrument` subclass
            that detect the multipactor.

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
        nrows = len(instruments_to_plot)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=True,
            **fig_kw
        )

        # ensure that axes is an iterable
        if nrows == 1:
            axes = [axes, ]

        axes = dict(zip(instruments_to_plot, axes))

        axe = None
        for instrument_class, axe in axes.items():
            axe.grid(True)
            axe.set_ylabel(instrument_class.ylabel())
        assert isinstance(axe, Axes)
        if axe is not None:
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

        .. todo::
            not sure if ``keep_one_frame_over`` work as expected

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

        .. todo::
            Optimize. Surely I do not need to redraw everything at every
            iteration.

        .. todo::
            Clarify. This not very clean nor Pythonic

        """
        # subplot_kw = {'xlabel': r'Probe position [m]'}
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
                if pick_up.name not in pick_ups_to_exclude]

        def _plot_pick_ups_single_time_step(
                step_idx: int
        ) -> Sequence[StemContainer]:
            """Plot as stem the instrument signals.

            Parameters
            ----------
            step_idx : int
                Current measurement point.

            """
            if step_idx % keep_one_frame_over != 0:
                pass

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
