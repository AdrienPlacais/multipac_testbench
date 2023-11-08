#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store and treat data from pick-ups."""
import os.path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.container import StemContainer
import matplotlib.animation as animation

from multipac_testbench.pick_up import PickUp
from multipac_testbench.file_configuration import FileConfiguration


class MultipactorTest:
    """Holds a mp test with several probes."""

    def __init__(self,
                 filepath: str,
                 file_config: FileConfiguration,
                 skiprows: int = 1,
                 delimiter: str | None = None) -> None:
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

        .. todo::
            Proper data treatment to accept default file.

        """
        self._filepath = filepath
        data = np.loadtxt(filepath, skiprows=skiprows, delimiter=delimiter)

        self.pick_ups = self._instantiate_pick_ups(data, file_config)
        self._data = data

    def __str__(self) -> str:
        """Print the voltage of the MP zone of every pick-up."""
        out = (pick_up.__str__() for pick_up in self.pick_ups)
        return '\n'.join(out)

    def _instantiate_pick_ups(self,
                              data: np.ndarray,
                              file_config: FileConfiguration) -> list[PickUp]:
        """Create the pick-ups."""
        pick_ups = []
        for i, pick_up_name in enumerate(file_config.names):
            position = file_config.positions[i]
            electric_field_idx = file_config.electric_field_idx[i]
            mp_current_idx = file_config.mp_current_idx[i]
            pick_ups.append(PickUp(pick_up_name,
                                   position,
                                   data[:, 0],
                                   data[:, electric_field_idx],
                                   data[:, mp_current_idx],
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
                      **fig_kw
                      ) -> None:
        """Plot the measured current and voltage @ every pick-up."""
        subplot_kw = {'xlabel': 'Measurement index'}
        fig, field_ax, current_ax = self._electric_field_and_current_plots(
            subplot_kw,
            **fig_kw)

        for pick_up in self.pick_ups:
            if pick_up.name in to_exclude:
                continue

            pick_up.plot_electric_field(field_ax, draw_mp_zones=False)
            pick_up.plot_mp_current(current_ax, draw_mp_zones=True)

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
        fig, field_ax, current_ax = self._electric_field_and_current_plots(
            subplot_kw,
            **fig_kw)

        to_ignore = to_exclude + to_ignore_for_limits
        x_lim = self._get_limits('position', to_ignore)
        y_lim1 = self._get_limits('electric_field_probe', to_ignore)
        y_lim2 = self._get_limits('mp_current_probe', to_ignore)

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

            heads_field = [pick_up.electric_field_probe[step_idx]
                           for pick_up in self.pick_ups
                           if pick_up.name not in to_exclude
                           ]
            field_line = field_ax.stem(locs, heads_field)
            heads_current = [pick_up.mp_current_probe[step_idx]
                             for pick_up in self.pick_ups
                             if pick_up.name not in to_exclude
                             ]
            current_line = current_ax.stem(locs, heads_current)
            return field_line, current_line

        frames = len(self.pick_ups[0].electric_field_probe)
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

    def _electric_field_and_current_plots(self,
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
