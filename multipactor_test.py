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
                                   n_mp_zones=0)
                            )
        return pick_ups

    def plot_pick_ups(self,
                      to_exclude: tuple[str, ...] = (),
                      png_path: str = '',
                      **kwargs
                      ) -> None:
        """Plot the measured current and voltage @ every pick-up."""
        fig, field_ax, current_ax = self._electric_field_and_current_plots(
            **kwargs)

        for pick_up in self.pick_ups:
            if pick_up.name in to_exclude:
                continue

            pick_up.plot_electric_field(field_ax)
            pick_up.plot_mp_current(current_ax)

        field_ax.grid(True)
        current_ax.grid(True)
        field_ax.legend()

        if png_path == '':
            png_path = os.path.splitext(self._filepath)[0] + '.png'
        fig.savefig(png_path)

    def animate_pick_ups(self,
                         to_exclude: tuple[str, ...] = (),
                         gif_path: str = '',
                         fps: int = 50,
                         keep_one_frame_over: int = 1,
                         **kwargs,
                         ) -> None:
        """Plot what pick-up measure with time.

        .. todo::
            Name of pick-ups in x axis.

        """
        fig, field_ax, current_ax = self._electric_field_and_current_plots(
            **kwargs)
        current_ax.set_xlabel(r'Probe position $[m]$')

        def _plot_pick_ups_single_time_step(
                step_idx: int
                ) -> tuple[StemContainer, StemContainer] | None:
            """Plot as stem the current and voltage of pick ups at position.

            Parameters
            ----------
            field_ax : Axes
                Axes holding electric field plot.
            current_ax : Axes
                Axes holding MP current plot.
            step_idx : int
                Current measurement point.
            to_exclude : tuple[str, ...] | None
                Name of the pick-ups to exclude from the plot.

            """
            if step_idx % keep_one_frame_over != 0:
                return
            field_ax.clear()
            current_ax.clear()

            field_ax.set_ylim([0., 0.25])
            current_ax.set_xlim([-1., 7.])
            current_ax.set_ylim([0., 90.])

            locs = [pick_up.position
                    for pick_up in self.pick_ups
                    if pick_up.name not in to_exclude
                    ]
            heads_current = [pick_up.mp_current_probe[step_idx]
                             for pick_up in self.pick_ups
                              if pick_up.name not in to_exclude
                             ]
            heads_field = [pick_up.electric_field_probe[step_idx]
                           for pick_up in self.pick_ups
                            if pick_up.name not in to_exclude
                           ]
            field_line = field_ax.stem(heads_field)
            current_line = current_ax.stem(heads_current)
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

    def _electric_field_and_current_plots(self, **kwargs
                                          ) -> tuple[Figure, Axes, Axes]:
        """Set a figure with two Axes for electric field and MP current."""
        fig = plt.figure(**kwargs)
        field_ax = fig.add_subplot(2, 1, 1)
        current_ax = fig.add_subplot(2, 1, 2, sharex=field_ax)

        field_ax.set_ylabel(r'Field probe $[V]$')
        current_ax.set_xlabel('Measurement index')
        current_ax.set_ylabel(r'MP current $[\mu A]$')

        field_ax.grid(True)
        current_ax.grid(True)
        return fig, field_ax, current_ax
