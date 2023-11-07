#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store and treat data from pick-ups."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.container import StemContainer
import matplotlib.animation as animation

from pick_up import PickUp


class MultipactorTest:
    """Holds a mp test with several probes."""

    def __init__(self, filepath: str) -> None:
        """Load the file and create the pick-ups."""
        data = np.loadtxt(filepath, skiprows=1, delimiter=';')
        self.pick_ups = self._instantiate_pick_ups(data)
        self._data = data

    def _instantiate_pick_ups(self, data: np.ndarray) -> list[PickUp]:
        """
        Create the pick-ups.

        .. todo::
            Avoid these hard-coded names, positions, indexes.

        """
        pick_up_names = ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7')
        positions = {'E1': 0.,
                     'E2': 1.,
                     'E3': 2.,
                     'E4': 3.,
                     'E5': 4.,
                     'E6': 5.,
                     'E7': 6.,
                     }
        pick_ups = []
        for i, pick_up_name in enumerate(pick_up_names):
            pick_ups.append(PickUp(pick_up_name,
                                   positions[pick_up_name],
                                   data[:, 0],
                                   data[:, i + 4],
                                   data[:, i + 11],
                                   n_mp_zones=0)
                            )
        return pick_ups

    def plot_pick_ups(self, to_exclude: tuple[str, ...] | None = None
                      ) -> None:
        """Plot the measured current and voltage @ every pick-up."""
        fig = plt.figure(1)
        field_ax = fig.add_subplot(2, 1, 1)
        current_ax = fig.add_subplot(2, 1, 2, sharex=field_ax)

        field_ax.set_ylabel('Field probe (V)')
        current_ax.set_xlabel('Measurement index')
        current_ax.set_ylabel('MP current (uA)')

        for pick_up in self.pick_ups:
            if to_exclude is not None and pick_up.name in to_exclude:
                continue

            pick_up.plot_electric_field(field_ax)
            pick_up.plot_mp_current(current_ax)

        field_ax.grid(True)
        current_ax.grid(True)
        field_ax.legend()

    def animate_pick_ups(self, to_exclude: tuple[str, ...] | None = None
                         ) -> None:
        """Plot what pick-up measure with time.

        .. todo::
            Name of pick-ups in x axis.

        """
        fig = plt.figure(2)
        field_ax = fig.add_subplot(2, 1, 1)
        current_ax = fig.add_subplot(2, 1, 2, sharex=field_ax)

        field_ax.set_ylabel('Field probe (V)')
        current_ax.set_xlabel('Probe position (m)')

        current_ax.set_ylabel('MP current (uA)')

        field_ax.grid(True)
        current_ax.grid(True)

        def _plot_pick_ups_single_time_step(
                step_idx: int
                ) -> tuple[StemContainer, StemContainer]:
            """Plot as stem the current and voltage of pick ups at position.

            Parameters
            ----------
            field_ax : Axes
                field_ax
            current_ax : Axes
                current_ax
            step_idx : int
                step_idx
            to_exclude : tuple[str, ...] | None
                to_exclude

            """
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

        file = "/home/placais/Documents/test.gif"
        writergif = animation.PillowWriter(fps=50)
        ani.save(file, writer=writergif)


