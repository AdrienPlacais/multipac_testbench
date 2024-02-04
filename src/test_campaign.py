#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store data from several :class:`.MultipactorTest`."""
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import numpy as np

from multipac_testbench.src.multipactor_test import MultipactorTest
from multipac_testbench.src.theoretical.somersalo import (
    measured_to_somersalo_coordinates,
    plot_somersalo_analytical
)


class TestCampaign(list):
    """Hold several multipactor tests together."""

    def __init__(self, multipactor_tests: list[MultipactorTest]) -> None:
        """Create the object from the list of :class:`.MultipactorTest`."""
        super().__init__(multipactor_tests)

    @classmethod
    def from_filepaths(cls,
                       filepaths: Sequence[Path],
                       frequencies: Sequence[float],
                       swrs: Sequence[float],
                       config: dict,
                       sep: str = ';') -> Self:
        """Instantiate the :class:`.MultipactorTest` and :class:`TestCampaign`.

        Parameters
        ----------
        filepaths : Sequence[Path]
           Filepaths to the LabViewer files.
        frequencies : Sequence[float]
            Frequencies matching the filepaths.
        swrs : Sequence[float]
            SWRs matching the filepaths.
        config : dict
            Configuration of the test bench.
        sep : str
            Delimiter between the columns.

        Returns
        -------
        TestCampaign
            List of :class:`.MultipactorTest`.

        """
        args = zip(filepaths, frequencies, swrs, strict=True)

        multipactor_tests = [
            MultipactorTest(filepath, config, freq_mhz, swr, sep=sep)
            for filepath, freq_mhz, swr in args
        ]
        return cls(multipactor_tests)

    def somersalo(self,
                  multipactor_measured_at: str,
                  **fig_kw) -> tuple[Figure, Axes, Axes]:
        """Create a Somersalo plot, with theoretical results and measured."""
        xlim = (-1.5, 3.5)
        # xlim = (0, 3.5)  # Somersalo original
        fig, ax1, ax2 = self._somersalo_base_plot(xlim=xlim, **fig_kw)

        log_power = np.linspace(xlim[0], xlim[1], 51)
        self._add_somersalo_analytical(log_power, ax1, ax2)
        self._add_somersalo_measured(multipactor_measured_at, ax1, ax2)
        ax1.grid(True)
        return fig, ax1, ax2

    def _somersalo_base_plot(self,
                             xlim: tuple[float, float],
                             **fig_kw) -> tuple[Figure, Axes, Axes]:
        fig = plt.figure(**fig_kw)
        ax1 = fig.add_subplot(
            111,
            xlabel=r"$\log_{10}(P~\mathrm{[kW]})$",
            ylabel=r"$\log_{10}((f_\mathrm{GHz} d_\mathrm{mm})^4$"
            + r"$\dot Z_\Omega)$",
            xlim=xlim,
            ylim=(2.2, 9.2),
            # ylim=(7.4, 9.2),  # Somersalo original
        )
        ax1.grid(True)
        ax2 = plt.twinx(ax1)
        ax2.set_ylabel(
            r"$\log_{10}((f_\mathrm{GHz} d_\mathrm{mm})^4 \dot Z_\Omega^2)$")
        ax2.set_ylim(3.8, 11)
        # ax2.set_ylim(9.1, 11)  # Somersalo original
        return fig, ax1, ax2

    def _add_somersalo_analytical(self,
                                  log_power: np.ndarray,
                                  ax1: Axes,
                                  ax2: Axes) -> None:
        """Cmpute and plot all Somersalo.

        .. todo::
            For some reason, two point is plotted on the one point ax instead
            of the two point...

        """
        orders_one = (1, 2, 3, 4, 5, 6, 7)
        orders_two = (1, )
        plot_somersalo_analytical('one', log_power, orders_one, ax1)
        # plot_somersalo_analytical('two', log_power, orders_two, ax2, ls='--')

    def _add_somersalo_measured(self,
                                multipactor_measured_at: str,
                                ax1: Axes,
                                ax2: Axes) -> None:
        """Represent the theoretical plots from Somersalo.

        .. todo::
            Determine what this function should precisely return. As for now,
            it returns last lower and upper power barriers. Alternatives would
            be to plot every power that led to multipactind during last power
            cycle, or every power that led to multipacting during whole test.

        """
        for mp_test in self:
            somersalo_data = mp_test.data_for_somersalo(
                multipactor_measured_at)
            one_point, two_point = measured_to_somersalo_coordinates(
                **somersalo_data)
            ax1.scatter(one_point[:, 0], one_point[:, 1],
                        marker='o', label=str(mp_test))
            ax2.scatter(two_point[:, 0], two_point[:, 1],
                        marker='*', label=str(mp_test))
        ax1.legend()
        ax2.legend()
