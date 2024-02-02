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
from multipac_testbench.src.theoretical.somersalo import \
    plot_somersalo_analytical


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

    def somersalo(self, **fig_kw) -> tuple[Figure, Axes, Axes]:
        """Create a Somersalo plot, with theoretical results and measured."""
        xlim = (0., 3.5)
        fig, ax1, ax2 = self._somersalo_base_plot(xlim=xlim, **fig_kw)

        log_power = np.linspace(xlim[0], xlim[1], 51)
        self._add_somersalo_analytical(log_power, ax1, ax2)
        self._add_somersalo_measured(ax1, ax2)
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
            ylim=(7.4, 9.2),
        )
        ax1.grid(True)
        ax2 = plt.twinx(ax1)
        ax2.set_ylabel(
            r"$\log_{10}((f_\mathrm{GHz} d_\mathrm{mm})^4 \dot Z_\Omega^2)$")
        ax2.set_ylim(9.1, 11)
        return fig, ax1, ax2

    def _add_somersalo_analytical(self,
                                  log_power: np.ndarray,
                                  ax1: Axes,
                                  ax2: Axes) -> None:
        """Cmpute and plot all Somersalo."""
        one_orders = (1, 2)
        two_orders = (2, 3)
        for points, orders, ax in zip(['one', 'two'],
                                      [one_orders, two_orders],
                                      [ax1, ax2]):
            plot_somersalo_analytical(points, log_power, orders, ax)
        ax1.grid(True)

    def _add_somersalo_measured(*args: Axes) -> None:
        """Represent the theoretical plots from Somersalo."""
        print("Somersalo measured not implemented.")
