#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store data from several :class:`.MultipactorTest`."""
from abc import ABCMeta
from collections.abc import Callable, Sequence
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
from multipac_testbench.src.theoretical.susceptibility import \
    measured_to_susceptibility_coordinates


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

    def add_post_treater(self, *args, **kwargs) -> None:
        """Add post-treatment functions to instruments."""
        for test in self:
            test.add_post_treater(*args, **kwargs)

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
        for test in self:
            test.detect_multipactor(multipac_detector,
                                    instrument_class,
                                    power_is_growing_kw)

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

    def susceptibility_plot(self,
                            multipactor_measured_at: str,
                            electric_field_at: str,
                            fig_kw: dict | None = None,
                            ax_kw: dict | None = None) -> tuple[Figure, Axes]:
        """Create a scusceptiblity chart."""
        fig, ax1 = self._susceptibility_base_plot(fig_kw, ax_kw)

        for mp_test in self:
            susceptibility_data = mp_test.data_for_susceptibility(
                multipactor_measured_at,
                electric_field_at)
            points = measured_to_susceptibility_coordinates(
                **susceptibility_data)
            ax1.scatter(points[:, 0], points[:, 1], label=str(mp_test))
        ax1.legend()
        return fig, ax1

    def _susceptibility_base_plot(self,
                                  fig_kw: dict | None = None,
                                  ax_kw: dict | None = None,
                                  ) -> tuple[Figure, Axes]:
        """Create the base figure."""
        if fig_kw is None:
            fig_kw = {}
        fig = plt.figure(**fig_kw)

        if ax_kw is None:
            ax_kw = {}
        ax1 = fig.add_subplot(
            111,
            xlabel=r"$f \times d~[\mathrm{MHz~cm}]$",
            ylabel=r"Threshold $V~[\mathrm{V}]$",
            **ax_kw,
        )
        ax1.set_xscale('log', base=10)
        ax1.set_yscale('log', base=10)
        ax1.grid(True)
        return fig, ax1