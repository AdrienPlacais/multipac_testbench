#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store data from several :class:`.MultipactorTest`.

.. todo::
    Implement the `plot_barriers_vs_swr` and `plot_barriers_vs_frequency`
    methods.

"""
import warnings
from abc import ABCMeta
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint
from multipac_testbench.src.multipactor_band.multipactor_bands import \
    MultipactorBands
from multipac_testbench.src.multipactor_test import MultipactorTest
from multipac_testbench.src.theoretical.somersalo import (
    plot_somersalo_analytical, plot_somersalo_measured, somersalo_base_plot)
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
                       info: Sequence[str] = (),
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
        info : Sequence[str]
            Other information string to identify each multipactor test.
        sep : str
            Delimiter between the columns.

        Returns
        -------
        TestCampaign
            List of :class:`.MultipactorTest`.

        """
        if len(info) == 0:
            info = ['' for _ in filepaths]
        args = zip(filepaths, frequencies, swrs, info, strict=True)

        multipactor_tests = [
            MultipactorTest(filepath, config, freq_mhz, swr, info, sep=sep)
            for filepath, freq_mhz, swr, info in args
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
            *args,
            power_is_growing_kw: dict[str, int | float] | None = None,
            measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (
            ),
            **kwargs,
    ) -> list[list[MultipactorBands]]:
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
        measurement_points_to_exclude : Sequence[IMeasurementPoint | str] = ()
            :class:`.IMeasurementPoint` where you do not want to know if there
            is multipacting.

        Returns
        -------
        nested_multipactor_bands : list[list[MultipactorBands]]
            :class:`.MultipactorBands` objects holding when multipactor
            happens. They are sorted first by :class:`.MultipactorTest` (outer
            level), then per :class:`.Instrument` of class ``instrument_class``
            (inner level).

        """
        nested_multipactor_bands = [
            test.detect_multipactor(
                multipac_detector=multipac_detector,
                instrument_class=instrument_class,
                *args,
                power_is_growing_kw=power_is_growing_kw,
                measurement_points_to_exclude=measurement_points_to_exclude,
                **kwargs)
            for test in self]
        return nested_multipactor_bands

    def somersalo_chart(self,
                        multipactor_measured_at: str | None = None,
                        multipactor_bands: Sequence[MultipactorBands] | None = None,
                        orders_one_point: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
                        orders_two_point: tuple[int, ...] = (1, ),
                        **fig_kw) -> tuple[Figure, Axes, Axes]:
        """Create a Somersalo plot, with theoretical results and measured.

        .. todo::
            For some reason, two point is plotted on the one point ax instead
            of the two point...

        Parameters
        ----------
        multipactor_measured_at : str
            Name of the :class:`.IMeasurementPoint` where the multipactor is
            detected. It must have a :class:`.MultipactorBands` attribute,
            which is set by the :meth:`TestCampaign.detect_multipactor` method.
        multipactor_bands : Sequence[MultipactorBands]
            An object holding the multipactor information for every
            :class:`.MultipactorTest` in ``self``.
        orders_one_point : tuple[int, ...], optional
            The multipactor orders to plot for one point multipactor. The
            default is orders 1 to 8, as in Somersalo's plot.
        orders_two_point : tuple[int, ...]
            The multipactor orders to plot for two point multipactor. The
            default is order 1 only, as in Somersalo's plot.
        fig_kw :
            Other keyword arguments passed to the Figure constructor.

        Returns
        -------
        Figure :
            Holds the plotted figure.
        Axes :
            Left axis (one-point multipactor).
        Axes :
            Right axis (two-point multipactor).

        """
        log_power = np.linspace(-1.5, 3.5, 2)
        xlim = (log_power[0], log_power[-1])
        ylim_one_point = (2.2, 9.2)
        ylim_two_point = (3.8, 11.)

        fig, ax1, ax2 = somersalo_base_plot(xlim=xlim,
                                            ylim_one_point=ylim_one_point,
                                            ylim_two_point=ylim_two_point,
                                            **fig_kw)
        one_point_kw = {'points': 'one',
                        'orders': orders_one_point,
                        'ax': ax1,
                        'ls': '-'}
        two_point_kw = {'points': 'two',
                        'orders': orders_two_point,
                        'ax': ax2,
                        'ls': '--'}
        for kwargs in (one_point_kw, two_point_kw):
            plot_somersalo_analytical(log_power=log_power, **kwargs)

        self._add_somersalo_measured(
            ax1, ax2,
            multipactor_measured_at=multipactor_measured_at,
            multipactor_bands=multipactor_bands,
        )

        ax1.grid(True)
        return fig, ax1, ax2

    def _add_somersalo_measured(self,
                                ax1: Axes, ax2: Axes,
                                multipactor_measured_at: str | None = None,
                                multipactor_bands: Sequence[MultipactorBands] | None = None,
                                **plot_kw
                                ) -> None:
        """Put the measured multipacting limits on Somersalo plot.

        .. todo::
            Determine what this function should precisely plot. As for now,
            it plots last lower and upper power barriers. Alternatives would
            be to plot every power that led to multipacting during last power
            cycle, or every power that led to multipacting during whole test.

        """
        if multipactor_bands is None:
            multipactor_bands = [None for _ in self]

        zipper = zip(self, multipactor_bands, strict=True)
        for mp_test, mp_bands in zipper:
            if len(mp_bands) > 1:
                raise NotImplementedError(f"{mp_bands = }, but only one pair "
                                          "power--mp band is allowed")
            somersalo_data = mp_test.data_for_somersalo(
                multipactor_measured_at,
                mp_bands[0])
            plot_somersalo_measured(mp_test_name=str(mp_test),
                                    somersalo_data=somersalo_data,
                                    ax1=ax1, ax2=ax2,
                                    **plot_kw)

    def susceptibility_chart(self,
                            electric_field_at: str,
                            multipactor_measured_at: str | None = None,
                            multipactor_bands: Sequence[MultipactorBands] | None = None,
                            fig_kw: dict | None = None,
                            ax_kw: dict | None = None) -> tuple[Figure, Axes]:
        """Create a susceptiblity chart."""
        fig, ax1 = self._susceptibility_base_plot(fig_kw, ax_kw)

        if multipactor_bands is None:
            multipactor_bands = [None for _ in self]
        zipper = zip(self, multipactor_bands, strict=True)

        for mp_test, mp_bands in zipper:
            if len(mp_bands) > 1:
                raise NotImplementedError(f"{mp_bands = }, but only one pair "
                                          "field probe--mp band is allowed")
            susceptibility_data = mp_test.data_for_susceptibility(
                electric_field_at,
                multipactor_measured_at=multipactor_measured_at,
                multipactor_bands=mp_bands[0],
            )
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

    def animate_instruments_vs_position(
            self,
            *args,
            out_folder: str | None = None,
            iternum: int = 100,
            **kwargs
    ) -> list[animation.FuncAnimation]:
        """Call all :meth:`.MultipactorTest.animate_instruments_vs_position`"""
        animations = []
        for i, test in enumerate(self):
            gif_path = None
            if out_folder is not None:
                gif_path = test.output_filepath(out_folder, ".gif")
            animation = test.animate_instruments_vs_position(*args,
                                                             gif_path=gif_path,
                                                             num=iternum + i,
                                                             **kwargs)
            animations.append(animation)
        return animations

    def reconstruct_voltage_along_line(self, *args, **kwargs) -> None:
        """Call all :meth:`.MultipactorTest.reconstruct_voltage_along_line`."""
        for test in self:
            test.reconstruct_voltage_along_line(*args, **kwargs)

    def plot_instruments_vs_time(self,
                                 *args,
                                 out_folder: str | None = None,
                                 iternum: int = 300,
                                 **kwargs) -> None:
        """Call all :meth:`.MultipactorTest.plot_instruments_vs_time`."""
        for i, test in enumerate(self):
            png_path = None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
            _ = test.plot_instruments_vs_time(
                *args,
                num=iternum + i,
                png_path=png_path,
                **kwargs
            )
        return

    def scatter_instruments_data(self,
                                 *args,
                                 out_folder: str | None = None,
                                 iternum: int = 200,
                                 **kwargs) -> None:
        """Call all :meth:`.MultipactorTest.scatter_instruments_data`."""
        for i, test in enumerate(self):
            png_path = None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
            _ = test.scatter_instruments_data(
                *args,
                num=iternum + i,
                png_path=png_path,
                **kwargs
            )
        return

    def plot_instruments_y_vs_instrument_x(self,
                                           *args,
                                           out_folder: str | None = None,
                                           iternum: int = 250,
                                           **kwargs) -> None:
        """Call :meth:`.MultipactorTest.plot_instruments_y_vs_instrument_x`."""
        for i, test in enumerate(self):
            png_path = None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
            _ = test.plot_instruments_y_vs_instrument_x(
                *args,
                num=iternum + i,
                png_path=png_path,
                **kwargs
            )
        return

    def plot_barriers_vs_frequency(self) -> None:
        """Plot evolution of mp barriers with frequency."""
        raise NotImplementedError

    def plot_barriers_vs_swr(self) -> None:
        """Plot evolution of mp barriers with SWR."""
        raise NotImplementedError

    def plot_multipactor_limits(
            self,
            *args,
            all_multipactor_bands: list[MultipactorBands] | None = None,
            out_folder: str | None = None,
            iternum: int = 300,
            **kwargs) -> None:
        """Call :meth:`.MultipactorTest.plot_multipactor_limits`.

        .. deprecated:: 1.4.0
            Use TestCampaign.plot_data_at_multipactor_thresholds instead.

        """
        match (all_multipactor_bands):
            case MultipactorBands() as same_multipactor_bands_for_everyone:
                multipactor_bands = [same_multipactor_bands_for_everyone
                                     for _ in self]
            case list() as one_multipactor_bands_per_simulation:
                assert len(one_multipactor_bands_per_simulation) == len(self)
                multipactor_bands = one_multipactor_bands_per_simulation
            case None:
                warnings.warn("In the future, it will be mandatory to pass in "
                              "the desired MultipactorBands object.",
                              DeprecationWarning)
                multipactor_bands = [None for _ in self]
            case _:
                raise IOError(f"{all_multipactor_bands = } not handled.")

        zipper = zip(self, multipactor_bands, strict=True)
        for i, (test, mp_band) in enumerate(zipper):
            png_path = None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
            _ = test.plot_multipactor_limits(
                *args,
                multipactor_bands=mp_band,
                num=iternum + i,
                png_path=png_path,
                **kwargs
            )
        return

    def plot_data_at_multipactor_thresholds(
            self,
            *args,
            seq_multipactor_bands: Sequence[Sequence[MultipactorBands]] | Sequence[MultipactorBands],
            out_folder: str | None = None,
            iternum: int = 350,
            **kwargs) -> None:
        """Call :meth:`.MultipactorTest.plot_data_at_multipactor_thresholds`.

        Parameters
        ----------
        args :
            Arguments passed to
            :meth:`.MultipactorTest.plot_data_at_multipactor_thresholds`.
        seq_multipactor_bands : Sequence[Sequence[MultipactorBands]] | \
                Sequence[MultipactorBands]
            :class:`.MultipactorBands` or lists of :class:`.MultipactorBands`
            for every :class:`.MultipactorTest` in ``self``.
        out_folder : str | None, optional
            Where figures should be saved. The default is None, in which case
            figures are not saved.
        iternum : int, optional
            First figure number. Iterated for every figure. The default is
            ``350``.
        kwargs :
            Keyword arguments passed to
            :meth:`MultipactorTest.plot_data_at_multipactor_thresholds`.

        """
        zipper = zip(self, seq_multipactor_bands, strict=True)
        for i, (test, multipactor_bands) in enumerate(zipper):
            png_path = None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
            _ = test.plot_data_at_multipactor_thresholds(
                *args,
                multipactor_bands=multipactor_bands,
                num=iternum + i,
                png_path=png_path,
                **kwargs
            )
        return
