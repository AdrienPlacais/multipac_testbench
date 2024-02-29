#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to store data from several :class:`.MultipactorTest`.

.. todo::
    Implement the `plot_barriers_vs_swr` and `plot_barriers_vs_frequency`
    methods.

"""
from abc import ABCMeta
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import multipac_testbench.src.instruments as ins
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from multipac_testbench.src.instruments.power import ForwardPower
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint
from multipac_testbench.src.multipactor_band.campaign_multipactor_bands \
    import CampaignMultipactorBands
from multipac_testbench.src.multipactor_band.instrument_multipactor_bands \
    import InstrumentMultipactorBands
from multipac_testbench.src.multipactor_test import MultipactorTest
from multipac_testbench.src.theoretical.somersalo import (
    plot_somersalo_analytical, plot_somersalo_measured, somersalo_base_plot,
    somersalo_scaling_law)
from multipac_testbench.src.theoretical.susceptibility import \
    measured_to_susceptibility_coordinates
from multipac_testbench.src.util import helper, log_manager, plot


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
                       sep: str = ';',
                       **kwargs,
                       ) -> Self:
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

        logfile = Path(filepaths[0].parent / "multipac_testbench.log")
        log_manager.set_up_logging(logfile_file=logfile)

        multipactor_tests = [
            MultipactorTest(filepath,
                            config,
                            freq_mhz,
                            swr,
                            info,
                            sep=sep,
                            verbose=i == 0,
                            **kwargs)
            for i, (filepath, freq_mhz, swr, info) in enumerate(args)
        ]
        return cls(multipactor_tests)

    def add_post_treater(self, *args, **kwargs) -> None:
        """Add post-treatment functions to instruments."""
        for test in self:
            test.add_post_treater(*args, **kwargs)

    def sweet_plot(
        self,
        *args,
        campaign_multipactor_bands: CampaignMultipactorBands | list[None] | None = None,
        png_folder: str | None = None,
        csv_folder: str | None = None,
        **kwargs
    ) -> list[Axes] | list[np.ndarray[Axes]]:
        """Recursively call :meth:`.MultipactorTest.sweet_plot`."""
        axes = []
        if campaign_multipactor_bands is None:
            campaign_multipactor_bands = [None for _ in self]
        zipper = zip(self, campaign_multipactor_bands, strict=True)
        for test, band in zipper:
            png_path = None
            if png_folder is not None:
                png_path = test.output_filepath(png_folder, ".png")

            csv_path = None
            if csv_folder is not None:
                csv_path = test.output_filepath(csv_folder, ".csv")

            axes.append(test.sweet_plot(*args,
                                        png_path=png_path,
                                        test_multipactor_bands=band,
                                        csv_path=csv_path,
                                        **kwargs))
        return axes

    def plot_thresholds(self,
                        instrument_id_plot: ABCMeta,
                        campaign_multipactor_bands: CampaignMultipactorBands,
                        *args,
                        png_folder: str | None = None,
                        csv_folder: str | None = None,
                        **kwargs
                        ) -> list[Axes] | list[np.ndarray[Axes]]:
        """Recursively call :meth:`.MultipactorTest.plot_thresholds`."""
        axes = []
        zipper = zip(self, campaign_multipactor_bands, strict=True)
        for test, multipactor_bands in zipper:
            png_path = None
            if png_folder is not None:
                png_path = test.output_filepath(png_folder, ".png")

            csv_path = None
            if csv_folder is not None:
                csv_path = test.output_filepath(csv_folder, ".csv")

            axes.append(test.plot_thresholds(
                instrument_id_plot,
                multipactor_bands,
                *args,
                png_path=png_path,
                csv_path=csv_path,
                **kwargs))
        return axes

    def at_last_threshold(
            self,
            instrument_id: ABCMeta,
            campaign_multipactor_bands: CampaignMultipactorBands,
            *args,
            **kwargs) -> pd.DataFrame:
        """Make a resume of data measured at last thresholds."""
        zipper = zip(self, campaign_multipactor_bands, strict=True)
        df_thresholds = [
            test.at_last_threshold(instrument_id, band, *args, **kwargs)
            for test, band in zipper]
        return pd.concat(df_thresholds)

    def detect_multipactor(
            self,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_class: ABCMeta,
            *args,
            power_is_growing_kw: dict[str, int | float] | None = None,
            measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (
            ),
            debug: bool = False,
            **kwargs,
    ) -> CampaignMultipactorBands:
        """Create the :class:`.InstrumentMultipactorBands` objects.

        Parameters
        ----------
        multipac_detector : Callable[[np.ndarray], np.ndarray[np.bool_]]
            Function that takes in the ``data`` of an :class:`ins.Instrument` and
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
        debug : bool, optional
            To plot the data used for multipactor detection, where power grows,
            where multipactor is detected. The default is False.

        Returns
        -------
        nested_instrument_multipactor_bands : list[list[InstrumentMultipactorBands]]
            :class:`.InstrumentMultipactorBands` objects holding when multipactor
            happens. They are sorted first by :class:`.MultipactorTest` (outer
            level), then per :class:`ins.Instrument` of class ``instrument_class``
            (inner level).

        """
        tests_multipactor_bands = [
            test.detect_multipactor(
                multipac_detector=multipac_detector,
                instrument_class=instrument_class,
                *args,
                power_is_growing_kw=power_is_growing_kw,
                measurement_points_to_exclude=measurement_points_to_exclude,
                debug=debug,
                **kwargs)
            for test in self]
        campaign_multipactor_bands = CampaignMultipactorBands(
            tests_multipactor_bands)
        return campaign_multipactor_bands

    def somersalo_chart(self,
                        multipactor_bands: CampaignMultipactorBands,
                        orders_one_point: tuple[int, ...] = (
                            1, 2, 3, 4, 5, 6, 7),
                        orders_two_point: tuple[int, ...] = (1, ),
                        **fig_kw) -> tuple[Figure, Axes, Axes]:
        """Create a Somersalo plot, with theoretical results and measured.

        .. todo::
            For some reason, two point is plotted on the one point ax instead
            of the two point...

        Parameters
        ----------
        instrument_multipactor_bands : Sequence[InstrumentMultipactorBands]
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

        self._add_somersalo_measured(ax1, ax2, multipactor_bands)
        ax1.grid(True)
        return fig, ax1, ax2

    def _add_somersalo_measured(self,
                                ax1: Axes, ax2: Axes,
                                multipactor_bands: CampaignMultipactorBands,
                                **plot_kw
                                ) -> None:
        """Put the measured multipacting limits on Somersalo plot.

        .. todo::
            Determine what this function should precisely plot. As for now,
            it plots last lower and upper power barriers. Alternatives would
            be to plot every power that led to multipacting during last power
            cycle, or every power that led to multipacting during whole test.

        """
        zipper = zip(self, multipactor_bands, strict=True)
        for test, bands in zipper:
            somersalo_data = test.data_for_somersalo(bands)
            plot_somersalo_measured(mp_test_name=str(test),
                                    somersalo_data=somersalo_data,
                                    ax1=ax1, ax2=ax2,
                                    **plot_kw)

    def check_somersalo_scaling_law(
        self,
        multipactor_bands: CampaignMultipactorBands | Sequence[InstrumentMultipactorBands],
        show_fit: bool = True,
        remove_last_point_for_fit: bool = False,
        use_theoretical_r: bool = False,
        full_output: bool = True,
        png_path: Path | None = None,
        **fig_kw,
    ) -> Axes:
        r"""Represent evolution of forward power threshold with :math:`R`.

        Somersalo et al. [1]_ link the mixed wave (:math:`MW`) forward power
        with the traveling wave (:math:`TW`) forward power through reflection
        coefficient :math:`R`.

        .. math::

            P_\mathrm{MW} \sim \frac{1}{(1 + R)^2}P_\mathrm{TW}

        .. note::
            Multipactor is detected on a global level, i.e. multipactor
            threshold is reached when multipactor is detected anywhere in the
            system.

        .. todo::
            Clean this anti-patternic method.

        .. [1] Erkki Somersalo, Pasi Yla-Oijala, Dieter Proch et Jukka \
               Sarvas. «Computational methods for analyzing electron \
               multipacting in RF structures». In : Part. Accel. 59 (1998), p.\
               107-141. url : http://cds.cern.ch/record/1120302/files/p107.pdf.

        Parameters
        ----------
        campaign_multipactor_bands : CampaignMultipactorBands | Sequence[InstrumentMultipactorBands]
            Object holding the information on where multipactor happens. If a
            :class:`.CampaignMultipactorBands` object is given, take every
            :class:`.TestMultipactorBands` in it and merge it. You can also
            provide one :class:`.InstrumentMultipactorBands` per multipactor
            test.
        show_fit : bool, optional
            To perform a fit and plot it. The default is True.
        png_path : Path | None, optional
            If provided, the resulting figure will be saved at this location.
            The default is None.
        remove_last_point_for_fit : bool, optional
            A dirty patch to remove the last point from the fit. Used in a
            study were I wanted to plot this point but exclude it from the fit.
            The default is False.
        use_theoretical_r : bool, optional
            Another patch to allow fitting and plotting using the theoretical
            reflection coefficient instead of the one calculated from
            :math:`P_f` and :math:`P_r`. The default is False.
        fig_kw :
            Other keyword arguments passed to Figure.

        Returns
        -------
        Axes

        """
        frequencies = set([test.freq_mhz for test in self])
        if len(frequencies) != 1:
            raise NotImplementedError("Plot over several freqs to implement")

        zipper = zip(self, multipactor_bands, strict=True)
        data_for_somersalo = [
            test.data_for_somersalo_scaling_law(band, use_theoretical_r)
            for (test, band) in zipper]
        df_for_somersalo = pd.concat(data_for_somersalo).filter(like='Lower')
        x_col = df_for_somersalo.filter(like='ReflectionCoefficient').columns
        y_col = df_for_somersalo.filter(like='ForwardPower').columns

        axe = df_for_somersalo.plot(
            x=x_col.values[0],
            y=y_col,
            xlabel=ins.ReflectionCoefficient.ylabel(),
            ylabel=ForwardPower.ylabel(),
            grid=True,
            ms=15,
            marker='+',
            **fig_kw)

        if show_fit:
            raise NotImplementedError
            R = np.linspace(0, 1, 101)
            r_fit = df_for_somersalo[df_for_somersalo.columns[0]]
            p_fit = df_for_somersalo[df_for_somersalo.columns[1]]
            if remove_last_point_for_fit:
                r_fit = r_fit[:-1]
                p_fit = p_fit[:-1]

            result = curve_fit(f=somersalo_scaling_law,
                               xdata=r_fit,
                               ydata=p_fit,
                               full_output=full_output)
            popt = result[0]
            tmp_str = r'$P_{TW}$ '
            if full_output:
                r_squared = helper.r_squared(result[2]['fvec'], p_fit)
                tmp_str = f'Fit ({tmp_str} = {popt[0]:3.1f}W, $r^2$ = {r_squared:3.3f})'
            else:
                tmp_str = f'Fit ({tmp_str} = {popt[0]:3.1f}W)'
            df_fitted = pd.DataFrame(
                {'$R$': R,
                 tmp_str: somersalo_scaling_law(R, *popt)
                 })
            df_fitted.plot(ax=axe,
                           x=0,
                           y=1,
                           grid=True)
        # if png_path is not None:
            # fig.savefig(png_path)

        return axe

    def voltage_thresholds(
            self,
            campaign_multipactor_bands: CampaignMultipactorBands,
            measurement_points_to_exclude: Sequence[str] = (),
            png_path: Path | None = None,
            png_kwargs: dict | None = None,
            csv_path: Path | None = None,
            csv_kwargs: dict | None = None,
            **kwargs,
    ) -> tuple[Axes, pd.DataFrame]:
        """Plot the lower and upper thresholds as voltage.

        Parameters
        ----------
        campaign_multipactor_bands : CampaignMultipactorBands
            campaign_multipactor_bands
        measurement_points_to_exclude : Sequence[str]
            measurement_points_to_exclude
        png_path : Path | None
            png_path
        png_kwargs : dict | None
            png_kwargs
        csv_path : Path | None
            csv_path
        csv_kwargs : dict | None
            csv_kwargs
        kwargs :
            kwargs

        Returns
        -------
        tuple[Axes, pd.DataFrame]

        """
        frequencies = set([test.freq_mhz for test in self])
        if len(frequencies) != 1:
            raise NotImplementedError("Plot over several freqs to implement")

        voltages = self.at_last_threshold(
            ins.FieldProbe,
            campaign_multipactor_bands,
            measurement_points_to_exclude=measurement_points_to_exclude)

        axes = voltages.filter(like='Lower').plot(grid=True,
                                                  ylabel="Thresholds $V$ [V]",
                                                  marker='o',
                                                  ms=10,
                                                  **kwargs,
                                                  )
        axes.set_prop_cycle(None)
        axes = voltages.filter(like='Upper').plot(grid=True,
                                                  ax=axes,
                                                  ylabel="Thresholds $V$ [V]",
                                                  marker='^',
                                                  ms=10,
                                                  **kwargs,
                                                  )
        if png_path is not None:
            if png_kwargs is None:
                png_kwargs = {}
            plot.save_figure(axes, png_path, **png_kwargs)
        if csv_path is not None:
            if csv_kwargs is None:
                csv_kwargs = {}
            plot.save_dataframe(df_to_plot, csv_path, **csv_kwargs)
        return axes, voltages

    # to redo
    def susceptibility_chart(self,
                             electric_field_at: str,
                             instrument_multipactor_bands: Sequence[InstrumentMultipactorBands],
                             fig_kw: dict | None = None,
                             ax_kw: dict | None = None) -> tuple[Figure, Axes]:
        """Create a susceptiblity chart."""
        fig, ax1 = self._susceptibility_base_plot(fig_kw, ax_kw)

        zipper = zip(self, instrument_multipactor_bands, strict=True)

        for mp_test, mp_bands in zipper:
            if len(mp_bands) > 1:
                raise NotImplementedError(f"{mp_bands = }, but only one pair "
                                          "field probe--mp band is allowed")
            susceptibility_data = mp_test.data_for_susceptibility(
                electric_field_at,
                instrument_multipactor_bands=mp_bands[0],
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

    def plot_instruments_vs_time(
        self,
        *args,
        seq_instrument_multipactor_bands: Sequence[Sequence[InstrumentMultipactorBands]
                                                   ] | Sequence[InstrumentMultipactorBands] | None = None,
        out_folder: str | None = None,
        iternum: int = 300,
        **kwargs
    ) -> None:
        """Call all :meth:`.MultipactorTest.plot_instruments_vs_time`.

        .. deprecated:: 1.5.0
            Use :meth:`TestCampaign.sweet_plot` instead.

        """
        if seq_instrument_multipactor_bands is None:
            seq_instrument_multipactor_bands = [None for _ in self]
        zipper = zip(self, seq_instrument_multipactor_bands, strict=True)
        for i, (test, instrument_multipactor_bands) in enumerate(zipper):
            png_path = None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
            _ = test.plot_instruments_vs_time(
                *args,
                instrument_multipactor_bands=instrument_multipactor_bands,
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
        """Call :meth:`.MultipactorTest.plot_instruments_y_vs_instrument_x`.

        .. deprecated:: 1.5.0
            Use :meth:`TestCampaign.sweet_plot` instead.

        """
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

    def plot_data_at_multipactor_thresholds(
            self,
            *args,
            seq_instrument_multipactor_bands: Sequence[Sequence[InstrumentMultipactorBands]] | Sequence[InstrumentMultipactorBands],
            out_folder: str | None = None,
            iternum: int = 350,
            **kwargs) -> None:
        """Call :meth:`.MultipactorTest.plot_data_at_multipactor_thresholds`.

        .. deprecated:: 1.5.0
            Use :meth:`TestCampaign.plot_thresholds` instead.

        Parameters
        ----------
        args :
            Arguments passed to
            :meth:`.MultipactorTest.plot_data_at_multipactor_thresholds`.
        seq_instrument_multipactor_bands : Sequence[Sequence[InstrumentMultipactorBands]] | \
                Sequence[InstrumentMultipactorBands]
            :class:`.InstrumentMultipactorBands` or lists of :class:`.InstrumentMultipactorBands`
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
        zipper = zip(self, seq_instrument_multipactor_bands, strict=True)
        for i, (test, instrument_multipactor_bands) in enumerate(zipper):
            png_path, csv_path = None, None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
                csv_path = test.output_filepath(out_folder, ".csv")
            _ = test.plot_data_at_multipactor_thresholds(
                *args,
                instrument_multipactor_bands=instrument_multipactor_bands,
                num=iternum + i,
                png_path=png_path,
                csv_path=csv_path,
                **kwargs
            )
        return
