"""Define an object to store data from several :class:`.MultipactorTest`."""

import logging
from abc import ABCMeta
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import multipac_testbench.instruments as ins
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.multipactor_band.campaign_multipactor_bands import (
    CampaignMultipactorBands,
)
from multipac_testbench.multipactor_band.instrument_multipactor_bands import (
    InstrumentMultipactorBands,
)
from multipac_testbench.multipactor_test import MultipactorTest
from multipac_testbench.theoretical.somersalo import (
    fit_somersalo_scaling,
    plot_somersalo_analytical,
    plot_somersalo_measured,
    somersalo_base_plot,
)
from multipac_testbench.util import log_manager, plot
from multipac_testbench.util.types import MULTIPAC_DETECTOR_T


class TestCampaign(list):
    """Hold several multipactor tests together."""

    def __init__(self, multipactor_tests: list[MultipactorTest]) -> None:
        """Create the object from the list of :class:`.MultipactorTest`."""
        super().__init__(multipactor_tests)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[Path],
        frequencies: Sequence[float],
        swrs: Sequence[float],
        config: dict,
        info: Sequence[str] = (),
        sep: str = ";",
        **kwargs,
    ) -> Self:
        """Instantiate the :class:`.MultipactorTest` and :class:`TestCampaign`.

        Parameters
        ----------
        filepaths :
           Filepaths to the LabViewer files.
        frequencies :
            Frequencies matching the filepaths.
        swrs :
            SWRs matching the filepaths.
        config :
            Configuration of the test bench.
        info :
            Other information string to identify each multipactor test.
        sep :
            Delimiter between the columns.

        Returns
        -------
        test_campaign :
            List of :class:`.MultipactorTest`.

        """
        if len(info) == 0:
            info = ["" for _ in filepaths]
        args = zip(filepaths, frequencies, swrs, info, strict=True)

        logfile = Path(filepaths[0].parent / "multipac_testbench.log")
        log_manager.set_up_logging(logfile_file=logfile)

        multipactor_tests = [
            MultipactorTest(
                filepath,
                config,
                freq_mhz,
                swr,
                info,
                sep=sep,
                verbose=i == 0,
                **kwargs,
            )
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
        campaign_multipactor_bands: (
            CampaignMultipactorBands | list[None] | None
        ) = None,
        png_folder: str | None = None,
        csv_folder: str | None = None,
        all_on_same_plot: bool = False,
        **kwargs,
    ) -> (
        tuple[list[Axes], pd.DataFrame]
        | tuple[list[list[Axes]], list[pd.DataFrame]]
    ):
        """Recursively call :meth:`.MultipactorTest.sweet_plot`.

        Parameters
        ----------
        args :
            Arguments that are passed to :meth:`.MultipactorTest.sweet_plot`.
        campaign_multipactor_bands :
            Object holding the :class:`.TestMultipactorBands` corresponding to
            each :class:`.MultipactorTest` stored in ``self``. The default is
            None, in which case the multipactor zones are not drawn.
        png_folder :
            If provided, all the created figures will be saved there.
        csv_folder :
            If provided, all the created DataFrame will be saved there.
        all_on_same_plot :
            If all the data from all the :class:`.MultipactorTest` should be
            drawn on the same Axes.
        kwargs :
            Other keyword arguments passed to
            :meth:`.MultipactorTest.sweet_plot`.

        Returns
        -------
        axes :
            Holds plotted fig.
        data :
            Holds data used to create the plot.

        """
        all_axes = []
        all_df = []
        if campaign_multipactor_bands is None:
            campaign_multipactor_bands = [None for _ in self]
        zipper = zip(self, campaign_multipactor_bands, strict=True)

        if all_on_same_plot:
            return self._sweet_plot_same_plot(zipper, *args, **kwargs)

        for test, band in zipper:
            png_path = None
            if png_folder is not None:
                png_path = test.output_filepath(png_folder, ".png")

            csv_path = None
            if csv_folder is not None:
                csv_path = test.output_filepath(csv_folder, ".csv")

            axes, df_plot = test.sweet_plot(
                *args,
                png_path=png_path,
                test_multipactor_bands=band,
                csv_path=csv_path,
                **kwargs,
            )
            all_axes.append(axes)
            all_df.append(df_plot)
        return all_axes, all_df

    def _sweet_plot_same_plot(
        self,
        zipper: zip,
        *args,
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[list[Axes], pd.DataFrame]:
        """Plot the various signals on the same Axes."""
        if len(args) > 1:
            logging.warning(
                "I am not sure how the interaction of all_on_same_plot with "
                "several instruments plotted will go."
            )
        axes = None
        all_df = []
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (test, band) in enumerate(zipper):
            axes, df_plot = test.sweet_plot(
                *args,
                test_multipactor_bands=band,
                axes=axes,
                column_names=str(test),
                title=" ",
                test_color=colors[i],
                **kwargs,
            )
            all_df.append(df_plot)
        assert axes is not None
        df_to_plot = pd.concat(all_df, axis=1)

        if png_path is not None:
            if png_kwargs is None:
                png_kwargs = {}
            plot.save_figure(axes, png_path, **png_kwargs)
        if csv_path is not None:
            if csv_kwargs is None:
                csv_kwargs = {}
            plot.save_dataframe(df_to_plot, csv_path, **csv_kwargs)

        return axes, df_to_plot

    def plot_thresholds(
        self,
        instrument_id_plot: ABCMeta,
        campaign_multipactor_bands: CampaignMultipactorBands,
        *args,
        png_folder: str | None = None,
        csv_folder: str | None = None,
        **kwargs,
    ) -> tuple[list[list[Axes]], list[pd.DataFrame]]:
        """Recursively call :meth:`.MultipactorTest.plot_thresholds`."""
        all_axes = []
        all_df = []
        zipper = zip(self, campaign_multipactor_bands, strict=True)
        for test, multipactor_bands in zipper:
            png_path = None
            if png_folder is not None:
                png_path = test.output_filepath(png_folder, ".png")

            csv_path = None
            if csv_folder is not None:
                csv_path = test.output_filepath(csv_folder, ".csv")

            axes, df_plot = test.plot_thresholds(
                instrument_id_plot,
                multipactor_bands,
                *args,
                png_path=png_path,
                csv_path=csv_path,
                **kwargs,
            )
            all_axes.append(axes)
            all_df.append(df_plot)
        return all_axes, all_df

    def at_last_threshold(
        self,
        instrument_id: ABCMeta | Sequence[ABCMeta],
        campaign_multipactor_bands: CampaignMultipactorBands,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Make a resume of data measured at last thresholds."""
        zipper = zip(self, campaign_multipactor_bands, strict=True)
        df_thresholds = [
            test.at_last_threshold(instrument_id, band, *args, **kwargs)
            for test, band in zipper
        ]
        return pd.concat(df_thresholds)

    def detect_multipactor(
        self,
        multipac_detector: MULTIPAC_DETECTOR_T,
        instrument_class: ABCMeta,
        *args,
        power_growth_mask_kw: dict[str, int | float] | None = None,
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        debug: bool = False,
        **kwargs,
    ) -> CampaignMultipactorBands:
        """Create the :class:`.InstrumentMultipactorBands` objects.

        Parameters
        ----------
        multipac_detector :
            Function that takes in the ``data`` of an :class:`.Instrument` and
            returns an array, where True means multipactor and False no
            multipactor.
        instrument_class :
            Type of instrument on which ``multipac_detector`` should be
            applied.
        power_growth_mask_kw :
            Keyword arguments passed to the function that determines when power
            is increasing, when it is decreasing
            (:meth:`.ForwardPower.growth_mask`).
        measurement_points_to_exclude :
            :class:`.IMeasurementPoint` where you do not want to know if there
            is multipacting.
        debug :
            To plot the data used for multipactor detection, where power grows,
            where multipactor is detected.

        Returns
        -------
        nested_instrument_multipactor_bands :
            :class:`.InstrumentMultipactorBands` objects holding when multipactor
            happens. They are sorted first by :class:`.MultipactorTest` (outer
            level), then per :class:`.Instrument` of class ``instrument_class``
            (inner level).

        """
        tests_multipactor_bands = [
            test.detect_multipactor(
                multipac_detector=multipac_detector,
                instrument_class=instrument_class,
                *args,
                power_growth_mask_kw=power_growth_mask_kw,
                measurement_points_to_exclude=measurement_points_to_exclude,
                debug=debug,
                **kwargs,
            )
            for test in self
        ]
        campaign_multipactor_bands = CampaignMultipactorBands(
            tests_multipactor_bands
        )
        return campaign_multipactor_bands

    def somersalo_chart(
        self,
        multipactor_bands: CampaignMultipactorBands,
        orders_one_point: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
        orders_two_point: tuple[int, ...] = (1,),
        **fig_kw,
    ) -> tuple[Figure, Axes, Axes]:
        """Create a Somersalo plot, with theoretical results and measured.

        .. todo::
            For some reason, two point is plotted on the one point ax instead
            of the two point...

        Parameters
        ----------
        instrument_multipactor_bands :
            An object holding the multipactor information for every
            :class:`.MultipactorTest` in ``self``.
        orders_one_point :
            The multipactor orders to plot for one point multipactor. The
            default is orders 1 to 8, as in Somersalo's plot.
        orders_two_point :
            The multipactor orders to plot for two point multipactor. The
            default is order 1 only, as in Somersalo's plot.
        fig_kw :
            Other keyword arguments passed to the Figure constructor.

        Returns
        -------
        fig :
            Holds the plotted figure.
        ax1 :
            Left axis (one-point multipactor).
        ax2 :
            Right axis (two-point multipactor).

        """
        log_power = np.linspace(-1.5, 3.5, 2)
        xlim = (log_power[0], log_power[-1])
        ylim_one_point = (2.2, 9.2)
        ylim_two_point = (3.8, 11.0)

        fig, ax1, ax2 = somersalo_base_plot(
            xlim=xlim,
            ylim_one_point=ylim_one_point,
            ylim_two_point=ylim_two_point,
            **fig_kw,
        )
        one_point_kw = {
            "points": "one",
            "orders": orders_one_point,
            "ax": ax1,
            "ls": "-",
        }
        two_point_kw = {
            "points": "two",
            "orders": orders_two_point,
            "ax": ax2,
            "ls": "--",
        }
        for kwargs in (one_point_kw, two_point_kw):
            plot_somersalo_analytical(log_power=log_power, **kwargs)

        self._add_somersalo_measured(ax1, ax2, multipactor_bands)
        ax1.grid(True)
        return fig, ax1, ax2

    def _add_somersalo_measured(
        self,
        ax1: Axes,
        ax2: Axes,
        multipactor_bands: CampaignMultipactorBands,
        **plot_kw,
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
            plot_somersalo_measured(
                mp_test_name=str(test),
                somersalo_data=somersalo_data,
                ax1=ax1,
                ax2=ax2,
                **plot_kw,
            )

    def check_somersalo_scaling_law(
        self,
        multipactor_bands: (
            CampaignMultipactorBands | Sequence[InstrumentMultipactorBands]
        ),
        show_fit: bool = True,
        use_theoretical_r: bool = False,
        full_output: bool = True,
        drop_idx: list[int] | None = None,
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        **fig_kw,
    ) -> tuple[Axes, pd.DataFrame]:
        r"""Represent evolution of forward power threshold with :math:`R`.

        Somersalo et al. :cite:`Somersalo1998` link the mixed wave (:math:`MW`)
        forward power with the traveling wave (:math:`TW`) forward power
        through reflection coefficient :math:`R`.

        .. math::

            P_\mathrm{MW} \sim \frac{1}{(1 + R)^2}P_\mathrm{TW}

        .. note::
            Multipactor is detected on a global level, i.e. multipactor
            threshold is reached when multipactor is detected anywhere in the
            system. Also, we represent the thresholds that were measured during
            the last half-power cycle.

        .. todo::
            Columns in the output file are illogic
            xx | P_measured | R_measured | R_fit | P_fit

        Parameters
        ----------
        campaign_multipactor_bands :
            Object holding the information on where multipactor happens. If a
            :class:`.CampaignMultipactorBands` object is given, take every
            :class:`.TestMultipactorBands` in it and merge it. You can also
            provide one :class:`.InstrumentMultipactorBands` per multipactor
            test.
        show_fit :
            To perform a fit and plot it.
        use_theoretical_r :
            Another patch to allow fitting and plotting using the theoretical
            reflection coefficient instead of the one calculated from
            :math:`P_f` and :math:`P_r`.
        png_path :
            If provided, the resulting figure will be saved at this location.
        png_kwargs :
            Other keyword arguments passed to the :func:`.save_figure`
            function.
        csv_path :
            If provided, the data to produce the figure will be saved in this
            location.
        csv_kwargs :
            Other keyword arguments passed to the :func:`.save_dataframe`
            function.
        fig_kw :
            Other keyword arguments passed to Figure.

        Returns
        -------
        axes :
            Holds the plot.
        data :
            Holds the data that was plotted.

        """
        frequencies = {test.freq_mhz for test in self}
        if len(frequencies) != 1:
            raise NotImplementedError("Plot over several freqs to implement")

        zipper = zip(self, multipactor_bands, strict=True)
        data_for_somersalo = [
            test.data_for_somersalo_scaling_law(band, use_theoretical_r)
            for (test, band) in zipper
        ]
        df_somersalo = pd.concat(data_for_somersalo).filter(like="Lower")

        x_col = df_somersalo.filter(like="ReflectionCoefficient").columns
        y_col = df_somersalo.filter(like="ForwardPower").columns
        axes = df_somersalo.plot(
            x=x_col.values[0],
            y=y_col,
            xlabel=ins.ReflectionCoefficient.ylabel(),
            ylabel=ins.ForwardPower.ylabel(),
            grid=True,
            ms=15,
            marker="+",
            **fig_kw,
        )

        if drop_idx is not None:
            df_somersalo.drop(df_somersalo.index[drop_idx], inplace=True)

        if show_fit:
            df_fit = fit_somersalo_scaling(
                df_somersalo, full_output=full_output, plot=True, axes=axes
            )
            df_somersalo = pd.concat([df_somersalo, df_fit], axis=1)

        if png_path is not None:
            if png_kwargs is None:
                png_kwargs = {}
            plot.save_figure(axes, png_path, **png_kwargs)
        if csv_path is not None:
            if csv_kwargs is None:
                csv_kwargs = {}
            plot.save_dataframe(df_somersalo, csv_path, **csv_kwargs)

        return axes, df_somersalo

    def voltage_thresholds(
        self,
        campaign_multipactor_bands: CampaignMultipactorBands,
        measurement_points_to_exclude: Sequence[str] = (),
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        **fig_kw,
    ) -> tuple[Axes, pd.DataFrame]:
        """Plot the lower and upper thresholds as voltage.

        Parameters
        ----------
        campaign_multipactor_bands :
            Object holding where multipactor happens for every test.
        measurement_points_to_exclude :
            Some measurement points to exclude. The default is an empty tuple.
        png_path :
            If provided, the resulting figure will be saved at this location.
            The default is None.
        png_kwargs :
            Other keyword arguments passed to the :func:`.save_figure`
            function. The default is None.
        csv_path :
            If provided, the data to produce the figure will be saved in this
            location. The default is None.
        csv_kwargs :
            Other keyword arguments passed to the :func:`.save_dataframe`
            function.
        fig_kw :
            Other keyword arguments passed to the :meth:`pandas.DataFrame.plot`
            method.

        Returns
        -------
        tuple[Axes, pd.DataFrame]
        axes :
            Plotted axes.
        data :
            Corresponding data.

        """
        frequencies = {test.freq_mhz for test in self}
        if len(frequencies) != 1:
            raise NotImplementedError("Plot over several freqs to implement")

        voltages = self.at_last_threshold(
            ins.FieldProbe,
            campaign_multipactor_bands,
            measurement_points_to_exclude=measurement_points_to_exclude,
        )

        axes = voltages.filter(like="Lower").plot(
            grid=True,
            ylabel="Thresholds $V$ [V]",
            marker="o",
            ms=10,
            **fig_kw,
        )
        axes.set_prop_cycle(None)
        axes = voltages.filter(like="Upper").plot(
            grid=True,
            ax=axes,
            ylabel="Thresholds $V$ [V]",
            marker="^",
            ms=10,
            **fig_kw,
        )
        if png_path is not None:
            if png_kwargs is None:
                png_kwargs = {}
            plot.save_figure(axes, png_path, **png_kwargs)
        if csv_path is not None:
            if csv_kwargs is None:
                csv_kwargs = {}
            plot.save_dataframe(voltages, csv_path, **csv_kwargs)
        return axes, voltages

    def susceptibility(
        self,
        campaign_multipactor_bands: CampaignMultipactorBands,
        measurement_points_to_exclude: Sequence[str] = (),
        keep_only_travelling: bool = True,
        tol: float = 1e-6,
        gap_in_cm: float | None = None,
        xlabel: str = r"$f\times d~[\mathrm{MHz\cdot cm}]$",
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        **fig_kw,
    ) -> tuple[Axes, pd.DataFrame]:
        """Create a susceptiblity chart.

        Parameters
        ----------
        campaign_multipactor_bands :
            Object holding where multipactor happens for every test.
        measurement_points_to_exclude :
            Some measurement points to exclude.
        keep_only_travelling :
            To remove points where :math:`SWR` is not unity.
        tol :
            Tolerance over the :math:`SWR` when performing the
            ``keep_only_travelling`` check.
        gap_in_cm :
            Gap of the system. If not provided, we take the value of MULTIPAC
            test bench.
        xlabel :
            The xlabel for the plot. The default is good enough.
        png_path :
            If provided, the resulting figure will be saved at this location.
        png_kwargs :
            Other keyword arguments passed to the :func:`.save_figure`
            function.
        csv_path :
            If provided, the data to produce the figure will be saved in this
            location.
        csv_kwargs :
            Other keyword arguments passed to the :func:`.save_dataframe`
            function.
        fig_kw :
            Other keyword arguments passed to the :meth:`pandas.DataFrame.plot`
            method.

        Returns
        -------
        axes :
            Plotted axes.
        data :
            Corresponding data.

        """
        df_susceptibility = self.at_last_threshold(
            ins.FieldProbe,
            campaign_multipactor_bands,
            measurement_points_to_exclude=measurement_points_to_exclude,
        )

        frequencies = np.array([test.freq_mhz for test in self])
        if gap_in_cm is None:
            gap_in_cm = 0.5 * (3.878 - 1.687)
            logging.info(f"Used default {gap_in_cm = }")

        df_susceptibility[xlabel] = frequencies * gap_in_cm
        df_susceptibility.set_index(xlabel, inplace=True)

        if keep_only_travelling:
            swr = [test.swr for test in self]
            is_travelling = [abs(x - 1.0) < tol for x in swr]
            df_susceptibility = df_susceptibility[is_travelling]

        axes = df_susceptibility.filter(like="Lower").plot(
            marker="o", lw=0.0, **fig_kw
        )
        axes.set_prop_cycle(None)
        axes = df_susceptibility.filter(like="Upper").plot(
            ax=axes,
            ylabel="Measured voltage [V]",
            marker="^",
            lw=0.0,
            grid=True,
            logx=True,
            logy=True,
            **fig_kw,
        )
        if png_path is not None:
            if png_kwargs is None:
                png_kwargs = {}
            plot.save_figure(axes, png_path, **png_kwargs)
        if csv_path is not None:
            if csv_kwargs is None:
                csv_kwargs = {}
            plot.save_dataframe(df_susceptibility, csv_path, **csv_kwargs)
        return axes, df_susceptibility

    def animate_instruments_vs_position(
        self,
        *args,
        out_folder: str | None = None,
        iternum: int = 100,
        **kwargs,
    ) -> list[animation.FuncAnimation]:
        """Call all :meth:`.MultipactorTest.animate_instruments_vs_position`"""
        animations = []
        for i, test in enumerate(self):
            gif_path = None
            if out_folder is not None:
                gif_path = test.output_filepath(out_folder, ".gif")
            animation = test.animate_instruments_vs_position(
                *args, gif_path=gif_path, num=iternum + i, **kwargs
            )
            animations.append(animation)
        return animations

    def reconstruct_voltage_along_line(self, *args, **kwargs) -> None:
        """Call all :meth:`.MultipactorTest.reconstruct_voltage_along_line`."""
        for test in self:
            test.reconstruct_voltage_along_line(*args, **kwargs)

    def scatter_instruments_data(
        self,
        *args,
        out_folder: str | None = None,
        iternum: int = 200,
        **kwargs,
    ) -> None:
        """Call all :meth:`.MultipactorTest.scatter_instruments_data`."""
        for i, test in enumerate(self):
            png_path = None
            if out_folder is not None:
                png_path = test.output_filepath(out_folder, ".png")
            _ = test.scatter_instruments_data(
                *args, num=iternum + i, png_path=png_path, **kwargs
            )
        return
