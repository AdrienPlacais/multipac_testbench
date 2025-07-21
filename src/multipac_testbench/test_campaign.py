"""Define an object to store data from several :class:`.MultipactorTest`."""

from __future__ import annotations

import logging
import math
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Self, TypeVar

import matplotlib.pyplot as plt
import multipac_testbench.instruments as ins
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from multipac_testbench.instruments.power import ForwardPower
from multipac_testbench.instruments.reflection_coefficient import (
    ReflectionCoefficient,
)
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.multipactor_band.campaign_multipactor_bands import (
    CampaignMultipactorBands,
)
from multipac_testbench.multipactor_test import MultipactorTest
from multipac_testbench.multipactor_test.loader import TRIGGER_POLICIES
from multipac_testbench.theoretical.somersalo import (
    fit_somersalo_scaling,
    plot_somersalo_analytical,
    plot_somersalo_measured,
    somersalo_base_plot,
)
from multipac_testbench.threshold.threshold import THRESHOLD_DETECTOR_T
from multipac_testbench.threshold.threshold_set import ThresholdSet
from multipac_testbench.util import log_manager, plot
from multipac_testbench.util.types import MULTIPAC_DETECTOR_T

T = TypeVar("T", bound=Callable[..., Any])


class TestCampaign(list[MultipactorTest]):
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
        trigger_policy: TRIGGER_POLICIES = "keep_all",
        index_col: str = "Sample index",
        is_raw: bool = False,
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
        trigger_policy :
            How consecutive measures at the same power should be treated.
        index_col :
            Name of column holding measurement index.
        is_raw :
            If set to ``True``, input data files are considered to be raw, ie
            to contain acquisition voltages instead of physical quantities.

        Returns
        -------
        test_campaign :
            List of :class:`.MultipactorTest`.

        """
        if len(info) == 0:
            info = ["" for _ in filepaths]

        logfile = Path(filepaths[0].parent / "multipac_testbench.log")
        log_manager.set_up_logging(logfile_file=logfile)

        args = zip(filepaths, frequencies, swrs, info, strict=True)
        multipactor_tests = []
        for filepath, freq_mhz, swr, info in args:
            try:
                multipactor_test = MultipactorTest(
                    filepath,
                    config,
                    freq_mhz,
                    swr,
                    info,
                    sep=sep,
                    trigger_policy=trigger_policy,
                    index_col=index_col,
                    is_raw=is_raw,
                    **kwargs,
                )
            except Exception as e:
                logging.error(
                    f"Exception raised during loading of {filepath}:\n{e}"
                    "Skipping this file."
                )
                continue
            multipactor_tests.append(multipactor_test)
        return cls(multipactor_tests)

    def add_post_treater(self, *args, **kwargs) -> None:
        """Add post-treatment functions to instruments."""
        for test in self:
            test.add_post_treater(*args, **kwargs)

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

    def determine_thresholds(
        self,
        multipac_detector: MULTIPAC_DETECTOR_T,
        instrument_class: ABCMeta,
        power_growth_array_kw: dict[str, Any] | None = None,
        threshold_reducer: THRESHOLD_DETECTOR_T | None = None,
        **kwargs,
    ) -> dict[MultipactorTest, ThresholdSet]:
        """Determine every :class:`.MultipactorTest` multipactor thresholds.

        Parameters
        ----------
        multipac_detector :
            Function that takes in the ``data`` of an :class:`.Instrument`
            and returns an array, where True means multipactor and False no
            multipactor.
        instrument_class :
            Type of instrument on which ``multipac_detector`` should be
            applied.
        power_growth_array_kw :
            Keyword arguments passed to :meth:`.PowerSetpoint.growth_array`.
        threshold_reducer :
            If provided, we consider that multipactor appears when one
            detecting :class:`.Instrument` detected it (``"any"``), or only
            when all detecting :class:`.Instrument` measured it (``"all"``).

        Returns
        -------
            Objects holding all lower and upper thresholds, detected by
            ``multipac_detector`` applied on every instance of
            ``instrument_class`` for every :class:`.MultipactorTest`.

        """
        return {
            test: test.determine_thresholds(
                multipac_detector,
                instrument_class,
                power_growth_array_kw,
                threshold_reducer=threshold_reducer,
                **kwargs,
            )
            for test in self
        }

    def reconstruct_voltage_along_line(self, *args, **kwargs) -> None:
        """Call all :meth:`.MultipactorTest.reconstruct_voltage_along_line`."""
        for test in self:
            test.reconstruct_voltage_along_line(*args, **kwargs)

    def sweet_plot(
        self,
        *args,
        thresholds_sets: dict[MultipactorTest, ThresholdSet] | None = None,
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
        thresholds_sets :
            Thresholds to plot corresponding to some or each
            :class:`.MultipactorTest`.
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
        if all_on_same_plot:
            return self._sweet_plot_same_plot(
                *args,
                thresholds_sets=thresholds_sets,
                png_folder=png_folder,
                csv_folder=csv_folder,
                **kwargs,
            )

        all_axes = []
        all_df = []

        for test in self:
            png_path = (
                test.output_filepath(png_folder, ".png")
                if png_folder
                else None
            )

            csv_path = (
                test.output_filepath(csv_folder, ".csv")
                if csv_folder
                else None
            )

            axes, df_plot = test.plotter.sweet_plot(
                *args,
                threshold_set=(thresholds_sets or {}).get(test, None),
                png_path=png_path,
                csv_path=csv_path,
                **kwargs,
            )
            all_axes.append(axes)
            all_df.append(df_plot)
        return all_axes, all_df

    def _sweet_plot_same_plot(
        self,
        *args,
        thresholds_sets: dict[MultipactorTest, ThresholdSet] | None = None,
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[list[Axes], pd.DataFrame]:
        """Plot the various signals on the same Axes.

        Parameters
        ----------
        args :
            Arguments that are passed to :meth:`.MultipactorTest.sweet_plot`.
        thresholds_sets :
            Thresholds to plot corresponding to some or each
            :class:`.MultipactorTest`.
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

        if len(args) > 1:
            logging.warning(
                "I am not sure how the interaction of all_on_same_plot with "
                "several instruments plotted will go."
            )
        axes = None
        all_df = []
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, test in enumerate(self):
            axes, df_plot = test.plotter.sweet_plot(
                *args,
                threshold_set=(thresholds_sets or {}).get(test, None),
                axes=axes,
                column_names=str(test),
                title=" ",
                test_color=colors[i],
                **kwargs,
            )
            all_df.append(df_plot)
        assert axes is not None
        df_to_plot = pd.concat(all_df, axis=1)

        if png_path:
            plot.save_figure(axes, png_path, **(png_kwargs or {}))
        if csv_path:
            plot.save_dataframe(df_to_plot, csv_path, **(csv_kwargs or {}))

        return axes, df_to_plot

    def plot_thresholds(
        self,
        to_plot: ABCMeta,
        thresholds_sets: dict[MultipactorTest, ThresholdSet],
        *args,
        png_folder: str | None = None,
        csv_folder: str | None = None,
        **kwargs,
    ) -> tuple[list[list[Axes]], list[pd.DataFrame]]:
        """Recursively call :meth:`.MultipactorTest.plot_thresholds`."""
        all_axes = []
        all_df = []
        for test, threshold_set in thresholds_sets.items():
            png_path = (
                test.output_filepath(png_folder, ".png")
                if png_folder is not None
                else None
            )
            csv_path = (
                test.output_filepath(csv_folder, ".csv")
                if csv_folder is not None
                else None
            )

            axes, df_plot = test.plot_thresholds(
                to_plot,
                threshold_set,
                *args,
                png_path=png_path,
                csv_path=csv_path,
                **kwargs,
            )
            all_axes.append(axes)
            all_df.append(df_plot)
        return all_axes, all_df

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
        thresholds_sets: dict[MultipactorTest, ThresholdSet],
        show_fit: bool = True,
        use_theoretical_r: bool = False,
        full_output: bool = True,
        add_upper_thresholds: bool = False,
        axes: Axes | None = None,
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        **fig_kw,
    ) -> tuple[
        Axes | None,
        dict[float, pd.DataFrame] | None,  # df_low per freq
        dict[float, pd.DataFrame] | None,  # df_upp per freq
        dict[float, pd.DataFrame | None] | None,  # df_fit per freq
    ]:
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
        thresholds_sets :
            Contains the threshold to represent. For every instrument, will
            plot the last lower and upper thresholds found. In general, you
            will want to give :class:`.ThresholdSet` corresponding to
            multipactor detected anywhere in the line (use ``threshold_reducer
            = "any"`` in :meth:`.ThresholdSet.from_instruments`). Additionally,
            you can also average the last :class:`.Threshold` by giving an
            :class:`.AveragedThresholdSet`.
        show_fit :
            To perform a fit and plot it.
        use_theoretical_r :
            Another patch to allow fitting and plotting using the theoretical
            reflection coefficient instead of the one calculated from
            :math:`P_f` and :math:`P_r`.
        axes :
            Axes to plot if provided.
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
        axes : Axes | None,
            Holds the plot.
        df_low : dict[float, pd.DataFrame] | None
            Holds forward power at every lower threshold, by RF frequency.
        df_upp : dict[float, pd.DataFrame] | None
            Holds forward power at every upper threshold, by RF frequency.
        df_fit : dict[float, pd.DataFrame] | None
            Holds forward power at every lower threshold, as fitted on
            ``df_low``, for the different RF frequencies.

        """

        def build_dataframe(
            r_values: list[float], p_values: list[float], label: str
        ) -> pd.DataFrame:
            return (
                pd.DataFrame({"R": r_values, label: p_values})
                .sort_values("R")
                .dropna()
            )

        plot_kwargs = {
            "x": "R",
            "xlabel": ReflectionCoefficient.ylabel(),
            "ylabel": ForwardPower.ylabel(),
            "grid": True,
            "ms": 15,
            **fig_kw,
        }

        tests_by_freq: dict[float, list[MultipactorTest]] = defaultdict(list)
        for test in self:
            tests_by_freq[test.freq_mhz].append(test)

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = [c["color"] for c in prop_cycle]
        freq_to_color = {
            freq: colors[i % len(colors)]
            for i, freq in enumerate(sorted(tests_by_freq.keys()))
        }

        df_lows, df_upps, df_fits = {}, {}, {}
        df_low, df_upp, df_fit = None, None, None
        for freq_mhz, tests in sorted(tests_by_freq.items()):
            lower_rs, lower_ps, upper_rs, upper_ps = [], [], [], []

            for test in tests:
                args = test.data_for_somersalo_scaling_law(
                    thresholds_sets[test],
                    use_theoretical_r=use_theoretical_r,
                )
                lower_r, lower_p, upper_r, upper_p = args
                lower_rs.append(lower_r)
                lower_ps.append(lower_p)
                upper_rs.append(upper_r)
                upper_ps.append(upper_p)

            suffix = f"{freq_mhz:.0f}MHz"
            df_low = build_dataframe(lower_rs, lower_ps, f"P_low ({suffix})")
            df_upp = build_dataframe(upper_rs, upper_ps, f"P_high ({suffix})")
            if not df_low.empty:
                axes = df_low.plot(
                    marker="o",
                    label=f"P_low ({suffix})",
                    ax=axes,
                    c=freq_to_color[freq_mhz],
                    **plot_kwargs,
                )
            if add_upper_thresholds and not df_upp.empty:
                axes = df_upp.plot(
                    marker="*",
                    label=f"P_high ({suffix})",
                    ax=axes,
                    c=freq_to_color[freq_mhz],
                    **plot_kwargs,
                )

            df_fit = None
            if show_fit:
                if not df_low.empty:
                    df_fit = fit_somersalo_scaling(
                        df_low,
                        full_output=full_output,
                        plot=True,
                        axes=axes,
                        freq_mhz=suffix,
                        c=freq_to_color[freq_mhz],
                    )

            df_lows[freq_mhz] = df_low
            df_upps[freq_mhz] = df_upp
            df_fits[freq_mhz] = df_fit

        if axes is None:
            logging.error("No Somersalo scaling law data plotted.")
            return None, df_lows, df_upps, df_fits

        if png_path is not None:
            plot.save_figure(axes, png_path, **(png_kwargs or {}))
        if csv_path:
            csv_stem = csv_path.stem
            csv_dir = csv_path.parent
            for freq_mhz in df_lows:
                for df, name in zip(
                    (df_lows, df_upps, df_fits), ("low", "upp", "fit")
                ):
                    file = csv_dir / f"{csv_stem}_{freq_mhz:.0f}MHz_{name}.csv"
                    plot.save_dataframe(
                        df[freq_mhz], file, **(csv_kwargs or {})
                    )
        return axes, df_lows, df_upps, df_fits

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
        """Create a susceptibility chart.

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
            is_travelling = [math.isclose(x, 1.0, abs_tol=tol) for x in swr]
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
