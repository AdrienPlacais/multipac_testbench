"""Define an object to store and treat data from pick-ups.

.. todo::
    Allow to trim data (remove noisy useless data at end of exp)

.. todo::
    name of pick ups in animation

.. todo::
    histograms for mp voltages? Maybe then add a gaussian fit, then we can
    determine the 3sigma multipactor limits?

.. todo::
    ``to_ignore``, ``to_exclude`` arguments should have more consistent names.

"""

import itertools
import logging
from abc import ABCMeta
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import multipac_testbench.instruments as ins
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.measurement_point.factory import (
    IMeasurementPointFactory,
)
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.multipactor_band.instrument_multipactor_bands import (
    InstrumentMultipactorBands,
)
from multipac_testbench.multipactor_band.test_multipactor_bands import (
    TestMultipactorBands,
)
from multipac_testbench.util import plot
from multipac_testbench.util.animate import get_limits
from multipac_testbench.util.helper import (
    flatten,
    output_filepath,
    split_rows_by_masks,
    types_match,
)
from multipac_testbench.util.types import MULTIPAC_DETECTOR_T
from numpy.typing import NDArray


class MultipactorTest:
    """Holds a mp test with several probes."""

    def __init__(
        self,
        filepath: Path,
        config: dict,
        freq_mhz: float,
        swr: float,
        info: str = "",
        sep: str = "\t",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        r"""Create all the pick-ups.

        Parameters
        ----------
        filepath :
            Path to the results file produced by LabViewer.
        config :
            Configuration ``TOML`` of the testbench.
        freq_mhz :
            Frequency of the test in :unit:`MHz`.
        swr :
            Expected Voltage Signal Wave Ratio.
        info :
            An additional string to identify this test in plots.
        sep :
            Delimiter between two columns in ``filepath``.
        verbose :
            To print information on the structure of the test bench, as it was
            understood.

        """
        self.filepath = filepath
        try:
            df_data = pd.read_csv(
                filepath, sep=sep, index_col="Sample index", **kwargs
            )
        except Exception as e:
            logging.error(
                f"There was a mismatch is the number of columns in {filepath}"
                ". Check that the number of column header match the number of "
                "columns, and that the trailing comments in the first lines "
                "were removed."
            )
            logging.exception(e)
            raise e
        self._n_points = len(df_data)
        self.df_data = df_data

        if df_data.index[0] != 0:
            logging.error(
                "Your Sample index column does not start at 0. I should patch "
                "this, but meanwhile expect some index mismatches."
            )

        imeasurement_point_factory = IMeasurementPointFactory(
            freq_mhz=freq_mhz
        )
        imeasurement_points = imeasurement_point_factory.run(
            config,
            df_data,
            verbose,
        )
        #: Where all diagnostics at a specific pick-up are defined (e.g.
        #: current probe)
        self.pick_ups = imeasurement_points[1]
        #: Where all diagnostics which are not a specific position are stored
        #: (e.g. forward/reflected power)
        self.global_diagnostics = imeasurement_points[0]

        self.freq_mhz = freq_mhz
        self.swr = swr
        self.info = info

    def __str__(self) -> str:
        """Print info on object."""
        out = [f"{self.freq_mhz}MHz", f"SWR {self.swr}"]
        if len(self.info) > 0:
            out.append(f"{self.info}")
        return ", ".join(out)

    def add_post_treater(
        self, *args, only_pick_up_which_name_is: tuple[str, ...] = (), **kwargs
    ) -> None:
        """Add post-treatment functions to instruments."""
        pick_ups = self.pick_ups
        if len(only_pick_up_which_name_is) > 0:
            pick_ups = [
                pick_up
                for pick_up in self.pick_ups
                if pick_up.name in only_pick_up_which_name_is
            ]

        for pick_up in pick_ups:
            pick_up.add_post_treater(*args, **kwargs)

    def sweet_plot(
        self,
        *ydata: ABCMeta,
        xdata: ABCMeta | None = None,
        exclude: Sequence[str] = (),
        tail: int | None = None,
        xlabel: str = "",
        ylabel: str | Iterable = "",
        grid: bool = True,
        title: str | list[str] = "",
        test_multipactor_bands: TestMultipactorBands | None = None,
        column_names: str | list[str] = "",
        test_color: str | None = None,
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        axes: list[Axes] | None = None,
        masks: dict[str, NDArray[np.bool]] | None = None,
        drop_repeated_x: bool = False,
        **kwargs,
    ) -> tuple[list[Axes], pd.DataFrame]:
        """Plot ``ydata`` versus ``xdata``.

        .. todo::
            Kwargs mixed up between the different methods.

        Parameters
        ----------
        *ydata :
            Class of the instruments to plot.
        xdata :
            Class of instrument to use as x-data. If there is several
            instruments which have this class, only one ``ydata`` is allowed
            and number of ``x`` and ``y`` instruments must match. The default
            is None, in which case data is plotted vs sample index.
        exclude :
            Name of the instruments that you do not want to see plotted.
        tail :
            Specify this to only plot the last ``tail`` points. Useful to
            select only the last power cycle.
        xlabel :
            Label of x axis.
        ylabel :
            Label of y axis.
        grid :
            To show the grid.
        title :
            Title of the plot or of the subplots.
        test_multipactor_bands :
            If provided, information is added to the plot to show where
            multipactor happens.
        column_names :
            To override the default column names. This is used in particular
            with the method :meth:`.TestCampaign.sweet_plot` when
            ``all_on_same_plot=True``.
        test_color :
            Color used by :meth:`.TestCampaign.sweet_plot` when
            ``all_on_same_plot=True``. It overrides the :class:`.Instrument`
            color and is used to discriminate every :class:`.MultipactorTest`
            from another.
        png_path :
            If specified, save the figure at ``png_path``.
        csv_path :
            If specified, save the data used to produce the plot in
            ``csv_path``.
        masks :
            A dictionary where each key is a suffix used to label the split
            columns, and each value is a boolean mask of the same length as the
            input data. Keys must start with two underscores (``__``) to enable
            consistent column naming and compatibility with downstream styling
            logic (e.g., grouping lines by base column in plots). If multiple
            masks are ``True`` at the same row index, a ``ValueError`` is
            raised.
        drop_repeated_x :
            If True, remove consecutive rows with identical x values.
        **kwargs :
            Other keyword arguments passed to :meth:`pandas.DataFrame.plot`,
            :meth:`._set_y_data`, :func:`.create_df_to_plot`,
            :func:`.set_labels`.

        Returns
        -------
        axes :
            Objects holding the plot.
        df_to_plot :
            DataFrame holding the data that is plotted.

        """
        data_to_plot, x_columns = self._set_x_data(xdata, exclude=exclude)
        data_to_plot, y_columns, color = self._set_y_data(
            data_to_plot,
            *ydata,
            exclude=exclude,
            column_names=column_names,
            masks=masks,
            **kwargs,
        )
        if test_color is not None:
            color = test_color

        df_to_plot = plot.create_df_to_plot(
            data_to_plot,
            tail=tail,
            column_names=column_names,
            drop_repeated_x=drop_repeated_x,
            **kwargs,
        )

        x_column, y_column = plot.match_x_and_y_column_names(
            x_columns, y_columns
        )

        if not xlabel:
            xlabel = xdata.name if isinstance(xdata, Instrument) else ""

        if axes is None:
            if not title:
                title = str(self)

            _, dic_axes = plot.create_fig(
                title=title if isinstance(title, str) else title[0],
                instruments_to_plot=ydata,
                xlabel=xlabel,
            )
            axes = list(dic_axes.values())

        axes = plot.actual_plot(
            df_to_plot,
            x_column,
            y_column,
            axes=axes,
            grid=grid,
            color=color,
            **kwargs,
        )

        plot.set_labels(
            axes, *ydata, xdata=xdata, xlabel=xlabel, ylabel=ylabel, **kwargs
        )

        if test_multipactor_bands is not None:
            plot.add_instrument_multipactor_bands(
                test_multipactor_bands, axes, twinx=True
            )

        if png_path is not None:
            if png_kwargs is None:
                png_kwargs = {}
            plot.save_figure(axes, png_path, **png_kwargs)
        if csv_path is not None:
            if csv_kwargs is None:
                csv_kwargs = {}
            plot.save_dataframe(df_to_plot, csv_path, **csv_kwargs)
        return axes, df_to_plot

    def _set_x_data(
        self,
        xdata: ABCMeta | None,
        exclude: Sequence[str] = (),
    ) -> tuple[list[pd.Series], list[str] | None]:
        """Set the data that will be used for x-axis.

        Parameters
        ----------
        xdata :
            Class of an instrument, or None (in this case, use default index).
        exclude :
            Name of instruments to exclude.

        Returns
        -------
        data_to_plot :
            Contains the data used for x axis.
        x_columns :
            Name of the column(s) used for x axis.

        """
        if xdata is None:
            return [], None

        instruments = self.get_instruments(
            xdata, instruments_to_ignore=exclude
        )
        x_columns = [
            instrument.name
            for instrument in instruments
            if instrument.name not in exclude
        ]

        data_to_plot = []
        for instrument in instruments:
            if isinstance(instrument.data_as_pd, pd.DataFrame):
                logging.error(
                    f"You want to plot {instrument}, which data is 2D. Not "
                    "supported."
                )
                continue
            data_to_plot.append(instrument.data_as_pd)

        return data_to_plot, x_columns

    def _set_y_data(
        self,
        data_to_plot: list[pd.Series | pd.DataFrame],
        *ydata: ABCMeta,
        exclude: Sequence[str] = (),
        column_names: str | list[str] = "",
        masks: dict[str, NDArray[np.bool]] | None = None,
        **kwargs,
    ) -> tuple[list[pd.Series], list[list[str]], dict[str, str]]:
        """Set the y-data that will be plotted.

        Parameters
        ----------
        data_to_plot :
            List already containing the x-data, or nothing if the index is to
            be used.
        *ydata :
            The class of the instruments to plot.
        exclude :
            Name of some instruments to exclude.
        column_names :
            To override the default column names. This is used in particular
            with the method :meth:`.TestCampaign.sweet_plot`, when
            ``all_on_same_plot=True``.
        masks :
            A dictionary where each key is a suffix used to label the split
            columns, and each value is a boolean mask of the same length as the
            input data. Keys must start with two underscores (``__``) to enable
            consistent column naming and compatibility with downstream styling
            logic (e.g., grouping lines by base column in plots). If multiple
            masks are ``True`` at the same row index, a ``ValueError`` is
            raised.
        kwargs :
            Other keyword arguments.

        Returns
        -------
        data_to_plot :
            List containing all the series that will be plotted.
        y_columns :
            Contains, for every subplot, the name of the columns to plot.
            If ``column_names`` is provided, it overrides the given
            ``y_columns``.
        color :
            Dictionary linking column names in ``df_to_plot`` to HTML colors.
            Used to keep the same color between different instruments at the
            same :class:`.PickUp`.

        """
        instruments = [self.get_instruments(y) for y in ydata]
        y_columns = []
        color: dict[str, str] = {}

        for sublist in instruments:
            sub_ycols = []

            for instrument in sublist:
                if instrument.name in exclude:
                    logging.debug(
                        f"Skipping {instrument} because it is excluded."
                    )
                    continue

                df = instrument.data_as_pd
                if masks is not None:
                    df = split_rows_by_masks(df, masks=masks)

                data_to_plot.append(df)

                if isinstance(ser := df, pd.Series):
                    sub_ycols.append(ser.name)
                    color[ser.name] = instrument.color
                    continue

                names = df.columns.to_list()
                if masks is not None:
                    names = [names]
                sub_ycols.extend(names)

                for name in flatten(names):
                    color[name] = instrument.color

            y_columns.append(sub_ycols)

        if column_names:
            logging.info("Instrument.color attribute will not be used.")
            if len(y_columns) > 1:
                logging.warning("This will lead to duplicate column names.")
            if isinstance(column_names, str):
                column_names = [column_names]

            y_columns = [column_names for _ in y_columns]

        return data_to_plot, y_columns, color

    def plot_thresholds(
        self,
        instrument_id: ABCMeta,
        multipactor_bands: TestMultipactorBands | InstrumentMultipactorBands,
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        instruments_to_ignore: Sequence[ins.Instrument | str] = (),
        title: str = "",
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[Axes | NDArray[Axes], pd.DataFrame]:
        """Plot instrument ``to_plot`` at every multipactor threshold.

        When ``to_plot`` is :class:`.ForwardPower` or :class:`.FieldProbe`,
        the output is the threshold. But this method works with any instrument
        type.

        .. todo::
            Add a way to fit exponential (?) law on the thresholds. Will need
            to change the x-axis.

        Parameters
        ----------
        instrument_id :
            Class of instrument to plot. Makes most sense with
            :class:`.ForwardPower` or :class:`.FieldProbe`.
        multipactor_bands :
            Object containing the indexes of multipacting. If only a
            :class:`.InstrumentMultipactorBands` is given, all plotted
            instruments will use it.
        measurement_points_to_exclude :
            To exclude some pick-ups.
        instruments_to_ignore :
            To exclude some instruments.
        png_path :
            If provided, figure will be saved there.
        png_kwargs :
            Keyword arguments for the :meth:`matplotlib.figure.Figure.savefig`
            method.
        csv_path :
            If provided, plotted data will be saved there.
        csv_kwargs :
            Keyword arguments for the :meth:`pandas.DataFrame.to_csv` method.

        Returns
        -------
        axes :
            Hold plotted axes.
        df_thresholds :
            The data used to produce the plot.

        """
        zipper = self.instruments_and_multipactor_bands(
            instrument_id,
            multipactor_bands,
            raise_no_match_error=True,
            global_diagnostics=True,
            measurement_points_to_exclude=measurement_points_to_exclude,
            instruments_to_ignore=instruments_to_ignore,
        )
        if not title:
            title = str(self)

        thresholds = [
            instrument.at_thresholds(multipactor_band)
            for instrument, multipactor_band in zipper
        ]
        df_thresholds = pd.concat(thresholds, axis=1)
        axes = df_thresholds.filter(like="Lower").plot(
            marker="o",
            ms=10,
            title=title,
            **kwargs,
        )
        axes.set_prop_cycle(None)
        axes = df_thresholds.filter(like="Upper").plot(
            ax=axes,
            grid=True,
            marker="^",
            ms=10,
            xlabel="Half-power cycle #",
            ylabel=instrument_id.ylabel(),
            **kwargs,
        )
        if png_path is not None:
            if png_kwargs is None:
                png_kwargs = {}
            plot.save_figure(axes, png_path, **png_kwargs)
        if csv_path is not None:
            if csv_kwargs is None:
                csv_kwargs = {}
            plot.save_dataframe(df_thresholds, csv_path, **csv_kwargs)
        return axes, df_thresholds

    def instruments_and_multipactor_bands(
        self,
        instruments_id: ABCMeta,
        multipactor_bands: TestMultipactorBands | InstrumentMultipactorBands,
        raise_no_match_error: bool = True,
        global_diagnostics: bool = True,
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        instruments_to_ignore: Sequence[ins.Instrument | str] = (),
    ) -> zip:
        """Match the instruments with their multipactor bands.

        Parameters
        ----------
        instruments_id :
            Class of instrument under study.
        multipactor_bands :
            All multipactor bands, among which we will be looking. If only one
            is given (:class:`.InstrumentMultipactorBands`), then all
            :class:`.Instrument` will be matched with the same identical
            :class:`.InstrumentMultipactorBands`.
        raise_no_match_error :
            If an error should be raised when no
            :class:`.InstrumentMultipactorBands` match an
            :class:`.Instrument`.
        global_diagnostics :
            If :class:`.InstrumentMultipactorBands` that were obtained from a
            global diagnostic should be matched.
        measurement_points_to_exclude :
            :class:`.Instrument` at this pick-ups are skipped.
        instruments_to_ignore :
            :class:`.Instrument` in this sequence are skipped.

        Returns
        -------
        zipper :
            Object matching every :class:`.Instrument` with the appropriate
            :class:`.InstrumentMultipactorBands`.

        """
        instruments = self.get_instruments(
            instruments_id,
            measurement_points_to_exclude,
            instruments_to_ignore,
        )

        matching_mp_bands = [
            instrument.multipactor_band_at_same_position(
                multipactor_bands,
                raise_no_match_error=raise_no_match_error,
                global_diagnostics=global_diagnostics,
            )
            for instrument in instruments
        ]
        zipper = zip(instruments, matching_mp_bands, strict=True)
        return zipper

    def at_last_threshold(
        self,
        instrument_id: ABCMeta | Sequence[ABCMeta],
        multipactor_bands: TestMultipactorBands | InstrumentMultipactorBands,
        **kwargs,
    ) -> pd.DataFrame:
        """Give the ``instrument_id`` measurements at last threshold."""
        if isinstance(instrument_id, Sequence):
            all_df_thresholds = [
                self.at_last_threshold(
                    single_instrument_id, multipactor_bands, **kwargs
                )
                for single_instrument_id in instrument_id
            ]
            return pd.concat(all_df_thresholds, axis=1)

        zipper = self.instruments_and_multipactor_bands(
            instrument_id, multipactor_bands, **kwargs
        )
        df_thresholds = pd.concat(
            [
                instrument.at_thresholds(band).tail(1)
                for instrument, band in zipper
            ],
            axis=1,
        )
        df_thresholds.index = [str(self)]
        return df_thresholds

    def detect_multipactor(
        self,
        multipac_detector: MULTIPAC_DETECTOR_T,
        instrument_class: ABCMeta,
        power_growth_mask_kw: dict[str, Any] | None = None,
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        debug: bool = False,
        **kwargs,
    ) -> TestMultipactorBands:
        """Create the :class:`.TestMultipactorBands` object.

        Parameters
        ----------
        multipac_detector :
            Function that takes in the ``data`` of an :class:`.Instrument`
            and returns an array, where True means multipactor and False no
            multipactor.
        instrument_class :
            Type of instrument on which ``multipac_detector`` should be
            applied.
        power_growth_mask_kw :
            Keyword arguments passed to :meth:`.ForwardPower.growth_mask`.
        measurement_points_to_exclude :
            Some measurement points that should not be considered.
        debug :
            To plot the data used for multipactor detection, where power grows,
            where multipactor is detected.

        Returns
        -------
        test_multipactor_bands :
            Objets containing when multipactor happens, according to
            ``multipac_detector``, at every pick-up holding an
            :class:`.Instrument` of type ``instrument_class``.

        """
        growth_mask = self._power_growth_mask(power_growth_mask_kw)

        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude
        )

        instrument_multipactor_bands = [
            measurement_point.detect_multipactor(
                multipac_detector,
                instrument_class,
                growth_mask,
                debug,
                info=f" {self}",
            )
            for measurement_point in measurement_points
        ]
        test_multipactor_bands = TestMultipactorBands(
            instrument_multipactor_bands, growth_mask
        )
        return test_multipactor_bands

    def _power_growth_mask(
        self, growth_mask_kw: dict[str, Any] | None = None
    ) -> NDArray[np.bool]:
        """Determine where the power is growing.

        Parameters
        ----------
        growth_mask_kw :
            Keyword arguments passed to :meth:`.ForwardPower.growth_mask`.

        Returns
        -------
            ``True`` where power increases, ``False`` where it decreases.

        """
        forward_power = self.get_instrument(ins.ForwardPower)
        assert isinstance(forward_power, ins.ForwardPower), (
            f"{forward_power} is a {type(forward_power)} instead of a "
            "ForwardPower."
        )
        if growth_mask_kw is None:
            growth_mask_kw = {}
        mask = forward_power.growth_mask(**growth_mask_kw)
        return mask

    def animate_instruments_vs_position(
        self,
        instruments_to_plot: Sequence[ABCMeta],
        gif_path: Path | None = None,
        fps: int = 50,
        keep_one_frame_over: int = 1,
        interval: int | None = None,
        only_first_frame: bool = False,
        last_frame: int | None = None,
        **fig_kw,
    ) -> animation.FuncAnimation | list[Axes]:
        """Represent measured signals with probe position.

        .. todo::
            ``last_frame`` badly handled: gif will be as long as if the
            ``last_frame`` was not set, except that images won't be updated
            after the last frame.

        """
        fig, axes_instruments = self._prepare_animation_fig(
            instruments_to_plot, **fig_kw
        )

        frames = self._n_points - 1
        artists = self._plot_instruments_single_time_step(
            0,
            keep_one_frame_over=keep_one_frame_over,
            axes_instruments=axes_instruments,
            artists=None,
        )
        if only_first_frame:
            return list(axes_instruments.keys())

        def update(step_idx: int) -> Sequence[Artist]:
            """Update the ``artists`` defined in outer scope.

            Parameters
            ----------
            step_idx :
                Step that shall be plotted.

            Returns
            -------
            artists :
                Updated artists.

            """
            self._plot_instruments_single_time_step(
                step_idx,
                keep_one_frame_over=keep_one_frame_over,
                axes_instruments=axes_instruments,
                artists=artists,
                last_frame=last_frame,
            )
            assert artists is not None
            return artists

        if interval is None:
            interval = int(200 / keep_one_frame_over)

        ani = animation.FuncAnimation(
            fig, update, frames=frames, interval=interval, repeat=True
        )

        if gif_path is not None:
            writergif = animation.PillowWriter(fps=fps)
            ani.save(gif_path, writer=writergif)
        return ani

    def _prepare_animation_fig(
        self,
        to_plot: Sequence[ABCMeta],
        measurement_points_to_exclude: tuple[str, ...] = (),
        instruments_to_ignore_for_limits: tuple[str, ...] = (),
        instruments_to_ignore: Sequence[ins.Instrument | str] = (),
        **fig_kw,
    ) -> tuple[Figure, dict[Axes, list[ins.Instrument]]]:
        """Create the figure and axes for the animation.

        Parameters
        ----------
        to_plot :
            Classes of instruments you want to see.
        measurement_points_to_exclude :
            Measurement points that should not appear.
        instruments_to_ignore_for_limits :
            Instruments to plot, but that can go off limits.
        instruments_to_ignore :
            Instruments that will not even be plotted.
        fig_kw :
            Other keyword arguments for Figure.

        Returns
        -------
        fig :
         Figure holding the axes.
        axes_instruments :
            Links the instruments to plot with the Axes they should be plotted
            on.

        """
        fig, instrument_class_axes = plot.create_fig(
            str(self), to_plot, xlabel="Position [m]", **fig_kw
        )

        for instrument_class, axe in instrument_class_axes.items():
            axe.set_ylabel(instrument_class.ylabel())

        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude
        )

        axes_instruments = {
            axe: self._instruments_by_class(
                instrument_class,
                measurement_points,
                instruments_to_ignore=instruments_to_ignore,
            )
            for instrument_class, axe in instrument_class_axes.items()
        }

        y_limits = get_limits(
            axes_instruments, instruments_to_ignore_for_limits
        )
        axe = None
        for axe, y_lim in y_limits.items():
            axe.set_ylim(y_lim)

        return fig, axes_instruments

    def _plot_instruments_single_time_step(
        self,
        step_idx: int,
        keep_one_frame_over: int,
        axes_instruments: dict[Axes, list[ins.Instrument]],
        artists: Sequence[Artist] | None = None,
        last_frame: int | None = None,
    ) -> Sequence[Artist] | None:
        """Plot all instruments signal at proper axe and time step."""
        if step_idx % keep_one_frame_over != 0:
            return

        if last_frame is not None and step_idx > last_frame:
            return

        sample_index = step_idx + 1

        if artists is None:
            artists = [
                instrument.plot_vs_position(sample_index, axe=axe)
                for axe, instruments in axes_instruments.items()
                for instrument in instruments
            ]
            return artists

        i = 0
        for instruments in axes_instruments.values():
            for instrument in instruments:
                instrument.plot_vs_position(sample_index, artist=artists[i])
                i += 1
        return artists

    def scatter_instruments_data(
        self,
        instruments_to_plot: Sequence[ABCMeta],
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        instrument_multipactor_bands: (
            Sequence[InstrumentMultipactorBands] | None
        ) = None,
        png_path: Path | None = None,
        **fig_kw,
    ) -> tuple[Figure, list[Axes]]:
        """Plot the data measured by instruments.

        This plot results in important amount of points. It becomes interesting
        when setting different colors for multipactor/no multipactor points and
        can help see trends.

        .. todo::
            Also show from global diagnostic

        .. todo::
            User should be able to select: reconstructed or measured electric
            field.

        .. todo::
            Fix this. Or not? This is not the most explicit way to display
            data...

        """
        raise NotImplementedError("currently broken")
        if fig_kw is None:
            fig_kw = {}
        fig, instrument_class_axes = plot.create_fig(
            str(self), instruments_to_plot, xlabel="Probe index", **fig_kw
        )
        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude
        )

        instrument_multipactor_bands = (
            self._get_proper_instrument_multipactor_bands(
                multipactor_measured_at=measurement_points,
                instrument_multipactor_bands=instrument_multipactor_bands,
                measurement_points_to_exclude=measurement_points_to_exclude,
            )
        )

        for i, measurement_point in enumerate(measurement_points):
            measurement_point.scatter_instruments_data(
                instrument_class_axes,
                xdata=float(i),
            )

        fig, axes = plot.finish_fig(
            fig, instrument_class_axes.values(), png_path
        )
        return fig, axes

    def _instruments_by_class(
        self,
        instrument_class: ABCMeta,
        measurement_points: Sequence[IMeasurementPoint] | None = None,
        instruments_to_ignore: Sequence[ins.Instrument | str] = (),
    ) -> list[ins.Instrument]:
        """Get all instruments of desired class from ``measurement_points``.

        But remove the instruments to ignore.

        Parameters
        ----------
        instrument_class :
            Class of the desired instruments.
        measurement_points :
            The measurement points from which you want the instruments. The
            default is None, in which case we look into every
            :class:`.IMeasurementPoint` attribute of self.
        instruments_to_ignore :
            The :class:`.Instrument` or instrument names you do not want.

        Returns
        -------
        instruments :
            All the instruments matching the required conditions.

        """
        if measurement_points is None:
            measurement_points = self.get_measurement_points()

        instruments_2d = [
            measurement_point.get_instruments(
                instrument_class,
                instruments_to_ignore=instruments_to_ignore,
            )
            for measurement_point in measurement_points
        ]
        instruments = [
            instrument
            for instrument_1d in instruments_2d
            for instrument in instrument_1d
        ]
        return instruments

    def _instruments_by_name(
        self,
        instrument_names: Sequence[str],
    ) -> list[ins.Instrument]:
        """Get all instruments of desired name from ``measurement_points``.

        But remove the instruments to ignore.

        Parameters
        ----------
        instrument_name :
            Name of the desired instruments.

        Returns
        -------
        instruments :
            All the instruments matching the required conditions.

        """
        all_measurement_points = self.get_measurement_points()
        instruments = [
            instr
            for measurement_point in all_measurement_points
            for instr in measurement_point.instruments
            if instr.name in instrument_names
        ]
        if len(instrument_names) != len(instruments):
            logging.warning(
                f"You asked for {instrument_names = }, I give you "
                f"{[instr.name for instr in instruments]} which has a "
                "different length."
            )
        return instruments

    def get_measurement_points(
        self,
        names: Sequence[str] | None = None,
        to_exclude: Sequence[str | IMeasurementPoint] = (),
    ) -> Sequence[IMeasurementPoint]:
        """Get all or some measurement points.

        Parameters
        ----------
        names :
            If given, only the :class:`.IMeasurementPoint` which name is in
            ``names`` will be returned.
        to_exclude :
            List of objects or objects names to exclude from returned list.

        Returns
        -------
        i_measurement_points :
            The desired objects.

        """
        names_to_exclude = [
            x if isinstance(x, str) else x.name for x in to_exclude
        ]

        measurement_points = [
            x
            for x in self.pick_ups + [self.global_diagnostics]
            if x is not None and x.name not in names_to_exclude
        ]

        if names is not None and len(names) > 0:
            return [x for x in measurement_points if x.name in names]
        return measurement_points

    def get_measurement_point(
        self,
        name: str | None = None,
        to_exclude: Sequence[str | IMeasurementPoint] = (),
    ) -> IMeasurementPoint:
        """Get all or some measurement points. Ensure there is only one.

        Parameters
        ----------
        name :
            If given, only the :class:`.IMeasurementPoint` which name is in
            ``names`` will be returned.
        to_exclude :
            List of objects or objects names to exclude from returned list.

        Returns
        -------
        measurement_point :
            The desired object.

        """
        if name is not None:
            name = (name,)
        measurement_points = self.get_measurement_points(name, to_exclude)
        assert len(measurement_points) == 1, (
            "Only one IMeasurementPoint " "should match."
        )
        return measurement_points[0]

    def get_instruments(
        self,
        instruments_id: (
            ABCMeta
            | Sequence[ABCMeta]
            | Sequence[str]
            | Sequence[ins.Instrument]
        ),
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        instruments_to_ignore: Sequence[ins.Instrument | str] = (),
    ) -> list[ins.Instrument]:
        """Get all instruments matching ``instrument_id``."""
        match (instruments_id):
            case list() | tuple() as instruments if types_match(
                instruments, ins.Instrument
            ):
                return instruments

            case list() | tuple() as names if types_match(names, str):
                out = self._instruments_by_name(names)

            case list() | tuple() as classes if types_match(classes, ABCMeta):
                measurement_points = self.get_measurement_points(
                    to_exclude=measurement_points_to_exclude
                )
                out_2d = [
                    self._instruments_by_class(
                        instrument_class,
                        measurement_points,
                        instruments_to_ignore=instruments_to_ignore,
                    )
                    for instrument_class in classes
                ]
                out = list(itertools.chain.from_iterable(out_2d))

            case ABCMeta() as instrument_class:
                measurement_points = self.get_measurement_points(
                    to_exclude=measurement_points_to_exclude
                )
                out = self._instruments_by_class(
                    instrument_class,
                    measurement_points,
                    instruments_to_ignore=instruments_to_ignore,
                )
            case _:
                raise OSError(
                    f"instruments is {type(instruments_id)} which ",
                    "is not supported.",
                )
        return out

    def get_instrument(
        self,
        instrument_id: ABCMeta | str | ins.Instrument,
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        instruments_to_ignore: Sequence[ins.Instrument | str] = (),
    ) -> ins.Instrument | None:
        """Get a single instrument matching ``instrument_id``."""
        match (instrument_id):
            case ins.Instrument():
                return instrument_id
            case str() as instrument_name:
                instruments = self.get_instruments((instrument_name,))
            case ABCMeta() as instrument_class:
                instruments = self.get_instruments(
                    instrument_class,
                    measurement_points_to_exclude,
                    instruments_to_ignore,
                )

        if len(instruments) == 0:
            raise OSError("No instrument found.")
        if len(instruments) > 1:
            logging.warning("Several instruments found. Returning first one.")
        return instruments[0]

    def reconstruct_voltage_along_line(
        self,
        name: str,
        probes_to_ignore: Sequence[str | ins.FieldProbe] = (),
    ) -> None:
        """Reconstruct the voltage profile from the e field probes."""
        e_field_probes = self._instruments_by_class(
            ins.FieldProbe, self.pick_ups, probes_to_ignore
        )
        assert self.global_diagnostics is not None

        forward_power = self.get_instrument(ins.ForwardPower)
        reflection = self.get_instrument(ins.ReflectionCoefficient)

        reconstructed = ins.Reconstructed(
            name=name,
            raw_data=None,
            e_field_probes=e_field_probes,
            forward_power=forward_power,
            reflection=reflection,
            freq_mhz=self.freq_mhz,
        )
        reconstructed.fit_voltage()

        self.global_diagnostics.add_instrument(reconstructed)

        return

    def data_for_somersalo(
        self,
        test_multipactor_bands: TestMultipactorBands,
    ) -> dict[str, float | list[float]]:
        """Get the data required to create the Somersalo plot.

        .. todo::
            Allow representation of several pick-ups.

        """
        last_powers = self.at_last_threshold(
            ins.ForwardPower, test_multipactor_bands
        ).iloc[0]
        z_ohm = 50.0
        d_mm = 0.5 * (38.78 - 16.87)
        logging.warning(f"Used default {d_mm = }")
        somersalo_data = {
            "powers_kw": [
                last_powers.iloc[0] * 1e-3,
                last_powers.iloc[1] * 1e-3,
            ],
            "z_ohm": z_ohm,
            "d_mm": d_mm,
            "freq_ghz": self.freq_mhz * 1e-3,
        }
        return somersalo_data

    def data_for_somersalo_scaling_law(
        self,
        multipactor_bands: TestMultipactorBands | InstrumentMultipactorBands,
        use_theoretical_r: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Get the data necessary to plot the Somersalo scaling law.

        In particular, the power thresholds measured during the last half power
        cycle, and the reflection coefficient :math:`R` at the corresponding
        time steps. Lower and upper thresholds are returned, even if Somersalo
        scaling law does not concern the upper threshold.

        Parameters
        ----------
        multipactor_bands :
            Object telling where multipactor happens. If it is a
            :class:`.TestMultipactorBands`, we merge all the
            :class:`.InstrumentMultipactorBands` in it, to know where the first
            (``several_bands_politics='keep_first'``) multipactor happened,
            anywhere in the testbench (``union='relaxed'``). You can also
            provide directly an :class:`.InstrumentMultipactorBands`; we will
            take its last :class:`.MultipactorBand`.
        use_theoretical_r :
            If set to True, we return the :math:`R` corresponding to the
            user-defined :math:`SWR`.
        kwargs :
            Other keyword arguments passed to :meth:`.at_last_threshold`.

        Returns
        -------
        data :
            Holds the lower and upper :math:`P_f` during last half power cycle,
            as well as reflection coefficient :math:`R` at same time steps.

        """
        if isinstance(multipactor_bands, TestMultipactorBands):
            multipactor_bands = multipactor_bands.merge(
                union="relaxed",
                info_test=str(self),
                several_bands_politics="keep_lowest",
            )

        instruments = ins.ForwardPower, ins.ReflectionCoefficient
        df_somersalo = self.at_last_threshold(
            instruments, multipactor_bands, **kwargs
        )

        if use_theoretical_r:
            if np.isinf(self.swr):
                reflection_coeff = 1.0
            else:
                reflection_coeff = (self.swr - 1.0) / (self.swr + 1.0)
            cols = df_somersalo.filter(like="ReflectionCoefficient").columns
            df_somersalo[cols] = reflection_coeff

        return df_somersalo

    def output_filepath(self, out_folder: str, extension: str) -> Path:
        """Create consistent path for output files."""
        filepath = output_filepath(
            self.filepath, self.swr, self.freq_mhz, out_folder, extension
        )
        return filepath
