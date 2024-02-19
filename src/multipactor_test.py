#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

.. todo::
    Find a way to check consistency between the MultipactorBands and
    corresponding Instruments. First idea would be to check their respective
    positions.

"""
import itertools
from abc import ABCMeta
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from multipac_testbench.src.instruments.electric_field.field_probe import \
    FieldProbe
from multipac_testbench.src.instruments.electric_field.reconstructed import \
    Reconstructed
from multipac_testbench.src.instruments.instrument import Instrument
from multipac_testbench.src.instruments.powers import Powers
from multipac_testbench.src.measurement_point.factory import \
    IMeasurementPointFactory
from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint
from multipac_testbench.src.multipactor_band.multipactor_bands import \
    MultipactorBands
from multipac_testbench.src.multipactor_band.util import match_with_mp_band
from multipac_testbench.src.util import plot
from multipac_testbench.src.util.animate import get_limits
from multipac_testbench.src.util.helper import output_filepath


class MultipactorTest:
    """Holds a mp test with several probes."""

    def __init__(self,
                 filepath: Path,
                 config: dict,
                 freq_mhz: float,
                 swr: float,
                 info: str = '',
                 sep: str = '\t',
                 verbose: bool = False,
                 ) -> None:
        r"""Create all the pick-ups.

        Parameters
        ----------
        filepath : Path
            Path to the results file produced by LabViewer.
        config : dict
            Configuration ``.toml`` of the testbench.
        freq_mhz : float
            Frequency of the test in :math:\mathrm{MHz}:
        swr : float
            Expected Voltage Signal Wave Ratio.
        info : str, optional
            An additional string to identify this test in plots.
        sep : str
            Delimiter between two columns in ``filepath``.
        verbise : bool, optional
            To print information on the structure of the test bench, as it was
            understood. The default is False.

        """
        self.filepath = filepath
        df_data = pd.read_csv(filepath, sep=sep, index_col="Sample index")
        self._n_points = len(df_data)
        self.df_data = df_data

        if df_data.index[0] != 0:
            print("MultipactorTest.__init__ warning! Your Sample index column "
                  "does not start at 0. I should patch this, but meanwhile "
                  " expect some index mismatches.")

        imeasurement_point_factory = IMeasurementPointFactory()
        imeasurement_points = imeasurement_point_factory.run(config,
                                                             df_data,
                                                             verbose)
        self.global_diagnostics, self.pick_ups = imeasurement_points

        self.freq_mhz = freq_mhz
        self.swr = swr
        self.info = info

    def __str__(self) -> str:
        """Print info on object."""
        out = [f"{self.freq_mhz}MHz", f"SWR {self.swr}"]
        if len(self.info) > 0:
            out.append(f"{self.info}")
        return ', '.join(out)

    def add_post_treater(self,
                         *args,
                         only_pick_up_which_name_is: tuple[str, ...] = (),
                         **kwargs) -> None:
        """Add post-treatment functions to instruments."""
        pick_ups = self.pick_ups
        if len(only_pick_up_which_name_is) > 0:
            pick_ups = [pick_up for pick_up in self.pick_ups
                        if pick_up.name in only_pick_up_which_name_is]

        for pick_up in pick_ups:
            pick_up.add_post_treater(*args, **kwargs)

    def detect_multipactor(
            self,
            multipac_detector: Callable[[np.ndarray], np.ndarray[np.bool_]],
            instrument_class: ABCMeta,
            power_is_growing_kw: dict[str, int | float] | None = None,
            measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (
            ),
    ) -> list[MultipactorBands]:
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

        Returns
        -------
        detected_multipactor_bands : list[MultipactorBands]
            Objets containing when multipactor happens, according to
            ``multipac_detector``, at every pick-up holding an
            :class:`.Instrument` of type ``instrument_class``.

        """
        powers = self.get_instrument(Powers)

        detected_multipactor_bands = []
        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)
        for measurement_point in measurement_points:
            multipactor_bands = measurement_point.detect_multipactor(
                multipac_detector,
                instrument_class
            )
            # if not hasattr(measurement_point, 'multipactor_bands'):
            if multipactor_bands is None:
                continue
            if not isinstance(powers, Powers):
                continue

            if power_is_growing_kw is None:
                power_is_growing_kw = {}
            power_is_growing = powers.where_is_growing(**power_is_growing_kw)
            multipactor_bands.power_is_growing = power_is_growing
            detected_multipactor_bands.append(multipactor_bands)
        return detected_multipactor_bands

    def plot_instruments_vs_time(
        self,
        instruments_class_to_plot: Sequence[ABCMeta],
        measurement_points_to_exclude: Sequence[str | IMeasurementPoint] = (),
        png_path: Path | None = None,
        raw: bool = False,
        plot_multipactor: bool = False,
        multipactor_bands: Sequence[MultipactorBands] | None = None,
        **fig_kw,
    ) -> tuple[Figure, list[Axes]]:
        """Plot signals measured by ``instruments_to_plot``.

        .. todo::
            Add a ``instruments_to_exclude`` argument. Could replace
            ``measurement_points_to_exclude``.

        Parameters
        ----------
        instruments_to_plot : tuple[ABCMeta, ...]
            Subclass of the :class:`.Instrument` to plot.
        measurement_points_to_exclude : tuple[str, ...], optional
            Name of the measurement points that should not be plotted. The
            default is an empty tuple.
        png_path : Path | None, optional
            If provided, the resulting figure is saved at this path. The
            default is None.
        raw : bool, optional
            If the data that should be plotted is the raw data before
            post-treatment. The default is False. Note that when the
            :attr:`.Instrument.post_treaters` list is empty, raw data is
            plotted even if ``raw==True``.
        multipactor_plots : bool, optional
            To add arrows to detect multipactor. The default is False.
        fig_kw :
            Keyword arguments passed to the ``Figure``.

        Returns
        -------
        fig : Figure
            The created figure.
        axes : Axes
            The created axes.

        """
        fig, instrument_class_axes = plot.create_fig(
            str(self),
            instruments_class_to_plot,
            xlabel='Measurement index',
            **fig_kw)

        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)

        if multipactor_bands is None or not plot_multipactor:
            multipactor_bands = [None for _ in measurement_points]
            zipper = zip(measurement_points, multipactor_bands, strict=True)
            # zipper = zip(measurement_points, multipactor_bands, strict=True)
        else:
            zipper = match_with_mp_band(
                measurement_points,
                multipactor_bands,
                assert_positions_match=True,
                find_matching_pairs=True,
                assert_every_obj_has_multipactor_bands=False,
            )

        for measurement_point, mp_bands in zipper:
            measurement_point.plot_instruments_vs_time(
                instrument_class_axes,
                instruments_class_to_plot,
                raw=raw)

            if mp_bands is not None:
                self._add_multipactor_vs_time(measurement_point,
                                              instrument_class_axes,
                                              mp_bands)

        plot.finish_fig(fig, instrument_class_axes.values(), png_path)
        return fig, [axes for axes in instrument_class_axes.values()]

    def _add_multipactor_vs_time(
        self,
        measurement_point: IMeasurementPoint,
        instrument_class_axes: dict[ABCMeta, Axes],
        multipactor_bands: MultipactorBands,
    ) -> None:
        """Show with arrows when multipactor happens.

        Parameters
        ----------
        measurement_point : IMeasurementPoint
            :class:`.PickUp` or :class:`.GlobalDiagnostic` under study.
        instrument_class_axes : dict[ABCMeta, Axes]
            Links instrument class with the axes.
        multipactor_bands : MultipactorBands
            Should correspond to the :class:`IMeasurementPoint` under study.

        """
        for plotted_instrument_class, axe in instrument_class_axes.items():
            measurement_point._add_multipactor_vs_time(
                axe,
                plotted_instrument_class,
                multipactor_bands)

    def animate_instruments_vs_position(
            self,
            instruments_to_plot: Sequence[ABCMeta],
            gif_path: Path | None = None,
            fps: int = 50,
            keep_one_frame_over: int = 1,
            interval: int | None = None,
            **fig_kw,
    ) -> animation.FuncAnimation:
        """Represent measured signals with probe position."""
        fig, axes_instruments = self._prepare_animation_fig(
            instruments_to_plot,
            **fig_kw
        )

        frames = self._n_points - 1
        artists = self._plot_instruments_single_time_step(
            0,
            keep_one_frame_over=keep_one_frame_over,
            axes_instruments=axes_instruments,
            artists=None,
        )

        def update(step_idx: int) -> Sequence[Artist]:
            """Update the ``artists`` defined in outer scope.

            Parameters
            ----------
            step_idx : int
                Step that shall be plotted.

            Returns
            -------
            artists : Sequence[Artist]
                Updated artists.

            """
            self._plot_instruments_single_time_step(
                step_idx,
                keep_one_frame_over=keep_one_frame_over,
                axes_instruments=axes_instruments,
                artists=artists,
            )
            assert artists is not None
            return artists

        if interval is None:
            interval = int(200 / keep_one_frame_over)

        ani = animation.FuncAnimation(fig,
                                      update,
                                      frames=frames,
                                      interval=interval,
                                      repeat=True)

        if gif_path is not None:
            writergif = animation.PillowWriter(fps=fps)
            ani.save(gif_path, writer=writergif)
        return ani

    def _plot_instruments_single_time_step(
            self,
            step_idx: int,
            keep_one_frame_over: int,
            axes_instruments: dict[Axes, list[Instrument]],
            artists: Sequence[Artist] | None = None,
    ) -> Sequence[Artist] | None:
        """Plot all instruments signal at proper axe and time step."""
        if step_idx % keep_one_frame_over != 0:
            return

        sample_index = step_idx + 1

        if artists is None:
            artists = [instrument.plot_vs_position(sample_index, axe=axe)
                       for axe, instruments in axes_instruments.items()
                       for instrument in instruments]
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
        multipactor_bands: Sequence[MultipactorBands] | None = None,
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
        fig, instrument_class_axes = plot.create_fig(str(self),
                                                     instruments_to_plot,
                                                     xlabel='Probe index',
                                                     **fig_kw)
        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)

        multipactor_bands = self._get_proper_multipactor_bands(
            multipactor_measured_at=measurement_points,
            multipactor_bands=multipactor_bands,
            measurement_points_to_exclude=measurement_points_to_exclude)

        for i, measurement_point in enumerate(measurement_points):
            measurement_point.scatter_instruments_data(instrument_class_axes,
                                                       xdata=float(i),
                                                       )

        fig, axes = plot.finish_fig(fig,
                                    instrument_class_axes.values(),
                                    png_path)
        return fig, axes

    def _instruments_by_class(
            self,
            instrument_class: ABCMeta,
            measurement_points: Sequence[IMeasurementPoint] | None = None,
            instruments_to_ignore: Sequence[Instrument | str] = (),
    ) -> list[Instrument]:
        """Get all instruments of desired class from ``measurement_points``.

        But remove the instruments to ignore.

        Parameters
        ----------
        instrument_class : ABCMeta
            Class of the desired instruments.
        measurement_points : Sequence[IMeasurementPoint] | None, optional
            The measurement points from which you want the instruments. The
            default is None, in which case we look into every
            :class:`IMeasurementPoint` attribute of self.
        instruments_to_ignore : Sequence[Instrument | str], optional
            The :class:`.Instrument` or instrument names you do not want. The
            default is an empty tuple, in which case no instrument is ignored.

        Returns
        -------
        instruments : list[Instrument]
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
        instruments = [instrument
                       for instrument_1d in instruments_2d
                       for instrument in instrument_1d]
        return instruments

    def _instruments_by_name(
            self,
            instrument_names: Sequence[str],
    ) -> list[Instrument]:
        """Get all instruments of desired name from ``measurement_points``.

        But remove the instruments to ignore.

        Parameters
        ----------
        instrument_name : Sequence[str]
            Name of the desired instruments.

        Returns
        -------
        instruments : list[Instrument]
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
            print("MultipactorTest._instruments_by_name warning: ",
                  f"you asked for {instrument_names = }, I give you ",
                  f"{[instr.name for instr in instruments]} which has a ",
                  "different length.")
        return instruments

    def get_measurement_points(
        self,
        names: Sequence[str] | None = None,
        to_exclude: Sequence[str | IMeasurementPoint] = (),
    ) -> Sequence[IMeasurementPoint]:
        """Get all or some measurement points.

        Parameters
        ----------
        names : Sequence[str], optional
            If given, only the :class:`.IMeasurementPoint` which name is in
            ``names`` will be returned.
        to_exclude : Sequence[str | IMeasurementPoint], optional
            List of objects or objects names to exclude from returned list.

        Returns
        -------
        i_measurement_points : Sequence[IMeasurementPoint]
            The desired objects.

        """
        names_to_exclude = [x if isinstance(x, str) else x.name
                            for x in to_exclude]

        measurement_points = [
            x for x in self.pick_ups + [self.global_diagnostics]
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
        name : str | None, optional
            If given, only the :class:`.IMeasurementPoint` which name is in
            ``names`` will be returned.
        to_exclude : Sequence[str | IMeasurementPoint], optional
            List of objects or objects names to exclude from returned list.

        Returns
        -------
        measurement_point : IMeasurementPoint
            The desired object.

        """
        if name is not None:
            name = name,
        measurement_points = self.get_measurement_points(name, to_exclude)
        assert len(measurement_points) == 1, ("Only one IMeasurementPoint "
                                              "should match.")
        return measurement_points[0]

    def get_instruments(
            self,
            instruments_id: ABCMeta | Sequence[ABCMeta] | Sequence[str] | Sequence[Instrument],
            measurement_points_to_exclude: Sequence[IMeasurementPoint
                                                    | str] = (),
            instruments_to_ignore: Sequence[Instrument | str] = (),
    ) -> list[Instrument]:
        """Get all instruments matching ``instrument_id``."""
        match (instruments_id):
            case list() | tuple() as instruments if types_match(instruments,
                                                                Instrument):
                return instruments

            case list() | tuple() as names if types_match(names, str):
                out = self._instruments_by_name(names)

            case list() | tuple() as classes if types_match(classes, ABCMeta):
                measurement_points = self.get_measurement_points(
                    to_exclude=measurement_points_to_exclude)
                out_2d = [self._instruments_by_class(
                    instrument_class,
                    measurement_points,
                    instruments_to_ignore=instruments_to_ignore)
                    for instrument_class in classes]
                out = list(itertools.chain.from_iterable(out_2d))

            case ABCMeta() as instrument_class:
                measurement_points = self.get_measurement_points(
                    to_exclude=measurement_points_to_exclude)
                out = self._instruments_by_class(
                    instrument_class,
                    measurement_points,
                    instruments_to_ignore=instruments_to_ignore)
            case _:
                raise IOError(f"instruments is {type(instruments_id)} which is ",
                              "not supported.")
        return out

    def get_instrument(
            self,
            instrument_id: ABCMeta | str | Instrument,
            measurement_points_to_exclude: Sequence[IMeasurementPoint
                                                    | str] = (),
            instruments_to_ignore: Sequence[Instrument | str] = (),
    ) -> Instrument | None:
        """Get a single instrument matching ``instrument_id``."""
        match (instrument_id):
            case Instrument():
                return instrument_id
            case str() as instrument_name:
                instruments = self.get_instruments((instrument_name, ))
            case ABCMeta() as instrument_class:
                instruments = self.get_instruments(
                    instrument_class,
                    measurement_points_to_exclude,
                    instruments_to_ignore)

        if len(instruments) == 0:
            raise IOError("No instrument found.")
        if len(instruments) > 1:
            print("multipactor_test.get_instrument warning! Several "
                  "instruments found. Returning first one.")
        return instruments[0]

    def _prepare_animation_fig(
        self,
        instruments_to_plot: Sequence[ABCMeta],
        measurement_points_to_exclude: tuple[str, ...] = (),
        instruments_to_ignore_for_limits: tuple[str, ...] = (),
        instruments_to_ignore: Sequence[Instrument | str] = (),
        **fig_kw,
    ) -> tuple[Figure, dict[Axes, list[Instrument]]]:
        """Prepare the figure and axes for the animation.

        Parameters
        ----------
        instruments_to_plot : tuple[ABCMeta, ...]
            Classes of instruments you want to see.
        measurement_points_to_exclude : tuple[str, ...]
            Measurement points that should not appear.
        instruments_to_ignore_for_limits : tuple[str, ...]
            Instruments to plot, but that can go off limits.
        instruments_to_ignore : Sequence[Instrument | str]
            Instruments that will not even be plotted.
        fig_kw :
            Other keyword arguments for Figure.

        Returns
        -------
        fig : Figure
         Figure holding the axes.
        axes_instruments : dict[Axes, list[Instrument]]
            Links the instruments to plot with the Axes they should be plotted
            on.

        """
        fig, instrument_class_axes = plot.create_fig(str(self),
                                                     instruments_to_plot,
                                                     xlabel='Position [m]',
                                                     **fig_kw)

        for instrument_class, axe in instrument_class_axes.items():
            axe.set_ylabel(instrument_class.ylabel())

        measurement_points = self.get_measurement_points(
            to_exclude=measurement_points_to_exclude)

        axes_instruments = {
            axe: self._instruments_by_class(
                instrument_class,
                measurement_points,
                instruments_to_ignore=instruments_to_ignore)
            for instrument_class, axe in instrument_class_axes.items()
        }

        y_limits = get_limits(axes_instruments,
                              instruments_to_ignore_for_limits)
        axe = None
        for axe, y_lim in y_limits.items():
            axe.set_ylim(y_lim)

        return fig, axes_instruments

    def _get_limits(
            self,
            axes_instruments: dict[Axes, Sequence[Instrument]],
            instruments_to_ignore_for_limits: Sequence[Instrument | str] = (),
    ) -> dict[Axes, tuple[float, float]]:
        """Get limits of demanded instruments.

        .. note::
            Currently not used, not to be used. ``self.df_data`` is not
            synchronized with the ``ydata`` from the instruments.

        """
        names_to_ignore = [x if isinstance(x, str) else x.name
                           for x in instruments_to_ignore_for_limits]
        limits = {}
        for axe, instruments in axes_instruments.items():
            names = [instrument.name for instrument in instruments
                     if instrument.name not in names_to_ignore
                     and not instrument.is_2d]
            df_data = self.df_data[names]
            lower = df_data.min(axis=None)
            upper = df_data.max(axis=None)
            amplitude = lower - upper
            limits[axe] = (lower - .1 * amplitude, upper + .1 * amplitude)
        return limits

    def reconstruct_voltage_along_line(
            self,
            name: str,
            probes_to_ignore: Sequence[str | FieldProbe] = (),
    ) -> None:
        """Reconstruct the voltage profile from the e field probes."""
        e_field_probes = self._instruments_by_class(FieldProbe,
                                                    self.pick_ups,
                                                    probes_to_ignore)
        assert self.global_diagnostics is not None
        powers = self.get_instrument(Powers)

        reconstructed = Reconstructed(
            name=name,
            raw_data=None,
            e_field_probes=e_field_probes,
            powers=powers,
            freq_mhz=self.freq_mhz,
        )
        reconstructed.fit_voltage()

        self.global_diagnostics.add_instrument(reconstructed)

        return

    def plot_data_at_multipactor_thresholds(
        self,
        instruments_id_plot: ABCMeta | Sequence[Instrument] | Sequence[str],
        multipactor_bands: MultipactorBands | Sequence[MultipactorBands],
        measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (),
        instruments_to_ignore: Sequence[Instrument | str] = (),
        png_path: Path | None = None,
        **fig_kw,
    ) -> tuple[Figure, Axes]:
        """Plot the data measured by some instruments at thresholds.

        .. todo::
            Docstring, match_with_mp_band keywords

        New version of `plot_multipactor_limits`.

        """
        instruments_to_plot = self.get_instruments(
            instruments_id_plot,
            measurement_points_to_exclude,
            instruments_to_ignore)

        if isinstance(multipactor_bands, MultipactorBands):
            multipactor_bands = [multipactor_bands
                                 for _ in instruments_to_plot]

        instrument_types = list(types(instruments_to_plot))
        fig, instrument_class_axes = plot.create_fig(
            str(self),
            instrument_types,
            xlabel="Measurement index",
            **fig_kw
        )
        axe = [axe for axe in instrument_class_axes.values()][0]

        zipper = match_with_mp_band(instruments_to_plot,
                                    multipactor_bands,
                                    assert_positions_match=True,
                                    find_matching_pairs=False,
                                    # should already match
                                    )
        for instrument, mp_bands in zipper:
            lower_values, upper_values = instrument.values_at_barriers(
                mp_bands)
            lower_values.plot(ax=axe, kind='line', drawstyle='steps-post')
            color = axe.get_lines()[-1].get_color()
            upper_values.plot(ax=axe,
                              kind='line',
                              drawstyle='steps-post',
                              color=color,
                              ls='--')
        axe.grid(True)
        plot.finish_fig(fig, instrument_class_axes.values(), png_path)
        return fig, axe

    def data_for_somersalo(self,
                           multipactor_bands: MultipactorBands,
                           ) -> dict[str, float | list[float]]:
        """Get the data required to create the Somersalo plot.

        .. todo::
            Allow representation of several pick-ups.

        """
        powers = self.get_instrument(Powers)
        assert powers is not None
        last_powers = powers.values_at_barriers_fully_conditioned(
            multipactor_bands)

        z_ohm = 50.
        d_mm = .5 * (38.78 - 16.87)
        print("MultipactorTest.data_for_somersalo warning! Used default "
              f"{d_mm = }")
        somersalo_data = {
            'powers_kw': [last_powers[0][0] * 1e-3, last_powers[1][0] * 1e-3],
            'z_ohm': z_ohm,
            'd_mm': d_mm,
            'freq_ghz': self.freq_mhz * 1e-3,
        }
        return somersalo_data

    def data_for_susceptibility(
            self,
            electric_field_at: IMeasurementPoint | str,
            multipactor_bands: MultipactorBands,
    ) -> dict[str, float | list[float]]:
        """Get the data required to create the susceptibility plot.

        .. todo::
            Allow representation of several pick-ups.

        """
        if isinstance(electric_field_at, str):
            electric_field_at = self.get_measurement_point(
                electric_field_at)
        electric_field = electric_field_at.get_instrument(FieldProbe)
        assert electric_field is not None
        last_fields = electric_field.values_at_barriers_fully_conditioned(
            multipactor_bands)

        d_mm = .5 * (38.78 - 16.87)
        print("MultipactorTest.data_for_susceptibility warning! Used default "
              f"{d_mm = }")
        somersalo_data = {
            'voltages_v': [last_fields[0], last_fields[1]],
            'd_cm': d_mm * 1e-1,
            'freq_mhz': self.freq_mhz,
        }
        return somersalo_data

    def data_for_somersalo_scaling_law(self,
                                       multipactor_bands: MultipactorBands,
                                       use_theoretical_r: bool = False,
                                       ) -> pd.Series:
        """Get the data necessary to plot the Somersalo scaling law.

        .. todo::
            Proper docstring.

        """
        powers = self.get_instrument(Powers)
        assert isinstance(powers, Powers)

        last_low_idx = multipactor_bands[-1][-1]
        reflection_coeff = powers.gamma[last_low_idx]
        if use_theoretical_r:
            if np.isinf(self.swr):
                reflection_coeff = 1.
            else:
                reflection_coeff = (self.swr - 1.) / (self.swr + 1.)
        last_forward_power = powers.forward[last_low_idx]
        ser = pd.Series(
            {'$R$': reflection_coeff,
             r'Lower multipactor threshold $P_{f, low}$': last_forward_power,
             }
        )
        return ser

    def data_for_perez(self,
                       multipactor_bands: Sequence[MultipactorBands],
                       measurement_points_to_exclude: Sequence[str | IMeasurementPoint] = (
                       ),
                       probes_conditioned_during_test: Sequence[str] = (),
                       ) -> pd.Series:
        """Get the data necessary to check if Perez was right.

        .. todo::
            Proper docstring.

        .. todo::
            There are way cleaner ways to do this.

        .. todo::
            If multipactor happens somewhere and then is conditioned, this
            information does not appear and we plot the last detected
            multipactor.
            Hence the dirty patch probes_conditioned_during_test.

        """
        field_probes = self.get_instruments(
            FieldProbe,
            measurement_points_to_exclude=measurement_points_to_exclude)
        zipper = match_with_mp_band(field_probes,
                                    multipactor_bands,
                                    assert_positions_match=True,
                                    find_matching_pairs=False)

        v_thresholds = {}
        for field_probe, mp_band in zipper:
            if mp_band is None or len(mp_band) == 0:
                v_thresholds[field_probe.name + " low"] = np.NaN
                v_thresholds[field_probe.name + " high"] = np.NaN
                continue

            last_multipactor_band = mp_band[-1]

            # Dirty patch
            if field_probe.name in probes_conditioned_during_test:
                last_multipactor_band = None

            if last_multipactor_band is None:
                v_thresholds[field_probe.name + " low"] = np.NaN
                v_thresholds[field_probe.name + " high"] = np.NaN
                continue

            last_multipac_idx = last_multipactor_band[-1]
            last_lower_threshold = field_probe.ydata[last_multipac_idx]
            v_thresholds[field_probe.name + " low"] = last_lower_threshold

            if not last_multipactor_band.upper_threshold_was_reached:
                last_higher_threshold = np.NaN
            else:
                first_multipac_idx = last_multipactor_band[0]
                last_higher_threshold = field_probe.ydata[first_multipac_idx]

            v_thresholds[field_probe.name + " high"] = last_higher_threshold

        tmp_str = r"$SWR_{theor.}$"
        name = f"{tmp_str} = {self.swr}"
        df_thresholds = pd.Series(v_thresholds, name=name)
        return df_thresholds

    def plot_instruments_y_vs_instrument_x(
            self,
            instrument_ids_x: Sequence[ABCMeta] | Sequence[str] | Sequence[Instrument],
            instrument_ids_y: Sequence[ABCMeta] | Sequence[str] | Sequence[Instrument],
            measurement_points_to_exclude: Sequence[IMeasurementPoint | str] = (
            ),
            instruments_to_ignore: Sequence[Instrument | str] = (),
            tail: int = -1,
            fig_kw: dict | None = None,
    ) -> Axes:
        """Plot data measured by ``instrument_y`` vs ``instrument_x``."""
        instruments_y = self.get_instruments(instrument_ids_y,
                                             measurement_points_to_exclude,
                                             instruments_to_ignore)

        instruments_x = self.get_instruments(instrument_ids_x,
                                             measurement_points_to_exclude,
                                             instruments_to_ignore)
        if len(instruments_x) == 1:
            instruments_x = [instruments_x[0] for _ in instruments_y]

        zipper = zip(instruments_x, instruments_y, strict=True)

        if fig_kw is None:
            fig_kw = {}

        axes = None
        for x, y in zipper:
            dict_to_plot = {x.name: x.ydata_as_pd,
                            y.name: y.ydata_as_pd}

            df_to_plot = pd.DataFrame(dict_to_plot)

            axes = df_to_plot.tail(tail).plot(x=0,
                                              xlabel=x.ylabel(),
                                              ylabel=y.ylabel(),
                                              ax=axes,
                                              grid=True,
                                              )
        assert axes is not None
        return axes

    def output_filepath(self, out_folder: str, extension: str) -> Path:
        """Create consistent path for output files."""
        filepath = output_filepath(self.filepath,
                                   self.swr,
                                   self.freq_mhz,
                                   out_folder,
                                   extension)
        return filepath


def types(my_list: Sequence) -> set[type]:
    """Get all different types in given list."""
    return set(type(x) for x in my_list)


def types_match(my_list: Sequence, to_match: type) -> bool:
    """Check if all elements of ``my_list`` have type ``type``."""
    return types(my_list) == {to_match}
