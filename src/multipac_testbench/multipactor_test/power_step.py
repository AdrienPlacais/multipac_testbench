"""Define an object corresponding to a power step file."""

import functools
import logging
import random
from abc import ABCMeta
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from multipac_testbench.instruments import Penning, Power, Sync, Trigger
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.multipactor_test import MultipactorTest
from multipac_testbench.multipactor_test.helper import (
    POWERSTEP_FILE_RECOGNIZER_T,
    default_powerstep_file_valider,
    powerstep_files,
    take_maximum,
    take_median,
)
from multipac_testbench.multipactor_test.loader import save
from multipac_testbench.multipactor_test.reduction_info import ReductionInfo
from multipac_testbench.threshold.threshold_set import ThresholdSet
from multipac_testbench.util.files import load_config
from multipac_testbench.util.log_manager import suppress_log_messages
from multipac_testbench.util.types import REDUCER_T
from numpy.typing import NDArray

#: For :class:`.Instrument` for which taking the median over the trigger window
#: does not make sense.
#: For :class:`.Penning`, this is because the signal takes time to rise and the
#: main info lies after the trigger window.
#: For :class:`.Power`, this is because the signals are delayed wrt to the
#: trigger window.
_DEFAULT_SPECIAL_REDUCERS: dict[str | type[Instrument], REDUCER_T] = {
    Penning: take_maximum,
    Power: take_maximum,
}


class PowerStep(MultipactorTest):
    """This object is basically a MultipactorTest. But for one power step."""

    #: Log messages to suppress, as they are very noisy in this context.
    log_messages_to_suppress = [
        "points were removed in R calculation, where reflected power was ",
        "column_header = 'NI9205_dBm' not present in provided file. Skipping",
        "Applied trigger_policy = ",
        "Adding a post_treater to ",
        "not present in provided file. Skipping associated instrument",
    ]

    def __init__(
        self,
        filepath: Path,
        config: dict[str, Any] | str | Path,
        sample_index: int,
        freq_mhz: float | None = None,
        swr: float = 1.0,
        info: str = "",
        sep: str = "\t",
        index_col: str = "Index",
        out_index_col: str = "Sample index",
        comment: str = "#",
        create_virtual_instruments: bool = True,
        **kwargs,
    ) -> None:
        """Create object like if it was a :class:`.MultipactorTest`.

        The differences are:

        - ``index_col`` is by default ``"Index"``, like in the ``MV`` files.
        - ``trigger_policy`` is always ``"keep_all"``, as other values would be
          meaningless.
        - ``remove_metadata_columns`` is always True, as the rightmost metadata
          columns hold strings, messing up with the ``REDUCER_T`` funcs.

        Keys such as ``freq_mhz`` or ``swr`` are not used to create
        :class:`.MultipactorTest` files; however the let you perform
        :meth:`PowerStep.sweet_plot`.

        Parameters
        ----------
        filepath :
            Path to the results file produced by LabViewer.
        config :
            Configuration ``TOML`` of the testbench.
        sample_index :
            Index of power step.
        freq_mhz :
            Frequency of the test in :unit:`MHz`.
        swr :
            Expected Voltage Signal Wave Ratio.
        info :
            An additional string to identify this test in plots.
        sep :
            Delimiter between two columns in ``filepath``.
        index_col :
            Name of the column holding index data.
        out_index_col :
            Where to store ``sample_index`` in the output file.
        comment :
            Comment character.
        create_virtual_instruments :
            If virtual instruments should be created.
        kwargs :
            Other kwargs passed to :func:`.load`.

        """
        config = config if isinstance(config, dict) else load_config(config)
        with suppress_log_messages("", self.log_messages_to_suppress):
            super().__init__(
                filepath=filepath,
                config=config,
                freq_mhz=freq_mhz,
                swr=swr,
                info=f"Sample index #{sample_index}" + info,
                sep=sep,
                index_col=index_col,
                trigger_policy="keep_all",
                remove_metadata_columns=True,
                comment=comment,
                create_virtual_instruments=create_virtual_instruments,
                **kwargs,
            )
        #: Position of the step in the complete :class:`.MultipactorTest`
        self._sample_index = sample_index
        self._out_index_col = out_index_col

        sync = self.get_instrument(Sync, raise_missing_error=False)
        if sync is None:
            return

        assert isinstance(sync, Sync)
        trigger_instrument = Trigger.from_sync(
            n_points=self._n_points, sync=sync
        )
        if self.global_diagnostics is not None:
            self.global_diagnostics.add_instrument(trigger_instrument)
        self.test_conditions.trigger = int(sync.trigger)

    def to_single_values(
        self,
        generic_reducer: REDUCER_T | None = None,
        special_reducers: (
            Mapping[str | type[Instrument], REDUCER_T] | None
        ) = None,
        operate_on_raw_data: bool = True,
    ) -> pd.Series:
        """Convert arrays of :class:`.Instrument` values to single floats.

        Parameters
        ----------
        generic_reducer :
            Function converting array to float. The default is
            :func:`.take_median` applied to the pulse window ``[pre_trigger,
            pre_trigger + trigger]`` as read from :attr:`test_conditions`.
            Falls back to :func:`.take_maximum` on the full array if those
            values are unavailable, and logs an error.
        special_reducers :
            Different functions to apply to specific columns. Keys can be an
            instrument name string (e.g. ``"NI9205_Penning1"``) for a single
            instrument, or an instrument class (e.g. :class:`.Penning`) to
            apply to all instruments of that type. String keys take priority
            over class keys. Defaults to ``{Penning: take_maximum, Power:
            take_maximum}``. If you really do not want to use defaults, provide
            an empty dict.

        Note
        ----
        As the synchronism of the watt-metre is bad, measured powers are
        shifted wrt NI9205 measurements. So you will generally want to take the
        maximum of ``NI9205_Power1`` and ``NI9205_Power2`` columns, which is
        handled by the default ``special_reducers``.

        """
        if generic_reducer is None:
            pre_trig = self.test_conditions.pre_trigger
            trigger = self.test_conditions.trigger
            if pre_trig is None or trigger is None:
                logging.error(
                    f"{self}: pre_trigger or trigger is None; "
                    "falling back to take_maximum on the full array."
                )
                generic_reducer = take_maximum
            else:
                generic_reducer = functools.partial(
                    take_median,
                    first_index=pre_trig,
                    last_index=pre_trig + trigger,
                )
        if special_reducers is None:
            special_reducers = _DEFAULT_SPECIAL_REDUCERS

        def dispatch(instrument: Instrument) -> float:
            """Compute proper reducer for given instrument.

            First, we check if the name of the :class:`.Instrument` is present
            in ``special_reducers``. If not, we check if its type is present in
            ``special_reducers``. If not, we fall back on the generic reducer.

            The generic reducer is the median over the trigger window if
            ``pre_trigger`` and ``trigger`` are defined, else it is the maximum
            of the signal.

            Parameters
            ----------
            instrument :
                Instance which we want to reduce data to a single float.

            Returns
            -------
            float
                A single value representing measurement of the ``instrument``
                on a single :class:`.PowerStep`.

            """
            values = (
                instrument._raw_data.to_numpy()
                if operate_on_raw_data
                else instrument.data
            )
            name = instrument.name

            if name in special_reducers:
                actual_reducer = special_reducers[name]
            else:
                actual_reducer = generic_reducer

                for key, r in special_reducers.items():
                    if isinstance(key, type) and isinstance(instrument, key):
                        actual_reducer = r
                        break
            instrument.reduction_info = ReductionInfo.from_reducer(
                actual_reducer, operated_on_raw=operate_on_raw_data
            )
            return actual_reducer(values)

        all_data = {
            instrument.name: dispatch(instrument)
            for instrument in self.instruments
        }
        assert (
            self._out_index_col not in all_data
        ), f"{self._out_index_col} collides with an instrument name."
        all_data[self._out_index_col] = self._sample_index
        return pd.Series(all_data)

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
        threshold_set: ThresholdSet | None = None,
        global_instruments: bool = False,
        global_multipactor: bool = False,
        column_names: str | list[str] = "",
        test_color: str | None = None,
        png_path: Path | None = None,
        png_kwargs: dict | None = None,
        csv_path: Path | None = None,
        csv_kwargs: dict | None = None,
        axes: list[Axes] | None = None,
        masks: dict[str, NDArray[np.bool]] | None = None,
        drop_repeated_x: bool = False,
        pre_trig: int | None = None,
        trig: int | None = None,
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
        threshold_set :
            If provided, mark lower (circle) and upper (star) thresholds on top
            of every :class:`.Instrument` data.
        global_instruments :
            If instruments not position-specific (eg :class:`.ForwardPower`)
            should have their thresholds plotted.
        global_multipactor :
            If multipactor not position-specific (eg thresholds created by
            merging several other multipactor arrays) should have their
            thresholds plotted.
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
        csv_kwargs :
            Keyword arguments passed to :func:`.plot.save_dataframe`.
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
        pre_trig :
            Index at which the pulse should start. If both ``pre_trig`` and
            ``trig`` are provided, the times with power on are highlighted in
            red.
        trig :
            Pulse duration in indexes. If both ``pre_trig`` and ``trig`` are
            provided, the times with power on are highlighted in red.
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
        axes, df = super().sweet_plot(
            *ydata,
            xdata=xdata,
            exclude=exclude,
            tail=tail,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            title=title,
            threshold_set=threshold_set,
            global_instruments=global_instruments,
            global_multipactor=global_multipactor,
            column_names=column_names,
            test_color=test_color,
            png_path=png_path,
            png_kwargs=png_kwargs,
            csv_path=csv_path,
            csv_kwargs=csv_kwargs,
            axes=axes,
            masks=masks,
            drop_repeated_x=drop_repeated_x,
            **kwargs,
        )
        if pre_trig is not None and trig is not None:
            for ax in axes:
                ax.axvspan(
                    xmin=pre_trig,
                    xmax=pre_trig + trig,
                    facecolor="r",
                    alpha=0.1,
                )
        return axes, df

    @property
    def trigger(self) -> int | None:
        """Trigger sample index. Shorthand for ``test_conditions.trigger``."""
        return self.test_conditions.trigger


class PowerStepSet:
    """Define all the files constituting a :class:`.MultipactorTest`."""

    def __init__(
        self,
        folder: Path,
        config: dict[str, Any] | str | Path,
        freq_mhz: float | None = None,
        swr: float = 1.0,
        info: str = "",
        sep: str = "\t",
        index_col: str = "Index",
        out_index_col: str = "Sample index",
        file_recognizer: POWERSTEP_FILE_RECOGNIZER_T | None = None,
        comment: str = "#",
        create_virtual_instruments: bool = True,
        are_raw: bool = True,
        **kwargs,
    ) -> None:
        """Load all ``MV`` files in ``folder``, create :class:`.PowerStep`.

        Parameters
        ----------
        folder :
            Directory holding all the power step files of a test.
        config :
            Configuration file for the test.
        freq_mhz :
            Explicit frequency of the test in :unit:`MHz`. Prefer defining a
            :class:`.FrequencySetpoint` in the ``TOML``.
        swr :
            Expected Voltage Signal Wave Ratio.
        info :
            An additional string to identify this test in plots.
        sep :
            Column delimiter.
        index_col :
            Name of the column holding indexes in every power step file.
        out_index_col :
            Name of column where sample indexes will be stored.
        file_recognizer :
            Function taking in a filepath, and determining if it is a valid
            power step file. If not provided, set to
            :func:`.default_powerstep_file_valider`.
        comment :
            Comment delimiter, to skip the first lines in the source ``CSV``.
        create_virtual_instruments :
            If virtual instruments should be created.
        are_raw :
            Set to True if the ``CSV`` files in ``folder`` hold acquisition
            voltages, set to False if they hold physical quantities.

        """
        self._folder = folder
        self._freq_mhz = freq_mhz
        self._swr = swr
        self._info = info
        self._config: dict[str, Any] = (
            config if isinstance(config, dict) else load_config(config)
        )
        self._are_raw = are_raw
        file_recognizer = (
            file_recognizer
            if file_recognizer
            else default_powerstep_file_valider
        )
        file_index_mapping = powerstep_files(folder, file_recognizer)

        self._power_steps = [
            PowerStep(
                filepath=filepath,
                config=self._config,
                freq_mhz=self._freq_mhz,
                swr=self._swr,
                sample_index=sample_index,
                sep=sep,
                index_col=index_col,
                out_index_col=out_index_col,
                comment=comment,
                create_virtual_instruments=create_virtual_instruments,
                is_raw=are_raw,
                **kwargs,
            )
            for filepath, sample_index in file_index_mapping.items()
        ]
        if len(self) == 0:
            logging.warning(f"No valid file found in {folder}")

        some_power_steps = random.sample(self._power_steps, 10)
        triggers = [
            p.trigger for p in some_power_steps if p.trigger is not None
        ]
        if len(triggers) > 5:
            self._trigger = int(np.median(triggers))
        else:
            self._trigger = None
            logging.warning("Trigger value could not be calculated.")

    def __iter__(self) -> Iterator[PowerStep]:
        """Iterate over :class:`.PowerStep` objects.

        Yields
        ------
        PowerStep
            The stored :class:`.PowerSample` objects, sorted by sample index.

        """
        return iter(self._power_steps)

    def __str__(self) -> str:
        """Print out origin folder, number of loaded files."""
        return f"PowerStepSet holding {len(self)} files from {self._folder}"

    def __len__(self) -> int:
        """Get number of loaded files."""
        return len(self._power_steps)

    def get_power_step(self, sample_index: int) -> PowerStep:
        """Return the :class:`.PowerStep` with the given ``sample_index``.

        Parameters
        ----------
        sample_index :
            The index as stored in :attr:`.PowerStep._sample_index`.

        Raises
        ------
        KeyError
            If no :class:`.PowerStep` with that index exists.

        """
        for step in self._power_steps:
            if step._sample_index == sample_index:
                return step
        raise KeyError(f"No PowerStep with {sample_index=} in {self}")

    def to_dataframe(
        self,
        reducer: REDUCER_T | None = None,
        special_reducers: (
            Mapping[str | type[Instrument], REDUCER_T] | None
        ) = None,
        index_col: str = "Sample index",
    ) -> pd.DataFrame:
        """
        Reduce all :class:`.PowerStep` to single :class:`pandas.DataFrame`.

        Parameters
        ----------
        reducer :
            Passed to :meth:`.PowerStep.to_single_values`.
        special_reducers :
            Passed to :meth:`.PowerStep.to_single_values`.
        index_col :
            Name of the column holding sample indexes.

        Returns
        -------
            One row per :class:`.PowerStep`, one column per instrument.

        """
        series = (
            power_step.to_single_values(
                generic_reducer=reducer,
                special_reducers=special_reducers,
            )
            for power_step in sorted(self, key=lambda step: step._sample_index)
        )
        return pd.concat(series, axis=1).transpose().set_index(index_col)

    def to_multipactor_test_file(
        self,
        csv_path: Path,
        reducer: REDUCER_T | None = None,
        index_col: str = "Sample index",
        special_reducers: (
            Mapping[str | type[Instrument], REDUCER_T] | None
        ) = None,
        sep: str = ",",
        **kwargs,
    ) -> None:
        """Create a file that can be loaded by :class:`.MultipactorTest`.

        Parameters
        ----------
        power_steps :
            All the power steps of the file.
        csv_path :
            Where the resulting ``CSV`` will be stored.
        reducer :
            Function converting array to float. The default in LabViewer is to
            take the maximum. If not set, we also use this.
        index_col :
            Name of the column that will contain each power step index.
        special_reducers :
            Different functions to apply to some specific columns.
        sep :
            Column delimiter in the resulting file.
        **kwargs :
            Other keyword arguments passed down to
            :meth:`pandas.DataFrame.to_csv`.

        """
        df = self.to_dataframe(
            reducer=reducer,
            special_reducers=special_reducers,
            index_col=index_col,
        )
        save(csv_path, df, sep=sep, **kwargs)
        return

    def to_multipactor_test(
        self,
        csv_path: Path,
        reducer: REDUCER_T | None = None,
        special_reducers: (
            Mapping[str | type[Instrument], REDUCER_T] | None
        ) = None,
        **kwargs,
    ) -> MultipactorTest:
        """
        Write the summary ``CSV`` and load it as a :class:`.MultipactorTest`.

        Convenience wrapper around :meth:`to_multipactor_test_file` that
        additionally back-links the returned :class:`.MultipactorTest` to this
        :class:`.PowerStepSet`, enabling
        :meth:`.MultipactorTest.interactive_sweet_plot`.

        Parameters
        ----------
        csv_path :
            Where the summary ``CSV`` will be written (and loaded from).
        reducer :
            Passed to :meth:`to_multipactor_test_file`.
        special_reducers :
            Passed to :meth:`to_multipactor_test_file`.
        **kwargs :
            Passed to :class:`.MultipactorTest` constructor (e.g.
            ``trigger_policy``, ``info``).

        Returns
        -------
            A fully-constructed :class:`.MultipactorTest` with
            :attr:`~.MultipactorTest.power_step_set` set to ``self``.

        """
        logging.warning(
            "PowerStepSet.to_multipactor_test() is deprecated. Use "
            "MultipactorTest.from_folder() instead.",
        )
        self.to_multipactor_test_file(
            csv_path,
            reducer=reducer,
            special_reducers=special_reducers,
        )
        test = MultipactorTest(
            filepath=csv_path,
            config=self._config,
            freq_mhz=self._freq_mhz,
            swr=self._swr,
            info=self._info,
            trigger=self._trigger,
            is_raw=self._are_raw,
            **kwargs,
        )
        test.power_step_set = self

        if self._power_steps:
            info_by_name = {
                instrument.name: instrument.reduction_info
                for instrument in self._power_steps[0].instruments
                if instrument.reduction_info is not None
            }
            for instrument in test.instruments:
                instrument.reduction_info = info_by_name.get(instrument.name)
        return test
