"""Define an object to hold all thresholds of a multipactor test."""

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.threshold.helper import (
    sorter_index_then_way,
    threshold_df_column_header,
)
from multipac_testbench.threshold.threshold import (
    THRESHOLD_DETECTOR_T,
    THRESHOLD_FILTER_T,
    PowerExtremum,
    Threshold,
    create_power_extrema,
    create_thresholds,
)
from multipac_testbench.util.types import MULTIPAC_DETECTOR_T
from numpy.typing import NDArray


class ThresholdSet:

    def __init__(
        self,
        thresholds: Iterable[Threshold],
        power_extrema: Iterable[PowerExtremum],
    ) -> None:
        """Create object.

        Parameters
        ----------
        thresholds :
            Multipactor thresholds detected during a :class:`.MultipactorTest`.
        power_extrema :
            Power minima/maxima delimiting the different power cycles in the
            :class:`.MultipactorTest`.

        """
        self._thresholds = sorted(thresholds, key=lambda t: t.sample_index)
        self.extrema = sorted(power_extrema, key=lambda p: p.sample_index)
        self._warn_instruments_at_same_position()

    @classmethod
    def from_instruments(
        cls,
        multipac_detector: MULTIPAC_DETECTOR_T,
        detecting_instruments: Iterable[Instrument],
        growth_array: NDArray[np.float64],
        predicate: THRESHOLD_FILTER_T | None = None,
        threshold_reducer: THRESHOLD_DETECTOR_T | None = None,
    ) -> Self:
        """Create a ThresholdSet using the specified detection strategy.

        Parameters
        ----------
        multipac_detector :
            Function that takes in the ``data`` of an :class:`.Instrument`
            and returns an array, where True means multipactor and False no
            multipactor.
        detecting_instruments :
            Instruments to apply ``multipac_detector`` on.
        growth_array :
            Holds ``1.0`` where power increases, ``0.0`` where it is stable,
            ``-1.0`` where it decreases.
        threshold_reducer :
            - not provided: thresholds are computed for each instrument
              independently.
            - "any": thresholds appear when multipactor is detected by *any*
              of the provided detecting instrument.
            - "all": thresholds appear when multipactor is detected by *all*
              the provided detecting instrument.
        predicate :
            Function filtering the thresholds. Applied *after*
            ``threshold_reducer``.

        """
        if threshold_reducer is None:
            thresholds = [
                threshold
                for instr in detecting_instruments
                if isinstance(instr.position, float)
                for threshold in create_thresholds(
                    multipac_detector(instr.data),
                    growth_array,
                    detecting_instrument=instr.name,
                    position=instr.position,
                    predicate=predicate,
                    color=instr.color,
                )
            ]
        elif threshold_reducer in {"any", "all"}:
            multipactors = [
                multipac_detector(instr.data)
                for instr in detecting_instruments
                if isinstance(instr.position, float)
            ]
            reducer = np.any if threshold_reducer == "any" else np.all
            combined = reducer(multipactors, axis=0)
            thresholds = create_thresholds(
                combined,
                growth_array,
                detecting_instrument=threshold_reducer,
                position=np.nan,
                predicate=predicate,
                color=(0, 0, 0),
            )
        else:
            raise ValueError(f"Unknown {threshold_reducer = }")

        power_extrema = create_power_extrema(growth_array)
        return cls(thresholds, power_extrema)

    @classmethod
    def last(
        cls, threshold_set: Self, predicate: THRESHOLD_FILTER_T | None = None
    ) -> Self:
        """
        Create object holding the last threshold measured by every instrument.

        See Also
        --------
        :class:`AveragedThresholdSet`

        Parameters
        ----------
        threshold_set :
            Holds all the detected thresholds.
        predicate :
            Additional predicate, *eg* to exclude thresholds measured during
            the first power cycles, from a specific detecting instrument, of
            a certain type...

        Returns
        -------
            Holds only one lower and one upper :class:`.Threshold` per
            detecting instrument: the last one measured during the test.

        """
        filtered_thresholds = tuple(
            [t for t in threshold_set if predicate is None or predicate(t)]
        )
        last_thresholds_by_instr: dict[str, Threshold] = {}
        for t in filtered_thresholds[::-1]:
            if t.detecting_instrument in last_thresholds_by_instr:
                continue
            last_thresholds_by_instr[t.detecting_instrument] = t
        return cls(last_thresholds_by_instr.values(), threshold_set.extrema)

    @classmethod
    def subset(
        cls, threshold_set: Self, predicate: THRESHOLD_FILTER_T
    ) -> Self:
        """Return object holding a subset of ``threshold_set``.

        ``predicate`` is used to filter on the :class:`.Threshold`.

        """
        thresholds = [t for t in threshold_set if predicate(t)]
        return cls(thresholds=thresholds, power_extrema=threshold_set.extrema)

    @classmethod
    def extreme(
        cls, threshold_set: Self, predicate: THRESHOLD_FILTER_T | None = None
    ) -> Self:
        """Create object holding only the most *extreme* :class:`.Threshold`.

        For each half cycle:

        - If power increases: keep first lower and last upper threshold.
        - If power decreases: keep first upper and last lower threshold.

          - If there was still multipactor somewhere when the half power cycle
            ended (e.g. instrument with a lower but no upper threshold), no
            upper threshold is added.
        - If direction is undetermined: skip the cycle.

        Parameters
        ----------
        threshold_set :
            The full set of thresholds.
        predicate :
            A function to select relevant thresholds.

        Returns
        -------
            A new object containing only selected extreme thresholds.

        """
        assert len(threshold_set.detecting_instruments()) <= 1, (
            "This method currently does not handle detection from several "
            "instruments."
        )

        subset = []
        for key, thresholds in threshold_set._thresholds_by_half_power_cycle(
            predicate=predicate
        ).items():
            if not thresholds:
                continue

            direction = key.split("(", 1)[-1].removesuffix(")").strip()
            if direction not in {"increasing", "decreasing"}:
                logging.warning(f"Skipped undetermined cycle: {key}")
                continue

            if direction == "increasing":
                first = min(thresholds, key=sorter_index_then_way)
                if first.nature == "lower" and first.way == "enter":
                    subset.append(first)

                last = max(thresholds, key=sorter_index_then_way)
                if last.nature == "upper" and last.way == "exit":
                    subset.append(last)
                continue

            if direction == "decreasing":
                first = min(thresholds, key=sorter_index_then_way)
                if first.nature == "upper" and first.way == "enter":
                    subset.append(first)

                last = max(thresholds, key=sorter_index_then_way)
                if last.nature == "lower" and last.way == "exit":
                    subset.append(last)

        return cls(thresholds=subset, power_extrema=threshold_set.extrema)

    def __iter__(self) -> Iterator[Threshold]:
        """Iterate over stored :class:`.Threshold` objects.

        Yields
        ------
        Threshold
            The stored :class:`.Threshold` objects, sorted by sample index.

        """
        return iter(self._thresholds)

    def remove_singularities(self, min_consecutive: int = 1) -> None:
        """Remove fugitive :class:`.Threshold`.

        If two :class:`.Threshold` are detected by the same
        :class:`.Instrument` and their :attr:`.Threshold.sample_index` are
        separated by ``min_consecutive - 1`` or less, both objects are removed.

        Parameters
        ----------
        min_consecutive :
            :class:`.Threshold` objects separated by less than
            ``min_consecutive`` sample index are removed. The default
            ``min_consecutive=1`` removes multipactor spanning over a single
            sample index.

        """
        by_instr: dict[str, list[Threshold]] = defaultdict(list)
        for t in self._thresholds:
            by_instr[t.detecting_instrument].append(t)

        cleaned_thresholds = []
        for thresholds in by_instr.values():
            thresholds.sort(key=lambda t: t.sample_index)
            to_remove: list[Threshold] = []
            for i in range(len(thresholds) - 1):
                current = thresholds[i]
                next_ = thresholds[i + 1]
                if (
                    abs(current.sample_index - next_.sample_index)
                    >= min_consecutive
                ):
                    continue
                to_remove.append(current)
                to_remove.append(next_)
            cleaned = [t for t in thresholds if t not in to_remove]
            cleaned_thresholds.extend(cleaned)

        self._thresholds = cleaned_thresholds

    def _warn_instruments_at_same_position(self) -> None:
        """Verify bijection between detecting instruments pos and name."""
        pos_to_names: dict[float, str] = {}
        warned_positions = set()
        for threshold in self._thresholds:
            name = threshold.detecting_instrument
            pos = threshold.position
            if pos in pos_to_names and pos_to_names[pos] != name:
                if pos not in warned_positions:
                    msg = (
                        "Multiple instruments detected at the same position "
                        f"{pos}:\n- {pos_to_names[pos]}\n- {name}"
                    )
                    logging.warning(msg)
                    warned_positions.add(pos)
            pos_to_names[pos] = name

    def sample_indexes(
        self, *, predicate: THRESHOLD_FILTER_T | None = None
    ) -> list[int]:
        """Return sample indexes matching optional filter."""
        return [
            t.sample_index for t in self if predicate is None or predicate(t)
        ]

    def apply_to(self, instrument: Instrument) -> NDArray[np.float64]:
        """Extract instrument data at threshold sample indexes."""
        idx = self.sample_indexes()
        return instrument.data[idx]

    def at(
        self, position: float, tol: float = 1e-10, return_global: bool = False
    ) -> list[Threshold]:
        """Gather all thresholds measured at ``position``.

        Parameters
        ----------
        position :
            Where you want the thresholds.
        tol :
            Tolerance over the position.
        return_global :
            To return global multipactors, and also return all thresholds when
            ``position`` is ``np.nan``. ``np.nan`` position are associated with
            "global" instruments, such as :class:`.ForwardPower`, and with
            "global" multipactors, such as obtained by crossing several
            :class:`.Instrument` data.

        Returns
        -------
            All multipactor thresholds detected at this position.

        """
        return [
            x
            for x in self._thresholds
            if math.isclose(x.position, position, abs_tol=tol)
            or return_global
            and (np.isnan(x.position) or np.isnan(position))
        ]

    def data_at_thresholds(
        self,
        instruments: Iterable[Instrument],
        tol: float = 1e-10,
        global_instruments: bool = False,
        global_multipactor: bool = False,
        xdata_instrument: Instrument | None = None,
        unique_x_value: float | None = None,
    ) -> pd.DataFrame:
        """Return instrument values at threshold sample indices.

        We match :class:`.Threshold` and :class:`.Instrument` objects by
        position.

        Parameters
        ----------
        instruments :
            Instruments to which data must be plotted. Must have ``.position``
            and ``.data`` attributes.
        tol :
            Tolerance for position matching.
        global_instruments :
            If instruments not position-specific (eg :class:`.ForwardPower`)
            should be returned.
        global_multipactor :
            If multipactor not position-specific (eg thresholds created by
            merging several other multipactor arrays) should be returned.
        xdata_instrument :
            Its data is returned at every threshold. It results in a unique
            ``xdata`` column, without ``nan``, that can be used as a common
            x-data for plotting.
        unique_x_value :
            If given, this value will replace every value of the
            ``xdata_instrument`` column.

        Returns
        -------
            Columns are named by detecting instrument + threshold nature:
            ``"NI9205_E4 @ upper threshold (according to NI9205_MP4l)"``. If
            ``xdata_instrument`` was given, also return this instrument values
            at every sample index (can be unique value if ``unique_x_value``
            was given). Indexes are the sample indices at every threshold.

        """
        #           {column:  {sample_index: instrument value}}
        result: dict[str, dict[int, float]] = defaultdict(dict)

        for threshold in self:
            for instrument in instruments:
                is_close = (
                    math.isclose(
                        instrument.position, threshold.position, abs_tol=tol
                    )
                    or (global_instruments and np.isnan(instrument.position))
                    or (global_multipactor and np.isnan(threshold.position))
                )

                if not is_close:
                    continue

                label = threshold_df_column_header(instrument, threshold)
                idx = threshold.sample_index
                result[label][idx] = instrument.data[idx]

        if xdata_instrument is None:
            return pd.DataFrame({k: pd.Series(v) for k, v in result.items()})

        xlabel = xdata_instrument.ylabel()
        result[xlabel] = {
            t.sample_index: xdata_instrument.data[t.sample_index] for t in self
        }
        df = pd.DataFrame({k: pd.Series(v) for k, v in result.items()})
        if unique_x_value is not None:
            df[xlabel] = unique_x_value
        return df

    def according_to(
        self, instrument: Instrument | str | THRESHOLD_DETECTOR_T
    ) -> list[Threshold]:
        """Give thresholds measured by ``instrument``."""
        if isinstance(instrument, Instrument):
            detecting_name = instrument.name
        else:
            detecting_name = instrument

        thresholds: list[Threshold] = []
        for x in self:
            if isinstance(x.detecting_instrument, Instrument):
                matching = x.detecting_instrument.name
            else:
                matching = x.detecting_instrument
            if detecting_name == matching:
                thresholds.append(x)
        return thresholds

    def remove_detected_by(
        self, instrument: Instrument | str | THRESHOLD_DETECTOR_T
    ) -> None:
        """Remove thresholds detected by ``instrument``."""
        to_remove = self.according_to(instrument)
        cleaned = [t for t in self if t not in to_remove]
        self._thresholds = cleaned
        logging.info(
            f"Removed the {len(to_remove)} thresholds detected by {instrument}"
        )

    def get_threshold_label_color_map(
        self, instruments: Sequence[Instrument]
    ) -> dict[str, tuple[float, float, float] | None]:
        """Maps threshold dataframe column headers to corresponding colors.

        Assumes :attr:`.Threshold.color` is already set to the corresponding
        :class:`.Instrument` color.

        Returns
        -------
            Mapping from a header looking like ``"NI9205_E4 @ upper threshold
            (according to NI9205_MP4l)"``, to the threshold color (usually,
            this is detecting instrument color).

        """
        label_to_color = {}
        for threshold in self:
            for instrument in instruments:
                header = threshold_df_column_header(instrument, threshold)
                label_to_color[header] = threshold.color
        return label_to_color

    def detecting_instruments(self) -> set[str | THRESHOLD_DETECTOR_T]:
        """Return instruments that detected at least one threshold."""
        return {t.detecting_instrument for t in self}

    def _thresholds_by_half_power_cycle(
        self, predicate: THRESHOLD_FILTER_T | None = None
    ) -> dict[str, list[Threshold]]:
        """Group thresholds by half power cycle, based on sample index range.

        Each group includes thresholds between two consecutive extrema:
        ``[extremum_i.sample_index, extremum_{i+1}.sample_index)``

        The dictionary key is of the form:
        - "0 (increasing)" if power increases over the interval
        - "1 (decreasing)" if power decreases over the interval
        - "2 (undetermined)" if direction cannot be determined

        .. note::
           Not ultra efficient. To update if necessary.

        Parameters
        ----------
        predicate :
            Filter the :class:`.Threshold` instances.

        Returns
        -------
            Dictionary mapping half-cycle index to thresholds within that
            range. Keys are sorted by increasing power cycle index values.

        """
        thresholds_by_cycle: dict[str, list[Threshold]] = {}

        for i, (ext1, ext2) in enumerate(
            zip(self.extrema[:-1], self.extrema[1:])
        ):
            if ext1.nature == "minimum" and ext2.nature == "maximum":
                direction = "increasing"
            elif ext1.nature == "maximum" and ext2.nature == "minimum":
                direction = "decreasing"
            else:
                direction = "undetermined"

            key = f"{i} ({direction})"

            thresholds = [
                t
                for t in self
                if ext1.sample_index <= t.sample_index < ext2.sample_index
                and (predicate is None or predicate(t))
            ]
            thresholds_by_cycle[key] = thresholds

        return thresholds_by_cycle


class AveragedThresholdSet(ThresholdSet):
    """Holds average of several thresholds.

    The main difference with a classic ``ThresholdSet`` is that its
    :meth:`.data_at_thresholds` is overriden to return data averaged from
    several :class:`.Threshold`.

    """

    @classmethod
    def from_threshold_set(
        cls,
        threshold_set: ThresholdSet,
        predicate: THRESHOLD_FILTER_T | None = None,
    ) -> Self:
        """Create an object holding averaged thresholds.

        Parameters
        ----------
        threshold_set :
            The thresholds to average.
        predicate :
            To filter thresholds to average. A typical example would be
            ``lambda t: t.sample_index > 200`` to keep only conditioned
            thresholds.

        Returns
        -------
            Object containing "averaged" thresholds. It contains one lower and
            one upper threshold per detecting instrument (if already present in
            the original :class:`.ThresholdSet`).

        """
        subset = [
            t for t in threshold_set if predicate is None or predicate(t)
        ]
        return cls(subset, threshold_set.extrema)

    def data_at_thresholds(self, *args, **kwargs) -> pd.DataFrame:
        """Return average of instrument values at threshold sample indices.

        Keep the xdata column as a representative index: for each y-column,
        compute the median of its xdata values.

        Parameters
        ----------
        instruments :
            Instruments to which data must be plotted. Must have ``.position``
            and ``.data`` attributes.
        tol :
            Tolerance for position matching.
        global_instruments :
            If instruments not position-specific (eg :class:`.ForwardPower`)
            should be returned.
        global_multipactor :
            If multipactor not position-specific (eg thresholds created by
            merging several other multipactor arrays) should be returned.
        xdata_instrument :
            Its data is returned at every threshold. It results in a unique
            ``xdata`` column, without ``nan``, that can be used as a common
            x-data for plotting.

        Returns
        -------
            Columns are named by detecting instrument + threshold nature.
            Only index is average (median) of instruments values at the various
            thresholds.

        """
        df = super().data_at_thresholds(*args, **kwargs)
        if df.index.name is None:
            return df.median().to_frame().T

        xname = df.index.name
        records = []

        for col in df.columns:
            y = df[col].dropna()
            if y.empty:
                continue
            x_median = y.index.to_series().median()
            y_median = y.median()

            row = pd.Series({col: y_median}, name=x_median)
            records.append(row)

        df = pd.DataFrame(records).sort_index()
        df.index.name = xname
        df = df.reindex(sorted(df.columns), axis=1)
        return df
