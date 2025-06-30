"""Define an object to hold all thresholds of a multipactor test."""

import itertools
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.threshold.helper import extract_detecting_name
from multipac_testbench.threshold.threshold import (
    THRESHOLD_DETECTOR_T,
    PowerExtremum,
    Threshold,
    create_power_extrema,
    create_thresholds,
)
from multipac_testbench.util.types import MULTIPAC_DETECTOR_T
from numpy.typing import NDArray

THRESHOLD_FILTER_T = Callable[[Threshold], bool]


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
        self._thresholds = sorted(
            thresholds,
            key=lambda t: t.sample_index,
        )
        self._extrema = sorted(
            power_extrema,
            key=lambda p: p.sample_index,
        )
        self._warn_instruments_at_same_position()

    @classmethod
    def from_instruments(
        cls,
        multipac_detector: MULTIPAC_DETECTOR_T,
        detecting_instruments: Iterable[Instrument],
        growth_array: NDArray[np.float64],
    ) -> Self:
        """Create the :class:`.ThresholdSet` object.

        This method is used in :meth:`.MultipactorTest.determine_thresholds`.

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

        """
        nested_thresholds = [
            create_thresholds(
                multipac_detector(instrument.data),
                growth_array,
                str(instrument),
                instrument.position,
                instrument.color,
            )
            for instrument in detecting_instruments
            if isinstance(instrument.position, float)
        ]
        thresholds = itertools.chain(*nested_thresholds)
        power_extrema = create_power_extrema(growth_array)
        return cls(thresholds, power_extrema)

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
        separated by ``0`` or less, both objects are removed.

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
        """Assert bijection between detecting instruments pos and name."""
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
        """Give thresholds measured at a given position.

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
        list[Threshold]
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
        instruments: Sequence[Instrument],
        tol: float = 1e-10,
        global_instruments: bool = False,
        global_multipactor: bool = False,
    ):
        """Return instrument values at threshold sample indices.

        Parameters
        ----------
        instruments :
            Instruments to search from. Must have ``.position`` and ``.data``
            attributes.
        tol :
            Tolerance for position matching.
        global_instruments :
            If instruments not position-specific (eg :class:`.ForwardPower`)
            should be returned.
        global_multipactor :
            If multipactor not position-specific (eg thresholds created by
            merging several other multipactor arrays) should be returned.

        Returns
        -------
        pd.DataFrame
            Columns are named by detecting instrument + threshold nature.
            Indexes are the corresponding sample indices.

        """
        result: dict[str, dict[int, float]] = defaultdict(dict)

        for threshold in self:
            for instrument in instruments:
                far_away = math.isclose(
                    instrument.position, threshold.position, abs_tol=tol
                )
                if far_away:
                    continue

                if not global_instruments and np.isnan(instrument.position):
                    continue

                if not global_multipactor and np.isnan(threshold.position):
                    continue

                label = f"{threshold.detecting_instrument} {threshold.nature}"
                idx = threshold.sample_index
                result[label][idx] = instrument.data[idx]
                break  # Assume one match is enough

        return pd.DataFrame({k: pd.Series(v) for k, v in result.items()})

    def according_to(
        self, instrument: Instrument | str | THRESHOLD_DETECTOR_T
    ) -> list[Threshold]:
        """Give thresholds measured by ``instrument``."""
        if isinstance(instrument, Instrument):
            instrument = str(Instrument)
        return [
            x
            for x in self
            if extract_detecting_name(x.detecting_instrument) == instrument
        ]

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

    #     """Get lower thresholds."""
    #     return [x for x in self if x.nature == "lower"]
    #
    # def uppers(self) -> list[Threshold]:
    #     """Get upper thresholds."""
    #     return [x for x in self if x.nature == "upper"]

    def get_threshold_label_color_map(
        self,
    ) -> dict[str, tuple[float, float, float] | None]:
        """Return a mapping from threshold label to color.

        Assumes :attr:`.Threshold.color` is already set to the corresponding
        :class:`.Instrument` color.

        Returns
        -------
        dict[str, str]
            Mapping from ``"<detecting_instrument> <nature>"`` to the threshold
            color.

        """
        return {
            f"{th.detecting_instrument} {th.nature}": th.color for th in self
        }
