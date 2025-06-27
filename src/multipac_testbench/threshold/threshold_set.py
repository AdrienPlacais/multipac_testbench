"""Define an object to hold all thresholds of a multipactor test."""

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.threshold.threshold import PowerExtremum, Threshold
from numpy.typing import NDArray

THRESHOLD_FILTER_T = Callable[[Threshold], bool]


class ThresholdSet:

    def __init__(self, events: Iterable[Threshold | PowerExtremum]) -> None:
        """Create object.

        Parameters
        ----------
        events :
            Objects to store.

        """
        self._thresholds = sorted(
            [x for x in events if isinstance(x, Threshold)],
            key=lambda t: t.sample_index,
        )
        self._extrema = [x for x in events if isinstance(x, PowerExtremum)]

    def __iter__(self) -> Iterator[Threshold]:
        """Iterate over stored Threshold objects, in order of sample index.

        Yields
        ------
        Threshold
            The stored Threshold objects, sorted by sample index.

        """
        return iter(self._thresholds)

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
            if abs(x.position - position) < tol
            or return_global
            and (np.isnan(x.position) or np.isnan(position))
        ]

    def data_at_threshold(
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
                far_away = abs(instrument.position - threshold.position) < tol
                if far_away:
                    continue

                if not global_instruments and np.isnan(instrument.position):
                    continue

                if not global_multipactor and np.isnan(threshold.position):
                    continue

                label = (
                    f"{threshold.detecting_instrument} "
                    f"{threshold.nature.capitalize()}"
                )
                idx = threshold.sample_index
                result[label][idx] = instrument.data[idx]
                break  # Assume one match is enough

        return pd.DataFrame({k: pd.Series(v) for k, v in result.items()})

    #
    # def according_to(
    #     self, instrument: Instrument | str | THRESHOLD_DETECTOR_T
    # ) -> list[Threshold]:
    #     """Give thresholds measured by ``instrument``."""
    #     if isinstance(instrument, Instrument):
    #         instrument = str(Instrument)
    #     return [x for x in self if str(x.detecting_instrument) == instrument]
    #
    # def lowers(self) -> list[Threshold]:
    #     """Get lower thresholds."""
    #     return [x for x in self if x.nature == "lower"]
    #
    # def uppers(self) -> list[Threshold]:
    #     """Get upper thresholds."""
    #     return [x for x in self if x.nature == "upper"]


class ThresholdsFactory:
    """Create a :class:`.Thresholds` object."""
