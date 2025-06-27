"""Define an object to hold all thresholds of a multipactor test."""

from collections.abc import Callable, Iterable, Iterator, Sequence

import numpy as np
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.threshold.threshold import (
    THRESHOLD_DETECTOR_T,
    THRESHOLD_NATURE_T,
    PowerExtremum,
    Threshold,
)
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

    # def at(self, position: float, tol: float = 1e-10) -> list[Threshold]:
    #     """Give thresholds measured at a given position."""
    #     return [x for x in self if abs(x.position - position) < tol]
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
