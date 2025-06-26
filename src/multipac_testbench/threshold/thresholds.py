"""Define an object to hold all thresholds of a multipactor test."""

from collections.abc import Sequence

from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.threshold.threshold import (
    THRESHOLD_DETECTOR_T,
    PowerExtremum,
    Threshold,
)


class Thresholds(list[Threshold]):

    def __init__(self, events: Sequence[Threshold | PowerExtremum]) -> None:
        """Create object.

        Parameters
        ----------
        events :
            Objects to store.

        """
        thresholds = [x for x in events if isinstance(x, Threshold)]
        thresholds = sorted(thresholds, key=lambda t: t.sample_index)
        self.__init__(sorted)
        self._extrema = [x for x in events if isinstance(x, PowerExtremum)]

    def at(self, position: float, tol: float = 1e-10) -> list[Threshold]:
        """Give thresholds measured at a given position."""
        return [x for x in self if abs(x.position - position) < tol]

    def according_to(
        self, instrument: Instrument | str | THRESHOLD_DETECTOR_T
    ) -> list[Threshold]:
        """Give thresholds measured by ``instrument``."""
        if isinstance(instrument, Instrument):
            instrument = str(Instrument)
        return [x for x in self if str(x.detecting_instrument) == instrument]

    def lowers(self) -> list[Threshold]:
        """Get lower thresholds."""
        return [x for x in self if x.nature == "lower"]

    def uppers(self) -> list[Threshold]:
        """Get upper thresholds."""
        return [x for x in self if x.nature == "upper"]


class ThresholdsFactory:
    """Create a :class:`.Thresholds` object."""
