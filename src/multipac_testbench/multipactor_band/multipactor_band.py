"""Keep track of a single multipactor band.

A multipactor band a sequence of consecutive measurement points where
multipactor happens. A :class:`MultipactorBand` is defined on half a power
cycle.

"""

from dataclasses import dataclass


@dataclass
class IMultipactorBand:
    """Mother class of :class:`MultipactorBand` and :class:`NoMultipactorBand`."""

    #: Index of the half-power cycle, starting at zero.
    pow_index: int


@dataclass
class MultipactorBand(IMultipactorBand):
    """First and last index of a multipactor zone.

    Parameters
    ----------
    first_index :
        Index where multipactor is first detected.
    last_index :
        Index where multipactor is last detected.
    reached_second_threshold :
        If multipactor disappeared at ``upper_index``.
    power_grows :
        If the object corresponds to a half-power cycle where power was
        growing.
    lower_index :
        Index corresponding to lower threshold.
    upper_index :
        Index corresponding to upper threshold. It is ``None`` if
        ``power_grows=False``.

    """

    first_index: int
    last_index: int
    reached_second_threshold: bool
    power_grows: bool

    def __post_init__(self) -> None:
        """Set complementary quantities."""
        self.lower_index, self.upper_index = self._set_lower_upper_indexes()

    def __repr__(self) -> str:
        """Inform."""
        out = f"Half-power cycle #{self.pow_index}: "
        out += f"{self.first_index = }, {self.last_index = }, "
        out += f"{self.lower_index = }, {self.upper_index = }"
        return out

    def _set_lower_upper_indexes(self) -> tuple[int, int | None]:
        """Determine lower/upper threshold indexes."""
        lower_index, upper_index = self.first_index, self.last_index
        if not self.power_grows:
            upper_index, lower_index = lower_index, upper_index

        if not self.reached_second_threshold:
            self.upper_index = None

        return lower_index, upper_index


class NoMultipactorBand(IMultipactorBand):
    """A dummy object to keep track of cycles without multipactor."""

    def __init__(self, pow_index: int) -> None:
        """Instantiate object."""
        super().__init__(pow_index)

    def __repr__(self) -> str:
        """Inform."""
        return f"Half-power cycle #{self.pow_index}: no MP."
