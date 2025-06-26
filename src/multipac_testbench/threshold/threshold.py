"""Define an object to hold a single multipactor threshold.

Also define a place-holder to mark when a minimum or maximum of threshold was
reached.

"""

import logging
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from multipac_testbench.instruments import Instrument
from numpy.typing import NDArray

THRESHOLD_NATURE_T = Literal["upper", "lower"]
THRESHOLD_DETECTOR_T = Literal["any", "all"]
POWER_EXTREMUM_T = Literal["minimum", "maximum"]


@dataclass
class Threshold:
    """Holds a single multipactor threshold."""

    sample_index: int
    nature: THRESHOLD_NATURE_T
    detecting_instrument: Instrument | THRESHOLD_DETECTOR_T

    def __post_init__(self) -> None:
        """Add some info."""
        self.position = (
            self.detecting_instrument.position
            if isinstance(self.detecting_instrument, Instrument)
            else np.nan
        )


@dataclass
class PowerExtremum:
    """Place-holder for reaching a minimum or maximum of power."""

    sample_index: int
    nature: POWER_EXTREMUM_T

    def __eq__(self, other: object) -> bool:
        """Test that two extrema represent the same thing."""
        if not isinstance(other, PowerExtremum):
            return False
        return (
            self.sample_index == other.sample_index
            and self.nature == other.nature
        )


def power_extrema(growth_array: NDArray[np.float64]) -> list[PowerExtremum]:
    """Create power extrema.

    Parameters
    ----------
    power_growth_mask :
        Holds ``1.0`` where it grows, ``-1.0`` where it decreases, and ``0.0``
        where it changes. We use the position of those np.nan to determine
        power extrema.

    """
    extrema: list[PowerExtremum] = [PowerExtremum(0, "minimum")]
    i_max = len(growth_array) - 1

    if growth_array[1] != 1.0:
        logging.warning(
            "User should manually trim exceedent powers in order to avoid "
            "flat minima at the start of the test."
        )
    if growth_array[-1] != -1.0:
        logging.warning(
            "User should manually trim exceedent powers in order to avoid "
            "flat minima at the end of the test."
        )

    for i in range(1, i_max):
        if growth_array[i] != 0.0:
            continue

        prev = growth_array[i - 1]
        next = growth_array[i + 1]

        if prev == 1.0 and next == -1.0:
            extrema.append(PowerExtremum(i, "maximum"))
            continue
        if prev == -1.0 and next == 1.0:
            extrema.append(PowerExtremum(i, "minimum"))
            continue

        logging.warning(
            f"Detected noise or plateau around {i = }. Ignoring..."
        )

    extrema.append(PowerExtremum(i_max, "minimum"))
    return extrema
