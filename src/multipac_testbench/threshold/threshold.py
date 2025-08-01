"""Define an object to hold a single multipactor threshold.

Also define a place-holder to mark when a minimum or maximum of threshold was
reached.

.. todo::
   Fix the typing of ``THRESHOLD_FILTER_T``.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

THRESHOLD_NATURE_T = Literal["upper", "lower"]
THRESHOLD_WAY_T = Literal["enter", "exit"]
THRESHOLD_DETECTOR_T = Literal["any", "all"]
THRESHOLD_DETECTOR = ("any", "all")
POWER_EXTREMUM_T = Literal["minimum", "maximum"]

#: Function taking in a :class:`.Threshold`, and returning a boolean.
THRESHOLD_FILTER_T = Callable[["Threshold"], bool]


@dataclass
class Threshold:
    """Holds a single multipactor threshold.

    .. todo::
        Handle isolated mp zones? Characterized by two Threshold objects at
        same position, same indexes. One is upper, other is lower. One is
        enter, other is exit

    """

    #: At which sample index the threshold was detected.
    sample_index: int
    #: If the threshold is a lower threshold or an upper threshold.
    nature: THRESHOLD_NATURE_T
    #: If the threshold was measured during an entry or an exit of the
    #: multipator band
    way: THRESHOLD_WAY_T
    #: Name of the instrument that detected this threshold.
    detecting_instrument: str | THRESHOLD_DETECTOR_T
    #: Position of the object that detected this threshold.
    position: float
    #: Color of the :class:`.Instrument` that detected this threshold.
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


def create_thresholds(
    multipactor: NDArray[np.bool],
    growth_array: NDArray[np.float64],
    detecting_instrument: str | THRESHOLD_DETECTOR_T,
    position: float,
    predicate: THRESHOLD_FILTER_T | None = None,
    color: tuple[float, float, float] | None = None,
) -> list[Threshold]:
    """Create threshold objects corresponding to a single detecting instrument.

    Parameters
    ----------
    multipactor :
        Array where True means multipactor and False no multipactor, according
        to ``detecting_instrument``.
    growth_array :
        Holds ``1.0`` where power grows, ``-1.0`` where it decreases, and
        ``0.0`` at transition points. Used to determine threshold nature
        (lower/upper).
    detecting_instrument :
        Name of :class:`.Instrument` that created the ``multipactor`` array.
    position :
        Position of :class:`.Instrument` that created the ``multipactor``
        array.
    predicate :
        Function filtering the created thresholds.
    color :
        Color of the detecting instrument.

    Returns
    -------
    list[Threshold]
        All multipactor thresholds detected by the :class:`.Instrument` named
        ``detecting_instrument``, filtered by ``predicate``.

    """
    thresholds: list[Threshold] = []
    actual_color = color if color is not None else (1.0, 1.0, 1.0)

    if multipactor[0]:
        logging.warning(
            "Multipactor detected at the start of the test. May cause "
            "instabilities."
        )
        thresholds.append(
            Threshold(
                0,
                "lower",
                "enter",
                detecting_instrument,
                position,
                color=actual_color,
            )
        )

    delta_mp = np.diff(multipactor.astype(np.float64))
    for i, delta in enumerate(delta_mp, start=1):
        if delta == 0.0:
            continue

        if delta > 0.0:
            way = "enter"
            # Transition: No MP [i - 1] -> MP [i]
            # so we enter multipactor at [i]
            i_threshold = i
            nature = "lower" if growth_array[i_threshold] > 0 else "upper"
        else:
            way = "exit"
            # Transition: MP [i - 1] -> no MP [i]
            # so last detected multipactor was at [i - 1]
            i_threshold = i - 1
            nature = "upper" if growth_array[i_threshold] > 0 else "lower"

        thresholds.append(
            Threshold(
                i_threshold,
                nature,
                way,
                detecting_instrument,
                position,
                color=actual_color,
            )
        )
    return [t for t in thresholds if predicate is None or predicate(t)]


@dataclass
class PowerExtremum:
    """Place-holder for reaching a minimum or maximum of power."""

    #: At which sample index the power reached an extremum
    sample_index: int
    #: If the extremum is mini/maxi
    nature: POWER_EXTREMUM_T

    def __eq__(self, other: object) -> bool:
        """Test that two extrema represent the same thing."""
        if not isinstance(other, PowerExtremum):
            return False
        return (
            self.sample_index == other.sample_index
            and self.nature == other.nature
        )


def create_power_extrema(
    growth_array: NDArray[np.float64],
) -> list[PowerExtremum]:
    """Create power extrema.

    Parameters
    ----------
    growth_array :
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
