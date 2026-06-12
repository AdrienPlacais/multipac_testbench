"""Define an object to hold a single multipactor threshold.

Also define a place-holder to mark when a minimum or maximum of threshold was
reached.

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

THRESHOLD_NATURE_T = Literal["upper", "lower"]
THRESHOLD_WAY_T = Literal["enter", "exit"]
THRESHOLD_DETECTOR_T = Literal["any", "all"]
THRESHOLD_DETECTOR = ("any", "all")
POWER_EXTREMUM_T = Literal["minimum", "maximum"]

#: Function taking in a :class:`.Threshold`, and returning a boolean.
THRESHOLD_FILTER_T = Callable[["Threshold"], bool]


class ThresholdFilter(Protocol):
    """Function taking in a :class:`.Threshold`, and returning a boolean.

    This resolves, in contrary to classic:

    .. code-block:: python

        THRESHOLD_FILTER_T = Callable[["Threshold"], bool]

    """

    def __call__(self, threshold: Threshold) -> bool: ...


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

    @property
    def is_global(self) -> bool:
        """Tell if threshold is global by checking if ``position`` is nan."""
        return bool(np.isnan(self.position))


def create_thresholds(
    multipactor: NDArray[np.bool],
    growth_array: NDArray[np.float64],
    detecting_instrument: str | THRESHOLD_DETECTOR_T,
    position: float,
    threshold_predicate: ThresholdFilter | None = None,
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
    threshold_predicate :
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
    return [
        t
        for t in thresholds
        if threshold_predicate is None or threshold_predicate(t)
    ]


@dataclass
class PowerExtremum:
    """Place-holder for reaching a minimum or maximum of power."""

    #: At which sample index the power reached an extremum.
    sample_index: int
    #: If the extremum is mini/maxi
    nature: POWER_EXTREMUM_T
    #: Clear interpretation of what was detected
    info: Literal[
        "First point forced to a minimum",
        "Trough of a seesaw profile",
        "Trough of a triangle profile",
        "Last point forced to a minimum",
        "Peak of a seesaw profile",
        "Peak of a triangle profile",
    ]
    #: If the extremum corresponds to a small power change (power follows a
    #: triangular profile), in opposition to significant power change (seesaw
    #: profile)
    smooth: bool = True

    def __eq__(self, other: object) -> bool:
        """Test that two extrema represent the same thing."""
        if not isinstance(other, PowerExtremum):
            return False
        return (
            self.sample_index == other.sample_index
            and self.nature == other.nature
            and self.smooth == other.smooth
        )


_GROWTH_STATUS_T = Literal["increasing", "decreasing", "constant"]


def create_power_extrema(
    growth_array: NDArray[np.float64],
) -> list[PowerExtremum]:
    """Create power extrema.

    Supports triangular-like and seesaw profiles. Seesaw detection does not
    work properly if ``growth_array`` was generated using
    :func:`.noisy_array_is_growing` (:class:`.ForwardPower`). Prefer using
    :func:`.not_noisy_array_is_growing` (:class:`.PowerSetpoint`).

    Parameters
    ----------
    growth_array :
        Holds ``1.0`` where it grows, ``-1.0`` where it decreases, and ``0.0``
        where it changes. We use the position of those np.nan to determine
        power extrema.

    """
    growth_status = np.empty(np.shape(growth_array), dtype=object)
    growth_status[np.where(growth_array == -1.0)] = "decreasing"
    growth_status[np.where(growth_array == 0.0)] = "constant"
    growth_status[np.where(growth_array == 1.0)] = "increasing"

    extrema: list[PowerExtremum] = [
        PowerExtremum(0, "minimum", info="First point forced to a minimum")
    ]
    i_max = len(growth_array) - 1

    if growth_status[1] != "increasing":
        logging.info(
            "Signal does not start rising. Consider trimming leading points."
        )
    if growth_status[-1] != "decreasing":
        logging.info(
            "Signal does not end falling. Consider trimming trailing points."
        )

    current: _GROWTH_STATUS_T
    prev: _GROWTH_STATUS_T

    for i in range(1, i_max):
        prev, current = growth_status[i - 1 : i + 1]
        if current == "constant":
            continue

        if prev == "increasing" and current == "decreasing":
            extrema.append(
                PowerExtremum(i, "maximum", info="Peak of a triangle profile")
            )
            continue
        elif prev == "decreasing" and current == "increasing":
            extrema.append(
                PowerExtremum(
                    i, "minimum", info="Trough of a triangle profile"
                )
            )
            continue

    extrema.append(
        PowerExtremum(i_max, "minimum", info="Last point forced to a minimum")
    )

    # Post-process: adjacent extrema are seesaw pairs
    for j in range(1, len(extrema)):
        prev_extrem = extrema[j - 1]
        curr_extrem = extrema[j]
        if curr_extrem.sample_index - prev_extrem.sample_index == 1:
            for ext in (curr_extrem, prev_extrem):
                ext.smooth = False
                ext.info = ext.info.replace("triangle", "seesaw")  # type: ignore
    return extrema
