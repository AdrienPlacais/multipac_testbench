"""Define functions to detect where multipactor happens."""

from typing import Any

import numpy as np
from multipac_testbench.util.filtering import (
    remove_isolated_false,
    remove_isolated_true,
)
from numpy.typing import NDArray


def quantity_is_above_threshold(
    quantity: NDArray[np.float64],
    threshold: float,
    consecutive_criterion: int = 0,
    minimum_number_of_points: int = 1,
    **kwargs: Any,
) -> NDArray[np.bool]:
    """Detect where ``quantity`` is above a given threshold.

    Parameters
    ----------
    quantity :
        Array of measured multipactor quantity.
    threshold :
        Quantity value above which multipactor is detected.
    consecutive_criterion :
        If provided, we gather multipactor zones that were separated by
        ``consecutive_criterion`` measure points or less.
    minimum_number_of_points :
        If provided, the multipactor must happen on at least
        ``minimum_number_of_points`` consecutive points, otherwise we consider
        that it was a measurement flaw.

    Returns
    -------
    multipactor :
        True where multipactor was detected.

    """
    multipactor = quantity >= threshold

    if consecutive_criterion > 0:
        multipactor = remove_isolated_false(multipactor, consecutive_criterion)

    if minimum_number_of_points > 1:
        multipactor = remove_isolated_true(
            multipactor, minimum_number_of_points
        )

    return multipactor


def start_and_end_of_contiguous_true_zones(
    multipactor: NDArray[np.bool],
) -> list[tuple[int, int]]:
    """Get indexes of the entry and exit of contiguous multipactor zones.

    Parameters
    ----------
    multipactor :
        Iterable where True means there is multipactor, False no multipactor,
        and np.nan undetermined.

    Returns
    -------
    zones :
        List of first and last index of every multipactor band (multipactor
        contiguous zone).

    """
    diff = np.where(np.diff(multipactor))[0]
    n_changes = diff.size

    starts = (diff[::2] + 1).tolist()
    ends = (diff[1::2] + 1).tolist()

    # Multipacting zones are "closed"
    if n_changes % 2 == 0:
        # Multipacting zones are not closed
        if multipactor[0]:
            starts, ends = ends, starts
            starts.insert(0, 0)
            ends.append(None)

    # One multipacting zone is "open"
    else:
        ends.append(None)

        if multipactor[0]:
            starts, ends = ends, starts
            starts = ends
            starts.insert(0, 0)

    zones = [(start, end) for start, end in zip(starts, ends)]
    return zones
