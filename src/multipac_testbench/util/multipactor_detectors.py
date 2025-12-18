"""Define functions to detect where multipactor happens."""

from typing import Any

import numpy as np
from multipac_testbench.util.filtering import (
    clean_boolean_mask,
)
from multipac_testbench.util.post_treaters import running_mean
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
        True where multipactor was detected.

    """
    multipactor = quantity >= threshold
    return clean_boolean_mask(
        multipactor,
        min_true=minimum_number_of_points,
        max_false_gap=consecutive_criterion,
    )


def quantity_is_above_local_average(
    quantity: NDArray[np.float64],
    baseline_window: int = 300,
    threshold_factor: float = 3.0,
    consecutive_criterion: int = 0,
    minimum_number_of_points: int = 1,
    **kwargs,
) -> NDArray[np.bool]:
    """Detect where ``quantity`` is above the local average.

    Procedure is the following:

    #. Compute running mean (slow trend).
    #. Compute array of residuals between actual data and running mean.
    #. Average array of residuals to get mean difference level.
    #. Multipactor happens where residuals are above the noise level, scaled
       by ``threshold_factor``.

    Parameters
    ----------
    quantity :
        Array of measured multipactor quantity.
    baseline_window :
        Window size (in samples) for baseline estimation. Set it to two power
        cycles for a good first estimation.
    threshold_factor :
        Multiplier for noise level above baseline. This can be negative!
    consecutive_criterion :
        If provided, we gather multipactor zones that were separated by
        ``consecutive_criterion`` measure points or less.
    minimum_number_of_points :
        If provided, the multipactor must happen on at least
        ``minimum_number_of_points`` consecutive points, otherwise we consider
        that it was a measurement flaw.

    Returns
    -------
        True where multipactor was detected.

    """
    slow_trend = running_mean(quantity, n_mean=baseline_window)
    residual = quantity - slow_trend
    noise_level = np.median(np.abs(residual))
    multipactor = residual > threshold_factor * noise_level

    return clean_boolean_mask(
        multipactor,
        min_true=minimum_number_of_points,
        max_false_gap=consecutive_criterion,
    )


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
