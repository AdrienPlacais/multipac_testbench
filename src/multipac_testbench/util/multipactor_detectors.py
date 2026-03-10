"""Define functions to detect where multipactor happens."""

from typing import Any

import numpy as np
from multipac_testbench.util.filtering import (
    clean_boolean_mask,
)
from multipac_testbench.util.post_treaters import lower_envelope, running_mean
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

    1. Compute running mean (slow trend).
    2. Compute array of residuals between actual data and running mean.
    3. Average array of residuals to get mean difference level.
    4. Multipactor happens where residuals are above the noise level, scaled
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
        ``consecutive_criterion`` measurement points or less.
    minimum_number_of_points :
        If provided, the multipactor must happen on at least
        ``minimum_number_of_points`` consecutive points, otherwise we consider
        that it was a measurement flaw.

    Returns
    -------
        True where multipactor was detected.

    """
    slow_trend = running_mean(quantity, n_mean=baseline_window)
    residual, threshold = residual_threshold(
        quantity, slow_trend, factor=threshold_factor
    )
    multipactor = residual > threshold

    return clean_boolean_mask(
        multipactor,
        min_true=minimum_number_of_points,
        max_false_gap=consecutive_criterion,
    )


def quantity_is_above_lower_envelope(
    quantity: NDArray[np.float64],
    envelope_window: int = 150,
    threshold_factor: float = 3.0,
    consecutive_criterion: int = 0,
    minimum_number_of_points: int = 1,
) -> NDArray[np.bool]:
    """Detect where ``quantity`` is above the local average.

    Procedure is the following:

    1. Compute lower envelope (slow trend without local bumps).
    2. Compute array of residuals between actual data and lower envelope.
    3. Average array of residuals to get mean difference level.
    4. Multipactor happens wmin_widthhere residuals are above the noise level,
       scaled by ``threshold_factor``.

    Parameters
    ----------
    quantity :
        Array of measured multipactor quantity.
    envelope_window :
        Window size (in samples) for lower envelope calculation. Set it to one
        power cycle for a good first estimation.
    threshold_factor :
        Multiplier for noise level above lower envelope. This can be negative!
    consecutive_criterion :
        If provided, we gather multipactor zones that were separated by
        ``consecutive_criterion`` measurement points or less.
    minimum_number_of_points :
        If provided, the multipactor must happen on at least
        ``minimum_number_of_points`` consecutive points, otherwise we consider
        that it was a measurement flaw.

    Returns
    -------
        True where multipactor was detected.

    """
    envelope = lower_envelope(quantity, envelope_window)
    residual, threshold = residual_threshold(
        quantity, envelope, factor=threshold_factor
    )
    multipactor = residual > threshold

    return clean_boolean_mask(
        multipactor,
        min_true=minimum_number_of_points,
        max_false_gap=consecutive_criterion,
    )


def residual_threshold(
    data: NDArray[np.float64],
    baseline: NDArray[np.float64],
    factor: float = 1.0,
) -> tuple[NDArray[np.float64], float]:
    """Compute residuals and a robust noise-based detection threshold.

    The baseline is assumed to represent typical non-multipactor behavior.
    Residuals are defined as positive excursions of ``data`` above this
    baseline. A robust estimate of the residual noise amplitude is obtained
    from the median absolute deviation and scaled to define a detection
    threshold.

    Notes
    -----
    Noise-based limit above which residuals are classified as multipactor.

    Parameters
    ----------
    data :
        :class:`.Instrument` data.
    baseline :
        Estimate of the non-multipactor baseline. Typically obtained with
        :func:`.running_mean` or :func:`.lower_envelope`.
    factor :
        Multiplicative factor applied to the noise estimate to set the
        detection strictness. Higher values reduce false positives.

    Returns
    -------
    NDArray[np.float64]
        Difference ``data - baseline``.
    float
        Noise-based limit above which residuals are classified as multipactor.

    """
    residual = data - baseline
    return residual, factor * float(np.median(np.abs(residual)))


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
