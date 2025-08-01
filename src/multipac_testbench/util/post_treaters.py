"""Define various smoothing/smoothing functions for measured data."""

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CONVOLUTION_MODES = Literal["full", "same", "valid"]


def running_mean(
    input_data: NDArray[np.float64],
    n_mean: int,
    mode: CONVOLUTION_MODES = "full",
    **kwargs,
) -> NDArray[np.float64]:
    """Compute the runnning mean. Taken from `this link`_.

    .. _this link: https://stackoverflow.com/questions/13728392/\
moving-average-or-running-mean

    See Also
    --------
    :func:`numpy.convolve`

    Parameters
    ----------
    input_data :
        Data to smooth of shape ``N``.
    n_mean :
        Number of points on which running mean is ran.
    mode :
        - By default, mode is ``'full``'.  This returns the convolution
          at each point of overlap, with an output shape of ``(N+M-1,)``. At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.
        - ``'same'``: Mode ``'same'`` returns output of length ``max(M, N)``.
          Boundary effects are still visible.
        - ``'valid'``: Mode ``'valid'`` returns output of length ``max(M, N) -
          min(M, N) + 1``. The convolution product is only given for points
          where the signals overlap completely. Values outside the signal
          boundary have no effect.

        (taken from numpy documentation)

    Returns
    -------
        Smoothed data.

    """
    return np.convolve(input_data, np.ones(n_mean) / n_mean, mode=mode).astype(
        np.float64
    )


def average_y_for_nearby_x_within_distance(
    y_values: NDArray[np.float64],
    x_values: NDArray[np.float64],
    tol: float = 1e-6,
    max_index_distance: int = 1,
    keep_shape: bool = True,
) -> NDArray[np.float64]:
    """Average ``y_values`` measured at nearly identical ``x_values``

    This function groups values in ``y_values`` that correspond to ``x_values``
    which are numerically close (within ``tol``) and occur within
    ``max_index_distance`` of each other. These grouped ``y_values`` are
    averaged, and the result is returned either in a shape-preserving format or
    as a compact array, depending on ``keep_shape``.

    Parameters
    ----------
    y_values :
        The dependent variable values to average.
    x_values :
        The independent variable values used to group corresponding
        ``y_values``.
    tol :
        Maximum absolute difference under which ``x_values`` are considered
        equal.
    max_index_distance :
        Maximum index separation allowed when grouping similar ``x_values``.
        Prevents averaging across distant, unrelated measurements.
    keep_shape :
        If ``True``, the returned array has the same shape as the input, with
        only the first element of each group containing the average and others
        filled with ``np.nan``. If ``False`` (not recommended), returns a
        compact array with only the averaged values.

    Returns
    -------
        The averaged ``y_values``, either shape-preserving or compact.

    Raises
    ------
    ValueError
        If ``x_values`` and ``y_values`` do not have the same shape.

    Examples
    --------
    >>> x = np.array([100.0, 100.0, 200.0, 200.0])
    >>> y = np.array([1.0, 3.0, 10.0, 14.0])
    >>> average_y_for_nearby_x_within_distance(y, x)
    array([2.0, nan, 12.0, nan])

    >>> average_y_for_nearby_x_within_distance(y, x, keep_shape=False)
    array([2.0, 12.0])

    """
    if y_values.shape != x_values.shape:
        raise ValueError("x_data and y_data must have the same shape.")

    averaged = np.full_like(y_values, np.nan)

    used = np.zeros_like(x_values, dtype=bool)
    n = len(x_values)

    for i in range(n):
        if used[i]:
            continue

        xi = x_values[i]
        group_indices = [i]

        for j in range(i + 1, n):
            if used[j]:
                continue
            if (
                math.isclose(x_values[j], xi, abs_tol=tol)
                and (j - group_indices[-1]) <= max_index_distance
            ):
                group_indices.append(j)

        if len(group_indices) > 1:
            avg = np.mean(y_values[group_indices])
            if keep_shape:
                averaged[group_indices[0]] = avg
            else:
                averaged[group_indices[0]] = avg
        elif keep_shape:
            averaged[i] = y_values[i]

        for idx in group_indices:
            used[idx] = True

    if not keep_shape:
        averaged = averaged[~np.isnan(averaged)]

    return averaged


def drop_x_where_y_is_nan(
    x_values: NDArray[np.float64], y_values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``x_values`` without indexes where ``y_values`` is ``np.nan``.

    This can be used in for :class:`.RPA` when some current data is dropped
    (``keep_shape = False``) but we still want the same shape for potential.

    """
    indexes = ~np.isnan(y_values)
    return x_values[indexes]


def replace_data_under_threshold(
    input_data: NDArray[np.float64],
    threshold: float,
    replace_value: float,
    min_consecutive: int = 1,
) -> NDArray[np.float64]:
    """Replace data where ``min_consecutive`` values are below ``threshold``.

    Data is replaced by ``replace_value``.

    Parameters
    ----------
    input_data :
        Data to filter.
    threshold :
        Threshold under which data is considered noise.
    replace_value :
        Value to replace the data with.
    min_consecutive :
        Minimum number of consecutive values below the threshold required for
        replacement.

    Returns
    -------
        Modified data array.

    """
    data = input_data.copy()
    mask = data < threshold

    i = 0
    while i < len(mask):
        if mask[i]:
            start = i
            while i < len(mask) and mask[i]:
                i += 1
            end = i
            if end - start >= min_consecutive:
                data[start:end] = replace_value
        else:
            i += 1

    return data


def return_constant(
    input_data: NDArray[np.float64], constant: float
) -> NDArray[np.float64]:
    """Always return same value."""
    return np.full_like(input_data, constant)
