"""Define various smoothing functions for measured data."""

import logging
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import minimum_filter1d, uniform_filter1d

#: Available modes for the running mean routine.
CONVOLUTION_MODES_T = Literal[
    "reflect", "constant", "nearest", "mirror", "wrap"
]
DEPRECATED_CONVOLUTION_MODES_T = Literal["full", "same", "valid"]
#: Deprecated modes for the running mean routine.
DEPRECATED_CONVOLUTION_MODES = ("full", "same", "valid")


def running_mean(
    input_data: NDArray[np.float64],
    n_mean: int,
    mode: CONVOLUTION_MODES_T | DEPRECATED_CONVOLUTION_MODES_T = "nearest",
    **kwargs,
) -> NDArray[np.float64]:
    """Compute the running mean.

    .. deprecated:: 1.9.0
       The modes listed in :data:`DEPRECATED_CONVOLUTION_MODES` will call the
       :func:`numpy.convolve` function. Prefer the :data:`CONVOLUTION_MODES_T`,
       which will call :func:`scipy.ndimage.uniform_filter1d`.

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
        Convolution modes. Refer to :func:`scipy.ndimage.uniform_filter1d`
        documentation, which is the actual function that is called.

    Returns
    -------
        Smoothed data.

    """
    if mode in DEPRECATED_CONVOLUTION_MODES:
        logging.warning(
            "Using deprecated mode, based on np.convolve. Prefer the scipy."
            "ndimage.uniform_filter1d function, which is more robust on the "
            "edges."
        )
        return np.convolve(
            input_data, np.ones(n_mean) / n_mean, mode=mode
        ).astype(np.float64)
    return uniform_filter1d(input_data, size=n_mean, mode=mode)


def lower_envelope(
    input_data: NDArray[np.float64],
    envelope_window: int,
    mode: CONVOLUTION_MODES_T = "nearest",
) -> NDArray[np.float64]:
    """Compute the lower envelope, i.e. the signal minimum without the bumps.

    .. todo::
       Illustration in documentation.

    .. todo::
       May also use :func:`scipy.ndimage.percentile_filter`. May be more robust
       with noisy data or occasional downward spikes. To test.

    Parameters
    ----------
    input_data :
        Data to smooth of shape ``N``.
    envelope_window :
        Number of points on which lower envelope is taken.
    mode :
        Convolution modes. Refer to :func:`scipy.ndimage.minimum_filter1d`
        documentation, which is the actual function that is called.

    Returns
    -------
        Smoothed data.

    """
    return minimum_filter1d(input_data, size=envelope_window, mode=mode)


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


def get_data_above_noise(
    data: NDArray,
    noise_level: float | None = None,
    level: float | None = None,
) -> NDArray:
    """Detect signal above noise, return it.

    Parameters
    ----------
    data :
        Array of data in pulsed shape, such as power pulse or synch.
    noise_level :
        Absolute level of noise. If not provided, we use ``level`` to compute
        it.
    level :
        Noise percentile wrt ``data``, in :unit:`\\%`.

    Returns
    -------
    NDArray
        Only the widest pulse detected in the signal.

    """
    if noise_level is None:
        if level is None:
            raise ValueError("You must provide ``noise_level`` or ``level``.")
        noise_level = np.percentile(np.abs(data), level)
    above = data > noise_level

    # Detect rising and falling edges
    padded = np.concatenate(([0], above.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        logging.warning("No active pulse found above the noise floor.")

    # Take the longest one
    best = int(np.argmax(ends - starts))
    return data[int(starts[best]) : int(ends[best])]
