"""Define useful functions to filter data.

.. todo:: Merge the two remove_isolated functions

"""

import logging
from collections import Counter
from typing import overload

import numpy as np
from numpy.typing import NDArray


def remove_trailing_true(
    data: NDArray[np.bool],
    n_trailing_points_to_check: int = 50,
    array_name_for_warning: str = "",
) -> NDArray[np.bool]:
    """Replace trailing ``True`` by False.

    Parameters
    ----------
    data :
        Boolean array to treat.
    n_trailing_points_to_check :
        The number of points at the end of array that shall be checked. The
        default is 50, which is a good balance to remove unwanted starts of new
        power cycle at the end of the array.
    array_name_for_warning :
        Name of the array, to print a more informative warning message.

    Returns
    -------
    data :
        Boolean array without trailing True.

    """
    trailing_true = np.where(data[-n_trailing_points_to_check:])[0].shape[0]
    if trailing_true == 0:
        return data

    if array_name_for_warning:
        logging.warning(
            f"There was {trailing_true} 'True' points in the last "
            f"{n_trailing_points_to_check} points of the "
            f"{array_name_for_warning} array. Setting it to False."
        )
    data[-n_trailing_points_to_check:] = False
    return data


@overload
def array_is_growing(
    array: NDArray[np.float64],
    index: int,
    width: int = 10,
    tol: float = 1e-5,
    undetermined_value: bool = True,
    default_first_value: bool = True,
) -> bool: ...


@overload
def array_is_growing(
    array: NDArray[np.float64],
    index: int,
    width: int = 10,
    tol: float = 1e-5,
    undetermined_value: None = None,
    default_first_value: bool = True,
) -> None: ...


def array_is_growing(
    array: NDArray[np.float64],
    index: int,
    width: int = 10,
    tol: float = 1e-5,
    undetermined_value: bool | None = None,
    default_first_value: bool = True,
) -> bool | None:
    """Tell if ``array`` is locally increasing at ``index``.

    Parameters
    ----------
    array :
        Array under study.
    index :
        Where you want to know if we increase.
    width :
        Width of the sample to determine increase.
    tol :
        If absolute value of variation between ``array[idx-width/2]`` and
        ``array[idx+width/2]`` is lower than ``tol``, we return a ``NaN``.
    default_first_value :
        Default return for the first values. The default is True, which means
        that we suppose that power increases at the start.
    undetermined_value :
        Default value for when the output is undetermined.

    Returns
    -------
    is_growing :
        If the array is locally increasing, ``undetermined_value`` if
        undetermined.

    """
    semi_width = width // 2
    if index <= semi_width:
        return default_first_value
    if index >= len(array) - semi_width:
        return undetermined_value

    local_diff = array[index + semi_width] - array[index - semi_width]
    if abs(local_diff) < tol:
        return undetermined_value
    if local_diff < 0.0:
        return False
    return True


def remove_isolated_true(
    array: NDArray[np.bool], minimum_number_of_points: int
) -> NDArray[np.bool]:
    """Remove 'True' observed on less than ``minimum_number_of_points`` points.

    Basically the same as ``_merge_consecutive``.

    """
    n_points = array.size
    window_width = minimum_number_of_points + 2
    indexer = (
        np.arange(window_width)[None, :]
        + np.arange(n_points + 1 - window_width)[:, None]
    )

    window: NDArray[np.bool]
    for i, window in enumerate(array[indexer]):
        if window[0]:
            # True at start of window
            continue

        if window[-1]:
            # True at end of window
            continue

        if not window.any():
            # Not a single True in the window
            continue

        # True in isolated points in the window: do something!!
        array[indexer[i]] = False

    return array


def remove_isolated_false(
    array: NDArray[np.bool], consecutive_criterion: int
) -> NDArray[np.bool]:
    """
    Merge multipac zones separated by ``consecutive_criterion`` points.

    For the window slicing:
    https://stackoverflow.com/a/42258242/12188681

    We explore ``array`` with a slicing window of width
    ``consecutive_criterion + 2``. If there is multipactor at the two
    extremities of the window, but some of the points inside the window do not
    have multipacting, we say that multipactor happend here anyway.

    """
    n_points = array.size
    window_width = consecutive_criterion + 2
    indexer = (
        np.arange(window_width)[None, :]
        + np.arange(n_points + 1 - window_width)[:, None]
    )

    for i, window in enumerate(array[indexer]):
        if not window[0]:
            # no multipactor at start of window
            continue

        if not window[-1]:
            # no multipactor at end of window
            continue

        if window.all():
            # already multipactor everywhere in the window
            continue

        # multipactor at the start and end of window, with "holes" between
        array[indexer[i]] = True

    return array


def retrieve_power_sweep(
    power: NDArray[np.float64],
) -> tuple[int, int, int, float]:
    """Determine start and end of power sweeps, number of points per power.

    Parameters
    ----------
    power :
        Power array. Typically, content of the ``NI9205_dBm`` array.

    Returns
    -------
    int :
        Index at which power sweep started.
    int :
        Index at which power sweep stopped.
    int :
        Number of contiguous points at same power (the "triggers to wait for"
        parameter in LabVIEW).
    float :
        Power step in :unit:`dBm`.

    """
    repetitions = _most_common_count_of_contiguous_identical(power)
    start = _staircase_beginning(power, repetitions)
    delta = _determine_delta(power, start, repetitions)
    end = _find_end(power, start, repetitions, delta)
    return start, end, repetitions, delta


def _most_common_count_of_contiguous_identical(
    power: NDArray[np.float64], tol: float = 1e-10
) -> int:
    """Get most common count of contiguous identical values in ``power``."""
    if len(power) == 0:
        raise ValueError("Power array is empty.")

    power_step_lengths = []
    current_power = power[0]
    current_length = 1

    for p in power[1:]:
        if abs(p - current_power) < tol:
            current_length += 1
            continue
        power_step_lengths.append(current_length)
        current_power = p
        current_length = 1
    power_step_lengths.append(current_length)

    most_common = Counter(power_step_lengths).most_common(1)
    return most_common[0][0]


def _staircase_beginning(
    power: NDArray[np.float64], repetitions: int, tol: float = 1e-10
) -> int:
    """Find the index where a consistent, ascending staircase sweep starts.

    The sweep must consist of at least 3 steps (i.e., 3 ``repetitions``
    values), where:

    - each step is a block of repeated values (within ``tol``),
    - step heights are constant (within ``tol``),
    - steps increase monotonically.

    Parameters
    ----------
    power :
        1D array of power values.
    repetitions :
        Number of identical values per step.
    tol :
        Tolerance for comparing floating point values.

    Returns
    -------
    int
        Index in ``power`` where the staircase starts.

    Raises
    ------
    ValueError
        If no valid staircase is found.

    """
    n = len(power)
    max_start = n - 3 * repetitions
    # We will need at least 3 blocks to verify step height

    for i in range(max_start + 1):
        # Three candidate stairs
        candidate_stairs = [
            power[i + j * repetitions : i + (j + 1) * repetitions]
            for j in range(3)
        ]
        if any(len(stair) < repetitions for stair in candidate_stairs):
            continue

        heights = [np.mean(block) for block in candidate_stairs]
        if any(
            np.max(np.abs(block - val)) >= tol
            for block, val in zip(candidate_stairs, heights)
        ):
            # There are different values in current block
            # ("current stair is not flat")
            continue

        delta_height_1 = heights[1] - heights[0]
        delta_height_2 = heights[2] - heights[1]

        if abs(delta_height_1 - delta_height_2) > tol or delta_height_1 <= tol:
            # The stairs have inconsistent height, or are not ascending
            continue

        return i

    raise ValueError("Start of consistent staircase sweep not found.")


def _determine_delta(
    power: NDArray[np.float64],
    start: int,
    repetitions: int,
    tol: float = 1e-10,
) -> float:
    """Determine the delta (step height) of the staircase sweep.

    Parameters
    ----------
    power :
        Array of power values.
    start :
        Starting index of the sweep.
    repetitions :
        Number of identical values per step.
    tol :
        Tolerance for floating point comparisons.

    Returns
    -------
    float
        The most common step height (delta).

    Raises
    ------
    ValueError
        If the first steps do not have consistent height.
    """
    step_blocks = [
        power[start + i * repetitions : start + (i + 1) * repetitions]
        for i in range(3)
    ]
    if any(len(block) < repetitions for block in step_blocks):
        raise ValueError("Not enough data to determine delta.")

    means = [np.mean(block) for block in step_blocks]
    step1 = means[1] - means[0]
    step2 = means[2] - means[1]

    if abs(step1 - step2) > tol:
        raise ValueError("Inconsistent step height in initial sweep steps.")

    return float(step1)


def _find_end(
    power: NDArray[np.float64],
    start: int,
    repetitions: int,
    delta: float,
    tol: float = 1e-10,
) -> int:
    """Find the index where the sweep ends.

    Parameters
    ----------
    power :
        Array of power values.
    start :
        Starting index of the sweep.
    repetitions :
        Number of identical values per step.
    delta :
        Step height of the sweep.
    tol :
        Tolerance for floating point comparisons.

    Returns
    -------
    int
        The index after the last valid step of the sweep.

    """
    prev_mean = np.mean(power[start : start + repetitions])
    i = 1

    while True:
        block_start = start + i * repetitions
        block_end = block_start + repetitions
        if block_end > len(power):
            break

        block = power[block_start:block_end]
        mean = np.mean(block)
        step = mean - prev_mean

        if not np.isclose(abs(step), delta, atol=tol):
            break

        prev_mean = mean
        i += 1

    return start + i * repetitions
