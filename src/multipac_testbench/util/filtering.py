"""Define useful functions to filter data.

.. todo:: Merge the two remove_isolated functions

"""

import logging
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
