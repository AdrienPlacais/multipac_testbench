"""Define useful functions to filter data."""

import logging
from typing import overload

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_closing, binary_opening


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
    no_change_value: bool = True,
    default_first_value: bool | None = True,
) -> bool: ...


@overload
def array_is_growing(
    array: NDArray[np.float64],
    index: int,
    width: int = 10,
    tol: float = 1e-5,
    no_change_value: None = None,
    default_first_value: bool | None = True,
) -> bool | None: ...


def array_is_growing(
    array: NDArray[np.float64],
    index: int,
    width: int = 10,
    tol: float = 1e-5,
    no_change_value: bool | None = None,
    default_first_value: bool | None = True,
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
        ``array[idx+width/2]`` is lower than ``tol``, we return
        ``no_change_value``.
    default_first_value :
        Default return for the first values. The default is True, which means
        that we suppose that power increases at the start.
    no_change_value :
        Default value for when no change in array was detected.

    Returns
    -------
        If the array is locally increasing, ``no_change_value`` if array is
        locally constant.

    """
    semi_width = width // 2
    if index < semi_width:
        return default_first_value
    if index >= len(array) - semi_width:
        return no_change_value

    local_diff = array[index + semi_width] - array[index - semi_width]
    if abs(local_diff) < tol:
        return no_change_value
    if local_diff < 0.0:
        return False
    return True


def clean_boolean_mask(
    mask: NDArray, min_true: int, max_false_gap: int
) -> NDArray[np.bool]:
    """Remove isolated True and False from ``mask``.

    Parameters
    ----------
    mask :
        Boolean mask array. Typically, a multipactor array.
    min_true :
        Minimum size for the multipactor zone. Under this number of samples,
        we consider that multipactor detection was a false positive.
    max_false_gap :
        Maximum distance between two multipactor zones. Under this number of
        samples, we consider the multipactor detection was a false negative,
        and the two neighboring multipactor zones are actually a single zone.

    Returns
    -------
        A copy of ``mask`` without the isolated True/False points.

    """
    structure_true = np.ones(min_true, dtype=np.bool)
    structure_false = np.ones(max_false_gap, dtype=np.bool)

    if len(structure_true) > 1:
        mask = binary_opening(mask, structure=structure_true)
    if len(structure_false) > 1:
        mask = binary_closing(mask, structure=structure_false)

    return mask
