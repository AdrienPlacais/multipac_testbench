"""Define useful functions to filter data."""
from typing import overload
import numpy as np


def remove_trailing_true(data: np.ndarray[np.bool_],
                         n_trailing_points_to_check: int = 50,
                         array_name_for_warning: str = '',
                         ) -> np.ndarray[np.bool_]:
    """Replace trailing ``True`` by False.

    Parameters
    ----------
    data : np.ndarray[np.bool_]
        Boolean array to treat.
    n_trailing_points_to_check : int, optional
        The number of points at the end of array that shall be checked. The
        default is 50, which is a good balance to remove unwanted starts of new
        power cycle at the end of the array.
    array_name_for_warning : str, optional
        Name of the array, to print a more informative warning message.

    Returns
    -------
    np.ndarray[np.bool_]
        Boolean array without trailing True.

    """
    trailing_true = np.where(data[-n_trailing_points_to_check:])[0].shape[0]
    if trailing_true == 0:
        return data

    if array_name_for_warning:
        print("util.filtering.remove_trailing_true warning: there was "
              f"{trailing_true} 'True' points in the last "
              f"{n_trailing_points_to_check} "
              f"points of the {array_name_for_warning} array. Setting it to "
              "False.")
    data[-n_trailing_points_to_check:] = False
    return data


@overload
def array_is_growing(array: np.ndarray,
                     index: int,
                     width: int = 10,
                     tol: float = 1e-5,
                     undetermined_value: bool = True,
                     default_first_value: bool = True,
                     ) -> bool: ...


@overload
def array_is_growing(array: np.ndarray,
                     index: int,
                     width: int = 10,
                     tol: float = 1e-5,
                     undetermined_value: None = None,
                     default_first_value: bool = True,
                     ) -> None: ...


def array_is_growing(array: np.ndarray,
                     index: int,
                     width: int = 10,
                     tol: float = 1e-5,
                     undetermined_value: bool | None = None,
                     default_first_value: bool = True,
                     ) -> bool | None:
    """Tell if ``array`` is locally increasing at ``index``.

    Parameters
    ----------
    array : np.ndarray
        Array under study.
    index : int
        Where you want to know if we increase.
    width : int, optional
        Width of the sample to determine increase. The default is ``10``.
    tol : float, optional
        If absolute value of variation between ``array[idx-width/2]`` and
        ``array[idx+width/2]`` is lower than ``tol``, we return a ``NaN``. The
        default is ``1e-5``.
    default_first_value : bool, optional
        Default return for the first values. The default is True, which means
        that we suppose that power increases at the start.
    undetermined_value : bool | None, optional
        Default value for when the output is undetermined. The default is None.

    Returns
    -------
    bool | None
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
    if local_diff < 0.:
        return False
    return True
