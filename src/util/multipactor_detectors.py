#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define functions to detect where multipactor happens."""
from typing import Any, Sequence
import numpy as np


def quantity_is_above_threshold(quantity: np.ndarray,
                                threshold: float,
                                consecutive_criterion: int = 0,
                                minimum_number_of_points: int = 1,
                                **kwargs: Any) -> np.ndarray:
    """Detect where ``quantity`` is above a given threshold.

    Parameters
    ----------
    quantity : np.ndarray
        Array of measured multipactor quantity.
    threshold : float
        Quantity value above which multipactor is detected.
    consecutive_criterion : int, optional
        If provided, we gather multipactor zones that were separated by
        ``consecutive_criterion`` measure points or less.
    minimum_number_of_points : int, optional
        If provided, the multipactor must happen on at least
        ``minimum_number_of_points`` consecutive points, otherwise we consider
        that it was a measurement flaw. The default is 1.


    Returns
    -------
    np.ndarray
        True where multipactor was detected.

    """
    multipactor = quantity >= threshold

    if consecutive_criterion > 0:
        multipactor = _merge_consecutive(multipactor, consecutive_criterion)

    if minimum_number_of_points > 1:
        multipactor = _remove_isolated(multipactor, minimum_number_of_points)

    return multipactor


def _merge_consecutive(multipactor: np.ndarray,
                       consecutive_criterion: int) -> np.ndarray:
    """
    Merge multipac zones separated by ``consecutive_criterion`` points.

    For the window slicing:
    https://stackoverflow.com/a/42258242/12188681

    We explore ``multipactor`` with a slicing window of width
    ``consecutive_criterion + 2``. If there is multipactor at the two
    extremities of the window, but some of the points inside the window do not
    have multipacting, we say that multipactor happend here anyway.

    """
    n_points = multipactor.size
    window_width = consecutive_criterion + 2
    indexer = np.arange(window_width)[None, :] \
        + np.arange(n_points + 1 - window_width)[:, None]

    for i, window in enumerate(multipactor[indexer]):
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
        multipactor[indexer[i]] = True

    return multipactor


def _remove_isolated(multipactor: np.ndarray,
                     minimum_number_of_points: int) -> np.ndarray:
    """
    Remove mp zones observed on less than ``minimum_number_of_points`` points.

    Basically the same as ``_merge_consecutive``.

    """
    n_points = multipactor.size
    window_width = minimum_number_of_points + 2
    indexer = np.arange(window_width)[None, :] \
        + np.arange(n_points + 1 - window_width)[:, None]

    for i, window in enumerate(multipactor[indexer]):
        if window[0]:
            # multipactor at start of window
            continue

        if window[-1]:
            # multipactor at end of window
            continue

        if (~window).any():
            # not a single multipactor point
            continue

        # no multipacting in the window, except at isolated points
        multipactor[indexer[i]] = False

    return multipactor


def start_and_end_of_contiguous_true_zones(array: np.ndarray
                                           ) -> Sequence[tuple[int, int]]:
    """Get indexes of the contiguous zones where array is True."""
    diff = np.where(np.diff(array))[0]
    n_changes = diff.size

    starts = (diff[::2] + 1).tolist()
    ends = (diff[1::2] + 1).tolist()

    # Multipacting zones are "closed"
    if n_changes % 2 == 0:
        # Multipacting zones are not closed, this is non-multipacting zones
        # that are
        if array[0]:
            starts, ends = ends, starts
            starts.insert(0, 0)
            ends.append(None)

    # One multipacting zone is "open"
    else:
        ends.append(None)

        if array[0]:
            starts, ends = ends, starts
            starts = ends
            starts.insert(0, 0)

    zones = [(start, end) for start, end in zip(starts, ends)]
    return zones