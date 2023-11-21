#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define functions to detect where multipactor happens."""
from typing import Any
import numpy as np


def detect_multipactor(quantity: np.ndarray,
                       detection_method: str,
                       **kwargs: Any) -> np.ndarray[np.int64]:
    """Detect the multipactor using proper method.

    Parameters
    ----------
    quantity : np.ndarray
        Array to check.
    detection_method : str
        Name of the detection method. Must be in ``IMPLEMENTED_DETECTORS``.
    kwargs : Any
        Transmitted to the multipactor detection method.

    Returns
    -------
    np.ndarray[np.int64]
        Indexes of the measurements where multipactor is detected.

    """
    if detection_method not in IMPLEMENTED_DETECTORS:
        list_of_implemented_detection_methods = \
            list(IMPLEMENTED_DETECTORS.keys())
        print(f"Warning! You asked for {detection_method = }, which is not in "
              f"{list_of_implemented_detection_methods = }. Using "
              "'above_threshold'.")
        detection_method = 'above_threshold'

    detector = IMPLEMENTED_DETECTORS[detection_method]
    return detector(quantity, **kwargs)


def _quantity_is_above_threshold(quantity: np.ndarray,
                                 threshold: float,
                                 **kwargs: Any) -> np.ndarray[np.int64]:
    """Detect where ``quantity`` is above a given threshold.

    Parameters
    ----------
    quantity : np.ndarray
        Array of measured multipactor quantity.
    threshold : float
        Quantity value above which multipactor is detected.

    Returns
    -------
    np.ndarray[np.int64]
        Indexes of the measurements where multipactor is detected.

    """
    return np.where(quantity >= threshold)[0]


IMPLEMENTED_DETECTORS = {
    'above_threshold': _quantity_is_above_threshold,
}  #:
