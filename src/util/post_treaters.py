#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define various smoothing/smoothing functions for measured data."""
import numpy as np


def running_mean(input_data: np.ndarray,
                 n_mean: int,
                 mode: str = 'full',
                 **kwargs) -> np.ndarray:
    """Compute the runnning mean. Taken from `this link`_.

    .. _this link: https://stackoverflow.com/questions/13728392/\
moving-average-or-running-mean

    Parameters
    ----------
    input_data : np.ndarray
        Data to smooth of shape ``N``.
    n_mean : int
        Number of points on which running mean is ran.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

        (taken from numpy documentation)

    Returns
    -------
    np.ndarray
        Smoothed data.

    """
    return np.convolve(input_data, np.ones(n_mean) / n_mean, mode=mode)
