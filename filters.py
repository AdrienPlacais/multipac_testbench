#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define various smoothing/smoothing functions for measured data."""
import numpy as np


def smooth(smooth_function_name: str,
           input_data: np.ndarray,
           *args,
           **kwargs) -> np.ndarray:
    """Call the proper smoothing function."""
    if smooth_function_name not in IMPLEMENTED:
        print(f"Warning! {smooth_function_name = } is not in the list of "
              f"implemented smooths, i.e. {tuple(IMPLEMENTED.keys())}."
              "Falling back on 'running_mean'.")
        smooth_function_name = 'running_mean'
    smooth_function = IMPLEMENTED[smooth_function_name]
    return smooth_function(input_data, *args, **kwargs)


def _running_mean(input_data: np.ndarray,
                  n_mean: int,
                  mode: str = 'full',
                  **kwargs
                  ) -> np.ndarray:
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
        From :func:`np.convolve` documentation.
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

    Returns
    -------
    np.ndarray
        Smoothen data.

    """
    return np.convolve(input_data, np.ones(n_mean) / n_mean, mode=mode)


IMPLEMENTED = {
    'running_mean': _running_mean,
}  #:
