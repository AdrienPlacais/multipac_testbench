#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produce the Somersalo plots.

Numerical implementation of the work from Somersalo et al.[1]_.

.. [1] Erkki Somersalo, Pasi Yla-Oijala, Dieter Proch et Jukka Sarvas. \
«Computational methods for analyzing electron multipacting in RF structures». \
In : Part. Accel. 59 (1998), p. 107-141. \
url : http://cds.cern.ch/record/1120302/files/p107.pdf.

"""
import numpy as np


def one_point(log_power: np.ndarray,
              order: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute one-point multipactor bands of order ``order``.

    Parameters
    ----------
    log_power : np.ndarray
        Log (base 10) of power in kW.
    order : int
        Order of the multipacting band.

    Returns
    -------
    np.ndarray, np.ndarray
        Lower and upper multipactor limits, in :math:`(\mathrm{GHz} \times
        \mathrm{mm})^4 \times \Omega`.

    """
    low = 3. * log_power
    upp = 3.5 * log_power
    return low, upp


def two_point(log_power: np.ndarray,
              order: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute two-point multipactor bands of order ``order``.

    Parameters
    ----------
    log_power : np.ndarray
        Log (base 10) of power in kW.
    order : int
        Order of the multipacting band.

    Returns
    -------
    np.ndarray, np.ndarray
        Lower and upper multipactor limits, in :math:`(\mathrm{GHz} \times
        \mathrm{mm})^4 \times \Omega^2`.

    """
    low = 4. * log_power
    upp = 4.5 * log_power
    return low, upp
