#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produce the Somersalo plots.

Numerical implementation of the work from Somersalo et al.[1]_.

.. [1] Erkki Somersalo, Pasi Yla-Oijala, Dieter Proch et Jukka Sarvas. \
«Computational methods for analyzing electron multipacting in RF structures». \
In : Part. Accel. 59 (1998), p. 107-141. \
url : http://cds.cern.ch/record/1120302/files/p107.pdf.

"""
from collections.abc import Callable, Sequence
from typing import overload

from matplotlib.axes import Axes
import numpy as np
import pandas as pd


@overload
def _one_point(log_power: np.ndarray,
               order: int,
               to_dict: bool = True) -> dict[str, np.ndarray]: ...


@overload
def _one_point(log_power: np.ndarray,
               order: int,
               to_dict: bool = False) -> tuple[np.ndarray, np.ndarray]: ...


def _one_point(log_power: np.ndarray,
               order: int,
               to_dict: bool = False
               ) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    r"""Compute one-point multipactor bands of order ``order``.

    Parameters
    ----------
    log_power : np.ndarray
        Log (base 10) of power in kW.
    order : int
        Order of the multipacting band.

    Returns
    -------
    np.ndarray, np.ndarray | dict[str, np.ndarray]
        Lower and upper multipactor limits, in :math:`(\mathrm{GHz} \times
        \mathrm{mm})^4 \times \Omega`.

    """
    low = 3. * log_power + order
    upp = 3.5 * log_power + order
    if to_dict:
        out = {
            f'One-point order {order} (lower lim)': low,
            f'One-point order {order} (upper lim)': upp,
        }
        return out
    return low, upp


@overload
def _two_point(log_power: np.ndarray,
               order: int,
               to_dict: bool = True) -> dict[str, np.ndarray]: ...


@overload
def _two_point(log_power: np.ndarray,
               order: int,
               to_dict: bool = False) -> tuple[np.ndarray, np.ndarray]: ...


def _two_point(log_power: np.ndarray,
               order: int,
               to_dict: bool = False
               ) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    r"""Compute two-point multipactor bands of order ``order``.

    Parameters
    ----------
    log_power : np.ndarray
        Log (base 10) of power in kW.
    order : int
        Order of the multipacting band.

    Returns
    -------
    np.ndarray, np.ndarray | dict[str, np.ndarray]
        Lower and upper multipactor limits, in :math:`(\mathrm{GHz} \times
        \mathrm{mm})^4 \times \Omega^2`.

    """
    low = 4. * log_power + order
    upp = 4.5 * log_power + order

    if to_dict:
        out = {
            f'Two-point order {order} (lower lim)': low,
            f'Two-point order {order} (upper lim)': upp,
        }
        return out
    return low, upp


def plot_somersalo_analytical(points: str | int,
                              log_power: np.ndarray,
                              orders: Sequence[int],
                              ax: Axes,
                              **plot_kw,
                              ) -> None:
    """Compute and plot single Somersalo plot, several orders."""
    fun = _proper_somersalo_func(points)
    somersalo_theory = {}
    for order in orders:
        somersalo_theory.update(fun(log_power, order, to_dict=True))
    df_somersalo = pd.DataFrame(somersalo_theory, index=log_power)
    df_somersalo.plot(ax=ax, **plot_kw)


def _proper_somersalo_func(points: str | int) -> Callable:
    """Get one or two point Somersalo function."""
    one_allowed = ('one', 'One', 1)
    two_allowed = ('two', 'Two', 2)
    if points in one_allowed:
        return _one_point
    if points in two_allowed:
        return _two_point

    raise IOError(f"{points = } not recognized. Must be in {one_allowed}"
                  f"or {two_allowed}")
