#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produce the Somersalo plots.

Numerical implementation of the work from Somersalo et al.[1]_.

.. [1] Erkki Somersalo, Pasi Yla-Oijala, Dieter Proch et Jukka Sarvas. \
«Computational methods for analyzing electron multipacting in RF structures». \
In : Part. Accel. 59 (1998), p. 107-141. \
url : http://cds.cern.ch/record/1120302/files/p107.pdf.

"""
from collections.abc import Callable, Iterable, Sequence
from typing import overload

from matplotlib.axes import Axes
import numpy as np
import pandas as pd


SOMERSALO_ANALYTICAL_PARAMETERS_ONE = {
    1: ((1.69, 2.97, 7.40, 8.69), (1.59, 2.99, 7.40, 8.79)),
    2: ((1.39, 2.97, 7.40, 8.99), (1.31, 3.11, 7.40, 9.19)),
    3: ((1.08, 2.87, 7.40, 9.19), (1.04, 2.83, 7.40, 9.19)),
    4: ((0.860, 2.64, 7.40, 9.19), (0.833, 2.61, 7.41, 9.19)),
    5: ((0.679, 2.46, 7.40, 9.19), (0.674, 2.44, 7.40, 9.19)),
    6: ((0.533, 2.31, 7.40, 9.19), (0.520, 2.30, 7.40, 9.20)),
    7: ((0.411, 2.19, 7.40, 9.19), (0.399, 2.18, 7.40, 9.19)),
    8: ((0.310, 2.08, 7.41, 9.20), (0.310, 2.07, 7.42, 9.19)),  # wrong
}

SOMERSALO_ANALYTICAL_PARAMETERS_TWO = {
    1: ((1.54, 2.98, 9.08, 10.7), (1.51, 2.99, 9.08, 10.7))  # wrong
}


def webdplotdigitizerpoints_to_data(log_power: np.ndarray,
                                    x_0: float, x_1: float,
                                    y_0: float, y_1: float
                                    ) -> np.ndarray:
    """Fit the webplot points and compute Somersalo curve."""
    a = (y_1 - y_0) / (x_1 - x_0)
    b = a * y_1 - x_1
    return a * log_power + b


@overload
def _one_point_analytical(log_power: np.ndarray,
                          order: int,
                          to_dict: bool = True) -> dict[str, np.ndarray]: ...


@overload
def _one_point_analytical(log_power: np.ndarray,
                          order: int,
                          to_dict: bool = False
                          ) -> tuple[np.ndarray, np.ndarray]: ...


def _one_point_analytical(
    log_power: np.ndarray,
        order: int,
        to_dict: bool = False
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    r"""Compute one-point multipactor bands of order ``order``.

    .. note::
        For now, use data manually extracted from Somersalo's fig.

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
    if order not in SOMERSALO_ANALYTICAL_PARAMETERS_ONE:
        raise NotImplementedError
    param_low, param_upp = SOMERSALO_ANALYTICAL_PARAMETERS_ONE[order]
    low = webdplotdigitizerpoints_to_data(log_power, *param_low)
    upp = webdplotdigitizerpoints_to_data(log_power, *param_upp)

    if to_dict:
        out = {
            f'One-point order {order} (lower lim)': low,
            f'One-point order {order} (upper lim)': upp,
        }
        return out
    return low, upp


@overload
def _two_point_analytical(log_power: np.ndarray,
                          order: int,
                          to_dict: bool = True) -> dict[str, np.ndarray]: ...


@overload
def _two_point_analytical(log_power: np.ndarray,
                          order: int,
                          to_dict: bool = False
                          ) -> tuple[np.ndarray, np.ndarray]: ...


def _two_point_analytical(
    log_power: np.ndarray,
    order: int,
    to_dict: bool = False
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    r"""Compute two-point multipactor bands of order ``order``.

    .. note::
        For now, use data manually extracted from Somersalo's fig.

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
    if order not in SOMERSALO_ANALYTICAL_PARAMETERS_TWO:
        raise NotImplementedError
    param_low, param_upp = SOMERSALO_ANALYTICAL_PARAMETERS_TWO[order]
    low = webdplotdigitizerpoints_to_data(log_power, *param_low)
    upp = webdplotdigitizerpoints_to_data(log_power, *param_upp)

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
    fun = _somersalo_analytical_fun(points)
    somersalo_theory = {}
    for order in orders:
        somersalo_theory.update(fun(log_power, order, to_dict=True))
    df_somersalo = pd.DataFrame(somersalo_theory, index=log_power)
    df_somersalo.plot(ax=ax, **plot_kw)


def _somersalo_analytical_fun(points: str | int) -> Callable:
    """Get one or two point Somersalo function."""
    one_allowed = ('one', 'One', 1)
    two_allowed = ('two', 'Two', 2)
    if points in one_allowed:
        return _one_point_analytical
    if points in two_allowed:
        return _two_point_analytical

    raise IOError(f"{points = } not recognized. Must be in {one_allowed}"
                  f"or {two_allowed}")


def measured_to_somersalo_coordinates(powers_kw: np.ndarray | list[float],
                                      d_mm: float,
                                      freq_ghz: float,
                                      z_ohm: float,
                                      ) -> tuple[np.ndarray, np.ndarray]:
    """Convert measured data to coordinates for Somersalo plot."""
    x_coordinates = np.log10(powers_kw)
    n_coordinates = len(x_coordinates)
    y_coordinates_one = np.full(
        n_coordinates,
        np.log10((d_mm * freq_ghz)**4 * z_ohm)
    )
    y_coordinates_two = np.full(
        n_coordinates,
        np.log10((d_mm * freq_ghz)**4 * z_ohm**2)
    )
    one_point = np.column_stack((x_coordinates, y_coordinates_one))
    two_point = np.column_stack((x_coordinates, y_coordinates_two))
    return one_point, two_point
