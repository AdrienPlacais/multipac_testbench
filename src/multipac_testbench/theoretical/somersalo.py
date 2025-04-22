"""Produce the Somersalo plots.

Numerical implementation of the work from Somersalo et al.
:cite:`Somersalo1998`.

"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from multipac_testbench.util import helper
from numpy.typing import NDArray
from scipy.optimize import curve_fit

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


def somersalo_base_plot(
    xlim: tuple[float, float] = (0.0, 3.5),
    ylim_one_point: tuple[float, float] = (7.4, 9.2),
    ylim_two_point: tuple[float, float] = (9.1, 11.0),
    **fig_kw,
) -> tuple[Figure, Axes, Axes]:
    """Create base figure and axes for Somersalo.

    Parameters
    ----------
    xlim :
        Limits for the x axis. The default matches the figure from Somersalo's
        paper.
    ylim_one_point :
        Limits for the one-point (left) y axis. The default matches the figure
        from Somersalo's paper.
    ylim_two_point :
        Limits for the two-point (right) y axis. The default matches the figure
        from Somersalo's paper.
    fig_kw :
        Other keyword arguments passed to the Figure constructor.

    Returns
    -------
    out :
        Figure, left and right Axis.

    """
    fig = plt.figure(**fig_kw)
    ax1 = fig.add_subplot(
        111,
        xlabel=r"$\log_{10}(P~\mathrm{[kW]})$",
        ylabel=r"$\log_{10}((f_\mathrm{GHz} d_\mathrm{mm})^4$"
        + r"$\dot Z_\Omega)$",
        xlim=xlim,
        ylim=ylim_one_point,
    )
    ax1.grid(True)
    ax2 = plt.twinx(ax1)
    ax2.set_ylabel(
        r"$\log_{10}((f_\mathrm{GHz} d_\mathrm{mm})^4 \dot Z_\Omega^2)$"
    )
    ax2.set_ylim(ylim_two_point)
    return fig, ax1, ax2


def webdplotdigitizerpoints_to_data(
    log_power: NDArray[np.float64],
    x_0: float,
    x_1: float,
    y_0: float,
    y_1: float,
) -> NDArray[np.float64]:
    """Fit the webplot points and compute Somersalo curve."""
    a = (y_1 - y_0) / (x_1 - x_0)
    b = a * y_1 - x_1
    return a * log_power + b


def _one_point_analytical(
    log_power: NDArray[np.float64],
    order: int,
) -> pd.DataFrame:
    r"""Compute one-point multipactor bands of order ``order``.

    .. note::
        For now, use data manually extracted from Somersalo's fig.

    Parameters
    ----------
    log_power :
        Log (base 10) of power in :unit:`kW`.
    order :
        Order of the multipacting band.

    Returns
    -------
    df_one_point :
        Lower and upper multipactor limits, in :math:`(\mathrm{GHz} \times
        \mathrm{mm})^4 \times \Omega`.

    """
    if order not in SOMERSALO_ANALYTICAL_PARAMETERS_ONE:
        raise NotImplementedError
    param_low, param_upp = SOMERSALO_ANALYTICAL_PARAMETERS_ONE[order]
    low = webdplotdigitizerpoints_to_data(log_power, *param_low)
    upp = webdplotdigitizerpoints_to_data(log_power, *param_upp)

    df_one_point = pd.DataFrame(
        data={
            f"One-point order {order} (lower lim)": low,
            f"One-point order {order} (upper lim)": upp,
        },
        index=log_power,
    )
    return df_one_point


def _two_point_analytical(
    log_power: NDArray[np.float64],
    order: int,
) -> pd.DataFrame:
    r"""Compute two-point multipactor bands of order ``order``.

    .. note::
        For now, use data manually extracted from Somersalo's fig.

    Parameters
    ----------
    log_power :
        Log (base 10) of power in kW.
    order :
        Order of the multipacting band.

    Returns
    -------
    pd.DataFrame
        Lower and upper multipactor limits, in :math:`(\mathrm{GHz} \times
        \mathrm{mm})^4 \times \Omega^2`.

    """
    if order not in SOMERSALO_ANALYTICAL_PARAMETERS_TWO:
        raise NotImplementedError
    param_low, param_upp = SOMERSALO_ANALYTICAL_PARAMETERS_TWO[order]
    low = webdplotdigitizerpoints_to_data(log_power, *param_low)
    upp = webdplotdigitizerpoints_to_data(log_power, *param_upp)

    df_two_point = pd.DataFrame(
        data={
            f"Two-point order {order} (lower lim)": low,
            f"Two-point order {order} (upper lim)": upp,
        },
        index=log_power,
    )
    return df_two_point


SOMERSALO_ANALYTICAL_FUNC = {
    "one": _one_point_analytical,
    "One": _one_point_analytical,
    1: _one_point_analytical,
    "two": _two_point_analytical,
    "Two": _two_point_analytical,
    2: _two_point_analytical,
}


def plot_somersalo_analytical(
    points: str | int,
    log_power: np.ndarray,
    orders: Sequence[int],
    ax: Axes,
    **plot_kw,
) -> None:
    """Compute and plot single Somersalo plot, several orders."""
    func = SOMERSALO_ANALYTICAL_FUNC[points]
    df_somersalos = (func(log_power, order) for order in orders)
    df_somersalo = pd.concat(df_somersalos, axis=1)
    df_somersalo.plot(ax=ax, **plot_kw)


def _somersalo_coordinates(
    powers_kw: np.ndarray | list[float],
    d_mm: float,
    freq_ghz: float,
    z_ohm: float,
    mp_test_name: str,
) -> pd.DataFrame:
    """Convert measured data to coordinates for Somersalo plot."""
    x_coordinates = np.log10(powers_kw)
    n_coordinates = len(x_coordinates)
    y_coordinates_one = np.full(
        n_coordinates, np.log10((d_mm * freq_ghz) ** 4 * z_ohm)
    )
    y_coordinates_two = np.full(
        n_coordinates, np.log10((d_mm * freq_ghz) ** 4 * z_ohm**2)
    )
    df_somersalo = pd.DataFrame(
        {
            f"One-point {mp_test_name}": y_coordinates_one,
            f"Two-point {mp_test_name}": y_coordinates_two,
        },
        index=x_coordinates,
    )
    return df_somersalo


def plot_somersalo_measured(
    mp_test_name: str,
    somersalo_data: dict[str, float | np.ndarray],
    ax1: Axes,
    ax2: Axes,
    **plot_kw,
) -> None:
    """Plot the data on Somersalo plot."""
    df_somersalo = _somersalo_coordinates(
        mp_test_name=mp_test_name, **somersalo_data
    )
    for (_, series), ax, marker in zip(
        df_somersalo.items(), (ax1, ax2), ("o", "*")
    ):
        series.plot(ax=ax, marker=marker, lw=0, legend=True, **plot_kw)


def somersalo_scaling_law(reflected: np.ndarray, p_tw: float) -> np.ndarray:
    r"""Compute :math:`P_{MW} = f(P_{TW}, R)`.

    Somersalo et al. :cite:`Somersalo1998` link the mixed wave (:math:`MW`)
    forward power with the traveling wave (:math:`TW`) forward power through
    reflection coefficient :math:`R`.

    .. math::

        P_\mathrm{MW} \sim \frac{1}{(1 + R)^2}P_\mathrm{TW}

    Parameters
    ----------
    reflected :
        Reflection coefficient.
    p_tw :
        Traveling Wave lower threshold (power).

    Returns
    -------
    p_mw :
        Mixed Wave lower threshold.

    """
    p_mw = p_tw / (1.0 + reflected) ** 2
    return p_mw


def fit_somersalo_scaling(
    df_somersalo: pd.DataFrame,
    full_output: bool,
    plot: bool,
    axes: Axes | None = None,
    ls: str = "--",
    **fit_plot_kw,
) -> pd.DataFrame:
    """Fit the Somersalo scaling law over measurements.

    Parameters
    ----------
    df_somersalo :
        DataFrame holding reflection coefficient and forward power. We take the
        proper columns by looking for ``'ReflectionCoefficient'`` and
        ``'ForwardPower'`` column names.
    full_output :
        To ask for more detailed information on the fit process.
    plot :
        To plot the fitted scaling law or not.
    axes :
        Axes on which scaling law will be drawn. If not provided, a new Axe
        will be created.

    Returns
    -------
    df_fitted :
        Holds the fitted Somersalo scaling law.

    """
    R = np.linspace(0, 1, 101)
    r_fit = df_somersalo.filter(like="ReflectionCoefficient").values.ravel()
    p_fit = df_somersalo.filter(like="ForwardPower").values.ravel()

    result = curve_fit(
        f=somersalo_scaling_law,
        xdata=r_fit,
        ydata=p_fit,
        full_output=full_output,
    )
    popt = result[0]
    somer_index = r"$P_{TW}$ "

    if full_output:
        r_squared = helper.r_squared(result[2]["fvec"], p_fit)
        somer_index = (
            f"Fit ({somer_index} = {popt[0]:3.1f}W, $r^2$ = {r_squared:3.3f})"
        )

    else:
        somer_index = f"Fit ({somer_index} = {popt[0]:3.1f}W)"

    df_fitted = pd.DataFrame(
        {"$R$": R, somer_index: somersalo_scaling_law(R, *popt)}
    )
    if plot:
        df_fitted.plot(ax=axes, x=0, y=1, grid=True, ls=ls, **fit_plot_kw)
    return df_fitted
