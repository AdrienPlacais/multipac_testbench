"""Define useful relations."""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def powers_to_reflection(
    forward_power: NDArray[np.float64],
    reflected_power: NDArray[np.float64],
    name: str,
    warn_reflected_higher_than_forward: bool = True,
    warn_gamma_too_close_to_unity: bool = True,
    tol: float = 5e-2,
) -> pd.Series:
    r"""Compute the reflection coefficient :math:`R`.

    We use the definition:

    .. math::

        R = \frac{V_r}{V_f} = \sqrt{\frac{P_r}{P_f}}

    """
    reflection_coefficient = np.abs(np.sqrt(reflected_power / forward_power))

    mask = reflection_coefficient > 1.0
    n_invalid = np.count_nonzero(mask)
    if n_invalid > 0:
        reflection_coefficient[mask] = np.nan
        if warn_reflected_higher_than_forward:
            logging.warning(
                f"{n_invalid} points were removed in R calculation, where "
                "reflected power was higher than forward power."
            )

    mask = np.isclose(reflection_coefficient, 1.0, atol=tol)
    n_invalid = np.count_nonzero(mask)
    if n_invalid > 0:
        reflection_coefficient[mask] = np.nan
        if warn_gamma_too_close_to_unity:
            logging.warning(
                f"{n_invalid} points were removed in R calculation, where "
                "reflected power was too close to forward power. Tolerance "
                f"was: {tol = }."
            )
    return pd.Series(reflection_coefficient, name=name)


def reflection_to_swr(
    reflection_coefficient: NDArray[np.float64], name: str
) -> pd.Series:
    r"""Compute the :math:`SWR`.

    We use the definition:

    .. math::

        SWR = \frac{1 + R}{1 - R}

    where :math:`R` is the reflection coefficient.

    """
    swr = (1.0 + reflection_coefficient) / (1.0 - reflection_coefficient)
    return pd.Series(swr, name=name)


def swr_to_reflection(swr: NDArray[np.float64], name: str) -> pd.Series:
    r"""Compute the reflection coefficient :math:`R`.

    We use the relation:

    .. math::

        R = \frac{SWR - 1}{SWR + 1}

    """
    return pd.Series((swr - 1.0) / (swr + 1.0), name=name)
