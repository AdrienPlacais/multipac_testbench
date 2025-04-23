r"""Define the reflection coefficient virtual probe.

As for now, it is always a real, i.e. it is :math:`R = |\Gamma|`.

"""

import logging
from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.power import ForwardPower, ReflectedPower
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from numpy.typing import NDArray


class ReflectionCoefficient(VirtualInstrument):
    r"""Store the reflection coefficient.

    We use the definition:

    .. math::

        R = \frac{V_r}{V_f} = \sqrt{\frac{P_r}{P_f}}

    where :math:`P_r` is the reflected power and :math:`P_f` is the forward
    power.
    This object is created by :meth:`.InstrumentFactory.run_virtual` when there
    is one :class:`.ForwardPower` and one :class:`.ReflectedPower` in its
    ``instruments`` argument.

    """

    @classmethod
    def from_powers(
        cls,
        forward: ForwardPower,
        reflected: ReflectedPower,
        name: str = "Reflection_coefficient",
        **kwargs,
    ) -> Self:
        """Compute the reflection coefficient from given :class:`.Power`."""
        data = _compute_reflection_coef(forward.data, reflected.data)
        ser_data = pd.Series(data, name=name)
        return cls(name=name, raw_data=ser_data, position=np.nan, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "Reflection coefficient $R$"


def _compute_reflection_coef(
    forward_data: NDArray[np.float64],
    reflected_data: NDArray[np.float64],
    warn_reflected_higher_than_forward: bool = True,
    warn_gamma_too_close_to_unity: bool = True,
    tol: float = 5e-2,
) -> NDArray[np.float64]:
    r"""Compute the reflection coefficient :math:`R`."""
    reflection_coefficient = np.abs(np.sqrt(reflected_data / forward_data))

    invalid_indexes = np.where(reflection_coefficient > 1.0)[0]
    n_invalid = len(invalid_indexes)
    if n_invalid > 0:
        reflection_coefficient[invalid_indexes] = np.nan
        if warn_reflected_higher_than_forward:
            logging.warning(
                f"{n_invalid} points were removed in R calculation, where "
                "reflected power was higher than forward power."
            )

    invalid_indexes = np.where(np.abs(reflection_coefficient - 1.0) < tol)[0]
    n_invalid = len(invalid_indexes)
    if n_invalid > 0:
        reflection_coefficient[invalid_indexes] = np.nan
        if warn_gamma_too_close_to_unity:
            logging.warning(
                f"{n_invalid} points were removed in R calculation, where "
                "reflected power was too close to forward power. Tolerance "
                f"was: {tol = }."
            )
    return reflection_coefficient
