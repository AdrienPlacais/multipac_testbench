r"""Define the SWR virtual probe.

It is the Voltage Standing Wave Ratio.

"""

from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.reflection_coefficient import (
    ReflectionCoefficient,
)
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from numpy.typing import NDArray


class SWR(VirtualInstrument):
    r"""Store the Standing Wave Ratio.

    We use the definition:

    .. math::

        SWR = \frac{1 + R}{1 - R}

    where :math:`R` is the reflection coefficient.

    This object is created by :meth:`.InstrumentFactory.run_virtual` when there
    is one :class:`.ForwardPower` and one :class:`.ReflectedPower` in its
    ``instruments`` argument.

    """

    def __init__(
        self,
        name: str,
        raw_data: pd.Series,
        reflection_coefficient: ReflectionCoefficient,
        **kwargs,
    ) -> None:
        """Create object, save :class:`.ReflectionCoefficient` object."""
        super().__init__(name, raw_data, **kwargs)

        self._reflection_coefficient = reflection_coefficient
        self._reflection_coefficient.register_callback(self.recompute)

    def recompute(self) -> pd.Series:
        """Recompute SWR.

        This method is called when one of the :class:`.Power` attributes or
        the :class:`.ReflectionCoefficient` data is changed.

        """
        self._raw_data = _compute_swr(
            self._reflection_coefficient.data, self.name
        )
        return self._raw_data

    @classmethod
    def from_reflection_coefficient(
        cls,
        reflection_coefficient: ReflectionCoefficient,
        name: str = "SWR",
        **kwargs,
    ) -> Self:
        """Compute the SWR from given :class:`.ReflectionCoefficient`."""
        return cls(
            name=name,
            raw_data=_compute_swr(reflection_coefficient.data, name),
            position=np.nan,
            reflection_coefficient=reflection_coefficient,
            **kwargs,
        )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "$SWR$"


def _compute_swr(
    reflection_coefficient: NDArray[np.float64], name: str
) -> pd.Series:
    """Compute the :math:`SWR`."""
    swr = (1.0 + reflection_coefficient) / (1.0 - reflection_coefficient)
    return pd.Series(swr, name=name)
