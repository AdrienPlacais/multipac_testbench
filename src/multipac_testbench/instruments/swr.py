r"""Define the SWR virtual probe.

It is the Voltage Standing Wave Ratio.

"""

from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.power import ForwardPower, ReflectedPower
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

    @classmethod
    def from_powers(
        cls,
        forward: ForwardPower,
        reflected: ReflectedPower,
        name: str = "SWR",
        **kwargs,
    ) -> Self:
        """Compute the reflection coefficient from given :class:`.Power`."""
        reflection_coefficient = ReflectionCoefficient(forward, reflected)
        return cls.from_reflection_coefficient(
            reflection_coefficient, name=name, **kwargs
        )

    @classmethod
    def from_reflection_coefficient(
        cls,
        reflection_coefficient: ReflectionCoefficient,
        name: str = "SWR",
        **kwargs,
    ) -> Self:
        """Compute the SWR from given :class:`.ReflectionCoefficient`."""
        data = _compute_swr(reflection_coefficient.data)
        ser_data = pd.Series(data, name=name)
        return cls(name=name, raw_data=ser_data, position=np.nan, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "$SWR$"


def _compute_swr(
    reflection_coefficient: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the :math:`SWR`."""
    swr = (1.0 + reflection_coefficient) / (1.0 - reflection_coefficient)
    return swr
