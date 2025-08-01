r"""Define the reflection coefficient virtual probe.

As for now, it is always a real, i.e. it is :math:`R = |\Gamma|`.

"""

from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.power import ForwardPower, ReflectedPower
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from multipac_testbench.util.physics import powers_to_reflection


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

    def __init__(
        self,
        name: str,
        raw_data: pd.Series,
        forward: ForwardPower,
        reflected: ReflectedPower,
        **kwargs,
    ) -> None:
        """Create object, save :class:`.Power` objects."""
        super().__init__(name, raw_data, **kwargs)

        self._forward = forward
        self._forward.register_callback(self.recompute)
        self._reflected = reflected
        self._reflected.register_callback(self.recompute)

    def recompute(self) -> pd.Series:
        """Recompute reflection coefficient.

        This method is called when one of the stored :class:`.Power` attributes
        data is changed.

        Note
        ----
        Also triggers the recalculation of :class:`.SWR`.

        """
        self._raw_data = powers_to_reflection(
            self._forward.data,
            self._reflected.data,
            self.name,
        )
        self._notify_callbacks()
        return self._raw_data

    @classmethod
    def from_powers(
        cls,
        forward: ForwardPower,
        reflected: ReflectedPower,
        name: str = "Reflection_coefficient",
        **kwargs,
    ) -> Self:
        """Compute the reflection coefficient from given :class:`.Power`."""
        return cls(
            name=name,
            raw_data=powers_to_reflection(forward.data, reflected.data, name),
            position=np.nan,
            forward=forward,
            reflected=reflected,
            **kwargs,
        )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "Reflection coefficient $R$"
