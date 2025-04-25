"""Define the RPA."""

from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from numpy.typing import NDArray


class RPAPotential(Instrument):
    """A probe to measure potential on RPA grid."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Grid potential [V]"


class RPACurrent(Instrument):
    """A probe to measure collected current on RPA."""

    def __init__(self, *args, **kwargs) -> None:
        """Just instantiate."""
        return super().__init__(*args, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"RPA current [$\mu$A]"


class RPA(VirtualInstrument):
    """Store the multipactor electrons energy distribution.

    This object is created by :meth:`.InstrumentFactory.run_virtual` when there
    is one :class:`.RPACurrent` and one :class:`.RPAPotential` in its
    ``instruments`` argument.

    """

    @classmethod
    def from_current_and_potential(
        cls,
        rpa_current: RPACurrent,
        rpa_potential: RPAPotential,
        name: str = "RPA",
        **kwargs,
    ) -> Self:
        """Compute the distribution from the current and grid potential."""
        data = _compute_energy_distribution(
            rpa_current.data, rpa_potential.data
        )
        ser_data = pd.Series(data, name=name)
        return cls(name=name, raw_data=ser_data, position=np.nan, **kwargs)

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"Energy distribution [$\mu$A/V]"


def _compute_energy_distribution(
    current: NDArray[np.float64], potential: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Derivate signal to obtain distribution."""
    return np.diff(current) / np.diff(potential)
