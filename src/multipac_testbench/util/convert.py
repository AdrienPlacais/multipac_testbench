"""Define functions to switch between physical units."""

import numpy as np
from numpy.typing import NDArray


def watt_to_dBm(power: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert power in :unit:`W` to :unit:`dBm`."""
    return 10.0 * np.log10(power) + 30.0


def dBm_to_watt(power: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert power in :unit:`dBm` to :unit:`W`."""
    return 10.0 ** ((power - 30.0) * 0.1)
