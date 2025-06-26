"""Provide tests for :class:`.Threshold`, :class:`.PowerExtremum`."""

import numpy as np
from multipac_testbench.threshold.threshold import PowerExtremum, power_extrema


def test_power_extrem_creation() -> None:
    """Test normal behavior of extremum creation."""
    growth_array = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0])
    returned = power_extrema(growth_array)
    expected = [
        PowerExtremum(0, "minimum"),
        PowerExtremum(4, "maximum"),
        PowerExtremum(7, "minimum"),
    ]
    assert returned == expected
