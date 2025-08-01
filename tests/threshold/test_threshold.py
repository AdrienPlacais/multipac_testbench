"""Provide tests for :class:`.Threshold`, :class:`.PowerExtremum`.

.. todo::
    Test other things than "normal behavior"

"""

import numpy as np
from multipac_testbench.threshold.threshold import (
    PowerExtremum,
    Threshold,
    create_power_extrema,
    create_thresholds,
)


def test_tresholds_creation() -> None:
    """Test normal behavior of thresholds creation."""
    # fmt: off
    growth_array = np.array(
        [
            0.0, 1.0, 1.0, 1.0,
            0.0, -1.0, -1.0, -1.0,
            0.0, 1.0, 1.0, 1.0,
            0.0, -1.0, -1.0, -1.0,
            0.0,
        ]
    )
    multipactor = np.array(
        [
            False, False, True, True, # | did not reached second threshold
            True, True, True, False,  # |
            False, False, True, False, # | here, we reached second threshold
            False, True, True, False,  # |
            False
        ])
    # fmt: on
    expected = [
        Threshold(2, "lower", "enter", "_", 42.0),
        Threshold(6, "lower", "exit", "_", 42.0),
        Threshold(10, "lower", "enter", "_", 42.0),
        Threshold(10, "upper", "exit", "_", 42.0),
        Threshold(13, "upper", "enter", "_", 42.0),
        Threshold(14, "lower", "exit", "_", 42.0),
    ]
    returned = create_thresholds(multipactor, growth_array, "_", 42.0)
    assert returned == expected


def test_power_extrema_creation() -> None:
    """Test normal behavior of extremum creation."""
    growth_array = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0])
    returned = create_power_extrema(growth_array)
    expected = [
        PowerExtremum(0, "minimum"),
        PowerExtremum(4, "maximum"),
        PowerExtremum(7, "minimum"),
    ]
    assert returned == expected
