"""Define tests for :class:`.Instrument`.

.. todo::
    Very incomplete.

"""

import numpy as np
import pandas as pd
from multipac_testbench.instruments import Instrument


def test_growth_mask_classic() -> None:
    """Check that growth mask works as expected."""
    data = pd.Series([-5, 0, 5, 0, -5])
    expected = np.array([True, True, True, False, False])
    i = Instrument("test instrument", data, np.nan)
    returned = i.growth_mask(width=2)
    assert (expected == returned).all()


def test_growth_array() -> None:
    """Check that :func:`.power_extrema` will have correct args."""
    # fmt: off
    data = pd.Series([
        -10, -5, 0, 5,
        10, 5, 0, -5,
        -10, -5, 0, 5,
        10, 5, 0, -5,
        -10,
    ])
    expected = np.array(
        [0.0, 1.0, 1.0, 1.0,
         0.0, -1.0, -1.0, -1.0,
         0.0, 1.0, 1.0, 1.0,
         0.0, -1.0, -1.0, -1.0,
         0.0
    ])
    # fmt: on
    i = Instrument("test instrument", data, np.nan)
    returned = i.growth_array()
    assert (expected == returned).all()
