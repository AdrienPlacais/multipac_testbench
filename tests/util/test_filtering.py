"""Provide test functions for the :mod:`.filtering` module."""

import numpy as np
from multipac_testbench.util.filtering import array_is_growing


def test_array_is_growing_classic() -> None:
    """Check growth detection in a simple classic case."""
    array = np.hstack(
        (np.linspace(-3.0, 3.0, 7), np.linspace(2.0, -3.0, 6)),
        dtype=np.float64,
    )

    default_first_value = False
    undetermined_value = None

    returned = [
        array_is_growing(
            array,
            i,
            width=2,
            default_first_value=default_first_value,
            no_change_value=undetermined_value,
        )
        for i in range(len(array))
    ]
    # fmt: off
    expected = [
        default_first_value, True, True, True, True, True,
        undetermined_value,
        False, False, False, False, False, undetermined_value
    ]
    # fmt: on
    assert returned == expected
