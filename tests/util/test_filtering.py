"""Provide test functions for the :mod:`.filtering` module."""

import numpy as np
import pytest
from multipac_testbench.util.filtering import retrieve_power_sweep


def test_retrieve_power_sweep_nominal():
    """Check that we can retrieve power sweep in the normal behavior."""
    # fmt: off
    power = np.array([
        # Recording started
        -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0,
        # Power sweep started (start keeping data here)
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        # Power sweep ended (stop keeping data here)
        -10.0, -10.0, -10.0, -10.0, -10.0,
        # RF output turned off
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Recording ended
    ])
    # fmt: on
    start, end, repetitions, delta = (8, 59, 3, 5.0)
    returned = retrieve_power_sweep(power)
    assert (start, end, repetitions, delta) == returned


def test_retrieve_power_sweep_no_power_change_at_start():
    """Check that we can retrieve power sweep in the normal behavior.

    Here, we also check that start of power sweep can be retrieved even if
    there was no change in power at the start.

    """
    # fmt: off
    power = np.array([
        # Recording started
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        # Power sweep started (start keeping data here)
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        # Power sweep ended (stop keeping data here)
        -10.0, -10.0, -10.0, -10.0, -10.0,
        # RF output turned off
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Recording ended
    ])
    # fmt: on
    start, end, repetitions, delta = (8, 59, 3, 5.0)
    returned = retrieve_power_sweep(power)
    assert (start, end, repetitions, delta) == returned


def test_retrieve_power_sweep_start_ascending():
    """Check that we can retrieve power sweep in the normal behavior.

    Check for corner case: the power before the actual power sweep start is
    ``delta`` :unit:`dBm` above ``power_start``.

    """
    # fmt: off
    power = np.array([
        # Recording started
        -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
        # Power sweep started (start keeping data here)
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        # Power sweep ended (stop keeping data here)
        -10.0, -10.0, -10.0, -10.0, -10.0,
        # RF output turned off
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Recording ended
    ])
    # fmt: on
    start, end, repetitions, delta = (8, 59, 3, 5.0)
    returned = retrieve_power_sweep(power)
    assert (start, end, repetitions, delta) == returned


def test_retrieve_power_sweep_interrupted():
    """Check that we can retrieve power sweep even if test was interrupted.

    Case of stopping power sweep with ``STOP`` button in LabView.

    """
    # fmt: off
    power = np.array([
        # Recording started
        -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0,
        # Power sweep started (start keeping data here)
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        # Last complete power step (stop keeping data here)
        -10.0, -10.0,
        # RF output turned off
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Recording ended
    ])
    # fmt: on
    start, end, repetitions, delta = (8, 56, 3, 5.0)
    returned = retrieve_power_sweep(power)
    assert (start, end, repetitions, delta) == returned


@pytest.mark.xfail(
    reason="Not really a problem if some points at the end do not belong to sweep."
)
def test_retrieve_power_sweep_rf_turned_off():
    """Check that we can retrieve power sweep even if test was interrupted.

    Case of stopping power with ``RF output: OFF`` button in LabView.

    """
    # fmt: off
    power = np.array([
        # Recording started
        -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0,
        # Power sweep started (start keeping data here)
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        # Last complete power step (stop keeping data here)
        # RF output turned off
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # Recording ended
    ])
    # fmt: on
    start, end, repetitions, delta = (8, 56, 3, 5.0)
    returned = retrieve_power_sweep(power)
    assert (start, end, repetitions, delta) == returned


def test_retrieve_power_sweep_recording_stopped():
    """Check that we can retrieve power sweep even if test was interrupted.

    Case of stopping recording.

    """
    # fmt: off
    power = np.array([
        # Recording started
        -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0,
        # Power sweep started (start keeping data here)
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        -10.0, -10.0, -10.0,
        -5.0, -5.0, -5.0,
        0.0, 0.0, 0.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0,
        5.0, 5.0, 5.0,
        0.0, 0.0, 0.0,
        -5.0, -5.0, -5.0,
        # Last complete power step (stop keeping data here)
        -10.0,
        # Recording ended
    ])
    # fmt: on
    start, end, repetitions, delta = (8, 56, 3, 5.0)
    returned = retrieve_power_sweep(power)
    assert (start, end, repetitions, delta) == returned
