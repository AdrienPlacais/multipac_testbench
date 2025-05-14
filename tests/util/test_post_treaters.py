"""Define tests for the post-treaters."""

import numpy as np
import pytest
from multipac_testbench.util.post_treaters import (
    average_y_for_nearby_x_within_distance,
)
from numpy.testing import assert_array_almost_equal


def test_average_y_at_same_x_classic() -> None:
    """Test normal use of function."""
    y = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
    x = np.array([100.0, 100.0, 200.0, 200.0, 300.0, 300.0])
    expected = np.array([0.5, np.nan, 10.5, np.nan, 20.5, np.nan])
    result = average_y_for_nearby_x_within_distance(y, x)
    assert_array_almost_equal(expected, result)


def test_average_y_at_same_x_no_keep_shape() -> None:
    """Test when shape of output should be changed."""
    y = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
    x = np.array([100.0, 100.0, 200.0, 200.0, 300.0, 300.0])
    expected = np.array([0.5, 10.5, 20.5])
    result = average_y_for_nearby_x_within_distance(y, x, keep_shape=False)
    assert_array_almost_equal(expected, result)


def test_average_y_at_same_x_low_max_distance() -> None:
    """Test when average is done on neighbors only."""
    # fmt: off
    y = np.array([0.0, 10.0, 20.0, 10.5,
                  0.5, 11.0, 20.5, 11.5,
                  1.0, 12.0, 21.0, 12.5])
    # Represents three identical cycles
    x = np.array([100.0, 200.0, 300.0, 200.0,
                  100.0, 200.0, 300.0, 200.0,
                  100.0, 200.0, 300.0, 200.0])
    # No consecutive points with max_distance = 1 leads to nothing being
    # performed
    expected = np.array([0.0, 10.0, 20.0, 10.5,
                         0.5, 11.0, 20.5, 11.5,
                         1.0, 12.0, 21.0, 12.5])
    # fmt: on
    result = average_y_for_nearby_x_within_distance(y, x, max_index_distance=1)
    assert_array_almost_equal(expected, result)


def test_average_y_at_same_x_high_max_distance() -> None:
    """Test when average is done on whole test."""
    # fmt: off
    y = np.array([0.0, 10.0, 20.0, 10.5,
                  0.5, 11.0, 20.5, 11.5,
                  1.0, 12.0, 21.0, 12.5])

    # Represents three identical cycles
    x = np.array([100.0, 200.0, 300.0, 200.0,
                  100.0, 200.0, 300.0, 200.0,
                  100.0, 200.0, 300.0, 200.0])
    # No consecutive points but max_distance = 1000 leads to one point per
    # x_data
    expected = np.array([0.5, 11.25, 20.5, np.nan,
                         np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan, np.nan])
    # fmt: on
    result = average_y_for_nearby_x_within_distance(
        y, x, max_index_distance=1000
    )
    assert_array_almost_equal(expected, result)


def test_average_y_at_same_x_shape_mismatch_raises() -> None:
    """Test ValueError is raised when x_data and y_data have different shapes."""
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([1.0, 2.0])
    with pytest.raises(
        ValueError, match="x_data and y_data must have the same shape."
    ):
        average_y_for_nearby_x_within_distance(y, x)


def test_average_y_at_same_x_tol_behavior() -> None:
    """Test that tolerance affects grouping of x values."""
    y = np.array([1.0, 3.0])
    x = np.array([100.0, 100.01])

    # With high tol, they should be averaged
    expected = np.array([2.0, np.nan])
    result = average_y_for_nearby_x_within_distance(y, x, tol=1.0)
    assert_array_almost_equal(expected, result)

    # With very small tol, they should not be averaged
    expected = np.array([1.0, 3.0])
    result = average_y_for_nearby_x_within_distance(y, x, tol=1e-10)
    assert_array_almost_equal(expected, result)


def test_average_y_at_same_x_on_different_numbers_of_points() -> None:
    """Test that tolerance affects grouping of x values."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 21.0])
    x = np.array([100.0, 100.0, 100.0, 100.0, 200.0, 300.0, 300.0])
    expected = np.array([2.5, np.nan, np.nan, np.nan, 10.0, 20.5, np.nan])
    result = average_y_for_nearby_x_within_distance(y, x)
    assert_array_almost_equal(expected, result)
