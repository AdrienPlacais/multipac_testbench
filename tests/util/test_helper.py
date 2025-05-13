"""Provide tests for the helper functions."""

import numpy as np
import pandas as pd
import pytest
from multipac_testbench.util.helper import split_rows_by_mask
from pandas.testing import assert_frame_equal


def test_split_rows_by_mask_series() -> None:
    """Test splitting a Series based on a boolean mask."""
    ser = pd.Series([1, 2, 3], name="data")
    mask = np.array([True, False, True])
    result = split_rows_by_mask(ser, mask, ("__a", "__b"))
    expected = pd.DataFrame(
        {
            "data__a": [1.0, np.nan, 3.0],
            "data__b": [np.nan, 2.0, np.nan],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_mask_dataframe() -> None:
    """Test splitting a DataFrame based on a boolean mask."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    mask = np.array([True, False, True])
    result = split_rows_by_mask(df, mask, ("__a", "__b"))
    expected = pd.DataFrame(
        {
            "col1__a": [1.0, np.nan, 3.0],
            "col1__b": [np.nan, 2.0, np.nan],
            "col2__a": [4.0, np.nan, 6.0],
            "col2__b": [np.nan, 5.0, np.nan],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_mask_empty_input() -> None:
    """Test splitting an empty Series."""
    ser = pd.Series([], dtype=float)
    mask = np.array([], dtype=bool)
    result = split_rows_by_mask(ser, mask)
    expected = pd.DataFrame(
        {
            "0__increasing": pd.Series([], dtype=float),
            "0__decreasing": pd.Series([], dtype=float),
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_mask_all_true() -> None:
    """Test when all values are assigned to the 'true' side of the split."""
    ser = pd.Series([10, 20, 30])
    mask = np.array([True, True, True])
    result = split_rows_by_mask(ser, mask, ("__yes", "__no"))
    expected = pd.DataFrame(
        {
            "0__yes": [10, 20, 30],
            "0__no": [np.nan, np.nan, np.nan],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_mask_all_false() -> None:
    """Test when all values are assigned to the 'false' side of the split."""
    ser = pd.Series([10, 20, 30])
    mask = np.array([False, False, False])
    result = split_rows_by_mask(ser, mask, ("__yes", "__no"))
    expected = pd.DataFrame(
        {
            "0__yes": [np.nan, np.nan, np.nan],
            "0__no": [10, 20, 30],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_mask_invalid_mask_length() -> None:
    """Test that a ValueError is raised when mask length doesn't match input."""
    ser = pd.Series([1, 2, 3])
    mask = np.array([True, False])
    with pytest.raises(ValueError):
        split_rows_by_mask(ser, mask)
