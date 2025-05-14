"""Provide tests for the helper functions."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from multipac_testbench.util.helper import (
    split_rows_by_masks,
)
from pandas.testing import assert_frame_equal


def test_split_rows_by_masks_series() -> None:
    ser = pd.Series([1, 2, 3], name="data")
    masks = {
        "__a": np.array([True, False, True]),
        "__b": np.array([False, True, False]),
    }
    result = split_rows_by_masks(ser, masks)
    expected = pd.DataFrame(
        {
            "data__a": [1.0, np.nan, 3.0],
            "data__b": [np.nan, 2.0, np.nan],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_masks_dataframe() -> None:
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    masks = {
        "__a": np.array([True, False, True]),
        "__b": np.array([False, True, False]),
    }
    result = split_rows_by_masks(df, masks)
    expected = pd.DataFrame(
        {
            "col1__a": [1.0, np.nan, 3.0],
            "col1__b": [np.nan, 2.0, np.nan],
            "col2__a": [4.0, np.nan, 6.0],
            "col2__b": [np.nan, 5.0, np.nan],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_masks_empty_series() -> None:
    ser = pd.Series([], dtype=float)
    masks = {
        "__increasing": np.array([], dtype=bool),
        "__decreasing": np.array([], dtype=bool),
    }
    result = split_rows_by_masks(ser, masks)
    expected = pd.DataFrame(
        {
            "0__increasing": pd.Series([], dtype=float),
            "0__decreasing": pd.Series([], dtype=float),
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_masks_all_true() -> None:
    ser = pd.Series([10, 20, 30])
    masks = {
        "__yes": np.array([True, True, True]),
        "__no": np.array([False, False, False]),
    }
    result = split_rows_by_masks(ser, masks)
    expected = pd.DataFrame(
        {
            "0__yes": [10, 20, 30],
            "0__no": [np.nan, np.nan, np.nan],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_masks_all_false() -> None:
    ser = pd.Series([10, 20, 30])
    masks = {
        "__yes": np.array([False, False, False]),
        "__no": np.array([True, True, True]),
    }
    result = split_rows_by_masks(ser, masks)
    expected = pd.DataFrame(
        {
            "0__yes": [np.nan, np.nan, np.nan],
            "0__no": [10, 20, 30],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_masks_invalid_mask_length() -> None:
    ser = pd.Series([1, 2, 3])
    masks = {
        "__bad": np.array([True, False]),  # invalid length
    }
    with pytest.raises(ValueError):
        split_rows_by_masks(ser, masks)


def test_split_rows_by_masks_series_three_masks() -> None:
    """Test splitting a Series using three disjoint masks."""
    ser = pd.Series([100, 200, 300, 400], name="value")
    masks = {
        "__low": np.array([True, False, False, False]),
        "__mid": np.array([False, True, True, False]),
        "__high": np.array([False, False, False, True]),
    }
    result = split_rows_by_masks(ser, masks)
    expected = pd.DataFrame(
        {
            "value__low": [100.0, np.nan, np.nan, np.nan],
            "value__mid": [np.nan, 200.0, 300.0, np.nan],
            "value__high": [np.nan, np.nan, np.nan, 400.0],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_masks_dataframe_three_masks() -> None:
    """Test splitting a DataFrame using three disjoint masks."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40],
        }
    )
    masks = {
        "__first": np.array([True, False, False, False]),
        "__middle": np.array([False, True, True, False]),
        "__last": np.array([False, False, False, True]),
    }
    result = split_rows_by_masks(df, masks)
    expected = pd.DataFrame(
        {
            "a__first": [1.0, np.nan, np.nan, np.nan],
            "a__middle": [np.nan, 2.0, 3.0, np.nan],
            "a__last": [np.nan, np.nan, np.nan, 4.0],
            "b__first": [10.0, np.nan, np.nan, np.nan],
            "b__middle": [np.nan, 20.0, 30.0, np.nan],
            "b__last": [np.nan, np.nan, np.nan, 40.0],
        }
    )
    assert_frame_equal(result, expected)


def test_split_rows_by_masks_overlapping_masks_raises() -> None:
    """Test that overlapping masks raise a ValueError."""
    df = pd.Series([1, 2, 3], name="x")
    masks = {
        "__m1": np.array([True, False, True]),
        "__m2": np.array([False, True, True]),  # Overlap at index 2
    }
    with pytest.raises(ValueError, match="Masks must be disjoint"):
        split_rows_by_masks(df, masks)


def test_split_rows_by_masks_no_double_underscore() -> None:
    ser = pd.Series([1, 2, 3], name="data")
    masks = {
        "__a": np.array([True, False, True]),
        "b": np.array([False, True, False]),
    }
    with patch("logging.warning") as mock_warning:
        result = split_rows_by_masks(ser, masks)
        expected = pd.DataFrame(
            {
                "data__a": [1.0, np.nan, 3.0],
                "datab": [np.nan, 2.0, np.nan],
            }
        )
        assert_frame_equal(result, expected)
        mock_warning.assert_called_once()
