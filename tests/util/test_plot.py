"""Provide tests for plotting functions."""

import logging

import numpy as np
import pandas as pd
import pytest
from multipac_testbench.util.plot import (
    create_df_to_plot,
    styles_from_column_cycle,
)
from pandas.testing import assert_frame_equal


def test_create_df_to_plot_basic_concat() -> None:
    """Test basic concatenation of multiple Series."""
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([4, 5, 6], name="b")
    result = create_df_to_plot([s1, s2])
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_frame_equal(result, expected)


def test_create_df_to_plot_mixed_inputs() -> None:
    """Test basic concatenation of multiple Series."""
    s1 = pd.Series([1, 2, 3], name="a")
    df2 = pd.DataFrame({"b": [4, 5, 6], "c": [7, 8, 9]})
    result = create_df_to_plot([s1, df2])
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    assert_frame_equal(result, expected)


def test_create_df_to_plot_tail() -> None:
    """Test slicing with tail."""
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([4, 5, 6], name="b")
    result = create_df_to_plot([s1, s2], tail=2)
    expected = pd.DataFrame({"a": [2, 3], "b": [5, 6]}, index=[1, 2])
    assert_frame_equal(result, expected)


def test_create_df_to_plot_rename_columns() -> None:
    """Test renaming columns with a list of new names."""
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([4, 5, 6], name="b")
    result = create_df_to_plot([s1, s2], column_names=["x", "y"])
    expected = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert_frame_equal(result, expected)


def test_create_df_to_plot_rename_columns_mixed_inputs() -> None:
    """Test basic concatenation of multiple Series."""
    s1 = pd.Series([1, 2, 3], name="a")
    df2 = pd.DataFrame({"b": [4, 5, 6], "c": [7, 8, 9]})
    result = create_df_to_plot([s1, df2], column_names=["x", "y", "z"])
    expected = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
    assert_frame_equal(result, expected)


def test_create_df_to_plot_rename_single_string_column_name() -> None:
    """Test renaming columns with a single string (for one column)."""
    s1 = pd.Series([1, 2, 3], name="a")
    result = create_df_to_plot([s1], column_names="x")
    expected = pd.DataFrame({"x": [1, 2, 3]})
    assert_frame_equal(result, expected)


def test_create_df_to_plot_duplicate_columns_removed() -> None:
    """Test that duplicate columns are removed from the result."""
    s = pd.Series([1, 2, 3], name="dup")
    result = create_df_to_plot([s, s])
    expected = pd.DataFrame({"dup": [1, 2, 3]})
    assert_frame_equal(result, expected)


def test_create_df_to_plot_invalid_column_rename_length() -> None:
    """Test assertion is raised when number of new names doesn't match."""
    s1 = pd.Series([1, 2, 3], name="a")
    s2 = pd.Series([4, 5, 6], name="b")
    with pytest.raises(AssertionError):
        create_df_to_plot([s1, s2], column_names=["only_one_name"])


def test_create_df_to_plot_warning_and_padding(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test with inputs of differing lengths and verify warning and padding."""

    ser1 = pd.Series([1, 2], name="a")
    ser2 = pd.Series([10, 20, 30], name="b")

    with caplog.at_level(logging.WARNING):
        df = create_df_to_plot([ser1, ser2])

    warnings = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "Not all data sources have the same length" in warnings[0].message

    assert df.shape == (3, 2)
    assert np.isnan(df.iloc[2, 0])
    assert df.iloc[2, 1] == 30


def test_styles_from_column_cycle_basic() -> None:
    """Test that line styles are correctly assigned and cycled."""
    columns = ["col__a", "col__b", "col__c"]
    linestyles = ["-", "--"]
    result = styles_from_column_cycle(columns, linestyles)

    expected = {
        "col__a": {"linestyle": "-"},
        "col__b": {"linestyle": "--"},
        "col__c": {"linestyle": "-"},  # cycles back
    }
    assert result == expected


def test_styles_from_column_cycle_multiple_bases() -> None:
    """Test cycling styles independently for different base column names."""
    columns = ["x__1", "x__2", "y__1", "y__2", "y__3"]
    linestyles = [":", "-."]
    result = styles_from_column_cycle(columns, linestyles)

    expected = {
        "x__1": {"linestyle": ":"},
        "x__2": {"linestyle": "-."},
        "y__1": {"linestyle": ":"},
        "y__2": {"linestyle": "-."},
        "y__3": {"linestyle": ":"},  # cycles back for 'y'
    }
    assert result == expected


def test_styles_from_column_cycle_empty() -> None:
    """Test that an empty column list returns an empty dictionary."""
    result = styles_from_column_cycle([], ["-"])
    assert result == {}


def test_styles_from_column_cycle_single_style() -> None:
    """Test with only one style, all columns get the same style."""
    columns = ["col__a", "col__b", "col__c"]
    result = styles_from_column_cycle(columns, ["--"])
    expected = {
        "col__a": {"linestyle": "--"},
        "col__b": {"linestyle": "--"},
        "col__c": {"linestyle": "--"},
    }
    assert result == expected
