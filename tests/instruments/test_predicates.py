"""Check that predicates and filtering logic works as expected.

.. note::
   We use ``pytest_lazy_fixtures``. It allows to use a fixture in a pytest
   parameters.

"""

from abc import ABCMeta

import numpy as np
import pandas as pd
import pytest
from multipac_testbench.instruments import Instrument
from multipac_testbench.instruments.current_probe import CurrentProbe
from multipac_testbench.instruments.penning import Penning
from multipac_testbench.instruments.predicates import (
    INSTRUMENT_FILTER,
    INSTRUMENTS_ID,
    InstrumentFilteringError,
    combine_predicates,
    dummy_instrument_filter,
    filter_instruments,
    instrument_excluder,
    instrument_name_selector,
    instrument_type_selector,
    measurement_point_excluder,
)
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from pytest_lazy_fixtures import lf, lfc

# Sentinel value indicating that a test case should raise an error.
RAISES = object()


@pytest.fixture
def data() -> pd.Series:
    return pd.Series([1, 2, 3])


@pytest.fixture
def instruments(data: pd.Series) -> list[Instrument]:
    """Give dummy instruments."""
    return [
        Instrument("first", data, np.nan),
        Instrument("second", data, np.nan),
        CurrentProbe("third", data, np.nan),
        Penning("fourth", data, np.nan),
        CurrentProbe("fifth", data, np.nan),
    ]


@pytest.fixture
def class_ids(instruments: list[Instrument]) -> list[ABCMeta]:
    """Create a list of :class:`.Instrument` types. Alphabetically sorted."""
    return sorted({type(i) for i in instruments}, key=lambda cls: cls.__name__)


@pytest.fixture
def str_ids(instruments: list[Instrument]) -> list[str]:
    """Create a list of :class:`.Instrument` names."""
    return [str(i) for i in instruments]


@pytest.fixture
def object_ids(instruments: list[Instrument]) -> list[Instrument]:
    """Create a list of :class:`.Instrument` objets."""
    return instruments


class _FakeMeasurementPoint(IMeasurementPoint):
    def __init__(self, instruments: list[Instrument]) -> None:
        self.instruments = instruments


@pytest.fixture
def measurement_points(
    instruments: list[Instrument],
) -> list[IMeasurementPoint]:
    """Give dummy measurement points."""
    return [
        _FakeMeasurementPoint(instruments=instruments[:3]),
        _FakeMeasurementPoint(instruments=instruments[3:]),
    ]


# =============================================================================
# Test type of returned values
# =============================================================================
def test_filter_return_type(
    class_ids: list[ABCMeta], str_ids: list[str], object_ids: list[Instrument]
) -> None:
    """Check that :func:`.predicates.filter` returns the correct type."""
    filtered = filter_instruments(class_ids, dummy_instrument_filter)
    assert isinstance(filtered, list)
    assert all(isinstance(x, ABCMeta) for x in filtered)

    filtered = filter_instruments(str_ids, dummy_instrument_filter)
    assert isinstance(filtered, list)
    assert all(isinstance(x, str) for x in filtered)

    filtered = filter_instruments(object_ids, dummy_instrument_filter)
    assert isinstance(filtered, list)
    assert all(isinstance(x, Instrument) for x in filtered)


# =============================================================================
# Test returned values
# =============================================================================
@pytest.mark.parametrize(
    "instruments_id, predicate, expected",
    [
        # =====================================================================
        # Dummy filter
        # =====================================================================
        pytest.param(
            lf("class_ids"),
            dummy_instrument_filter,
            [CurrentProbe, Instrument, Penning],
            id="Dummy filter applied on [ABCMeta] input",
        ),
        pytest.param(
            lf("str_ids"),
            dummy_instrument_filter,
            ["first", "second", "third", "fourth", "fifth"],
            id="Dummy filter applied on [str] input",
        ),
        pytest.param(
            lf("object_ids"),
            dummy_instrument_filter,
            lf("instruments"),
            id="Dummy filter applied on [Instrument instances] input",
        ),
        # =====================================================================
        # Selection by name
        # =====================================================================
        pytest.param(
            lf("class_ids"),
            instrument_name_selector("second"),
            RAISES,
            id="Name filter applied on [ABCMeta] input",
        ),
        pytest.param(
            lf("str_ids"),
            instrument_name_selector("second"),
            ["second"],
            id="Name filter applied on [str] input",
        ),
        pytest.param(
            lf("object_ids"),
            instrument_name_selector("second"),
            lfc(lambda instruments: [instruments[1]]),
            id="Name filter applied on [Instrument instances] input",
        ),
        pytest.param(
            lf("str_ids"),
            instrument_name_selector(["second", "fourth"]),
            ["second", "fourth"],
            id="Name filter applied on [str] input, several names",
        ),
        # =====================================================================
        # Selection by type
        # =====================================================================
        pytest.param(
            lf("class_ids"),
            instrument_type_selector(CurrentProbe),
            [CurrentProbe],
            id="CurrentProbe filter applied on [ABCMeta] input",
        ),
        pytest.param(
            lf("str_ids"),
            instrument_type_selector(CurrentProbe),
            RAISES,
            id="CurrentProbe filter applied on [str] input",
        ),
        pytest.param(
            lf("object_ids"),
            instrument_type_selector(CurrentProbe),
            lfc(lambda instruments: [instruments[2], instruments[4]]),
            id="CurrentProbe filter applied on [Instrument instances] input",
        ),
        pytest.param(
            lf("class_ids"),
            instrument_type_selector(Instrument),
            [CurrentProbe, Instrument, Penning],
            id="Instrument filter applied on [ABCMeta] input",
        ),
        pytest.param(
            lf("str_ids"),
            instrument_type_selector(Instrument),
            RAISES,
            id="Instrument filter applied on [str] input",
        ),
        pytest.param(
            lf("object_ids"),
            instrument_type_selector(Instrument),
            lf("instruments"),
            id="Instrument filter applied on [Instrument instances] input",
        ),
        pytest.param(
            lf("object_ids"),
            instrument_type_selector([CurrentProbe, Penning]),
            lfc(lambda instruments: instruments[2:]),
            id="CurrentProbe+Penning filter applied on [Instrument instances] "
            "input",
        ),
        # =====================================================================
        # Instrument exclusion
        # =====================================================================
        pytest.param(
            lf("class_ids"),
            instrument_excluder(["second", "fourth"]),
            RAISES,
            id="Exclude 2 and 4 applied on [ABCMeta] input should raise "
            "error.",
        ),
        pytest.param(
            lf("str_ids"),
            instrument_excluder(["second", "fourth"]),
            ["first", "third", "fifth"],
            id="Exclude 2 and 4 applied on [str] input.",
        ),
        pytest.param(
            lf("object_ids"),
            instrument_excluder(["second", "fourth"]),
            lfc(
                lambda instruments: [
                    instruments[0],
                    instruments[2],
                    instruments[4],
                ]
            ),
            id="Exclude 2 and 4 applied on [Instrument instances] input.",
        ),
        # =====================================================================
        # Some combinations
        # =====================================================================
        pytest.param(
            lf("object_ids"),
            combine_predicates(
                instrument_type_selector(CurrentProbe),
                instrument_excluder(["third"]),
            ),
            lfc(
                lambda instruments: [
                    instruments[4],
                ]
            ),
            id=(
                "Only CurrentProbe, exclude 3rd applied on [Instrument "
                "instances] input."
            ),
        ),
    ],
)
def test_filter_returned_values(
    instruments_id: INSTRUMENTS_ID,
    predicate: INSTRUMENT_FILTER,
    expected: INSTRUMENTS_ID | object,
) -> None:
    """Check that :func:`.predicates.filter` returns the expected values."""
    if expected is RAISES:
        with pytest.raises(InstrumentFilteringError):
            filter_instruments(instruments_id, predicate)
        return

    filtered = filter_instruments(instruments_id, predicate)
    assert filtered == expected


# =============================================================================
# Test construction of predicates
# =============================================================================
@pytest.mark.parametrize(
    "instruments_to_ignore",
    [
        pytest.param(CurrentProbe, id="Bare ABCMeta raises."),
        pytest.param([CurrentProbe], id="Sequence of ABCMeta raises."),
        pytest.param(
            [CurrentProbe, Penning], id="Sequence of multiple ABCMeta raises."
        ),
    ],
)
def test_instrument_excluder_invalid_input(
    instruments_to_ignore: INSTRUMENTS_ID,
) -> None:
    """Check that :func:`.instrument_excluder` raises on invalid input."""
    with pytest.raises(InstrumentFilteringError):
        instrument_excluder(instruments_to_ignore)


# =============================================================================
# Test filtering on MeasurementPoints
# =============================================================================
@pytest.mark.parametrize(
    "instruments_id, predicate, expected",
    [
        pytest.param(
            lf("object_ids"),
            lfc(
                lambda measurement_points: measurement_point_excluder(
                    measurement_points[1:]
                )
            ),
            lfc(lambda instruments: instruments[:3]),
            id="Exclude mp2, applied on [Instrument instances] input.",
        ),
        pytest.param(
            lf("object_ids"),
            lfc(
                lambda measurement_points: measurement_point_excluder(
                    measurement_points[:1]
                )
            ),
            lfc(lambda instruments: instruments[3:]),
            id="Exclude mp1, applied on [Instrument instances] input.",
        ),
        pytest.param(
            lf("object_ids"),
            lfc(
                lambda measurement_points: measurement_point_excluder(
                    measurement_points
                )
            ),
            [],
            id="Exclude all measurement points, applied on [Instrument "
            "instances] input.",
        ),
        pytest.param(
            lf("str_ids"),
            lfc(
                lambda measurement_points: measurement_point_excluder(
                    measurement_points[1:]
                )
            ),
            lfc(lambda instruments: [str(i) for i in instruments[:3]]),
            id="Exclude mp2, applied on [str] input.",
        ),
        pytest.param(
            lf("class_ids"),
            lfc(
                lambda measurement_points: measurement_point_excluder(
                    measurement_points[:1]
                )
            ),
            RAISES,
            id="Exclude mp2, applied on [ABCMeta] input raises.",
        ),
    ],
)
def test_measurement_point_excluder(
    instruments_id: INSTRUMENTS_ID,
    predicate: INSTRUMENT_FILTER,
    expected: INSTRUMENTS_ID | object,
) -> None:
    """Check that :func:`.measurement_point_excluder` returns expected values."""
    if expected is RAISES:
        with pytest.raises(InstrumentFilteringError):
            filter_instruments(instruments_id, predicate)
        return

    filtered = filter_instruments(instruments_id, predicate)
    assert filtered == expected
