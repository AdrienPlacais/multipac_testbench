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
    filter,
    instrument_excluder,
    instrument_name_selector,
    instrument_type_selector,
)
from pytest_lazy_fixtures import lf, lfc


@pytest.fixture
def data() -> pd.Series:
    return pd.Series([1, 2, 3])


@pytest.fixture
def instruments(data: pd.Series) -> list[Instrument]:
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


@pytest.mark.implementation
def test_filter_return_type(
    class_ids: list[ABCMeta], str_ids: list[str], object_ids: list[Instrument]
) -> None:
    """Check that :func:`.predicates.filter` returns the correct type."""
    filtered = filter(class_ids, dummy_instrument_filter)
    assert isinstance(filtered, list)
    assert all(isinstance(x, ABCMeta) for x in filtered)

    filtered = filter(str_ids, dummy_instrument_filter)
    assert isinstance(filtered, list)
    assert all(isinstance(x, str) for x in filtered)

    filtered = filter(object_ids, dummy_instrument_filter)
    assert isinstance(filtered, list)
    assert all(isinstance(x, Instrument) for x in filtered)


@pytest.mark.implementation
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
            None,
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
            None,  # Should raise an error
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
            None,  # Should raise an error
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
            id="CurrentProbe+Penning filter applied on [Instrument instances] input",
        ),
        # =====================================================================
        # Instrument exclusion
        # =====================================================================
        pytest.param(
            lf("class_ids"),
            instrument_excluder(["second", "fourth"]),
            None,
            id="Exclude 2 and 4 applied on [ABCMeta] input.",
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
                instrument_excluder("third"),
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
    expected: INSTRUMENTS_ID | None,
) -> None:
    """Check that :func:`.predicates.filter` returns the expected values."""
    if expected is not None:
        filtered = filter(instruments_id, predicate)
        assert filtered == expected
        return

    with pytest.raises(InstrumentFilteringError):
        filter(instruments_id, predicate)
