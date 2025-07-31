"""Define tests for :class:`.ThresholdSet`."""

from collections.abc import Iterable, Mapping

from multipac_testbench import ThresholdSet
from multipac_testbench.threshold.threshold import (
    THRESHOLD_FILTER_T,
    PowerExtremum,
    Threshold,
)


class MockThresholdSet(ThresholdSet):
    def __init__(
        self,
        by_half_power_cycle: Mapping[str, list[Threshold]],
        thresholds: Iterable[Threshold] | None = None,
        power_extrema: Iterable[PowerExtremum] | None = None,
    ) -> None:
        super().__init__(thresholds or [], power_extrema or [])
        self._by_half_power_cycle = by_half_power_cycle

    def _thresholds_by_half_power_cycle(
        self, predicate: THRESHOLD_FILTER_T | None = None
    ) -> dict[str, list[Threshold]]:
        return dict(self._by_half_power_cycle)


def test_extreme_thresholds_ideal():
    """Test with plain multipactor with clean lower and upper.

    One full power cycle, with one multipactor band in each.

    """
    thresholds_by_half_power = {
        "0 (increasing)": [
            a := Threshold(2, "lower", "enter", "_", 0),
            b := Threshold(4, "upper", "exit", "_", 0),
        ],
        "1 (decreasing)": [
            c := Threshold(12, "upper", "enter", "_", 0),
            d := Threshold(14, "lower", "exit", "_", 0),
        ],
    }
    threshold_set = MockThresholdSet(thresholds_by_half_power)
    extreme = ThresholdSet.extreme(threshold_set)
    assert extreme._thresholds == [a, b, c, d]


def test_extreme_thresholds_mixed_up_ideal():
    """Test with plain multipactor with clean lower and upper.

    One full power cycle, with one multipactor band in each. But we mix up the
    position of thresholds in the dicts. Note that mixing the keys of
    ``thresholds_by_half_power`` would lead to a bug.

    """
    thresholds_by_half_power = {
        "0 (increasing)": [
            b := Threshold(4, "upper", "exit", "_", 0),
            a := Threshold(2, "lower", "enter", "_", 0),
        ],
        "1 (decreasing)": [
            d := Threshold(14, "lower", "exit", "_", 0),
            c := Threshold(12, "upper", "enter", "_", 0),
        ],
    }
    threshold_set = MockThresholdSet(thresholds_by_half_power)
    extreme = ThresholdSet.extreme(threshold_set)
    assert extreme._thresholds == [a, b, c, d]


def test_extreme_thresholds_did_not_reach():
    """Test with plain multipactor with clean lower and no upper.

    One full power cycle, with one multipactor band we did not exit.

    """
    thresholds_by_half_power = {
        "0 (increasing)": [
            a := Threshold(2, "lower", "enter", "_", 0),
        ],
        "1 (decreasing)": [
            d := Threshold(14, "lower", "exit", "_", 0),
        ],
    }
    threshold_set = MockThresholdSet(thresholds_by_half_power)
    extreme = ThresholdSet.extreme(threshold_set)
    assert extreme._thresholds == [a, d]


def test_extreme_thresholds_two_bands_per_cycle():
    """Test with two bands in every half cycle."""
    thresholds_by_half_power = {
        "0 (increasing)": [
            a := Threshold(2, "lower", "enter", "_", 0),
            Threshold(4, "upper", "exit", "_", 0),
            Threshold(6, "lower", "enter", "_", 0),
            b := Threshold(8, "upper", "exit", "_", 0),
        ],
        "1 (decreasing)": [
            c := Threshold(12, "upper", "enter", "_", 0),
            Threshold(14, "lower", "exit", "_", 0),
            Threshold(16, "upper", "enter", "_", 0),
            d := Threshold(18, "lower", "exit", "_", 0),
        ],
    }
    threshold_set = MockThresholdSet(thresholds_by_half_power)
    extreme = ThresholdSet.extreme(threshold_set)
    assert extreme._thresholds == [a, b, c, d]


def test_extreme_thresholds_one_full_band_one_half():
    """Test with one complete band, but did not exit second one."""
    thresholds_by_half_power = {
        "0 (increasing)": [
            a := Threshold(2, "lower", "enter", "_", 0),
            Threshold(4, "upper", "exit", "_", 0),
            Threshold(6, "lower", "enter", "_", 0),
        ],
        "1 (decreasing)": [
            Threshold(14, "lower", "exit", "_", 0),
            Threshold(16, "upper", "enter", "_", 0),
            d := Threshold(18, "lower", "exit", "_", 0),
        ],
    }
    threshold_set = MockThresholdSet(thresholds_by_half_power)
    extreme = ThresholdSet.extreme(threshold_set)
    assert extreme._thresholds == [a, d]


def test_extreme_thresholds_single_point():
    """Test when MP one one single power step."""
    thresholds_by_half_power = {
        "0 (increasing)": [
            a := Threshold(2, "lower", "enter", "_", 0),
            b := Threshold(2, "upper", "exit", "_", 0),
        ],
        "1 (decreasing)": [
            c := Threshold(16, "upper", "enter", "_", 0),
            d := Threshold(16, "lower", "exit", "_", 0),
        ],
    }
    threshold_set = MockThresholdSet(thresholds_by_half_power)
    extreme = ThresholdSet.extreme(threshold_set)
    assert extreme._thresholds == [a, b, c, d]


def test_extreme_thresholds_mixed_up_single_point():
    """Test when MP one one single power step.

    We mix them up, to that first threshold of the list is not the more
    logical.

    """
    thresholds_by_half_power = {
        "0 (increasing)": [
            b := Threshold(2, "upper", "exit", "_", 0),
            a := Threshold(2, "lower", "enter", "_", 0),
        ],
        "1 (decreasing)": [
            d := Threshold(16, "lower", "exit", "_", 0),
            c := Threshold(16, "upper", "enter", "_", 0),
        ],
    }
    threshold_set = MockThresholdSet(thresholds_by_half_power)
    extreme = ThresholdSet.extreme(threshold_set)
    assert extreme._thresholds == [a, b, c, d]
