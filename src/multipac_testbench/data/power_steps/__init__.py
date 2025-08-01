"""Provide a set of data for :class:`.PowerStepSet` and :class:`.PowerStep`."""

from importlib import resources

dir = resources.files(__name__)
power_step_set_example = dir / "250620_120130_140MHz_SWR1_14_RAW_CSV"
power_step_example = power_step_set_example / "250620-12031527-Event_7_dBm.csv"
