"""Define functions to help parametrize :func:`.quantity_is_above_local_average`.

In particular, plots intermediate lines.

"""

from abc import ABCMeta
from functools import partial

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from multipac_testbench import MultipactorTest
from multipac_testbench.instruments import (
    CurrentProbe,
    Instrument,
    VirtualInstrument,
)
from multipac_testbench.instruments.power import ForwardPower
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.util.multipactor_detectors import (
    quantity_is_above_local_average,
)
from multipac_testbench.util.post_treaters import running_mean


class ConstructionLine(VirtualInstrument):
    """Define a fake instrument type.

    It will hold intermediate calculations from
    :func:`.quantity_is_above_local_average` to help tuning this multipactor
    detector.

    """

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "Construction lines"


def study(
    instrument_class: ABCMeta,
    test: MultipactorTest,
    baseline_window: int = 300,
    threshold_factor: float = 3.0,
    consecutive_criterion: int = 0,
    minimum_number_of_points: int = 1,
) -> list[Axes]:
    measurement_point, instrument, slow_trend = add_slow_trend_instrument(
        instrument_class, test, baseline_window
    )
    add_construction_lines(
        measurement_point, instrument, slow_trend, threshold_factor
    )

    detector = partial(
        quantity_is_above_local_average,
        baseline_window=baseline_window,
        threshold_factor=threshold_factor,
        consecutive_criterion=consecutive_criterion,
        minimum_number_of_points=minimum_number_of_points,
    )

    threshold_set = test.determine_thresholds(
        detector, instrument_class, instruments_to_ignore=(slow_trend,)
    )

    to_plot = (instrument_class, CurrentProbe, ForwardPower)
    axes, _ = test.sweet_plot(*to_plot, threshold_set=threshold_set)

    # for ax in axes:
    #     lines = ax.get_lines()
    #
    #     # Show position of every measurement point
    #     lines[0].set_marker("o")
    #     lines[0].set_markersize(2)
    #     lines[0].set_color("grey")

    return axes


def add_slow_trend_instrument(
    instrument_class: ABCMeta, test: MultipactorTest, baseline_window: int
) -> tuple[IMeasurementPoint, Instrument, Instrument]:
    """Add fake instrument based on ``instrument_class``, with smoothened data.

    The fake instrument is a copy of the multipactor-detecting instrument, so
    it will have the same type.

    For now, ``test`` can only have one instance of ``instrument_class``.

    """
    instrument = test.get_instrument(instrument_class)

    measurement_points = [
        mp
        for mp in test.get_measurement_points()
        if instrument in mp.instruments
    ]
    assert len(measurement_points) == 1
    measurement_point = measurement_points[0]

    slow_trend = instrument.replace(
        name=instrument.name + f" smoothened over {baseline_window} samples",
        color=(0, 0, 0),
    )
    smoother = partial(running_mean, n_mean=baseline_window)
    slow_trend.add_post_treater(smoother)

    measurement_point.add_instrument(slow_trend)
    return measurement_point, instrument, slow_trend


def add_construction_lines(
    measurement_point: IMeasurementPoint,
    instrument: Instrument,
    slow_trend: Instrument,
    threshold_factor: float,
) -> None:
    """Add fake instruments holding intermediate data.

    They are :class:`.ConstructionLine` instances.

    """
    residuals = slow_trend.data - instrument.data
    residual = ConstructionLine(
        name="Residual",
        raw_data=pd.Series(residuals),
        position=instrument.position,
    )
    limits = np.full_like(
        residuals, np.median(np.abs(residual.data)) * threshold_factor
    )
    limit = ConstructionLine(
        name=f"Limit; {threshold_factor = }",
        raw_data=pd.Series(limits),
        position=instrument.position,
    )
    measurement_point.add_instrument(residual, limit)
