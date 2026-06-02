"""Define functions to help parametrize :func:`.quantity_is_above_lower_envelope`.

In particular, plots intermediate lines.

"""

from abc import ABCMeta
from functools import partial

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from multipac_testbench import MultipactorTest
from multipac_testbench.instruments import Instrument, VirtualInstrument
from multipac_testbench.instruments.power import ForwardPower
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.util.multipactor_detectors import (
    quantity_is_above_lower_envelope,
    residual_threshold,
)
from multipac_testbench.util.post_treaters import lower_envelope


class ConstructionLine(VirtualInstrument):
    """Define a fake instrument type.

    It will hold intermediate calculations from
    :func:`.quantity_is_above_lower_envelope` to help tuning this multipactor
    detector.

    We set ``relatable_thresholds`` to ``False`` to avoid marking thresholds
    positions on these plots.

    """

    relatable_thresholds = False

    def __init__(
        self, *args, relatable_thresholds: bool = False, **kwargs
    ) -> None:
        super().__init__(
            *args, relatable_thresholds=relatable_thresholds, **kwargs
        )

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return "Construction lines"


def study(
    instrument_class: ABCMeta,
    test: MultipactorTest,
    envelope_window: int = 150,
    threshold_factor: float = 0.8,
    consecutive_criterion: int = 0,
    minimum_number_of_points: int = 10,
) -> list[Axes]:
    detector = partial(
        quantity_is_above_lower_envelope,
        envelope_window=envelope_window,
        threshold_factor=threshold_factor,
        consecutive_criterion=consecutive_criterion,
        minimum_number_of_points=minimum_number_of_points,
    )

    measurement_point, instrument, envelope = add_lower_envelope_instrument(
        instrument_class, test, envelope_window
    )
    construction_lines = add_construction_lines(
        measurement_point, instrument, envelope, threshold_factor
    )
    threshold_set = test.determine_thresholds(
        detector,
        instrument_class,
        instruments_to_ignore=(envelope, *construction_lines),
    )

    to_plot = (instrument_class, ForwardPower)
    axes, _ = test.sweet_plot(
        *to_plot,
        threshold_set=threshold_set,
        global_instruments=True,
    )

    # for ax in axes:
    #     lines = ax.get_lines()
    #
    #     # Show position of every measurement point
    #     lines[0].set_marker("o")
    #     lines[0].set_markersize(2)
    #     lines[0].set_color("grey")

    return axes


def add_lower_envelope_instrument(
    instrument_class: ABCMeta, test: MultipactorTest, envelope_window: int
) -> tuple[IMeasurementPoint, Instrument, Instrument]:
    """Create fake instrument holding lower envelope.

    This instrument is a copy of the multipactor-detecting instrument, so it
    will have the same type.

    For now, ``test`` can only have one instance of ``instrument_class``.

    Parameters
    ----------
    instrument_class :
        Class of the detecting instrument.
    test :
        The multipactor test under study.
    envelope_window :
        Number of samples of the envelope window. Typically, one power cycle.

    Returns
    -------
    IMeasurementPoint
        Where the detecting instrument and the virtual instruments were added.
    Instrument
        Detecting instrument instance.
    Instrument
        :class:`.Instrument` instance holding the lower envelope of the
        detecting instrument data.

    """
    detecting = test.get_instrument(instrument_class)

    measurement_points = [
        mp
        for mp in test.get_measurement_points()
        if detecting in mp.instruments
    ]
    assert len(measurement_points) == 1
    measurement_point = measurement_points[0]

    envelope = detecting.replace(
        name=detecting.name + f" lower envelope, {envelope_window} samples",
        data=pd.Series(
            lower_envelope(detecting.data, envelope_window=envelope_window)
        ),
        color=(0, 0, 0),
        relatable_thresholds=False,
    )

    measurement_point.add_instrument(envelope)
    return measurement_point, detecting, envelope


def add_construction_lines(
    measurement_point: IMeasurementPoint,
    detecting: Instrument,
    envelope: Instrument,
    threshold_factor: float,
) -> tuple[ConstructionLine, ConstructionLine]:
    """Add fake instruments holding intermediate data.

    They are :class:`.ConstructionLine` instances.

    """
    residual, threshold = residual_threshold(detecting.data, envelope.data)

    res = ConstructionLine(
        name="Residual",
        raw_data=pd.Series(residual),
        position=detecting.position,
    )
    limit = ConstructionLine(
        name=f"Limit; {threshold_factor = }",
        raw_data=pd.Series(np.full_like(residual, fill_value=threshold)),
        position=detecting.position,
    )
    measurement_point.add_instrument(res, limit)
    return res, limit
