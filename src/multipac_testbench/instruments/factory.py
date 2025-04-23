"""Define a class to create the proper :class:`.Instrument`."""

import logging
from collections.abc import Sequence
from typing import Any, Literal

import multipac_testbench.instruments as ins
import pandas as pd

STRING_TO_INSTRUMENT_CLASS = {
    "CurrentProbe": ins.CurrentProbe,
    "ElectricFieldProbe": ins.FieldProbe,
    "FieldProbe": ins.FieldProbe,
    "ForwardPower": ins.ForwardPower,
    "OpticalFibre": ins.OpticalFibre,
    "Penning": ins.Penning,
    "ReflectedPower": ins.ReflectedPower,
}  #:
INSTRUMENT_NAME_T = Literal[
    "CurrentProbe",
    "ElectricFieldProbe",
    "FieldProbe",
    "ForwardPower",
    "OpticalFibre",
    "Penning",
    "ReflectedPower",
]


class InstrumentFactory:
    """Class to create instruments."""

    def __init__(self, freq_mhz: float | None = None) -> None:
        """Set user-defined constants to create correspondig instrument."""
        self.freq_mhz = freq_mhz

    def run(
        self,
        name: str,
        df_data: pd.DataFrame,
        class_name: INSTRUMENT_NAME_T,
        column_header: str | list[str] | None = None,
        **instruments_kw: Any,
    ) -> ins.Instrument:
        """Take the proper subclass, instantiate it and return it.

        Parameters
        ----------
        name :
            Name of the instrument. For clarity, it should match the name of a
            column in ``df_data`` when it is possible.
        df_data :
            Content of the multipactor tests results ``CSV`` file.
        class_name :
            Name of the instrument class, as given in the ``TOML`` file.
        column_header :
            Name of the column(s) from which the data of the instrument will
            be taken. The default is None, in which case ``column_header`` is
            set to ``name``. In general it is not necessary to provide it. An
            exception is when several ``CSV`` columns should be loaded in the
            instrument.
        instruments_kw :
            Other keyword arguments in the ``TOML`` file.

        Returns
        -------
        instrument :
            Instrument properly subclassed.

        """
        assert class_name in STRING_TO_INSTRUMENT_CLASS, (
            f"{class_name = } not recognized, check STRING_TO_INSTRUMENT_CLASS"
            "in instrument/factory.py"
        )
        instrument_class = STRING_TO_INSTRUMENT_CLASS[class_name]

        if column_header is None:
            column_header = name

        raw_data = df_data[column_header]

        if isinstance(raw_data, pd.DataFrame):
            return instrument_class.from_pd_dataframe(
                name, raw_data, **instruments_kw
            )
        return instrument_class(name, raw_data, **instruments_kw)

    def run_virtual(
        self,
        instruments: Sequence[ins.Instrument],
        is_global: bool = False,
        **kwargs,
    ) -> list[ins.VirtualInstrument]:
        """Add the implemented :class:`.VirtualInstrument`.

        Parameters
        ----------
        instruments :
            The :class:`.Instrument` that were already created. They are used
            to compute derived quantities, in particular :math:`SWR` and
            :math:`R`.
        is_global :
            Tells if the :class:`.IMeasurementPoint` from which this method is
            called is global. It allows to forbid creation of one
            :class:`.Frequency` or one :class:`.SWR` instrument per
            :class:`.IMeasurementPoint`.
        kwargs :
            Other keyword arguments passed to :meth:`._power_related`.

        Returns
        -------
        virtuals :
            The created virtual instruments.

        """
        virtuals = []

        power_related = []
        if is_global:
            power_related = self._power_related(instruments, **kwargs)
        if len(power_related) > 0:
            virtuals += power_related

        n_points = len(instruments[0].data_as_pd)
        constants = []
        if is_global:
            constants = self._constant_values_defined_by_user(n_points)
        if len(constants) > 0:
            virtuals += constants

        return virtuals

    def _power_related(
        self, instruments: Sequence[ins.Instrument], **kwargs
    ) -> Sequence[ins.VirtualInstrument]:
        """Create :class:`.ReflectionCoefficient` and :class:`.SWR`."""
        forwards = [x for x in instruments if isinstance(x, ins.ForwardPower)]
        reflecteds = [
            x for x in instruments if isinstance(x, ins.ReflectedPower)
        ]
        if len(forwards) != 1 or len(reflecteds) != 1:
            logging.error(
                "Should have exactly one ForwardPower and one ReflectedPower "
                "instruments. Skipping SWR and R, this may create problems in "
                "the future."
            )
            return ()

        forward = forwards[0]
        reflected = reflecteds[0]
        reflection_coefficient = ins.ReflectionCoefficient.from_powers(
            forward, reflected, **kwargs
        )
        swr = ins.SWR.from_reflection_coefficient(
            reflection_coefficient, **kwargs
        )
        return reflection_coefficient, swr

    def _constant_values_defined_by_user(
        self,
        n_points: int,
    ) -> Sequence[ins.VirtualInstrument]:
        """Define a fake frequency probe. Maybe a fake SWR, fake R later."""
        constants = []
        if self.freq_mhz is not None:
            constants.append(
                ins.Frequency.from_user_defined_frequency(
                    self.freq_mhz, n_points
                )
            )
        return constants
