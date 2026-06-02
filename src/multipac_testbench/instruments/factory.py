"""Define a class to create the proper :class:`.Instrument`."""

import logging
from collections.abc import Sequence
from pprint import pformat
from typing import Any, Literal

import multipac_testbench.instruments as ins
import pandas as pd
from multipac_testbench.instruments.penning import DiffPenning, Penning
from multipac_testbench.instruments.rpa import RPA

STRING_TO_INSTRUMENT_CLASS = {
    "CurrentProbe": ins.CurrentProbe,
    "DiffPenning": ins.DiffPenning,
    "ElectricFieldProbe": ins.FieldProbe,
    "FieldProbe": ins.FieldProbe,
    "ForwardPower": ins.ForwardPower,
    "OpticalFibre": ins.OpticalFibre,
    "Penning": ins.Penning,
    "PowerSetpoint": ins.PowerSetpoint,
    "RPACurrent": ins.RPACurrent,
    "RPAPotential": ins.RPAPotential,
    "ReflectedPower": ins.ReflectedPower,
}  #:
INSTRUMENT_NAME_T = Literal[
    "CurrentProbe",
    "DiffPenning",
    "ElectricFieldProbe",
    "FieldProbe",
    "ForwardPower",
    "OpticalFibre",
    "Penning",
    "PowerSetpoint",
    "RPACurrent",
    "RPAPotential",
    "ReflectedPower",
]


class InstrumentFactory:
    """Class to create instruments."""

    def __init__(
        self,
        freq_mhz: float | None = None,
        is_raw: bool = False,
        create_virtual_instruments: bool = True,
        commented_lines: Sequence[str] | None = None,
    ) -> None:
        """Set user-defined constants to create correspondig instrument.

        Parameters
        ----------
        freq_mhz:
            Frequency in :unit:`MHz`.
        is_raw :
            If set to ``True``, input data files is considered to be raw, ie to
            contain acquisition voltages instead of physical quantities.
        create_virtual_instruments :
            If virtual instruments should be created.
        commented_lines :
            Lines from a power step ``CSV`` file header, stripped of their
            comment character. Will be ``None`` in the context of
            :class:`.MultipactorTest`, but will be set within
            :class:`.PowerStep`.

        """
        self.freq_mhz = freq_mhz
        self._is_raw = is_raw
        self._create_virtual_instruments = create_virtual_instruments
        self._commented_lines: Sequence[str] = commented_lines or ()

    def run(
        self,
        name: str,
        df_data: pd.DataFrame,
        class_name: INSTRUMENT_NAME_T,
        column_header: str | list[str] | None = None,
        header_key: str | None = None,
        **instruments_kw: Any,
    ) -> ins.Instrument | None:
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
        header_key :
            Key to look for in a power step ``CSV`` header. Used to instantiate
            :class:`.HeaderConstant` in the context of :class:`.PowerStep`.
        instruments_kw :
            Other keyword arguments in the ``TOML`` file.

        Returns
        -------
            Instrument properly subclassed.

        """
        assert class_name in STRING_TO_INSTRUMENT_CLASS, (
            f"{class_name = } not recognized, allowed values are:\n"
            f"{pformat(INSTRUMENT_NAME_T)}\nSee: instruments/factory.py"
        )
        instrument_class = STRING_TO_INSTRUMENT_CLASS[class_name]

        if column_header is None:
            column_header = name

        if column_header not in df_data:
            logging.error(
                f"{column_header = } not present in provided file. Skipping "
                "associated instrument."
            )
            return

        raw_data = df_data[column_header]

        if isinstance(raw_data, pd.DataFrame):
            return instrument_class.from_pd_dataframe(
                name, raw_data, **instruments_kw
            )
        return instrument_class(
            name,
            raw_data,
            is_raw=self._is_raw,
            freq_mhz=self.freq_mhz,
            **instruments_kw,
        )

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
            to compute derived quantities, eg :math:`SWR` and :math:`R`.
        is_global :
            Tells if the :class:`.IMeasurementPoint` from which this method is
            called is global. It allows to forbid creation of one
            :class:`.Frequency` or one :class:`.SWR` instrument per
            :class:`.IMeasurementPoint`.
        kwargs :
            Other keyword arguments passed to :meth:`._power_related`.

        Returns
        -------
            The created virtual instruments.

        """
        if not self._create_virtual_instruments:
            return []
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

        rpa = self._rpa_related(instruments, **kwargs)
        if rpa is not None:
            virtuals.append(rpa)

        diff_pennings = self._pressure_related(instruments, **kwargs)
        if diff_pennings is not None:
            virtuals += diff_pennings

        return virtuals

    def _power_related(
        self, instruments: Sequence[ins.Instrument], **kwargs
    ) -> list[ins.VirtualInstrument]:
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
            return []

        forward = forwards[0]
        reflected = reflecteds[0]
        reflection_coefficient = ins.ReflectionCoefficient.from_powers(
            forward, reflected, **kwargs
        )
        swr = ins.SWR.from_reflection_coefficient(
            reflection_coefficient, **kwargs
        )
        return [reflection_coefficient, swr]

    def _rpa_related(
        self, instruments: Sequence[ins.Instrument], **kwargs
    ) -> RPA | None:
        """Create :class:`.RPA`."""
        rpa_potentials = [
            x for x in instruments if isinstance(x, ins.RPAPotential)
        ]
        rpa_currents = [
            x for x in instruments if isinstance(x, ins.RPACurrent)
        ]
        if len(rpa_currents) == 0 and len(rpa_currents) == 0:
            logging.debug("No RPA defined. Skipping.")
            return
        if len(rpa_potentials) != 1 or len(rpa_currents) != 1:
            logging.error(
                "Should have exactly one RPAPotential and one RPAPotential "
                "instruments. Skipping."
            )
            return

        potential = rpa_potentials[0]
        current = rpa_currents[0]
        rpa = ins.RPA.from_current_and_potential(
            rpa_current=current, rpa_potential=potential, **kwargs
        )
        return rpa

    def _pressure_related(
        self, instruments: Sequence[ins.Instrument], **kwargs
    ) -> list[DiffPenning] | None:
        """Create :class:`.DiffPenning`."""
        pennings = [x for x in instruments if isinstance(x, Penning)]
        if len(pennings) == 0:
            logging.debug("No Penning defined. Skipping.")
            return
        diff_pennings = [
            ins.DiffPenning.from_penning(
                penning=penning, name=f"DiffPenning_{i}", **kwargs
            )
            for i, penning in enumerate(pennings)
        ]
        return diff_pennings

    def _constant_values_defined_by_user(
        self,
        n_points: int,
    ) -> list[ins.VirtualInstrument]:
        """Define a fake frequency probe. Maybe a fake SWR, fake R later."""
        constants = []
        if self.freq_mhz is not None:
            constants.append(
                ins.Frequency.from_user_defined_frequency(
                    self.freq_mhz, n_points
                )
            )
        return constants
