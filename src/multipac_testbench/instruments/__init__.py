"""This subpackage holds instrument (current, voltage, etc)."""

from multipac_testbench.instruments.current_probe import CurrentProbe
from multipac_testbench.instruments.electric_field.field_probe import (
    FieldProbe,
)
from multipac_testbench.instruments.electric_field.i_electric_field import (
    IElectricField,
)
from multipac_testbench.instruments.electric_field.reconstructed import (
    FieldPowerError,
    Reconstructed,
)
from multipac_testbench.instruments.frequency import Frequency
from multipac_testbench.instruments.header_constant import (
    FrequencySetpoint,
    HeaderConstant,
    NewPowerSetpoint,
    PolarizationSetpoint,
)
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.optical_fibre import OpticalFibre
from multipac_testbench.instruments.penning import DiffPenning, Penning
from multipac_testbench.instruments.power import (
    ForwardPower,
    Power,
    PowerSetpoint,
    ReflectedPower,
)
from multipac_testbench.instruments.reflection_coefficient import (
    ReflectionCoefficient,
)
from multipac_testbench.instruments.rpa import RPA, RPACurrent, RPAPotential
from multipac_testbench.instruments.swr import SWR
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument

__all__ = [
    "CurrentProbe",
    "DiffPenning",
    "FieldPowerError",
    "FieldProbe",
    "ForwardPower",
    "Frequency",
    "IElectricField",
    "Instrument",
    "FrequencySetpoint",
    "NewPowerSetpoint",
    "OpticalFibre",
    "Penning",
    "PolarizationSetpoint",
    "HeaderConstant",
    "Power",
    "PowerSetpoint",
    "Reconstructed",
    "ReflectedPower",
    "ReflectionCoefficient",
    "RPA",
    "RPACurrent",
    "RPAPotential",
    "SWR",
    "VirtualInstrument",
]
