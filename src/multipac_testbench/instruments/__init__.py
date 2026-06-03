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
from multipac_testbench.instruments.header_constant import (
    CurrentCalibre,
    Frequency,
    FrequencySetpoint,
    HeaderConstant,
    PolarizationSetpoint,
    PostTrigger,
    PowerSetpoint,
    PreTrigger,
)
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.instruments.optical_fibre import OpticalFibre
from multipac_testbench.instruments.penning import DiffPenning, Penning
from multipac_testbench.instruments.power import (
    ForwardPower,
    Power,
    ReflectedPower,
    Sync,
)
from multipac_testbench.instruments.reflection_coefficient import (
    ReflectionCoefficient,
)
from multipac_testbench.instruments.rpa import RPA, RPACurrent, RPAPotential
from multipac_testbench.instruments.swr import SWR
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument

__all__ = [
    "CurrentCalibre",
    "CurrentProbe",
    "DiffPenning",
    "FieldPowerError",
    "FieldProbe",
    "ForwardPower",
    "Frequency",
    "FrequencySetpoint",
    "HeaderConstant",
    "IElectricField",
    "Instrument",
    "PowerSetpoint",
    "OpticalFibre",
    "Penning",
    "PolarizationSetpoint",
    "PostTrigger",
    "Power",
    "PreTrigger",
    "RPA",
    "RPACurrent",
    "RPAPotential",
    "Reconstructed",
    "ReflectedPower",
    "ReflectionCoefficient",
    "SWR",
    "Sync",
    "VirtualInstrument",
]
