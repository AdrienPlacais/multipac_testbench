"""Define instruments whose constant value is read from a ``CSV`` file header.

In general, such instruments actually record a set point.

"""

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.power import Sync
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from multipac_testbench.multipactor_test.helper import parse_header_value
from multipac_testbench.util.filtering import not_noisy_array_is_growing
from multipac_testbench.util.types import POST_TREATER_T
from numpy.typing import NDArray


class StepConstant(VirtualInstrument):
    """A :class:`.VirtualInstrument` with a constant value from a ``CSV`` header.

    The value is read once from the commented lines at the top of a
    :class:`.PowerStep` ``CSV`` file and repeated for all time samples.

    Note
    ----
    When associated with a :class:`.PowerStep`, this :class:`.Instrument`
    always store constant data.
    When associated with a :class:`.MultipactorTest`, it will generally be
    constant (eg :class:`FrequencySetpoint`), but may vary (eg
    :class:`PolarizationSetpoint` studies).

    """

    #: Whether this property is expected to not vary from ``CSV`` to ``CSV``.
    _should_be_constant: bool = True
    #: Name in :class:`.TestConditions`
    _field_name: str | None = None

    def __init__(
        self,
        name: str,
        raw_data: pd.Series,
        position: NDArray[np.float64] | float = np.nan,
        **kwargs,
    ) -> None:
        super().__init__(name, raw_data, position, **kwargs)

    @classmethod
    def from_single_csv_header(
        cls,
        name: str,
        commented_lines: Sequence[str],
        n_points: int,
        header_key: str | None = None,
        **kwargs,
    ) -> Self:
        """Instantiate from a parsed ``CSV`` header.

        Parameters
        ----------
        name :
            Name of the instrument and of the underlying pandas Series. Used as
            a column header when corresponding :class:`.MultipactorTest` is
            exported.
        commented_lines :
            Lines from the file header, stripped of their comment character.
        n_points :
            Number of time samples; the constant is repeated this many times.
        header_key :
            Header key to look up. Falls back to ``name`` when ``None``.
        kwargs :
            Passed to the constructor.

        Returns
        -------
            Instantiated instrument.

        """
        key = header_key if header_key is not None else name
        value = parse_header_value(commented_lines, key)
        raw_data = pd.Series(np.full(n_points, value), name=name)
        return cls(name=name, raw_data=raw_data, **kwargs)

    @property
    def _transfer_functions(self) -> list[POST_TREATER_T]:
        """Forbid use of transfer functions.

        Setpoints are generally not meant to be post-processed. But this can be
        overriden.

        """
        return []

    def growth_array(self, **kwargs) -> NDArray[np.float64]:
        return not_noisy_array_is_growing(self.data, **kwargs)


class CurrentCalibre(StepConstant):
    """Store the current calibre.

    It influences the calibration constant of the :class:`.CurrentProbe`.


    The header key should in general be ``Gamme_mA``.

    """

    _field_name = "current_calibre"


class FrequencySetpoint(StepConstant):
    """Store the frequency read in the header, in :unit:`MHz`.

    The header key should in general be ``SM300_Frequency``.

    """

    _field_name = "freq_mhz"

    @classmethod
    def from_single_csv_header(cls, *args, **kwargs) -> Self:
        """Instantiate from a parsed ``CSV`` header.

        Data is converted from :unit:`Hz` to :unit:`MHz`.

        See Also
        --------
        :class:`StepConstant`

        """
        freq = super().from_single_csv_header(*args, **kwargs)
        freq._raw_data /= 1e3
        return freq

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"RF frequency $f~[\mathrm{MHz}]$"


class MissingFrequencyError(ValueError):
    """Error raise when frequency is mandatory but unknown."""


class Frequency(FrequencySetpoint):
    """Alias to :class:`.FrequencySetpoint`."""


class PowerSetpoint(StepConstant):
    """Store the power asked by user.

    It should be preferred over :class:`.ForwardPower` to determine wether
    power is growing, as it is much more robust.

    The header key should in general be ``SM300_Level``.

    Note
    ----
    Does not inherit from :class:`.Power`.

    """

    _should_be_constant = False

    @classmethod
    def ylabel(cls) -> str:
        return r"Power setpoint $[\mathrm{dBm}]$"

    def growth_mask(
        self,
        minimum_number_of_points: int = 0,
        n_trailing_points_to_check: int = 0,
        width: int = 2,
        **kwargs,
    ) -> NDArray[np.bool]:
        raise ValueError("should not be used i think")
        return super().growth_mask(
            minimum_number_of_points=minimum_number_of_points,
            n_trailing_points_to_check=n_trailing_points_to_check,
            width=width,
            **kwargs,
        )


class PolarizationSetpoint(StepConstant):
    """Store the probes polarization read from the :class:`.PowerStep` header.

    The header key should in general be ``Polarisation_2``.

    """

    _field_name = "polarization"

    @classmethod
    def ylabel(cls) -> str:
        return r"Probes polarization $[\mathrm{V}]$"


class PostTrigger(StepConstant):
    """Store the post-trigger.

    The header key should in general be ``NI9205 _Post-Trig``.

    """

    _field_name = "post_trigger"


class PreTrigger(StepConstant):
    """Store the pre-trigger.

    The header key should in general be ``NI9205_Pre-Trig``.

    """

    _field_name = "pre_trigger"


class Trigger(StepConstant):
    """Store the trigger.

    This one is special because it is not present in the header, but must be
    calculated from a :class:`.Sync` signal.

    """

    _field_name = "trigger"

    @classmethod
    def from_sync(
        cls, n_points: int, sync: Sync, name: str = "NI9205_Sync", **kwargs
    ) -> Self:
        """Instantiate from a complete :class:`.Sync` signal."""
        raw_data = pd.Series(np.full(n_points, sync.trigger), name=name)
        return cls(name=name, raw_data=raw_data, **kwargs)
