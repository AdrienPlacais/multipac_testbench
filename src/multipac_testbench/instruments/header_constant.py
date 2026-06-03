"""Define instruments whose constant value is read from a ``CSV`` file header.

In general, such instruments actually record a set point.

"""

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from multipac_testbench.instruments.virtual_instrument import VirtualInstrument
from multipac_testbench.multipactor_test.helper import parse_header_value
from multipac_testbench.util.types import POST_TREATER_T
from numpy.typing import NDArray


class HeaderConstant(VirtualInstrument):
    """A :class:`.VirtualInstrument` with a constant value from a ``CSV`` header.

    The value is read once from the commented lines at the top of a
    :class:`.PowerStep` ``CSV`` file and repeated for all time samples.

    Note
    ----
    When associated with a :class:`.PowerStep`, this :class:`.Instrument`
    always store constant data.
    When associated with a :class:`.MultipactorTest`, it will generally be
    constant (eg :class:`FrequencySetPoint`), but may vary (eg
    :class:`PolarizationSetpoint` studies).

    """

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


class PolarizationSetpoint(HeaderConstant):
    """Store the probes polarization read from the :class:`.PowerStep` header.

    The header key should in general be ``Polarisation_2``.

    """

    @classmethod
    def ylabel(cls) -> str:
        return r"Probes polarization $[\mathrm{V}]$"


class NewPowerSetpoint(HeaderConstant):
    """Store the power asked by user.

    It should be preferred over :class:`.ForwardPower` to determine wether
    power is growing, as it is much more robust.

    The header key should in general be ``SM300_Level``.

    Note
    ----
    Does not inherit from :class:`Power`.

    """

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
        return super().growth_mask(
            minimum_number_of_points=minimum_number_of_points,
            n_trailing_points_to_check=n_trailing_points_to_check,
            width=width,
            **kwargs,
        )


class FrequencySetpoint(HeaderConstant):
    """Store the frequency set by the user.

    By default, the frequency is in :unit:`MHz`.

    The header key should in general be ``SM300_Frequency``.

    """

    @classmethod
    def from_single_csv_header(cls, *args, **kwargs) -> Self:
        """Instantiate from a parsed ``CSV`` header.

        Data is converted from :unit:`Hz` to :unit:`MHz`.

        See Also
        --------
        :class:`HeaderConstant`

        """
        freq = super().from_single_csv_header(*args, **kwargs)
        freq._raw_data /= 1e3
        return freq

    @classmethod
    def ylabel(cls) -> str:
        """Label used for plots."""
        return r"RF frequency $f~[\mathrm{MHz}]$"
