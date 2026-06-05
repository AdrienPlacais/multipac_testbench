"""Define a compact record of the conditions under which a test was run."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from multipac_testbench.instruments.step_constant import StepConstant

if TYPE_CHECKING:
    from multipac_testbench.measurement_point.global_diagnostics import (
        GlobalDiagnostics,
    )


@functools.lru_cache(maxsize=1)
def _warn_freq_mhz_override_once() -> None:
    """Warn once (process-wide) when freq_mhz overrides FrequencySetpoint."""
    logging.warning(
        "freq_mhz was supplied explicitly but FrequencySetpoint is also "
        "defined in the TOML. The explicit value takes precedence. "
        "Consider removing the freq_mhz argument and relying on the TOML."
    )


_INT_FIELDS = {"pre_trigger", "post_trigger", "trigger", "current_calibre"}


@dataclass
class TestConditions:
    """Compact record of the conditions under which a test was run."""

    freq_mhz: float
    swr: float
    info: str = ""
    polarization: float = np.nan
    pre_trigger: int | None = None
    trigger: int | None = None
    post_trigger: int | None = None
    current_calibre: Literal[1, 10] | None = None

    @classmethod
    def from_components(
        cls,
        freq_mhz: float | None,
        swr: float,
        info: str = "",
        trigger: int | None = None,
        global_diagnostics: GlobalDiagnostics | None = None,
    ) -> TestConditions:
        """
        Build from constructor args and :class:`.StepConstant` instruments.

        Scans ``global_diagnostics`` for :class:`.StepConstant` instruments
        whose :attr:`_should_be_constant` is ``True`` and whose
        :attr:`_field_name` matches a field of this dataclass. Warns if a
        supposed-to-be-constant instrument has more than one distinct value,
        and stores the median in that case.

        Parameters
        ----------
        freq_mhz :
            Explicit frequency override. Wins over :class:`.FrequencySetpoint`
            if both are present (warns once). Pass ``None`` to rely entirely on
            the TOML.
        swr :
            Standing Wave Ratio, always supplied by the user.
        info :
            Human-readable label for this test.
        trigger :
            Number of steps with power ON during a power pulse.
        global_diagnostics :
            Searched for :class:`.StepConstant` instruments.

        Returns
        -------
            Populated :class:`TestConditions`.

        Raises
        ------
        ValueError
            If ``freq_mhz`` is ``None`` and no :class:`.FrequencySetpoint` is
            found.

        """
        # Local import avoids circular dependency at module level.
        from multipac_testbench.instruments.step_constant import (
            FrequencySetpoint,
        )

        kwargs: dict[str, float] = {}
        freq_from_instrument: float | None = None

        if global_diagnostics is not None:
            for instrument in global_diagnostics.instruments:
                if not isinstance(instrument, StepConstant):
                    continue
                if not instrument._should_be_constant:
                    continue
                field_name = instrument._field_name
                if field_name is None:
                    continue

                series = instrument.data_as_pd
                if series.nunique() > 1:
                    logging.warning(
                        f"{instrument} should generally be constant but has "
                        f"{series.nunique()} distinct values. Saving median in"
                        " TestConditions."
                    )
                value_f = float(np.nanmedian(instrument.data))
                value = (
                    int(round(value_f))
                    if field_name in _INT_FIELDS
                    else value_f
                )
                if isinstance(instrument, FrequencySetpoint):
                    freq_from_instrument = value
                else:
                    kwargs[field_name] = value

        final_freq = None
        if freq_from_instrument is not None:
            final_freq = freq_from_instrument
        if freq_mhz is not None:
            _warn_freq_mhz_override_once()
            final_freq = freq_mhz
        if final_freq is None:
            raise ValueError(
                "freq_mhz must be provided either as a constructor argument "
                "or via FrequencySetpoint defined in the TOML."
            )

        return cls(
            freq_mhz=final_freq, swr=swr, info=info, trigger=trigger, **kwargs
        )

    def __str__(self) -> str:
        """Give info on the object."""
        out = [f"{self.freq_mhz}MHz"]
        if self.info:
            out.append(self.info)
        return ", ".join(out)
