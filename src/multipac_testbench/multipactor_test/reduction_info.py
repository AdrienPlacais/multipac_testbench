"""Track how an :class:`.Instrument` was reduced to a single value."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from multipac_testbench.util.types import REDUCER_T


@dataclass
class ReductionInfo:
    """Record of how a single :class:`.Instrument` was reduced."""

    reducer_name: str
    first_index: int | None
    last_index: int | None
    operated_on_raw: bool

    @classmethod
    def from_reducer(
        cls,
        reducer: REDUCER_T,
        operated_on_raw: bool,
    ) -> ReductionInfo:
        """Build from a reducer function, unpacking partials if needed."""
        if isinstance(reducer, functools.partial):
            name = reducer.func.__name__
            first_index = reducer.keywords.get("first_index")
            last_index = reducer.keywords.get("last_index")
        else:
            name = reducer.__name__
            first_index = None
            last_index = None
        return cls(
            reducer_name=name,
            first_index=first_index,
            last_index=last_index,
            operated_on_raw=operated_on_raw,
        )

    def __str__(self) -> str:
        window = (
            f"[{self.first_index}, {self.last_index}]"
            if self.first_index is not None
            else "full array"
        )
        raw = "raw" if self.operated_on_raw else "treated"
        return f"{self.reducer_name} over {window} ({raw})"
