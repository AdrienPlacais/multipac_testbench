#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an object to hold test bench configuration."""
from dataclasses import dataclass


@dataclass
class FileConfiguration:
    """
    Store in a compact way the meaning of every mp test file column.

    Attributes
    ----------
    names: tuple[str, ...]
        Name of pick-ups.
    positions: tuple[float, ...]
        Position in m of pick-ups.
    electric_field_idx: tuple[int, ...]
        Column index of electric field data.
    mp_current_idx: tuple[int, ...]
        Column index of multipactor current data.

    """

    names: tuple[str, ...]
    positions: tuple[float, ...]
    electric_field_idx: tuple[int, ...]
    mp_current_idx: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check inputs validity."""
        self._check_inputs_length()

    def _check_inputs_length(self) -> None:
        """Ensure that inputs have same length."""
        ref_length = len(self.names)
        for parameter in ('positions', 'electric_field_idx', 'mp_current_idx'):
            attribute = getattr(self, parameter)
            assert len(attribute) == ref_length, f"Parameter {parameter} has "\
                f"length {len(attribute)} instead of {ref_length}."
