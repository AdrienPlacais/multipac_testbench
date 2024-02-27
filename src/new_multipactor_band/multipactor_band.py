#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of a single multipactor band.

i.e.: a set of measurement points where multipactor happens. A
:class:`MultipactorBand` is defined on half a power cycle.

.. todo::
    Modify the :class:`.Instrument` so that we can call
    Instrument[MultipactorBand] and get their data at the corresponding
    indexes.

"""


class MultipactorBand:
    """First and last index of a multipactor zone."""

    def __init__(self,
                 first_index: int,
                 last_index: int,
                 reached_second_threshold: bool,
                 power_grows: bool) -> None:
        """Instantiate object."""
        self.first_index = first_index
        self.last_index = last_index

        self.lower_index, self.upper_index = first_index, last_index
        if not power_grows:
            self.upper_index, self.lower_index = first_index, last_index

        if not reached_second_threshold:
            self.upper_index = None
        self.indexes = [i for i in range(self.first_index,
                                         self.last_index + 1)]
        self.reached_second_threshold = reached_second_threshold
