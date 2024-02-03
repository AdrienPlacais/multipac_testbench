#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep track of a single multipactor band.

i.e.: a set of measurement points where multipactor happens.

.. todo::
    Modify the :class:`.Instrument` so that we can call
    Instrument[MultipactorBand] and get their data at the corresponding
    indexes.

"""
import numpy as np


class MultipactorBand(list):
    """
    A set of contiguous indexes, corresponding to multipactor at a given pos.

    Also, holds some methods to ease plotting.

    """

    def __init__(self,
                 first_index: int,
                 last_index: int,
                 detector_instrument_name: str
                 ) -> None:
        """Create the objects with its indexes."""
        indexes = [i for i in range(first_index, last_index + 1)]
        super().__init__(indexes)

        self.detector_instrument_name = detector_instrument_name

        # this is position in MultipactorBands object, so may be unnecessary
        self.multipactor_band_index: int

    def to_range(self) -> range:
        """Convert objet to a ``range`` object holding all ``indexes``.

        .. todo::
            Delete this if not used.

        """
        return range(self[0], self[-1] + 1)


if __name__ == '__main__':
    my_indexes = [3, 8]
    my_band = MultipactorBand(my_indexes[0], my_indexes[-1])

    my_instrument_data = np.linspace(0, 10, 11)
    my_mp_data = my_instrument_data[my_band]
    print(my_mp_data)
