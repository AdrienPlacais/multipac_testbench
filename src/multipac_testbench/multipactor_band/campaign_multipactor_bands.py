"""Store all the :class:`.TestMultipactorBands` of several tests."""

from collections.abc import Sequence

from multipac_testbench.multipactor_band.test_multipactor_bands import (
    TestMultipactorBands,
)


class CampaignMultipactorBands(list):
    """Hold multipactor bands measured during several tests."""

    def __init__(
        self, tests_multipactor_bands: Sequence[TestMultipactorBands | None]
    ) -> None:
        """Instantiate the object."""
        super().__init__(tests_multipactor_bands)
