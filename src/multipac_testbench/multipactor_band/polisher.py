"""Handle post-treatment of the :class:`.MultipactorBand`."""

from itertools import groupby
from typing import Literal

from multipac_testbench.multipactor_band.multipactor_band import (
    IMultipactorBand,
    MultipactorBand,
    NoMultipactorBand,
)


# =============================================================================
# Helpers
# =============================================================================
def group_by_power_cycle(
    input_list: list[IMultipactorBand],
) -> list[list[MultipactorBand] | NoMultipactorBand]:
    """Put in sublists the :class:`.IMultipactorBand` of same power cycle.

    :class:`.NoMultipactorBand` are not put in a sublist.

    """
    grouped = [
        [band for band in bands]
        for _, bands in groupby(input_list, key=lambda x: x.pow_index)
    ]
    for i, group in enumerate(grouped):
        if len(group) == 1 and isinstance(group[0], NoMultipactorBand):
            grouped[i] = group[0]  # type: ignore
    return grouped  # type: ignore


# =============================================================================
# Actual workers
# =============================================================================
def _keep_all_bands(bands: list[IMultipactorBand]) -> list[IMultipactorBand]:
    """Allow several :class:`.MultipactorBand` in same power cycle."""
    return bands


def _keep_largest(bands: list[IMultipactorBand]) -> list[IMultipactorBand]:
    """Return :class:`.MultipactorBand` spanning on most points."""
    grouped = group_by_power_cycle(bands)
    largest_bands = []
    for group in grouped:
        if isinstance(group, NoMultipactorBand):
            largest_bands.append(group)
            continue

        widths = [x.last_index - x.lower_index for x in group]
        largest_band = group[widths.index(max(widths))]
        largest_bands.append(largest_band)

    return largest_bands


def _keep_lowest(bands: list[IMultipactorBand]) -> list[IMultipactorBand]:
    """Return :class:`.MultipactorBand` at lowest power."""
    grouped = group_by_power_cycle(bands)
    lowest_thresholds = []
    for group in grouped:
        if isinstance(group, NoMultipactorBand):
            lowest_thresholds.append(group)
            continue

        if group[0].power_grows:
            lowest_thresholds.append(group[0])
            continue
        lowest_thresholds.append(group[-1])

    return lowest_thresholds


def _keep_highest(bands: list[IMultipactorBand]) -> list[IMultipactorBand]:
    """Return :class:`.MultipactorBand` at highest power."""
    grouped = group_by_power_cycle(bands)
    highest_thresholds = []
    for group in grouped:
        if isinstance(group, NoMultipactorBand):
            highest_thresholds.append(group)
            continue

        if not group[0].power_grows:
            highest_thresholds.append(group[0])
            continue
        highest_thresholds.append(group[-1])

    return highest_thresholds


def _merge(bands: list[IMultipactorBand]) -> list[IMultipactorBand]:
    """Merge the provided :class:`.IMultipactorBand` in same half-power cycle.

    With this method, we keep only one :class:`.IMultipactorBand` per half
    power cycle, spanning from start of first band to end of last band.

    Parameters
    ----------
    multipactor_bands :
        List of multipactor bands, some of which may belong to the same half
        power cycle.

    Returns
    -------
    merged :
        List of multipactor bands, only one per half power cycle.

    """
    grouped = group_by_power_cycle(bands)
    merged = []
    for group in grouped:
        if isinstance(group, NoMultipactorBand):
            merged.append(group)
            continue

        reached_second_threshold = all(
            [x.reached_second_threshold for x in group]
        )
        new = MultipactorBand(
            group[0].pow_index,
            group[0].first_index,
            group[-1].last_index,
            reached_second_threshold,
            group[0].power_grows,
        )
        merged.append(new)
    return merged


POLISHERS = {
    "keep_all": _keep_all_bands,
    "keep_largest": _keep_largest,
    "keep_lowest": _keep_lowest,
    "keep_highest": _keep_highest,
    "merge": _merge,
}
POLISHER_T = Literal[
    "keep_all", "keep_largest", "keep_lowest", "keep_highest", "merge"
]


# =============================================================================
# Main function
# =============================================================================
def polish(
    bands: list[IMultipactorBand], politics: POLISHER_T
) -> list[IMultipactorBand]:
    """
    Clean the given ``bands`` when there is a power cycle with several bands.

    Parameters
    ----------
    bands :
        List of bands, where we can have multiple multipactor bands per power
        cycle.
    politics :
        What to to when several multipactor bands are found in the same half
        power cycle:

        - ``'keep_lowest'``: we keep :class:`.MultipactorBand` at the lowest
          powers.
        - ``'keep_highest'``: we keep :class:`.MultipactorBand` at the highest
          powers.
        - ``'keep_all'``: we keep all :class:`.MultipactorBand`.
        - ``'merge'``: the resulting :class:`.MultipactorBand` will span from
          start of first :class:`.MultipactorBand` to end of last.
        - ``'keep_largest'``: we keep the :class:`.MultipactorBand` that was
          measured on the largest number of points.

    """
    assert politics in POLISHERS, (
        f"You asked {politics = } but allowed " f"are {POLISHERS.keys()}"
    )
    polisher = POLISHERS[politics]
    return polisher(bands)
