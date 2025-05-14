"""Handle creation of the :class:`.MultipactorBand`."""

import numpy as np
from multipac_testbench.multipactor_band.multipactor_band import (
    IMultipactorBand,
    MultipactorBand,
    NoMultipactorBand,
)
from numpy.typing import NDArray


def _enter_a_mp_zone(
    first_index: int | None,
    last_index: int | None,
    index: int,
    info: str,
) -> int:
    """Enter a multipactor zone.

    .. note::
        This function does not create a :class:`.MultipactorBand`. It only
        initializes its ``first_index``, as we do not know when this
        multipactor is gonna end.

    Parameters
    ----------
    first_index :
        Index at which the previous multipactor started. We just check if it is
        None to catch eventual corner cases.
    last_index :
        Index at which the previous multipactor ended. We just check if it is
        None to catch eventual corner cases.
    index :
        Current index at which an entry in multipactor regime was detected.
    info :
        Information on the test and the multipactor instrument detector for
        more explicit error messages.

    Returns
    -------
    first_index :
        Index at which current multipactor starts.

    """
    assert first_index is None, (
        f"{info}: was previous MP zone correctly reinitialized? "
        f"{first_index = }, {index = }"
    )
    assert last_index is None, (
        f"{info}: was previous MP zone correctly reinitialized? "
        f"{last_index = }, {index = }"
    )
    first_index = index + 1
    return first_index


def _exit_a_mp_zone(
    first_index: int | None,
    last_index: int | None,
    power_grows: bool,
    pow_index: int,
    info: str,
    at_end_of_power_cycle: bool = False,
) -> tuple[None, None, MultipactorBand]:
    """Exit a multipactor zone.

    Parameters
    ----------
    first_index :
        Index of entry in the zone. If it is None, an error is raised.
    last_index :
        Current index, which is the the index of exit.
    current_band :
        Previous :class:`.MultipactorBand` in the same half-power cycle. If it
        is not None, it means that several zones were detected. Its handling is
        determined by ``several_bands_politics``.
    power_grows :
        If the power grows.
    pow_index :
        Index of the current power half cycle.
    info :
        To give more meaning to the error messages.
    reached_end_of_power_cycle :
        If this function is called when we reach the end of a half power cycle.

    Returns
    -------
    first_index :
        TODO
    last_index :
        TODO
    band :
        TODO

    """
    assert first_index is not None, (
        f"{info}: we are exiting a multipacting zone but I did not detect "
        f"when it started. Check what happened around {last_index = }."
    )
    assert last_index is not None

    band = MultipactorBand(
        pow_index,
        first_index,
        last_index,
        reached_second_threshold=not at_end_of_power_cycle,
        power_grows=power_grows,
    )
    first_index, last_index = None, None
    return first_index, last_index, band


def _init_half_power_cycle(
    info: str,
    pow_index: int = -1,
    index: int = 0,
    previous_band: IMultipactorBand | None = None,
) -> tuple[int | None, None, int, None]:
    """(Re)-init variables for a new half power cycle."""
    first_index, last_index = None, None
    pow_index += 1
    next_band = None

    if index == 0:
        return first_index, last_index, pow_index, next_band

    still_in_a_mp_zone = (
        isinstance(previous_band, MultipactorBand)
        and not previous_band.reached_second_threshold
    )
    if still_in_a_mp_zone:
        first_index = _enter_a_mp_zone(first_index, last_index, index, info)
    return first_index, last_index, pow_index, next_band


def _end_half_power_cycle(
    first_index: int | None,
    last_index: int | None,
    index: int,
    power_grows: bool,
    pow_index: int,
    info: str,
) -> MultipactorBand | None:
    """Start a new power cycle."""
    band = None
    if first_index is not None and last_index is None:
        last_index = index
        _, _, band = _exit_a_mp_zone(
            first_index,
            last_index,
            power_grows,
            pow_index,
            info,
            at_end_of_power_cycle=True,
        )
    return band


# =============================================================================
# Main function
# =============================================================================
def create(
    multipactor: NDArray[np.bool],
    power_growth_mask: NDArray[np.bool],
    info: str = "",
) -> list[MultipactorBand | None]:
    """Create the different :class:`.MultipactorBand`.

    Parameters
    ----------
    multipactor :
        True means multipactor, False no multipactor.
    power_growth_mask :
        True means power is growing, False it is decreasing.
    info :
        To give more meaning to the error messages.
    several_bands_politics :
        What to to when several multipactor bands are found in the same
        half-power cycle:

        - ``'keep_first'``: we keep first :class:`.MultipactorBand`
        - ``'keep_last'``: we keep last :class:`.MultipactorBand`
        - ``'keep_all'``: we keep all :class:`.MultipactorBand` (currently not
          implemented)
        - ``'merge'``: the final :class:`.MultipactorBand` spans from start
          of first :class:`.MultipactorBand` to end of last.

    Returns
    -------
    all_bands :
        One object per half power cycle (*i.e.* one object for power growth,
        one for power decrease). None means that no multipactor was detected.

    """
    delta_multipactor = np.diff(multipactor)
    delta_power_growth_mask = np.diff(power_growth_mask)
    zip_enum = enumerate(zip(delta_multipactor, delta_power_growth_mask))

    i_max = len(delta_power_growth_mask)

    all_bands = []
    first_index, last_index, pow_index, band = _init_half_power_cycle(info)

    for i, (change_in_multipactor, change_in_power_growth) in zip_enum:
        last_iter = i + 1 == i_max

        if not (change_in_multipactor or change_in_power_growth or last_iter):
            continue

        if change_in_power_growth or last_iter:
            # a current band is returned if we are still multipacting
            band = _end_half_power_cycle(
                first_index,
                last_index,
                i,
                bool(power_growth_mask[i]),
                pow_index,
                info,
            )
            if band is not None:
                all_bands.append(band)

            no_mp_during_this_cycle = (
                len(all_bands) == 0 or all_bands[-1].pow_index != pow_index
            )
            if no_mp_during_this_cycle:
                all_bands.append(NoMultipactorBand(pow_index))

            first_index, last_index, pow_index, band = _init_half_power_cycle(
                info, pow_index, i, band
            )
            continue

        if multipactor[i + 1]:
            first_index = _enter_a_mp_zone(first_index, last_index, i, info)
            continue

        first_index, last_index, band = _exit_a_mp_zone(
            first_index,
            last_index=i,
            power_grows=bool(power_growth_mask[i]),
            pow_index=pow_index,
            info=info,
        )
        all_bands.append(band)
        band = None
    return all_bands
