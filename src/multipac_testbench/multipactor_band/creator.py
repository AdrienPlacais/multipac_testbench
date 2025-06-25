"""Handle creation of the :class:`.MultipactorBand`."""

import logging

import numpy as np
from multipac_testbench.multipactor_band.multipactor_band import (
    IMultipactorBand,
    MultipactorBand,
    NoMultipactorBand,
)
from numpy.typing import NDArray


def _enter_a_mp_band(
    first_index: int | None, last_index: int | None, index: int, info: str
) -> int:
    """Enter a multipactor band.

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
        f"{info}: was previous MP zone correctly reinitialized? Maybe your "
        "multipactor detector is too sensitive and multipactor is detected at "
        "low powers, between two power ramps.\n"
        f"{first_index = }, {index = }"
    )
    assert last_index is None, (
        f"{info}: was previous MP zone correctly reinitialized?\n"
        f"{last_index = }, {index = }"
    )
    first_index = index + 1
    return first_index


def _exit_a_mp_band(
    first_index: int | None,
    last_index: int | None,
    power_grows: bool,
    pow_index: int,
    info: str,
    at_end_of_power_cycle: bool = False,
) -> tuple[None, None, MultipactorBand]:
    """Exit a multipactor band, re-init variables for the next one.

    Parameters
    ----------
    first_index :
        Index of entry in the zone. If it is None, an error is raised.
    last_index :
        Current index, which is the the index of exit.
    current_band :
        Previous :class:`.MultipactorBand` in the same half-power cycle. If it
        is not None, it means that several zones were detected.
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
    None
        Starting index of next multipactor zone, re-initialized to None.
    None
        Ending index of next multipactor zone, re-initialized to None.
    MultipactorBand
        Multipactor zone we are currently leaving, starting at the
        ``first_index`` and ending at the ``last_index`` that were provided as
        arguments.

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
    """(Re)-init variables for a new half power cycle.

    Parameters
    ----------
    info :
        For more descriptive error messages.
    pow_index :
        Index of previous half power cycle.
    index :
        Current measurement index.
    previous_band :
        Previous multipactor band object.

    Returns
    -------
    int | None
        Starting index of next multipactor zone. Will be set only if we did not
        exit multipactor during previous half power cycle.
    None
        End index of next multipactor zone, re-initialized to None as we do not
        know when it will end.
    int
        Index of current half power cycle.
    None
        [TODO:description]

    """
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
        first_index = _enter_a_mp_band(first_index, last_index, index, info)

    return first_index, last_index, pow_index, next_band


def _end_half_power_cycle(
    first_index: int | None,
    last_index: int | None,
    index: int,
    power_grows: bool,
    pow_index: int,
    info: str,
) -> MultipactorBand | None:
    """End the previous half power cycle.

    If we are in a multipactor zone, we also create the
    :class:`.MultipactorBand` and return it. A new :class:`.MultipactorBand`
    starting at current index will be created later.

    Parameters
    ----------
    first_index :
        Starting index of current multipactor band, if we are in one.
    last_index :
        End index of current multipactor band, if we are in one.
    index :
        Current measurement index.
    power_grows :
        If the power was growing in the ending half power cycle.
    pow_index :
        Current half power cycle index.
    info :
        For more descriptive error messages.

    Returns
    -------
    MultipactorBand | None
        A multipactor band is returned if we are in the middle of a multipactor
        band.

    """
    band = None
    if first_index is None or last_index is not None:
        return band

    last_index = index
    _, _, band = _exit_a_mp_band(
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
) -> list[IMultipactorBand]:
    """Create the :class:`.MultipactorBand`.

    Parameters
    ----------
    multipactor :
        True means multipactor, False no multipactor.
    power_growth_mask :
        True means power is growing, False it is decreasing.
    info :
        To give more meaning to the error messages.

    Returns
    -------
    bands : list[IMultipactorBand]
        One object per half power cycle (*i.e.* one object for power growth,
        one for power decrease). :class:`.IMultipactorBand` are subclassed in
        :class:`.MultipactorBand` and  :class:`.NoMultipactorBand`.

    """
    bands: list[IMultipactorBand] = []

    first_index, last_index, pow_index, band = _init_half_power_cycle(info)
    if multipactor[0]:
        logging.warning(
            "It seems that there was multipactor at the start of the test. "
            "I forced the start of a MultipactorBand to avoid errors later."
        )
        first_index = 0

    delta_multipactor = np.diff(multipactor)
    delta_power_growth_mask = np.diff(power_growth_mask)

    i_max = len(delta_power_growth_mask)
    zip_enum = enumerate(zip(delta_multipactor, delta_power_growth_mask))
    for i, (mp_status_changed, power_growth_changed) in zip_enum:
        reached_end_of_test = i + 1 == i_max
        reached_end_of_a_power_cycle = (
            power_growth_changed or reached_end_of_test
        )
        someting_to_do = mp_status_changed or reached_end_of_a_power_cycle
        if not someting_to_do:
            continue

        if reached_end_of_a_power_cycle:
            band = _end_half_power_cycle(
                first_index,
                last_index,
                i,
                bool(power_growth_mask[i]),
                pow_index,
                info,
            )
            if band is not None:
                # Happens when we left a half-power cycle but were still
                # multipacting
                bands.append(band)

            mp_detected_during_ending_power_cycle = (
                len(bands) > 0 and bands[-1].pow_index == pow_index
            )
            if not mp_detected_during_ending_power_cycle:
                bands.append(NoMultipactorBand(pow_index))

            first_index, last_index, pow_index, band = _init_half_power_cycle(
                info, pow_index, i, band
            )
            continue

        if multipactor[i + 1]:
            first_index = _enter_a_mp_band(first_index, last_index, i, info)
            continue

        first_index, last_index, band = _exit_a_mp_band(
            first_index,
            last_index=i,
            power_grows=bool(power_growth_mask[i]),
            pow_index=pow_index,
            info=info,
        )
        bands.append(band)
        band = None
    return bands
