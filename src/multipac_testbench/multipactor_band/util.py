"""Ensure that :class:`.InstrumentMultipactorBands` are consistently used."""

from collections.abc import Sequence

import numpy as np
from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.multipactor_band.instrument_multipactor_bands import (
    InstrumentMultipactorBands,
)


def _create_zip(
    obj: Sequence[Instrument] | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: (
        InstrumentMultipactorBands
        | Sequence[InstrumentMultipactorBands]
        | None
    ),
    assert_positions_match: bool = True,
) -> zip:
    """Zip ``obj`` with ``instrument_multipactor_bands``.

    Perform some checkings, to ensure consistency between the
    instruments/measure points under study and the
    :class:`.InstrumentMultipactorBands` object.

    Parameters
    ----------
    obj :
        An instrument or pick-up.
    instrument_multipactor_bands :
        Objects holding info on when multipactor appears. If there is only one
        :class:`.InstrumentMultipactorBands`, it will be applied on all the
        ``obj``. ``None`` is allowed, may be removed in the future.
    assert_positions_match :
        To check if position where multipactor was checked must match the
        position of ``obj``.

    Returns
    -------
    zipper :
        Object storing matching pairs of ``obj`` and
        :class:`.InstrumentMultipactorBands`.

    """
    if instrument_multipactor_bands is None:
        return zip(obj, [None for _ in obj])

    if isinstance(instrument_multipactor_bands, InstrumentMultipactorBands):
        instrument_multipactor_bands = [
            instrument_multipactor_bands for _ in obj
        ]

    assert len(obj) == len(instrument_multipactor_bands), (
        f"Mismatch between {obj} ({len(obj) = }) and multipactor bands "
        f"({len(instrument_multipactor_bands) = })"
    )
    zipper = zip(obj, instrument_multipactor_bands, strict=True)

    if not assert_positions_match:
        return zipper

    for single_obj, mp_bands in zip(
        obj, instrument_multipactor_bands, strict=True
    ):
        if mp_bands is None:
            continue
        if positions_match(single_obj, mp_bands):
            continue
        raise OSError(
            f"The position of {single_obj} ({single_obj.position}) does not "
            f"match the position of {instrument_multipactor_bands} "
            f"({mp_bands.position})."
        )
    return zipper


def positions_match(
    obj: Instrument | IMeasurementPoint,
    instrument_multipactor_bands: InstrumentMultipactorBands,
    tol: float = 1e-6,
) -> bool:
    """Check that positions of argument objects are consistent.

    Parameters
    ----------
    obj :
        An object with a ``position`` attribute. It it is ``np.nan``, it means
        that the object under study is "global" and we return True.
    instrument_multipactor_bands :
        The multipactor bands to check. If its ``position`` is ``np.nan``, it
        means that the multipactor is detected at the scale of the whole
        testbench. In this case, we return True.
    tol :
        Tolerance over the position matching.

    Returns
    -------
    flag :
        If the positions of ``obj`` and ``instrument_multipactor_bands`` match.

    """
    if instrument_multipactor_bands is None:
        return True
    if instrument_multipactor_bands.position is np.nan:
        return True

    obj_pos = getattr(obj, "position", None)
    assert obj_pos is not None, (
        "position attribute should never be None. It should be np.nan for "
        "global instruments / measurement points."
    )

    if obj_pos is np.nan:
        return True

    if abs(instrument_multipactor_bands.position - obj_pos) > tol:
        return False
    return True
