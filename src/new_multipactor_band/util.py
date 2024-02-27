#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensure that :class:`.InstrumentMultipactorBands` are consistently used."""
from abc import ABCMeta
from typing import overload
import numpy as np
from collections.abc import Callable, Sequence

from multipac_testbench.src.measurement_point.i_measurement_point import \
    IMeasurementPoint
from multipac_testbench.src.new_multipactor_band.instrument_multipactor_bands import \
    InstrumentMultipactorBands
from multipac_testbench.src.instruments.instrument import Instrument


@overload
def match_with_mp_band(
    obj: Instrument | Sequence[Instrument] | IMeasurementPoint | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: InstrumentMultipactorBands | Sequence[InstrumentMultipactorBands],
    assert_positions_match: bool,
    find_matching_pairs: bool = False,
    **matching_pair_kw: bool,
) -> zip: ...


@overload
def match_with_mp_band(
    obj: Sequence[Instrument] | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: Sequence[InstrumentMultipactorBands],
    assert_positions_match: bool,
    find_matching_pairs: bool = True,
    **matching_pair_kw: bool,
) -> zip: ...


def match_with_mp_band(
    obj: Instrument | Sequence[Instrument] | IMeasurementPoint | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: InstrumentMultipactorBands | Sequence[InstrumentMultipactorBands],
    assert_positions_match: bool = True,
    find_matching_pairs: bool = False,
    **matching_pair_kw: bool,
) -> zip:
    """Zip ``instruments`` with ``instrument_multipactor_bands``.

    Perform some checkings, to consistency between the instruments under study
    and the :class:`.InstrumentMultipactorBands` object.

    Parameters
    ----------
    obj: Instrument | Sequence[Instrument] | IMeasurementPoint | Sequence[IMeasurementPoint]
        Objects for which you need to find multipactor.
    instrument_multipactor_bands : InstrumentMultipactorBands | Sequence[InstrumentMultipactorBands]
        Objects holding info on when multipactor appears. If there is only one
        :class:`.InstrumentMultipactorBands`, it will be applied on all the
        ``obj``.
    assert_positions_match : bool, optional
        To check if position where multipactor was checked must match the
        position of the instruments. The default is True.
    find_matching_pairs : bool, optional
        Use this to ensure that for every item of ``obj``, there is a matching
        :class:`.InstrumentMultipactorBands`. Use this when there is a length mismatch,
        or when one of the items of ``obj`` does not have a
        :class:`InstrumentMultipactorBands`. The default is False.

    Returns
    -------
    zip
        Object storing matching pairs of ``obj`` item with
        :class:`.InstrumentMultipactorBands`.

    """
    if isinstance(obj, (Instrument, IMeasurementPoint)):
        obj = obj,

    if find_matching_pairs:
        instrument_multipactor_bands = _match_by_name(
            obj,
            instrument_multipactor_bands,
            **matching_pair_kw)

    zipper = _create_zip(
        obj,
        instrument_multipactor_bands,
        assert_positions_match=assert_positions_match)
    return zipper


def _create_zip(
    obj: Sequence[Instrument] | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: InstrumentMultipactorBands | Sequence[InstrumentMultipactorBands] | None,
    assert_positions_match: bool = True,
) -> zip:
    """Zip ``obj`` with ``instrument_multipactor_bands``.

    Perform some checkings, to ensure consistency between the
    instruments/measure points under study and the :class:`.InstrumentMultipactorBands`
    object.

    Parameters
    ----------
    obj : Sequence[Instrument] | Sequence[IMeasurementPoint]
        An instrument or pick-up.
    instrument_multipactor_bands : InstrumentMultipactorBands | Sequence[InstrumentMultipactorBands] | None
        Objects holding info on when multipactor appears. If there is only one
        :class:`.InstrumentMultipactorBands`, it will be applied on all the ``obj``. None
        is allowed, may be removed in the future.
    assert_positions_match : bool, optional
        To check if position where multipactor was checked must match the
        position of ``obj``. The default is True.

    Returns
    -------
    zip
        Object storing matching pairs of ``obj`` and
        :class:`.InstrumentMultipactorBands`.

    """
    if instrument_multipactor_bands is None:
        return zip(obj, [None for _ in obj])

    if isinstance(instrument_multipactor_bands, InstrumentMultipactorBands):
        instrument_multipactor_bands = [instrument_multipactor_bands for _ in obj]

    assert len(obj) == len(instrument_multipactor_bands), (
        f"Mismatch between {obj} ({len(obj) = }) and "
        f"multipactor bands ({len(instrument_multipactor_bands) = })"
    )
    zipper = zip(obj, instrument_multipactor_bands, strict=True)

    if not assert_positions_match:
        return zipper

    for single_obj, mp_bands in zip(obj, instrument_multipactor_bands, strict=True):
        if mp_bands is None:
            continue
        if positions_match(single_obj, mp_bands):
            continue
        raise IOError(f"The position of {single_obj} ({single_obj.position})"
                      f"does not match the position of {instrument_multipactor_bands} "
                      f"({mp_bands.position}).")
    return zipper


@overload
def _match_by_name(
    obj: Sequence[Instrument] | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: Sequence[InstrumentMultipactorBands],
    assert_every_obj_has_instrument_multipactor_bands: bool = True,
) -> Sequence[InstrumentMultipactorBands]: ...


@overload
def _match_by_name(
    obj: Sequence[Instrument] | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: Sequence[InstrumentMultipactorBands],
    assert_every_obj_has_instrument_multipactor_bands: bool = False,
) -> Sequence[InstrumentMultipactorBands | None]: ...


def _match_by_name(
    obj: Sequence[Instrument] | Sequence[IMeasurementPoint],
    instrument_multipactor_bands: Sequence[InstrumentMultipactorBands],
    assert_every_obj_has_instrument_multipactor_bands: bool = True,
) -> Sequence[InstrumentMultipactorBands | None]:
    """Find the ``instrument_multipactor_bands`` matching the given ``obj``.

    This is done by comparing the ``name`` attribute of ``obj`` with the
    ``instrument_name`` or ``measurement_point_name`` attributes of
    ``instrument_multipactor_bands``.

    Parameters
    ----------
    obj : Sequence[Instrument] | Sequence[IMeasurementPoint]
        An instrument or pick-up/global diagnostic. Must have a ``name``
        attribute.
    instrument_multipactor_bands : Sequence[InstrumentMultipactorBands]
        Length and ordering can be different from ``obj``.

    Returns
    -------
    matching_instrument_multipactor_bands : Sequence[InstrumentMultipactorBands | None]
        Same length and ordering as ``obj``.

    """
    band_name_getter = _band_name_getter(type(obj[0]))
    matching_instrument_multipactor_bands = []

    for single_obj in obj:
        single_matching = [band for band in instrument_multipactor_bands
                           if band_name_getter(band) == single_obj.name]
        if len(single_matching) == 0:
            if assert_every_obj_has_instrument_multipactor_bands:
                raise IOError("No InstrumentMultipactorBands was found for "
                              f"{single_obj = }")
            matching_instrument_multipactor_bands.append(None)
            continue

        if len(single_matching) > 1:
            raise IOError("Several InstrumentMultipactorBands match this object: "
                          "undefined policy.")
        matching_instrument_multipactor_bands.append(single_matching[0])
    return matching_instrument_multipactor_bands


def _band_name_getter(obj_type: ABCMeta) -> Callable[[InstrumentMultipactorBands], str]:
    """Get the proper function to get the :class:`.InstrumentMultipactorBands` names."""
    if issubclass(obj_type, Instrument):
        return lambda band: band.instrument_name
    if issubclass(obj_type,  IMeasurementPoint):
        return lambda band: band.measurement_point_name
    raise TypeError(f"{obj_type = } not supported")


def positions_match(obj: Instrument | IMeasurementPoint,
                    instrument_multipactor_bands: InstrumentMultipactorBands,
                    tol: float = 1e-6) -> bool:
    """
    Check that positions of ``obj`` and ``instrument_multipactor_bands`` are consistent.

    Parameters
    ----------
    obj : Instrument | IMeasurementPoint
        An object with a ``position`` attribute. It it is ``np.NaN``, it means
        that the object under study is "global" and we return True.
    instrument_multipactor_bands : InstrumentMultipactorBands
        The multipactor bands to check. If its ``position`` is ``np.NaN``, it
        means that the multipactor is detected at the scale of the whole
        testbench. In this case, we return True.
    tol : float, optional
        Tolerance over the position matching. The default is ``1e-6``.

    Returns
    -------
    bool
        If the positions of ``obj`` and ``instrument_multipactor_bands`` match or not.

    """
    if instrument_multipactor_bands is None:
        return True
    if instrument_multipactor_bands.position is np.NaN:
        return True

    obj_pos = getattr(obj, 'position', None)
    assert obj_pos is not None, "position attribute should never be None. It" \
        + " should be np.NaN for global instruments / measurement points."

    if obj_pos is np.NaN:
        return True

    if abs(instrument_multipactor_bands.position - obj_pos) > tol:
        return False
    return True
