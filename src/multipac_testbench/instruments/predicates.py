"""Provide helper functions for filtering :class:`.Instrument`."""

from abc import ABCMeta
from collections.abc import Collection
from typing import Callable, Sequence, cast, overload

from multipac_testbench.instruments import Instrument
from multipac_testbench.measurement_point.i_measurement_point import (
    IMeasurementPoint,
)
from multipac_testbench.util.helper import is_collection_of

#: A single :class:`Instrument` identifier, as accepted by filter predicates.
#: Used for building predicates, sometimes for applying predicates.
INSTRUMENT_ID = ABCMeta | Instrument | str
#: Identifier for one or several :class:`Instrument` classes or a collection of
#: instances/names. Used to apply predicates.
INSTRUMENTS_ID = (
    ABCMeta | Sequence[ABCMeta] | Sequence[str] | Sequence[Instrument]
)
#: Function to filter :class:`Instrument`. Returns True if it should be kept,
#: False if it should be discarded.
INSTRUMENT_FILTER = Callable[[INSTRUMENT_ID], bool]

#: Identifier for measurement points, used in deprecated filtering arguments.
MEASUREMENT_POINTS_ID = Collection[IMeasurementPoint] | Collection[str]


class InstrumentFilteringError(ValueError):
    """Error raised when filtering logic was inconsistent."""


def _to_name_set(instruments_to_ignore: INSTRUMENTS_ID) -> set[str]:
    """Normalize instruments or names into a set of strings.

    Raises
    ------
    InstrumentFilteringError
        If ``instruments_to_ignore`` contains or is an :class:`ABCMeta`.

    """
    _error = InstrumentFilteringError(
        "name-based filters must be created with (list of) names or "
        "Instrument instances."
    )
    if isinstance(instruments_to_ignore, ABCMeta):
        raise _error
    if not isinstance(instruments_to_ignore, Sequence) or isinstance(
        instruments_to_ignore, str
    ):
        instruments_to_ignore = (instruments_to_ignore,)
    if is_collection_of(instruments_to_ignore, ABCMeta):
        raise _error
    if is_collection_of(instruments_to_ignore, Instrument) or is_collection_of(
        instruments_to_ignore, str
    ):
        return {str(i) for i in instruments_to_ignore}
    raise _error


def dummy_instrument_filter(instrument_id: INSTRUMENT_ID) -> bool:
    """Create filter that does not filter anything."""
    return True


def instrument_type_selector(
    instrument_types: ABCMeta | Collection[ABCMeta],
) -> INSTRUMENT_FILTER:
    """Return instruments of any (sub)types of ``instrument_types``.

    .. todo::
        Accept ``str`` inputs?

    """
    if isinstance(instrument_types, ABCMeta):
        instrument_types = (instrument_types,)
    else:
        instrument_types = tuple(instrument_types)

    def predicate(instrument_id: INSTRUMENT_ID) -> bool:
        match instrument_id:
            case type():
                return issubclass(instrument_id, instrument_types)
            case Instrument():
                return isinstance(instrument_id, instrument_types)
            case str():
                raise InstrumentFilteringError(
                    "Cannot determine instrument type from a "
                    f"{type(instrument_id)}."
                )
            case _:
                raise InstrumentFilteringError(
                    f"instrument_id is a {type(instrument_id)}, which is not "
                    "supported."
                )

    return predicate


def instrument_name_selector(
    instrument_names: str | Collection[str],
) -> INSTRUMENT_FILTER:
    """Return instruments whose name matches any of ``instrument_names``.

    Parameters
    ----------
    instrument_names :
        Instrument name(s) to match against. Can be a single string or a
        sequence of strings.

    Returns
    -------
        A predicate that returns True if the instrument's name (str, repr, or
        .name) matches any of the provided names.

    """
    if isinstance(instrument_names, str):
        instrument_names = (instrument_names,)
    else:
        instrument_names = tuple(instrument_names)

    name_set = set(instrument_names)

    def predicate(instrument_id: INSTRUMENT_ID) -> bool:
        match instrument_id:
            case type():
                raise InstrumentFilteringError(
                    "name-based selectors apply on Instrument instances "
                    "names, they do not work on Instrument classes."
                )
            case Instrument():
                return any(
                    name in name_set
                    for name in (
                        str(instrument_id),
                        repr(instrument_id),
                        instrument_id.name,
                    )
                )
            case str():
                return instrument_id in name_set
            case _:
                raise InstrumentFilteringError(
                    f"instrument_id is a {type(instrument_id)}, which is not "
                    "supported."
                )

    return predicate


def instrument_excluder(
    instruments_to_ignore: INSTRUMENTS_ID,
) -> INSTRUMENT_FILTER:
    """Create filter that rejects ``instruments_to_ignore``.

    Parameters
    ----------
    instruments_to_ignore :
        Instrument(s) name(s), ``str``, ``repr`` or instances.

    Returns
    -------
        Can be applied to :class:`.Instrument` instances or
        :class:`.Instrument` names. Does not make any sense with
        :class:`.Instrument` types.

    """
    as_set_of_names = _to_name_set(instruments_to_ignore)

    def predicate(instrument_id: INSTRUMENT_ID) -> bool:
        """Reject instruments whose name is in ``as_set``.

        We compare :meth:`.Instrument.__str__`,
        :meth:`.Instrument.__repr__` and :attr:`.Instrument.name`.

        """
        match instrument_id:
            case type():
                raise InstrumentFilteringError(
                    "name-based filters apply on Instrument instances "
                    "names, they do not work on Instrument classes."
                )
            case Instrument():
                return as_set_of_names.isdisjoint(
                    (
                        str(instrument_id),
                        repr(instrument_id),
                        instrument_id.name,
                    )
                )
            case str():
                return instrument_id not in as_set_of_names
            case _:
                raise InstrumentFilteringError(
                    f"instrument_id is a {type(instrument_id)}, which is not "
                    "supported."
                )

    return predicate


def measurement_point_excluder(
    measurement_points: MEASUREMENT_POINTS_ID,
) -> INSTRUMENT_FILTER:
    """Create filter that rejects instruments located at `measurement_points`.

    .. todo::
       Implement predicate applying on measurement points names?

    Parameters
    ----------
    measurement_points :
        :class:`.IMeasurementPoint` instances or names.

    Returns
    -------
        A predicate that returns True if the instrument is NOT in a measurement
        point to exclude (*ie* if the instrument should be kept).

    """
    if is_collection_of(measurement_points, IMeasurementPoint):
        names_to_exclude = {
            i.name for point in measurement_points for i in point.instruments
        }

    else:
        raise NotImplementedError(
            "To filter on measurement_points, you must provide actual objects,"
            " not just their names."
        )

    def predicate(instrument_id: INSTRUMENT_ID) -> bool:
        """Reject instruments belonging to a given measurement point."""
        match instrument_id:
            case type():
                raise InstrumentFilteringError(
                    "name-based filters apply on Instrument instances "
                    "names, they do not work on Instrument classes."
                )
            case Instrument():
                return instrument_id.name not in names_to_exclude
            case str():
                return instrument_id not in names_to_exclude
            case _:
                raise InstrumentFilteringError(
                    f"instrument_id is a {type(instrument_id)}, which is not "
                    "supported."
                )

    return predicate


def combine_predicates(*predicates: INSTRUMENT_FILTER) -> INSTRUMENT_FILTER:
    """Combine multiple predicates into a single filter using logical AND.

    Parameters
    ----------
    *predicates :
        Filters to combine. All must return ``True`` for an instrument to pass.

    Returns
    -------
        A filter that returns ``True`` only if all ``predicates`` return
        ``True``. Returns ``True`` for any input if no predicates are given.

    """

    def combined(instrument_id: INSTRUMENT_ID) -> bool:
        return all(p(instrument_id) for p in predicates)

    return combined


@overload
def filter_instruments(
    instruments_id: Sequence[ABCMeta], predicate: INSTRUMENT_FILTER
) -> list[ABCMeta]: ...
@overload
def filter_instruments(
    instruments_id: Sequence[str], predicate: INSTRUMENT_FILTER
) -> list[str]: ...
@overload
def filter_instruments(
    instruments_id: Sequence[Instrument], predicate: INSTRUMENT_FILTER
) -> list[Instrument]: ...
@overload
def filter_instruments(
    instruments_id: ABCMeta, predicate: INSTRUMENT_FILTER
) -> ABCMeta: ...


def filter_instruments(
    instruments_id: INSTRUMENTS_ID, predicate: INSTRUMENT_FILTER
) -> list[ABCMeta] | list[Instrument] | list[str] | ABCMeta:
    """Apply ``predicate`` filter on given instruments."""
    if isinstance(instruments_id, ABCMeta):
        raise NotImplementedError("Filtering class is not implemented yet.")
    return cast(
        list[ABCMeta] | list[Instrument] | list[str],
        [x for x in instruments_id if predicate(x)],
    )
