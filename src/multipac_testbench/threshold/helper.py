"""Define utility functions related to thresholds."""

from collections import defaultdict

from multipac_testbench.instruments.instrument import Instrument
from multipac_testbench.threshold.threshold import THRESHOLD_WAY_T, Threshold


def threshold_df_column_header(
    instrument: Instrument,
    threshold: Threshold,
) -> str:
    """Create consistent column header for threshold dataframes.

    Parameters
    ----------
    instrument :
        Object that measured data in currently labelled column.
    threshold :
        Threshold for which you want a header.

    Returns
    -------
    str
        Threshold dataframe column header, looking like: ``"NI9205_E4 @ upper
        threshold (according to NI9205_MP4l)"``.

    """
    if isinstance(threshold.detecting_instrument, Instrument):
        detecting_name = threshold.detecting_instrument.name
    else:
        detecting_name = threshold.detecting_instrument
    header = (
        f"{instrument.name} @ {threshold.nature} threshold "
        f"(according to {detecting_name})"
    )
    return header


def extract_measured_name(label: str) -> str:
    """Get instrument name from a thresholds df column header.

    Parameters
    ----------
    label :
        Threshold dataframe column header, looking like: ``"NI9205_E4 @ upper
        threshold (according to NI9205_MP4l)"``.

    Returns
    -------
    str
        Measure instrument name, like ``"NI9205_E4"``.

    """
    if "@" in label:
        return label.split("@", 1)[0][:-1]

    raise ValueError(f"{label = } not recognized.")


def extract_detecting_name(label: str) -> str:
    """Get detecting instrument name from a thresholds df column header.

    Parameters
    ----------
    label :
        Threshold dataframe column header, looking like: ``"NI9205_E4 @ upper
        threshold (according to NI9205_MP4l)"``.

    Returns
    -------
    str
        Detecting instrument name, like ``"NI9205_MP4l"``.

    """
    if "(" in label:
        return label.rsplit("(", 1)[1].split(" ")[2][:-1]

    raise ValueError(f"{label = } not recognized.")


def reached_last_upper(thresholds: list[Threshold]) -> bool:
    """Determine if all lower thresholds have a matching upper."""
    grouped: dict[str, list[Threshold]] = defaultdict(list)
    for t in thresholds:
        grouped[t.detecting_instrument].append(t)
    for thresh in grouped.values():
        sorted_thresholds = sorted(thresh, key=lambda t: t.sample_index)
        expecting = None
        for t in sorted_thresholds:
            if t.nature == "lower":
                expecting = "upper"
            elif t.nature == "upper" and expecting == "upper":
                expecting = None
        if expecting == "upper":
            return False  # still expecting an upper threshold
    return True


def sorter_index_then_way(threshold: Threshold) -> tuple[int, int]:
    """Give ``sample_index`` and ``way`` of ``threshold``.

    Used to sort thresholds:
    - first, by sample index
    - in case of equality, ``"enter" < "exit"``.

    """
    ways: dict[THRESHOLD_WAY_T, int] = {"enter": 0, "exit": 1}
    return threshold.sample_index, ways[threshold.way]
