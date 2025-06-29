"""Define utility functions related to thresholds."""


def extract_detecting_name(label: str) -> str:
    """Get instrument name from a thresholds df column header.

    Parameters
    ----------
    label :
        Column header, looking like ``"CurrentProbe (NI9205_MP4l) upper"``.

    Returns
    -------
        Detecting instrument name, like ``"NI9205_MP4l"``.

    """
    if "(" in label:
        return label.rsplit("(", 1)[1].split(")")[0]

    raise ValueError(f"{label = } not recognized.")
