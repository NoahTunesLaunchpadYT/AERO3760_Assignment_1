"""
Note: Do not rename this module to time.py, as this will cause
Python's standard time module to be overridden.
"""

import datetime as dt


def utc_to_j0(utc: dt.datetime) -> float:
    """Converts a UTC datetime to J0 (midnight) of the corresponding
    Julian date.

    Args:
        utc (dt.datetime): UTC date

    Returns:
        float: J0 Julian date
    """
    Y = utc.year
    M = utc.month
    D = utc.day

    j0 = ...    # TODO: Equation (5.48)
    return j0


def utc_to_jd(utc: dt.datetime) -> float:
    """Converts a UTC date to its Julian Day

    Args:
        utc (dt.datetime): UTC date

    Returns:
        float: Julian date
    """
    j0 = utc_to_j0(utc)

    h = utc.hour
    m = utc.minute
    s = utc.second

    total_hours = ...       # TODO

    jd = j0 + total_hours / 24.0      # Equation (5.47)
    return jd


def sidereal_time(utc: dt.datetime, east_longitude: float = 0) -> float:
    """Converts a UTC time to sidereal time. By default, returns
    Greenwhich Mean Sidereal Time (i.e. east_longitude = 0).

    Args:
        utc (dt.datetime): UTC time
        east_longitude (float, optional): East longitude in degrees. Defaults to 0.

    Returns:
        float: Sidereal time in degrees
    """
    j0 = utc_to_j0(utc)

    T0 = ...    # TODO: Equation (5.49)

    theta_g0 = ...  # TODO: Equation (5.50)
    theta_g = ...   # TODO: Equation (5.51)

    sidereal_local = theta_g + east_longitude
    return sidereal_local % 360.0       # Modulo 360 to keep in range [0, 360)