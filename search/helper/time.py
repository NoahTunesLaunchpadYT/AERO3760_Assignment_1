import numpy as np
from helper.constants import *

def calendar_to_jd(Y: int, M: int, D: int, h: int, m: int, s: float) -> float:
    if Y < 1901 or Y >= 2100:
        raise ValueError("Year must be between 1901 and 2099")
    ut = (h + m/60 + s/3600)/24.0
    j0 = 367*Y - int((7*(Y + int((M + 9)/12)))/4) + int((275*M)/9) + D + 1721013.5
    jd = j0 + ut

    return jd

def gmst_from_jd(jd):
    """Greenwich Mean Sidereal Angle (radians) from Julian Date."""
    jd = np.asarray(jd, float)
    T = (jd - 2451545.0) / 36525.0
    theta_deg = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - (T**3)/38710000.0
    return np.radians(theta_deg % 360.0)
