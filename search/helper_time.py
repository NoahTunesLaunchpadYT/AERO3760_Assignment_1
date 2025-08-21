import numpy as np
from helper_constants import *

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  c, -s],
                     [0.0,  s,  c]])

def kepler_to_rv(a_km, e, inc_deg, raan_deg, aop_deg, tru_deg, mu=MU_EARTH):
    # Convert degrees to radians
    i = np.radians(inc_deg)
    Ω = np.radians(raan_deg)
    ω = np.radians(aop_deg)
    ν = np.radians(tru_deg)

    p = a_km * (1.0 - e**2)

    r_pf = np.array([
        p / (1.0 + e*np.cos(ν)) * np.cos(ν),
        p / (1.0 + e*np.cos(ν)) * np.sin(ν),
        0.0
    ])
    v_pf = np.array([
        -np.sqrt(mu/p) * np.sin(ν),
         np.sqrt(mu/p) * (e + np.cos(ν)),
         0.0
    ])

    # Rotation from perifocal to ECI
    R = rot_z(Ω).dot(rot_x(i)).dot(rot_z(ω))
    r_eci = R.dot(r_pf)
    v_eci = R.dot(v_pf)
    return r_eci, v_eci

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
