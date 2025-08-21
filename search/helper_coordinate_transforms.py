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

def enu_matrix(lat_deg, lon_deg):
    """
    ECEF -> ENU rotation at site(s).
    Inputs:
        lat_deg, lon_deg : scalars or 1D arrays of equal length
    Output:
        - (3, 3) if scalars
        - (N, 3, 3) if arrays of length N
    """
    lat = np.radians(np.asarray(lat_deg))
    lon = np.radians(np.asarray(lon_deg))

    if lat.ndim == 0 and lon.ndim == 0:   # scalar case
        sinp, cosp = np.sin(lat), np.cos(lat)
        sinl, cosl = np.sin(lon), np.cos(lon)

        E = np.array([-sinl,        cosl,         0.0])
        N = np.array([-sinp*cosl,  -sinp*sinl,   cosp])
        U = np.array([ cosp*cosl,   cosp*sinl,   sinp])
        return np.array([E, N, U])

    # array case
    if lat.shape != lon.shape:
        raise ValueError("lat_deg and lon_deg must have the same shape")

    sinp, cosp = np.sin(lat), np.cos(lat)
    sinl, cosl = np.sin(lon), np.cos(lon)

    # Each of these has shape (N, 3)
    E = np.column_stack((-sinl,         cosl,           np.zeros_like(lat)))
    N = np.column_stack((-sinp*cosl,   -sinp*sinl,      cosp))
    U = np.column_stack(( cosp*cosl,    cosp*sinl,      sinp))

    # Stack into shape (N, 3, 3)
    Q = np.stack((E, N, U), axis=1)
    return Q

def geodetic_to_eci(lat_deg, local_sidereal_time, h_m):
    return geodetic_to_ecef(lat_deg, local_sidereal_time, h_m)

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """geodetic -> ECEF (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h_km = h_m * 1e-3
    sinp, cosp = np.sin(lat), np.cos(lat)
    R_φ = R_EARTH / np.sqrt(1.0 - (2.0*FLATTENING - FLATTENING**2) * sinp*sinp)

    R_c = R_φ + h_km
    R_s = (1-FLATTENING)**2 * R_φ + h_km

    x = (R_c) * cosp * np.cos(lon)
    y = (R_c) * cosp * np.sin(lon)
    z = np.full_like(lon, R_s * sinp)
    return np.array([x, y, z])