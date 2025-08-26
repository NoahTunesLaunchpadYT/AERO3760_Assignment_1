import numpy as np
from helper.constants import *

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

def kepler_to_rv(a_km, e, inc_deg, raan_deg, aop_deg, ta_deg, mu=MU_EARTH):
    # Convert degrees to radians
    i = np.radians(inc_deg)
    Ω = np.radians(raan_deg)
    ω = np.radians(aop_deg)
    ν = np.radians(ta_deg)

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

def az_el_sat_from_gs(gs_traj, sat_traj, return_distance=False):
    """
    Compute azimuth/elevation of a satellite as seen from a ground station,
    using the station's time-varying ENU basis in ECI.

    Args:
        gs_traj: Ground station trajectory with attributes JD, R, E_eci, N_eci, U_eci
        sat_traj: Satellite trajectory with attributes JD, R
        return_distance (bool): If True, also return the satellite-ground
                                distance (same length as az/el arrays).

    Returns:
        az_deg (n,), el_deg (n,), vis_mask (n,) 
        If return_distance=True, also returns:
        distance_m (n,)
    """
    # Ensure same time base length
    if gs_traj.JD.shape[0] != sat_traj.JD.shape[0]:
        raise ValueError("GS and SAT trajectories must be sampled on the same JD grid")

    Rg = gs_traj.R                    # (n,3)
    Rs = sat_traj.R                   # (n,3)
    rho = Rs - Rg                     # LOS in ECI (n,3)

    # Local ENU basis in ECI
    E_eci = getattr(gs_traj, 'E_eci', None)
    N_eci = getattr(gs_traj, 'N_eci', None)
    U_eci = getattr(gs_traj, 'U_eci', None)
    if E_eci is None or N_eci is None or U_eci is None:
        raise RuntimeError("GroundStationTrajectory is missing E_eci/N_eci/U_eci. "
                        "Make sure you ran GroundStationPropagator and stored these.")

    # Project LOS onto ENU        
    e = np.einsum('ij,ij->i', rho, E_eci)  # East component
    n = np.einsum('ij,ij->i', rho, N_eci)  # North component
    u = np.einsum('ij,ij->i', rho, U_eci)  # Up component

    # Azimuth measured from North toward East
    az = np.degrees(np.arctan2(e, n))      # [-180, 180]
    az = (az + 360.0) % 360.0

    # Elevation and range
    rho_norm = np.linalg.norm(rho, axis=1)
    el = np.degrees(np.arcsin(np.clip(u / rho_norm, -1.0, 1.0)))

    vis = el >= 0.0

    if return_distance:
        return az, el, vis, rho_norm
    else:
        return az, el, vis

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

def ENU_from_az_el_rng(az_deg, el_deg, rng_km):
    """
    Convert azimuth/elevation/range to a vector in East-North-Up (ENU).
    Inputs:
        az_deg: Azimuth angle in degrees, measured from North toward East.
        el_deg: Elevation angle in degrees.
        rng_km: Range in kilometers.

    Returns:
    np.ndarray: A 3-element array representing the ENU unit vectors.
    """
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)

    # Compute the ENU unit vectors
    e = np.cos(el_rad) * np.sin(az_rad)
    n = np.cos(el_rad) * np.cos(az_rad)
    u = np.sin(el_rad)

    return rng_km * np.array([e, n, u])

def u_from_az_el(az, el, *, degrees: bool = True, az_from: str = "north"):
    """
    Direction cosines (unit vector) in ENU from azimuth/elevation.

    Parameters
    ----------
    az : float or ndarray
        Azimuth angle. If az_from="north", az=0 points to North and increases toward East.
        If az_from="east", az=0 points to East and increases toward North.
    el : float or ndarray
        Elevation angle above the local horizon (−90°..+90°).
    degrees : bool
        If True, inputs are in degrees. If False, radians.
    az_from : {"north","east"}
        Azimuth reference direction.

    Returns
    -------
    u : ndarray (..., 3)
        Unit vector [E, N, U] in the local ENU frame.
    """
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)

    if degrees:
        az = np.deg2rad(az)
        el = np.deg2rad(el)

    ce, se = np.cos(el), np.sin(el)
    ca, sa = np.cos(az), np.sin(az)

    if az_from == "north":
        # Az measured from North, increasing to the East (common geodesy convention)
        E = ce * sa
        N = ce * ca
    elif az_from == "east":
        # Az measured from East, increasing to the North
        E = ce * ca
        N = ce * sa
    else:
        raise ValueError("az_from must be 'north' or 'east'")

    U = se
    u = np.stack((E, N, U), axis=-1)

    return u