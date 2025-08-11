import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.1363     # km
OMEGA_E  = 7.2921159e-5  # rad/s (Earth rotation)
A_WGS84 = 6378.137            # km
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)  # first eccentricity^2

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

def calendar_to_jd(Y, M, D, h, m, s):
    '''Convert calendar date to Julian Date.'''
    if M <= 2:
        Y -= 1
        M += 12
    A = int(Y / 100)
    B = 2 - A + int(A / 4)
    day_fraction = (h + m / 60 + s / 3600) / 24
    JD = int(365.25 * (Y + 4716)) + int(30.6001 * (M + 1)) + D + day_fraction + B - 1524.5
    return JD

def gmst_from_jd(jd):
    """Greenwich Mean Sidereal Angle (radians) from Julian Date."""
    jd = np.asarray(jd, float)
    T = (jd - 2451545.0) / 36525.0
    theta_deg = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - (T**3)/38710000.0
    return np.radians(theta_deg % 360.0)

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """WGS-84 geodetic -> ECEF (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h_km = h_m * 1e-3
    sinp, cosp = np.sin(lat), np.cos(lat)
    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sinp*sinp)
    x = (N + h_km) * cosp * np.cos(lon)
    y = (N + h_km) * cosp * np.sin(lon)
    z = (N * (1.0 - E2_WGS84) + h_km) * sinp
    return np.array([x, y, z])

def enu_matrix(lat_deg, lon_deg):
    """ECEF -> ENU rotation at site (rows are E,N,U unit vectors)."""
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    sinp, cosp = np.sin(lat), np.cos(lat)
    sinl, cosl = np.sin(lon), np.cos(lon)
    E = np.array([-sinl,            cosl,           0.0])
    N = np.array([-sinp*cosl, -sinp*sinl,     cosp])
    U = np.array([ cosp*cosl,  cosp*sinl,     sinp])  # surface-normal (Up) for geodetic lat
    return np.vstack((E, N, U))

# --- helpers ---
def _segmented_polar_arrays(az_deg, el_deg, min_el_deg: float):
    """Return theta,r arrays with NaNs at invisibility and azimuth wrap gaps."""
    theta = np.radians(az_deg)
    r = el_deg
    vis = el_deg >= float(min_el_deg)
    # break on invisibility
    split = ~vis.copy()
    # break on large azimuth wrap between samples (>180 deg)
    dth = np.abs(np.diff(theta))
    wrap = dth > np.pi
    jumps = np.zeros_like(theta, dtype=bool)
    jumps[1:] = wrap
    split |= jumps
    theta_plot = theta.copy(); r_plot = r.copy()
    theta_plot[split] = np.nan; r_plot[split] = np.nan
    return theta_plot, r_plot, vis, theta, r