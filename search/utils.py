import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.1363     # km
OMEGA_E  = 7.2921159e-5  # rad/s (Earth rotation)

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