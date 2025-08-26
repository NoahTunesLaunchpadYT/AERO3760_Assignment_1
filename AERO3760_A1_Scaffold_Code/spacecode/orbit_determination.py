from typing import Tuple
import numpy as np
import spacetools.constants as const

# The import path may need to change depending on your file structure
from universal import universal_variable_from_r, stumpff_S, stumpff_C


def gibbs(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, mu: float) -> np.ndarray:
    """Performs Gibbs method of orbit determination. 

    Note: Make sure the units of all inputs are consistent.

    Args:
        r1 (np.ndarray): First ECI vector
        r2 (np.ndarray): Second ECI vector
        r3 (np.ndarray): Third ECI vector
        mu (float): Gravitational constant of the primary body.

    Returns:
        np.ndarray: The velocity vector at the second observation.
    """
    # All equation numbers are from Orbital Mechanics for Engineering Students
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)

    # Intermediate equations
    N = r1_mag * (np.cross(r2, r3)) + r2_mag * (np.cross(r3, r1)) + r3_mag * (np.cross(r1, r2))
    D = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)
    S = r1 * (r2_mag - r3_mag) + r2 * (r3_mag - r1_mag) + r3 * (r1_mag - r2_mag)

    # Compute the velocity vector at r2
    N_mag = np.linalg.norm(N)
    D_mag = np.linalg.norm(D)

    v2 = np.sqrt(mu / (N_mag*D_mag)) * ((np.sqrt(D, r2)) / r2_mag + S)

    return v2


def lagrange_coefficients_universal(
    r1: np.ndarray,
    r2: np.ndarray,
    mu: float,
    z: float,
    is_prograde: bool
) -> tuple:
    """_summary_

    Args:
        r1 (np.ndarray): _description_
        r2 (np.ndarray): _description_
        mu (float): _description_
        z (float): _description_
        is_prograde (bool): _description_

    Returns:
        tuple: _description_
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # -------------------- Find the change in true anomaly from r1 to r2
    norm_vec = np.cross(r1, r2)     # Vector normal to the orbital plane

    if (is_prograde and norm_vec[2] >= 0) or (not is_prograde and norm_vec[2] < 0):
        delta_theta = np.arccos(np.clip(np.dot(r1, r2) / (r1_mag * r2_mag), -1.0, 1.0)) 
    else:
        delta_theta = 2 * np.pi - np.arccos(np.clip(np.dot(r1, r2) / (r1_mag * r2_mag), -1.0, 1.0))
    
    # -------------------- Calculate the Lagrange Coefficients
    Sz = stumpff_S(z)
    Cz = stumpff_C(z)

    A = np.sin(delta_theta) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(delta_theta)))
    y = r1_mag + r2_mag + A * np.sqrt((z*Sz - 1) / np.sqrt(Cz))

    f = 1 - y / r1_mag
    g = A * np.sqrt(y) / np.sqrt(mu)
    fdot = np.sqrt(mu) / (r1_mag * r2_mag) * np.sqrt(y / Cz) * (z * Sz - 1)
    gdot = 1 - y / r2_mag

    return f, g, fdot, gdot


def lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    dt: float,
    mu: float,
    is_prograde: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Solves Lambert's problem by finding the required velocity vectors at r1 and r2.

    Note: The units of r1, r2 and mu must be consistent.

    Args:
        r1 (np.ndarray): Starting position vector
        r2 (np.ndarray): Ending position vector
        dt (float): Time of flight between r1 and r2
        mu (float): Gravitational parameter of the primary body
        is_prograde (bool): True is the orbit is prograde, False if retrograde.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The velocity vectors at r1 and r2.
    """
    z = universal_variable_from_r(r1, r2, dt, mu, is_prograde)
    f, g, fdot, gdot = lagrange_coefficients_universal(r1, r2, mu, z, is_prograde)

    v1 = 1 / g * (r2 - f * r1)
    v2 = 1 / g * (gdot * r2 - r1)
    
    return v1, v2


def gauss(
    obs1: Tuple[np.ndarray, np.ndarray, float],
    obs2: Tuple[np.ndarray, np.ndarray, float],
    obs3: Tuple[np.ndarray, np.ndarray, float],
    mu: float
) -> np.ndarray:
    """Performs Gauss' method of angle only orbit determination.

    Each observation consists of a bearing (unit) vector from the ground station,
    the ground station's ECI position, and relative timestamp in seconds. Absolute
    UTC isn't required, timestamps need only reflect correct time intervals
    e.g. t1 = 6, t2 = 7, t3 = 8 is valid if all measurements were taken
    one second apart, even if in reality, t2 = 25-07-2025 12:54:05.28.   

    WARNING: obs1, obs2 and obs3 must not lie on the same plane.

    Args:
        obs1 (Tuple[np.ndarray, np.ndarray, float]): First observation
        obs2 (Tuple[np.ndarray, np.ndarray, float]): Second observation
        obs3 (Tuple[np.ndarray, np.ndarray, float]): Third observation
        mu (float): Gravitational parameter of the primary body

    Returns:
        np.ndarray: ECI velocity vector at the second observation.
    """
    # Bearing vector, Ground Station ECI position, and timestamp
    q1, R1, t1 = obs1
    q2, R2, t2 = obs2
    q3, R3, t3 = obs3

    # Time intervals between observations
    tau_1 = t1 - t2             # Equation (5.98)
    tau_3 = t3 - t2
    tau = tau_3 - tau_1         # Equation (5.101)

    # Intermediate equations
    p1 = np.cross(q2, q3)
    p2 = np.cross(q1, q3)
    p3 = np.cross(q1, q2)

    D0 = np.dot(q1, p1)        # Equation (5.108)

    D11, D12, D13 = np.dot(R1, p1), np.dot(R1, p2), np.dot(R1, p3)
    D21, D22, D23 = np.dot(R2, p1), np.dot(R2, p2), np.dot(R2, p3)
    D31, D32, D33 = np.dot(R3, p1), np.dot(R3, p2), np.dot(R3, p3)

    # Solve for the satellite range at obs2 by forming the range polynomial
    R2_mag2 = np.dot(R2, R2)        # squared magnitude

    A = 1 / D0 * (-D12 * tau_3/tau + D22 + D32 * tau_1/tau)
    B = 1 / (6 * D0) * (D12 * (tau_3**2 - tau**2) * (tau_3/tau) + D32 * (tau**2 - tau_1**2) * (tau_1/tau))
    E = R2 * q2

    a = -(A**2 + 2*A*E + R2_mag2)
    b = -2*mu*B*(A + E)
    c = -mu**2 * B**2
    
    r2_mag = 20000 # km    # Select a suitable initial guess (hint: it is not 0)
    tolerance = 1e-6
    update = 1

    # Perform Newton's method
    while abs(update) > tolerance:
        F = r2_mag**8 + a * r2_mag**6 + b * r2_mag**3 + c
        dFdx = 8 * r2_mag**7 + 6 * a * r2_mag**5 + 3 * b * r2_mag**2
        update = F / dFdx
        r2_mag = r2_mag - update


    q2_mag = A + mu * B / r2_mag**3

    q1_mag = 1 / D0 * ((6*(D31 * tau_1/tau_3 + D21 * tau/tau_3) * r2_mag**3 + mu * D31 * (tau**2 - tau_1**2) * tau_1/tau_3)
                       / (6*r2_mag**3 + mu * (tau**2 - tau_3**2))
                       - D11
                        )
    
    q3_mag = 1 / D0 * ((6*(D13 * tau_3/tau_1 - D23 * tau/tau_1) * r2_mag**3 + mu * D13 * (tau**2 - tau_3**2) * tau_3/tau_1)
                       / (6*r2_mag**3 + mu * (tau**2 - tau_3**2))
                       - D21
                       )

    r1 = R1 + q1 * (q1_mag)
    r2 = R2 + q2 * (q2_mag)
    r3 = R3 + q3 * (q3_mag)

    f_1 = 1 - 1/2 * mu/r2_mag**3 * tau_1**2
    f_3 = 1 - 1/2 * mu/r2_mag**3 * tau_3**2

    g_1 = tau_1 - 1/6 * mu/r2_mag**3 * tau_1**3
    g_3 = tau_3 - 1/6 * mu/r2_mag**3 * tau_3**3

    v_2 = 1 / (f_1 * g_3 - f_3 * g_1) * (-f_3 * r1 + f_1 * r3)   # Equation (5.114)

    return v_2

    # Calculate the Lagrange Coefficients
