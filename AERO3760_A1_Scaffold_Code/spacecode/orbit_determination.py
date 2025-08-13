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
    N = ...     # TODO: Equation (5.13)
    D = ...     # TODO: Equation (5.14)
    S = ...     # TODO: Equation (5.21)

    # Compute the velocity vector at r2
    N_mag = ...
    D_mag = ...

    v2 = ...    # TODO: Equation (5.22)

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

    # TODO: Equation (5.26) Hint: be careful of radians
    if (is_prograde and norm_vec[2] >= 0) or (not is_prograde and norm_vec[2] < 0):
        delta_theta = ...
    else:
        delta_theta = ...
    
    # -------------------- Calculate the Lagrange Coefficients
    Sz = stumpff_S(z)
    Cz = stumpff_C(z)

    A = ...     # TODO: Equation (5.35)
    y = ...     # TODO: Equation (5.38)

    # TODO: Equations(5.46a-d)
    f = ...
    g = ...
    fdot = ...
    gdot = ...

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

    v1 = ...    # TODO: Equation (5.28)
    v2 = ...    # TODO: Equation (5.29)

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

    A = ...     # TODO: Equation (5.112b)
    B = ...     # TODO: Equation (5.112c)
    E = ...     # TODO: Equation (5.115b)

    a = ...     # TODO: Equation (5.117a)
    b = ...     # TODO: Equation (5.117b)
    c = ...     # TODO: Equation (5.117c)
    
    r2 = ...    # Select a suitable initial guess (hint: it is not 0)
    tolerance = 1e-6
    update = 1

    # Perform Newton's method
    while abs(update) > tolerance:
        F = ...     # TODO: Equation (5.116)
        dFdx = ...  # TODO: Derivative of Equation (5.116)
        update = ...
        r2 = r2 - update
    
    # Calculate the Lagrange Coefficients
    