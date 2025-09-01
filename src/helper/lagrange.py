import numpy as np

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
    try:
        Sz = stumpff_S(z)
        Cz = stumpff_C(z)
    except Exception:
        return None

    A = np.sin(delta_theta) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(delta_theta)))
    y = r1_mag + r2_mag + A * (z*Sz - 1) / np.sqrt(Cz)

    f = 1 - y / r1_mag
    g = A * np.sqrt(y) / np.sqrt(mu)
    fdot = np.sqrt(mu) / (r1_mag * r2_mag) * np.sqrt(y / Cz) * (z * Sz - 1)
    gdot = 1 - y / r2_mag

    return f, g, fdot, gdot

def stumpff_S(z: float) -> float:
    """Calculates the Stumpff function S(z)

    Args:
        z (float): universal anomaly

    Returns:
        float
    """
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / z**(3/2)
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (-z)**(3/2)
    else:
        return 1/6


def stumpff_C(z: float) -> float:
    """Calculates the Stumpff function C(z)

    Args:
        z (float): universal anomaly

    Returns:
        float
    """
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / -z
    else:
        return 1/2


def universal_variable_from_r(
    r1: np.ndarray,
    r2: np.ndarray,
    dt: float,
    mu: float,
    is_prograde: bool
) -> float:
    """Calculates the universal variable of r2 relative r1.

    The universal variable `z = (1 / a) * chi^2`, where `a` is the
    semi-major axis and `chi` is the universal variable.

    Args:
        r1 (np.ndarray): Starting position vector
        r2 (np.ndarray): Findal position vector
        dt (float): Time of flight between r1 and r2
        mu (float): Gravitational parameter of the primary body
        is_prograde (bool): True if the orbit is prograde, False if retrograde

    Returns:
        float: The universal variable.
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # -------------------- Find the change in true anomaly from r1 to r2
    norm_vec = np.cross(r1, r2)     # Vector normal to the orbital plane

    if (is_prograde and norm_vec[2] >= 0) or (not is_prograde and norm_vec[2] < 0):
        delta_theta = np.arccos(np.clip(np.dot(r1, r2) / (r1_mag * r2_mag), -1.0, 1.0)) 
    else:
        delta_theta = 2 * np.pi - np.arccos(np.clip(np.dot(r1, r2) / (r1_mag * r2_mag), -1.0, 1.0))

    A = np.sin(delta_theta) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(delta_theta)))

    # Note: The following three functions are called inner (or nested) functions
    def y_func(z: float) -> float:
        Sz = stumpff_S(z)
        Cz = stumpff_C(z)

        return r1_mag + r2_mag + A * (z * Sz - 1) / np.sqrt(Cz)

    def F(z: float) -> float:
        # Hint: The variables `A` and `dt` are already defined
        Sz = stumpff_S(z)
        Cz = stumpff_C(z)
        y = y_func(z)

        return (y / Cz) ** (3/2) * Sz + A * np.sqrt(y) - np.sqrt(mu)*dt
    
    def dFdz(z: float) -> float:
        # Derivative of F(z) w.r.t to z
        # Hint: The variables `A` and `dt` are already defined
        Sz = stumpff_S(z)
        Cz = stumpff_C(z)
        y = y_func(z)

        if z == 0:
            return np.sqrt(2) / 40 * y**(3/2) + (A/8)*(np.sqrt(y) + A * np.sqrt(1/(2*np.sqrt(y))))
        else:
            return (y / Cz) ** (3/2) * (1 / (2*z) * (Cz - 3/2 * Sz/Cz) + 3/4 * Sz**2/Cz) + A/8 * (3 * Sz/Cz * np.sqrt(y) + A * np.sqrt(Cz / y))

    # Initial guess for the universal variable
    z = 0
    tolerance = 1e-6
    update = 1

    while abs(update) > tolerance:
        update = F(z) / dFdz(z)
        z = z - update
    
    return z

def universal_variable_from_dt_r_v_alpha(dt: float, r0_mag: float, v0_r: float, alpha: float, mu: float) -> float:
    """Calculates the universal variable from the given parameters.

    Args:
        dt (float): Time of flight
        r0_mag (float): Magnitude of the initial position vector
        v0_r (float): Radial component of the initial velocity vector
        alpha (float): Flight path angle
        mu (float): Gravitational parameter

    Returns:
        float: The universal variable.
    """
    Chi = np.sqrt(mu) * abs(alpha) * dt
    tolerance = 1e-8
    ratio = 1

    while abs(ratio) > tolerance:
        try:
            C = stumpff_C(alpha*Chi**2)
            S = stumpff_S(alpha*Chi**2)

            f = r0_mag*v0_r / np.sqrt(mu) * Chi**2 * C + (1 - alpha*r0_mag) * Chi**3 * S + r0_mag * Chi - np.sqrt(mu) * dt
            df = r0_mag*v0_r / np.sqrt(mu) * Chi * (1 - alpha*Chi**2*S) + (1 - alpha*r0_mag) * Chi**2 * C + r0_mag

            ratio = f / df
            Chi -= ratio
        except Exception:
            return None

    return Chi