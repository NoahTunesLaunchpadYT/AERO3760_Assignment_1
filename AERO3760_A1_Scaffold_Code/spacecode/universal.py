import numpy as np


def stumpff_S(z: float) -> float:
    """Calculates the Stumpff function S(z)

    Args:
        z (float)

    Returns:
        float
    """
    # TODO: complete the cases for the S(z) stumpff function
    if z > 0:
        return ...
    elif z < 0:
        return ...
    else:
        return ...


def stumpff_C(z: float) -> float:
    """Calculates the Stumpff function C(z)

    Args:
        z (float)

    Returns:
        float
    """
    # TODO: complete the cases for the C(z) stumpff function
    if z > 0:
        return ...
    elif z < 0:
        return ...
    else:
        return ...
    

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

    # TODO: Equation (5.26) Hint: be careful of radians
    if (is_prograde and norm_vec[2] >= 0) or (not is_prograde and norm_vec[2] < 0):
        delta_theta = ...
    else:
        delta_theta = ...
    

    # -------------------- Compute the Universal Anomaly
    A = ...     # TODO: Equation (5.35)

    # Note: The following three functions are called inner (or nested) functions
    def y(z: float) -> float:
        Sz = stumpff_S(z)
        Cz = stumpff_C(z)

        return ...  # TODO: Equation (5.38)

    def F(z: float) -> float:
        # Hint: The variables `A` and `dt` are already defined
        Sz = stumpff_S(z)
        Cz = stumpff_C(z)
        y = y(z)

        return ...  # TODO: Equation (5.40)
    
    def dFdz(z: float) -> float:
        # Derivative of F(z) w.r.t to z
        # Hint: The variables `A` and `dt` are already defined
        Sz = stumpff_S(z)
        Cz = stumpff_C(z)
        y = y(z)

        # TODO: Equation (5.43)
        if z == 0:
            return ...
        else:
            return ...
    

    # TODO: Perform Newton's method to find z
    z = 0               # Initial guess for the universal variable
    tolerance = 1e-6
    update = 1

    while abs(update) > tolerance:
        update = ...    # TODO
        z = z - update
    
    return z