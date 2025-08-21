import numpy as np
from helper_constants import *

# ---------- Dynamics ----------
def accel_two_body_j2(r_eci, mu=MU_EARTH, Re=R_EARTH, j2=J2):
    x, y, z = r_eci
    r2 = x*x + y*y + z*z
    r  = np.sqrt(r2)

    # Two-body
    a_tb = -mu * r_eci / (r**3)

    # ---- Correct J2 (uses r^5 in the denominator) ----
    z2 = z*z
    r5 = r2 * r2 * r
    k  = 1.5 * j2 * mu * (Re**2) / r5
    f  = 5.0 * z2 / r2
    a_j2 = np.array([k * x * (f - 1.0),
                     k * y * (f - 1.0),
                     k * z * (f - 3.0)])

    return a_tb + a_j2

def rhs_j2(t, y):
    r = y[0:3]
    v = y[3:6]
    a = accel_two_body_j2(r)
    out = np.empty(6)
    out[0:3] = v
    out[3:6] = a
    return out