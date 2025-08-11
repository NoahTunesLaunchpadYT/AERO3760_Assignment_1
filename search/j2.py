import numpy as np
from scipy.integrate import solve_ivp

# ---- Earth constants (km, s) ----
MU_EARTH = 398600.4418        # km^3/s^2
R_EARTH  = 6378.1363          # km
J2       = 1.08262668e-3

# ---------- Dynamics ----------
def accel_two_body_j2(r_eci, mu=MU_EARTH, Re=R_EARTH, j2=J2):
    x, y, z = r_eci
    r2 = x*x + y*y + z*z
    r  = np.sqrt(r2)

    # Two-body
    a_tb = -mu * r_eci / (r**3)

    # ---- Correct J2 (uses r^5 in the denominator) ----
    z2 = z*z
    r5 = r2 * r2 * r              # <-- correct r^5
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

# # ---------- Propagation wrapper ----------
# def propagate_ivp_j2(r0_km, v0_km_s, t0_s, tf_s,
#                      rtol=1e-9, atol=1e-9, method="DOP853",
#                      max_step=np.inf, dense=False, sample_dt=None):
#     """
#     Adaptive propagation with SciPy solve_ivp (ECI, J2).
#       - r0_km, v0_km_s: length-3 arrays
#       - [t0_s, tf_s]: time span in seconds
#       - rtol/atol: integrator tolerances (tight enough for multi-day J2)
#       - method: "DOP853" (great default), or "RK45"
#       - max_step: cap step size if you want (e.g., 120.0)
#       - dense: if True, keeps a dense_output function for later sampling
#       - sample_dt: if provided, returns uniform samples every sample_dt seconds

#     Returns:
#       T (n,), R (n,3), V (n,3)
#     """
#     y0 = np.hstack([np.array(r0_km, float), np.array(v0_km_s, float)])

#     sol = solve_ivp(rhs_j2,
#                     (t0_s, tf_s),
#                     y0,
#                     method=method,
#                     rtol=rtol,
#                     atol=atol,
#                     max_step=max_step,
#                     dense_output=dense or (sample_dt is not None))

#     if not sol.success:
#         raise RuntimeError("Propagation failed: " + sol.message)

#     if sample_dt is None:
#         # Use solver output times (non-uniform)
#         T = sol.t
#         Y = sol.y.T
#     else:
#         # Uniform sampling via dense_output
#         tout = np.arange(t0_s, tf_s + 1e-9, sample_dt)
#         Y = sol.sol(tout).T
#         T = tout

#     R = Y[:, 0:3]
#     V = Y[:, 3:6]
#     return T, R, V

# def main():
#     # r0, v0 from your Satellite.current_state()
#     # r0, v0 = sat.current_state()
#     # Example:
#     r0 = np.array([7000.0, 0.0, 0.0])
#     v0 = np.array([0.0, 7.5, 1.0])

#     t0 = 0.0
#     tf = 10*24*3600.0  # 10 days
#     T, R, V = propagate_ivp_j2(
#         r0, v0, t0, tf,
#         method="DOP853",         # or "RK45"
#         rtol=1e-8, atol=1e-10,   # slightly looser
#         max_step=300.0,          # or leave as np.inf and just sample with sample_dt
#         sample_dt=60.0
#     )
#     print("Done")
    
# if __name__ == "__main__":
#     main()