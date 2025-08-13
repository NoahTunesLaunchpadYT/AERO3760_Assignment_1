# propagators.py
from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp

# Expect these from your modules:
# - rhs_j2(t, y): 6D state derivative with J2
# - geodetic_to_ecef(lat_deg, lon_deg, h_m) -> (3,) km
# - gmst_from_jd(jd) -> radians
# - enu_matrix(lat_deg, lon_deg) -> 3x3 (rows E,N,U)
from j2 import rhs_j2
from utils import geodetic_to_ecef, gmst_from_jd, enu_matrix


# ----------------- Abstract Propagator -----------------
class Propagator(ABC):
    @abstractmethod
    def propagate(self, *args, **kwargs):
        """Return trajectory data; subclasses define the exact signature."""
        pass


# ----------------- Satellite Propagator (SciPy + J2) -----------------
class SatellitePropagator(Propagator):
    def __init__(self, method: str = "DOP853", rtol: float = 1e-8, atol: float = 1e-10, max_step: float = np.inf):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

    def propagate(self, sat, jd_array: list[float] | tuple[float, ...] | np.ndarray):
        """
        Run the satellite over a JD timebase.
        Converts JD -> elapsed seconds for the solver, then samples back at the same JD times.

        sat must have: sat.r0 (3,), sat.v0 (3,) in km / km/s
        Returns: JD (n,), R (n,3) km, V (n,3) km/s
        """
        JD = np.asarray(jd_array, float)
        t_samples = (JD - JD[0]) * 86400.0
        t0 = 0.0
        tf = float(t_samples[-1])

        r0, v0 = sat.current_state()
        y0 = np.hstack([r0, v0])

        try:
            sol = solve_ivp(rhs_j2, (t0, tf), y0,
                            method=self.method,
                            rtol=self.rtol, atol=self.atol,
                            max_step=self.max_step,
                            dense_output=True)
            if not sol.success:
                raise RuntimeError("Propagation failed: " + sol.message)
        except Exception as e:
            raise RuntimeError(f"Propagation error: {e}, y0: {y0}")

        Y = sol.sol(t_samples).T
        R = Y[:, 0:3]
        V = Y[:, 3:6]
        return JD, R, V

# ----------------- Ground Station Propagator (ECI + Up/ENU in ECI) -----------------
class GroundStationPropagator(Propagator):
    def propagate(self, gs, jd_array: list[float] | tuple[float, ...] | np.ndarray):
        """
        gs: data-only object with gs.lat_deg, gs.lon_deg, gs.h_m
        jd_array: absolute Julian Dates (days) at which to sample

        Returns:
          JD (n,), R_eci (n,3) km, V_eci (n,3) km/s (zeros),
          up_eci (n,3) unit,
          E_eci (n,3), N_eci (n,3), U_eci (n,3)  # ENU basis vectors expressed in ECI
        """
        JD = np.asarray(jd_array, float)

        # Station fixed frames
        r_ecef = geodetic_to_ecef(gs.lat_deg, gs.lon_deg, gs.h_m)   # (3,) km
        ENU = enu_matrix(gs.lat_deg, gs.lon_deg)                    # 3x3 rows E,N,U
        E_ecef, N_ecef, U_ecef = ENU[0], ENU[1], ENU[2]             # each (3,)

        # Earth rotation
        theta = gmst_from_jd(JD)                                    # (n,)
        ct, st = np.cos(theta), np.sin(theta)

        # Rotate ECEF -> ECI for position
        x =  ct * r_ecef[0] - st * r_ecef[1]
        y =  st * r_ecef[0] + ct * r_ecef[1]
        z =  np.full_like(ct, r_ecef[2], dtype=float)
        R_eci = np.vstack((x, y, z)).T                              # (n,3)

        # Rotate ENU basis rows to ECI (per time)
        def rot_row(row: np.ndarray) -> np.ndarray:
            rx =  ct * row[0] - st * row[1]
            ry =  st * row[0] + ct * row[1]
            rz =  np.full_like(ct, row[2], dtype=float)
            return np.vstack((rx, ry, rz)).T                        # (n,3)

        E_eci = rot_row(E_ecef)                                     # (n,3)
        N_eci = rot_row(N_ecef)                                     # (n,3)
        U_eci = rot_row(U_ecef)                                     # (n,3)

        # Up (normal) vector in ECI: same as U_eci, normalised (should already be unit)
        up_eci = U_eci / np.linalg.norm(U_eci, axis=1, keepdims=True)

        # Ground station "velocity" in this simple model = 0 (use if you need station ECI rates later)
        V_eci = np.zeros_like(R_eci)

        return JD, R_eci, V_eci, up_eci, E_eci, N_eci, U_eci
