# propagators.py
from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp

# Expect these from your modules:
# - rhs_j2(t, y): 6D state derivative with J2
# - geodetic_to_ecef(lat_deg, lon_deg, h_m) -> (3,) km
# - gmst_from_jd(jd) -> radians
# - enu_matrix(lat_deg, lon_deg) -> 3x3 (rows E,N,U)
from helper_j2 import rhs_j2
from helper_time import gmst_from_jd
from helper_coordinate_transforms import geodetic_to_eci, enu_matrix


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
        JD = np.asarray(jd_array, float)            # Array
        gmst_rad = gmst_from_jd(JD)                 # Array
        lst_rad_array = gmst_rad + np.radians(gs.lon_deg) # Array
        lat_rad_array = np.full_like(lst_rad_array, np.radians(gs.lat_deg))

        r_eci = geodetic_to_eci(lat_rad_array, lst_rad_array, gs.h_m)  # (3,n) km
        enu_rotation_matrix = enu_matrix(lat_rad_array, lst_rad_array) # Get the East-North-Up (ENU) basis vectors in the ECI frame   # 3x3 rows E,N,U
        east_basis_eci, north_basis_eci, up_basis_eci = enu_rotation_matrix[0], enu_rotation_matrix[1], enu_rotation_matrix[2]             # each (3,)                      # (n,3)

        # Up (normal) vector in ECI: same as U_eci, normalised (should already be unit)
        up_eci = up_basis_eci / np.linalg.norm(up_basis_eci, axis=1, keepdims=True)

        # TODO: Remove the above line

        # Ground station "velocity" in this simple model = 0 (use if you need station ECI rates later)
        v_eci = np.zeros_like(r_eci)

        return JD, r_eci, v_eci, up_eci, east_basis_eci, north_basis_eci, up_basis_eci
