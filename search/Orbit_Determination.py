import numpy as np
from Trajectory import SatelliteTrajectory, GroundStationTrajectory
from helper.coordinate_transforms import az_el_sat_from_gs, ENU_from_az_el_rng, enu_matrix, geodetic_to_eci, u_from_az_el
from helper.time import gmst_from_jd
from helper.constants import MU_EARTH as mu
from helper.constants import R_EARTH
from helper.lagrange import lagrange_coefficients_universal, universal_variable_from_r
from Satellite import Satellite
from dataclasses import dataclass

from matplotlib import pyplot as plt

class Observation:
    def __init__(self, jd: float, kind: str, values: np.ndarray):
        self.jd = jd
        self.kind = kind
        self.values = values

class Sensor:
    def __init__(self, id: str, range_error: float, angular_error: float, time_error: float):
        self.id = id
        self.range_error = range_error
        self.angular_error = angular_error
        self.time_error = time_error

    def observe(self):
        pass
    
class SatelliteLaserRanging(Sensor):
    def __init__(self, id: str, range_error: float, angular_error: float, time_error: float):
        super().__init__(id, range_error, angular_error, time_error)

    def observe(self, sat_trajectory: SatelliteTrajectory, gs_trajectory: GroundStationTrajectory, jd_times: tuple):
        if len(jd_times) != 3:
            raise ValueError("Three Julian dates are required.")

        observations = []

        az_arr, el_arr, vis_arr, rng_km_arr = az_el_sat_from_gs(gs_trajectory, sat_trajectory, return_distance=True)
        
        for jd_time in jd_times:
            az = np.interp(jd_time, sat_trajectory.JD, az_arr)
            el = np.interp(jd_time, sat_trajectory.JD, el_arr)
            rng_km = np.interp(jd_time, sat_trajectory.JD, rng_km_arr)
            print(f"[SatelliteLaserRanging] Range (km): {rng_km}")

            obs = Observation(jd=jd_time, kind="az_el_rng", values=np.array([az, el, rng_km]))
            observations.append(obs)

        return observations
    
class RadiometricTracking(Sensor):
    def __init__(self, id: str, range_error: float, angular_error: float, time_error: float):
        super().__init__(id, range_error, angular_error, time_error)

    def observe(self, sat_trajectory: SatelliteTrajectory, gs_trajectory: GroundStationTrajectory, jd_times: tuple):
        if len(jd_times) != 2:
            raise ValueError("Two Julian dates are required.")

        observations = []

        for jd_time in jd_times:
            az_arr, el_arr, vis_arr, rng_km_arr = az_el_sat_from_gs(gs_trajectory, sat_trajectory, return_distance=True)
            az = np.interp(jd_time, sat_trajectory.JD, az_arr)
            el = np.interp(jd_time, sat_trajectory.JD, el_arr)
            rng_km = np.interp(jd_time, sat_trajectory.JD, rng_km_arr)

            obs = Observation(jd=jd_time, kind="az_el_rng_time", values=np.array([az, el, rng_km, jd_time]))
            observations.append(obs)

        return observations

class OpticalTracking(Sensor):
    def __init__(self, id: str, range_error: float, angular_error: float, time_error: float):
        super().__init__(id, range_error, angular_error, time_error)

    def observe(self, sat_trajectory: SatelliteTrajectory, gs_trajectory: GroundStationTrajectory, jd_times: tuple):
        if len(jd_times) != 3:
            raise ValueError("Three Julian dates are required.")

        observations = []

        for jd_time in jd_times:
            az_arr, el_arr, vis_arr = az_el_sat_from_gs(gs_trajectory, sat_trajectory)
            az = np.interp(jd_time, sat_trajectory.JD, az_arr)
            el = np.interp(jd_time, sat_trajectory.JD, el_arr)

            obs = Observation(jd=jd_time, kind="az_el", values=np.array([az, el]))
            observations.append(obs)

        return observations

@dataclass
class GibbsObservables:
    def __init__(self,
                 r1_eci_km, r2_eci_km, r3_eci_km,
                 t1_jd: float, t2_jd: float, t3_jd: float):
        self.r1_eci_km = r1_eci_km
        self.r2_eci_km = r2_eci_km
        self.r3_eci_km = r3_eci_km
        self.t1_jd = float(t1_jd)
        self.t2_jd = float(t2_jd)
        self.t3_jd = float(t3_jd)

    def plot_vectors_3d(self, R_EARTH: float = 6378.1363, *, ax=None, show=True, block=True):
        """
        Plot r1, r2, r3 as 3D vectors from the origin (ECI), label each tip with its JD,
        and draw the Earth as a sphere of radius R_EARTH (km).

        Parameters
        ----------
        R_EARTH : float
            Earth radius in km.
        ax : matplotlib 3D axes (optional)
            If provided, draw on it; otherwise create a new figure/axes.
        show : bool
            If True, call plt.show() at the end.

        Returns
        -------
        fig, ax : the figure and 3D axes used.
        """
        # Prepare axes
        created_ax = False
        if ax is None:
            fig = plt.figure(figsize=(7.5, 7))
            ax = fig.add_subplot(111, projection="3d")
            created_ax = True
        else:
            fig = ax.figure

        # Draw Earth as a sphere
        u = np.linspace(0, 2*np.pi, 80)
        v = np.linspace(0, np.pi, 40)
        x = R_EARTH * np.outer(np.cos(u), np.sin(v))
        y = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, rstride=2, cstride=2, color="C0", alpha=0.15, linewidth=0)

        # Plot the three vectors
        vecs = [
            (self.r1_eci_km, f"r1 @ {self.t1_jd:.6f} JD", "C1"),
            (self.r2_eci_km, f"r2 @ {self.t2_jd:.6f} JD", "C2"),
            (self.r3_eci_km, f"r3 @ {self.t3_jd:.6f} JD", "C3"),
        ]
        lim = R_EARTH
        for r, label, color in vecs:
            ax.plot([0, r[0]], [0, r[1]], [0, r[2]], color=color, lw=2, label=label)
            ax.scatter([r[0]], [r[1]], [r[2]], color=color, s=30)
            lim = max(lim, np.linalg.norm(r))

        # Labels and legend
        ax.set_xlabel("ECI X [km]")
        ax.set_ylabel("ECI Y [km]")
        ax.set_zlabel("ECI Z [km]")
        ax.set_title("Gibbs Observables — ECI Position Vectors")

        # Set symmetric limits and roughly equal aspect
        pad = 0.1 * lim
        ax.set_xlim([-lim - pad, lim + pad])
        ax.set_ylim([-lim - pad, lim + pad])
        ax.set_zlim([-lim - pad, lim + pad])

        ax.legend(loc="upper left", fontsize=9)

        if show and created_ax:
            plt.show(block=block)
        return fig, ax

@dataclass
class LambertObservables:
    r1_eci_km: np.ndarray
    r2_eci_km: np.ndarray
    dt_seconds: float
    t1_jd: float
    t2_jd: float

@dataclass
class GaussObservables:
    u1_eci: np.ndarray
    u2_eci: np.ndarray
    u3_eci: np.ndarray
    r1_gs_eci_km: np.ndarray
    r2_gs_eci_km: np.ndarray
    r3_gs_eci_km: np.ndarray
    dt12_s: float
    dt32_s: float
    t1_jd: float
    t2_jd: float
    t3_jd: float

class ObservationReducer:
    def for_gibbs(self, gs, observables: list[Observation]) -> GibbsObservables:
        obs3 = sorted([o for o in observables if o.kind == "az_el_rng"], key=lambda o:o.jd)[:3]
        if len(obs3) < 3:
            raise ValueError("Gibbs requires 3 az/el/range observations")

        r_list, t_list = [], []
        for o in obs3:
            jd = o.jd
            gmst_rad = gmst_from_jd(jd)
            lst_deg = np.degrees(gmst_rad) + gs.lon_deg
            lat_deg = gs.lat_deg
            r_gs_eci = geodetic_to_eci(lat_deg, lst_deg, gs.h_m)  # (3,n) km
            R_ENU_ECI = enu_matrix(lat_deg, lst_deg).T   # 3×3
            az_deg, el_deg, rng_km = o.values
            print(f"[ObservationReducer] jd: {jd}, gmst_rad: {gmst_rad}, lst_deg: {lst_deg}, lat_deg: {lat_deg}, R_ENU_ECI: {R_ENU_ECI}, az_deg: {az_deg}, el_deg: {el_deg}, rng_km: {rng_km}")
            q_ENU = ENU_from_az_el_rng(az_deg, el_deg, rng_km)
            print(f"[ObservationReducer] r_ENU: {q_ENU}")
            q_ECI = R_ENU_ECI @ q_ENU
            r_eci = r_gs_eci + q_ECI
            print(f"[ObservationReducer] r_eci: {r_eci}")
            r_list.append(r_eci)
            t_list.append(jd)

            print(f"[ObservationReducer] rng_km: {rng_km}, r_ENU: {q_ENU}, r_list: {r_list}")
        return GibbsObservables(r1_eci_km=r_list[0], r2_eci_km=r_list[1], r3_eci_km=r_list[2],
                                t1_jd=t_list[0], t2_jd=t_list[1], t3_jd=t_list[2])

    def for_lambert(self, gs, observables: list[Observation]) -> LambertObservables:
        print("WWEEEE")
        print([o.kind for o in observables])
        obs2 = sorted([o for o in observables if o.kind == "az_el_rng_time"], key=lambda o:o.jd)[:2]

        if len(obs2) < 2:
            raise ValueError("Lambert requires 2 az/el/range observations")
        
        r_list, t_list = [], []
        dt_s = (obs2[1].jd - obs2[0].jd) * 86400.0

        for o in obs2:
            jd = o.jd
            gmst_rad = gmst_from_jd(jd)
            lst_deg = np.degrees(gmst_rad) + gs.lon_deg
            lat_deg = gs.lat_deg
            R_ENU_ECI = enu_matrix(lat_deg, lst_deg)   # 3×3
            az_deg, el_deg, rng_km, _ = o.values
            r_ENU = ENU_from_az_el_rng(az_deg, el_deg, rng_km)
            r_list.append(R_ENU_ECI @ r_ENU)
            t_list.append(jd)

        return LambertObservables(r_list[0], r_list[1], dt_s, obs2[0].jd, obs2[1].jd)

    def for_gauss(self, gs, observables: list[Observation]) -> GaussObservables:
        obs3 = sorted([o for o in observables if o.kind == "az_el"], key=lambda o:o.jd)[:3]
        if len(obs3) < 3:
            raise ValueError("Gauss requires 3 az/el observations")

        r_gs_list, u_list, t_list = [], [], []

        for o in obs3:
            jd = o.jd
            gmst_rad = gmst_from_jd(jd)
            lst_deg = np.degrees(gmst_rad) + gs.lon_deg
            lat_deg = gs.lat_deg
            r_gs_eci = geodetic_to_eci(lat_deg, lst_deg, gs.h_m)  # (3,n) km
            R_ENU_ECI = enu_matrix(lat_deg, lst_deg).T   # 3×3
            az_deg, el_deg = o.values
            u_ENU = u_from_az_el(az_deg, el_deg)  # Unit vector
            r_gs_list.append(r_gs_eci)
            u_list.append(R_ENU_ECI @ u_ENU)
            t_list.append(jd)

        print(f"[ObservationReducer] t_list: {t_list}")

        dt12 = (t_list[0] - t_list[1]) * 86400.0
        dt32 = (t_list[2] - t_list[1]) * 86400.0

        print(f"[ObservationReducer] dt12: {dt12}, dt32: {dt32}")

        return GaussObservables(u_list[0], u_list[1], u_list[2], r_gs_list[0], r_gs_list[1], r_gs_list[2], dt12, dt32, t_list[0], t_list[1], t_list[2])

class OrbitDeterminationSolver:
    def __init__(self):
        self.reducer = ObservationReducer()

    def determine_orbit(self, gs, observables: list[Observation], method: str = "Gibbs"):
        if method == "Gibbs":
            observables = self.reducer.for_gibbs(gs, observables)
            observables.plot_vectors_3d(block=False)
            r_km, r_km_s = self.gibbs(observables)
        elif method == "Lambert":
            observables = self.reducer.for_lambert(gs, observables)
            r_km, r_km_s = self.lambert(observables)
        elif method == "Gauss":
            observables = self.reducer.for_gauss(gs, observables)
            r_km, r_km_s = self.gauss(observables)
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"[OrbitDeterminationSolver]: Building satellite with r_km: {r_km}, r_km_s: {r_km_s}")
        sat = Satellite((r_km, r_km_s), start_jd=observables.t2_jd)

        return sat

    def gibbs(self, observables: GibbsObservables) -> tuple[np.ndarray, np.ndarray]:
        '''
            Implements the Gibbs method for orbit determination.

            arguments:
                observables (GibbsObservables): The observables containing position and time information.

            returns:
                tuple[np.ndarray, np.ndarray]: The position and velocity vectors at t2.
        '''
        r1 = observables.r1_eci_km
        r2 = observables.r2_eci_km
        r3 = observables.r3_eci_km

        print(f"r1: {r1}, r2: {r2}, r3: {r3}")

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

        v2 = np.sqrt(mu / (N_mag*D_mag)) * ((np.cross(D, r2)) / r2_mag + S)

        return r2, v2

    def lambert(self, observables: LambertObservables) -> tuple[np.ndarray, np.ndarray]:
        """Implements the Lambert method for orbit determination.

        Args:
            observables (LambertObservables): The observables containing position and time information.

        Returns:
            tuple[np.ndarray, np.ndarray]: The position and velocity vectors at t2.
        """
        r1 = observables.r1_eci_km
        r2 = observables.r2_eci_km
        dt = observables.dt_seconds

        is_prograde = np.dot(np.cross(r1, r2), r2) > 0

        z = universal_variable_from_r(r1, r2, dt, mu, is_prograde)
        f, g, fdot, gdot = lagrange_coefficients_universal(r1, r2, mu, z, is_prograde)

        v1 = 1 / g * (r2 - f * r1)
        v2 = 1 / g * (gdot * r2 - r1)

        return r2, v2

    def gauss(self, observables: GaussObservables) -> tuple[np.ndarray, np.ndarray]:
        """Implements the Gauss method for orbit determination.

        Args:
            observables (GaussObservables): The observables containing position and time information.

        Returns:
            tuple[np.ndarray, np.ndarray]: The position and velocity vectors at t2.
        """
        q1, q2, q3 = observables.u1_eci, observables.u2_eci, observables.u3_eci
        R1, R2, R3 = observables.r1_gs_eci_km, observables.r2_gs_eci_km, observables.r3_gs_eci_km

        # Time intervals between observations
        tau_1 = observables.dt12_s
        tau_3 = observables.dt32_s
        tau = tau_3 - tau_1         # Equation (5.101)

        print(f"[OrbitDeterminationSolver] tau_1: {tau_1}, tau_3: {tau_3}")
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
        E = np.dot(R2, q2)

        print(f"[OrbitDeterminationSolver] A: {A}, B: {B}, E: {E}")

        a = -(A**2 + 2*A*E + R2_mag2)
        b = -2*mu*B*(A + E)
        c = -mu**2 * B**2
        
        r2_mag = 20000 # km    # Select a suitable initial guess (hint: it is not 0)
        tolerance = 1e-6
        update = 1

        # Print the shape of each variable
        print(f"[OrbitDeterminationSolver] r2_mag shape: {r2_mag}")
        print(f"[OrbitDeterminationSolver] a shape: {a}")
        print(f"[OrbitDeterminationSolver] b shape: {b}")
        print(f"[OrbitDeterminationSolver] c shape: {c}")

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

        v2 = 1 / (f_1 * g_3 - f_3 * g_1) * (-f_3 * r1 + f_1 * r3)   # Equation (5.114)

        return r2, v2
