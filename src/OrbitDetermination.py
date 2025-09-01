import numpy as np
from Trajectory import SatelliteTrajectory, GroundStationTrajectory
from helper.coordinate_transforms import az_el_sat_from_gs, ENU_from_az_el_rng, enu_matrix, geodetic_to_eci, u_from_az_el
from helper.time import gmst_from_jd
from helper.constants import MU_EARTH as mu
from helper.constants import R_EARTH
from helper.plotting import set_axis_limits_from_points
from helper.lagrange import lagrange_coefficients_universal, universal_variable_from_r, stumpff_C, stumpff_S, universal_variable_from_dt_r_v_alpha
from Satellite import Satellite
from dataclasses import dataclass


from matplotlib import pyplot as plt

class Observation:
    """Represents a satellite observation.
    """    
    def __init__(self, jd: float, kind: str, values: np.ndarray):
        self.jd = jd
        self.kind = kind
        self.values = values

class Sensor:
    """Represents a sensor for satellite observation.
    """    
    def __init__(self, id: str, range_error: float = 0, angular_error: float = 0, time_error: float = 0):
        self.id = id
        self.range_error = range_error
        self.angular_error = angular_error
        self.time_error = time_error

    def observe(self):
        pass
    
class SatelliteLaserRanging(Sensor):
    """Represents a laser ranging sensor for satellite observation.
    """    
    def __init__(self, id: str, range_error: float = 0, angular_error: float = 0, time_error: float = 0):
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

            # Apply Error
            az_error = abs(360*self.angular_error)
            el_error = abs(360*self.angular_error)
            rng_error = abs(rng_km*self.range_error)

            d_az = np.random.normal(0, az_error)
            d_el = np.random.normal(0, el_error)
            d_rng = np.random.normal(0, rng_error)

            az += d_az
            el += d_el
            rng_km += d_rng

            obs = Observation(jd=jd_time, kind="az_el_rng", values=np.array([az, el, rng_km]))
            observations.append(obs)

        return observations
    
class RadiometricTracking(Sensor):
    """Represents a radiometric tracking sensor for satellite observation.
    """
    def __init__(self, id: str, range_error: float = 0, angular_error: float = 0, time_error: float = 0):
        super().__init__(id, range_error, angular_error, time_error)

    def observe(self, sat_trajectory: SatelliteTrajectory, gs_trajectory: GroundStationTrajectory, jd_times: tuple):
        if len(jd_times) != 2:
            raise ValueError("Two Julian dates are required.")

        observations = []
        previous_jd_time = jd_times[0]

        for jd_time in jd_times:
            az_arr, el_arr, vis_arr, rng_km_arr = az_el_sat_from_gs(gs_trajectory, sat_trajectory, return_distance=True)
            az = np.interp(jd_time, sat_trajectory.JD, az_arr)
            el = np.interp(jd_time, sat_trajectory.JD, el_arr)
            rng_km = np.interp(jd_time, sat_trajectory.JD, rng_km_arr)

            # Apply Error
            az_error = abs(360*self.angular_error)
            el_error = abs(360*self.angular_error)
            rng_error = abs(rng_km*self.range_error)
            jd_error = abs((jd_time - previous_jd_time)*self.time_error)
            previous_jd_time = jd_time

            d_az = np.random.normal(0, az_error)
            d_el = np.random.normal(0, el_error)
            d_rng = np.random.normal(0, rng_error)
            d_jd = np.random.normal(0, jd_error)

            az += d_az
            el += d_el
            rng_km += d_rng
            jd_time += d_jd

            obs = Observation(jd=jd_time, kind="az_el_rng_time", values=np.array([az, el, rng_km, jd_time]))
            observations.append(obs)

        return observations

class OpticalTracking(Sensor):
    """Represents an optical tracking sensor for satellite observation.
    """
    def __init__(self, id: str, range_error: float = 0, angular_error: float = 0, time_error: float = 0):
        super().__init__(id, range_error, angular_error, time_error)

    def observe(self, sat_trajectory: SatelliteTrajectory, gs_trajectory: GroundStationTrajectory, jd_times: tuple):
        if len(jd_times) != 3:
            raise ValueError("Three Julian dates are required.")

        observations = []

        previous_jd_time = jd_times[0]

        for jd_time in jd_times:
            az_arr, el_arr, vis_arr = az_el_sat_from_gs(gs_trajectory, sat_trajectory, return_distance=False)
            az = np.interp(jd_time, sat_trajectory.JD, az_arr)
            el = np.interp(jd_time, sat_trajectory.JD, el_arr)

            # Apply Error
            az_error = abs(360*self.angular_error)
            el_error = abs(360*self.angular_error)
            jd_error = abs((jd_time - previous_jd_time)*self.time_error)
            previous_jd_time = jd_time

            d_az = np.random.normal(0, az_error)
            d_el = np.random.normal(0, el_error)
            d_jd = np.random.normal(0, jd_error)

            az += d_az
            el += d_el
            jd_time += d_jd

            obs = Observation(jd=jd_time, kind="az_el_time", values=np.array([az, el, jd_time]))
            observations.append(obs)

        return observations

@dataclass
class GibbsObservables:
    """Represents the Gibbs observables for satellite orbit determination.
    """
    def __init__(self,
                 r1_eci_km, r2_eci_km, r3_eci_km,
                 t1_jd: float, t2_jd: float, t3_jd: float):
        self.r1_eci_km = r1_eci_km
        self.r2_eci_km = r2_eci_km
        self.r3_eci_km = r3_eci_km
        self.t1_jd = float(t1_jd)
        self.t2_jd = float(t2_jd)
        self.t3_jd = float(t3_jd)

    def plot_vectors_3d(self, R_EARTH: float = 6378.1363, trajectory=None, *, ax=None, show=True, block=True):
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

        if trajectory is not None and hasattr(trajectory, "R"):
            R_sat = np.asarray(trajectory.R)
            if R_sat.ndim == 2 and R_sat.shape[0] == 3 and R_sat.shape[1] >= 1:
                R_sat = R_sat.T  # make it Nx3
            if R_sat.ndim == 2 and R_sat.shape[1] == 3:
                # Path style
                k = dict(color="k", lw=1.3, alpha=0.85, label="Satellite path")
                ax.plot(R_sat[:, 0], R_sat[:, 1], R_sat[:, 2], **k)
            set_axis_limits_from_points(ax, [R_sat])
        else:
            set_axis_limits_from_points(ax, [self.r1_eci_km, self.r2_eci_km])


        # Labels and legend
        ax.set_xlabel("ECI X [km]")
        ax.set_ylabel("ECI Y [km]")
        ax.set_zlabel("ECI Z [km]")
        ax.set_title("Gibbs Observables — ECI Position Vectors")

        ax.legend(loc="upper left", fontsize=9)

        if show and created_ax:
            plt.show(block=block)
        return fig, ax

@dataclass
class LambertObservables:
    """Represents the Lambert observables for satellite orbit determination.
    """    
    r1_eci_km: np.ndarray
    r2_eci_km: np.ndarray
    dt_seconds: float
    t1_jd: float
    t2_jd: float

    def plot_vectors_3d(self, R_EARTH: float = 6378.1363, trajectory=None, *, ax=None, show=True, block=True):
        """
        Plot r1 and r2 as 3D vectors from ECI origin, label the tips with their JDs,
        optionally draw the transfer trajectory, and render the Earth as a sphere.

        Parameters
        ----------
        R_EARTH : float
            Earth radius in km.
        trajectory : object with attribute `R` (optional)
            If provided, expects positions in km shaped (N,3) or (3,N).
        ax : matplotlib 3D axes (optional)
            If provided, draw on it; otherwise create a new figure/axes.
        show : bool
            If True, call plt.show() at the end.
        block : bool
            Passed through to plt.show(block=...).

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

        # Plot r1 and r2 vectors
        pairs = [
            (self.r1_eci_km, f"r1 @ {self.t1_jd:.6f} JD", "C1"),
            (self.r2_eci_km, f"r2 @ {self.t2_jd:.6f} JD", "C2"),
        ]
        for r, label, color in pairs:
            ax.plot([0, r[0]], [0, r[1]], [0, r[2]], color=color, lw=2, label=label)
            ax.scatter([r[0]], [r[1]], [r[2]], color=color, s=30)

        # Optional trajectory path
        if trajectory is not None and hasattr(trajectory, "R"):
            R_sat = np.asarray(trajectory.R)
            if R_sat.ndim == 2 and R_sat.shape[0] == 3 and R_sat.shape[1] >= 1:
                R_sat = R_sat.T  # make it Nx3
            if R_sat.ndim == 2 and R_sat.shape[1] == 3:
                ax.plot(R_sat[:, 0], R_sat[:, 1], R_sat[:, 2],
                        color="k", lw=1.3, alpha=0.85, label="Transfer path")
            set_axis_limits_from_points(ax, [R_sat])
        else:
            set_axis_limits_from_points(ax, [self.r1_eci_km, self.r2_eci_km])


        # Labels, limits, legend
        ax.set_xlabel("ECI X [km]")
        ax.set_ylabel("ECI Y [km]")
        ax.set_zlabel("ECI Z [km]")
        minutes = self.dt_seconds / 60.0
        ax.set_title(f"Lambert Observables — ECI Vectors (Δt ≈ {minutes:.2f} min)")


        ax.legend(loc="upper left", fontsize=9)

        if show and created_ax:
            plt.show(block=block)
        return fig, ax

class GaussObservables:
    """Represents the Gauss observables for satellite orbit determination.
    """
    def __init__(self, q1_eci: np.ndarray, q2_eci: np.ndarray, q3_eci: np.ndarray, r1_gs_eci_km: np.ndarray, r2_gs_eci_km: np.ndarray, r3_gs_eci_km: np.ndarray, dt12_s: float, dt32_s: float, t1_jd: float, t2_jd: float, t3_jd: float):
        self.q1_eci = q1_eci
        self.q2_eci = q2_eci
        self.q3_eci = q3_eci
        self.r1_gs_eci_km = r1_gs_eci_km
        self.r2_gs_eci_km = r2_gs_eci_km
        self.r3_gs_eci_km = r3_gs_eci_km
        self.dt12_s = dt12_s
        self.dt32_s = dt32_s
        self.t1_jd = t1_jd
        self.t2_jd = t2_jd
        self.t3_jd = t3_jd
    
    def plot_vectors_3d(self,
                        R_EARTH: float = R_EARTH,
                        los_length_km: float = 12000.0,
                        trajectory=None,
                        path_kwargs: dict | None = None,
                        *, show: bool = True, block: bool = True):
        """
        Visualise Gauss observables.
        Left: Earth, GS position vectors (ECI), and line-of-sight rays from each GS tip. (Optionally the satellite path.)
        Right: Direction-only unit vectors u1,u2,u3 drawn from the origin.

        Parameters
        ----------
        R_EARTH : float
            Earth radius in km for the sphere.
        los_length_km : float
            Length used to draw the LOS rays (for visibility only).
        trajectory : object, optional
            Any object with attribute .R (array-like Nx3 or 3xN, km). If it also has .JD, points at t1,t2,t3 are marked.
        path_kwargs : dict, optional
            Matplotlib kwargs for the satellite path line (e.g., {'color':'k','lw':1,'alpha':0.9}).
        show : bool
            If True, call plt.show() at the end.
        block : bool
            If True and show=True, block on plt.show().
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # --- Unpack (typo fix: use q1_eci/q2_eci/q3_eci) ---
        u1, u2, u3 = self.q1_eci, self.q2_eci, self.q3_eci
        R1, R2, R3 = self.r1_gs_eci_km, self.r2_gs_eci_km, self.r3_gs_eci_km

        # Ensure unit vectors are normalised (defensive)
        def _unit(v):
            n = np.linalg.norm(v)
            return v / n if n != 0 else v

        u1, u2, u3 = _unit(u1), _unit(u2), _unit(u3)

        # Precompute LOS endpoints
        L1 = R1 + los_length_km * u1
        L2 = R2 + los_length_km * u2
        L3 = R3 + los_length_km * u3

        # Figure & subplots
        fig = plt.figure(figsize=(13.5, 6.5))
        ax_geo = fig.add_subplot(1, 2, 1, projection="3d")
        ax_dir = fig.add_subplot(1, 2, 2, projection="3d")

        # ---------- Left subplot: Earth + GS vectors + LOS rays ----------
        # Earth
        u = np.linspace(0, 2*np.pi, 80)
        v = np.linspace(0, np.pi, 40)
        x = R_EARTH * np.outer(np.cos(u), np.sin(v))
        y = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
        ax_geo.plot_surface(x, y, z, rstride=2, cstride=2, color="C0", alpha=0.15, linewidth=0)

        # Colours/labels for the three epochs
        items = [
            ("C1", R1, L1, f"t1 = {self.t1_jd:.6f} JD"),
            ("C2", R2, L2, f"t2 = {self.t2_jd:.6f} JD"),
            ("C3", R3, L3, f"t3 = {self.t3_jd:.6f} JD"),
        ]

        # Draw GS position vectors and LOS rays
        for ci, Rg, Lg, label in items:
            ax_geo.plot([0, Rg[0]], [0, Rg[1]], [0, Rg[2]], color=ci, lw=2)
            ax_geo.scatter([Rg[0]], [Rg[1]], [Rg[2]], color=ci, s=22, depthshade=True)

            ax_geo.plot([Rg[0], Lg[0]], [Rg[1], Lg[1]], [Rg[2], Lg[2]],
                        color=ci, lw=2, ls="--", label=label)
            ax_geo.scatter([Lg[0]], [Lg[1]], [Lg[2]], color=ci, s=18, marker="^", depthshade=True)

        # --- OPTIONAL: overlay satellite path from trajectory.R ---
        pts = [R1, R2, R3, L1, L2, L3, np.array([0, 0, 0])]
        if trajectory is not None and hasattr(trajectory, "R"):
            R_sat = np.asarray(trajectory.R)
            if R_sat.ndim == 2 and R_sat.shape[0] == 3 and R_sat.shape[1] >= 1:
                R_sat = R_sat.T  # make it Nx3
            if R_sat.ndim == 2 and R_sat.shape[1] == 3:
                # Path style
                k = dict(color="k", lw=1.3, alpha=0.85, label="Satellite path")
                if path_kwargs:
                    k.update(path_kwargs)

                ax_geo.plot(R_sat[:, 0], R_sat[:, 1], R_sat[:, 2], **k)

                # Include sat points in limits
                pts.append(R_sat)

                # If JD available, mark closest samples to t1,t2,t3
                if hasattr(trajectory, "JD"):
                    JD = np.asarray(trajectory.JD).ravel()
                    def _idx_for(tjd):
                        return int(np.argmin(np.abs(JD - tjd)))
                    i1, i2, i3 = _idx_for(self.t1_jd), _idx_for(self.t2_jd), _idx_for(self.t3_jd)
                    marks = [("C1", i1, "o"), ("C2", i2, "s"), ("C3", i3, "^")]
                    for ci, ii, m in marks:
                        ax_geo.scatter([R_sat[ii, 0]], [R_sat[ii, 1]], [R_sat[ii, 2]],
                                    color=ci, s=36, marker=m, edgecolor="k", linewidths=0.6,
                                    label=None)
                        
        # Axes styling
        ax_geo.set_xlabel("ECI X [km]")
        ax_geo.set_ylabel("ECI Y [km]")
        ax_geo.set_zlabel("ECI Z [km]")
        ax_geo.set_title("Gauss Observables")

        # Limits based on all drawn points (and ensure Earth-centred)
        try:
            set_axis_limits_from_points(ax_geo, pts)
        except Exception:
            P = np.vstack([p if p.ndim == 2 else np.atleast_2d(p) for p in pts])
            lim = float(np.max(np.linalg.norm(P, axis=1)))
            ax_geo.set_xlim(-lim, lim); ax_geo.set_ylim(-lim, lim); ax_geo.set_zlim(-0.4*lim, 0.4*lim)

        ax_geo.legend(loc="upper right", fontsize=9)

        shadow_kw = dict(color="0.3", lw=1.2, ls="--", alpha=0.5)

        def _shadow_line(ax, p0, p1, **kw):
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [-10000, -10000], **kw)

        # 0) A light ground plane at z=0 (after limits are known)
        xlim, ylim = ax_geo.get_xlim(), ax_geo.get_ylim()
        xx, yy = np.meshgrid(np.linspace(*xlim, 2), np.linspace(*ylim, 2))
        zz = np.ones_like(xx)*-10000
        ax_geo.plot_surface(xx, yy, zz, color="k", alpha=0.05, rstride=1, cstride=1, linewidth=0)

        # 2) Shadows for GS position vectors (from origin to each GS)
        for ci, Rg, Lg, label in items:
            _shadow_line(ax_geo, np.zeros(3), np.array([Rg[0], Rg[1], -10000]), **shadow_kw)

        # 3) Shadows for LOS rays (from GS to LOS tip)
        for ci, Rg, Lg, label in items:
            _shadow_line(ax_geo, np.array([Rg[0], Rg[1], -10000]), np.array([Lg[0], Lg[1], -10000]), **shadow_kw)

        # 4) Optional: shadow for the satellite path
        if trajectory is not None and 'R_sat' in locals():
            ax_geo.plot(R_sat[:, 0], R_sat[:, 1], np.ones_like(R_sat[:, 0])*-10000,
                        color="0.2", lw=1.0, alpha=0.35, label=None)

        # 5) Optional: shadows for unit vectors on the right subplot
        shadow_kw_dir = dict(color="0.4", lw=1.2, ls="--", alpha=0.6)
        for uu in (u1, u2, u3):
            ax_dir.plot([0, uu[0]], [0, uu[1]], [0, -10000], **shadow_kw_dir)

        # ---------- Right subplot: direction-only unit vectors from origin ----------
        U_items = [("C1", u1, "ρ1 (t1)"), ("C2", u2, "ρ2 (t2)"), ("C3", u3, "ρ3 (t3)")]
        for ci, uu, label in U_items:
            ax_dir.plot([0, uu[0]], [0, uu[1]], [0, uu[2]], color=ci, lw=2, label=label)
            ax_dir.scatter([uu[0]], [uu[1]], [uu[2]], color=ci, s=20)

            ax_dir.plot([0, uu[0]], [0, uu[1]], [-1.2, -1.2], color="gray", lw=1)

        ax_dir.set_xlabel("ECI X [unitless]")
        ax_dir.set_ylabel("ECI Y [unitless]")
        ax_dir.set_zlabel("ECI Z [unitless]")
        ax_dir.set_title("Line-of-Sight Unit Vectors (from origin)")
        ax_dir.set_xlim([-1.2   , 1.2]); ax_dir.set_ylim([-1.2, 1.2]); ax_dir.set_zlim([-1.2, 1.2])
        ax_dir.legend(loc="upper right", fontsize=9)

        if show:
            plt.show(block=block)

        return fig, (ax_geo, ax_dir)

class ObservationReducer:
    """Reduces observations for orbit determination.
    """
    def for_gibbs(self, gs, observables: list[Observation]) -> GibbsObservables:
        obs3 = sorted([o for o in observables if o.kind == "az_el_rng"], key=lambda o:o.jd)[:3]
        if len(obs3) < 3:
            raise ValueError("Gibbs requires 3 az/el/range observations")

        r_list, t_list = [], []
        for o in obs3:
            # Timnig
            jd = o.jd
            gmst_rad = gmst_from_jd(jd)

            # LLA
            lst_deg = np.degrees(gmst_rad) + gs.lon_deg
            lat_deg = gs.lat_deg

            # Ground Station ECI
            r_gs_eci = geodetic_to_eci(lat_deg, lst_deg, gs.h_m)  # (3,n) km
            R_ENU_ECI = enu_matrix(lat_deg, lst_deg).T   # 3×3
            
            # Satelite ENU
            az_deg, el_deg, rng_km = o.values
            q_ENU = ENU_from_az_el_rng(az_deg, el_deg, rng_km)

            # Satellite ECI
            q_ECI = R_ENU_ECI @ q_ENU
            r_eci = r_gs_eci + q_ECI

            # Adding to lists
            r_list.append(r_eci)
            t_list.append(jd)

        return GibbsObservables(r1_eci_km=r_list[0], r2_eci_km=r_list[1], r3_eci_km=r_list[2],
                                t1_jd=t_list[0], t2_jd=t_list[1], t3_jd=t_list[2])

    def for_lambert(self, gs, observables: list[Observation]) -> LambertObservables:
        obs2 = sorted([o for o in observables if o.kind == "az_el_rng_time"], key=lambda o:o.jd)[:2]

        if len(obs2) < 2:
            raise ValueError("Lambert requires 2 az/el/range observations")
        
        r_list, t_list = [], []
        dt_s = (obs2[1].jd - obs2[0].jd) * 86400.0

        for o in obs2:
            # Timing
            jd = o.jd
            gmst_rad = gmst_from_jd(jd)

            # LLA
            lst_deg = np.degrees(gmst_rad) + gs.lon_deg
            lat_deg = gs.lat_deg

            # Ground Station ECI
            r_gs_eci = geodetic_to_eci(lat_deg, lst_deg, gs.h_m)  # (3,n) km
            R_ENU_ECI = enu_matrix(lat_deg, lst_deg).T   # 3×3

            # Satellite ENU
            az_deg, el_deg, rng_km, _ = o.values
            q_ENU = ENU_from_az_el_rng(az_deg, el_deg, rng_km)

            # Satellite ECI
            q_ECI = R_ENU_ECI @ q_ENU
            r_eci = r_gs_eci + q_ECI

            # Adding to lists
            r_list.append(r_eci)
            t_list.append(jd)

        return LambertObservables(r_list[0], r_list[1], dt_s, obs2[0].jd, obs2[1].jd)

    def for_gauss(self, gs, observables: list[Observation]) -> GaussObservables:
        obs3 = sorted([o for o in observables if o.kind == "az_el_time"], key=lambda o:o.jd)[:3]
        if len(obs3) < 3:
            raise ValueError("Gauss requires 3 az/el observations")

        r_gs_list, u_list, t_list = [], [], []

        for o in obs3:
            az_deg, el_deg, jd = o.values
            gmst_rad = gmst_from_jd(jd)
            lst_deg = np.degrees(gmst_rad) + gs.lon_deg
            lat_deg = gs.lat_deg
            r_gs_eci = geodetic_to_eci(lat_deg, lst_deg, gs.h_m)  # (3,n) km
            R_ENU_ECI = enu_matrix(lat_deg, lst_deg).T   # 3×3
            u_ENU = u_from_az_el(az_deg, el_deg)  # Unit vector
            r_gs_list.append(r_gs_eci)
            u_list.append(R_ENU_ECI @ u_ENU)
            t_list.append(jd)

        dt12 = (t_list[0] - t_list[1]) * 86400.0
        dt32 = (t_list[2] - t_list[1]) * 86400.0

        return GaussObservables(u_list[0], u_list[1], u_list[2], r_gs_list[0], r_gs_list[1], r_gs_list[2], dt12, dt32, t_list[0], t_list[1], t_list[2])

class OrbitDeterminationSolver:
    """Solves orbit determination problems using various methods.
    """
    def __init__(self):
        self.reducer = ObservationReducer()

    def determine_orbit(self, gs, observables: list[Observation], method: str = "Gibbs", trajectory = None, plot: bool = False):
        if method == "Gibbs":
            observables = self.reducer.for_gibbs(gs, observables)
            r_km, r_km_s = self.gibbs(observables)
            if plot:
                observables.plot_vectors_3d(trajectory=trajectory, block=True)
        elif method == "Lambert":
            observables = self.reducer.for_lambert(gs, observables)
            result = self.lambert(observables)
            if result is None:
                # print("Lambert method failed: couldn't solve for Lagrange coefficients.")
                return None
            r_km, r_km_s = result
            if plot:
                observables.plot_vectors_3d(trajectory=trajectory,block=True)
        elif method == "Gauss":
            observables = self.reducer.for_gauss(gs, observables)
            result = self.gauss(observables)
            if result is None:
                # print("Gauss method failed: couldn't solve for universal variable.")
                return None
            r_km, r_km_s = result
            if plot:
                observables.plot_vectors_3d(trajectory=trajectory,block=True)
        else:
            raise ValueError(f"Unknown method: {method}")

        # print(f"Final Orbit: r (km): {r_km}, v (km/s): {r_km_s}")
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

        is_prograde = np.cross(r1, r2)[2] > 0

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
        q1, q2, q3 = observables.q1_eci, observables.q2_eci, observables.q3_eci
        R1, R2, R3 = observables.r1_gs_eci_km, observables.r2_gs_eci_km, observables.r3_gs_eci_km

        # Time intervals between observations
        tau_1 = observables.dt12_s
        tau_3 = observables.dt32_s
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
        E = np.dot(R2, q2)

        a = -(A**2 + 2*A*E + R2_mag2)
        b = -2*mu*B*(A + E)
        c = -mu**2 * B**2

        # Build polynomial: x^8 + a x^6 + b x^3 + c = 0
        coeffs = np.array([1.0, 0.0, a, 0.0, 0.0, b, 0.0, 0.0, c], float)
        roots = np.roots(coeffs)
        realpos = roots[np.isreal(roots)].real
        realpos = realpos[realpos > 0]
        r2_mag = np.max(realpos)

        q2_mag = A + mu * B / r2_mag**3

        q1_mag = 1 / D0 * ((6*(D31 * tau_1/tau_3 + D21 * tau/tau_3) * r2_mag**3 + mu * D31 * (tau**2 - tau_1**2) * tau_1/tau_3)
                        / (6*r2_mag**3 + mu * (tau**2 - tau_3**2))
                        - D11
                            )
        
        q3_mag = 1 / D0 * ((6*(D13 * tau_3/tau_1 - D23 * tau/tau_1) * r2_mag**3 + mu * D13 * (tau**2 - tau_3**2) * tau_3/tau_1)
                        / (6*r2_mag**3 + mu * (tau**2 - tau_3**2))
                        - D33
                        )

        r1 = R1 + q1 * (q1_mag)
        r2 = R2 + q2 * (q2_mag)
        r3 = R3 + q3 * (q3_mag)

        f1 = 1 - 1/2 * mu/r2_mag**3 * tau_1**2
        f3 = 1 - 1/2 * mu/r2_mag**3 * tau_3**2

        g1 = tau_1 - 1/6 * mu/r2_mag**3 * tau_1**3
        g3 = tau_3 - 1/6 * mu/r2_mag**3 * tau_3**3

        v2 = 1 / (f1 * g3 - f3 * g1) * (-f3 * r1 + f1 * r3)   # Equation (5.114)

        # Refining the estimates using the Gauss method
        for i in range(30):
            # Calculate magnitudes of r and v
            r_mag = np.linalg.norm(r2)
            v_mag = np.linalg.norm(v2)

            # Calculate reciprocal of semimajor axis
            alpha = 2 / r_mag - v_mag**2 / mu

            # Calculate the radial component of v2
            v_r = np.dot(r2, v2) / r_mag

            # Use Algorithm 3.3 to solve for universal variables chi1 and chi2
            chi1 = universal_variable_from_dt_r_v_alpha(tau_1, r_mag, v_r, alpha, mu)
            chi3 = universal_variable_from_dt_r_v_alpha(tau_3, r_mag, v_r, alpha, mu)

            if chi1 is None or chi3 is None:
                return None

            C1 = stumpff_C(alpha*chi1**2)
            C3 = stumpff_C(alpha*chi3**2)

            S1 = stumpff_S(alpha*chi1**2)
            S3 = stumpff_S(alpha*chi3**2)

            f1_prev, f3_prev, g1_prev, g3_prev = f1, f3, g1, g3

            f1 = 1 - chi1**2 / r_mag * C1
            f3 = 1 - chi3**2 / r_mag * C3
            g1 = tau_1 - 1/np.sqrt(mu) * chi1**3 * S1
            g3 = tau_3 - 1/np.sqrt(mu) * chi3**3 * S3

            # Average for faster convergence
            f1 = (f1_prev + f1) / 2
            f3 = (f3_prev + f3) / 2
            g1 = (g1_prev + g1) / 2
            g3 = (g3_prev + g3) / 2

            # Calculate c1 and c3 from 5.96, and 5.97
            c1 = g3 / (f1 * g3 - f3 * g1)
            c3 = -g1 / (f1 * g3 - f3 * g1)

            # Calculate q1, q2, and q3 from eqn 5.109, 5.111
            q1_mag = 1/D0 * (-D11 + 1/c1 * D21 - c3/c1 * D31)
            q2_mag = 1/D0 * (-c1 * D12 + D22 - c3 * D32)
            q3_mag = 1/D0 * (-c1/c3 * D13 + 1/c3 * D23 - D33)

            # Calculate r1, r2, and r3 with 5.86
            r1 = R1 + q1 * (q1_mag)
            r2 = R2 + q2 * (q2_mag)
            r3 = R3 + q3 * (q3_mag)

            # Calculate v2 with 5.118
            v2 = 1 / (f1 * g3 - f3 * g1) * (-f3 * r1 + f1 * r3)

        return r2, v2

def _u_enu_from_ra_dec(alpha_deg: float, delta_deg: float, lst_deg: float, lat_deg: float) -> np.ndarray:
    """
    Unit line-of-sight vector in ENU, given topocentric right ascension (alpha),
    declination (delta), local sidereal time (lst), and site latitude (lat).
    Follows the standard relations via hour angle H = LST - RA.
    """
    # Radians
    a = np.radians(alpha_deg)
    d = np.radians(delta_deg)
    th = np.radians(lst_deg)
    ph = np.radians(lat_deg)

    # Hour angle H = LST - RA
    H = th - a

    # ENU components from (H, δ, φ)
    e = -np.cos(d) * np.sin(H)
    n =  np.sin(d) * np.cos(ph) - np.cos(d) * np.sin(ph) * np.cos(H)
    u =  np.sin(d) * np.sin(ph) + np.cos(d) * np.cos(ph) * np.cos(H)

    return np.array([e, n, u], dtype=float)

def main():
    """
    Main function for satellite orbit determination. Tests the functionality of Gauss's method
    """
    # Observation set: (time_s, RA_deg, Dec_deg, LST_deg)
    obs = [
        (0.00,    43.537,  -8.7833, 44.506),
        (118.10,  54.420, -12.0740, 45.000),
        (237.58,  64.318, -15.1050, 45.499),
    ]

    # Site info (given): latitude 40 deg N, altitude 1 km
    lat_deg = 40.0
    h_m = 1000.0

    # Build u (ECI) and r_gs (ECI) for each epoch
    u_list = []
    r_gs_list = []
    t_jd_list = []

    # Arbitrary epoch (only time differences matter for Gauss)
    JD0 = 2458849.500000

    for t_s, ra_deg, dec_deg, lst_deg in obs:
        # Ground station ECI at this LST (longitude is implicit in LST)
        r_gs_eci = geodetic_to_eci(lat_deg, lst_deg, h_m)           # km
        R_ENU_ECI = enu_matrix(lat_deg, lst_deg).T                  # ENU -> ECI

        # Topocentric RA/Dec -> ENU unit vector, then to ECI
        u_ENU = _u_enu_from_ra_dec(ra_deg, dec_deg, lst_deg, lat_deg)
        u_ECI = R_ENU_ECI @ u_ENU

        u_list.append(u_ECI)
        r_gs_list.append(r_gs_eci)
        t_jd_list.append(JD0 + t_s / 86400.0)

    # Time deltas
    t1_jd, t2_jd, t3_jd = t_jd_list
    dt12_s = (t1_jd - t2_jd) * 86400.0
    dt32_s = (t3_jd - t2_jd) * 86400.0
    # Pack Gauss observables
    gauss_obs = GaussObservables(
        q1_eci=u_list[0],
        q2_eci=u_list[1],
        q3_eci=u_list[2],
        r1_gs_eci_km=r_gs_list[0],
        r2_gs_eci_km=r_gs_list[1],
        r3_gs_eci_km=r_gs_list[2],
        dt12_s=dt12_s,
        dt32_s=dt32_s,
        t1_jd=t1_jd,
        t2_jd=t2_jd,
        t3_jd=t3_jd,
    )

    # Run Gauss directly
    solver = OrbitDeterminationSolver()
    r2_km, v2_km_s = solver.gauss(gauss_obs)
    satellite = Satellite((r2_km, v2_km_s))
    a, e, inc, raan, aop, ta = satellite.keplerian_elements()
    print(f"Keplerian elements (JD {t2_jd:.8f}):")
    print(f"  Semi-major axis (a): {a:.3f} km")
    print(f"  Eccentricity (e): {e:.3f}")
    print(f"  RAAN (Ω): {raan:.3f} deg")
    print(f"  Argument of perigee (ω): {aop:.3f} deg")
    print(f"  True anomaly (ν): {ta:.3f} deg")

    print("\n[Gauss test]")
    print(f"t2 (JD): {t2_jd:.8f}")
    print(f"dt12_s: {dt12_s:.3f}, dt32_s: {dt32_s:.3f}")
    print(f"r2 (km): {r2_km}")
    print(f"v2 (km/s): {v2_km_s}")

if __name__ == "__main__":
    main()
