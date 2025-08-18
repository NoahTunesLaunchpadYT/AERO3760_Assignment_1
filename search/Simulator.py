# --- simulator.py ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import _sky_series_for_plot, gmst_from_jd
from tqdm.auto import tqdm  # at top of Simulator.py (or local import inside function)

# bring in your trajectory classes
from Trajectory import SatelliteTrajectory, GroundStationTrajectory
from Propagator import SatellitePropagator, GroundStationPropagator



class Simulator:
    """
    Stores satellites and ground stations in dicts.
    Owns JD timebase. Runs both propagators.
    Saves results as trajectories in:
      - self.satellite_trajectories
      - self.ground_station_trajectories
    """

    def __init__(self,
                 satellites: dict[str, object],
                 ground_stations: dict[str, object],
                 sat_prop: SatellitePropagator,
                 gs_prop: GroundStationPropagator) -> None:
        self.satellites: dict[str, object] = dict(satellites)
        self.ground_stations: dict[str, object] = dict(ground_stations)
        self.sat_prop = sat_prop
        self.gs_prop = gs_prop

        self.JD: np.ndarray | None = None
        self.start_time: tuple[int, int, int, int, int, float] | None = None

        self.satellite_trajectories: dict[str, SatelliteTrajectory] = {}
        self.ground_station_trajectories: dict[str, GroundStationTrajectory] = {}

    # ---- timebase (JD) ----
    @staticmethod
    def calendar_to_jd(Y: int, M: int, D: int, h: int, m: int, s: float) -> float:
        if Y < 1901 or Y >= 2100:
            raise ValueError("Year must be between 1901 and 2099")
        ut = (h + m/60 + s/3600)/24.0
        j0 = 367*Y - int((7*(Y + int((M + 9)/12)))/4) + int((275*M)/9) + D + 1721013.5
        jd = j0 + ut

        return jd

    def add_satellites(self,
                    sats: dict[str, object],
                    *,
                    overwrite: bool = False,
                    drop_old_trajectories: bool = False) -> dict[str, list[str]]:
        """
        Add a dictionary of satellites to the simulator.

        Args:
            sats: mapping {key -> Satellite-like object}. Each object should either
                expose .r0/.v0 (km, km/s) or implement .current_state() -> (r, v).
            overwrite: if False, existing keys are left unchanged and reported as skipped.
                    if True, replace existing entries.
            drop_old_trajectories: when overwriting, also remove any stored trajectory
                                for that key from self.satellite_trajectories.

        Returns:
            summary dict with lists: {"added": [...], "replaced": [...], "skipped": [...]}

        Notes:
            - This does NOT run propagation. Call `propagate_one(sat_key=...)`
            or `run_all(...)` afterwards if you want trajectories.
        """
        if not isinstance(sats, dict) or not sats:
            raise ValueError("sats must be a non-empty dict")

        def _is_satellite_like(x) -> bool:
            return (hasattr(x, "r0") and hasattr(x, "v0")) or hasattr(x, "current_state")

        added, replaced, skipped = [], [], []
        for k, v in sats.items():
            if not isinstance(k, str):
                raise TypeError(f"Satellite key must be str, got {type(k)!r}")
            if not _is_satellite_like(v):
                raise ValueError(f"Satellite '{k}' is missing r0/v0 or current_state()")

            if k in self.satellites:
                if overwrite:
                    self.satellites[k] = v
                    if drop_old_trajectories:
                        self.satellite_trajectories.pop(k, None)
                    replaced.append(k)
                else:
                    skipped.append(k)
            else:
                self.satellites[k] = v
                added.append(k)

        return {"added": added, "replaced": replaced, "skipped": skipped}

    def add_ground_stations(self,
                            gss: dict[str, object],
                            *,
                            overwrite: bool = False,
                            drop_old_trajectories: bool = False) -> dict[str, list[str]]:
        """
        Add a dictionary of ground stations to the simulator.

        Args:
            gss: mapping {key -> GroundStation-like object}. Each object should expose
                lat_deg, lon_deg (deg) and h_m (meters).
            overwrite: if False, existing keys are left unchanged and reported as skipped.
                    if True, replace existing entries.
            drop_old_trajectories: when overwriting, also remove any stored trajectory
                                for that key from self.ground_station_trajectories.

        Returns:
            summary dict with lists: {"added": [...], "replaced": [...], "skipped": [...]}

        Notes:
            - This does NOT run propagation. Call `propagate_one(gs_key=...)`
            or `run_all(...)` afterwards if you want trajectories.
        """
        if not isinstance(gss, dict) or not gss:
            raise ValueError("gss must be a non-empty dict")

        def _is_gs_like(x) -> bool:
            return all(hasattr(x, attr) for attr in ("lat_deg", "lon_deg", "h_m"))

        added, replaced, skipped = [], [], []
        for k, v in gss.items():
            if not isinstance(k, str):
                raise TypeError(f"Ground station key must be str, got {type(k)!r}")
            if not _is_gs_like(v):
                raise ValueError(f"Ground station '{k}' is missing lat_deg/lon_deg/h_m")

            if k in self.ground_stations:
                if overwrite:
                    self.ground_stations[k] = v
                    if drop_old_trajectories:
                        self.ground_station_trajectories.pop(k, None)
                    replaced.append(k)
                else:
                    skipped.append(k)
            else:
                self.ground_stations[k] = v
                added.append(k)

        return {"added": added, "replaced": replaced, "skipped": skipped}

    def build_timebase(self, Y: int, M: int, D: int, h: int, m: int, s: float,
                       tf_days: float, sample_dt_s: float) -> np.ndarray:
        jd0 = self.calendar_to_jd(Y, M, D, h, m, s)
        n = int(np.floor(tf_days*86400.0/sample_dt_s)) + 1
        self.JD = jd0 + np.arange(n, dtype=float) * (sample_dt_s/86400.0)
        self.start_time = (Y, M, D, h, m, s)
        return self.JD

    # ---- run everything (or subsets by keys) ----
    def run_all(self,
                sat_keys: list[str] | None = None,
                gs_keys: list[str] | None = None,
                *,
                progress: bool = True) -> None:
        if self.JD is None:
            raise ValueError("Build timebase first.")
        JD = self.JD

        s_keys = sat_keys if sat_keys is not None else list(self.satellites.keys())
        g_keys = gs_keys if gs_keys is not None else list(self.ground_stations.keys())

        total = len(g_keys) + len(s_keys)
        pbar = tqdm(total=total, desc="Propagating", unit="traj", leave=False) if progress else None

        try:
            # Ground stations
            for k in g_keys:
                gs = self.ground_stations[k]
                JDg, Rg, Vg, up_eci, E_eci, N_eci, U_eci = self.gs_prop.propagate(gs, JD)
                gtraj = GroundStationTrajectory(
                    JDg, Rg, Vg, self.start_time or (0, 0, 0, 0, 0, 0.0),
                    name=f"GS:{k}", enu_vectors=U_eci
                )
                # optional caches
                gtraj.up_vectors = up_eci
                gtraj.E_eci = E_eci; gtraj.N_eci = N_eci; gtraj.U_eci = U_eci
                self.ground_station_trajectories[k] = gtraj

                if pbar:
                    pbar.set_postfix_str(f"GS:{k}")
                    pbar.update(1)

            # Satellites
            for k in s_keys:
                sat = self.satellites[k]
                JDs, Rs, Vs = self.sat_prop.propagate(sat, JD)
                straj = SatelliteTrajectory(
                    JDs, Rs, Vs, self.start_time or (0, 0, 0, 0, 0, 0.0),
                    name=f"SAT:{k}"
                )
                self.satellite_trajectories[k] = straj

                if pbar:
                    pbar.set_postfix_str(f"SAT:{k}")
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()
    # ---- plotting using stored trajectories ----
    def plot_3d(self,
                sat_keys: list[str] | None = None,
                gs_keys: list[str] | None = None,
                show_earth: bool = True,
                earth_alpha: float = 0.15,
                earth_wire: bool = True,
                wire_steps: int = 24) -> None:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Earth
        if show_earth:
            Re = globals().get('R_EARTH', 6378.1363)
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            xs = Re * np.outer(np.cos(u), np.sin(v))
            ys = Re * np.outer(np.sin(u), np.sin(v))
            zs = Re * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, alpha=earth_alpha, linewidth=0, shade=True)
            if earth_wire:
                uw = np.linspace(0, 2*np.pi, wire_steps)
                vw = np.linspace(0, np.pi, wire_steps//2)
                xw = Re * np.outer(np.cos(uw), np.sin(vw))
                yw = Re * np.outer(np.sin(uw), np.sin(vw))
                zw = Re * np.outer(np.ones_like(uw), np.cos(vw))
                ax.plot_wireframe(xw, yw, zw, linewidth=0.3)

        s_keys = sat_keys if sat_keys is not None else list(self.satellite_trajectories.keys())
        g_keys = gs_keys  if gs_keys  is not None else list(self.ground_station_trajectories.keys())

        # GS
        for k in g_keys:
            G = self.ground_station_trajectories[k]
            R = G.R
            ax.plot(R[:,0], R[:,1], R[:,2], lw=1.2, label=f"GS:{k}")
            ax.scatter(R[0,0], R[0,1], R[0,2], s=30)
            ax.scatter(R[-1,0], R[-1,1], R[-1,2], s=30)

        # SATS
        for k in s_keys:
            S = self.satellite_trajectories[k]
            R = S.R
            ax.plot(R[:,0], R[:,1], R[:,2], lw=1.2, label=f"SAT:{k}")
            ax.scatter(R[0,0], R[0,1], R[0,2], s=30)
            ax.scatter(R[-1,0], R[-1,1], R[-1,2], s=30)

        # Your fixed scaling style (0.4 on z)
        all_R = []
        for k in g_keys: all_R.append(self.ground_station_trajectories[k].R)
        for k in s_keys: all_R.append(self.satellite_trajectories[k].R)
        all_R = np.vstack(all_R) if all_R else np.zeros((1,3))
        xs, ys, zs = all_R[:,0], all_R[:,1], all_R[:,2]
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
        mid = np.array([xs.mean(), ys.mean(), zs.mean()])
        ax.set_xlim(mid[0]-0.5*max_range, mid[0]+0.5*max_range)
        ax.set_ylim(mid[1]-0.5*max_range, mid[1]+0.5*max_range)
        ax.set_zlim(mid[2]-0.4*max_range, mid[2]+0.4*max_range)

        ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
        ax.set_title('ECI trajectories (3D)')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout(); plt.show()

    def animate_3d(self,
                sat_keys: list[str] | None = None,
                gs_keys: list[str] | None = None,
                step: int = 1,
                interval: int = 30,
                tail: int | None = 800,
                save_path: str | None = None,
                dpi: int = 120,
                repeat: bool = True,
                camera_spin: bool = False):   # set this False to allow free camera control
        s_keys = sat_keys if sat_keys is not None else list(self.satellite_trajectories.keys())
        g_keys = gs_keys  if gs_keys  is not None else list(self.ground_station_trajectories.keys())

        # Gather data
        series = []
        for k in g_keys:
            series.append(("GS:"+k, self.ground_station_trajectories[k].R[::step]))
        for k in s_keys:
            series.append(("SAT:"+k, self.satellite_trajectories[k].R[::step]))
        if not series:
            raise RuntimeError("No trajectories to animate. Run the simulator first.")

        # Determine axis limits from all paths
        all_R = np.vstack([R for _, R in series])
        xs, ys, zs = all_R[:,0], all_R[:,1], all_R[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Earth
        Re = globals().get('R_EARTH', 6378.1363)
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xsE = Re * np.outer(np.cos(u), np.sin(v))
        ysE = Re * np.outer(np.sin(u), np.sin(v))
        zsE = Re * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xsE, ysE, zsE, rstride=2, cstride=2, alpha=0.12, linewidth=0, shade=True)

        # Artists per series
        lines = []
        points = []
        for name, _ in series:
            ln, = ax.plot([], [], [], lw=1.3, label=name, animated=False)
            pt   = ax.plot([], [], [], marker='o', markersize=4, animated=False)[0]
            lines.append(ln); points.append(pt)

        # Fixed scaling style (0.4 on z)
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
        mid = np.array([xs.mean(), ys.mean(), zs.mean()])
        ax.set_xlim(mid[0]-0.5*max_range, mid[0]+0.5*max_range)
        ax.set_ylim(mid[1]-0.5*max_range, mid[1]+0.5*max_range)
        ax.set_zlim(mid[2]-0.4*max_range, mid[2]+0.4*max_range)
        ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
        ax.set_title('ECI trajectories (animated)')
        ax.legend(loc='upper right', fontsize=8)

        n_frames = min(R.shape[0] for _, R in series)
        if tail is None or tail <= 0:
            tail = n_frames

        def init():
            for ln, pt in zip(lines, points):
                ln.set_data([], []); ln.set_3d_properties([])
                pt.set_data([], []); pt.set_3d_properties([])
            # Do NOT set ax.view_init here; leave camera as-is for user control
            return (*lines, *points)

        def update(frame: int):
            for i, (_, R) in enumerate(series):
                i0 = max(0, frame - tail)
                seg = R[i0:frame+1]
                lines[i].set_data(seg[:,0], seg[:,1])
                lines[i].set_3d_properties(seg[:,2])
                points[i].set_data([R[frame,0]], [R[frame,1]])
                points[i].set_3d_properties([R[frame,2]])
            # Do NOT touch ax.view_init unless you explicitly want auto-spin
            if camera_spin:
                ax.view_init(elev=ax.elev, azim=(ax.azim + 0.5) % 360.0)
            return (*lines, *points)

        ani = animation.FuncAnimation(
            fig, update, frames=n_frames, init_func=init,
            interval=interval, blit=False, repeat=repeat, cache_frame_data=False
        )

        if save_path is not None:
            ext = save_path.lower().split('.')[-1]
            if ext == 'mp4':
                writer = animation.FFMpegWriter(fps=int(1000/interval), bitrate=1800)
                ani.save(save_path, writer=writer, dpi=dpi)
            elif ext in ('gif', 'agif'):
                writer = animation.PillowWriter(fps=int(1000/interval))
                ani.save(save_path, writer=writer, dpi=dpi)
            else:
                print("Unsupported extension; showing instead.")
        plt.show()
        return ani
    
    def _compute_az_el(self, gs_traj, sat_traj, return_distance=False):
        """
        Compute azimuth/elevation of a satellite as seen from a ground station,
        using the station's time-varying ENU basis in ECI.

        Args:
            gs_traj: Ground station trajectory with attributes JD, R, E_eci, N_eci, U_eci
            sat_traj: Satellite trajectory with attributes JD, R
            return_distance (bool): If True, also return the satellite-ground
                                    distance (same length as az/el arrays).

        Returns:
            az_deg (n,), el_deg (n,), vis_mask (n,) 
            If return_distance=True, also returns:
            distance_m (n,)
        """
        # Ensure same time base length
        if gs_traj.JD.shape[0] != sat_traj.JD.shape[0]:
            raise ValueError("GS and SAT trajectories must be sampled on the same JD grid")

        Rg = gs_traj.R                    # (n,3)
        Rs = sat_traj.R                   # (n,3)
        rho = Rs - Rg                     # LOS in ECI (n,3)

        # Local ENU basis in ECI
        E_eci = getattr(gs_traj, 'E_eci', None)
        N_eci = getattr(gs_traj, 'N_eci', None)
        U_eci = getattr(gs_traj, 'U_eci', None)
        if E_eci is None or N_eci is None or U_eci is None:
            raise RuntimeError("GroundStationTrajectory is missing E_eci/N_eci/U_eci. "
                            "Make sure you ran GroundStationPropagator and stored these.")

        # Project LOS onto ENU
        e = np.einsum('ij,ij->i', rho, E_eci)  # East component
        n = np.einsum('ij,ij->i', rho, N_eci)  # North component
        u = np.einsum('ij,ij->i', rho, U_eci)  # Up component

        # Azimuth measured from North toward East
        az = np.degrees(np.arctan2(e, n))      # [-180, 180]
        az = (az + 360.0) % 360.0

        # Elevation and range
        rho_norm = np.linalg.norm(rho, axis=1)
        el = np.degrees(np.arcsin(np.clip(u / rho_norm, -1.0, 1.0)))

        vis = el >= 0.0

        if return_distance:
            return az, el, vis, rho_norm
        else:
            return az, el, vis

    def plot_sky_anim_window(self, gs_key: str, sat_keys: list[str],
                            *, step=1, interval=40, tailsky=None, repeat=True,
                            min_elev_deg=0.0, distance_thresh_km=np.inf,
                            figsize=(7, 7), block=True):
        import numpy as np
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(90, 0)
        ax.set_thetagrids(range(0, 360, 30))
        ax.set_rgrids([0, 15, 30, 45, 60, 75, 90], angle=135)
        ax.set_title(f"Sky track (animated) from GS:{gs_key}  (elev ≥ {min_elev_deg:.0f}°)")

        gs_traj = self.ground_station_trajectories[gs_key]
        tracks = []
        n_frames = None
        for k in sat_keys:
            S = self.satellite_trajectories[k]
            az_full, el_full, _, dist_full = self._compute_az_el(gs_traj, S, return_distance=True)
            az = az_full[::step]; el = el_full[::step]; dist = dist_full[::step]

            th_s, r_s, th_d, r_d, vis, th_raw, r_raw = _sky_series_for_plot(
                az, el, dist, min_elev_deg=min_elev_deg,
                distance_thresh_km=distance_thresh_km
            )
            tracks.append({
                "name": f"SAT:{k}",
                "th_s": th_s, "r_s": r_s,
                "th_d": th_d, "r_d": r_d,
                "vis": vis, "th_raw": th_raw, "r_raw": r_raw
            })
            n_frames = len(th_raw) if n_frames is None else min(n_frames, len(th_raw))

        if tailsky is None or tailsky <= 0 or tailsky > n_frames:
            tailsky = n_frames

        # Two artists per sat: solid trail + dotted trail, plus head
        solid_lines, dotted_lines, heads = [], [], []
        for tr in tracks:
            ln_s, = ax.plot([], [], lw=1.6, label=tr["name"], animated=True)
            ln_d, = ax.plot([], [], lw=1.6, linestyle=':', animated=True)
            hd,   = ax.plot([], [], marker='o', ms=4, animated=True)
            solid_lines.append(ln_s); dotted_lines.append(ln_d); heads.append(hd)
        ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1.10), fontsize=8)

        def init():
            for ln_s, ln_d, hd in zip(solid_lines, dotted_lines, heads):
                ln_s.set_data([], []); ln_d.set_data([], []); hd.set_data([], [])
            return (*solid_lines, *dotted_lines, *heads)

        def update(frame: int):
            i0 = max(0, frame - tailsky + 1)
            for i, tr in enumerate(tracks):
                solid_lines[i].set_data(tr["th_s"][i0:frame+1], tr["r_s"][i0:frame+1])
                dotted_lines[i].set_data(tr["th_d"][i0:frame+1], tr["r_d"][i0:frame+1])
                if tr["vis"][frame]:
                    heads[i].set_data([tr["th_raw"][frame]], [tr["r_raw"][frame]])
                else:
                    heads[i].set_data([], [])
            return (*solid_lines, *dotted_lines, *heads)

        ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                    init_func=init, interval=interval,
                                    blit=True, repeat=repeat, cache_frame_data=False)
        plt.show(block=block)
        return fig, ani

    def plot_sky_static_window(self, gs_key: str, sat_keys: list[str],
                            *, min_elev_deg=0.0, distance_thresh_km=np.inf,
                            figsize=(7, 7), block=True):
        import numpy as np
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(90, 0)
        ax.set_thetagrids(range(0, 360, 30))
        ax.set_rgrids([0, 15, 30, 45, 60, 75, 90], angle=135)
        ax.set_title(f"Sky track (static) from GS:{gs_key}  (elev ≥ {min_elev_deg:.0f}°)")

        gs_traj = self.ground_station_trajectories[gs_key]
        for k in sat_keys:
            S = self.satellite_trajectories[k]
            az_deg, el_deg, _, dist_km = self._compute_az_el(gs_traj, S, return_distance=True)

            th_s, r_s, th_d, r_d, vis, th_raw, r_raw = _sky_series_for_plot(
                az_deg, el_deg, dist_km, min_elev_deg=min_elev_deg,
                distance_thresh_km=distance_thresh_km
            )

            # Solid (near)
            ax.plot(th_s, r_s, lw=1.6, label=f"SAT:{k}")
            # Dotted (far but visible)
            ax.plot(th_d, r_d, lw=1.6, linestyle=':')

            # Mark first/last visible samples (optional)
            if np.any(vis):
                idx = np.where(vis)[0]
                ax.plot([th_raw[idx[0]]], [r_raw[idx[0]]], marker='o', ms=4)
                ax.plot([th_raw[idx[-1]]], [r_raw[idx[-1]]], marker='s', ms=4)

        ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1.10), fontsize=8)
        plt.show(block=block)
        return fig

    def plot_eci_anim_window(self, gs_key: str, sat_keys: list[str],
                                *, step=1, interval=40, tail3d=None,
                                figsize=(8, 6), camera_spin=False, repeat=True, block=True):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import animation

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        gs_traj = self.ground_station_trajectories[gs_key]
        sats = {k: self.satellite_trajectories[k] for k in sat_keys}

        series3d = [("GS:"+gs_key, gs_traj.R[::step])]
        for k, S in sats.items():
            series3d.append(("SAT:"+k, S.R[::step]))
        all_R3d = np.vstack([R for _, R in series3d])
        xs3d, ys3d, zs3d = all_R3d[:,0], all_R3d[:,1], all_R3d[:,2]

        lines3d, pts3d = [], []
        for name, _ in series3d:
            # Create the line first to get its assigned colour from the cycle
            ln, = ax.plot([], [], [], lw=1.2, label=name, animated=True)
            color = ln.get_color()  # <- reuse this colour for the points
            pt, = ax.plot([], [], [], marker='o', markersize=6,
                        linestyle='None',  # no connecting line for the head
                        color=color,        # same colour as the line
                        markerfacecolor=color,
                        markeredgecolor=color,
                        animated=True)
            lines3d.append(ln); pts3d.append(pt)

        max_range3d = np.array([xs3d.max()-xs3d.min(),
                                ys3d.max()-ys3d.min(),
                                zs3d.max()-zs3d.min()]).max()
        mid3d = np.array([xs3d.mean(), ys3d.mean(), zs3d.mean()])
        ax.set_xlim(mid3d[0]-0.5*max_range3d, mid3d[0]+0.5*max_range3d)
        ax.set_ylim(mid3d[1]-0.5*max_range3d, mid3d[1]+0.5*max_range3d)
        ax.set_zlim(mid3d[2]-0.4*max_range3d, mid3d[2]+0.4*max_range3d)
        ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
        ax.set_title('ECI trajectories — animated')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(False)

        n_frames_3d = min(R.shape[0] for _, R in series3d)
        if tail3d is None or tail3d <= 0 or tail3d > n_frames_3d:
            tail3d = n_frames_3d

        def init():
            for ln, pt in zip(lines3d, pts3d):
                ln.set_data([], []); ln.set_3d_properties([])
                pt.set_data([], []); pt.set_3d_properties([])
            return (*lines3d, *pts3d)

        def update(frame: int):
            for i, (_, R) in enumerate(series3d):
                i0 = max(0, frame - tail3d + 1)
                seg = R[i0:frame+1]
                lines3d[i].set_data(seg[:,0], seg[:,1])
                lines3d[i].set_3d_properties(seg[:,2])
                pts3d[i].set_data([R[frame,0]], [R[frame,1]])
                pts3d[i].set_3d_properties([R[frame,2]])
            if camera_spin:
                ax.view_init(elev=ax.elev, azim=(ax.azim + 0.5) % 360.0)
            return (*lines3d, *pts3d)

        use_blit = not camera_spin
        ani = animation.FuncAnimation(fig, update, frames=n_frames_3d,
                                    init_func=init, interval=interval,
                                    blit=use_blit, repeat=repeat, cache_frame_data=False)
        plt.show(block=block)
        return fig, ani


    def plot_eci_static_window(self, gs_key: str, sat_keys: list[str],
                            *, figsize=(8, 6), show_earth=True,
                            earth_alpha=0.15, earth_wire=True, wire_steps=24, block=True):
        import numpy as np
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if self.JD is None:
            raise ValueError("Build the timebase and run the simulator first.")
        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        for k in sat_keys:
            if k not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{k}' not found")

        gs_traj = self.ground_station_trajectories[gs_key]
        sats = {k: self.satellite_trajectories[k] for k in sat_keys}

        if show_earth:
            Re = globals().get('R_EARTH', 6378.1363)
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            xs = Re * np.outer(np.cos(u), np.sin(v))
            ys = Re * np.outer(np.sin(u), np.sin(v))
            zs = Re * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs, ys, zs, rstride=2, cstride=2,
                            alpha=earth_alpha, linewidth=0, shade=True)
            if earth_wire:
                uw = np.linspace(0, 2*np.pi, wire_steps)
                vw = np.linspace(0, np.pi, wire_steps//2)
                xw = Re * np.outer(np.cos(uw), np.sin(vw))
                yw = Re * np.outer(np.sin(uw), np.sin(vw))
                zw = Re * np.outer(np.ones_like(uw), np.cos(vw))
                ax.plot_wireframe(xw, yw, zw, linewidth=0.3)

        Rg = gs_traj.R
        ax.plot(Rg[:,0], Rg[:,1], Rg[:,2], lw=1.2, label=f"GS:{gs_key}")
        ax.scatter(Rg[0,0], Rg[0,1], Rg[0,2], s=30)
        ax.scatter(Rg[-1,0], Rg[-1,1], Rg[-1,2], s=30)

        all_R = [Rg]
        for k, S in sats.items():
            R = S.R
            ax.plot(R[:,0], R[:,1], R[:,2], lw=1.2, label=f"SAT:{k}")
            ax.scatter(R[0,0], R[0,1], R[0,2], s=30)
            ax.scatter(R[-1,0], R[-1,1], R[-1,2], s=30)
            all_R.append(R)

        all_R = np.vstack(all_R)
        xs, ys, zs = all_R[:,0], all_R[:,1], all_R[:,2]
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
        mid = np.array([xs.mean(), ys.mean(), zs.mean()])
        ax.set_xlim(mid[0]-0.5*max_range, mid[0]+0.5*max_range)
        ax.set_ylim(mid[1]-0.5*max_range, mid[1]+0.5*max_range)
        ax.set_zlim(mid[2]-0.4*max_range, mid[2]+0.4*max_range)
        ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
        ax.set_title('ECI trajectories (3D) — static')
        ax.legend(loc='upper right', fontsize=8)

        plt.show(block=block)
        return fig

    def plot_all_five(self, gs_key: str, sat_keys: list[str], *,
                    step: int = 1,
                    interval: int = 40,
                    tail3d: int | None = 800,
                    tailsky: int | None = 600,
                    min_elev_deg: float = 0.0,
                    distance_thresh_km: float = np.inf,
                    repeat: bool = True):
        """
        Legacy wrapper: opens four separate non-blocking windows.
        """
        # 3D static
        fig1 = self.plot_eci_static_window(gs_key, sat_keys, block=False)

        # 3D animated
        fig2, ani2 = self.plot_eci_anim_window(gs_key, sat_keys, step=step,
                                            interval=interval, tail3d=tail3d,
                                            repeat=repeat, block=False)

        # Sky static (solid near, dotted far)
        fig3 = self.plot_sky_static_window(gs_key, sat_keys,
                                        min_elev_deg=min_elev_deg,
                                        distance_thresh_km=distance_thresh_km, block=False)

        # Sky animated (solid near, dotted far)
        fig4, ani4 = self.plot_sky_anim_window(gs_key, sat_keys,
                                            step=step, interval=interval,
                                            tailsky=tailsky, repeat=repeat,
                                            min_elev_deg=min_elev_deg,
                                            distance_thresh_km=distance_thresh_km, block=False)

        fig5 = self.plot_ground_tracks_window(gs_key, sat_keys)

        return (fig1, fig2, fig3, fig4, fig5), (ani2, ani4)

    def plot_elevation_visibility_distance(self,
                                        gs_key: str,
                                        *,
                                        sat_keys: list[str] | tuple[str, ...],
                                        min_elev_deg: float = 0.0,
                                        max_distance_km: float | None = None,
                                        show_all_distances: bool = True,
                                        fig_size: tuple[float, float] = (12, 9),
                                        save_plot_path: str | None = None):
        """
        One-stop plot with three rows:
        (1) Overlapping elevations for all satellites (with min-elev line)
        (2) Visibility raster (time on x, satellite on y; 1=visible)
        (3) Closest *visible* distance vs time (NaN when none visible), with optional threshold

        Also logs:
        - total time with ≥1 satellite visible
        - average number of satellites visible (duration-weighted)
        - visible time per satellite
        - portion of time when a visible satellite is within `max_distance_km` (if provided)

        Returns:
        dict with:
            'jd'                 : (n,) JD
            'count_visible'      : (n-1,) # visible per interval
            'closest_visible_km' : (n,)   min distance among visible sats (NaN if none)
            'chosen_sat'         : (n,)   satellite key chosen at each sample or None
            'per_sat'            : { key: {'elevation_deg': (n,), 'visible': (n,), 'distance_km': (n,)} }
            'stats'              : dict with 'time_with_any', 'avg_visible', 'per_sat_time', 'fraction_within'
        """
        if self.JD is None:
            raise ValueError("Build the timebase and run the propagators before calling this plot.")
        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        if not sat_keys:
            raise ValueError("sat_keys must be a non-empty list/tuple of satellite keys.")

        sat_keys = list(sat_keys)
        for sk in sat_keys:
            if sk not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{sk}' not found")

        gs_traj = self.ground_station_trajectories[gs_key]
        jd = self.JD
        n = jd.shape[0]
        if n < 2:
            raise ValueError("Timebase must contain at least two samples.")

        # Interval durations (seconds) for duration-weighted stats
        dt = np.diff(jd) * 86400.0
        total_duration = float(np.sum(dt))
        t_mid = 0.5 * (jd[:-1] + jd[1:])  # midpoints for interval plots

        # ---- Per-satellite series ----
        per_sat: dict[str, dict] = {}
        elev_stack = []
        dist_stack = []
        vis_stack  = []

        for sk in sat_keys:
            sat_traj = self.satellite_trajectories[sk]
            # Use your extended API: returns (az, el, vis, distance)
            az_deg, el_deg, _, rng_km = self._compute_az_el(gs_traj, sat_traj, return_distance=True)
            if el_deg.shape[0] != n or rng_km.shape[0] != n:
                raise ValueError(f"Mismatched timebase for satellite '{sk}'.")

            valid = np.isfinite(el_deg) & np.isfinite(rng_km)
            vis_mask = (el_deg >= float(min_elev_deg)) & valid & (rng_km <= max_distance_km)

            per_sat[sk] = {
                "elevation_deg": el_deg.copy(),
                "visible": vis_mask.copy(),
                "distance_km": rng_km.copy(),
            }

            elev_stack.append(el_deg)
            dist_stack.append(rng_km)
            vis_stack.append(vis_mask)

        elev_stack = np.vstack(elev_stack)  # (S, n)
        dist_stack = np.vstack(dist_stack)  # (S, n)
        vis_stack  = np.vstack(vis_stack)   # (S, n)
        S = len(sat_keys)

        # ---- Visibility counts per interval ----
        vis_intervals = vis_stack[:, :-1]                 # (S, n-1)
        count_visible = np.sum(vis_intervals, axis=0)     # (n-1,)

        # ---- Closest visible distance per sample ----
        masked_dist = np.where(vis_stack, dist_stack, np.inf)  # (S, n)
        min_dist = masked_dist.min(axis=0)                     # (n,)
        has_any_visible = np.any(vis_stack, axis=0)            # (n,)
        min_dist[~has_any_visible] = np.nan

        # Which satellite was chosen (by index -> key)
        chosen_idx = np.argmin(masked_dist, axis=0)            # (n,)
        chosen_idx[~has_any_visible] = -1
        chosen_sat = np.array([sat_keys[i] if i >= 0 else None for i in chosen_idx], dtype=object)

        # ---- Stats ----
        time_with_any = float(np.sum((count_visible > 0) * dt))
        avg_visible = float(np.sum(count_visible * dt) / total_duration) if total_duration > 0 else 0.0
        time_visible_per_sat = np.sum(vis_intervals * dt, axis=1)  # (S,)

        fraction_within = None
        if max_distance_km is not None and np.isfinite(max_distance_km):
            # need interval alignment: evaluate min_dist on [i] for interval [i, i+1)
            within = (has_any_visible[:-1]) & (min_dist[:-1] <= float(max_distance_km))
            time_within = float(np.sum(dt[within]))
            fraction_within = (time_within / total_duration) if total_duration > 0 else 0.0

        # Log nicely
        def _fmt_time(seconds: float) -> str:
            if seconds < 120: return f"{seconds:.1f}s"
            m = seconds / 60.0
            if m < 120: return f"{m:.1f} min"
            return f"{m/60.0:.2f} h"

        print(f"[VIS STATS] Total duration: {_fmt_time(total_duration)}")
        print(f"[VIS STATS] Time with ≥1 satellite: {_fmt_time(time_with_any)} "
            f"({100.0 * time_with_any / total_duration:.2f}%)")
        print(f"[VIS STATS] Average # visible: {avg_visible:.3f}")
        for sk, tsec in zip(sat_keys, time_visible_per_sat):
            print(f"[VIS STATS] {sk}: visible {_fmt_time(float(tsec))} "
                f"({100.0 * float(tsec) / total_duration:.2f}%)")
        if fraction_within is not None:
            pct = 100.0 * fraction_within
            print(f"[DIST STATS] Fraction of total time with a visible sat ≤ {max_distance_km:.1f} km: {pct:.2f}%")

        # ---- Plot (3 rows) ----
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=fig_size, sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 3]}
        )

        # Row 1: overlapping elevations
        for sk in sat_keys:
            ax1.plot(jd, per_sat[sk]["elevation_deg"], lw=1.0, label=sk)
        ax1.axhline(min_elev_deg, linestyle="--", linewidth=1.0, color="black")
        ax1.set_ylabel("Elevation [deg]")
        ax1.set_title(f"Elevations — GS: {gs_key} (min elev {min_elev_deg:.1f}°)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(ncols=2, fontsize=8, loc="upper right")

        # Row 2: visibility raster (interval-based)
        vis_img = vis_intervals.astype(int)
        ax2.imshow(vis_img, aspect="auto", interpolation="nearest",
                extent=[jd[0], jd[-1], -0.5, S - 0.5])
        ax2.set_yticks(range(S))
        ax2.set_yticklabels(sat_keys)
        ax2.set_ylabel("Satellite")
        visible_pct = 100.0 * (time_with_any / total_duration) if total_duration > 0 else 0.0
        ax2.set_title(f"Visibility (interval): {visible_pct:.1f}% of time has ≥1 visible")
        ax2.grid(False)

        # Row 3: closest visible distance
        if show_all_distances:
            for sk in sat_keys:
                ax3.plot(jd, per_sat[sk]["distance_km"], lw=0.8, alpha=0.35)
        ax3.plot(jd, min_dist, lw=2.0, color="black", label="Closest visible")
        if max_distance_km is not None and np.isfinite(max_distance_km):
            ax3.axhline(max_distance_km, linestyle="--", linewidth=1.0, color="black")
            if fraction_within is not None:
                ax3.text(0.01, 0.95, f"≤ {max_distance_km:.0f} km for {100*fraction_within:.1f}% of time",
                        transform=ax3.transAxes, va="top", ha="left", fontsize=9)
        ax3.set_ylabel("Range [km]")
        ax3.set_xlabel("Julian Date")
        ax3.set_title("Closest Visible Distance")
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_plot_path:
            try:
                fig.savefig(save_plot_path, dpi=160, bbox_inches="tight")
            except Exception as exc:
                print(f"[warn] could not save plot to '{save_plot_path}': {exc}")
        plt.show()

        return {
            "jd": jd.copy(),
            "count_visible": count_visible.astype(float),
            "closest_visible_km": min_dist,
            "chosen_sat": chosen_sat,
            "per_sat": per_sat,
            "stats": {
                "time_with_any": time_with_any,
                "avg_visible": avg_visible,
                "per_sat_time": {sk: float(t) for sk, t in zip(sat_keys, time_visible_per_sat)},
                "fraction_within": fraction_within,
                "total_duration": total_duration,
            },
        }

    def plot_ground_tracks_window(self, gs_key: str, sat_keys: list[str],
                                *, step: int = 1, figsize=(10, 6), block: bool = True):
        """
        Static ground tracks for multiple satellites.
        - Marks the ground station with a star (at its geodetic lon/lat).
        - Each satellite has a unique colour.
        - Handles lon wrap (-180..180) with NaN segmentation to avoid spurious lines.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        for k in sat_keys:
            if k not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{k}' not found")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # --- helper: ECI->ECEF->(lat,lon) for a trajectory S ---
        def _eci_to_latlon(traj):
            x, y, z = traj.R[::step, 0], traj.R[::step, 1], traj.R[::step, 2]
            theta = gmst_from_jd(self.JD[::step])               # array of angles
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x_e =  cos_t * x + sin_t * y
            y_e = -sin_t * x + cos_t * y
            z_e =  z
            r = np.linalg.norm(np.stack([x_e, y_e, z_e], axis=-1), axis=-1)
            lat = np.degrees(np.arcsin(z_e / r))
            lon = np.degrees(np.arctan2(y_e, x_e))
            lon = (lon + 180.0) % 360.0 - 180.0
            return lon, lat

        # --- plot each satellite with unique colour and wrap-safe segmentation ---
        def _plot_wrapped(lon, lat, label):
            lon = np.asarray(lon); lat = np.asarray(lat)
            dlon = np.diff(lon)
            # break line when we jump across the map edge
            breaks = np.where(np.abs(dlon) > 180.0)[0] + 1
            lon_seg = lon.astype(float).copy()
            lat_seg = lat.astype(float).copy()
            lon_seg[breaks] = np.nan
            lat_seg[breaks] = np.nan
            ln, = ax.plot(lon_seg, lat_seg, lw=1.5, label=label)
            # head marker in same colour (last point)
            ax.plot([lon[-1]], [lat[-1]], marker='o', ms=5,
                    color=ln.get_color(), linestyle='None')
            return ln

        # Ground station star (use its geodetic lat/lon)
        gs = self.ground_stations[gs_key] if hasattr(self, "ground_stations") else self.ground_station_trajectories[gs_key]
        gs_lat = float(getattr(gs, "lat_deg", None) if hasattr(gs, "lat_deg") else gs.lat_deg)
        gs_lon = float(getattr(gs, "lon_deg", None) if hasattr(gs, "lon_deg") else gs.lon_deg)
        gs_lon = (gs_lon + 180.0) % 360.0 - 180.0
        ax.plot([gs_lon], [gs_lat], marker='*', ms=12, color='k', label=f"GS:{gs_key}")

        # Satellites
        for k in sat_keys:
            S = self.satellite_trajectories[k]
            lon, lat = _eci_to_latlon(S)
            _plot_wrapped(lon, lat, label=f"SAT:{k}")

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title("Ground tracks (approx)")
        ax.set_xlim([-180, 180]); ax.set_ylim([-90, 90])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        plt.show(block=block)
        return fig
