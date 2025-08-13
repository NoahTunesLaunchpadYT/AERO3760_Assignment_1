# --- simulator.py ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from utils import _segmented_polar_arrays
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
        if M <= 2:
            Y -= 1; M += 12
        A = int(Y/100)
        B = 2 - A + int(A/4)
        frac = (h + m/60 + s/3600)/24.0
        return int(365.25*(Y+4716)) + int(30.6001*(M+1)) + D + frac + B - 1524.5

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
    
    def _compute_az_el(self, gs_traj, sat_traj):
        """
        Compute azimuth/elevation of a satellite as seen from a ground station,
        using the station's time-varying ENU basis in ECI.

        Returns:
            az_deg (n,), el_deg (n,), vis_mask (n,) where vis_mask = el_deg >= 0
        """
        # Ensure same time base length
        if gs_traj.JD.shape[0] != sat_traj.JD.shape[0]:
            raise ValueError("GS and SAT trajectories must be sampled on the same JD grid")

        Rg = gs_traj.R                    # (n,3)
        Rs = sat_traj.R                   # (n,3)
        rho = Rs - Rg                     # LOS in ECI (n,3)

        # Local ENU basis in ECI; rows are time samples
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

        # Elevation
        rho_norm = np.linalg.norm(rho, axis=1)
        el = np.degrees(np.arcsin(np.clip(u / rho_norm, -1.0, 1.0)))

        vis = el >= 0.0
        return az, el, vis

    def plot_sky_track(self,
                    gs_key: str,
                    sat_keys: list[str],
                    min_elev_deg: float = 0.0,
                    save_path: str | None = None):
        """
        Plot static sky tracks (az/el) for selected satellites as seen from a ground station.
        Uses NaN segmentation to avoid chords across horizon gaps and azimuth wraps.
        """
        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        for k in sat_keys:
            if k not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{k}' not found")

        import numpy as np
        import matplotlib.pyplot as plt

        gs_traj = self.ground_station_trajectories[gs_key]

        # Polar sky plot: azimuth clockwise from North; radius = elevation
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(90, 0)  # centre = 90° (zenith), edge = 0° (horizon)
        ax.set_thetagrids(range(0, 360, 30))
        ax.set_rgrids([0, 15, 30, 45, 60, 75, 90], angle=135)

        for k in sat_keys:
            sat_traj = self.satellite_trajectories[k]
            az_deg, el_deg, _ = self._compute_az_el(gs_traj, sat_traj)

            th_plot, r_plot, vis, th_raw, r_raw = _segmented_polar_arrays(
                az_deg, el_deg, min_elev_deg
            )
            ax.plot(th_plot, r_plot, lw=1.4, label=f"SAT:{k}")

            # Mark first/last visible points (over entire timebase)
            if np.any(vis):
                idx = np.where(vis)[0]
                ax.plot([th_raw[idx[0]]], [r_raw[idx[0]]], marker='o', ms=4)
                ax.plot([th_raw[idx[-1]]], [r_raw[idx[-1]]], marker='s', ms=4)

        ax.set_title(f"Sky track from GS:{gs_key} (elev ≥ {min_elev_deg:.0f}°)")
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.10), fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()
        return fig

    def animate_sky_track(self,
                        gs_key: str,
                        sat_keys: list[str],
                        step: int = 1,
                        interval: int = 40,
                        tail: int | None = 600,
                        min_elev_deg: float = 0.0,
                        save_path: str | None = None,
                        dpi: int = 120,
                        repeat: bool = True):
        """
        Animate sky tracks for selected satellites as seen from a ground station.
        Uses NaN segmentation to avoid chords across horizon gaps and azimuth wraps.
        """
        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        for k in sat_keys:
            if k not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{k}' not found")

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        gs_traj = self.ground_station_trajectories[gs_key]

        # Precompute per-satellite arrays (downsample by step)
        tracks = []  # list of dicts for each sat
        n_frames = None
        for k in sat_keys:
            sat_traj = self.satellite_trajectories[k]
            az_deg, el_deg, _ = self._compute_az_el(gs_traj, sat_traj)

            th_plot, r_plot, vis, th_raw, r_raw = _segmented_polar_arrays(
                az_deg[::step], el_deg[::step], min_elev_deg
            )
            tracks.append({
                "name": f"SAT:{k}",
                "theta_plot": th_plot,   # segmented for the trail
                "r_plot": r_plot,
                "vis": vis,              # visibility mask (downsampled)
                "theta_raw": th_raw,     # raw for head marker
                "r_raw": r_raw
            })
            n_frames = len(th_raw) if n_frames is None else min(n_frames, len(th_raw))

        if tail is None or tail <= 0:
            tail = n_frames

        # Polar sky plot setup
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(90, 0)
        ax.set_thetagrids(range(0, 360, 30))
        ax.set_rgrids([0, 15, 30, 45, 60, 75, 90], angle=135)
        ax.set_title(f"Sky track (animated) from GS:{gs_key} (elev ≥ {min_elev_deg:.0f}°)")

        # Artists
        line_art, head_art = [], []
        for tr in tracks:
            ln, = ax.plot([], [], lw=1.5, label=tr["name"], animated=False)
            hd, = ax.plot([], [], marker='o', ms=4, animated=False)
            line_art.append(ln); head_art.append(hd)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.10), fontsize=8)

        def init():
            for ln, hd in zip(line_art, head_art):
                ln.set_data([], []); hd.set_data([], [])
            return (*line_art, *head_art)

        def update(frame: int):
            for i, tr in enumerate(tracks):
                i0 = max(0, frame - tail)
                th_seg = tr["theta_plot"][i0:frame+1]
                r_seg  = tr["r_plot"][i0:frame+1]
                line_art[i].set_data(th_seg, r_seg)

                # Head marker only when visible at this frame
                if tr["vis"][frame]:
                    head_art[i].set_data([tr["theta_raw"][frame]], [tr["r_raw"][frame]])
                else:
                    head_art[i].set_data([], [])
            return (*line_art, *head_art)

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

    def plot_all_four(self,
                    gs_key: str,
                    sat_keys: list[str],
                    *,
                    step: int = 1,
                    interval: int = 40,
                    tail3d: int | None = 800,
                    tailsky: int | None = 600,
                    min_elev_deg: float = 0.0,
                    figsize=(14, 10),
                    show_earth: bool = True,
                    earth_alpha: float = 0.15,
                    earth_wire: bool = True,
                    wire_steps: int = 24,
                    camera_spin: bool = False,
                    save_path: str | None = None,
                    dpi: int = 120,
                    repeat: bool = True):
        """
        2x2 panel:
        (TL) Static 3D ECI plot
        (TR) 3D ECI animation (fixed camera; low-res Earth; blitting if possible)
        (BL) Static sky plot (polar, elevation as radius) with NaN segmentation
        (BR) Animated sky plot with NaN segmentation (blitting)

        Returns: (fig, ani)
        - fig: Matplotlib Figure
        - ani: FuncAnimation for the combined animations (TR, BR)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if self.JD is None:
            raise ValueError("Build the timebase and run the simulator first.")
        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        for k in sat_keys:
            if k not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{k}' not found")

        gs_traj = self.ground_station_trajectories[gs_key]
        sats = {k: self.satellite_trajectories[k] for k in sat_keys}

        # =========================
        # Figure & subplots (2x2)
        # =========================
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        g = fig.add_gridspec(2, 2)
        ax3d_static = fig.add_subplot(g[0, 0], projection='3d')
        ax3d_anim   = fig.add_subplot(g[0, 1], projection='3d')
        axsky_static = fig.add_subplot(g[1, 0], projection='polar')
        axsky_anim   = fig.add_subplot(g[1, 1], projection='polar')

        # ==========================================
        # (TL) Static 3D ECI plot (GS + satellites)
        # ==========================================
        if show_earth:
            Re = globals().get('R_EARTH', 6378.1363)
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            xs = Re * np.outer(np.cos(u), np.sin(v))
            ys = Re * np.outer(np.sin(u), np.sin(v))
            zs = Re * np.outer(np.ones_like(u), np.cos(v))
            ax3d_static.plot_surface(xs, ys, zs, rstride=2, cstride=2,
                                    alpha=earth_alpha, linewidth=0, shade=True)
            if earth_wire:
                uw = np.linspace(0, 2*np.pi, wire_steps)
                vw = np.linspace(0, np.pi, wire_steps//2)
                xw = Re * np.outer(np.cos(uw), np.sin(vw))
                yw = Re * np.outer(np.sin(uw), np.sin(vw))
                zw = Re * np.outer(np.ones_like(uw), np.cos(vw))
                ax3d_static.plot_wireframe(xw, yw, zw, linewidth=0.3)

        Rg = gs_traj.R
        ax3d_static.plot(Rg[:,0], Rg[:,1], Rg[:,2], lw=1.2, label=f"GS:{gs_key}")
        ax3d_static.scatter(Rg[0,0], Rg[0,1], Rg[0,2], s=30)
        ax3d_static.scatter(Rg[-1,0], Rg[-1,1], Rg[-1,2], s=30)

        all_R_static = [Rg]
        for k, S in sats.items():
            R = S.R
            ax3d_static.plot(R[:,0], R[:,1], R[:,2], lw=1.2, label=f"SAT:{k}")
            ax3d_static.scatter(R[0,0], R[0,1], R[0,2], s=30)
            ax3d_static.scatter(R[-1,0], R[-1,1], R[-1,2], s=30)
            all_R_static.append(R)

        all_R = np.vstack(all_R_static)
        xs_all, ys_all, zs_all = all_R[:,0], all_R[:,1], all_R[:,2]
        max_range = np.array([xs_all.max()-xs_all.min(),
                            ys_all.max()-ys_all.min(),
                            zs_all.max()-zs_all.min()]).max()
        mid = np.array([xs_all.mean(), ys_all.mean(), zs_all.mean()])
        ax3d_static.set_xlim(mid[0]-0.5*max_range, mid[0]+0.5*max_range)
        ax3d_static.set_ylim(mid[1]-0.5*max_range, mid[1]+0.5*max_range)
        ax3d_static.set_zlim(mid[2]-0.4*max_range, mid[2]+0.4*max_range)
        ax3d_static.set_xlabel('x [km]'); ax3d_static.set_ylabel('y [km]'); ax3d_static.set_zlabel('z [km]')
        ax3d_static.set_title('ECI trajectories (3D) — static')
        ax3d_static.legend(loc='upper right', fontsize=8)

        # =======================================
        # (TR) 3D animation — FAST (blitting)
        # =======================================
        # Gather series (downsampled)
        series3d = [("GS:"+gs_key, Rg[::step])]
        for k, S in sats.items():
            series3d.append(("SAT:"+k, S.R[::step]))
        all_R3d = np.vstack([R for _, R in series3d])
        xs3d, ys3d, zs3d = all_R3d[:,0], all_R3d[:,1], all_R3d[:,2]

        if show_earth:
            # low-res Earth, drawn once (no shading, antialias off)
            Re = globals().get('R_EARTH', 6378.1363)
            u2 = np.linspace(0, 2*np.pi, 36)
            v2 = np.linspace(0, np.pi, 18)
            xsE = Re * np.outer(np.cos(u2), np.sin(v2))
            ysE = Re * np.outer(np.sin(u2), np.sin(v2))
            zsE = Re * np.outer(np.ones_like(u2), np.cos(v2))
            ax3d_anim.plot_surface(xsE, ysE, zsE,
                                rstride=2, cstride=2,
                                alpha=0.10, linewidth=0,
                                antialiased=False, shade=False, zorder=0)

        lines3d, pts3d = [], []
        for name, _ in series3d:
            ln, = ax3d_anim.plot([], [], [], lw=1.2, label=name, animated=True)
            pt   = ax3d_anim.plot([], [], [], marker='o', markersize=6, animated=True)[0]
            lines3d.append(ln); pts3d.append(pt)

        max_range3d = np.array([xs3d.max()-xs3d.min(),
                                ys3d.max()-ys3d.min(),
                                zs3d.max()-zs3d.min()]).max()
        mid3d = np.array([xs3d.mean(), ys3d.mean(), zs3d.mean()])
        ax3d_anim.set_xlim(mid3d[0]-0.5*max_range3d, mid3d[0]+0.5*max_range3d)
        ax3d_anim.set_ylim(mid3d[1]-0.5*max_range3d, mid3d[1]+0.5*max_range3d)
        ax3d_anim.set_zlim(mid3d[2]-0.4*max_range3d, mid3d[2]+0.4*max_range3d)
        ax3d_anim.set_xlabel('x [km]'); ax3d_anim.set_ylabel('y [km]'); ax3d_anim.set_zlabel('z [km]')
        ax3d_anim.set_title('ECI trajectories — animated')
        ax3d_anim.legend(loc='upper right', fontsize=8)
        ax3d_anim.grid(False)

        n_frames_3d = min(R.shape[0] for _, R in series3d)
        if tail3d is None or tail3d <= 0 or tail3d > n_frames_3d:
            tail3d = n_frames_3d

        # =======================================
        # (BL) Static sky plot (with NaN segs)
        # =======================================
        axsky_static.set_theta_zero_location("N")
        axsky_static.set_theta_direction(-1)
        axsky_static.set_rlim(90, 0)  # radius = elevation: zenith at centre
        axsky_static.set_thetagrids(range(0, 360, 30))
        axsky_static.set_rgrids([0, 15, 30, 45, 60, 75, 90], angle=135)
        axsky_static.set_title(f"Sky track (static) from GS:{gs_key} (elev ≥ {min_elev_deg:.0f}°)")

        # =======================================
        # (BR) Animated sky plot (with NaN segs)
        # =======================================
        axsky_anim.set_theta_zero_location("N")
        axsky_anim.set_theta_direction(-1)
        axsky_anim.set_rlim(90, 0)
        axsky_anim.set_thetagrids(range(0, 360, 30))
        axsky_anim.set_rgrids([0, 15, 30, 45, 60, 75, 90], angle=135)
        axsky_anim.set_title(f"Sky track (animated) from GS:{gs_key} (elev ≥ {min_elev_deg:.0f}°)")

        # Precompute sky tracks (segmented) for both static & animated
        tracks = []   # for animation: dicts with arrays/labels
        n_frames_sky = None
        for k, S in sats.items():
            az_deg_full, el_deg_full, _ = self._compute_az_el(gs_traj, S)

            # Static (all samples, NaN-segmented)
            th_plot_s, r_plot_s, vis_s, _, _ = _segmented_polar_arrays(az_deg_full, el_deg_full, min_elev_deg)
            axsky_static.plot(th_plot_s, r_plot_s, lw=1.4, label=f"SAT:{k}")
            if np.any(vis_s):
                idx = np.where(vis_s)[0]
                axsky_static.plot([np.radians(az_deg_full[idx[0]])], [el_deg_full[idx[0]]], marker='o', ms=4)
                axsky_static.plot([np.radians(az_deg_full[idx[-1]])], [el_deg_full[idx[-1]]], marker='s', ms=4)

            # Animated (step-subsampled, NaN-segmented)
            az_deg = az_deg_full[::step]
            el_deg = el_deg_full[::step]
            th_plot, r_plot, vis, theta_raw, r_raw = _segmented_polar_arrays(az_deg, el_deg, min_elev_deg)
            tracks.append({
                "name": f"SAT:{k}",
                "theta_plot": th_plot,
                "r_plot": r_plot,
                "vis": vis,
                "theta_raw": theta_raw,
                "r_raw": r_raw
            })
            n_frames_sky = len(theta_raw) if n_frames_sky is None else min(n_frames_sky, len(theta_raw))

        axsky_static.legend(loc='upper right', bbox_to_anchor=(1.20, 1.10), fontsize=8)

        # Artists for animated sky (animated=True for blit)
        sky_lines, sky_heads = [], []
        for tr in tracks:
            ln, = axsky_anim.plot([], [], lw=1.5, label=tr["name"], animated=True)
            hd, = axsky_anim.plot([], [], marker='o', ms=4, animated=True)
            sky_lines.append(ln); sky_heads.append(hd)
        axsky_anim.legend(loc='upper right', bbox_to_anchor=(1.20, 1.10), fontsize=8)

        # ============================
        # Combined animation
        # ============================
        n_frames = min(n_frames_3d, n_frames_sky) if n_frames_sky is not None else n_frames_3d
        if tailsky is None or tailsky <= 0 or tailsky > n_frames:
            tailsky = n_frames

        def init():
            for ln, pt in zip(lines3d, pts3d):
                ln.set_data([], []); ln.set_3d_properties([])
                pt.set_data([], []); pt.set_3d_properties([])
            for ln, hd in zip(sky_lines, sky_heads):
                ln.set_data([], []); hd.set_data([], [])
            return (*lines3d, *pts3d, *sky_lines, *sky_heads)

        def update(frame: int):
            # 3D (fixed camera; only update artists)
            for i, (_, R) in enumerate(series3d):
                i0 = frame - tail3d + 1
                if i0 < 0: i0 = 0
                seg = R[i0:frame+1]
                lines3d[i].set_data(seg[:,0], seg[:,1])
                lines3d[i].set_3d_properties(seg[:,2])
                pts3d[i].set_data([R[frame,0]], [R[frame,1]])
                pts3d[i].set_3d_properties([R[frame,2]])
            # optional camera spin disables blitting benefits; keep False for speed
            if camera_spin:
                ax3d_anim.view_init(elev=ax3d_anim.elev, azim=(ax3d_anim.azim + 0.5) % 360.0)

            # Sky (NaN-segmented trail; head only if visible)
            i0s = frame - tailsky + 1
            if i0s < 0: i0s = 0
            for i, tr in enumerate(tracks):
                th_seg = tr["theta_plot"][i0s:frame+1]
                r_seg  = tr["r_plot"][i0s:frame+1]
                sky_lines[i].set_data(th_seg, r_seg)
                if tr["vis"][frame]:
                    sky_heads[i].set_data([tr["theta_raw"][frame]], [tr["r_raw"][frame]])
                else:
                    sky_heads[i].set_data([], [])
            return (*lines3d, *pts3d, *sky_lines, *sky_heads)

        # Try blitting when camera is fixed; fall back if backend/3D can't blit
        use_blit = not camera_spin
        try:
            ani = animation.FuncAnimation(
                fig, update, frames=n_frames, init_func=init,
                interval=interval, blit=use_blit, repeat=repeat, cache_frame_data=False
            )
        except Exception:
            ani = animation.FuncAnimation(
                fig, update, frames=n_frames, init_func=init,
                interval=interval, blit=False, repeat=repeat, cache_frame_data=False
            )

        if save_path:
            ext = save_path.lower().split('.')[-1]
            if ext == 'mp4':
                writer = animation.FFMpegWriter(fps=int(1000/interval), bitrate=1800)
                ani.save(save_path, writer=writer, dpi=dpi)
            elif ext in ('gif', 'agif'):
                writer = animation.PillowWriter(fps=int(1000/interval))
                ani.save(save_path, writer=writer, dpi=dpi)
            else:
                print("Unsupported extension for save_path; showing instead.")

        plt.show()
        return fig, ani

    def elevation_series(self,
                        gs_key: str,
                        sat_key: str,
                        min_elev_deg: float = 0.0,
                        return_az: bool = False,
                        *,
                        plot: bool = True,
                        fig_size: tuple[float, float] = (10, 6),
                        save_plot_path: str | None = None):
        """
        Compute the satellite's elevation at each simulator time step as seen from a
        given ground station, and report the percentage of time it is visible.

        Also plots (if plot=True) a figure with two subplots sharing the x-axis:
        - Top: elevation [deg] with the visibility threshold line.
        - Bottom: visibility mask (0/1) as a step plot.

        Returns:
            jd (np.ndarray): (n,) Julian Dates used by the simulator.
            elevation_deg (np.ndarray): (n,) elevation in degrees.
            visible_mask (np.ndarray): (n,) boolean array where elevation >= min_elev_deg.
            percent_visible (float): Percentage of samples with elevation >= min_elev_deg, in [0, 100].
            azimuth_deg (np.ndarray, optional): (n,) azimuth in degrees; only returned if return_az=True.
        """
        import numpy as np

        if self.JD is None:
            raise ValueError("Build the timebase and run the propagators before calling elevation_series().")

        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        if sat_key not in self.satellite_trajectories:
            raise KeyError(f"Satellite '{sat_key}' not found")

        gs_traj = self.ground_station_trajectories[gs_key]
        sat_traj = self.satellite_trajectories[sat_key]

        # Compute azimuth/elevation in the GS frame
        az_deg, el_deg, _ = self._compute_az_el(gs_traj, sat_traj)  # el_deg is (n,)

        if el_deg.shape[0] != self.JD.shape[0]:
            raise ValueError("Mismatched timebases: GS, SAT, and simulator JD must have the same length.")

        # Apply minimum elevation threshold for visibility
        visible_mask = el_deg >= float(min_elev_deg)

        # Percentage of visible samples (ignore NaNs if any)
        valid = np.isfinite(el_deg)
        total = int(valid.sum())
        percent_visible = 0.0 if total == 0 else 100.0 * float((visible_mask & valid).sum()) / float(total)

        # -------- Plotting --------
        if plot:
            import matplotlib.pyplot as plt

            jd = self.JD
            y_vis = visible_mask.astype(int)

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=fig_size, sharex=True,
                gridspec_kw={"height_ratios": [3, 1]}
            )

            # Top: elevation with threshold
            ax1.plot(jd, el_deg, lw=1.2)
            ax1.axhline(min_elev_deg, linestyle="--", linewidth=1.0)
            ax1.set_ylabel("Elevation [deg]")
            ax1.set_title(f"Elevation and Visibility (min elev {min_elev_deg:.1f}°) — {gs_key} vs {sat_key}")
            ax1.grid(True, alpha=0.3)

            # Bottom: visibility mask (0/1) as step
            ax2.step(jd, y_vis, where="post")
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_yticks([0, 1], labels=["Hidden", "Visible"])
            ax2.set_xlabel("Julian Date")
            ax2.set_ylabel("Vis")
            ax2.grid(True, axis="y", alpha=0.3)

            fig.tight_layout()

            if save_plot_path:
                try:
                    fig.savefig(save_plot_path, dpi=160, bbox_inches="tight")
                except Exception as exc:
                    print(f"[warn] could not save plot to '{save_plot_path}': {exc}")

            plt.show()

        # -------- Returns --------
        if return_az:
            return self.JD.copy(), el_deg.copy(), visible_mask, percent_visible, az_deg.copy()
        else:
            return self.JD.copy(), el_deg.copy(), visible_mask, percent_visible
