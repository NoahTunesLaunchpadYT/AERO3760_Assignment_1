# --- simulator.py ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helper.time import gmst_from_jd, calendar_to_jd
from tqdm.auto import tqdm  # at top of Simulator.py (or local import inside function)
from helper.coordinate_transforms import az_el_sat_from_gs
from helper.plotting import set_axis_limits_from_points
import spacetools

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
        jd0 = calendar_to_jd(Y, M, D, h, m, s)
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
                JDg, Rg, Vg, E_eci, N_eci, U_eci = self.gs_prop.propagate(gs, JD)
                   
                
                gtraj = GroundStationTrajectory(
                    JDg, Rg, Vg, self.start_time or (0, 0, 0, 0, 0, 0.0),
                    name=f"{k}", enu_vectors=U_eci
                )

                # optional caches
                gtraj.E_eci = E_eci
                gtraj.N_eci = N_eci
                gtraj.U_eci = U_eci
                self.ground_station_trajectories[k] = gtraj

                if pbar:
                    pbar.set_postfix_str(f"{k}")
                    pbar.update(1)

            # Satellites
            for k in s_keys:
                sat = self.satellites[k]
                start_jd = sat.start_jd
                if start_jd is None:
                    start_jd = JD[0]
                
                # indices where this sat is active
                active_mask = JD >= start_jd
                if not np.any(active_mask):
                    # nothing to do for this sat on this timebase
                    continue

                # Propagate only over the active tail
                JDa = JD[active_mask]
                JDa, Ra, Va = self.sat_prop.propagate(sat, JDa)  # unchanged propagator

                # Stitch back into global arrays
                n = JD.size
                R = np.full((n, 3), np.nan)
                V = np.full((n, 3), np.nan)
                R[active_mask] = Ra
                V[active_mask] = Va
                
                straj = SatelliteTrajectory(
                    JD, R, V, self.start_time or (0, 0, 0, 0, 0, 0.0),
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
            ax.plot(R[:,0], R[:,1], R[:,2], lw=1.2, label=f"{k}")
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

        set_axis_limits_from_points(ax, all_R)

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
            series.append((""+k, self.ground_station_trajectories[k].R[::step]))
        for k in s_keys:
            series.append(("SAT:"+k, self.satellite_trajectories[k].R[::step]))
        if not series:
            raise RuntimeError("No trajectories to animate. Run the simulator first.")
        
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

        # Determine axis limits from all paths
        all_R = np.vstack([R for _, R in series])

        set_axis_limits_from_points(ax, [all_R])

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
        ax.set_title(f"Sky track (animated) from {gs_key}  (min elev ≥ {min_elev_deg:.0f}°)")

        gs_traj = self.ground_station_trajectories[gs_key]
        tracks = []
        n_frames = None

        for k in sat_keys:
            S = self.satellite_trajectories[k]
            az_full, el_full, _, dist_full = az_el_sat_from_gs(gs_traj, S, return_distance=True)
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
        ax.set_title(f"Sky track (static) from {gs_key}  (min elev ≥ {min_elev_deg:.0f}°)")

        gs_traj = self.ground_station_trajectories[gs_key]
        
        
        for k in sat_keys:
            S = self.satellite_trajectories[k]
            az_deg, el_deg, _, dist_km = az_el_sat_from_gs(gs_traj, S, return_distance=True)

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

        series3d = [(""+gs_key, gs_traj.R[::step])]
        for k, S in sats.items():
            series3d.append(("SAT:"+k, S.R[::step]))
        all_R3d = np.vstack([R for _, R in series3d])

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

        set_axis_limits_from_points(ax, [all_R3d])

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
        if gs_key not in self.ground_station_trajectories.keys():
            raise KeyError(f"Ground station '{gs_key}' not found")
        for k in sat_keys:
            if k not in self.satellite_trajectories:
                print(self.satellite_trajectories.keys())
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
        ax.plot(Rg[:,0], Rg[:,1], Rg[:,2], lw=1.2, label=f"{gs_key}")
        ax.scatter(Rg[0,0], Rg[0,1], Rg[0,2], s=30)
        # ax.scatter(Rg[-1,0], Rg[-1,1], Rg[-1,2], s=30)

        all_R = [Rg]
        for k, S in sats.items():
            R = S.R
            ax.plot(R[:,0], R[:,1], R[:,2], lw=1.2, label=f"SAT:{k}")
            ax.scatter(R[0,0], R[0,1], R[0,2], s=30, label=f"Start:{k}")
            # ax.scatter(R[-1,0], R[-1,1], R[-1,2], s=30)
            all_R.append(R)

        all_R = np.vstack(all_R)

        set_axis_limits_from_points(ax, [all_R])
        
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

    def plot_sat_az_el_rng_vis(
        self,
        gs_key: str,
        sat_keys: list[str],
        *,
        min_elev_deg: float = 0.0,
        max_distance_km: float | None = None,
        max_elev_deg: float | None = None,
        show_azimuth: bool = True,
        show_elevation: bool = True,
        show_range: bool = True,
        show_visibility: bool = True,
        fig_size: tuple[float, float] = (6, 5),
        block: bool = True,
        save_plot_path: str | None = None,
    ):
        """
        Compute azimuth/elevation/range for multiple satellites from one ground station,
        derive visibility from (elev >= min_elev_deg) & (range <= max_distance_km if provided),
        then render up to 4 subplots with persistent per-satellite colours.

        Returns
        -------
        fig : matplotlib.figure.Figure
        out : dict
            {
            'x': (n,),
            'per_sat': {
                sk: {
                    'azimuth_deg': (n,),
                    'elevation_deg': (n,),
                    'range_km': (n,),
                    'visible': (n,) bool
                }, ...
            },
            'overall_visible': (n,) bool,
            'overall_visible_pct': float in [0, 100],
            'per_sat_visible_pct': { sk: float pct, ... },
            'colours': { sk: RGBA, ... }
            }
        """
        # ---- 1) Compute all series + visibility ----
        x, per_sat, overall_vis, per_sat_pct, overall_pct = self._compute_series_and_visibility(
            gs_key=gs_key,
            sat_keys=sat_keys,
            min_elev_deg=min_elev_deg,
            max_distance_km=max_distance_km,
        )

        # ---- 2) Colour map (persistent across calls) ----
        colours = self._get_sat_color_map(list(per_sat.keys()))

        # ---- 3) Decide which panels to draw ----
        panels = []
        if show_azimuth:    panels.append("az")
        if show_elevation:  panels.append("el")
        if show_range:      panels.append("rng")
        if show_visibility: panels.append("vis")
        if not panels:
            raise ValueError("At least one of the panels must be enabled.")

        fig, axes = plt.subplots(len(panels), 1, figsize=fig_size, sharex=True)
        if len(panels) == 1:
            axes = [axes]

        # ---- 4) Draw subplots via helpers ----
        idx = 0
        if "az" in panels:
            _subplot_azimuth(axes[idx], x, per_sat, colours, gs_key)
            idx += 1
        if "el" in panels:
            _subplot_elevation(axes[idx], x, per_sat, colours, gs_key,
                            min_elev_deg=min_elev_deg, max_elev_deg=max_elev_deg)
            idx += 1
        if "rng" in panels:
            _subplot_range(axes[idx], x, per_sat, colours, gs_key,
                        max_distance_km=max_distance_km)
            idx += 1
        if "vis" in panels:
            _subplot_visibility(axes[idx], x, per_sat, colours, gs_key,
                                overall_pct=overall_pct)
            idx += 1

        axes[-1].set_xlabel("Julian Date" if getattr(self, "JD", None) is not None else "Sample")

        fig.tight_layout()
        if save_plot_path:
            try:
                fig.savefig(save_plot_path, dpi=160, bbox_inches="tight")
            except Exception as exc:
                print(f"[warn] could not save plot to '{save_plot_path}': {exc}")

        if block:
            plt.show(block=True)

        return fig, {
            "x": x,
            "per_sat": per_sat,
            "overall_visible": overall_vis,
            "overall_visible_pct": overall_pct,
            "per_sat_visible_pct": per_sat_pct,
            "colours": colours,
        }

    def _compute_series_and_visibility(
        self,
        *,
        gs_key: str,
        sat_keys: list[str],
        min_elev_deg: float,
        max_distance_km: float | None,
    ):
        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found.")
        sat_keys = list(sat_keys)
        if not sat_keys:
            raise ValueError("sat_keys must be a non-empty iterable.")

        for sk in sat_keys:
            if sk not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{sk}' not found.")

        gs_traj = self.ground_station_trajectories[gs_key]

        # Use JD if available, else sample index after first compute
        if getattr(self, "JD", None) is not None:
            x = np.asarray(self.JD)
        else:
            x = None

        per_sat: dict[str, dict[str, np.ndarray]] = {}
        for sk in sat_keys:
            sat_traj = self.satellite_trajectories[sk]
            az_deg, el_deg, _, rng_km = az_el_sat_from_gs(gs_traj, sat_traj, return_distance=True)

            if x is None:
                x = np.arange(len(az_deg), dtype=float)

            az = np.asarray(az_deg, dtype=float)
            el = np.asarray(el_deg, dtype=float)
            rn = np.asarray(rng_km, dtype=float)

            # Visibility: elevation and range criteria
            valid = np.isfinite(el) & np.isfinite(rn)
            vis = valid & (el >= float(min_elev_deg))
            if max_distance_km is not None and np.isfinite(max_distance_km):
                vis &= (rn <= float(max_distance_km))

            per_sat[sk] = {
                "azimuth_deg": az,
                "elevation_deg": el,
                "range_km": rn,
                "visible": vis.astype(bool),
            }

        # Overall visibility (any sat visible at each sample)
        vis_stack = np.vstack([per_sat[sk]["visible"] for sk in sat_keys])  # (S, n)
        overall_vis = np.any(vis_stack, axis=0)
        n = overall_vis.size
        overall_pct = 100.0 * float(np.count_nonzero(overall_vis)) / float(n) if n else 0.0

        per_sat_pct = {
            sk: 100.0 * float(np.count_nonzero(per_sat[sk]["visible"])) / float(n) if n else 0.0
            for sk in sat_keys
        }

        return np.asarray(x, dtype=float), per_sat, overall_vis, per_sat_pct, overall_pct

    def _get_sat_color_map(self, sat_keys: list[str]):
        """
        Returns a persistent colour map {sat_key: rgba}. New satellites encountered
        are assigned the next colour from tab20 and stored on self for future calls.
        """
        if not hasattr(self, "_sat_color_map"):
            self._sat_color_map = {}

        cmap = plt.get_cmap("tab20")
        # Start from how many we’ve already assigned
        next_idx = len(self._sat_color_map)

        for sk in sat_keys:
            if sk not in self._sat_color_map:
                self._sat_color_map[sk] = cmap(next_idx % 20)
                next_idx += 1

        # Return only the colours for requested satellites (preserving the map object)
        return {sk: self._sat_color_map[sk] for sk in sat_keys}

    def spacetools_ground_track(self, sat_key: str,
                            *, step: int = 1, figsize=(10, 6), block: bool = True,
                            r: float | None = None):

        # normalise longitude to [0, 360)
        def _wrap360(lon_deg):
            lon = (np.asarray(lon_deg) % 360.0 + 360.0) % 360.0
            return lon
        
        # --- helper: ECI->ECEF->(lat,lon) for a trajectory S ---
        def _eci_to_latlon(traj):
            x, y, z = traj.R[::step, 0], traj.R[::step, 1], traj.R[::step, 2]
            theta = gmst_from_jd(self.JD[::step])               # array of angles [rad]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x_e =  cos_t * x + sin_t * y
            y_e = -sin_t * x + cos_t * y
            z_e =  z
            rnorm = np.linalg.norm(np.stack([x_e, y_e, z_e], axis=-1), axis=-1)
            lat = np.degrees(np.arcsin(z_e / rnorm))
            lon = np.degrees(np.arctan2(y_e, x_e))
            lon = _wrap360(lon)          # 0..360
            lon = (lon + 180)%360 - 180
            return lat, lon

        S = self.satellite_trajectories[sat_key]
        lat, lon = _eci_to_latlon(S)
        latlon = np.array(
            [lat, lon]
        )

        fig_groundtrack = plt.figure(figsize=figsize)
        ax_groundtrack = fig_groundtrack.add_subplot(111)

        spacetools.groundtrack(
            ax_groundtrack,
            latlon,
            arrows=True,                # True by default
            arrow_interval=3000,        # Interval between arrows in km, 3000 by default
            # Optional:
            arrow_kwargs={
                'linewidth': 1,         # Control arrow thickness
                'arrowstyle': '-|>',    # Arrow style, default is '-|>'. Try out '->'
                'mutation_scale': 10,   # Scale the size of the arrowhead (quite sensitive)
            },
            # Matplotlib keyword arguments:
            color='red',
            label=f'{sat_key}',
        )

        plt.title(f"Ground Track of Satellite '{sat_key}'")

        plt.show()

        return

    def plot_ground_tracks_window(self, gs_key: str, sat_keys: list[str],
                                *, step: int = 1, figsize=(10, 6), block: bool = True,
                                r: float | None = None):
        """
        Ground tracks (0..360° longitudes; 0° on the left).
        - Marks the ground station with a star (at its geodetic lon/lat).
        - Each satellite has a unique colour.
        - Handles lon wrap (0..360) with NaN segmentation to avoid spurious lines.
        - If r (degrees) is provided, draws a translucent geodesic visibility circle.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if gs_key not in self.ground_station_trajectories:
            raise KeyError(f"Ground station '{gs_key}' not found")
        for k in sat_keys:
            if k not in self.satellite_trajectories:
                raise KeyError(f"Satellite '{k}' not found")

        # normalise longitude to [0, 360)
        def _wrap360(lon_deg):
            lon = (np.asarray(lon_deg) % 360.0 + 360.0) % 360.0
            return lon

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # --- helper: ECI->ECEF->(lat,lon) for a trajectory S ---
        def _eci_to_latlon(traj):
            x, y, z = traj.R[::step, 0], traj.R[::step, 1], traj.R[::step, 2]
            theta = gmst_from_jd(self.JD[::step])               # array of angles [rad]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x_e =  cos_t * x + sin_t * y
            y_e = -sin_t * x + cos_t * y
            z_e =  z
            rnorm = np.linalg.norm(np.stack([x_e, y_e, z_e], axis=-1), axis=-1)
            lat = np.degrees(np.arcsin(z_e / rnorm))
            lon = np.degrees(np.arctan2(y_e, x_e))
            lon = _wrap360(lon)          # 0..360
            return lon, lat

        # --- plot each satellite with wrap-safe segmentation (0..360) ---
        def _plot_wrapped(lon, lat, label):
            lon = np.asarray(lon); lat = np.asarray(lat)
            dlon = np.diff(lon)
            # break line when we jump across 0/360 edge
            breaks = np.where(np.abs(dlon) > 180.0)[0] + 1
            lon_seg = lon.astype(float).copy()
            lat_seg = lat.astype(float).copy()
            lon_seg[breaks] = np.nan
            lat_seg[breaks] = np.nan
            ln, = ax.plot(lon_seg, lat_seg, lw=1.5, label=label)
            ax.plot([lon[-1]], [lat[-1]], marker='o', ms=5,
                    color=ln.get_color(), linestyle='None')
            return ln

        # Ground station star (use its geodetic lat/lon)
        gs = (self.ground_stations[gs_key]
            if hasattr(self, "ground_stations") else
            self.ground_station_trajectories[gs_key])
        gs_lat = float(getattr(gs, "lat_deg", None) if hasattr(gs, "lat_deg") else gs.lat_deg)
        gs_lon = float(getattr(gs, "lon_deg", None) if hasattr(gs, "lon_deg") else gs.lon_deg)
        gs_lon = float(_wrap360(gs_lon))
        ax.plot([gs_lon], [gs_lat], marker='*', ms=12, color='k', label=f"{gs_key}")

        # Optional visibility circle (geodesic, radius r degrees)
        if r is not None and r > 0:
            phi1 = np.radians(gs_lat)
            lam1 = np.radians(gs_lon)
            delta = np.radians(r)
            bearings = np.linspace(0.0, 2.0*np.pi, 361)

            sin_phi1, cos_phi1 = np.sin(phi1), np.cos(phi1)
            sin_d, cos_d = np.sin(delta), np.cos(delta)

            # Great-circle destination formula
            lat_c = np.arcsin(sin_phi1 * cos_d + cos_phi1 * sin_d * np.cos(bearings))
            lon_c = lam1 + np.arctan2(np.sin(bearings) * sin_d * cos_phi1,
                                    cos_d - sin_phi1 * np.sin(lat_c))

            lat_c = np.degrees(lat_c)
            lon_c = _wrap360(np.degrees(lon_c))

            # Split polygon at 0/360 to avoid a long wrap
            dlon = np.diff(lon_c)
            breaks = np.where(np.abs(dlon) > 180.0)[0] + 1
            lon_fill = lon_c.astype(float).copy()
            lat_fill = lat_c.astype(float).copy()
            lon_fill[breaks] = np.nan
            lat_fill[breaks] = np.nan

            ax.fill(lon_fill, lat_fill, alpha=0.12, color='C0',
                    edgecolor='C0', linewidth=1.0, linestyle='--',
                    label=f"Visibility r={r:.1f}°")

        # Satellites
        for k in sat_keys:
            S = self.satellite_trajectories[k]
            lon, lat = _eci_to_latlon(S)
            _plot_wrapped(lon, lat, label=f"SAT:{k}")

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        if len(sat_keys) > 1:
            title = "Ground tracks"
        else:
            title = "Ground track"
        ax.set_title(title)
        ax.set_xlim([0, 360]); ax.set_ylim([-90, 90])
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        plt.show(block=block)
        return fig

def _subplot_azimuth(ax, x, per_sat, colours, gs_key: str):
    """Plot azimuth over time for each satellite.
    """    
    for sk, series in per_sat.items():
        ax.plot(x, series["azimuth_deg"], lw=1.0, color=colours[sk], label=sk)
    ax.set_ylabel("Azimuth [deg]")
    ax.set_title(f"Azimuth over Time from {gs_key}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=min(3, len(per_sat)), loc="upper right")

def _subplot_elevation(ax, x, per_sat, colours, gs_key: str, *, min_elev_deg: float, max_elev_deg: float | None):
    """
    Plot elevation over time for each satellite.
    """
    for sk, series in per_sat.items():
        ax.plot(x, series["elevation_deg"], lw=1.0, color=colours[sk], label=sk)
    # Dotted reference lines
    ax.axhline(float(min_elev_deg), linestyle="--", linewidth=1.0, color="black", label=f"min elev {min_elev_deg:g}°")
    if max_elev_deg is not None and np.isfinite(max_elev_deg):
        ax.axhline(float(max_elev_deg), linestyle="--", linewidth=1.0, color="grey", label=f"max elev {max_elev_deg:g}°")
    ax.set_ylabel("Elevation [deg]")
    ax.set_title(f"Elevation over Time from {gs_key}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=min(3, len(per_sat)), loc="lower right")

def _subplot_range(ax, x, per_sat, colours, gs_key: str, *, max_distance_km: float | None):
    """
    Plot range over time for each satellite.
    """
    for sk, series in per_sat.items():
        rng = series["range_km"]/1000
        vis_mask = series["visible"]

        rng_segmented = rng.copy()
        not_rng_segmented = rng.copy()
        for i in range(len(rng)):
            if not vis_mask[i]:
                # Segment NaN where not visible
                rng_segmented[i] = np.nan
            else:
                not_rng_segmented[i] = np.nan

        # Plot visible portions (full opacity)
        ax.plot(x, rng_segmented,
                lw=2.0, color=colours[sk], label=sk)

        # Plot non-visible portions (faded)
        ax.plot(x, not_rng_segmented,
                lw=1.0, color=colours[sk], alpha=0.4)

    if max_distance_km is not None and np.isfinite(max_distance_km):
        ax.axhline(float(max_distance_km)/1000, linestyle="--", linewidth=1.0, color="black",
                   label=f"max range {max_distance_km:g} km")

    ax.set_ylabel("Range [km] 10^3")
    ax.set_title(f"Range over Time from {gs_key}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=min(3, len(per_sat)), loc="upper right")

def _subplot_visibility(ax, x, per_sat, colours, gs_key: str, *, overall_pct: float):
    """
    Plot visibility over time for each satellite.
    """
    # Raster: rows = satellites, columns = time samples, values = 0/1
    sat_keys = list(per_sat.keys())
    vis_stack = np.vstack([per_sat[sk]["visible"].astype(int) for sk in sat_keys])

    ax.imshow(
        vis_stack,
        aspect="auto",
        interpolation="nearest",
        extent=[x[0], x[-1], -0.5, len(sat_keys) - 0.5],
    )
    ax.set_yticks(range(len(sat_keys)))
    ax.set_yticklabels(sat_keys)
    ax.set_ylabel("Satellite")
    ax.set_title(f"Visibility: {overall_pct:.1f}% of the time")

def _sky_series_for_plot(az_deg, el_deg, dist_km, *, min_elev_deg, distance_thresh_km):
    """
    Build sky-track series for polar plotting.
    Returns theta_solid, r_solid, theta_dotted, r_dotted, vis_mask, theta_raw, r_raw
    where:
    - solid = visible AND dist <= threshold
    - dotted = visible AND dist  > threshold
    Segmentation (NaN) applied ONLY where not visible (el < min_elev_deg).
    No NaNs added for azimuth wraps.
    """
    az_deg = np.asarray(az_deg)
    el_deg = np.asarray(el_deg)
    dist_km = np.asarray(dist_km)

    # polar: theta in radians, radius as elevation (0 at centre = zenith, but we'll invert r-limits)
    theta = np.deg2rad(az_deg)
    r = el_deg

    visible = np.isfinite(el_deg) & (el_deg >= float(min_elev_deg))
    near = visible & (dist_km <= float(distance_thresh_km))
    far  = visible & (dist_km >  float(distance_thresh_km))

    # Apply NaN ONLY where not visible
    theta_solid = theta.copy()
    r_solid = r.copy()
    theta_solid[~near] = np.nan
    r_solid[~near] = np.nan

    theta_dotted = theta.copy()
    r_dotted = r.copy()
    theta_dotted[~far] = np.nan
    r_dotted[~far] = np.nan

    return theta_solid, r_solid, theta_dotted, r_dotted, visible, theta, r