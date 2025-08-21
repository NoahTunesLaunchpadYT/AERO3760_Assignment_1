import numpy as np
import matplotlib.pyplot as plt 
from helper_time import gmst_from_jd
from helper_constants import R_EARTH
# ----------------- Trajectory -----------------
class Trajectory:
    """
    Generic trajectory container:
      - JD: absolute Julian Dates for each sample (days)
      - R: ECI position [km], shape (n,3)
      - V: ECI velocity [km/s], shape (n,3)
      - start_time: (Y, M, D, h, m, s)
    """

    def __init__(
        self,
        jd_array: list[float] | tuple[float, ...] | np.ndarray,
        R_km: list[list[float]] | np.ndarray,
        V_km_s: list[list[float]] | np.ndarray,
        start_time: tuple[int, int, int, int, int, float],
        name: str | None = None,
    ) -> None:
        self.JD = np.array(jd_array, dtype=float)                  # days
        self.R = np.array(R_km, dtype=float).reshape(-1, 3)        # km
        self.V = np.array(V_km_s, dtype=float).reshape(-1, 3)      # km/s
        self.start_time = start_time
        self.name = name or "trajectory"

        if self.JD.shape[0] != self.R.shape[0] or self.R.shape != self.V.shape:
            raise ValueError("JD, R, V must have same length; R and V must be (n,3).")

    # -------- I/O --------
    def save_npz(self, path: str) -> None:
        np.savez(
            path,
            JD=self.JD,
            R=self.R,
            V=self.V,
            start_time=np.array(self.start_time, dtype=object),
            name=np.array(self.name, dtype=object),
        )

    @classmethod
    def load_npz(cls, path: str):
        data = np.load(path, allow_pickle=True)
        jd = data["JD"]
        R = data["R"]
        V = data["V"]
        start_time = tuple(data["start_time"])
        name = str(data.get("name", "trajectory"))
        return cls(jd, R, V, start_time, name=name)

    # -------- time helpers --------
    def elapsed_days(self) -> np.ndarray:
        return self.JD - self.JD[0]

    def elapsed_hours(self) -> np.ndarray:
        return self.elapsed_days() * 24.0

    # -------- plots --------
    def plot_r_components(self) -> None:
        t_hr = self.elapsed_hours()
        plt.figure()
        plt.plot(t_hr, self.R[:, 0], label="x [km]")
        plt.plot(t_hr, self.R[:, 1], label="y [km]")
        plt.plot(t_hr, self.R[:, 2], label="z [km]")
        plt.xlabel("Time [hours]"); plt.ylabel("Position [km]")
        plt.legend(); plt.title(f"ECI position vs time — {self.name}")
        plt.tight_layout(); plt.show()

    def plot_ground_track(self, block: bool = True) -> None:
        """
        Approx ground track: ECI -> ECEF by z-rotation with Earth's mean spin.
        (For high precision, replace with GMST/temporal model.)
        """
        x, y, z = self.R[:, 0], self.R[:, 1], self.R[:, 2]
        theta = gmst_from_jd(self.JD)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        x_e =  cos_t * x + sin_t * y
        y_e = -sin_t * x + cos_t * y
        z_e =  z

        r_norm = np.linalg.norm(self.R, axis=1)
        lat = np.degrees(np.arcsin(z_e / r_norm))
        lon = np.degrees(np.arctan2(y_e, x_e))
        lon = (lon + 180.0) % 360.0 - 180.0

        plt.figure()
        plt.plot(lon, lat, ".", ms=1)
        plt.xlabel("Longitude [deg]"); plt.ylabel("Latitude [deg]")
        plt.title(f"Ground track (approx) — {self.name}")
        plt.xlim([-180, 180]); plt.ylim([-90, 90])
        plt.grid(True); plt.tight_layout(); 
        plt.show(block=block)

    def plot_3d(
        self,
        show_earth: bool = True,
        earth_alpha: float = 0.15,
        earth_wire: bool = True,
        wire_steps: int = 24,
    ) -> None:
        """
        Plot the ECI trajectory in 3D with an optional Earth sphere.
        """
        R = self.R
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Orbit path and endpoints
        ax.plot(R[:,0], R[:,1], R[:,2], lw=1.2)
        ax.scatter(R[0,0], R[0,1], R[0,2], s=30)             # start
        ax.scatter(R[-1,0], R[-1,1], R[-1,2], s=30)          # end

        # Optional Earth
        if show_earth:
            Re = R_EARTH
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

        # Equal aspect
        xs, ys, zs = R[:,0], R[:,1], R[:,2]
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
        mid = np.array([xs.mean(), ys.mean(), zs.mean()])

        # Set limits
        ax.set_xlim(mid[0]-0.5*max_range, mid[0]+0.5*max_range)
        ax.set_ylim(mid[1]-0.5*max_range, mid[1]+0.5*max_range)
        ax.set_zlim(mid[2]-0.4*max_range, mid[2]+0.4*max_range)

        ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
        ax.set_title('ECI trajectory (3D)')
        plt.tight_layout()
        plt.show()

    def animate_3d(
        self,
        step: int = 1,
        interval: int = 30,
        tail: int | None = 500,
        save_path: str | None = None,
        dpi: int = 120,
        repeat: bool = True,
        camera_spin: bool = False,
    ):
        """
        Animate the ECI trajectory in 3D.
        - step: use every 'step'-th sample to speed up plotting
        - interval: ms between frames
        - tail: number of recent points to show as a trail (None for full)
        - save_path: if ends with .mp4 or .gif, saves animation (requires ffmpeg or imagemagick)
        - camera_spin: slowly rotate the view while animating
        """
        import matplotlib.animation as animation

        R = self.R[::step]
        n = R.shape[0]
        Re = R_EARTH

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Earth
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = Re * np.outer(np.cos(u), np.sin(v))
        ys = Re * np.outer(np.sin(u), np.sin(v))
        zs = Re * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, alpha=0.12, linewidth=0, shade=True)

        # Plot containers
        line, = ax.plot([], [], [], lw=1.5)
        sat  = ax.plot([], [], [], marker='o', markersize=4)[0]

        # Equal aspect
        xs, ys, zs = R[:,0], R[:,1], R[:,2]
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
        mid = np.array([xs.mean(), ys.mean(), zs.mean()])
        ax.set_xlim(mid[0]-0.5*max_range, mid[0]+0.5*max_range)
        ax.set_ylim(mid[1]-0.5*max_range, mid[1]+0.5*max_range)
        ax.set_zlim(mid[2]-0.4*max_range, mid[2]+0.4*max_range)
        ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
        ax.set_title('ECI trajectory (animated)')

        # Trail length
        if tail is None or tail <= 0:
            tail = n

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            sat.set_data([], [])
            sat.set_3d_properties([])
            return line, sat

        def update(frame):
            i0 = max(0, frame - tail)
            seg = R[i0:frame+1]
            line.set_data(seg[:,0], seg[:,1])
            line.set_3d_properties(seg[:,2])
            sat.set_data([R[frame,0]], [R[frame,1]])
            sat.set_3d_properties([R[frame,2]])

            if camera_spin:
                # rotate azimuth a bit each frame
                az = (frame * 0.5) % 360.0
                ax.view_init(elev=25, azim=az)
            return line, sat

        ani = animation.FuncAnimation(fig, update, frames=n, init_func=init,
                                      interval=interval, blit=True, repeat=repeat)

        if save_path is not None:
            # Choose writer based on extension
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
        return ani

class SatelliteTrajectory(Trajectory):
    """Trajectory for a satellite in ECI coordinates."""
    def __init__(
        self,
        jd_array: list[float] | tuple[float, ...] | np.ndarray,
        R_km: list[list[float]] | np.ndarray,
        V_km_s: list[list[float]] | np.ndarray,
        start_time: tuple[int, int, int, int, int, float],
        name: str | None = None,
    ) -> None:
        super().__init__(jd_array, R_km, V_km_s, start_time, name or "satellite")

class GroundStationTrajectory(Trajectory):
    """Trajectory for a ground station, optionally with precomputed ENU vectors."""
    def __init__(
        self,
        jd_array: list[float] | tuple[float, ...] | np.ndarray,
        R_km: list[list[float]] | np.ndarray,
        V_km_s: list[list[float]] | np.ndarray,
        start_time: tuple[int, int, int, int, int, float],
        name: str | None = None,
        enu_vectors: list[list[float]] | np.ndarray | None = None,
    ) -> None:
        super().__init__(jd_array, R_km, V_km_s, start_time, name or "ground_station")
        self.enu_vectors = None
        if enu_vectors is not None:
            self.enu_vectors = np.array(enu_vectors, dtype=float).reshape(-1, 3)

    def has_enu(self) -> bool:
        """Return True if ENU vectors have been computed and stored."""
        return self.enu_vectors is not None
