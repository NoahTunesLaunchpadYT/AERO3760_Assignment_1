import numpy as np
import matplotlib.pyplot as plt 
from utils import *

# ----------------- Trajectory -----------------
class Trajectory:
    def __init__(self, jd_array, R_km, V_km_s, start_time):
        """
        jd_array: Julian dates for each state sample
        R_km, V_km_s: position and velocity in km, km/s
        start_time: tuple (Y, M, D, h, m, s) of sim start
        """
        self.JD = np.array(jd_array, float)   # Julian Date at each sample
        self.R = np.array(R_km, float)
        self.V = np.array(V_km_s, float)
        self.start_time = start_time          # Store launch time

    def save_npz(self, path):
        np.savez(path, JD=self.JD, R=self.R, V=self.V, start_time=self.start_time)

    @staticmethod
    def load_npz(path):
        data = np.load(path, allow_pickle=True)
        return Trajectory(data["JD"], data["R"], data["V"], tuple(data["start_time"]))

    def elapsed_hours(self):
        return (self.JD - self.JD[0]) * 24.0
    
    # Simple ECI plots
    def plot_r_components(self):
        plt.figure()
        plt.plot(self.JD/3600.0, self.R[:,0], label='x [km]')
        plt.plot(self.JD/3600.0, self.R[:,1], label='y [km]')
        plt.plot(self.JD/3600.0, self.R[:,2], label='z [km]')
        plt.xlabel('Time [hours]'); plt.ylabel('Position [km]')
        plt.legend(); plt.title('ECI position vs time'); plt.tight_layout(); plt.show()

    # Ground track (assumes Earth as rotating sphere, no nutation/precession)
    def plot_ground_track(self, theta0=0.0):
        # Rotate ECI to ECEF by Earth rotation theta(t) about z
        x, y, z = self.R[:,0], self.R[:,1], self.R[:,2]
        theta = theta0 + OMEGA_E * self.T
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_e =  cos_t * x + sin_t * y
        y_e = -sin_t * x + cos_t * y
        z_e =  z

        r_norm = np.linalg.norm(self.R, axis=1)
        lat = np.degrees(np.arcsin(z_e / r_norm))
        lon = np.degrees(np.arctan2(y_e, x_e))
        # Wrap to [-180, 180]
        lon = (lon + 180.0) % 360.0 - 180.0

        plt.figure()
        plt.plot(lon, lat, '.', ms=1)
        plt.xlabel('Longitude [deg]'); plt.ylabel('Latitude [deg]')
        plt.title('Ground track (ECEF from ECI by z-rotation)')
        plt.xlim([-180, 180]); plt.ylim([-90, 90])
        plt.grid(True); plt.tight_layout(); plt.show()

    def plot_3d(self, show_earth=True, earth_alpha=0.15, earth_wire=True, wire_steps=24):
        """
        Plot the ECI trajectory in 3D with an optional Earth sphere.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        R = self.R
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Orbit path and endpoints
        ax.plot(R[:,0], R[:,1], R[:,2], lw=1.2)
        ax.scatter(R[0,0], R[0,1], R[0,2], s=30)             # start
        ax.scatter(R[-1,0], R[-1,1], R[-1,2], s=30)          # end

        # Optional Earth
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

    def animate_3d(self, step=1, interval=30, tail=500, save_path=None, dpi=120, repeat=True, camera_spin=False):
        """
        Animate the ECI trajectory in 3D.
        - step: use every 'step'-th sample to speed up plotting
        - interval: ms between frames
        - tail: number of recent points to show as a trail (None for full)
        - save_path: if ends with .mp4 or .gif, saves animation (requires ffmpeg or imagemagick)
        - camera_spin: slowly rotate the view while animating
        """
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        R = self.R[::step]
        n = R.shape[0]
        Re = globals().get('R_EARTH', 6378.1363)

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