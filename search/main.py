from Satellite import Satellite
from utils import *
from Propagator import PropagatorSciPy
from Simulator import Simulator

def main():
    # Example initial state (km, km/s)
    r0 = [7000.0, 0.0, 0.0]
    v0 = [0.0, 7.5, 1.0]

    sat = Satellite.from_state_vector(r0, v0)
    prop = PropagatorSciPy(method="DOP853", rtol=1e-8, atol=1e-10, max_step=np.inf)
    sim  = Simulator(prop)

    traj = sim.run(sat, t_days=1.0, sample_dt=60.0, append_final_state=True)
    traj.save_npz("traj_day1.npz")

    # Plots
    traj.plot_r_components()
    traj.plot_ground_track()

    # 3D static
    traj.plot_3d(show_earth=True)

    # 3D animation, quick preview
    traj.animate_3d(step=2, interval=20, tail=800, camera_spin=True)

    # Save to mp4 (needs ffmpeg installed)
    traj.animate_3d(step=3, interval=30, tail=600, save_path="orbit.mp4")

if __name__ == "__main__":
    main()