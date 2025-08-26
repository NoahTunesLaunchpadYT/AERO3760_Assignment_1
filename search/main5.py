from Satellite import Satellite
from GroundStation import GroundStation
from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
import numpy as np
from Orbit_Determination import SatelliteLaserRanging, RadiometricTracking, OpticalTracking, OrbitDeterminationSolver
from helper.time import calendar_to_jd
from helper.coordinate_transforms import enu_matrix

def main():
    print(enu_matrix(-33.9, 151.7))

    # Calendar Day
    Y = 2026
    M = 0
    D = 0
    # Time in Universal Time
    h = 0
    m = 0
    s = 0.0

    min_elev_deg = 30.0
    max_distance_km = 7500.0

    # Ground station: Sydney, ~50 m altitude
    gs = GroundStation(lat_deg=-33.9, lon_deg=151.7, h_m=1000.0)

    sat0 = Satellite.from_keplerian(
        e=0.0,
        a_km=11504.0,
        inc_deg=49.0,
        aop_deg=270.0,
        raan_deg=0.0,
        ta_deg=-45.0
    )

    sat1 = Satellite.from_state_vector(
        r_km=np.array([-1590.99025767, 8267.02788189, 3181.98051534]),
        v_km_s=np.array([-6.16934406, -2.13712347,  2.46773762])
    )

    sat2 = Satellite.from_state_vector(
        r_km=np.array([-9569.76166, -9410.00357,  8406.85572]),
        v_km_s=np.array([4.08650725, 4.06543281, -3.62177518])
    )

    # Propagators
    sat_prop = SatellitePropagator(method="RK45", rtol=1e-9, atol=1e-11, max_step=120.0)
    gs_prop  = GroundStationPropagator()

    # Empty Simulator
    sim = Simulator({"sat0": sat0, "sat1": sat1, "sat2": sat2}, {"NewCastle": gs}, sat_prop, gs_prop)

    # Start: 2026-01-11 12:00:00 UTC, run 2 hours at 30 s steps
    sim.build_timebase(Y=Y, M=M, D=D, h=h, m=m, s=s,
                       tf_days=1/24/60*20, sample_dt_s=3.0)

    sim.run_all()  # runs both propagators over the shared JD

    sim.plot_sat_az_el_rng_vis( gs_key="NewCastle", 
                                sat_keys=["sat0", "sat1", "sat2"], 
                                show_azimuth=True,
                                show_elevation=True,
                                show_range=True,
                                show_visibility=True,
                                min_elev_deg=min_elev_deg,
                                max_distance_km=max_distance_km,
                                    )

    sim.plot_all_five(gs_key="NewCastle", sat_keys=["sat0", "sat1", "sat2"], step=10)

    # # Orbit determination stuff
    # sat0_traj = sim.satellite_trajectories["sat0"]
    # gs_traj = sim.ground_station_trajectories["NewCastle"]

    # # Create Sensors
    # range_sensor = SatelliteLaserRanging(id="range_sensor", range_error=5.0, angular_error=0.1, time_error=0.01)
    # radio_sensor = RadiometricTracking(id="radio_sensor", range_error=5.0, angular_error=0.1, time_error=0.01)
    # optical_sensor = OpticalTracking(id="optical_sensor", range_error=5.0, angular_error=0.1, time_error=0.01)

    # # Pick times
    # jd_time1 = calendar_to_jd(Y, M, D, h, m + 10, s)
    # jd_time2 = calendar_to_jd(Y, M, D, h, m + 11, s)
    # jd_time3 = calendar_to_jd(Y, M, D, h, m + 12, s)

    # # Collect observations
    # range_obs = range_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))
    # radio_obs = radio_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2))
    # optical_obs = optical_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))

    # od_solver = OrbitDeterminationSolver()
    # predict_sat1 = od_solver.determine_orbit(gs, observables=range_obs, method="Gibbs")
    # predict_sat2 = od_solver.determine_orbit(gs, observables=radio_obs, method="Lambert")
    # predict_sat3 = od_solver.determine_orbit(gs, observables=optical_obs, method="Gauss")

    # sim.add_satellites({"sat1_predict": predict_sat1, "sat2_predict": predict_sat2, "sat3_predict": predict_sat3})
    # sim.run_all(sat_keys=["sat1_predict", "sat2_predict", "sat3_predict"])
    # sim.plot_all_five()

if __name__ == "__main__":
    main()