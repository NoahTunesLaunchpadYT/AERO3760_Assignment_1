from Satellite import Satellite
from GroundStation import GroundStation
from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
import numpy as np
from Orbit_Determination import SatelliteLaserRanging, RadiometricTracking, OpticalTracking, OrbitDeterminationSolver
from helper.time import calendar_to_jd

def main():
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

    e=0.0
    a_km=11504.0
    inc_deg=49.0
    aop_deg=270.0
    raan_deg=0.0
    ta_deg = 182.0

    # Ground station: Sydney, ~50 m altitude
    gs = GroundStation(lat_deg=-33.9, lon_deg=151.7, h_m=1000.0)

    sat0 = Satellite.from_keplerian(
        a_km=a_km,
        e=e,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        aop_deg=aop_deg,
        ta_deg=ta_deg
    )

    # Propagators
    sat_prop = SatellitePropagator(method="DOP853", rtol=1e-7, atol=1e-8, max_step=100.0) # or DOP853
    gs_prop  = GroundStationPropagator()

    # Empty Simulator
    sim = Simulator({"sat0": sat0}, {"NewCastle": gs}, sat_prop, gs_prop)

    sim.build_timebase(Y=Y, M=M, D=D, h=h, m=m, s=s,
                       tf_days=1, sample_dt_s=10.0)
    
    sim.run_all()  # runs both propagators over the shared JD

    sim.plot_sat_az_el_rng_vis( gs_key="NewCastle", 
                                sat_keys=["sat0"], 
                                show_azimuth=True,
                                show_elevation=True,
                                show_range=True,
                                show_visibility=True,
                                min_elev_deg=min_elev_deg,
                                max_distance_km=max_distance_km,
                                    )

    # Orbit determination stuff
    sat0_traj = sim.satellite_trajectories["sat0"]
    gs_traj = sim.ground_station_trajectories["NewCastle"]

    # Create Sensors
    range_sensor = SatelliteLaserRanging(id="range_sensor", range_error=0, angular_error=0.0, time_error=0.01)
    radio_sensor = RadiometricTracking(id="radio_sensor", range_error=0, angular_error=0.0, time_error=0.01)
    optical_sensor = OpticalTracking(id="optical_sensor", range_error=0, angular_error=0.0, time_error=0.01)

    # Pick times
    jd_time1 = calendar_to_jd(Y, M, D, h, m, s)
    jd_time2 = calendar_to_jd(Y, M, D, h, m + 5, s)
    jd_time3 = calendar_to_jd(Y, M, D, h, m + 10, s)

    # Collect observations
    range_obs = range_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))
    radio_obs = radio_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2))
    optical_obs = optical_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))

    od_solver = OrbitDeterminationSolver()
    predict_sat1 = od_solver.determine_orbit(gs, observables=range_obs, method="Gibbs")
    predict_sat2 = od_solver.determine_orbit(gs, observables=radio_obs, method="Lambert")
    # predict_sat3 = od_solver.determine_orbit(gs, observables=optical_obs, method="Gauss")

    sim.add_satellites({"sat1_predict": predict_sat1, "sat2_predict": predict_sat2})
    
    sim.run_all()
    sim.plot_all_five(gs_key="NewCastle", sat_keys=["sat0", "sat1_predict", "sat2_predict"], step=5)

if __name__ == "__main__":
    main()