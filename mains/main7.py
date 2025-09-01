from Satellite import Satellite
from GroundStation import GroundStation
from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
import numpy as np
from OrbitDetermination import SatelliteLaserRanging, RadiometricTracking, OpticalTracking, OrbitDeterminationSolver
from helper.time import calendar_to_jd
from helper.coordinate_transforms import enu_matrix

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
    sat_prop = SatellitePropagator(method="RK45", rtol=1e-9, atol=1e-11, max_step=120.0)
    gs_prop  = GroundStationPropagator()

    # Empty Simulator
    sim = Simulator({"sat0": sat0}, {"NewCastle": gs}, sat_prop, gs_prop)

    # Start: 2026-01-11 12:00:00 UTC, run 2 hours at 30 s steps
    sim.build_timebase(Y=Y, M=M, D=D, h=h, m=m, s=s,
                       tf_days=1, sample_dt_s=10.0)

    sim.run_all()  # runs both propagators over the shared JD

    # Orbit determination stuff
    sat0_traj = sim.satellite_trajectories["sat0"]
    gs_traj = sim.ground_station_trajectories["NewCastle"]

    # Pick times
    jd_time1 = calendar_to_jd(Y, M, D, h+1, m, s)
    jd_time2 = calendar_to_jd(Y, M, D, h+1, m+5, s)
    jd_time3 = calendar_to_jd(Y, M, D, h+1, m+15, s)

    od_solver = OrbitDeterminationSolver()

    print("\nTrue values:")
    print(f"    Semi-major axis (a): {a_km:.3f} km")
    print(f"    Eccentricity (e): {e:.6f}")
    print(f"    Inclination (i): {inc_deg:.6f} deg")
    print(f"    RAAN (Ω): {raan_deg:.6f} deg")
    print(f"    AOP (ω): {aop_deg:.6f} deg")
    print(f"    True Anomaly (ν): {ta_deg:.6f} deg")

    range_sensor = SatelliteLaserRanging(id="range_sensor", range_error=0.03/11000, angular_error=0.5/60/60/360, time_error=25e-12/60/5)
    radio_sensor = RadiometricTracking(id="radio_sensor", range_error=1/11000, angular_error=0.5/60/60/360, time_error=25e-12/60/5)
    optical_sensor = OpticalTracking(id="optical_sensor", range_error=0, angular_error=0.5/60/60/360, time_error=25e-12/60/5)

    trials = 1000

    gibbs_errors = []
    lambert_errors = []
    gauss_errors = []

    for _ in range(trials):
        # Collect observations
        range_obs = range_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))
        radio_obs = radio_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2))
        optical_obs = optical_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))

        predict_gibbs_sat = od_solver.determine_orbit(gs, observables=range_obs, method="Gibbs")
        predict_lambert_sat = od_solver.determine_orbit(gs, observables=radio_obs, method="Lambert")
        predict_gauss_sat = od_solver.determine_orbit(gs, observables=optical_obs, trajectory=sat0_traj, method="Gauss")
        
        a1, e1, inc1, raan1, aop1, ta1 = predict_gibbs_sat.keplerian_elements()
        a2, e2, inc2, raan2, aop2, ta2 = predict_lambert_sat.keplerian_elements()
        a3, e3, inc3, raan3, aop3, ta3 = predict_gauss_sat.keplerian_elements()

        # Calculate the absolute and percentage error of each element
        gibbs_abs_errors = [abs(a1 - a_km), abs(e1 - e), abs(inc1 - inc_deg), abs(raan1 - raan_deg), abs(aop1 - aop_deg), abs(ta1 - ta_deg)]
        lambert_abs_errors = [abs(a2 - a_km), abs(e2 - e), abs(inc2 - inc_deg), abs(raan2 - raan_deg), abs(aop2 - aop_deg), abs(ta2 - ta_deg)]
        gauss_abs_errors = [abs(a3 - a_km), abs(e3 - e), abs(inc3 - inc_deg), abs(raan3 - raan_deg), abs(aop3 - aop_deg), abs(ta3 - ta_deg)]

        gibbs_errors.append(gibbs_abs_errors)
        lambert_errors.append(lambert_abs_errors)
        gauss_errors.append(gauss_abs_errors)

    # Calculate mean errors for each method
    gibbs_mean_errors = np.mean(gibbs_errors, axis=0)
    lambert_mean_errors = np.mean(lambert_errors, axis=0)
    gauss_mean_errors = np.mean(gauss_errors, axis=0)

    # Print average errors for each method
    print(f"\nAverage Errors over {trials} trials:")
    print(f"\nErrors with sensor error:")
    print("    Gibbs:")
    print(f"        Absolute Errors   : a={gibbs_mean_errors[0]:.5f}, e={gibbs_mean_errors[1]:.5f}, i={gibbs_mean_errors[2]:.5f}, Ω={gibbs_mean_errors[3]:.5f}")
    print("    Lambert:")
    print(f"        Absolute Errors   : a={lambert_mean_errors[0]:.5f}, e={lambert_mean_errors[1]:.5f}, i={lambert_mean_errors[2]:.5f}, Ω={lambert_mean_errors[3]:.5f}")
    print("    Gauss:")
    print(f"        Absolute Errors   : a={gauss_mean_errors[0]:.5f}, e={gauss_mean_errors[1]:.5f}, i={gauss_mean_errors[2]:.5f}, Ω={gauss_mean_errors[3]:.5f}")

    sim.add_satellites({f"gibbs_predict": predict_gibbs_sat, f"lambert_predict": predict_lambert_sat, f"gauss_predict": predict_gauss_sat})
    sim.run_all()
    sim.plot_eci_static_window(gs_key="NewCastle", sat_keys=["sat0", f"gibbs_predict", f"lambert_predict", f"gauss_predict"], block=False)
    sim.plot_ground_tracks_window(gs_key="NewCastle", sat_keys=["sat0", f"gibbs_predict", f"lambert_predict", f"gauss_predict"], block=True)

    # sim.plot_all_five(gs_key="NewCastle", sat_keys=["sat0", f"gibbs_predict_{tag}", f"lambert_predict_{tag}", f"gauss_predict_{tag}"], step=5, )



if __name__ == "__main__":
    main()