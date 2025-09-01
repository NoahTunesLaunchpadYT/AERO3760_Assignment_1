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

    # Errors
    errors = [0.0, 0.001, 0.005, 0.01]

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

    for error in errors:
        range_sensor = SatelliteLaserRanging(id="range_sensor", range_error=error, angular_error=0, time_error=0)
        radio_sensor = RadiometricTracking(id="radio_sensor", range_error=error, angular_error=0, time_error=0)
        optical_sensor = OpticalTracking(id="optical_sensor", range_error=error, angular_error=0, time_error=0)

        # Collect observations
        range_obs = range_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))
        radio_obs = radio_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2))
        optical_obs = optical_sensor.observe(sat0_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))

        predict_gibbs_sat = od_solver.determine_orbit(gs, observables=range_obs, method="Gibbs")
        predict_lambert_sat = od_solver.determine_orbit(gs, observables=radio_obs, method="Lambert")
        predict_gauss_sat = od_solver.determine_orbit(gs, observables=optical_obs, trajectory=sat0_traj, method="Gauss", plot=True)
        
        a1, e1, inc1, raan1, aop1, ta1 = predict_gibbs_sat.keplerian_elements()
        a2, e2, inc2, raan2, aop2, ta2 = predict_lambert_sat.keplerian_elements()
        a3, e3, inc3, raan3, aop3, ta3 = predict_gauss_sat.keplerian_elements()

        # Calculate the absolute and percentage error of each element
        gibbs_abs_errors = [abs(a1 - a_km), abs(e1 - e), abs(inc1 - inc_deg), abs(raan1 - raan_deg), abs(aop1 - aop_deg), abs(ta1 - ta_deg)]
        lambert_abs_errors = [abs(a2 - a_km), abs(e2 - e), abs(inc2 - inc_deg), abs(raan2 - raan_deg), abs(aop2 - aop_deg), abs(ta2 - ta_deg)]
        gauss_abs_errors = [abs(a3 - a_km), abs(e3 - e), abs(inc3 - inc_deg), abs(raan3 - raan_deg), abs(aop3 - aop_deg), abs(ta3 - ta_deg)]

        gibbs_perc_errors = [abs_err / true_val * 100 if true_val != 0 else 0 for abs_err, true_val in zip(gibbs_abs_errors, [a_km, e, inc_deg, raan_deg, aop_deg, ta_deg])]
        lambert_perc_errors = [abs_err / true_val * 100 if true_val != 0 else 0 for abs_err, true_val in zip(lambert_abs_errors, [a_km, e, inc_deg, raan_deg, aop_deg, ta_deg])]
        gauss_perc_errors = [abs_err / true_val * 100 if true_val != 0 else 0 for abs_err, true_val in zip(gauss_abs_errors, [a_km, e, inc_deg, raan_deg, aop_deg, ta_deg])]

        print(f"\nPredicted values after {error*100:.2f}% sensor error:")
        print(f"    Gibbs   : a={a1:.3f} km, e={e1:.6f}, i={inc1:.6f} deg, Ω={raan1:.6f} deg, ω={aop1:.6f} deg, ν={ta1:.6f} deg")
        print(f"    Lambert : a={a2:.3f} km, e={e2:.6f}, i={inc2:.6f} deg, Ω={raan2:.6f} deg, ω={aop2:.6f} deg, ν={ta2:.6f} deg")
        print(f"    Gauss   : a={a3:.3f} km, e={e3:.6f}, i={inc3:.6f} deg, Ω={raan3:.6f} deg, ω={aop3:.6f} deg, ν={ta3:.6f} deg")

        print(f"\nErrors with {error*100:.2f}% sensor error:")
        print("    Gibbs:")
        print(f"        Absolute Errors   : a={gibbs_abs_errors[0]:.2f}, e={gibbs_abs_errors[1]:.2f}, i={gibbs_abs_errors[2]:.2f}, Ω={gibbs_abs_errors[3]:.2f}, ω={gibbs_abs_errors[4]:.2f}, ν={gibbs_abs_errors[5]:.2f}")
        print(f"        Percentage Errors : a={gibbs_perc_errors[0]:.2f}%, e={gibbs_perc_errors[1]:.2f}%, i={gibbs_perc_errors[2]:.2f}%, Ω={gibbs_perc_errors[3]:.2f}%, ω={gibbs_perc_errors[4]:.2f}%, ν={gibbs_perc_errors[5]:.2f}%")

        print("    Lambert:")
        print(f"        Absolute Errors   : a={lambert_abs_errors[0]:.2f}, e={lambert_abs_errors[1]:.2f}, i={lambert_abs_errors[2]:.2f}, Ω={lambert_abs_errors[3]:.2f}, ω={lambert_abs_errors[4]:.2f}, ν={lambert_abs_errors[5]:.2f}")
        print(f"        Percentage Errors : a={lambert_perc_errors[0]:.2f}%, e={lambert_perc_errors[1]:.2f}%, i={lambert_perc_errors[2]:.2f}%, Ω={lambert_perc_errors[3]:.2f}%, ω={lambert_perc_errors[4]:.2f}%, ν={lambert_perc_errors[5]:.2f}%")

        print("    Gauss:")
        print(f"        Absolute Errors   : a={gauss_abs_errors[0]:.2f}, e={gauss_abs_errors[1]:.2f}, i={gauss_abs_errors[2]:.2f}, Ω={gauss_abs_errors[3]:.2f}, ω={gauss_abs_errors[4]:.2f}, ν={gauss_abs_errors[5]:.2f}")
        print(f"        Percentage Errors : a={gauss_perc_errors[0]:.2f}%, e={gauss_perc_errors[1]:.2f}%, i={gauss_perc_errors[2]:.2f}%, Ω={gauss_perc_errors[3]:.2f}%, ω={gauss_perc_errors[4]:.2f}%, ν={gauss_perc_errors[5]:.2f}%")

        tag = f"{int(error*1000)}"  # 0, 1, 5, 10 (for 0%,0.1%,0.5%,1.0%)
        sim.add_satellites({f"gibbs_predict_{tag}": predict_gibbs_sat, f"lambert_predict_{tag}": predict_lambert_sat, f"gauss_predict_{tag}": predict_gauss_sat})
        sim.run_all()
        sim.plot_eci_static_window(gs_key="NewCastle", sat_keys=["sat0", f"gibbs_predict_{tag}", f"lambert_predict_{tag}", f"gauss_predict_{tag}"], block=False)
        sim.plot_ground_tracks_window(gs_key="NewCastle", sat_keys=["sat0", f"gibbs_predict_{tag}", f"lambert_predict_{tag}", f"gauss_predict_{tag}"], block=True)

        # sim.plot_all_five(gs_key="NewCastle", sat_keys=["sat0", f"gibbs_predict_{tag}", f"lambert_predict_{tag}", f"gauss_predict_{tag}"], step=5, )



if __name__ == "__main__":
    main()