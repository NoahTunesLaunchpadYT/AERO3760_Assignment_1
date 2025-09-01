# Noah West
# 530494928

# main.py
# Import Simulation Objects
from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
from GroundStation import GroundStation
from Satellite import Satellite
from OrbitDetermination import SatelliteLaserRanging, RadiometricTracking, OpticalTracking, OrbitDeterminationSolver
from helper.time import *

# Additional Classes
from OrbitDetermination import OrbitDeterminationSolver
from SensitivityAnalyser import SensitivityAnalyser

def question_1(sim: Simulator, 
               min_elev_deg: float, 
               max_distance_km: float):
    # Set orbital parameters of Satellite 1
    e0=0.0
    a0_km=11504
    inc0_deg=49
    aop0_deg=270
    raan0_deg=0.0
    ta0_deg = 182.0
    
    num_sats = 11
    sats = {}

    print(f"\nOrbital parameters of constellation of {num_sats} satellites: \n")
    print(f"{'Satellite:':<15} {'a (km)':>10} {'e':>8} {'inc (deg)':>10} {'raan (deg)':>10} {'aop (deg)':>10} {'ta (deg)':>10}")

    # Generate constellation of 11 equally spaced satellites
    for i in range(num_sats):
        e = e0
        a_km = a0_km
        inc_deg = inc0_deg
        aop_deg = aop0_deg
        raan_deg = (raan0_deg + i * 360.0 / num_sats)%360
        ta_deg = (ta0_deg - i * 7 * 360.0 / num_sats)%360

        # Create satellite object (dataclass)
        sats[f"Satellite {i+1}"] = Satellite.from_keplerian(e=e, a_km=a_km, inc_deg=inc_deg,
                                               aop_deg=aop_deg, raan_deg=raan_deg, ta_deg=ta_deg)

        print(f"{f'Satellite {i+1}':<15} {a_km:10.3f} {e:8.3f} {inc_deg:10.3f} {raan_deg:10.3f} {aop_deg:10.3f} {ta_deg:10.3f}")
    print("")

    # Add satellites to the simulation
    sim.add_satellites(sats)

    # Create observer in the Blue Mountains, ~1000 m altitude
    gs = GroundStation(lat_deg=-33.61, lon_deg=150.464444, h_m=1000.0)

    sim.add_ground_stations({"Blue Mountains": gs})

    # Run the simulation for one day
    sim.run_all()

    print("Plotting constellation animation...")
    sim.plot_eci_anim_window(
        gs_key="Blue Mountains",
        sat_keys=sats.keys(),
        step=10,
    )

    print("Plotting constellation static view...")
    sim.plot_eci_static_window(
        gs_key="Blue Mountains",
        sat_keys=sats.keys(),
    )

    print("Plotting satellite visibility...")
    sim.plot_sat_az_el_rng_vis(
        gs_key="Blue Mountains",
        sat_keys=sats.keys(),
        min_elev_deg=min_elev_deg,
        max_distance_km=max_distance_km,
        show_azimuth=False,
        show_elevation=True,
        show_range=True,
        show_visibility=True,
        fig_size=(8,8)
    )

    print("\n Consider Satellite 1:")
    period = sats["Satellite 1"].get_orbital_period()
    print(f"Orbital period of Satellite 1: {period:.2f} seconds")
    
    # Visualise orbits in ECI frame
    print(f"Simulating Satellite 1 for 24 hours...")
    sim.plot_eci_anim_window(gs_key="Blue Mountains", sat_keys=["Satellite 1"], step=10)
    sim.plot_eci_static_window(gs_key="Blue Mountains", sat_keys=["Satellite 1"])

    # Plot ground track
    print(f"Plotting Ground Track of Satellite 1...")
    sim.spacetools_ground_track("Satellite 1", figsize=(10, 6))

def question_2(sim: Simulator, min_elev_deg: float, max_distance_km: float):
    # -------- Question 2 --------
    print("\n \033[1mQuestion 2\033[0m")

    # Create and simulate a ground station
    gs = GroundStation(lat_deg=-33.9, lon_deg=151.7, h_m=1000.0)
    sim.add_ground_stations({"NewCastle": gs})
    sim.run_all(gs_keys=["NewCastle"], sat_keys=[])

    # Plot satellite visibility
    print("Plotting Range, Azimuth, and Elevation...")
    _, result = sim.plot_sat_az_el_rng_vis(gs_key="NewCastle", sat_keys=["Satellite 1"], min_elev_deg=min_elev_deg)
    sim.plot_sky_anim_window(gs_key="NewCastle", sat_keys=["Satellite 1"], min_elev_deg=min_elev_deg,step=20)
    sim.plot_sky_static_window(gs_key="NewCastle", sat_keys=["Satellite 1"], min_elev_deg=min_elev_deg)

    # Get maximum elevation and visibility percentage
    max_el = result["per_sat"]["Satellite 1"]["elevation_deg"].max()
    print(f"Maximum elevation of Satellite 1 as seen from NewCastle: {max_el:.2f} degrees")
    percent_visible = result["per_sat_visible_pct"]["Satellite 1"]
    print(f"Percentage of time Satellite 1 is visible from NewCastle: {percent_visible:.2f}%")


    # Pick observation times
    jd_time1 = calendar_to_jd(Y, M, D, h+1, m, s)
    jd_time2 = calendar_to_jd(Y, M, D, h+1, m+7, s)
    jd_time3 = calendar_to_jd(Y, M, D, h+1, m+25, s)

    # Create orbit determination solver
    od_solver = OrbitDeterminationSolver()
    sat_traj = sim.satellite_trajectories["Satellite 1"]
    gs_traj = sim.ground_station_trajectories["NewCastle"]


    print("Simulating Satellite Laser Ranger...")
    range_sensor = SatelliteLaserRanging(id="range_sensor")
    range_obs = range_sensor.observe(sat_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))
    predict_gibbs_sat = od_solver.determine_orbit(gs, observables=range_obs, method="Gibbs", trajectory=sat_traj, plot=True)

    print("Simulating Radiometric Tracker...")
    radio_sensor = RadiometricTracking(id="radio_sensor")
    radio_obs = radio_sensor.observe(sat_traj, gs_traj, jd_times=(jd_time1, jd_time2))
    predict_radio_sat = od_solver.determine_orbit(gs, observables=radio_obs, method="Lambert", trajectory=sat_traj, plot=True)

    print("Simulating Optical Tracker...")
    optical_sensor = OpticalTracking(id="optical_sensor")
    optical_obs = optical_sensor.observe(sat_traj, gs_traj, jd_times=(jd_time1, jd_time2, jd_time3))
    predict_optical_sat = od_solver.determine_orbit(gs, observables=optical_obs, method="Gauss", trajectory=sat_traj, plot=True)

    sim.add_satellites({f"gibbs_predict": predict_gibbs_sat,
                        f"lambert_predict": predict_radio_sat,
                        f"gauss_predict": predict_optical_sat})
    
    print("Recreating Satellite Trajectories...")
    sim.run_all(gs_keys=["NewCastle"], sat_keys=["Satellite 1", "gibbs_predict", "lambert_predict", "gauss_predict"])
    
    print("Plotting predicted orbits...")
    sim.plot_eci_anim_window(gs_key="NewCastle", sat_keys=["Satellite 1", "gibbs_predict", "lambert_predict", "gauss_predict"], step=10)
    sim.plot_eci_static_window(gs_key="NewCastle", sat_keys=["Satellite 1", "gibbs_predict", "lambert_predict", "gauss_predict"])

    # Test with sensor errors from 0.1% to 1%
    error_values = [i*0.001 for i in range(11)]

    # Take orbital parameter error as an average over 10 trials  
    repeats = 10 

    # Create sensitivity analyser
    analyser = SensitivityAnalyser(sim, sat_key="Satellite 1", gs_key="NewCastle", od_solver=od_solver)

    # Run sensitivity analysis
    print("Running sensitivity analysis:")

    # Range error sensitivity analysis
    print("Range error sensitivity analysis...")
    analyser.sensor_error_vs_predict_error(
        observation_times_jd=(jd_time1, jd_time2, jd_time3),
        error_type="range",  # 'range', 'angular' or 'temporal'
        error_values=error_values,
        methods=("Gibbs", "Lambert", "Gauss"),
        param_mask=(1, 1, 1, 1, 0, 0),
        repeats=repeats,
        ab_error_instead=(True, True, True, True, True, True),
        figsize=(5, 10),
        block=False,
    )

    # Angular error sensitivity analysis
    print("Running angular error analysis...")
    analyser.sensor_error_vs_predict_error(
        observation_times_jd=(jd_time1, jd_time2, jd_time3),
        error_type="angular",  # 'range', 'angular' or 'temporal'
        error_values=error_values,
        methods=("Gibbs", "Lambert", "Gauss"),
        param_mask=(1, 1, 1, 1, 0, 0),
        repeats=repeats,
        ab_error_instead=(True, True, True, True, True, True),
        figsize=(5, 10),
        block=False
    )

    # Temporal error sensitivity analysis
    print("Running temporal error analysis...")
    analyser.sensor_error_vs_predict_error(
        observation_times_jd=(jd_time1, jd_time2, jd_time3),
        error_type="temporal",  # 'range', 'angular' or 'temporal'
        error_values=error_values,
        methods=("Gibbs", "Lambert", "Gauss"),
        param_mask=(1, 1, 1, 1, 0, 0),
        repeats=repeats,
        ab_error_instead=(True, True, True, True, True, True),
        figsize=(5, 10),
        block=False
    )

    # All error sensitivity analysis
    print("Running error analysis with all sensors...")
    analyser.sensor_error_vs_predict_error(
        observation_times_jd=(jd_time1, jd_time2, jd_time3),
        error_type="all",  # 'range', 'angular', 'temporal', or 'all'
        error_values=error_values,
        methods=("Gibbs", "Lambert", "Gauss"),
        param_mask=(1, 1, 1, 1, 0, 0),
        repeats=repeats,
        ab_error_instead=(True, True, True, True, True, True),
        figsize=(5, 10),
        block=True
    )
    pass

def main():
    # -------- Question 1 --------
    print("\n \033[1mQuestion 1\033[0m")

    # Set requirements for 'visibility'
    min_elev_deg = 30.0
    max_distance_km = 7494.0 # = 299,792,458 * 0.025

    # Create objects responsible for generating state over time arrays
    sat_prop = SatellitePropagator(method="RK45", rtol=1e-9, atol=1e-11, max_step=100.0) # RK45 or DOP853
    gs_prop  = GroundStationPropagator()

    # Create and Empty Simulator with the propagators and ground station
    sim = Simulator({}, {}, sat_prop, gs_prop)

    # Specify start time, run 1 sidereal day at 30 s steps
    sim.build_timebase(Y=Y, M=M, D=D, h=h, m=m, s=s,
                       tf_days=24/24, sample_dt_s=30.0)
    
    # Run Question 1 code
    question_1(sim, min_elev_deg, max_distance_km)

    # Run Question 2 code
    question_2(sim, min_elev_deg, max_distance_km)

    print("\nAll done!")

# If run direction, run main
if __name__ == "__main__":
    main()