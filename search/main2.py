from VisibilityOptimiser import VisibilityOptimiser
from Satellite import Satellite
from GroundStation import GroundStation
from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator

def main():
    # Propagators
    sat_prop = SatellitePropagator(method="RK45", rtol=1e-9, atol=1e-11, max_step=120.0)
    gs_prop  = GroundStationPropagator()

    # Ground station: Sydney, ~50 m altitude
    gs = GroundStation(lat_deg=-33.8688, lon_deg=151.2093, h_m=50.0)

    # Empty Simulator
    sim = Simulator({}, {"Sydney": gs}, sat_prop, gs_prop)

    # Start: 2025-08-11 12:00:00 UTC, run 2 hours at 30 s steps
    sim.build_timebase(Y=2025, M=8, D=11, h=12, m=0, s=0.0,
                       tf_days=1, sample_dt_s=100.0)

    # Build timebase, add your GS (e.g., "Sydney"), then:
    opt = VisibilityOptimiser(sim)

    # print("Optimising visibility for Sydney ground station...")

    # best, all_results = opt.optimize_grid(
    #     gs_key="Sydney",            # your GS key
    #     a_km=14450, # 14450.0 # 12769.56
    #     raan_deg=0.0,
    #     aop_deg=270.0,
    #     inc_deg=None,                # None -> |GS latitude|
    #     e_range=(0.0, 0.3), n_e=5,
    #     ta_range=(0.0, 360.0), n_ta=10,
    #     min_perigee_alt_km=500.0,
    #     min_elev_deg=0.0,
    #     key_prefix="RES",
    #     verbose=True,
    #     save_plot_path="optimisation_grid.png",
    # )
    # print(best)

    best = {'key': 'custom', 
            'params': {'a_km': 14450.0, 'e': 0.0, 'ta_deg': 288.0, 'raan_deg': 0.0, 'inc_deg': 33.8688, 'aop_deg': 270.0},
            'gs_key': 'Sydney',
            'percent_visible': 25.954492865406866}
    opt.create_and_set_best_satellite(best)

    # sim.animate_3d(sat_keys=[best["key"]], step=10, camera_spin=True)

    # best = opt.refine_2d(steps=100,
    #             min_elev_deg=10.0,
    #             lr_e = 0.002,
    #             lr_ta = 10.0,
    #             min_perigee_alt_km=1500.0,
    #             save_plot_path="optimisation_2d.png"
    #             )
    
    # sim.animate_3d(sat_keys=[best["key"]], step=10, camera_spin=True)
    sim.satellite_trajectories[best["key"]].plot_ground_track()
    sim.ground_station_trajectories["Sydney"].plot_ground_track(block=True)
    # # 2) Local refinement (SPSA gradient descent
    # opt.refine_best_local(
    #     steps=10,
    #     min_elev_deg=0.0,
    #     lr0 = (200.0, 200.0, 10.0, 10.0, 10.0, 10.0),
    #     lock_aop_to_raan_plus_90=True,   # set True to enforce AOP = RAAN + 90°
    #     lock_inclination=True,
    #     seed=123,
    #     progress=True,
    # )


if __name__ == "__main__":
    main()