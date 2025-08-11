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
                       tf_days=3, sample_dt_s=100.0)

    # Build timebase, add your GS (e.g., "Sydney"), then:
    opt = VisibilityOptimiser(sim)

    print("Optimising visibility for Sydney ground station...")

    # best, all_results = opt.optimize_grid(
    #     gs_key="Sydney",            # your GS key
    #     a_km=12769.56,
    #     raan_deg=0.0,
    #     aop_deg=270.0,
    #     inc_deg=None,                 # None -> |GS latitude|
    #     e_range=(0.00, 0.30), n_e=10, # 0.0 .. 0.30 in 31 points
    #     ta_range=(0.0, 360.0), n_ta=10,
    #     min_perigee_alt_km=1500.0,
    #     min_elev_deg=0.0,
    #     key_prefix="RES",
    #     verbose=True
    # )
    # print(best)

    best = {'key': 'custom', 
            'params': {'a_km': 12769.56, 'e': 0.0, 'ta_deg': 144.0, 'raan_deg': 0.0, 'inc_deg': 33.8688, 'aop_deg': 270.0}, 
            'gs_key': 'Sydney',
            'percent_visible': 25.954492865406866}

    # opt.create_and_set_best_satellite(best)
    opt.create_and_set_best_satellite(best)
    sim.animate_3d(sat_keys=[best["key"]], step=10, camera_spin=True)

    best = opt.refine_2d(steps=100,
                min_elev_deg=0.0,
                lr_e = 0.01,   # set True to enforce AOP = RAAN + 90°
                lr_ta = 0.1,
                seed=123,
                progress=True,
                verbose=True,
                min_perigee_alt_km=1500.0,
                live_plot=True,
                save_plot_path="optimisation_2d.png"
                )
    
    sim.animate_3d(sat_keys=[best["key"]], step=10, camera_spin=True)

    # # 2) Local refinement (SPSA gradient descent
    # opt.refine_best_local(
    #     steps=100,
    #     min_elev_deg=0.0,
    #     lr0 = (200.0, 200.0, 10.0, 10.0, 10.0, 10.0),
    #     lock_aop_to_raan_plus_90=True,   # set True to enforce AOP = RAAN + 90°
    #     lock_inclination=True,
    #     seed=123,
    #     progress=True,
    # )


if __name__ == "__main__":
    main()