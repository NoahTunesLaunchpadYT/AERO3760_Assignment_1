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
    # A compact grid near ~100 trials:
    # best, all_results = opt.optimize_grid(
    #     gs_key="Sydney",
    #     rp_range=(6000, 7500),  # rp in km above Earth radius
    #     ra_range=(7500, 7500),  # ra in km above Earth radius
    #     ta_range=(0, 360),      # true anomaly in degrees
    #     raan_range=(-180, 180), # RAAN in degrees
    #     inc_range=(30, 90),     # inclination in degrees
    #     n_rp=10, 
    #     n_ra=1,      # rp/ra grid (≈ choose 2 = 10 valid pairs when ra>rp)
    #     n_ta=5,              # 2 true anomalies
    #     n_raan=5,            # 2 RAAN samples
    #     n_inc=3,             # 0°, 180° by default (adjust as needed)
    #     max_orbits=1000,      # cap just in case
    #     min_elev_deg=0,
    #     key_prefix="G",
    #     progress=True,
    #     verbose=True,
    # )

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


    # best = {'key': 'custom', 
    #         'params': {
    #             'rp_alt_km': 6335.3, 
    #             'ra_alt_km': 7500, 
    #             'ta_deg': 213.80, 
    #             'raan_deg': -38.43, 
    #             'inc_deg': 90.12, 
    #             'aop_deg': 54.92}, 
    #         'percent_visible': 31.98,
    #         'gs_key': 'Sydney',
    # }

    # best = {'key': 'custom', 
    #         'params': {
    #             'rp_alt_km': 7500, 
    #             'ra_alt_km': 7500, 
    #             'ta_deg': 0, 
    #             'raan_deg': 0, 
    #             'inc_deg': 33.0, 
    #             'aop_deg': 90}, 
    #         'percent_visible': 0,
    #         'gs_key': 'Sydney',
    # }

    best = {'key': 'RES0004', 
            'params': {'a_km': 12769.56, 'e': 0.0, 'ta_deg': 144.0, 'raan_deg': 0.0, 'inc_deg': 33.8688, 'aop_deg': 270.0}, 
            'gs_key': 'Sydney',
            'percent_visible': 25.954492865406866}

    # opt.create_and_set_best_satellite(best)

    sim.animate_3d(sat_keys=[best["key"]], step=10, camera_spin=True)

    best = opt.refine_2d(steps=100,
                min_elev_deg=0.0,
                lr_e = 0.01,   # set True to enforce AOP = RAAN + 90°
                lr_ta = 0.5,
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