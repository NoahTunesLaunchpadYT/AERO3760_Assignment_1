from VisibilityOptimiser import VisibilityOptimiser
from Satellite import Satellite
from GroundStation import GroundStation
from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
from search import grid_search_2d, refine_2d_generic
from Coverage import cover_circle_with_mask
import time

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

    def f(inc_deg, ta_deg):
        return opt.visibility_pct_objective(
            e=0.0,
            ta_deg=ta_deg,
            gs_key="Sydney",
            a_km=9129,
            raan_deg=72.0,
            inc_deg=inc_deg,
            aop_deg=270.0,
            min_elev_deg=30.0,
            min_perigee_alt_km=400.0
        )
    
    best_rec, _, _ = grid_search_2d(
        func=f,
        x_range=(0.0, 180.0), n_x=10, x_is_angle=True,
        y_range=(0.0, 360.0), n_y=10, y_is_angle=True,
        return_figure=True,
        x_label="Inclination Angle (inc)",
        y_label="True Anomaly (ta)",
        z_label="Visibility Percentage",
        title="Visibility Percentage vs Inclination and True Anomaly",
    )

    print(best_rec)

    # best, hist = refine_2d_generic(
    #     f,
    #     x0=best_rec["x"], y0=best_rec["y"],                  # starting (e, ta_deg)
    #     x_is_angle=True, y_is_angle=True, # e is scalar, ta is an angle
    #     x_bounds=(0.0, 360.0),              # optional bounds for e
    #     y_bounds=(0.0, 360.0),             # used for wrapping + axis limits
    #     steps=50,
    #     lr_x=2.0, lr_y=2.0,
    #     delta_x=1.0, delta_y=1.0,
    #     decay=0.97, decay_deltas=True,
    #     x_label="Eccentricity, e",
    #     y_label="True anomaly, ta [deg]",
    #     z_label="Visibility [%]",
    #     title_left="Visibility vs step",
    #     title_right="Visibility surface",
    #     live_plot=True, plot_every=1,
    #     fig_size=(12, 5),
    # )
    
    best = {'key': 'custom', 
            'params': {'a_km': 9129.0, 'e': 0, 'ta_deg': best_rec["y"], 'raan_deg': 0.0, 'inc_deg': best_rec["x"], 'aop_deg': 270.0},
            'gs_key': 'Sydney',
            'percent_visible': 25.954492865406866}
    # opt.create_and_set_best_satellite(best)

    # best = {'key': 'custom', 
    #         'params': {'a_km': 14450.0, 'e': 0.0, 'ta_deg': 108.0, 'raan_deg': 0.0, 'inc_deg': inc, 'aop_deg': 270.0},
    #         'gs_key': 'Sydney',
    #         'percent_visible': 25.954492865406866}

    opt.create_and_set_best_satellite(best)

    # sim.animate_3d(sat_keys=[best["key"]], step=10, camera_spin=True)

    
    # best = opt.refine_2d(steps=100,
    #             min_elev_deg=10.0,
    #             lr_e = 0.002,
    #             lr_ta = 10.0,
    #             min_perigee_alt_km=1500.0,
    #             save_plot_path="optimisation_2d.png"
    #             )
    


    # sim.satellite_trajectories[best["key"]].plot_ground_track()
    # sim.plot_all_four(gs_key="Sydney", sat_keys=[best["key"]], min_elev_deg=0.0)
    _, _, mask, _ =sim.elevation_series(gs_key="Sydney", sat_key=best["key"], min_elev_deg=0.0, plot=True, save_plot_path="elevation_series.png")
    res = cover_circle_with_mask(mask, n=4, require_full=False)

    print("Shifts (deg):", res.shifts_deg)
    print("Coverage %:", res.coverage_pct, "Covered all:", res.covered_all)

    for i in range(len(res.shifts_deg)):
        print(f"Shift {i+1}: {res.shifts_deg[i]} degrees")
        sat_params = best.copy()
        sat_params['params']['raan_deg'] = res.shifts_deg[i] % 360.0
        sat_params['params']['ta_deg'] = 108.0 - (res.shifts_deg[i] % 360.0) * 5
        sat_params['key'] = f"best_{i+1}"
        print(f"Satellite parameters for shift {i+1}: {sat_params}")
        opt.create_and_set_best_satellite(sat_params)

    sim.plot_all_four(gs_key="Sydney", sat_keys=[f"best_{i+1}" for i in range(len(res.shifts_deg))], min_elev_deg=0.0)

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