from VisibilityOptimiser import VisibilityOptimiser
from Satellite import Satellite
from GroundStation import GroundStation
from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
from search import grid_search_2d, refine_2d_generic
import time
from utils import _wrap_deg
import numpy as np

def main():
    Y = 2026
    M = 0
    D = 0
    h = 0
    m = 0
    s = 0.0

    min_elev_deg = 30.0
    max_distance = 7500.0

    e=0.0
    a_km=11546.8275
    inc_deg=48.529292
    aop_deg=270
    raan_deg=0.0
    ta_deg=182.0

    # Propagators
    sat_prop = SatellitePropagator(method="RK45", rtol=1e-9, atol=1e-11, max_step=120.0)
    gs_prop  = GroundStationPropagator()

    # Ground station: Sydney, ~50 m altitude
    gs = GroundStation(lat_deg=-33.61, lon_deg=150.464444, h_m=50.0)

    # Empty Simulator
    sim = Simulator({}, {"Sydney": gs}, sat_prop, gs_prop)

    # Start: 2026-01-11 12:00:00 UTC, run 2 hours at 30 s steps
    sim.build_timebase(Y=Y, M=M, D=D, h=h, m=m, s=s,
                       tf_days=4, sample_dt_s=100.0)

    # Build timebase, add your GS (e.g., "Sydney"), then:
    opt = VisibilityOptimiser(sim)

    # print("Optimising visibility for Sydney ground station...")
    def f(a_km, inc_deg):
        return opt.visibility_pct_objective(
            e=e,
            ta_deg=ta_deg,
            gs_key="Sydney",
            a_km=a_km,
            raan_deg=raan_deg,
            inc_deg=inc_deg,
            aop_deg=aop_deg,
            min_elev_deg=min_elev_deg,
            max_distance=max_distance,
            min_perigee_alt_km=400.0
        )
    
    # best_rec, _, _ = grid_search_2d(
    #     func=f,
    #     x_range=(11200, 12500), n_x=30, x_is_angle=False,
    #     y_range=(40, 55), n_y=11, y_is_angle=True,
    #     return_figure=True,
    #     x_label="Semi-major Axis (km)",
    #     y_label="Inclination (deg)",
    #     z_label="Visibility Percentage",
    #     title="Visibility Percentage vs Semi-major Axis and Inclination",
    # )

    best = {'key': 'custom', 
        'params': {'a_km': a_km, 'e': e, 'ta_deg': ta_deg, 'raan_deg': raan_deg, 'inc_deg': inc_deg, 'aop_deg': aop_deg},
        'gs_key': 'Sydney',
        'percent_visible': 25.954492865406866}

    opt.create_and_set_best_satellite(best)

    # best_rec, _ = refine_2d_generic(func=f, x0=best["params"]["a_km"], y0=best["params"]["inc_deg"], x_is_angle=False, y_is_angle=True, x_bounds=(11200, 12500), y_bounds=(40, 55), steps=100,
    #                                 lr_x=10, lr_y=0.1, delta_x=5, delta_y=0.05, decay=0.97, decay_deltas=True, seed=None)

    # best = {'key': 'custom', 
    #     'params': {'a_km': best_rec["x"], 'e': e, 'ta_deg': ta_deg, 'raan_deg': raan_deg, 'inc_deg': best_rec["y"], 'aop_deg': aop_deg},
    #     'gs_key': 'Sydney',
    #     'percent_visible': 25.954492865406866}

    # opt.create_and_set_best_satellite(best)

        # --- Plot single best satellite first (optional) ---

    # ==============================
    # Build a 10-sat constellation
    # ==============================
    # 21 works
    n = 11
    raan0 = best["params"]["raan_deg"]
    ta0   = best["params"]["ta_deg"]
    a_km  = best["params"]["a_km"]
    e     = best["params"]["e"]
    inc   = best["params"]["inc_deg"]
    aop   = best["params"]["aop_deg"]

    dRAAN = 360.0 / n
    dTA   = 7.0 * 360.0 / n  # phase so each sat has the same repeating ground track

    # Create satellites spaced in RAAN, phased in true anomaly
    sat_keys = []
    for i in range(n):
        raan_i = _wrap_deg(raan0 + i * dRAAN, 0.0)
        ta_i   = _wrap_deg(ta0   - i * dTA,   0.0)

        key = f"plane0_sat{i:02d}"
        sat_keys.append(key)

        # Construct the satellite (use whichever constructor your class supports)
        sat = Satellite.from_keplerian(
            a_km, e, inc,
            raan_i, aop, ta_i
        )

        # Add to the simulator (pick the correct method for your API)
        sim.satellites[key] = sat

    sim.run_all(sat_keys=sat_keys)
    # ==============================
    # Plot constellation results
    # ==============================
    res_dict = sim.plot_elevation_visibility_distance(
        gs_key="Sydney",
        sat_keys=sat_keys,
        min_elev_deg=min_elev_deg,
        max_distance_km=max_distance
    )

    mask = res_dict["per_sat"]["plane0_sat00"]["visible"]
    np.save("mask.npy", mask)

    sim.plot_all_four(
        gs_key="Sydney",
        sat_keys=sat_keys,
        step=10,
        min_elev_deg=min_elev_deg,
        distance_thresh_km=max_distance
    )

if __name__ == "__main__":
    main()