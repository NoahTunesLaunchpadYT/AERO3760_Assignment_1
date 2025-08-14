# main.py
import numpy as np

from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
from GroundStation import GroundStation
from Satellite import Satellite

def main():
    # --- build objects ---
    # Satellite: ~7000 km circular LEO, 51.6 deg inclination
    a_km   = 9845 # 9825   # semi-major axis
    e      = 0
    inc    = 0               # deg
    raan   = 0.0               # deg
    aop    = 0.0                # deg
    tru    = 0.0                # deg
    number_sats = 4
    satellites = {}

    for i in range(number_sats):
        # Adjust the true anomaly for each satellite to create a spread
        tru += (i / number_sats) * 360
        satellites[f"LEO{i+1}"] = Satellite.from_keplerian(a_km, e, inc, raan, aop, tru)

    # Ground station: Sydney, ~50 m altitude
    gs = GroundStation(lat_deg=-33.61, lon_deg=151.2093, h_m=50.0)

    ground_stations = {"Sydney": gs}

    # Propagators
    sat_prop = SatellitePropagator(method="DOP853", rtol=1e-9, atol=1e-11, max_step=120.0)
    gs_prop  = GroundStationPropagator()

    # Simulator
    sim = Simulator(satellites, ground_stations, sat_prop, gs_prop)

    # --- timebase ---
    # Start: 2025-08-11 12:00:00 UTC, run 2 hours at 30 s steps
    sim.build_timebase(Y=2025, M=8, D=11, h=12, m=0, s=0.0,
                       tf_days=1, sample_dt_s=30.0)

    # # --- run and plot ---
    sim.run_all()                 # runs both propagators over the shared JD
    # sim.plot_3d()                 # plots both the satellite and ground station
    # sim.animate_3d(step=1, camera_spin=True)  # animates the trajectory

    # sim.plot_sky_track(gs_key="Sydney", sat_keys=["LEO1"], min_elev_deg=10)
    # sim.animate_sky_track(gs_key="Sydney", sat_keys=["LEO1"], step=2, interval=40, tail=800, min_elev_deg=10)
    
    sim.plot_all_four(gs_key="Sydney", sat_keys=satellites.keys(), step=2, block=False)
    per_sat = sim.elevation_series(gs_key="Sydney", sat_keys=satellites.keys(), min_elev_deg=0.0, plot=True)
    sim.visible_distance_series(gs_key="Sydney", sat_keys=satellites.keys(), min_elev_deg=0.0, plot=True)
    # sim.animate_3d(sat_keys=satellites.keys(), step=2, camera_spin=True)

if __name__ == "__main__":
    main()
