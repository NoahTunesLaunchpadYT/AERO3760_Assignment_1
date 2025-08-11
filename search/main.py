# main.py
import numpy as np

from Simulator import Simulator
from Propagator import SatellitePropagator, GroundStationPropagator
from GroundStation import GroundStation
from Satellite import Satellite

def main():
    # --- build objects ---
    # Satellite: ~7000 km circular LEO, 51.6 deg inclination
    a_km   = 6378.137 + 7900.0   # semi-major axis
    e      = 0.001
    inc    = 51.6               # deg
    raan   = 40.0               # deg
    aop    = 0.0                # deg
    tru    = 0.0                # deg
    sat1 = Satellite.from_keplerian(a_km, e, inc, raan, aop, tru)
    sat2 = Satellite.from_keplerian(a_km, e, inc, raan, aop, tru + 120.0)  # another satellite with same orbit but different true anomaly
    sat3 = Satellite.from_keplerian(a_km, e, inc, raan, aop, tru + 240.0)  # another satellite with same orbit but different true anomaly
    
    # Ground station: Sydney, ~50 m altitude
    gs = GroundStation(lat_deg=-33.8688, lon_deg=151.2093, h_m=50.0)

    satellites = {"LEO1": sat1, "LEO2": sat2, "LEO3": sat3}
    ground_stations = {"Sydney": gs}

    # Propagators
    sat_prop = SatellitePropagator(method="DOP853", rtol=1e-9, atol=1e-11, max_step=120.0)
    gs_prop  = GroundStationPropagator()

    # Simulator
    sim = Simulator(satellites, ground_stations, sat_prop, gs_prop)

    # --- timebase ---
    # Start: 2025-08-11 12:00:00 UTC, run 2 hours at 30 s steps
    sim.build_timebase(Y=2025, M=8, D=11, h=12, m=0, s=0.0,
                       tf_days=10, sample_dt_s=30.0)

    # # --- run and plot ---
    sim.run_all()                 # runs both propagators over the shared JD
    # sim.plot_3d()                 # plots both the satellite and ground station
    # sim.animate_3d(step=1, camera_spin=True)  # animates the trajectory

    # sim.plot_sky_track(gs_key="Sydney", sat_keys=["LEO1"], min_elev_deg=10)
    # sim.animate_sky_track(gs_key="Sydney", sat_keys=["LEO1"], step=2, interval=40, tail=800, min_elev_deg=10)
    
    jd, elev, vis_mask, pct = sim.elevation_series(gs_key="Sydney", sat_key="LEO1", min_elev_deg=0.0)
    print(f"pct: {pct}")
    jd, elev, vis_mask, pct = sim.elevation_series(gs_key="Sydney", sat_key="LEO2", min_elev_deg=0.0)
    print(f"pct: {pct}")
    jd, elev, vis_mask, pct = sim.elevation_series(gs_key="Sydney", sat_key="LEO3", min_elev_deg=0.0)
    print(f"pct: {pct}")

    sim.plot_sky_track(gs_key="Sydney", sat_keys=["LEO1", "LEO2", "LEO3"], min_elev_deg=10)
    sim.animate_3d(sat_keys=["LEO1", "LEO2", "LEO3"], step=10, camera_spin=True)


if __name__ == "__main__":
    main()
