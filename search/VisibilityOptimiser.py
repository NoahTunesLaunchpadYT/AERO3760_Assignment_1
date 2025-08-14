# visibility_optimiser.py
import numpy as np
from tqdm.auto import tqdm
from Satellite import Satellite
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from search import grid_search_2d
import uuid

class VisibilityOptimiser:
    """
    Two-variable grid-search visibility optimiser for a fixed resonant orbit.

    Fixed:
      - a_km        : semi-major axis (e.g., 12769.56 km for 6:1 resonance)
      - raan_deg    : RAAN (default 0°)
      - inc_deg     : inclination (default = |GS latitude|; overrideable)
      - aop_deg     : argument of perigee (default 270° so apogee is at lowest latitude)

    Optimised:
      - e           : eccentricity
      - ta_deg      : true anomaly at epoch (shifts the ground track similarly to RAAN)

    Uses your Simulator timebase & ground station. Builds a grid of (e, ta),
    propagates all trials at once, and picks the orbit with the highest
    visibility percentage above a given minimum elevation.
    """

    def __init__(self, simulator) -> None:
        self.sim = simulator
        self.best = None  # {"key", "params", "percent_visible", "gs_key"}

    # ---------------- internal helpers ----------------
    def _earth_radius_km(self) -> float:
        return float(globals().get("R_EARTH", 6378.137))

    def _add_sats(self, sats: dict, overwrite: bool = True, drop_old_traj: bool = True):
        """
        Add satellites to the simulator, overwriting existing keys if requested,
        and optionally dropping any stale trajectories for those keys.
        """
        if hasattr(self.sim, "add_satellites"):
            self.sim.add_satellites(sats,
                                    overwrite=overwrite,
                                    drop_old_trajectories=drop_old_traj)
        else:
            # Minimal fallback
            for k, sat in sats.items():
                if drop_old_traj and hasattr(self.sim, "satellite_trajectories"):
                    self.sim.satellite_trajectories.pop(k, None)
                if overwrite or (k not in self.sim.satellites):
                    self.sim.satellites[k] = sat

    def _visibility_pct(self, gs_key: str, sat_key: str, min_elev_deg: float) -> float:
        """Percent of samples with elevation >= min_elev_deg."""
        if hasattr(self.sim, "elevation_series"):
            _, _, _, pct = self.sim.elevation_series(gs_key, sat_key, min_elev_deg=min_elev_deg)
            return float(pct)
        gs_traj = self.sim.ground_station_trajectories[gs_key]
        sat_traj = self.sim.satellite_trajectories[sat_key]
        _, el, _ = self.sim._compute_az_el(gs_traj, sat_traj)
        valid = np.isfinite(el)
        if valid.sum() == 0:
            return 0.0
        return 100.0 * (np.logical_and(valid, el >= float(min_elev_deg)).sum() / valid.sum())

    # ---------------- public API ----------------
    def create_and_set_best_satellite(self, best: dict):
        """
        Convenience: instantiate the best satellite found and attach it to the simulator.
        """
        a_km     = best["params"]["a_km"]
        e        = best["params"]["e"]
        ta_deg   = best["params"]["ta_deg"]
        raan_deg = best["params"]["raan_deg"]
        inc_deg  = best["params"]["inc_deg"]
        aop_deg  = best["params"]["aop_deg"]

        sat = Satellite.from_keplerian(a_km=float(a_km), e=float(e),
                                       inc_deg=float(inc_deg),
                                       raan_deg=float(raan_deg),
                                       aop_deg=float(aop_deg),
                                       tru_deg=float(ta_deg))
        key = best["key"]
        self._add_sats({key: sat}, overwrite=True, drop_old_traj=True)
        # Propagate just this sat and the GS
        if hasattr(self.sim, "propagate_one"):
            self.sim.propagate_one(sat_key=key)
        else:
            self.sim.run_all(sat_keys=[key], gs_keys=[best["gs_key"]])
        self.best = best

    # ---------- 2) Your specific objective: visibility from (e, ta) ----------
    def visibility_pct_objective(
        self,
        e: float,
        ta_deg: float,
        *,
        gs_key: str,
        a_km: float = 12769.56,
        raan_deg: float = 0.0,
        inc_deg: float | None = None,      # None -> |GS latitude|
        aop_deg: float = 270.0,
        min_elev_deg: float = 10.0,
        min_perigee_alt_km: float = 200.0,
        key_prefix: str = "RES",
        overwrite_existing: bool = True,
        drop_old_trajectories: bool = True,
    ) -> float:
        """
        Creates a satellite for the given (e, ta), adds it to the sim,
        propagates, and returns visibility percentage for gs_key at min_elev_deg.

        Note: This evaluates per point. For maximum performance, consider
        batching (build all sats, run once) if you need huge grids.
        """
        if self.sim.JD is None:
            raise ValueError("Build the Simulator timebase first (build_timebase).")
        if gs_key not in getattr(self.sim, "ground_stations", {}):
            raise KeyError(f"Ground station '{gs_key}' not found in sim.ground_stations")

        # Inclination default: |GS latitude|
        if inc_deg is None:
            gs = self.sim.ground_stations[gs_key]
            inc_use = float(abs(getattr(gs, "lat_deg", 0.0)))
        else:
            inc_use = float(inc_deg)
        inc_use = max(0.0, min(180.0, inc_use))

        # Physical perigee constraint
        RE = self._earth_radius_km()
        e_max_phys = 1.0 - (RE + float(min_perigee_alt_km)) / float(a_km)
        if e < 0.0 or e > e_max_phys:
            # Return a very poor score if infeasible (keeps the grid search generic)
            return -1.0

        # Build the satellite
        from Satellite import Satellite
        sat_key = f"{key_prefix}{uuid.uuid4().hex[:8]}"
        sat = Satellite.from_keplerian(
            a_km=float(a_km),
            e=float(e),
            inc_deg=inc_use,
            raan_deg=float(raan_deg),
            aop_deg=(float(aop_deg) % 360.0),
            tru_deg=float(ta_deg),
        )

        # Add to sim
        if hasattr(self, "_add_sats"):
            self._add_sats({sat_key: sat}, overwrite=overwrite_existing, drop_old_traj=drop_old_trajectories)
        else:
            if overwrite_existing or sat_key not in self.sim.satellites:
                self.sim.satellites[sat_key] = sat
            if drop_old_trajectories:
                self.sim.satellite_trajectories.pop(sat_key, None)

        # Propagate this one satellite
        self.sim.run_all(sat_keys=[sat_key], gs_keys=[gs_key], progress=False)

        # Score
        pct = float(self._visibility_pct(gs_key, sat_key, min_elev_deg=min_elev_deg))

        # Optional clean up: keep or remove the sat/trajectory if you like
        # del self.sim.satellites[sat_key]
        # self.sim.satellite_trajectories.pop(sat_key, None)

        return pct
