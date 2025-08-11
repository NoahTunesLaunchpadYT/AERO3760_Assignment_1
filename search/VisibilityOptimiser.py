# visibility_optimiser.py
import numpy as np
from tqdm.auto import tqdm
from Satellite import Satellite

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

    def optimize_grid(self,
                      gs_key: str,
                      *,
                      # Fixed geometry (resonant orbit setup)
                      a_km: float = 12769.56,
                      raan_deg: float = 0.0,
                      inc_deg: float | None = None,   # None -> use |GS latitude|
                      aop_deg: float = 270.0,
                      # Search variables (ranges & counts)
                      e_range: tuple[float, float] = (0.0, 0.25),
                      n_e: int = 16,
                      ta_range: tuple[float, float] = (0.0, 360.0),
                      n_ta: int = 48,
                      # Physical constraint
                      min_perigee_alt_km: float = 200.0,
                      # Scoring
                      min_elev_deg: float = 10.0,
                      key_prefix: str = "RES",
                      overwrite_existing: bool = True,
                      drop_old_trajectories: bool = True,
                      progress: bool = True,
                      verbose: bool = True):
        """
        Grid-search over (e, ta_deg) with a, RAAN, INC, AOP held fixed.

        Args:
            gs_key: ground station key present in sim.ground_stations
            a_km:   semi-major axis (e.g., 12769.56 km for 6:1 resonance)
            raan_deg: fixed RAAN (deg)
            inc_deg:  fixed inclination; if None, uses |GS latitude|
            aop_deg:  fixed argument of perigee (deg), default 270°
            e_range:  (emin, emax) initial range for eccentricity
            n_e:      number of e samples (inclusive endpoints)
            ta_range: (0, 360) by default; if spans full circle, endpoint is excluded
            n_ta:     number of true anomaly samples
            min_perigee_alt_km: perigee altitude floor; trims e_max to keep rp >= RE+floor
            min_elev_deg: visibility elevation mask for scoring
        """
        if self.sim.JD is None:
            raise ValueError("Build the Simulator timebase first (build_timebase).")
        if gs_key not in getattr(self.sim, "ground_stations", {}):
            raise KeyError(f"Ground station '{gs_key}' not found in sim.ground_stations")

        RE = self._earth_radius_km()

        # Inc: default to |GS latitude| if not provided
        if inc_deg is None:
            gs = self.sim.ground_stations[gs_key]
            inc_use = float(abs(getattr(gs, "lat_deg", 0.0)))
        else:
            inc_use = float(inc_deg)
        # Keep within [0, 180]
        inc_use = max(0.0, min(180.0, inc_use))

        # Enforce perigee floor: e <= 1 - (RE + min_alt)/a
        e_floor, e_ceil = float(e_range[0]), float(e_range[1])
        e_max_phys = 1.0 - (RE + float(min_perigee_alt_km)) / float(a_km)
        if e_max_phys < 0.0:
            raise ValueError("Chosen a_km & perigee floor are inconsistent (a too small).")
        e_hi = min(e_ceil, e_max_phys)
        if e_hi < e_floor:
            raise ValueError(f"e_range too high for perigee floor. "
                             f"Max allowed ≈ {e_max_phys:.4f} for a={a_km} km and floor={min_perigee_alt_km} km.")

        # Build grids
        def _lin(lo, hi, n, angle=False):
            if n <= 0:
                return np.array([], dtype=float)
            if angle and np.isclose((hi - lo) % 360.0, 0.0):
                return np.linspace(lo, hi, n, endpoint=False)
            return np.linspace(lo, hi, n, endpoint=True)

        e_grid  = _lin(e_floor, e_hi, n_e, angle=False)
        ta_grid = _lin(*ta_range, n_ta, angle=True)

        # Build trial satellites
        sats: dict[str, object] = {}
        params_by_key: dict[str, dict] = {}
        keys: list[str] = []

        count = 0
        for e in e_grid:
            for ta in ta_grid:
                key = f"{key_prefix}{count:04d}"
                sat = Satellite.from_keplerian(a_km=float(a_km), e=float(e),
                                               inc_deg=inc_use,
                                               raan_deg=float(raan_deg),
                                               aop_deg=(float(aop_deg) % 360.0),
                                               tru_deg=float(ta))
                sats[key] = sat
                params_by_key[key] = {
                    "a_km": float(a_km),
                    "e": float(e),
                    "ta_deg": float(ta),
                    "raan_deg": float(raan_deg),
                    "inc_deg": float(inc_use),
                    "aop_deg": float(aop_deg) % 360.0,
                }
                keys.append(key)
                count += 1

        if count == 0:
            raise RuntimeError("No orbits generated: check n_e/n_ta and ranges.")

        # Add to simulator and propagate all trials + this GS
        self._add_sats(sats, overwrite=overwrite_existing, drop_old_traj=drop_old_trajectories)
        # Propagate everyone at once for consistency of JD sampling
        self.sim.run_all(sat_keys=keys, gs_keys=[gs_key])

        # Score visibility (progress bar)
        results = []
        best_key, best_pct = None, -1.0
        iterator = tqdm(keys, desc="Scoring visibility", unit="sat", leave=False) if progress else keys
        for key in iterator:
            pct = self._visibility_pct(gs_key, key, min_elev_deg=min_elev_deg)
            rec = {"key": key, "params": params_by_key[key], "percent_visible": float(pct)}
            results.append(rec)
            if pct > best_pct:
                best_pct, best_key = pct, key
                if progress:
                    iterator.set_postfix_str(f"best={best_pct:.2f}% ({best_key})")

        best = next(r for r in results if r["key"] == best_key)

        if verbose:
            p = best["params"]
            print(
                f"[BEST] {best_key}  vis ≥{min_elev_deg:.1f}°: {best['percent_visible']:.2f}%  |  "
                f"a={p['a_km']:.2f} km, e={p['e']:.4f}, ta={p['ta_deg']:.1f}°, "
                f"inc={p['inc_deg']:.1f}°, raan={p['raan_deg']:.1f}°, aop={p['aop_deg']:.1f}°"
            )

        # Save "best" on the instance for later reuse
        self.best = {
            "key": best_key,
            "params": best["params"].copy(),
            "percent_visible": float(best["percent_visible"]),
            "gs_key": gs_key,
        }

        return best, results

    def refine_2d(self,
                *,
                steps: int = 200,
                min_elev_deg: float = 10.0,
                # learning rates for (e, ta_deg)
                lr_e: float = 0.02,
                lr_ta: float = 2.0,
                # SPSA perturbation sizes for (e, ta_deg)
                delta_e: float = 0.01,
                delta_ta: float = 1.0,
                decay: float = 0.97,       # geometric decay of lr and delta
                seed: int | None = None,
                progress: bool = True,
                verbose: bool = True,
                min_perigee_alt_km: float = 200.0,
                # --- live plotting / logging ---
                live_plot: bool = True,
                plot_every: int = 1,
                log_path: str | None = "visibility_refine2d_log.txt",
                save_plot_path: str | None = None):
        """
        2-D SPSA refiner over (e, ta_deg) starting from self.best.

        Fixed (taken from self.best['params']): a_km, raan_deg, inc_deg, aop_deg
        Optimised: e (bounded by perigee floor), ta_deg (wrapped to [0, 360))

        Live outputs:
        - Interactive plot of visibility % vs. step (current & best)
        - Tab-separated log file (step, current %, best %, e, ta_deg, fixed params)

        Returns:
        self.best (dict): {"params": {...}, "percent_visible": float, "key": str, "gs_key": str}
        """
        from tqdm.auto import tqdm
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        from Satellite import Satellite

        if not self.best:
            raise RuntimeError("Run optimize_grid(...) first; self.best is not set.")
        gs_key = self.best.get("gs_key")
        if gs_key is None:
            raise RuntimeError("self.best['gs_key'] missing; re-run grid search.")

        # Ensure GS trajectory exists (in case caller didn't just run grid search)
        if gs_key not in getattr(self.sim, "ground_station_trajectories", {}):
            if hasattr(self.sim, "propagate_one"):
                self.sim.propagate_one(gs_key=gs_key)
            else:
                self.sim.run_all(sat_keys=[], gs_keys=[gs_key], progress=False)

        # Fixed params from best
        pfix = self.best["params"]
        a_km     = float(pfix["a_km"])
        raan_deg = float(pfix["raan_deg"])
        inc_deg  = float(pfix["inc_deg"])
        aop_deg  = float(pfix["aop_deg"])

        # Start point (e, ta)
        e0      = float(pfix["e"])
        ta0_deg = float(pfix["ta_deg"])

        # Bounds / wrap
        RE = float(globals().get("R_EARTH", 6378.137))
        e_max_phys = 1.0 - (RE + float(min_perigee_alt_km)) / a_km
        if e_max_phys <= 0.0:
            raise ValueError("a_km too small for the requested perigee floor.")

        def _wrap_ta(x_deg: float) -> float:
            return float(x_deg) % 360.0

        def _project(e: float, ta_deg: float) -> tuple[float, float]:
            e = max(0.0, min(e_max_phys, float(e)))
            ta_deg = _wrap_ta(ta_deg)
            return e, ta_deg

        # Evaluate a candidate (return visibility %)
        def _score(e: float, ta_deg: float) -> float:
            sat = Satellite.from_keplerian(
                a_km=a_km, e=e, inc_deg=inc_deg,
                raan_deg=raan_deg, aop_deg=aop_deg, tru_deg=ta_deg
            )
            tmp_key = "_RES_2D_OPT_"
            if hasattr(self.sim, "add_satellites"):
                self.sim.add_satellites({tmp_key: sat}, overwrite=True, drop_old_trajectories=True)
            else:
                self.sim.satellites[tmp_key] = sat
                if hasattr(self.sim, "satellite_trajectories"):
                    self.sim.satellite_trajectories.pop(tmp_key, None)

            if hasattr(self.sim, "propagate_one"):
                self.sim.propagate_one(sat_key=tmp_key)
            else:
                self.sim.run_all(sat_keys=[tmp_key], gs_keys=[], progress=False)

            return float(self._visibility_pct(gs_key, tmp_key, min_elev_deg=min_elev_deg))

        # Init
        rng = np.random.default_rng(seed)
        e, ta_deg = _project(e0, ta0_deg)
        best_e, best_ta = e, ta_deg
        best_pct = _score(e, ta_deg)

        # Live logging
        f_log = None
        try:
            if log_path:
                f_log = open(log_path, "w", encoding="utf-8")
                f_log.write("# step\tpct_now\tbest_pct\te\tta_deg\ta_km\traan_deg\tinc_deg\taop_deg\n")
                f_log.flush()
        except Exception as exc:
            print(f"[warn] could not open log file '{log_path}': {exc}")
            f_log = None

        # Live plotting
        if live_plot:
            plt.ion()
            fig, ax = plt.subplots()
            ln_now,  = ax.plot([], [], lw=1.5, label="current %")
            ln_best, = ax.plot([], [], lw=1.5, label="best %")
            ax.set_xlabel("Step")
            ax.set_ylabel(f"Visibility ≥{min_elev_deg:.0f}° [%]")
            ax.set_title("2D refinement (e, ta)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            xs_hist, now_hist, best_hist = [], [], []
            fig.canvas.draw(); fig.canvas.flush_events()
        else:
            fig = None
            xs_hist, now_hist, best_hist = [], [], []

        # Schedules
        lr_e_k, lr_ta_k = float(lr_e), float(lr_ta)
        de_k, dta_k     = float(delta_e), float(delta_ta)

        iterator = tqdm(range(steps), desc="Refining 2D (SPSA)", unit="step", leave=False) if progress else range(steps)
        for k in iterator:
            s_e  = rng.choice([-1.0, 1.0])
            s_ta = rng.choice([-1.0, 1.0])

            # Two-sided probe
            e_p, ta_p = _project(e + de_k * s_e,      ta_deg + dta_k * s_ta)
            e_m, ta_m = _project(e - de_k * s_e,      ta_deg - dta_k * s_ta)

            f_plus  = -_score(e_p, ta_p)
            f_minus = -_score(e_m, ta_m)

            # Gradient estimate
            ge  = (f_plus - f_minus) / max(2.0 * de_k * s_e, 1e-12)
            gta = (f_plus - f_minus) / max(2.0 * dta_k * s_ta, 1e-12)

            # Step
            e, ta_deg = _project(e - lr_e_k * ge, ta_deg - lr_ta_k * gta)

            # Evaluate & track best
            pct_now = _score(e, ta_deg)
            if pct_now > best_pct:
                best_pct = pct_now
                best_e, best_ta = e, ta_deg

            # Logging
            if f_log is not None:
                try:
                    f_log.write(f"{k}\t{pct_now:.6f}\t{best_pct:.6f}\t"
                                f"{e:.8f}\t{ta_deg:.6f}\t"
                                f"{a_km:.3f}\t{raan_deg:.3f}\t{inc_deg:.3f}\t{aop_deg:.3f}\n")
                    f_log.flush()
                    os.fsync(f_log.fileno())
                except Exception as exc:
                    print(f"[warn] log write failed at step {k}: {exc}")

            # Live plot update
            xs_hist.append(k); now_hist.append(pct_now); best_hist.append(best_pct)
            if live_plot and (k % max(1, plot_every) == 0):
                try:
                    ln_now.set_data(xs_hist, now_hist)
                    ln_best.set_data(xs_hist, best_hist)
                    ax.relim(); ax.autoscale_view()
                    fig.canvas.draw(); fig.canvas.flush_events()
                except Exception:
                    pass

            # Decay
            lr_e_k  *= decay
            lr_ta_k *= decay
            de_k    *= decay
            dta_k   *= decay

            if progress:
                iterator.set_postfix_str(f"best={best_pct:.2f}%  e={best_e:.4f} ta={best_ta:.1f}°")

        # Save result
        self.best = {
            "key": self.best.get("key", "RES-2D-BEST"),
            "params": {
                "a_km": a_km,
                "e": float(best_e),
                "ta_deg": float(best_ta),
                "raan_deg": raan_deg,
                "inc_deg": inc_deg,
                "aop_deg": aop_deg,
            },
            "percent_visible": float(best_pct),
            "gs_key": gs_key,
        }

        # Save plot if requested
        if live_plot and save_plot_path:
            try:
                fig.savefig(save_plot_path, dpi=140, bbox_inches="tight")
            except Exception as exc:
                print(f"[warn] could not save plot to '{save_plot_path}': {exc}")

        if live_plot:
            try:
                plt.ioff(); plt.show(block=False)
            except Exception:
                pass

        if f_log is not None:
            try: f_log.close()
            except Exception: pass

        if verbose:
            print(f"[2D LOCAL BEST] vis ≥{min_elev_deg:.1f}°: {best_pct:.2f}%  |  "
                f"a={a_km:.2f} km, e={best_e:.5f}, ta={best_ta:.2f}°, "
                f"inc={inc_deg:.2f}°, raan={raan_deg:.2f}°, aop={aop_deg:.2f}°")

        return self.best
