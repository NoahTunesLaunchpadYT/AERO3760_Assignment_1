# visibility_optimiser.py (or drop into your module)

import numpy as np
from tqdm.auto import tqdm
from Satellite import Satellite

class VisibilityOptimiser:
    """
    Grid-search visibility optimiser.

    For a given ground station and simulator timebase, create a grid of
    (rp, ra, ta, raan, inc, aop), propagate all trial satellites at once,
    and select the orbit with the highest visibility percentage above a
    minimum elevation.
    """

    def __init__(self, simulator) -> None:
        self.sim = simulator
        self.best = None

    # ---------------- internal helpers ----------------
    def _earth_radius_km(self) -> float:
        return float(globals().get("R_EARTH", 6378.137))

    def _add_sats(self, sats: dict, overwrite: bool = True, drop_old_traj: bool = True):
        satellite = Satellite.from_peri_apo()
        self.sim.add_satellites({"custom": satellite})

    def _visibility_pct(self, gs_key: str, sat_key: str, min_elev_deg: float) -> float:
        """Percent of samples with elevation >= min_elev_deg."""
        if hasattr(self.sim, "elevation_series"):
            _, _, vis_mask, pct = self.sim.elevation_series(gs_key, sat_key, min_elev_deg=min_elev_deg)
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
        tru_deg = best.get("params", {}).get("ta_deg", 0.0)
        rp_km = best.get("params", {}).get("rp_alt_km", 0.0) + self._earth_radius_km()
        ra_km = best.get("params", {}).get("ra_alt_km", 0.0) + self._earth_radius_km()
        raan_deg = best.get("params", {}).get("raan_deg", 0.0)
        inc_deg = best.get("params", {}).get("inc_deg", 0.0)
        aop_deg = best.get("params", {}).get("aop_deg", 0.0)
        
        satellite = Satellite.from_peri_apo(tru_deg=float(tru_deg),
                                rp_km=float(rp_km),
                                ra_km=float(ra_km),
                                raan_deg=float(raan_deg),
                                inc_deg=float(inc_deg),
                                aop_deg=float(aop_deg))

        self.sim.add_satellites({best["key"]: satellite})
        self.sim.run_all(sat_keys=[best["key"]], gs_keys=[best.get("gs_key", "default")], progress=True)
        self.best = best

    def optimize_grid(self,
                      gs_key: str,
                      *,
                      # ranges (km altitudes, degrees for angles)
                      rp_range: tuple[float, float] = (400.0, 7500.0),
                      ra_range: tuple[float, float] = (400.0, 7500.0),
                      ta_range: tuple[float, float] = (0.0, 360.0),
                      raan_range: tuple[float, float] = (-180.0, 180.0),
                      inc_range: tuple[float, float] = (0.0, 180.0),
                      # grid counts
                      n_rp: int = 5,
                      n_ra: int = 5,
                      n_ta: int = 3,
                      n_raan: int = 4,
                      n_inc: int = 3,
                      # execution / scoring
                      max_orbits: int | None = None,
                      min_elev_deg: float = 10.0,
                      key_prefix: str = "GRID",
                      overwrite_existing: bool = True,
                      drop_old_trajectories: bool = True,
                      progress: bool = True,
                      verbose: bool = True):
        """
        Grid-search over (rp, ra, ta, raan, inc) with AOP fixed as (RAAN + 90°) mod 360.

        Ranges specify inclusive endpoints for linear spacing (n_* points each).
        Only (rp, ra) pairs with ra > rp are kept.
        """
        if self.sim.JD is None:
            raise ValueError("Build the Simulator timebase first (build_timebase).")
        if gs_key not in getattr(self.sim, "ground_stations", {}):
            raise KeyError(f"Ground station '{gs_key}' not found in sim.ground_stations")

        RE = self._earth_radius_km()

        # Helper: linspace; avoid duplicating end when full-circle
        def _lin(lo, hi, n, angle=False):
            if n <= 0:
                return np.array([], dtype=float)
            if angle and np.isclose((hi - lo) % 360.0, 0.0):
                return np.linspace(lo, hi, n, endpoint=False)
            return np.linspace(lo, hi, n, endpoint=True)

        rp_alts = _lin(*rp_range, n_rp, angle=False)
        ra_alts = _lin(*ra_range, n_ra, angle=False)
        ta_grid = _lin(*ta_range, n_ta, angle=True)
        raan_grid = _lin(*raan_range, n_raan, angle=True)
        inc_grid = _lin(*inc_range, n_inc, angle=False)

        # Build trial satellites (cap if requested)
        from Satellite import Satellite

        sats: dict[str, object] = {}
        params_by_key: dict[str, dict] = {}
        keys: list[str] = []

        count = 0
        cap = max_orbits if (max_orbits is not None and max_orbits > 0) else np.inf

        for rp_alt in rp_alts:
            for ra_alt in ra_alts:
                if ra_alt <= rp_alt:
                    continue
                rp_km = RE + float(rp_alt)
                ra_km = RE + float(ra_alt)

                for ta in ta_grid:
                    for raan in raan_grid:
                        aop = (float(raan) + 90.0) % 360.0  # enforce AOP = RAAN + 90°
                        for inc in inc_grid:
                            key = f"{key_prefix}{count:04d}"
                            sat = Satellite.from_peri_apo(
                                tru_deg=float(ta),
                                rp_km=rp_km,
                                ra_km=ra_km,
                                raan_deg=float(raan),
                                inc_deg=float(inc),
                                aop_deg=aop,
                            )
                            sats[key] = sat
                            params_by_key[key] = {
                                "rp_alt_km": float(rp_alt),
                                "ra_alt_km": float(ra_alt),
                                "ta_deg": float(ta),
                                "raan_deg": float(raan),
                                "inc_deg": float(inc),
                                "aop_deg": float(aop),
                            }
                            keys.append(key)

                            count += 1
                            if count >= cap:
                                break
                        if count >= cap: break
                    if count >= cap: break
            if count >= cap: break

        if count == 0:
            raise RuntimeError("No orbits generated: check ranges, grid counts, and ra > rp constraint.")

        # Add to simulator and propagate all trials + this GS
        self._add_sats(sats, overwrite=overwrite_existing, drop_old_traj=drop_old_trajectories)
        self.sim.run_all(sat_keys=keys, gs_keys=[gs_key], progress=progress)

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
                f"rp={p['rp_alt_km']:.0f} km, ra={p['ra_alt_km']:.0f} km, "
                f"inc={p['inc_deg']:.1f}°, raan={p['raan_deg']:.1f}°, "
                f"aop={p['aop_deg']:.1f}°, ta={p['ta_deg']:.1f}°"
            )

        # Save "best" on the instance for refinement later
        self.best = {
            "key": best_key,
            "params": best["params"].copy(),
            "percent_visible": float(best["percent_visible"]),
            "gs_key": gs_key,
        }

        return best, results

    def refine_best_local(self,
                        *,
                        steps: int = 300,
                        min_elev_deg: float = 10.0,
                        alt_bounds: tuple[float, float] = (400.0, 7500.0),
                        # per-parameter update scales (km, km, deg, deg, deg, deg)
                        lr0: tuple[float, float, float, float, float, float] = (50.0, 50.0, 2.0, 2.0, 2.0, 2.0),
                        # per-parameter perturbation scales for SPSA
                        delta0: tuple[float, float, float, float, float, float] = (25.0, 25.0, 1.0, 1.0, 1.0, 1.0),
                        decay: float = 0.95,
                        seed: int | None = None,
                        progress: bool = True,
                        verbose: bool = True,
                        lock_aop_to_raan_plus_90: bool = False,
                        lock_inclination: bool = False,
                        lock_perigee: bool = False,
                        # live plot / logging controls
                        live_plot: bool = True,
                        plot_every: int = 1,
                        log_path: str | None = "visibility_refine_log.txt",
                        save_plot_path: str | None = None):
        """
        SPSA-like local refinement to maximise visibility % from self.best['gs_key'].

        θ = [rp_alt_km, ra_alt_km, ta_deg, raan_deg, inc_deg, aop_deg]
        Locks:
        - lock_inclination: fix inc_deg to this value (deg) if provided.
        - lock_perigee:     fix rp_alt_km to this value (km) if provided.
        - lock_aop_to_raan_plus_90: enforce aop = (raan + 90°) mod 360 each step.
        """
        import matplotlib.pyplot as plt
        import os

        if not hasattr(self, "best") or not isinstance(self.best, dict):
            raise RuntimeError("No baseline found. Run optimize_grid(...) first.")
        gs_key = self.best.get("gs_key", None)
        if gs_key is None:
            raise RuntimeError("self.best is missing 'gs_key'. Re-run optimize_grid(...)")

        rng = np.random.default_rng(seed)
        RE = float(globals().get("R_EARTH", 6378.137))

        # Ensure GS trajectory exists once
        if gs_key not in getattr(self.sim, "ground_station_trajectories", {}):
            if hasattr(self.sim, "propagate_one"):
                self.sim.propagate_one(gs_key=gs_key)
            else:
                self.sim.run_all(sat_keys=[], gs_keys=[gs_key], progress=False)

        from Satellite import Satellite

        def _wrap_0_360(x): return float(x) % 360.0
        def _wrap_m180_180(x):
            y = (float(x) + 180.0) % 360.0 - 180.0
            if y == 180.0: y = -180.0
            return y

        # Determine which dims are locked
        # order: [rp, ra, ta, raan, inc, aop]
        # order: [rp, ra, ta, raan, inc, aop]
        locked = np.array([False, False, False, False, False, False])
        rp_lock_val = None
        inc_lock_val = None

        if lock_perigee:
            rp_lock_val = float(self.best["params"]["rp_alt_km"])
            locked[0] = True

        if lock_inclination:
            inc_lock_val = float(self.best["params"]["inc_deg"])
            locked[4] = True

        if lock_aop_to_raan_plus_90:
            locked[5] = True  # AOP is a function of RAAN; we freeze its own dimension
            
        free_idx = np.where(~locked)[0]
        if free_idx.size == 0:
            raise ValueError("All parameters are locked—nothing to optimise.")

        def _project_params(p):
            lo, hi = alt_bounds

            # rp / ra with constraints
            rp = float(np.clip(p[0], lo, hi))
            if locked[0]:  # perigee locked to current best
                rp = rp_lock_val

            ra = float(np.clip(p[1], lo, hi))

            # Only fix if strictly ra < rp (allow circular ra == rp)
            if ra < rp:
                if locked[0]:
                    # can't move rp; push ra up (bounded)
                    ra = min(hi, rp + 1.0)  # tiny separation
                else:
                    mid = 0.5 * (ra + rp)
                    eps = 1.0
                    rp = max(lo, mid - eps)
                    ra = min(hi, mid + eps)
                    if ra < rp:
                        ra = min(hi, rp + 1.0)

            # angles
            ta   = float(p[2]) % 360.0
            raan = (float(p[3]) + 180.0) % 360.0 - 180.0
            if raan == 180.0: raan = -180.0

            inc = float(p[4])
            if inc < 0.0: inc = 0.0
            if inc > 180.0: inc = 180.0
            if locked[4]:
                inc = inc_lock_val

            aop = float(p[5]) % 360.0
            if lock_aop_to_raan_plus_90:
                aop = (raan + 90.0) % 360.0

            return np.array([rp, ra, ta, raan, inc, aop], dtype=float)

        def _evaluate(params):
            """Return visibility percent (higher is better)."""
            rp_alt, ra_alt, ta, raan, inc, aop = params.tolist()
            sat = Satellite.from_peri_apo(
                tru_deg=float(ta),
                rp_km=RE + float(rp_alt),
                ra_km=RE + float(ra_alt),
                raan_deg=float(raan),
                inc_deg=float(inc),
                aop_deg=float(aop),
            )
            tmp_key = "_GD_OPT_"
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

        # Start from grid best
        p0 = np.array([
            self.best["params"]["rp_alt_km"],
            self.best["params"]["ra_alt_km"],
            self.best["params"]["ta_deg"],
            self.best["params"]["raan_deg"],
            self.best["params"]["inc_deg"],
            self.best["params"]["aop_deg"],
        ], dtype=float)

        # Apply initial locks if requested
        if locked[0]: p0[0] = rp_lock_val
        if locked[4]: p0[4] = inc_lock_val
        if locked[5]: p0[5] = _wrap_0_360(_wrap_m180_180(p0[3]) + 90.0)

        p = _project_params(p0)

        # Current/best
        best_p = p.copy()
        best_pct = _evaluate(best_p)

        # Live logging / plotting setup (same as before) ...........................
        import matplotlib.pyplot as plt
        import os
        f_log = None
        try:
            if log_path:
                f_log = open(log_path, "w", encoding="utf-8")
                f_log.write("# step\tpct_now\tbest_pct\trp_alt_km\tra_alt_km\tta_deg\traan_deg\tinc_deg\taop_deg\n")
                f_log.flush()
        except Exception as e:
            print(f"[warn] could not open log file '{log_path}': {e}")
            f_log = None

        if live_plot:
            plt.ion()
            fig, ax = plt.subplots()
            line_now,  = ax.plot([], [], lw=1.5, label="current %")
            line_best, = ax.plot([], [], lw=1.5, label="best %")
            ax.set_xlabel("Step")
            ax.set_ylabel(f"Visibility ≥{min_elev_deg:.0f}° [%]")
            ax.set_title("Local refinement (SPSA)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            xs_hist, now_hist, best_hist = [], [], []
            fig.canvas.draw(); fig.canvas.flush_events()
        else:
            xs_hist, now_hist, best_hist = [], [], []

        # Schedules
        lr = np.array(lr0, dtype=float)
        delta = np.array(delta0, dtype=float)

        it = tqdm(range(steps), desc="Refining (SPSA)", unit="step", leave=False) if progress else range(steps)
        for k in it:
            # Build perturb vector touching only free dims
            sign = np.ones(6, dtype=float)
            sign[free_idx] = rng.choice([-1.0, 1.0], size=free_idx.size)

            ck = np.zeros(6, dtype=float)   # perturb scales
            ck[free_idx] = delta[free_idx]

            # Two-sided probe
            p_plus  = _project_params(p + ck * sign)
            p_minus = _project_params(p - ck * sign)

            f_plus  = -_evaluate(p_plus)
            f_minus = -_evaluate(p_minus)

            # Gradient estimate on free dims only
            g = np.zeros(6, dtype=float)
            den = 2.0 * ck[free_idx] * sign[free_idx]
            den = np.where(np.abs(den) < 1e-12, np.sign(den) * 1e-12, den)
            g[free_idx] = (f_plus - f_minus) / den

            # Learning rates: zero for locked dims
            ak = np.zeros(6, dtype=float)
            ak[free_idx] = lr[free_idx]

            # Update and project
            p = _project_params(p - ak * g)

            # Decay schedules for free dims only
            lr[free_idx]    *= decay
            delta[free_idx] *= decay

            # Evaluate, track best
            pct_now = _evaluate(p)
            if pct_now > best_pct:
                best_pct = pct_now
                best_p = p.copy()

            # Logging
            if f_log is not None:
                try:
                    f_log.write(f"{k}\t{pct_now:.6f}\t{best_pct:.6f}\t"
                                f"{p[0]:.6f}\t{p[1]:.6f}\t{p[2]:.6f}\t{p[3]:.6f}\t{p[4]:.6f}\t{p[5]:.6f}\n")
                    f_log.flush(); os.fsync(f_log.fileno())
                except Exception as e:
                    print(f"[warn] log write failed at step {k}: {e}")

            # Live plot
            xs_hist.append(k); now_hist.append(pct_now); best_hist.append(best_pct)
            if live_plot and (k % max(1, plot_every) == 0):
                try:
                    line_now.set_data(xs_hist, now_hist)
                    line_best.set_data(xs_hist, best_hist)
                    ax.relim(); ax.autoscale_view()
                    fig.canvas.draw(); fig.canvas.flush_events()
                except Exception:
                    pass

            if progress:
                it.set_postfix_str(f"best={best_pct:.2f}%")

        # Save best and report
        self.best = {
            "key": self.best.get("key", "BEST"),
            "params": {
                "rp_alt_km": float(best_p[0]),
                "ra_alt_km": float(best_p[1]),
                "ta_deg": float(best_p[2]),
                "raan_deg": float(best_p[3]),
                "inc_deg": float(best_p[4]),
                "aop_deg": float(best_p[5]),
            },
            "percent_visible": float(best_pct),
            "gs_key": gs_key,
        }

        if live_plot and save_plot_path:
            try:
                fig.savefig(save_plot_path, dpi=140, bbox_inches="tight")
            except Exception as e:
                print(f"[warn] could not save plot to '{save_plot_path}': {e}")
        if live_plot:
            try:
                plt.ioff(); plt.show(block=False)
            except Exception:
                pass

        if f_log is not None:
            try: f_log.close()
            except Exception: pass

        if verbose:
            p = self.best["params"]
            print(
                f"[LOCAL BEST] vis ≥{min_elev_deg:.1f}°: {best_pct:.2f}%  |  "
                f"rp={p['rp_alt_km']:.1f} km, ra={p['ra_alt_km']:.1f} km, "
                f"inc={p['inc_deg']:.2f}°, raan={p['raan_deg']:.2f}°, "
                f"aop={p['aop_deg']:.2f}°, ta={p['ta_deg']:.2f}°"
            )
            self.sim.animate_3d(sat_keys=[self.best["key"]], step=10, camera_spin=True)

        return {"params": self.best["params"], "percent_visible": self.best["percent_visible"]}
