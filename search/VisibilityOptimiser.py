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
                decay: float = 0.97,               # geometric decay of lr
                decay_deltas: bool = True,         # optionally decay deltas too
                seed: int | None = None,
                progress: bool = True,
                verbose: bool = True,
                min_perigee_alt_km: float = 200.0,
                # --- live plotting / logging ---
                live_plot: bool = True,
                plot_every: int = 1,
                fig_size: tuple[float, float] = (12, 5),
                log_path: str | None = "visibility_refine2d_log.txt",
                save_plot_path: str | None = None,
                # --- SPSA debug/visualisation ---
                clip_ta_step_deg: float | None = None,   # e.g. 2.0 caps TA step magnitude
                show_probes: bool = True,                 # draw plus/minus probe on param-space plot
                show_step_arrow: bool = True,             # draw the current step segment
                log_perturbations: bool = True,
                perturb_log_path: str | None = "visibility_refine2d_perturb.tsv"):
        """
        2-D SPSA refiner over (e, ta_deg) starting from self.best.

        Fixed (from self.best['params']): a_km, raan_deg, inc_deg, aop_deg
        Optimised: e  (bounded by perigee floor), ta_deg (wrapped to [0, 360))

        Live plots:
        - Left: visibility % vs step (current & best)
        - Right: path through (ta_deg, e), point color = visibility %
            + optional dotted line between SPSA probe points, and a short segment
            showing the applied step.

        Logs:
        - log_path: per-step summary (current %, best %, e, ta, fixed params)
        - perturb_log_path: rich SPSA info (signs, deltas, probes, vis_plus/minus, gradient, step, new point)
        """
        # ------------------------- imports -------------------------
        from tqdm.auto import tqdm
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from Satellite import Satellite

        # ------------------------- checks --------------------------
        if not getattr(self, "best", None):
            raise RuntimeError("Run optimize_grid(...) first; self.best is not set.")
        gs_key = self.best.get("gs_key")
        if gs_key is None:
            raise RuntimeError("self.best['gs_key'] missing; re-run grid search.")

        # Ensure GS trajectory exists once
        if gs_key not in getattr(self.sim, "ground_station_trajectories", {}):
            if hasattr(self.sim, "propagate_one"):
                self.sim.propagate_one(gs_key=gs_key)
            else:
                self.sim.run_all(sat_keys=[], gs_keys=[gs_key], progress=False)

        # ------------------------- helpers -------------------------
        def _wrap_ta(x_deg: float) -> float:
            return float(x_deg) % 360.0

        def _e_max_physical(a_km_: float) -> float:
            RE_ = float(globals().get("R_EARTH", 6378.137))
            return 1.0 - (RE_ + float(min_perigee_alt_km)) / float(a_km_)

        def _project(e_: float, ta_: float, e_max_: float) -> tuple[float, float]:
            e_ = max(0.05, min(e_max_, float(e_)))
            ta_ = _wrap_ta(ta_)
            return e_, ta_

        def _score(a_km_: float, raan_: float, inc_: float, aop_: float,
                e_: float, ta_: float) -> float:
            """Build+propagate a temp sat at (e,ta), return visibility %."""
            sat = Satellite.from_keplerian(
                a_km=a_km_, e=e_, inc_deg=inc_,
                raan_deg=raan_, aop_deg=aop_, tru_deg=ta_
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

        # ------------------------- setup ---------------------------
        pfix = self.best["params"]
        a_km     = float(pfix["a_km"])
        raan_deg = float(pfix["raan_deg"])
        inc_deg  = float(pfix["inc_deg"])
        aop_deg  = float(pfix["aop_deg"])

        e        = float(pfix["e"])
        ta_deg   = float(pfix["ta_deg"])

        e_max = _e_max_physical(a_km)
        if e_max <= 0.0:
            raise ValueError("a_km too small for the requested perigee floor.")

        # Project start and baseline score
        e, ta_deg = _project(e, ta_deg, e_max)
        score_fn = lambda ee, tt: _score(a_km, raan_deg, inc_deg, aop_deg, * _project(ee, tt, e_max))

        rng = np.random.default_rng(seed)
        best_e, best_ta = e, ta_deg
        best_pct = score_fn(e, ta_deg)

        # ------------------------- logging -------------------------
        def _open_tsv(path: str | None, header: str):
            if not path: return None
            try:
                f = open(path, "w", encoding="utf-8")
                f.write(header + "\n"); f.flush()
                return f
            except Exception as exc:
                print(f"[warn] could not open log file '{path}': {exc}")
                return None

        # Per-step summary log
        f_log = _open_tsv(
            log_path,
            "# step\tpct_now\tbest_pct\te\tta_deg\ta_km\traan_deg\tinc_deg\taop_deg"
        )

        # Rich SPSA perturbation log (one row per iter)
        f_pert = None
        if log_perturbations:
            f_pert = _open_tsv(
                perturb_log_path,
                "# step\ts_e\ts_ta\tde_k\tdta_k\t"
                "e\tta_deg\te_plus\tta_plus\te_minus\tta_minus\t"
                "vis_plus\tvis_minus\tgrad_e\tgrad_ta\td_e\td_ta\te_new\tta_new\tvis_now\tbest_pct"
            )

        def _log_step(step, pct_now):
            if f_log is None: return
            try:
                f_log.write(f"{step}\t{pct_now:.6f}\t{best_pct:.6f}\t"
                            f"{e:.8f}\t{ta_deg:.6f}\t"
                            f"{a_km:.3f}\t{raan_deg:.3f}\t{inc_deg:.3f}\t{aop_deg:.3f}\n")
                f_log.flush(); os.fsync(f_log.fileno())
            except Exception as exc:
                print(f"[warn] log write failed at step {step}: {exc}")

        def _log_pert(step, s_e, s_ta, de_k, dta_k,
                    e0, ta0, e_p, ta_p, e_m, ta_m,
                    vis_p, vis_m, ge, gta, d_e, d_ta, e1, ta1, vis_now):
            if f_pert is None: return
            try:
                f_pert.write(
                    f"{step}\t{s_e:+.0f}\t{s_ta:+.0f}\t{de_k:.6f}\t{dta_k:.6f}\t"
                    f"{e0:.8f}\t{ta0:.6f}\t{e_p:.8f}\t{ta_p:.6f}\t{e_m:.8f}\t{ta_m:.6f}\t"
                    f"{vis_p:.6f}\t{vis_m:.6f}\t{ge:.6f}\t{gta:.6f}\t"
                    f"{d_e:.6f}\t{d_ta:.6f}\t{e1:.8f}\t{ta1:.6f}\t{vis_now:.6f}\t{best_pct:.6f}\n"
                )
                f_pert.flush(); os.fsync(f_pert.fileno())
            except Exception as exc:
                print(f"[warn] perturb log write failed at step {step}: {exc}")

        # ------------------------- plotting ------------------------
        if live_plot:
            plt.ion()
            fig, (ax_fit, ax_space) = plt.subplots(1, 2, figsize=fig_size)

            # Fitness
            ln_now,  = ax_fit.plot([], [], lw=1.5, label="current %")
            ln_best, = ax_fit.plot([], [], lw=1.5, label="best %")
            ax_fit.set_xlabel("Step")
            ax_fit.set_ylabel(f"Visibility ≥{min_elev_deg:.0f}° [%]")
            ax_fit.set_title("2D refinement: fitness over time")
            ax_fit.grid(True, alpha=0.3); ax_fit.legend(loc="lower right")

            # Param space
            ax_space.set_xlabel("True anomaly, ta [deg]")
            ax_space.set_ylabel("Eccentricity, e")
            ax_space.set_title("2D parameter space (color = visibility %)")
            ax_space.set_xlim(0, 360); ax_space.set_ylim(0.0, e_max * 1.02)
            ax_space.grid(True, alpha=0.3)

            norm = Normalize(vmin=0.0, vmax=max(100.0, best_pct))
            sc = ax_space.scatter([], [], c=[], s=24, cmap="viridis", norm=norm)
            path_line, = ax_space.plot([], [], lw=0.9, alpha=0.6)
            best_marker, = ax_space.plot([best_ta], [best_e], marker="*", ms=10)

            # NEW: probe visuals + step segment (reused each iter; not accumulating)
            probe_line = None
            probe_plus_pt = None
            probe_minus_pt = None
            step_seg = None
            if show_probes:
                probe_line, = ax_space.plot([], [], ":", lw=1.1, alpha=0.35)
                probe_plus_pt,  = ax_space.plot([], [], "o", ms=4, alpha=0.7)
                probe_minus_pt, = ax_space.plot([], [], "o", ms=4, alpha=0.7)
            if show_step_arrow:
                step_seg, = ax_space.plot([], [], "-", lw=1.2, alpha=0.6)

            # histories
            xs_hist, now_hist, best_hist = [], [], []
            ta_hist, e_hist, vis_hist = [], [], []

            fig.tight_layout(); fig.canvas.draw(); fig.canvas.flush_events()
        else:
            fig = None
            xs_hist, now_hist, best_hist = [], [], []
            ta_hist, e_hist, vis_hist = [], [], []
            # dummy placeholders:
            ax_fit = ln_now = ln_best = ax_space = sc = path_line = best_marker = None
            probe_line = probe_plus_pt = probe_minus_pt = step_seg = None

        # ------------------------- schedules -----------------------
        lr_e_k, lr_ta_k = float(lr_e), float(lr_ta)
        de_k, dta_k     = float(delta_e), float(delta_ta)

        iterator = tqdm(range(steps), desc="Refining 2D (SPSA)", unit="step", leave=False) if progress else range(steps)
        for k in iterator:
            e0, ta0 = e, ta_deg

            # SPSA perturbation signs
            s_e  = rng.choice([-1.0, 1.0])
            s_ta = rng.choice([-1.0, 1.0])

            # Two-sided probe (project each)
            e_p, ta_p = _project(e0 + de_k * s_e,  ta0 + dta_k * s_ta, e_max)
            e_m, ta_m = _project(e0 - de_k * s_e,  ta0 - dta_k * s_ta, e_max)

            vis_plus  = score_fn(e_p, ta_p)
            vis_minus = score_fn(e_m, ta_m)
            f_plus  = -vis_plus
            f_minus = -vis_minus

            # Proper signed denominators (don’t clamp with max)
            ge  = (f_plus - f_minus) / (2.0 * de_k  * s_e)
            gta = (f_plus - f_minus) / (2.0 * dta_k * s_ta)

            # Step (with optional TA clip)
            d_e  =  lr_e_k * ge
            d_ta =  lr_ta_k * gta
            if clip_ta_step_deg is not None:
                d_ta = float(np.clip(d_ta, -abs(clip_ta_step_deg), abs(clip_ta_step_deg)))

            e, ta_deg = _project(e0 - d_e, ta0 - d_ta, e_max)

            # Evaluate & track best
            vis_now = score_fn(e, ta_deg)
            if vis_now > best_pct:
                best_pct = vis_now
                best_e, best_ta = e, ta_deg

            # --- logging ---
            _log_step(k, vis_now)
            _log_pert(k, s_e, s_ta, de_k, dta_k,
                    e0, ta0, e_p, ta_p, e_m, ta_m,
                    vis_plus, vis_minus, ge, gta, d_e, d_ta, e, ta_deg, vis_now)

            # --- histories for plots ---
            xs_hist.append(k); now_hist.append(vis_now); best_hist.append(best_pct)
            ta_hist.append(ta_deg); e_hist.append(e); vis_hist.append(vis_now)

            # --- live plot update (throttled) ---
            if live_plot and (k % max(1, plot_every) == 0):
                try:
                    # fitness
                    ln_now.set_data(xs_hist, now_hist)
                    ln_best.set_data(xs_hist, best_hist)
                    ax_fit.relim(); ax_fit.autoscale_view()

                    # scatter + path
                    import numpy as _np
                    offs = _np.column_stack([ta_hist, e_hist])
                    sc.set_offsets(offs)
                    sc.set_array(_np.array(vis_hist, dtype=float))
                    sc.set_clim(vmin=min(vis_hist), vmax=max(vis_hist))
                    path_line.set_data(ta_hist, e_hist)
                    best_marker.set_data([best_ta], [best_e])

                    # NEW: probes & step segment visuals
                    if show_probes:
                        probe_line.set_data([ta_m, ta_p], [e_m, e_p])
                        probe_plus_pt.set_data([ta_p], [e_p])
                        probe_minus_pt.set_data([ta_m], [e_m])
                    if show_step_arrow:
                        step_seg.set_data([ta0, ta_deg], [e0, e])

                    fig.canvas.draw(); fig.canvas.flush_events()
                except Exception:
                    pass

            # Decay schedules
            lr_e_k  *= decay
            lr_ta_k *= decay
            if decay_deltas:
                de_k    *= decay
                dta_k   *= decay

            if progress:
                iterator.set_postfix_str(f"best={best_pct:.2f}%  e={best_e:.4f} ta={best_ta:.1f}°")

        # ------------------------- wrap up -------------------------
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

        for f in (f_log, f_pert):
            if f is not None:
                try: f.close()
                except Exception: pass

        if verbose:
            print(f"[2D LOCAL BEST] vis ≥{min_elev_deg:.1f}°: {best_pct:.2f}%  |  "
                f"a={a_km:.2f} km, e={best_e:.5f}, ta={best_ta:.2f}°, "
                f"inc={inc_deg:.2f}°, raan={raan_deg:.2f}°, aop={aop_deg:.2f}°")

        return self.best
