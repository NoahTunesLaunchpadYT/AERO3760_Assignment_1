import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize

def linspace_range(lo: float, hi: float, n: int, *, angle_wrap=False) -> np.ndarray:
    """
    Evenly spaced grid. If angle_wrap=True and the range is a full wrap
    (e.g. 0..360 or 10..370), we exclude the endpoint to avoid duplicates.
    """
    if n <= 0:
        return np.array([], dtype=float)
    if angle_wrap and np.isclose((hi - lo) % 360.0, 0.0):
        return np.linspace(lo, hi, n, endpoint=False)
    return np.linspace(lo, hi, n, endpoint=True)

# ---------- 1) Generic 2D grid search with 3D surface plot ----------
def grid_search_2d(
    func,                               # callable f(x, y) -> float
    *,
    x_range: tuple[float, float],
    n_x: int,
    y_range: tuple[float, float],
    n_y: int,
    x_is_angle: bool = False,
    y_is_angle: bool = False,
    fig_size: tuple[float, float] = (7.5, 6.0),
    cmap: str = "viridis",
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "f(x, y)",
    title: str | None = None,
    progress: bool = True,
    return_figure: bool = False,
):
    """
    Runs a grid search over (x, y) for any 2-var function and plots the surface.

    Returns:
        best: dict with {'x', 'y', 'value'}
        results: list of dicts [{'x', 'y', 'value'}, ...]
        (fig) if return_figure=True
    """
    # Build grids
    x_grid = linspace_range(*x_range, n_x, angle_wrap=x_is_angle)
    y_grid = linspace_range(*y_range, n_y, angle_wrap=y_is_angle)

    # Evaluate
    results = []
    best_rec = {"x": None, "y": None, "value": -np.inf}
    iterator_x = x_grid
    iterator_y = y_grid

    # Optional progress print
    if progress:
        total = len(x_grid) * len(y_grid)
        done = 0

    for x in iterator_x:
        for y in iterator_y:
            val = float(func(x, y))
            rec = {"x": float(x), "y": float(y), "value": val}
            results.append(rec)
            if val > best_rec["value"]:
                best_rec = rec
            if progress:
                done += 1
                if done % max(1, total // 20) == 0:
                    print(f"[grid] {done}/{total} evaluated. Best so far: {best_rec['value']:.3f} at (x={best_rec['x']:.3f}, y={best_rec['y']:.3f})")

    if not results:
        raise RuntimeError("Empty grid. Check counts and ranges.")

    # Prepare data for plotting
    xs = np.array([r["x"] for r in results], dtype=float)
    ys = np.array([r["y"] for r in results], dtype=float)
    zs = np.array([r["value"] for r in results], dtype=float)

    tri = Triangulation(xs, ys)

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    if title:
        ax.set_title(title)

    # Axes limits: if a full wrap on angle, set a nice 0..360
    def _axis_limit(ax_setlim, lo, hi, is_angle):
        if is_angle and np.isclose((hi - lo) % 360.0, 0.0):
            ax_setlim(0.0, 360.0)
        else:
            ax_setlim(float(lo), float(hi))

    _axis_limit(ax.set_xlim, *x_range, x_is_angle)
    _axis_limit(ax.set_ylim, *y_range, y_is_angle)

    norm = Normalize(vmin=float(zs.min()), vmax=float(zs.max()))
    surf = ax.plot_trisurf(tri, zs, linewidth=0.1, antialiased=True, cmap=cmap, norm=norm)

    cbar = fig.colorbar(surf, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label(z_label)

    fig.tight_layout()
    plt.show()

    if return_figure:
        return best_rec, results, fig
    return best_rec, results

def refine_2d_generic(
    func,                               # callable: f(x, y) -> float (to MAXIMISE)
    *,
    # starting point
    x0: float,
    y0: float,
    # variable semantics
    x_is_angle: bool = False,
    y_is_angle: bool = False,
    # bounds (for non-angles, used to clip; for angles, optional for axis limits)
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    # optimisation schedule
    steps: int = 200,
    lr_x: float = 0.5,
    lr_y: float = 0.5,
    delta_x: float = 0.25,
    delta_y: float = 0.25,
    decay: float = 0.97,
    decay_deltas: bool = True,
    seed: int | None = None,
    progress: bool = True,
    verbose: bool = True,
    # plotting
    live_plot: bool = True,
    plot_every: int = 1,
    fig_size: tuple[float, float] = (12, 5),
    save_plot_path: str | None = None,
    # labels
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "score",
    title_left: str = "Fitness over time",
    title_right: str = "Objective surface (height & colour = score)",
    # debug visuals
    show_probes: bool = True,
    show_step_arrow: bool = True,
    clip_step_x: float | None = None,      # optional per-iter cap on step magnitude
    clip_step_y: float | None = None,
    # ---- NEW: zoom controls ----
    zoom: bool = True,                     # turn auto-zoom on/off
    zoom_window: int = 60,                 # how many most-recent points to frame
    zoom_pad_frac_xy: float = 0.12,        # ~12% padding on x/y ranges
    zoom_pad_frac_z: float = 0.15,         # ~15% padding on z range
    respect_bounds: bool = True,           # do not zoom beyond provided bounds
):
    """
    Objective-agnostic 2D SPSA gradient ascent with live plotting.
    Left: fitness vs step (current & best)
    Right: 3D trisurf of sampled (x, y) with score as height & colour.

    Auto-zoom keeps both panels tightly framed around the last N samples.

    Returns:
        best: dict(x=..., y=..., value=...)
        history: dict with arrays: steps, x, y, values, best_values
    """

    rng = np.random.default_rng(seed)

    # ------------- helpers -------------
    def _wrap_angle(v: float, base: float = 0.0) -> float:
        """Wrap to [base, base + 360)."""
        return (float(v) - base) % 360.0 + base

    def _project_scalar(v: float, is_angle: bool, bounds: tuple[float, float] | None) -> float:
        if is_angle:
            base = bounds[0] if bounds is not None else 0.0
            return _wrap_angle(v, base)
        if bounds is not None:
            lo, hi = float(bounds[0]), float(bounds[1])
            return float(np.clip(v, lo, hi))
        return float(v)

    def _project(x: float, y: float) -> tuple[float, float]:
        return (_project_scalar(x, x_is_angle, x_bounds),
                _project_scalar(y, y_is_angle, y_bounds))

    def _apply_zoom(ax_fit, ax3d, X, Y, Z):
        """Auto-zoom both panels around recent data with padding."""
        if X.size == 0:
            return
        # recent window
        Xw = X[-zoom_window:] if X.size > zoom_window else X
        Yw = Y[-zoom_window:] if Y.size > zoom_window else Y
        Zw = Z[-zoom_window:] if Z.size > zoom_window else Z

        # 1) Fitness panel y-lims
        y_min, y_max = float(np.min(Zw)), float(np.max(Zw))
        if np.isfinite([y_min, y_max]).all():
            if y_max == y_min:
                y_min -= 1.0
                y_max += 1.0
            pad_y = (y_max - y_min) * 0.08
            ax_fit.set_ylim(y_min - pad_y, y_max + pad_y)

        # 2) 3D panel x/y/z limits with padding, respecting bounds if asked
        def _pad_range(lo, hi, frac, min_span=1e-9):
            if not np.isfinite([lo, hi]).all():
                return lo, hi
            span = max(hi - lo, min_span)
            pad = span * frac
            return lo - pad, hi + pad

        xlo, xhi = float(np.min(Xw)), float(np.max(Xw))
        ylo, yhi = float(np.min(Yw)), float(np.max(Yw))
        zlo, zhi = float(np.min(Zw)), float(np.max(Zw))

        xlo, xhi = _pad_range(xlo, xhi, zoom_pad_frac_xy)
        ylo, yhi = _pad_range(ylo, yhi, zoom_pad_frac_xy)
        zlo, zhi = _pad_range(zlo, zhi, zoom_pad_frac_z)

        if respect_bounds and x_bounds is not None:
            xlo = max(xlo, float(x_bounds[0])); xhi = min(xhi, float(x_bounds[1]))
        if respect_bounds and y_bounds is not None:
            ylo = max(ylo, float(y_bounds[0])); yhi = min(yhi, float(y_bounds[1]))

        # Avoid degenerate limits
        if xhi <= xlo: xhi = xlo + 1e-6
        if yhi <= ylo: yhi = ylo + 1e-6
        if zhi <= zlo: zhi = zlo + 1e-6

        ax3d.set_xlim(xlo, xhi)
        ax3d.set_ylim(ylo, yhi)
        ax3d.set_zlim(zlo, zhi)

    # ------------- init -------------
    x, y = _project(x0, y0)
    val = float(func(x, y))
    best = {"x": x, "y": y, "value": val}
    best_val = val

    # histories
    step_hist, val_hist, best_hist = [0], [val], [best_val]
    xs, ys, zs = [x], [y], [val]  # points for 3D surface

    # ------------- plotting setup -------------
    if live_plot:
        plt.ion()
        fig = plt.figure(figsize=fig_size)
        # left: fitness timeline
        ax_fit = fig.add_subplot(1, 2, 1)
        ln_now,  = ax_fit.plot([], [], lw=1.5, label="current")
        ln_best, = ax_fit.plot([], [], lw=1.5, label="best")
        ax_fit.set_xlabel("Step"); ax_fit.set_ylabel(z_label); ax_fit.set_title(title_left)
        ax_fit.grid(True, alpha=0.3); ax_fit.legend(loc="lower right")

        # right: 3D surface
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
        ax3d.set_xlabel(x_label); ax3d.set_ylabel(y_label); ax3d.set_zlabel(z_label)
        ax3d.set_title(title_right)

        # initial axis limits (optional)
        if x_bounds is not None:
            ax3d.set_xlim(float(x_bounds[0]), float(x_bounds[1]))
        if y_bounds is not None:
            ax3d.set_ylim(float(y_bounds[0]), float(y_bounds[1]))

        norm = Normalize(vmin=val, vmax=val)
        tri = None
        surf = None
        sc_pts = ax3d.scatter([], [], [], s=10)
        cbar = None

        probe_line = probe_plus = probe_minus = step_seg = None
        if show_probes:
            probe_line, = ax3d.plot([], [], [], ":", lw=1.0, alpha=0.35)
            probe_plus,  = ax3d.plot([], [], [], "o", ms=4, alpha=0.7)
            probe_minus, = ax3d.plot([], [], [], "o", ms=4, alpha=0.7)
        if show_step_arrow:
            step_seg, = ax3d.plot([], [], [], "-", lw=1.2, alpha=0.6)

        fig.tight_layout(); fig.canvas.draw(); fig.canvas.flush_events()
    else:
        fig = ax_fit = ax3d = ln_now = ln_best = None
        norm = tri = surf = sc_pts = cbar = None
        probe_line = probe_plus = probe_minus = step_seg = None

    # ------------- schedules -------------
    lr_x_k, lr_y_k = float(lr_x), float(lr_y)
    dx_k,  dy_k    = float(delta_x), float(delta_y)

    # ------------- main loop -------------
    for k in range(1, steps + 1):
        x0k, y0k = x, y

        sx = rng.choice([-1.0, 1.0])
        sy = rng.choice([-1.0, 1.0])

        xp, yp = _project(x0k + dx_k * sx, y0k + dy_k * sy)
        xm, ym = _project(x0k - dx_k * sx, y0k - dy_k * sy)

        vp = float(func(xp, yp))
        vm = float(func(xm, ym))

        ge = ((-vp) - (-vm)) / (2.0 * dx_k * sx)
        gy = ((-vp) - (-vm)) / (2.0 * dy_k * sy)

        step_x = lr_x_k * ge
        step_y = lr_y_k * gy
        if clip_step_x is not None:
            step_x = float(np.clip(step_x, -abs(clip_step_x),  abs(clip_step_x)))
        if clip_step_y is not None:
            step_y = float(np.clip(step_y, -abs(clip_step_y),  abs(clip_step_y)))

        x, y = _project(x0k - step_x, y0k - step_y)
        val  = float(func(x, y))

        if val > best_val:
            best_val = val
            best = {"x": x, "y": y, "value": val}

        step_hist.append(k); val_hist.append(val); best_hist.append(best_val)
        xs.append(x); ys.append(y); zs.append(val)

        if live_plot and (k % max(1, plot_every) == 0):
            try:
                # Left panel
                ln_now.set_data(step_hist, val_hist)
                ln_best.set_data(step_hist, best_hist)
                if zoom:
                    _apply_zoom(ax_fit, ax3d,
                                np.asarray(xs), np.asarray(ys), np.asarray(zs))
                else:
                    ax_fit.relim(); ax_fit.autoscale_view()

                # Right panel
                X = np.asarray(xs, dtype=float)
                Y = np.asarray(ys, dtype=float)
                Z = np.asarray(zs, dtype=float)

                norm.vmin = float(Z.min())
                norm.vmax = float(Z.max())

                if X.size >= 3:
                    tri = Triangulation(X, Y)
                    if surf is not None:
                        try: surf.remove()
                        except Exception: pass
                    surf = ax3d.plot_trisurf(tri, Z, linewidth=0.1, antialiased=True,
                                             cmap="viridis", norm=norm)
                    if cbar is None:
                        cbar = fig.colorbar(surf, ax=ax3d, pad=0.02, shrink=0.8)
                        cbar.set_label(z_label)
                    else:
                        cbar.update_normal(surf)

                sc_pts._offsets3d = (X, Y, Z)

                if show_probes:
                    probe_line.set_data_3d([xm, xp], [ym, yp], [vm, vp])
                    probe_plus.set_data_3d([xp], [yp], [vp])
                    probe_minus.set_data_3d([xm], [ym], [vm])
                if show_step_arrow:
                    step_seg.set_data_3d([x0k, x], [y0k, y], [best_hist[-2], val_hist[-1]])

                fig.canvas.draw(); fig.canvas.flush_events()
            except Exception:
                pass

        lr_x_k *= decay; lr_y_k *= decay
        if decay_deltas:
            dx_k *= decay; dy_k *= decay

        if progress and ((k % max(1, steps // 20)) == 0 or k == steps):
            print(f"[refine] step {k}/{steps} | best={best_val:.4f} @ ({best['x']:.4f}, {best['y']:.4f})")

    if live_plot and save_plot_path:
        try:
            fig.savefig(save_plot_path, dpi=140, bbox_inches="tight")
        except Exception as exc:
            print(f"[warn] could not save plot to '{save_plot_path}': {exc}")
        try:
            plt.ioff(); plt.show(block=False)
        except Exception:
            pass

    if verbose:
        print(f"[LOCAL BEST] {z_label}: {best['value']:.6f} at ({x_label}={best['x']:.6f}, {y_label}={best['y']:.6f})")

    history = {
        "steps": np.asarray(step_hist, dtype=int),
        "x":     np.asarray(xs, dtype=float),
        "y":     np.asarray(ys, dtype=float),
        "values": np.asarray(zs, dtype=float),
        "best_values": np.asarray(best_hist, dtype=float),
    }
    return best, history