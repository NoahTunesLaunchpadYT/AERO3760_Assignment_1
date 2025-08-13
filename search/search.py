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
