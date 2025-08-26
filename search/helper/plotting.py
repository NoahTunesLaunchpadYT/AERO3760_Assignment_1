import numpy as np

def set_axis_limits_from_points(ax, all_R):
    R = np.vstack(all_R)
    xs, ys, zs = R[:, 0], R[:, 1], R[:, 2]

    # If any axis has no finite data, fall back to a small cube
    if not (np.isfinite(xs).any() and np.isfinite(ys).any() and np.isfinite(zs).any()):
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.8, 0.8)
        return

    xs_min, xs_max = np.nanmin(xs), np.nanmax(xs)
    ys_min, ys_max = np.nanmin(ys), np.nanmax(ys)
    zs_min, zs_max = np.nanmin(zs), np.nanmax(zs)

    xs_mean, ys_mean, zs_mean = np.nanmean(xs), np.nanmean(ys), np.nanmean(zs)

    max_range = np.nanmax([xs_max - xs_min, ys_max - ys_min, zs_max - zs_min])
    if not np.isfinite(max_range) or max_range == 0:
        max_range = 1.0  # simple fallback

    mid = np.array([xs_mean, ys_mean, zs_mean], dtype=float)

    ax.set_xlim(mid[0] - 0.5 * max_range, mid[0] + 0.5 * max_range)
    ax.set_ylim(mid[1] - 0.5 * max_range, mid[1] + 0.5 * max_range)
    ax.set_zlim(mid[2] - 0.4 * max_range, mid[2] + 0.4 * max_range)