import numpy as np

def set_axis_limits_from_points(ax, all_R, R_EARTH: float = 6378.1363):
    """Set the axis limits for a 3D plot based on the given points.

    Args:
        ax (_type_): The 3D axis to set limits for.
        all_R (_type_): The points to consider for setting limits.
        R_EARTH (float, optional): The radius of the Earth. Defaults to 6378.1363.
    """    
    R = np.vstack(all_R)
    xs, ys, zs = R[:, 0], R[:, 1], R[:, 2]

    # If no valid data, just show Earth
    if not (np.isfinite(xs).any() and np.isfinite(ys).any() and np.isfinite(zs).any()):
        ax.set_xlim(-R_EARTH, R_EARTH)
        ax.set_ylim(-R_EARTH, R_EARTH)
        ax.set_zlim(-0.4 * R_EARTH, 0.4 * R_EARTH)
        return

    xs_min, xs_max = np.nanmin(xs), np.nanmax(xs)
    ys_min, ys_max = np.nanmin(ys), np.nanmax(ys)
    zs_min, zs_max = np.nanmin(zs), np.nanmax(zs)

    max_range = np.nanmax([xs_max - xs_min, ys_max - ys_min, zs_max - zs_min])
    if not np.isfinite(max_range) or max_range == 0:
        max_range = 1.0  # fallback

    # Ensure the cube fully contains the Earth
    max_range = max(max_range, 2 * R_EARTH)

    # Earth-centred: force centre at (0,0,0)
    ax.set_xlim(-0.5 * max_range, 0.5 * max_range)
    ax.set_ylim(-0.5 * max_range, 0.5 * max_range)
    ax.set_zlim(-0.4 * max_range, 0.4 * max_range)