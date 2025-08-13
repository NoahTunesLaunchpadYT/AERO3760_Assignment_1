import numpy as np
from dataclasses import dataclass

@dataclass
class CoverageResult:
    shifts_idx: list[int]          # chosen shifts in index units (0..m-1)
    shifts_deg: list[float]        # chosen shifts in degrees
    union_mask: np.ndarray         # bool, same length as input mask
    coverage_pct: float            # 0..100
    covered_all: bool              # True if union covers all bins (all True)
    gains: list[int]               # marginal gains at each chosen step

def cover_circle_with_mask(mask: np.ndarray,
                           n: int,
                           *,
                           require_full: bool = False,
                           plot: bool = True,
                           fig_size: tuple[float, float] = (11, 6),
                           save_plot_path: str | None = None,
                           return_figure: bool = False) -> CoverageResult:
    """
    Given a boolean mask over [0, 360) sampled uniformly (length m),
    choose up to n cyclic shifts of the mask to maximise union coverage.

    Optionally plots:
      - Top: union mask (0/1) as a step plot over angle.
      - Bottom: raster of each chosen shift (rows) showing where it's True.

    Returns:
        CoverageResult (and optionally the figure if return_figure=True).
    """
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("mask must be a non-empty 1D boolean array")
    if n <= 0:
        raise ValueError("n must be >= 1")

    m = mask.size
    deg_per_bin = 360.0 / m

    # If mask is all False, nothing can be covered regardless of shifts.
    if not mask.any():
        res = CoverageResult(
            shifts_idx=[],
            shifts_deg=[],
            union_mask=np.zeros_like(mask, dtype=bool),
            coverage_pct=0.0,
            covered_all=False,
            gains=[]
        )
        if plot:
            import matplotlib.pyplot as plt
            deg = np.arange(m) * deg_per_bin
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True,
                                           gridspec_kw={"height_ratios": [2, 3]})
            ax1.step(deg, np.zeros(m, dtype=int), where="post")
            ax1.set_ylabel("Union")
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_title("Coverage: 0.00% (no valid windows)")
            ax1.grid(True, axis="y", alpha=0.3)

            ax2.imshow(np.zeros((1, m), dtype=int), aspect="auto", interpolation="nearest")
            ax2.set_yticks([0], labels=["(none)"])
            ax2.set_xlabel("Angle [deg]")
            ax2.set_xlim(0, 360)
            fig.tight_layout()
            if save_plot_path:
                try:
                    fig.savefig(save_plot_path, dpi=160, bbox_inches="tight")
                except Exception as exc:
                    print(f"[warn] could not save plot to '{save_plot_path}': {exc}")
            plt.show()
            if return_figure:
                return res, fig
        return res

    union = np.zeros_like(mask, dtype=bool)
    chosen_shifts: list[int] = []
    chosen_deg: list[float] = []
    gains: list[int] = []

    for _ in range(n):
        not_covered = ~union
        best_gain = -1
        best_shift = 0

        # Brute-force over all m shifts
        for s in range(m):
            rolled = np.roll(mask, s)
            gain = int(np.count_nonzero(rolled & not_covered))
            if gain > best_gain:
                best_gain = gain
                best_shift = s

        if best_gain <= 0:
            break

        union |= np.roll(mask, best_shift)
        chosen_shifts.append(best_shift)
        chosen_deg.append(best_shift * deg_per_bin)
        gains.append(best_gain)

        if require_full and union.all():
            break

    coverage_pct = 100.0 * float(np.count_nonzero(union)) / float(m)
    covered_all = bool(union.all())

    result = CoverageResult(
        shifts_idx=chosen_shifts,
        shifts_deg=chosen_deg,
        union_mask=union,
        coverage_pct=coverage_pct,
        covered_all=covered_all,
        gains=gains
    )

    # -------- Plotting the overlays + union --------
    if plot:
        import matplotlib.pyplot as plt

        k = len(chosen_shifts)
        deg = np.arange(m) * deg_per_bin
        overlays = np.vstack([np.roll(mask, s) for s in chosen_shifts]) if k else np.zeros((1, m), dtype=bool)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=fig_size, sharex=True, gridspec_kw={"height_ratios": [2, 3]}
        )

        # Top: union (0/1) as a step plot
        ax1.step(deg, union.astype(int), where="post")
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_yticks([0, 1], labels=["Off", "On"])
        ax1.set_ylabel("Union")
        title_bits = [f"Coverage: {coverage_pct:.2f}%"]
        if k:
            title_bits.append(f"Shifts: {', '.join(f'{d:.1f}°' for d in chosen_deg)}")
        ax1.set_title(" | ".join(title_bits))
        ax1.grid(True, axis="y", alpha=0.3)

        # Bottom: raster image of each chosen shift (rows = shifts)
        # Use imshow so True shows as 1, False as 0. Keep default colormap.
        ax2.imshow(overlays.astype(int), aspect="auto", interpolation="nearest",
                   extent=[0, 360, -0.5, k - 0.5] if k else [0, 360, -0.5, 0.5])
        if k:
            ax2.set_yticks(range(k), labels=[f"{d:.1f}°" for d in chosen_deg])
        else:
            ax2.set_yticks([0], labels=["(none)"])
        ax2.set_ylabel("Shift (deg)")
        ax2.set_xlabel("Angle [deg]")
        ax2.grid(False)

        fig.tight_layout()

        if save_plot_path:
            try:
                fig.savefig(save_plot_path, dpi=160, bbox_inches="tight")
            except Exception as exc:
                print(f"[warn] could not save plot to '{save_plot_path}': {exc}")

        plt.show()

        if return_figure:
            return result, fig

    return result
