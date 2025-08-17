from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class CoverageResult:
    shifts_idx: list[int]          # chosen shifts in index units (0..m-1)
    shifts_deg: list[float]        # chosen shifts in degrees
    union_mask: np.ndarray         # bool, same length as input mask
    coverage_pct: float            # 0..100
    covered_all: bool              # True if union covers all bins (all True)
    gains: list[int]               # marginal gains at each chosen step

# ---------------------------
# 1) ALGORITHM (swappable)
# ---------------------------
def greedy_phase_select(mask: np.ndarray,
                        n: int,
                        *,
                        require_full: bool = False) -> tuple[list[int], np.ndarray, list[int]]:
    """
    Greedy set-cover style: at each step choose the shift that adds
    the most new coverage to the union.

    Returns:
        (shifts_idx, union_mask, gains)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("mask must be a non-empty 1D boolean array")
    if n <= 0:
        raise ValueError("n must be >= 1")

    m = mask.size
    union = np.zeros_like(mask, dtype=bool)
    chosen_shifts: list[int] = []
    gains: list[int] = []

    if not mask.any():
        # Nothing to cover; return early with empty selection
        return chosen_shifts, union, gains

    for _ in range(n):
        not_covered = ~union
        best_gain = -1
        best_shift = 0
        # Brute-force all m cyclic shifts
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
        gains.append(best_gain)

        if require_full and union.all():
            break

    return chosen_shifts, union, gains

# -------------------------------------------
# 2) PROCESSING + (optional) PLOTTING
# -------------------------------------------
def cover_circle_process(mask: np.ndarray,
                         n: int,
                         *,
                         strategy=greedy_phase_select,      # <-- inject any algorithm with same signature
                         require_full: bool = False,
                         plot: bool = True,
                         fig_size: tuple[float, float] = (11, 6),
                         save_plot_path: str | None = None,
                         return_figure: bool = False) -> CoverageResult | tuple[CoverageResult, plt.Figure]:
    """
    Validates inputs, runs a phase-selection strategy, computes degrees & coverage,
    and optionally plots the union + overlays.

    Args:
        mask: boolean array over [0, 360) sampled uniformly (length m)
        n: number of copies (shifts) to overlay
        strategy: callable (mask, n, require_full=...) -> (shifts_idx, union_mask, gains)
        require_full: stop early if full coverage is reached (strategy-dependent)
        plot: show plots
        fig_size, save_plot_path, return_figure: plotting controls

    Returns:
        CoverageResult (and optionally the matplotlib Figure if return_figure=True)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("mask must be a non-empty 1D boolean array")
    if n <= 0:
        raise ValueError("n must be >= 1")

    m = mask.size
    deg_per_bin = 360.0 / m

    # Run the algorithm
    shifts_idx, union_mask, gains = strategy(mask, n, require_full=require_full)

    # Degrees & stats
    shifts_deg = [s * deg_per_bin for s in shifts_idx]
    coverage_pct = 100.0 * float(np.count_nonzero(union_mask)) / float(m)
    covered_all = bool(union_mask.all())

    result = CoverageResult(
        shifts_idx=shifts_idx,
        shifts_deg=shifts_deg,
        union_mask=union_mask,
        coverage_pct=coverage_pct,
        covered_all=covered_all,
        gains=gains
    )

    # Optional plotting
    if plot:
        k = len(shifts_idx)
        deg = np.arange(m) * deg_per_bin
        overlays = np.vstack([np.roll(mask, s) for s in shifts_idx]) if k else np.zeros((1, m), dtype=bool)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=fig_size, sharex=True, gridspec_kw={"height_ratios": [2, 3]}
        )

        # Top: union (0/1)
        ax1.step(deg, union_mask.astype(int), where="post")
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_yticks([0, 1], labels=["Off", "On"])
        ax1.set_ylabel("Union")
        title_bits = [f"Coverage: {coverage_pct:.2f}%"]
        if k:
            title_bits.append(f"Shifts: {', '.join(f'{d:.1f}°' for d in shifts_deg)}")
        ax1.set_title(" | ".join(title_bits))
        ax1.grid(True, axis="y", alpha=0.3)

        # Bottom: raster of chosen shifts
        ax2.imshow(
            overlays.astype(int),
            aspect="auto",
            interpolation="nearest",
            extent=[0, 360, -0.5, max(k - 0.5, 0.5)]
        )
        if k:
            ax2.set_yticks(range(k), labels=[f"{d:.1f}°" for d in shifts_deg])
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
