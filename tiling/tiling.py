#!/usr/bin/env python3
"""
tile_mask_1d.py

Greedy tiling of a 1-D boolean mask with itself (n copies) using circular
shifts (wraparound on a ring) to maximise union coverage. Targets 100% coverage
if achievable.

Inputs:
  - mask.npy : 1-D boolean array

Outputs:
  - union.npy              : final boolean union (length L)
  - placements.json        : list of {"shift": int}
  - coverage_report.txt    : text summary
  - mask.png, union.png    : quick stripe visuals (optional; if Pillow available)

Usage:
  python tile_mask_1d.py --mask mask.npy --n 10 --progress --verbose

Notes:
  - Circular (torus) model: all shifts are modulo L.
  - At each step, we choose the shift of the mask that covers the maximum
    number of currently-uncovered positions (greedy).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False


def load_mask_1d(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    # Squeeze accidental extra dims like (L,1) or (1,L)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"mask must be 1-D, got shape {arr.shape}")
    if arr.dtype != np.bool_:
        arr = arr.astype(bool)
    return arr


def circ_corr_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Circular cross-correlation of two real 1-D arrays:
      c[k] = sum_i a[i] * b[(i+k) mod L]
    Implemented via FFT for speed.
    Returns real-valued array of length L.
    """
    L = a.size
    Fa = np.fft.rfft(a)
    Fb = np.fft.rfft(b)
    c = np.fft.irfft(Fa * np.conj(Fb), n=L)
    return np.real(c)


def greedy_tile_1d(mask: np.ndarray, n: int, verbose: bool, progress: bool):
    """
    Place up to n circular shifts of `mask` to maximise union coverage.

    Returns:
        union (bool, length L)
        placements (list of dict with {"shift": int})
        coverages (list of float in [0,1])  # cumulative coverage after each placement
    """
    L = mask.size
    if L == 0:
        raise ValueError("mask length is zero.")

    if not mask.any():
        # No true bits at all: impossible to gain coverage
        return mask.copy(), [], []

    union = np.zeros(L, dtype=bool)
    placements = []
    coverages = []

    total = L
    it = range(n)
    if progress and HAVE_TQDM:
        it = tqdm(it, desc="Tiling (1D)", leave=True)

    for step in it:
        # Remaining uncovered positions as float (1.0 where uncovered)
        remaining = (~union).astype(float)

        # Gain for shifting mask by k equals (# of mask-ones landing on remaining ones)
        # That's circular correlation remaining (*) mask.
        corr = circ_corr_1d(remaining, mask.astype(float))
        k = int(np.argmax(corr))
        # Apply this shift
        shifted = np.roll(mask, k)

        gain = int(np.count_nonzero(shifted & (~union)))
        if gain <= 0:
            if verbose:
                print(f"[info] step {step+1}/{n}: no additional coverage; stopping.")
            break

        union |= shifted
        cov = np.count_nonzero(union) / float(total)
        placements.append({"shift": k})
        coverages.append(cov)

        if verbose:
            print(f"[step {step+1}/{n}] +gain={gain:>4d}  coverage={cov*100:6.2f}%  shift={k}")

        if progress and HAVE_TQDM:
            it.set_postfix_str(f"cov={cov*100:.2f}%")

        if cov >= 1.0 - 1e-12:
            if verbose:
                print("[done] Reached ~100% coverage.")
            break

    return union, placements, coverages


def save_visual_1d(arr: np.ndarray, path: Path, height: int = 16):
    """
    Save a simple 1-D stripe PNG for quick inspection.
    White=True, Black=False.
    """
    if not HAVE_PIL:
        return
    L = arr.size
    img = (arr.astype(np.uint8) * 255).reshape(1, L)
    img = np.repeat(img, height, axis=0)
    Image.fromarray(img, mode="L").save(path)


def main():
    ap = argparse.ArgumentParser(description="Greedy circular tiling of a 1-D boolean mask with itself to maximise union coverage.")
    ap.add_argument("--mask", type=Path, default=Path("mask.npy"), help="Path to 1-D boolean mask .npy")
    ap.add_argument("--n", type=int, default=10, help="Maximum number of copies to place")
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bar if available")
    ap.add_argument("--verbose", action="store_true", help="Print per-step details")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory")
    args = ap.parse_args()

    try:
        mask = load_mask_1d(args.mask)
    except Exception as e:
        print(f"[error] failed to load mask: {e}", file=sys.stderr)
        sys.exit(1)

    L = mask.size
    if args.verbose:
        trues = int(mask.sum())
        print(f"[info] mask length: {L}, true count: {trues}")

    union, placements, coverages = greedy_tile_1d(mask, n=args.n, verbose=args.verbose, progress=args.progress)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    np.save(args.out_dir / "union.npy", union)
    with open(args.out_dir / "placements.json", "w", encoding="utf-8") as f:
        json.dump(placements, f, indent=2)

    coverage_pct = 100.0 * (np.count_nonzero(union) / union.size if union.size else 0.0)
    with open(args.out_dir / "coverage_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Mask: {args.mask}\n")
        f.write(f"Length: {L}\n")
        f.write(f"Copies placed: {len(placements)} / {args.n}\n")
        f.write(f"Final coverage: {coverage_pct:.4f}% of positions\n")
        if coverages:
            f.write(f"Coverage by step (%): {[round(100*c,3) for c in coverages]}\n")

    # Optional quick visuals
    save_visual_1d(mask, args.out_dir / "mask.png")
    save_visual_1d(union, args.out_dir / "union.png")

    if args.verbose:
        print(f"[done] final coverage = {coverage_pct:.2f}%")
        print(f"[done] outputs in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
