#!/usr/bin/env python3
"""
RPCA-based language change-point detection using wav2vec2-phone posterior features.

This script is aligned with this repository's phoneme feature format:
- Features are expected as .npy with 392 phoneme classes (from wav2vec2phone).
- In this repo, loaders commonly use np.load(path).T to get [T, 392].

Usage examples:
  # 1) Directly provide phoneme posterior feature matrix
  python rpca_language_change_wav2vec_phoneme.py --posterior-npy sample.npy --out-prefix runs/demo

  # 2) Provide an audio path and a feature root; script resolves <stem>.npy
  python rpca_language_change_wav2vec_phoneme.py --wav sample.wav --feature-root dataset/feature/seen_set_indicvoice

  # 3) Provide both wav and explicit feature file for that wav
  python rpca_language_change_wav2vec_phoneme.py --wav sample.wav --posterior-npy features/sample.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def svd_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return u @ np.diag(s_thresh) @ vt


def rpca_ialm(
    d: np.ndarray,
    lam: float | None = None,
    mu: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust PCA via inexact augmented Lagrangian method."""
    m, n = d.shape
    norm_d = np.linalg.norm(d, ord="fro") + 1e-12

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if mu is None:
        mu = (m * n) / (4.0 * (np.sum(np.abs(d)) + 1e-12))

    l = np.zeros_like(d)
    s = np.zeros_like(d)
    y = np.zeros_like(d)

    for i in range(max_iter):
        l = svd_threshold(d - s + (1.0 / mu) * y, 1.0 / mu)
        s = soft_threshold(d - l + (1.0 / mu) * y, lam / mu)

        z = d - l - s
        y = y + mu * z

        rel_err = np.linalg.norm(z, ord="fro") / norm_d
        if verbose and (i % 50 == 0 or rel_err < tol):
            print(f"[RPCA] iter={i:4d} rel_err={rel_err:.3e}")
        if rel_err < tol:
            break

    return l, s


def load_repo_phoneme_posteriors(posterior_npy: Path) -> np.ndarray:
    """
    Load wav2vec2-phone posterior matrix from this repo's expected feature format.

    Returns:
        post_tn: shape [T, Nph]
    """
    arr = np.load(posterior_npy)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D posterior matrix, got shape={arr.shape}")

    # In this repo: feature is loaded with np.load(path).T -> [T, 392].
    # So stored shape is often [392, T].
    if arr.shape[0] == 392 and arr.shape[1] != 392:
        post_tn = arr.T
    elif arr.shape[1] == 392:
        post_tn = arr
    else:
        # Fall back to whichever axis is smaller as phoneme dimension.
        post_tn = arr.T if arr.shape[0] < arr.shape[1] else arr

    # Convert logits or unnormalized values to pseudo-posteriors if needed.
    # Softmax row-wise if rows do not approximately sum to 1.
    row_sums = post_tn.sum(axis=1, keepdims=True)
    if not np.allclose(np.median(row_sums), 1.0, atol=0.1):
        z = post_tn - post_tn.max(axis=1, keepdims=True)
        expz = np.exp(z)
        post_tn = expz / (expz.sum(axis=1, keepdims=True) + 1e-12)

    return post_tn.astype(np.float64, copy=False)


def resolve_feature_path(wav: Path | None, posterior_npy: Path | None, feature_root: Path | None) -> Path:
    """
    Resolve which .npy phoneme posterior file to load.

    Priority:
    1) --posterior-npy, if given.
    2) --wav + --feature-root -> <feature_root>/<wav_stem>.npy
    """
    if posterior_npy is not None:
        if not posterior_npy.exists():
            raise FileNotFoundError(f"--posterior-npy not found: {posterior_npy}")
        return posterior_npy

    if wav is None:
        raise ValueError("Provide either --posterior-npy OR --wav (with --feature-root).")

    if feature_root is None:
        raise ValueError("When using --wav without --posterior-npy, provide --feature-root.")

    candidate = feature_root / f"{wav.stem}.npy"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find phoneme feature for wav '{wav}'. Expected: {candidate}. "
            "Either pass --posterior-npy directly or place the matching .npy in --feature-root."
        )
    return candidate


def temporal_pooling(post_tn: np.ndarray, frame_hop_sec: float, win_sec: float, hop_sec: float) -> np.ndarray:
    """Pool frame-level posteriors [T, Nph] -> matrix D [Nph, W]."""
    t, nph = post_tn.shape
    win_frames = max(1, int(round(win_sec / frame_hop_sec)))
    hop_frames = max(1, int(round(hop_sec / frame_hop_sec)))

    cols: list[np.ndarray] = []
    for start in range(0, t - win_frames + 1, hop_frames):
        cols.append(post_tn[start : start + win_frames].mean(axis=0))

    if len(cols) < 3:
        raise RuntimeError("Too few pooled windows. Use larger input or smaller win/hop.")

    d = np.stack(cols, axis=1)  # [Nph, W]
    return d


def robust_threshold(x: np.ndarray, k: float = 4.0) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return med + k * mad


def detect_change_points(s: np.ndarray, med_kernel: int, k: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return sparse energy, detected indices, threshold."""
    energy = np.linalg.norm(s, axis=0)
    if med_kernel >= 3 and med_kernel % 2 == 1:
        energy = medfilt(energy, kernel_size=med_kernel)

    thr = robust_threshold(energy, k=k)
    idx = np.where(energy > thr)[0]
    return energy, idx, thr


def main() -> None:
    parser = argparse.ArgumentParser(description="RPCA language change detection on wav2vec2-phone posteriors")
    parser.add_argument("--posterior-npy", type=Path, default=None, help="Path to phoneme posterior .npy")
    parser.add_argument("--wav", type=Path, default=None, help="Audio file to analyze (used to resolve matching .npy)")
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Directory containing phoneme features named <wav_stem>.npy (used with --wav)",
    )
    parser.add_argument("--frame-hop-sec", type=float, default=0.02, help="Frame hop used in posterior extraction")
    parser.add_argument("--win-sec", type=float, default=1.0, help="Pooling window size in seconds")
    parser.add_argument("--hop-sec", type=float, default=0.5, help="Pooling hop size in seconds")
    parser.add_argument("--med-kernel", type=int, default=5, help="Median filter kernel size (odd)")
    parser.add_argument("--k", type=float, default=4.0, help="Threshold factor in median + k*MAD")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("rpca_lang_change"), help="Output prefix")
    args = parser.parse_args()

    feature_path = resolve_feature_path(args.wav, args.posterior_npy, args.feature_root)
    if args.wav is not None:
        print(f"Input audio: {args.wav}")
    print(f"Using phoneme feature: {feature_path}")

    post_tn = load_repo_phoneme_posteriors(feature_path)
    print(f"Loaded posteriors: shape={post_tn.shape} (T, Nph)")

    d = temporal_pooling(post_tn, args.frame_hop_sec, args.win_sec, args.hop_sec)
    d = d / (d.sum(axis=0, keepdims=True) + 1e-12)
    print(f"Constructed D: shape={d.shape} (Nph, W)")

    l, s = rpca_ialm(d, max_iter=args.max_iter, tol=args.tol, verbose=args.verbose)
    energy, idx, thr = detect_change_points(s, med_kernel=args.med_kernel, k=args.k)

    change_times = idx * args.hop_sec
    print(f"Detected windows: {idx.tolist()}")
    print(f"Detected change times (sec): {change_times.tolist()}")
    print(f"Threshold: {thr:.6f}")
    print(f"Approx rank(L): {np.linalg.matrix_rank(l, tol=1e-3)}")
    print(f"S sparsity ratio (|S|<1e-6): {float(np.mean(np.abs(s) < 1e-6)):.4f}")

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(f"{args.out_prefix}_L.npy", l)
    np.save(f"{args.out_prefix}_S.npy", s)
    np.save(f"{args.out_prefix}_energy.npy", energy)
    np.save(f"{args.out_prefix}_change_idx.npy", idx)

    plt.figure(figsize=(10, 4))
    plt.plot(energy, label="Sparse energy ||S(:,w)||")
    plt.axhline(thr, linestyle="--", label=f"threshold (median + {args.k}*MAD)")
    for i in idx:
        plt.axvline(i, linestyle=":", linewidth=1)
    plt.xlabel("Window index")
    plt.ylabel("Energy")
    plt.title("RPCA-based language change detection")
    plt.legend()
    plt.tight_layout()
    fig_path = f"{args.out_prefix}_energy.png"
    plt.savefig(fig_path, dpi=160)
    print(f"Saved plot: {fig_path}")


if __name__ == "__main__":
    main()
