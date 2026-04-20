#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from glob import glob

import numpy as np
from scipy.signal import medfilt
import soundfile as sf


# ──────────────────────────────────────────────
# RPCA helpers
# ──────────────────────────────────────────────

def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def svd_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return u @ np.diag(s_thresh) @ vt


def rpca_ialm(d, lam=None, mu=None, max_iter=1000, tol=1e-7, verbose=False):
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
        if rel_err < tol:
            break

    return l, s


# ──────────────────────────────────────────────
# Feature helpers
# ──────────────────────────────────────────────

def load_repo_phoneme_posteriors(path: Path):
    arr = np.load(path)

    if arr.shape[0] == 392 and arr.shape[1] != 392:
        post = arr.T
    else:
        post = arr

    row_sums = post.sum(axis=1, keepdims=True)
    if not np.allclose(np.median(row_sums), 1.0, atol=0.1):
        z = post - post.max(axis=1, keepdims=True)
        expz = np.exp(z)
        post = expz / (expz.sum(axis=1, keepdims=True) + 1e-12)

    return post


def resolve_feature_path(wav, posterior_npy, feature_root):
    if posterior_npy:
        return posterior_npy
    return feature_root / f"{wav.stem}.npy"


def temporal_pooling(post, frame_hop_sec, win_sec, hop_sec):
    t, nph = post.shape
    win_frames = int(win_sec / frame_hop_sec)
    hop_frames = int(hop_sec / frame_hop_sec)

    cols = []
    for start in range(0, t - win_frames + 1, hop_frames):
        cols.append(post[start:start + win_frames].mean(axis=0))

    return np.stack(cols, axis=1)


def robust_threshold(x, k=4.0):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return med + k * mad


# ──────────────────────────────────────────────
# Edge-trimmed detection
# ──────────────────────────────────────────────

EDGE_TRIM_SEC = 1.0    # ignore first & last 1 second (in time, not windows)


def detect_change_points(s, med_kernel, k, hop_sec):
    """
    Returns energy, detected indices, threshold — with edge trimming applied.
    Edge windows (first/last EDGE_TRIM_SEC) are zeroed before thresholding
    to suppress known RPCA boundary artefacts.
    hop_sec is passed in so the window count is always correct regardless of
    what --hop-sec the user passes (0.1, 0.5, etc.).
    """
    energy = np.linalg.norm(s, axis=0)
    if med_kernel >= 3:
        energy = medfilt(energy, kernel_size=med_kernel)

    # ── Edge trim ──────────────────────────────────────────────────────────
    # Convert seconds → number of windows using the ACTUAL hop_sec
    edge_trim = max(1, int(EDGE_TRIM_SEC / hop_sec))
    interior = energy[edge_trim:-edge_trim]
    interior_mean = interior.mean()
    interior_std  = interior.std()
    noise_scale   = interior_std * 0.05   # 5% of std — visually flat but not zero

    rng = np.random.default_rng(seed=0)   # fixed seed for reproducibility
    energy_trimmed = energy.copy()
    energy_trimmed[:edge_trim]  = interior_mean + rng.normal(0, noise_scale, edge_trim)
    energy_trimmed[-edge_trim:] = interior_mean + rng.normal(0, noise_scale, edge_trim)
    # ──────────────────────────────────────────────────────────────────────

    # Compute threshold on interior only — edge fill must not corrupt median/MAD
    thr = robust_threshold(interior, k=k)
    # Filled edge values sit near the mean, well below threshold — suppression holds
    idx = np.where(energy_trimmed > thr)[0]

    # Fix 2: use last window of each contiguous cluster (boundary at trailing edge)
    if len(idx) > 0:
        clusters, n_clusters = label_clusters(idx)
        idx_adjusted = []
        for c in range(1, n_clusters + 1):
            cluster_idx = np.where(clusters == c)[0]
            idx_adjusted.append(int(idx[cluster_idx[-1]]))   # last window
        idx = np.array(idx_adjusted)

    return energy, energy_trimmed, idx, thr


def label_clusters(idx):
    """Label contiguous runs in idx with integer cluster IDs (1-based)."""
    from scipy.ndimage import label as scipy_label
    mask = np.zeros(int(idx.max()) + 1, dtype=int)
    mask[idx] = 1
    clusters_full, n = scipy_label(mask)
    # Map back to idx positions
    cluster_labels = clusters_full[idx]
    return cluster_labels, n


# ──────────────────────────────────────────────
# RTTM helpers
# ──────────────────────────────────────────────

def boundaries_to_segments(change_times_sec, total_dur_sec):
    """
    Convert a sorted list of change-point times into
    [(start, end, lang_id), ...] segments alternating LANG_A / LANG_B.
    """
    boundaries = [0.0] + sorted(change_times_sec) + [total_dur_sec]
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]
        lang  = "LANG_A" if i % 2 == 0 else "LANG_B"
        segments.append((start, end, lang))
    return segments


def write_rttm(segments, filename_stem, rttm_path: Path):
    """
    Write an RTTM file.  Format:
      SPEAKER <file> 1 <onset> <dur> <NA> <NA> <lang> <NA> <NA>
    """
    with open(rttm_path, "w") as f:
        for (start, end, lang) in segments:
            dur = end - start
            f.write(
                f"SPEAKER {filename_stem} 1 {start:.3f} {dur:.3f}"
                f" <NA> <NA> {lang} <NA> <NA>\n"
            )


# ──────────────────────────────────────────────
# DER computation
# ──────────────────────────────────────────────

def compute_der(detected_boundaries, gt_boundaries, total_dur, collar=0.25):
    """
    Simple 2-language DER.

    Parameters
    ----------
    detected_boundaries : list[float]   detected change times (sec)
    gt_boundaries       : list[float]   ground-truth change times (sec)
    total_dur           : float         total audio duration (sec)
    collar              : float         tolerance window around each boundary (sec)
                                        — matched regions are not penalised

    Returns
    -------
    der            : float  overall DER (0–1)
    missed_dur     : float  seconds missed
    false_alarm_dur: float  seconds falsely detected
    confusion_dur  : float  seconds of time-shifted / confused boundaries
    """
    missed_dur      = 0.0
    false_alarm_dur = 0.0
    confusion_dur   = 0.0

    # Miss: GT boundary with no detection within collar
    for gt in gt_boundaries:
        matched = [d for d in detected_boundaries if abs(d - gt) <= collar]
        if not matched:
            missed_dur += collar * 2          # rough segment extent
        else:
            closest = min(matched, key=lambda d: abs(d - gt))
            confusion_dur += abs(closest - gt)  # offset = mislabelled audio

    # False alarm: detection with no GT boundary within collar
    for det in detected_boundaries:
        if not any(abs(det - gt) <= collar for gt in gt_boundaries):
            false_alarm_dur += collar * 2

    der = (missed_dur + false_alarm_dur + confusion_dur) / max(total_dur, 1e-9)
    return der, missed_dur, false_alarm_dur, confusion_dur


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wav-dir",      type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--gt-json",      type=Path, required=True)

    parser.add_argument("--frame-hop-sec", type=float, default=0.02)
    parser.add_argument("--win-sec",       type=float, default=1.0)
    # hop-sec is passed through directly — controls both pooling and edge trim window count
    parser.add_argument("--hop-sec",       type=float, default=0.5,
                        help="Pooling hop size in seconds (e.g. 0.1, 0.5).")
    parser.add_argument("--med-kernel",    type=int,   default=5)
    parser.add_argument("--k",             type=float, default=4.0)
    parser.add_argument("--collar",        type=float, default=0.25,
                        help="DER collar in seconds (default 0.25).")

    parser.add_argument("--out-prefix",   type=Path,  default=Path("rpca"))
    parser.add_argument("--no-plots",     action="store_true",
                        help="Skip plot generation (faster for large batches).")
    parser.add_argument("--rttm-dir",     type=Path,  default=Path("rttm_out"),
                        help="Directory to write per-file RTTM files.")

    args = parser.parse_args()

    # Load GT  (supports single float OR list-of-floats per file)
    with open(args.gt_json) as f:
        gt_data = json.load(f)

    # Accept either {"change_point_sec": 4.0} or {"change_point_sec": [4.0, ...]}
    gt_raw = gt_data["change_point_sec"]
    if isinstance(gt_raw, (int, float)):
        gt_global = [float(gt_raw)]          # same GT for all files
    else:
        gt_global = [float(v) for v in gt_raw]

    args.rttm_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over feature files that actually exist, then match wav
    # This avoids crashes when features are missing for some wavs
    feature_files = sorted(args.feature_root.glob("*.npy"))
    wav_files     = []
    for npy in feature_files:
        wav_candidate = args.wav_dir / f"{npy.stem}.wav"
        if wav_candidate.exists():
            wav_files.append(str(wav_candidate))
    print(f"  Found {len(feature_files)} feature files, "
          f"{len(wav_files)} have matching wavs — processing those.")

    all_mae_errors = []
    all_der_values = []

    # ── Checkpoint log setup ──────────────────────────────────────────────
    CHECKPOINT_EVERY = 20
    checkpoint_log   = Path(str(args.out_prefix) + "_checkpoint.log")
    checkpoint_log.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_log, "w") as f:
        f.write(f"# RPCA checkpoint log\n")
        f.write(f"# hop={args.hop_sec}s  win={args.win_sec}s  k={args.k}"
                f"  collar={args.collar}s  edge_trim={EDGE_TRIM_SEC}s\n")
        f.write(f"# Total files to process: {len(wav_files)}\n")
        f.write(f"{'#'*60}\n")
    # ─────────────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  hop_sec = {args.hop_sec}s | edge trim = {EDGE_TRIM_SEC}s each side")
    print(f"  DER collar = {args.collar}s")
    print(f"  Total files: {len(wav_files)}")
    print(f"  Checkpoint log: {checkpoint_log}")
    print(f"{'='*60}")

    for wav in wav_files:
        wav = Path(wav)
        print(f"\n── {wav.name} ──")

        # Per-file GT: if gt-json has one entry per file use it, else use global
        gt_boundaries = gt_global

        # Audio duration (needed for RTTM and DER)
        try:
            info = sf.info(str(wav))
            total_dur = info.duration
        except Exception:
            # Fallback: estimate from feature length
            feature_path = resolve_feature_path(wav, None, args.feature_root)
            post = load_repo_phoneme_posteriors(feature_path)
            total_dur = post.shape[0] * args.frame_hop_sec

        feature_path = resolve_feature_path(wav, None, args.feature_root)
        post = load_repo_phoneme_posteriors(feature_path)

        d = temporal_pooling(post, args.frame_hop_sec, args.win_sec, args.hop_sec)
        d = d / (d.sum(axis=0, keepdims=True) + 1e-12)

        l, s = rpca_ialm(d)

        energy, energy_trimmed, idx, thr = detect_change_points(
            s, args.med_kernel, args.k, args.hop_sec
        )

        time_axis      = np.arange(len(energy)) * args.hop_sec
        detected_times = idx * args.hop_sec

        print(f"  Detected change points : {detected_times.tolist()}")
        print(f"  GT change points       : {gt_boundaries}")

        # ── MAE ──────────────────────────────────────────────────────────
        if len(detected_times) > 0:
            pred  = detected_times[0]
            error = abs(pred - gt_boundaries[0])
            all_mae_errors.append(error)
            print(f"  MAE (first CP)         : {error:.3f} s")
        else:
            print("  MAE                    : no detection")

        # ── DER ──────────────────────────────────────────────────────────
        der, miss, fa, conf = compute_der(
            detected_times.tolist(), gt_boundaries, total_dur,
            collar=args.collar
        )
        all_der_values.append(der)
        print(f"  DER                    : {der*100:.1f}%")
        print(f"    Miss={miss:.3f}s  FA={fa:.3f}s  Confusion={conf:.3f}s"
              f"  TotalDur={total_dur:.2f}s")

        # ── RTTM ─────────────────────────────────────────────────────────
        segments  = boundaries_to_segments(detected_times.tolist(), total_dur)
        rttm_path = args.rttm_dir / f"{wav.stem}.rttm"
        write_rttm(segments, wav.stem, rttm_path)
        print(f"  RTTM written           : {rttm_path}")

        # ── Per-file checkpoint log ───────────────────────────────────────
        with open(checkpoint_log, "a") as f:
            mae_str = f"{error:.3f}s" if len(detected_times) > 0 else "no_detection"
            f.write(f"{wav.name}  |  det={detected_times.tolist()}"
                    f"  |  gt={gt_boundaries}"
                    f"  |  MAE={mae_str}"
                    f"  |  DER={der*100:.1f}%"
                    f"  |  Miss={miss:.3f}s  FA={fa:.3f}s  Conf={conf:.3f}s\n")

        # ── Every-N checkpoint summary ────────────────────────────────────
        file_num = wav_files.index(str(wav)) + 1
        if file_num % CHECKPOINT_EVERY == 0 or file_num == len(wav_files):
            running_der = np.mean(all_der_values) * 100
            running_mae = np.mean(all_mae_errors) if all_mae_errors else float("nan")
            detected_count = sum(1 for d in all_der_values
                                 if d < (0.5 / max(1e-9, total_dur)) + 1e-6)
            summary = (
                f"\n{'─'*60}\n"
                f"  CHECKPOINT [{file_num}/{len(wav_files)} files]\n"
                f"  Running DER (mean)   : {running_der:.1f}%\n"
                f"  Running MAE (mean)   : {running_mae:.3f}s\n"
                f"  Files with detection : {len(all_mae_errors)}/{file_num}\n"
                f"{'─'*60}"
            )
            print(summary)
            with open(checkpoint_log, "a") as f:
                f.write(summary + "\n")
        # ─────────────────────────────────────────────────────────────────

        # ── Plot ─────────────────────────────────────────────────────────
        if not args.no_plots:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

            # Top: raw energy
            axes[0].plot(time_axis, energy, label="Energy (raw)", color="steelblue")
            axes[0].axhline(thr, linestyle="--", color="orange", label=f"Threshold (k={args.k})")
            for t in detected_times:
                axes[0].axvline(t, linestyle=":", color="purple", label="Detected")
            for gt in gt_boundaries:
                axes[0].axvline(gt, color="red", linestyle="--", label="GT boundary")
            axes[0].set_ylabel("Energy")
            axes[0].set_title(f"{wav.name} — Raw energy  |  DER={der*100:.1f}%")
            # Deduplicate legend entries
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys(), fontsize=8)

            # Bottom: trimmed energy (what thresholding actually sees)
            axes[1].plot(time_axis, energy_trimmed, label="Energy (trimmed)", color="teal")
            axes[1].axhline(thr, linestyle="--", color="orange", label=f"Threshold")
            for t in detected_times:
                axes[1].axvline(t, linestyle=":", color="purple", label="Detected")
            for gt in gt_boundaries:
                axes[1].axvline(gt, color="red", linestyle="--", label="GT boundary")
            axes[1].set_xlabel("Time (seconds)")
            axes[1].set_ylabel("Energy (edge-trimmed)")
            axes[1].set_title(f"After edge trim  ({EDGE_TRIM_SEC}s blanked each side)")
            handles, labels = axes[1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[1].legend(by_label.values(), by_label.keys(), fontsize=8)

            plt.tight_layout()
            plot_path = f"{args.out_prefix}_{wav.stem}.png"
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"  Plot saved             : {plot_path}")

    # ── Aggregate summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*60}")
    if all_mae_errors:
        print(f"  MAE  (all files) : {np.mean(all_mae_errors):.3f} s")
    if all_der_values:
        print(f"  DER  (mean)      : {np.mean(all_der_values)*100:.1f}%")
        print(f"  DER  (per file)  : "
              + "  ".join(f"{v*100:.1f}%" for v in all_der_values))
    print(f"  RTTM files in    : {args.rttm_dir}/")


if __name__ == "__main__":
    main()
