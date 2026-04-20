#!/usr/bin/env python3
"""
merge_audio_multi.py — Multi-language audio merger for diarization datasets.

Merging logic: crossfade join (librosa + numpy + soundfile).
  - Each language segment is collected by randomly picking files until
    the requested duration is reached.
  - Adjacent segments are joined with a linear crossfade (fade-out on the
    tail of segment i, fade-in on the head of segment i+1, overlap summed).
  - No other preprocessing (no resampling, no loudness normalisation).

Usage:
    python merge_audio_multi.py hin 4 eng 4
    python merge_audio_multi.py hin 4 eng 4 tam 4 --num-combinations 10
    python merge_audio_multi.py hin 4 eng 4 --out-dir output --manifest
    python merge_audio_multi.py --list-languages
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

# =============================================================================
# Silero VAD — loaded lazily only when --use-sad is passed
# =============================================================================
_vad_model  = None
_vad_utils  = None

def load_silero_vad():
    """Load Silero VAD model once and cache it."""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model, _vad_utils
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir = "snakers4/silero-vad",
            model       = "silero_vad",
            force_reload = False,
            trust_repo  = True,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero VAD loaded.")
        return model, utils
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Silero VAD: {exc}\n"
            "  Install with: pip install torch torchaudio\n"
            "  Silero VAD is fetched from torch.hub on first use."
        ) from exc


def extract_speech_only(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Run Silero VAD on waveform y and return only the speech regions
    concatenated together. Silences are removed entirely.

    Returns empty array if no speech detected.
    """
    import torch

    model, utils = load_silero_vad()
    get_speech_timestamps = utils[0]

    # Silero requires float32 torch tensor
    wav_tensor = torch.from_numpy(y).float()

    # get_speech_timestamps returns list of {start, end} sample dicts
    timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate     = sr,
        threshold         = 0.5,    # speech probability threshold
        min_silence_dur_ms= 300,    # merge gaps shorter than this
        min_speech_dur_ms = 100,    # drop speech bursts shorter than this
    )

    if not timestamps:
        return np.array([], dtype=np.float32)

    speech_chunks = [y[ts["start"]: ts["end"]] for ts in timestamps]
    return np.concatenate(speech_chunks).astype(np.float32)

# =============================================================================
# Language -> Path registry
# =============================================================================
LANGUAGE_PATHS: Dict[str, str] = {
    "asm": "data/asm",
    "ben": "data/ben",
    "eng": "data/eng",
    "guj": "data/guj",
    "hin": "data/hin",
    "kan": "data/kan",
    "mal": "data/mal",
    "mar": "data/mar",
    "odi": "data/odi",
    "pun": "data/pun",
    "tam": "data/tam",
    "tel": "data/tel",
}

LANGUAGE_NAMES: Dict[str, str] = {
    "asm": "Assamese",
    "ben": "Bengali",
    "eng": "English",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "odi": "Odia",
    "pun": "Punjabi",
    "tam": "Tamil",
    "tel": "Telugu",
}

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BOLD   = "\033[1m"
CYAN   = "\033[0;36m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
RESET  = "\033[0m"


# =============================================================================
# Terminal helpers
# =============================================================================

def print_language_table() -> None:
    print(f"\n{BOLD}{'─'*52}{RESET}")
    print(f"{BOLD}  Available languages{RESET}")
    print(f"{BOLD}{'─'*52}{RESET}")
    print(f"  {'Code':<8} {'Language':<14} {'Path':<26} {'Status'}")
    print(f"  {'────':<8} {'────────':<14} {'────':<26} {'──────'}")
    for code, path in LANGUAGE_PATHS.items():
        name   = LANGUAGE_NAMES.get(code, code)
        exists = Path(path).is_dir()
        status = f"{GREEN}found{RESET}" if exists else f"{RED}missing{RESET}"
        print(f"  {CYAN}{code:<8}{RESET} {name:<14} {path:<26} {status}")
    print(f"{BOLD}{'─'*52}{RESET}\n")


def prompt_language_durations() -> List[Tuple[str, float]]:
    print_language_table()
    print(f"{BOLD}Enter language-duration pairs.{RESET}")
    print(f"  Format : <code> <seconds>   e.g.  {CYAN}hin 4{RESET}")
    print(f"  Type   : {CYAN}done{RESET} when finished (minimum 2 languages)\n")

    pairs: List[Tuple[str, float]] = []
    valid_codes = set(LANGUAGE_PATHS.keys())

    while True:
        raw = input(f"  [{len(pairs)+1}] lang duration (or 'done'): ").strip()

        if raw.lower() in ("done", "q", "exit"):
            if len(pairs) < 2:
                print(f"  {YELLOW}Need at least 2 languages. Keep going.{RESET}")
                continue
            break

        parts = raw.split()
        if len(parts) != 2:
            print(f"  {RED}Expected exactly two tokens: <code> <seconds>{RESET}")
            continue

        code, dur_str = parts
        code = code.lower()

        if code not in valid_codes:
            print(f"  {RED}Unknown code '{code}'. Valid: {', '.join(sorted(valid_codes))}{RESET}")
            continue

        try:
            dur = float(dur_str)
        except ValueError:
            print(f"  {RED}Duration must be a number, got '{dur_str}'{RESET}")
            continue

        if dur <= 0:
            print(f"  {RED}Duration must be > 0{RESET}")
            continue

        pairs.append((code, dur))
        print(f"  {GREEN}Added {LANGUAGE_NAMES.get(code, code)} ({code}) - {dur}s{RESET}")

    return pairs


# =============================================================================
# File scanning (cached)
# =============================================================================
_file_scan_cache: Dict[Path, List[Path]] = {}


def find_audio_files(folder: Path) -> List[Path]:
    if folder in _file_scan_cache:
        logger.info("'%s': using cached file list (%d files)", folder, len(_file_scan_cache[folder]))
        return _file_scan_cache[folder]
    logger.info("Scanning '%s' ...", folder)
    files = [
        p for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    _file_scan_cache[folder] = files
    return files


# =============================================================================
# Audio collection  (librosa.load, no preprocessing)
# =============================================================================

def collect_segment(
    files: List[Path],
    target_sec: float,
    lang: str,
    file_index: int,
    verbose: bool = False,
    use_sad: bool = False,
) -> Tuple[np.ndarray, int, List[dict]]:
    """
    Take files starting from file_index (1-to-1 mapping per combination).
    Always reads from the beginning of each file.
    No shuffling, no random offsets, no preprocessing.

    If use_sad=True, runs Silero VAD on each file and strips silence before
    collecting, so only speech frames contribute to the target duration.

    Returns (waveform float32, sample_rate, manifest_entries).
    Raises RuntimeError if not enough audio is available.
    """
    collected: List[np.ndarray] = []
    manifest:  List[dict]       = []
    collected_samples = 0
    sr_out: int | None = None
    skipped = 0

    # Start from file_index so each combination uses a different file
    ordered = files[file_index:] + files[:file_index]

    for path in ordered:
        if sr_out is not None:
            if collected_samples >= int(target_sec * sr_out):
                break

        try:
            y, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if y.ndim == 2:          # stereo -> mono by averaging channels
                y = y.mean(axis=1)
        except Exception as exc:
            skipped += 1
            if verbose:
                logger.warning("  Skipping %s: %s", path.name, exc)
            continue

        if len(y) == 0:
            skipped += 1
            if verbose:
                logger.warning("  Skipping empty file: %s", path.name)
            continue

        # Apply SAD — strip silence, keep only speech regions
        if use_sad:
            y_speech = extract_speech_only(y, sr)
            if len(y_speech) == 0:
                skipped += 1
                if verbose:
                    logger.warning("  No speech detected — skipping: %s", path.name)
                continue
            if verbose:
                logger.info(
                    "  SAD: %.2fs -> %.2fs speech (removed %.2fs silence) %s",
                    len(y) / sr, len(y_speech) / sr,
                    (len(y) - len(y_speech)) / sr, path.name,
                )
            y = y_speech

        # Lock sample rate from first good file; skip mismatches
        if sr_out is None:
            sr_out = sr
        elif sr != sr_out:
            skipped += 1
            logger.warning(
                "  SR mismatch — skipping %s (%dHz != %dHz)",
                path.name, sr, sr_out,
            )
            continue

        target_samples = int(target_sec * sr_out)
        needed = target_samples - collected_samples
        clip   = y[:needed]

        start_sample = collected_samples
        collected.append(clip)
        collected_samples += len(clip)

        manifest.append({
            "file"        : str(path),
            "language"    : lang,
            "start_sample": start_sample,
            "end_sample"  : start_sample + len(clip),
        })

        logger.info(
            "  [%s] collected %.2fs / %.2fs  <- %s",
            lang, collected_samples / sr_out, target_sec, path.name,
        )

    # Error: no files loaded at all
    if sr_out is None:
        raise RuntimeError(
            f"No usable audio files found for language '{lang}' "
            f"({LANGUAGE_NAMES.get(lang, lang)})."
        )

    # Error: not enough audio
    collected_sec = collected_samples / sr_out
    if collected_samples < int(target_sec * sr_out):
        raise RuntimeError(
            f"\n"
            f"  +------ DURATION ERROR ----------------------------------+\n"
            f"  | Language  : {lang} ({LANGUAGE_NAMES.get(lang, lang)})\n"
            f"  | Requested : {target_sec:.2f}s\n"
            f"  | Collected : {collected_sec:.2f}s  ({len(files) - skipped} usable files)\n"
            f"  | Shortfall : {target_sec - collected_sec:.2f}s\n"
            f"  | Fix: reduce duration to <= {collected_sec:.1f}s\n"
            f"  |      or add more audio to '{LANGUAGE_PATHS.get(lang, lang)}'\n"
            f"  +--------------------------------------------------------+"
        )

    return np.concatenate(collected).astype(np.float32), sr_out, manifest


# =============================================================================
# Crossfade merge — exact logic from combine_audio.py
# =============================================================================

def crossfade_join(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    fade_duration: float = 0.5,
) -> np.ndarray:
    """
    Join two mono float32 waveforms with a linear crossfade.

    Exact replication of combine_audio.py:
        fade_samples = int(sr * fade_duration)
        fade_out     = linspace(1, 0, fade_samples)   -> applied to tail of y1
        fade_in      = linspace(0, 1, fade_samples)   -> applied to head of y2
        overlap      = y1_end_faded + y2_start_faded
        combined     = [y1_body | overlap | y2_tail]
    """
    fade_samples = int(sr * fade_duration)

    # Guard: shorten fade if either segment is too short
    fade_samples = min(fade_samples, len(y1), len(y2))

    if fade_samples == 0:
        return np.concatenate([y1, y2])

    # Fade curves  (same as combine_audio.py)
    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    fade_in  = np.linspace(0, 1, fade_samples, dtype=np.float32)

    # Apply fades and sum the overlap region
    y1_end_faded   = y1[-fade_samples:] * fade_out
    y2_start_faded = y2[:fade_samples]  * fade_in
    overlap        = y1_end_faded + y2_start_faded

    return np.concatenate([y1[:-fade_samples], overlap, y2[fade_samples:]])


def merge_all_segments(
    segments: List[np.ndarray],
    sr: int,
    fade_duration: float,
) -> np.ndarray:
    """Apply crossfade_join sequentially across all segments."""
    result = segments[0]
    for seg in segments[1:]:
        result = crossfade_join(result, seg, sr, fade_duration)
    return result


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge N language audio segments with crossfade.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "lang_dur_pairs", nargs="*", metavar="LANG_OR_DUR",
        help="Alternating codes and durations: hin 4 eng 4 tam 3 ..."
             " Omit for interactive mode.",
    )
    p.add_argument("--list-languages",   action="store_true",
                   help="Print available languages and exit.")
    p.add_argument("--base-dir",         default=".",
                   help="Fallback root dir if language not in registry (default: .)")
    p.add_argument("--out-dir",          default="output",
                   help="Output directory (default: ./output)")
    p.add_argument("--fade-sec",         type=float, default=0.5,
                   help="Crossfade duration in seconds (default: 0.5)")
    p.add_argument("--seed",             type=int,   default=42,
                   help="Base random seed (default: 42)")
    p.add_argument("--num-combinations", type=int,   default=1,
                   help="Number of merged files to generate (default: 1)")
    p.add_argument("--manifest",         action="store_true",
                   help="Write JSON manifest alongside each audio file")
    p.add_argument("--verbose",          action="store_true",
                   help="Print per-file loading details")
    p.add_argument("--use-sad",          action="store_true",
                   help="Strip silence using Silero VAD before collecting audio. "
                        "Requires: pip install torch torchaudio")
    return p.parse_args()


def parse_lang_dur_pairs(tokens: List[str]) -> List[Tuple[str, float]]:
    if len(tokens) % 2 != 0:
        print(
            f"{RED}[ERROR] Expected pairs: lang1 dur1 lang2 dur2 ...  "
            f"Got {len(tokens)} token(s){RESET}",
            file=sys.stderr,
        )
        sys.exit(1)

    pairs: List[Tuple[str, float]] = []
    valid = set(LANGUAGE_PATHS.keys())

    for i in range(0, len(tokens), 2):
        code    = tokens[i].lower()
        dur_str = tokens[i + 1]

        if code not in valid:
            print(
                f"{RED}[ERROR] Unknown language code '{code}'. "
                f"Valid: {', '.join(sorted(valid))}{RESET}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            dur = float(dur_str)
        except ValueError:
            print(
                f"{RED}[ERROR] Duration must be a number, got '{dur_str}'{RESET}",
                file=sys.stderr,
            )
            sys.exit(1)

        if dur <= 0:
            print(f"{RED}[ERROR] Duration must be > 0, got {dur}{RESET}", file=sys.stderr)
            sys.exit(1)

        pairs.append((code, dur))

    if len(pairs) < 2:
        print(f"{RED}[ERROR] Need at least 2 language-duration pairs.{RESET}", file=sys.stderr)
        sys.exit(1)

    return pairs


def resolve_lang_dir(code: str, base_dir: Path) -> Path:
    if code in LANGUAGE_PATHS:
        p = Path(LANGUAGE_PATHS[code]).resolve()
        logger.info("'%s' -> %s", code, p)
    else:
        p = (base_dir / code).resolve()
        logger.warning("'%s' not in registry -- fallback: %s", code, p)
    return p


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.list_languages:
        print_language_table()
        sys.exit(0)

    # Resolve pairs
    if args.lang_dur_pairs:
        pairs = parse_lang_dur_pairs(args.lang_dur_pairs)
    else:
        print(f"\n{BOLD}No languages specified -- starting interactive mode.{RESET}")
        pairs = prompt_language_durations()

    # Print plan
    print(f"\n{BOLD}Plan:{RESET}")
    total_s = sum(d for _, d in pairs)
    for code, dur in pairs:
        name = LANGUAGE_NAMES.get(code, code)
        print(f"  {CYAN}{code}{RESET} ({name:<12}) -> {dur}s")
    print(f"  {'Total':<18} -> {total_s}s per combination")
    print(f"  Combinations : {args.num_combinations}")
    print(f"  Crossfade    : {args.fade_sec}s")
    print(f"  SAD (Silero) : {'enabled' if args.use_sad else 'disabled'}")
    print(f"  Output dir   : {args.out_dir}\n")

    # Load VAD model upfront so the cost is paid once, not per combination
    if args.use_sad:
        load_silero_vad()

    # Resolve directories and scan files
    base_dir = Path(args.base_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_files: Dict[str, List[Path]] = {}
    errors: List[str] = []

    for code, _ in pairs:
        if code in lang_files:
            continue
        d = resolve_lang_dir(code, base_dir)
        if not d.is_dir():
            errors.append(
                f"Folder not found for '{code}' ({LANGUAGE_NAMES.get(code, code)}): '{d}'\n"
                f"  Update LANGUAGE_PATHS at the top of this script."
            )
        else:
            files = find_audio_files(d)
            if not files:
                errors.append(
                    f"No audio files found in '{d}' for language '{code}'.\n"
                    f"  Supported extensions: {sorted(SUPPORTED_EXTS)}"
                )
            else:
                lang_files[code] = files
                logger.info("'%s': %d file(s) found.", code, len(files))

    if errors:
        for e in errors:
            print(f"{RED}[ERROR] {e}{RESET}", file=sys.stderr)
        sys.exit(1)

    # Generate combinations
    logger.info("Generating %d combination(s) ...", args.num_combinations)
    success_count = 0

    for i in range(args.num_combinations):
        logger.info("-- Combination %d / %d --", i + 1, args.num_combinations)

        segments:       List[np.ndarray] = []
        all_manifest:   List[dict]       = []
        sr_combination: int | None       = None
        failed        = False
        sample_offset = 0

        for code, dur in pairs:
            try:
                audio, sr, manifest = collect_segment(
                    lang_files[code], dur, code, i, args.verbose, args.use_sad,
                )
            except RuntimeError as exc:
                print(f"{RED}{exc}{RESET}", file=sys.stderr)
                logger.warning("Skipping combination %d.", i + 1)
                failed = True
                break

            # Enforce consistent sample rate across all language segments
            if sr_combination is None:
                sr_combination = sr
            elif sr != sr_combination:
                logger.warning(
                    "Skipping combination %d — SR mismatch between languages: "
                    "'%s' locked at %dHz but previous language locked at %dHz. "
                    "Check that all language folders have the same sample rate.",
                    i + 1, code, sr, sr_combination,
                )
                failed = True
                break

            # Shift manifest offsets to global timeline
            for entry in manifest:
                entry["start_sample"] += sample_offset
                entry["end_sample"]   += sample_offset

            segments.append(audio)
            all_manifest.extend(manifest)
            sample_offset += len(audio)

        if failed:
            continue

        # Crossfade merge
        try:
            merged = merge_all_segments(segments, sr_combination, args.fade_sec)
        except Exception as exc:
            logger.error("Merge failed for combination %d: %s", i + 1, exc)
            continue

        # Build filename
        lang_tag = "_".join(f"{code}{int(dur)}s" for code, dur in pairs)
        name     = f"{lang_tag}__{i:03d}_{int(time.time())}"
        out_path = out_dir / f"{name}.wav"

        # Save with soundfile (same as combine_audio.py)
        try:
            sf.write(str(out_path), merged, sr_combination)
        except Exception as exc:
            logger.error("Save failed for combination %d: %s", i + 1, exc)
            continue

        size_mb      = out_path.stat().st_size / (1024 * 1024)
        duration_sec = len(merged) / sr_combination
        logger.info("Saved %s  (%.2f MB, %.2fs)", out_path.name, size_mb, duration_sec)

        # Manifest
        if args.manifest:
            manifest_doc = {
                "output_file"    : str(out_path),
                "sample_rate"    : sr_combination,
                "total_duration_s": duration_sec,
                "fade_duration_s": args.fade_sec,
                "languages": [
                    {
                        "code"                : code,
                        "name"                : LANGUAGE_NAMES.get(code, code),
                        "requested_duration_s": dur,
                    }
                    for code, dur in pairs
                ],
                "segments": all_manifest,
            }
            manifest_path = out_dir / f"{name}.json"
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest_doc, fh, indent=2, ensure_ascii=False)
            logger.info("Manifest: %s", manifest_path.name)

        success_count += 1

    print(f"\n{'='*52}")
    print(f"  {GREEN}Done.{RESET}  {success_count} / {args.num_combinations} combination(s) saved.")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()