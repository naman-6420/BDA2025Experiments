#!/usr/bin/env python3
"""
Extract frame-level phoneme posterior features from audio using Hugging Face
Wav2Vec2Phoneme CTC models.

This script produces repository-compatible .npy feature files for downstream
ConfPhoneme/XvecPhoneme/RPCA pipelines.

Output format:
- Saved matrix shape is [Nph, T] (default), so repo loaders using np.load(...).T
  get [T, Nph].

Example:
  python extract_wav2vec2phoneme_matrix.py \ 
    --wav /path/to/sample.wav \
    --out-npy /path/to/features/sample.npy \
    --model-id facebook/wav2vec2-lv-60-espeak-cv-ft
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForCTC


def load_audio_16k(path: Path, target_sr: int = 16000) -> np.ndarray:
    wav, _ = librosa.load(path, sr=target_sr)
    if wav.ndim != 1:
        wav = np.mean(wav, axis=-1)
    return wav.astype(np.float32, copy=False)


def batched_logits(
    model,
    processor,
    wav: np.ndarray,
    sr: int,
    chunk_sec: float,
    stride_sec: float,
    device: torch.device,
) -> np.ndarray:
    """Run long-audio inference with overlap and stitch center logits."""
    chunk = int(chunk_sec * sr)
    stride = int(stride_sec * sr)

    if len(wav) <= chunk:
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits[0].cpu().numpy()
        return logits

    parts = []
    pos = 0
    while pos < len(wav):
        x = wav[pos : pos + chunk]
        if len(x) == 0:
            break

        inputs = processor(x, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logit = model(inputs.input_values.to(device)).logits[0].cpu().numpy()  # [T', V]

        # Remove edge regions (except first/last chunk) to reduce boundary artifacts.
        n = logit.shape[0]
        cut = max(0, int(round((stride / max(1, len(x))) * n)))
        left = 0 if pos == 0 else cut
        right = n if (pos + chunk) >= len(wav) else max(left + 1, n - cut)
        parts.append(logit[left:right])

        if pos + chunk >= len(wav):
            break
        pos += (chunk - stride)

    return np.concatenate(parts, axis=0)


from glob import glob

def main() -> None:
    p = argparse.ArgumentParser()

    # existing args
    p.add_argument("--wav", type=Path, default=None)

    # ✅ NEW
    p.add_argument("--wav-dir", type=Path, default=None)

    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--model-id", type=str, default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--chunk-sec", type=float, default=20.0)
    p.add_argument("--stride-sec", type=float, default=2.0)
    p.add_argument("--force-dim", type=int, default=392)
    p.add_argument("--save-layout", type=str, default="nph_t")
    p.add_argument("--use-safetensors", action="store_true", default=True)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_id}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForCTC.from_pretrained(
        args.model_id,
        use_safetensors=args.use_safetensors,
    ).to(device).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ collect wavs
    if args.wav_dir:
        wav_files = sorted(glob(str(args.wav_dir / "*.wav")))
    else:
        wav_files = [str(args.wav)]

    for wav_path_str in wav_files:
        wav_path = Path(wav_path_str)

        print(f"\nProcessing: {wav_path.name}")

        wav = load_audio_16k(wav_path, args.sr)

        logits = batched_logits(
            model=model,
            processor=feature_extractor,
            wav=wav,
            sr=args.sr,
            chunk_sec=args.chunk_sec,
            stride_sec=args.stride_sec,
            device=device,
        )

        post = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

        if post.shape[1] != args.force_dim:
            if post.shape[1] > args.force_dim:
                post = post[:, :args.force_dim]
            else:
                pad = np.zeros((post.shape[0], args.force_dim - post.shape[1]))
                post = np.concatenate([post, pad], axis=1)

        out = post.T if args.save_layout == "nph_t" else post

        out_path = args.out_dir / f"{wav_path.stem}.npy"
        np.save(out_path, out.astype(np.float32))

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
