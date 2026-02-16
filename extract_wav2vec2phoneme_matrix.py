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


def main() -> None:
    p = argparse.ArgumentParser(description="Extract Wav2Vec2Phoneme posterior matrix (Nph x T)")
    p.add_argument("--wav", type=Path, required=True, help="Input WAV/FLAC audio path")
    p.add_argument("--out-npy", type=Path, required=True, help="Output .npy feature path")
    p.add_argument(
        "--model-id",
        type=str,
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        help="HF Wav2Vec2Phoneme CTC model id",
    )
    p.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    p.add_argument("--chunk-sec", type=float, default=20.0, help="Chunk length for long audio")
    p.add_argument("--stride-sec", type=float, default=2.0, help="Overlap stride for chunk stitching")
    p.add_argument(
        "--force-dim",
        type=int,
        default=392,
        help="Force output phoneme dim to this size via pad/truncate (repo default=392)",
    )
    p.add_argument(
        "--save-layout",
        type=str,
        default="nph_t",
        choices=["nph_t", "t_nph"],
        help="Output matrix layout: nph_t saves [Nph, T], t_nph saves [T, Nph]",
    )
    p.add_argument(
        "--use-safetensors",
        action="store_true",
        default=True,
        help="Load HF checkpoints via safetensors when available (recommended with older torch).",
    )
    args = p.parse_args()

    if args.stride_sec >= args.chunk_sec:
        raise ValueError("--stride-sec must be smaller than --chunk-sec")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_id}")
    # Use feature extractor (NOT full processor) to avoid tokenizer/phonemizer runtime deps
    # such as espeak/protobuf for pure acoustic forward inference.
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForCTC.from_pretrained(
        args.model_id,
        use_safetensors=args.use_safetensors,
    ).to(device).eval()

    wav = load_audio_16k(args.wav, args.sr)
    print(f"Audio loaded: {args.wav} ({len(wav)/args.sr:.2f}s)")

    logits = batched_logits(
        model=model,
        processor=feature_extractor,
        wav=wav,
        sr=args.sr,
        chunk_sec=args.chunk_sec,
        stride_sec=args.stride_sec,
        device=device,
    )  # [T, V]

    # Convert to frame-level posterior matrix.
    post = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()  # [T, V]

    # Force repo-compatible phoneme dimension if needed.
    nph = post.shape[1]
    if args.force_dim is not None and args.force_dim > 0 and nph != args.force_dim:
        if nph > args.force_dim:
            post = post[:, : args.force_dim]
        else:
            pad = np.zeros((post.shape[0], args.force_dim - nph), dtype=post.dtype)
            post = np.concatenate([post, pad], axis=1)
        print(f"Adjusted dim from {nph} -> {post.shape[1]}")

    out = post.T if args.save_layout == "nph_t" else post
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, out.astype(np.float32))
    print(f"Saved: {args.out_npy} shape={out.shape}")


if __name__ == "__main__":
    main()
