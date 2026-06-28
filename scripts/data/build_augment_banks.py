from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import torch
import torchaudio
from loguru import logger

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus"}


def _read_filelist(p: Path) -> list[Path]:
    if p is None:
        return []
    out: list[Path] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(Path(s))
    return out


def _collect_audio_files(dirs: list[Path], files: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for fp in files:
        fp = fp.expanduser()
        if fp.suffix.lower() in AUDIO_EXTS and fp.exists():
            rp = fp.resolve()
            if rp not in seen:
                out.append(rp)
                seen.add(rp)
    for d in dirs:
        d = d.expanduser()
        if not d.exists():
            continue
        if d.is_file():
            if d.suffix.lower() in AUDIO_EXTS:
                rp = d.resolve()
                if rp not in seen:
                    out.append(rp)
                    seen.add(rp)
            continue
        for fp in d.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in AUDIO_EXTS:
                rp = fp.resolve()
                if rp not in seen:
                    out.append(rp)
                    seen.add(rp)
    return out


def _load_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav.dim() == 2 and wav.size(0) == 1:
        wav = wav.squeeze(0)
    wav = wav.contiguous()
    return wav, int(sr)


def _resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr).contiguous()


def _rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt().item())


def _split_noise_segments(
    noise: torch.Tensor,
    seg_len: int,
    max_segments: int,
    gen: torch.Generator,
) -> list[torch.Tensor]:
    n = int(noise.numel())
    if seg_len <= 0 or n <= seg_len:
        return [noise.contiguous()]
    max_start = n - seg_len
    k = min(max_segments, max(1, n // seg_len))
    out: list[torch.Tensor] = []
    for _ in range(k):
        start = int(torch.randint(0, max_start + 1, (), generator=gen).item())
        out.append(noise[start : start + seg_len].contiguous())
    return out


def _process_noise_file(
    path: Path,
    target_sr: int,
    min_sec: float,
    max_sec: float | None,
    seg_sec: float,
    max_segments_per_file: int,
    min_rms: float,
    gen: torch.Generator,
) -> tuple[list[torch.Tensor], float, bool]:
    try:
        wav, sr = _load_mono(path)
        wav = wav.to(dtype=torch.float32)
        wav = _resample(wav, sr, target_sr)
        if wav.numel() == 0:
            return [], 0.0, False
        wav = wav - wav.mean()
        n = wav.numel()
        dur = n / float(target_sr)
        if dur < min_sec:
            return [], 0.0, False
        if max_sec is not None and dur > max_sec:
            max_len = int(max_sec * target_sr)
            if max_len > 0 and n > max_len:
                start = int(torch.randint(0, max(1, n - max_len + 1), (), generator=gen).item())
                wav = wav[start : start + max_len].contiguous()
                n = wav.numel()
                dur = n / float(target_sr)
        if _rms(wav) < min_rms:
            return [], 0.0, False
        seg_len = int(seg_sec * target_sr)
        segs = _split_noise_segments(wav, seg_len, max_segments_per_file, gen)
        kept = []
        kept_dur = 0.0
        for s in segs:
            if s.numel() == 0:
                continue
            if _rms(s) < min_rms:
                continue
            kept.append(s.cpu())
            kept_dur += s.numel() / float(target_sr)
        return kept, kept_dur, True
    except Exception:
        return [], 0.0, False


def _process_rir_file(
    path: Path,
    target_sr: int,
    min_sec: float,
    max_sec: float,
    min_rms: float,
) -> tuple[torch.Tensor | None, float, bool]:
    try:
        wav, sr = _load_mono(path)
        wav = wav.to(dtype=torch.float32)
        wav = _resample(wav, sr, target_sr)
        if wav.numel() == 0:
            return None, 0.0, False
        wav = wav - wav.mean()
        n = wav.numel()
        dur = n / float(target_sr)
        if dur < min_sec:
            return None, 0.0, False
        if _rms(wav) < min_rms:
            return None, 0.0, False
        peak = int(torch.argmax(wav.abs()).item())
        if peak > 0:
            wav = wav[peak:].contiguous()
        max_len = int(max_sec * target_sr)
        if max_len > 0 and wav.numel() > max_len:
            wav = wav[:max_len].contiguous()
        if wav.numel() == 0:
            return None, 0.0, False
        if _rms(wav) < min_rms:
            return None, 0.0, False
        return wav.cpu(), wav.numel() / float(target_sr), True
    except Exception:
        return None, 0.0, False


def _parse_paths(values: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for v in values:
        if not v:
            continue
        out.append(Path(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-dir", action="append", default=[])
    ap.add_argument("--noise-filelist", type=str, default=None)
    ap.add_argument("--out-noise-bank", type=str, default="data/augment/noise_bank.pt")
    ap.add_argument("--noise-min-sec", type=float, default=2.0)
    ap.add_argument("--noise-max-sec", type=float, default=120.0)
    ap.add_argument("--noise-seg-sec", type=float, default=30.0)
    ap.add_argument("--noise-max-segs-per-file", type=int, default=4)
    ap.add_argument("--noise-min-rms", type=float, default=1e-4)
    ap.add_argument("--rir-dir", action="append", default=[])
    ap.add_argument("--rir-filelist", type=str, default=None)
    ap.add_argument("--out-rir-bank", type=str, default="data/augment/rir_bank.pt")
    ap.add_argument("--rir-min-sec", type=float, default=0.05)
    ap.add_argument("--rir-max-sec", type=float, default=0.5)
    ap.add_argument("--rir-min-rms", type=float, default=1e-5)
    ap.add_argument("--out-stats", type=str, default="data/augment/augment_banks_stats.json")

    args = ap.parse_args()

    target_sr = int(args.target_sr)
    gen = torch.Generator()
    gen.manual_seed(int(args.seed))

    noise_dirs = _parse_paths(args.noise_dir)
    noise_files = _read_filelist(Path(args.noise_filelist)) if args.noise_filelist else []
    noise_paths = _collect_audio_files(noise_dirs, noise_files)

    rir_dirs = _parse_paths(args.rir_dir)
    rir_files = _read_filelist(Path(args.rir_filelist)) if args.rir_filelist else []
    rir_paths = _collect_audio_files(rir_dirs, rir_files)

    noise_bank: list[torch.Tensor] = []
    noise_total_sec = 0.0
    noise_ok = 0
    noise_fail = 0

    for p in noise_paths:
        segs, kept_sec, ok = _process_noise_file(
            p,
            target_sr=target_sr,
            min_sec=float(args.noise_min_sec),
            max_sec=float(args.noise_max_sec) if args.noise_max_sec > 0 else None,
            seg_sec=float(args.noise_seg_sec),
            max_segments_per_file=int(args.noise_max_segs_per_file),
            min_rms=float(args.noise_min_rms),
            gen=gen,
        )
        if ok and segs:
            noise_bank.extend(segs)
            noise_total_sec += float(kept_sec)
            noise_ok += 1
        else:
            noise_fail += 1

    rir_bank: list[torch.Tensor] = []
    rir_total_sec = 0.0
    rir_ok = 0
    rir_fail = 0

    for p in rir_paths:
        rir, kept_sec, ok = _process_rir_file(
            p,
            target_sr=target_sr,
            min_sec=float(args.rir_min_sec),
            max_sec=float(args.rir_max_sec),
            min_rms=float(args.rir_min_rms),
        )
        if ok and rir is not None:
            rir_bank.append(rir)
            rir_total_sec += float(kept_sec)
            rir_ok += 1
        else:
            rir_fail += 1

    out_noise = Path(args.out_noise_bank)
    out_noise.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuple(noise_bank), str(out_noise))

    out_rir = Path(args.out_rir_bank)
    out_rir.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuple(rir_bank), str(out_rir))

    stats = {
        "target_sr": target_sr,
        "noise": {
            "input_files": len(noise_paths),
            "ok_files": noise_ok,
            "failed_files": noise_fail,
            "bank_items": len(noise_bank),
            "total_seconds": noise_total_sec,
            "params": {
                "min_sec": float(args.noise_min_sec),
                "max_sec": args.noise_max_sec,
                "seg_sec": float(args.noise_seg_sec),
                "max_segs_per_file": int(args.noise_max_segs_per_file),
                "min_rms": float(args.noise_min_rms),
            },
        },
        "rir": {
            "input_files": len(rir_paths),
            "ok_files": rir_ok,
            "failed_files": rir_fail,
            "bank_items": len(rir_bank),
            "total_seconds": rir_total_sec,
            "params": {
                "min_sec": float(args.rir_min_sec),
                "max_sec": float(args.rir_max_sec),
                "min_rms": float(args.rir_min_rms),
            },
        },
        "outputs": {
            "noise_bank_pt": str(out_noise.resolve()),
            "rir_bank_pt": str(out_rir.resolve()),
        },
    }

    out_stats = Path(args.out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("{}", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
