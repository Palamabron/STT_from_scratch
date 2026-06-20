from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torchaudio
from loguru import logger


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


def _stable_name(audio_path: str) -> str:
    h = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:16]
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest", type=str, required=True)
    ap.add_argument("--output-manifest", type=str, required=True)
    ap.add_argument("--output-audio-dir", type=str, required=True)
    ap.add_argument("--target-sr", type=int, default=16000)
    args = ap.parse_args()

    in_manifest = Path(args.input_manifest)
    out_manifest = Path(args.output_manifest)
    out_audio_dir = Path(args.output_audio_dir)
    target_sr = int(args.target_sr)

    if not in_manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {in_manifest}")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0

    with (
        in_manifest.open("r", encoding="utf-8") as f_in,
        out_manifest.open("w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            obj: dict[str, Any] = json.loads(line)
            audio_fp = obj.get("audio_filepath")
            if not isinstance(audio_fp, str) or not audio_fp:
                n_fail += 1
                continue

            src = Path(audio_fp)
            if not src.exists():
                n_fail += 1
                continue

            try:
                wav, sr = _load_mono(src)
                wav = wav.to(dtype=torch.float32)
                wav = _resample(wav, sr, target_sr)

                out_name = _stable_name(str(src.resolve())) + ".wav"
                dst = (out_audio_dir / out_name).resolve()
                torchaudio.save(str(dst), wav.view(1, -1), sample_rate=target_sr)

                obj["audio_filepath"] = str(dst)
                obj["duration"] = float(wav.numel()) / float(target_sr)

                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception:
                n_fail += 1
                continue

    logger.info(
        "{}",
        json.dumps(
            {
                "input_manifest": str(in_manifest),
                "output_manifest": str(out_manifest),
                "output_audio_dir": str(out_audio_dir),
                "target_sr": target_sr,
                "ok": n_ok,
                "failed": n_fail,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
