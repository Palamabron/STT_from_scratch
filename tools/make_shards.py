from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(frozen=True, slots=True)
class ManifestItem:
    audio_path: str
    text: str
    duration: float | None
    language: str


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {e}") from e


def parse_item(obj: dict[str, Any]) -> ManifestItem | None:
    audio_path = obj.get("audio_filepath")
    text = (obj.get("text") or "").strip()
    if not isinstance(audio_path, str) or not audio_path:
        return None
    if not isinstance(text, str) or not text:
        return None

    lang = obj.get("language", "unknown")
    if not isinstance(lang, str) or not lang:
        lang = "unknown"

    dur = obj.get("duration", None)
    duration: float | None
    if dur is None:
        duration = None
    else:
        try:
            duration = float(dur)
            if duration <= 0:
                duration = None
        except (TypeError, ValueError):
            duration = None

    return ManifestItem(audio_path=audio_path, text=text, duration=duration, language=lang)


def _tar_add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    ti = tarfile.TarInfo(name=name)
    ti.size = len(payload)
    tar.addfile(ti, io.BytesIO(payload))


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def make_shards(
    manifest_path: Path,
    out_dir: Path,
    samples_per_shard: int,
    shard_prefix: str,
    start_shard: int = 0,
    max_shards: int | None = None,
    audio_ext: str = "wav",
    strict_exists: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "manifest_sharded.jsonl"
    _ensure_parent(manifest_out)

    shard_id = start_shard
    sample_in_shard = 0
    global_sample_id = 0

    tar_path = out_dir / f"{shard_prefix}-{shard_id:06d}.tar"
    tar = tarfile.open(tar_path, mode="w")

    def open_next_shard() -> None:
        nonlocal shard_id, sample_in_shard, tar, tar_path
        tar.close()
        shard_id += 1
        if max_shards is not None and (shard_id - start_shard) >= max_shards:
            raise StopIteration
        sample_in_shard = 0
        tar_path = out_dir / f"{shard_prefix}-{shard_id:06d}.tar"
        tar = tarfile.open(tar_path, mode="w")

    with manifest_out.open("w", encoding="utf-8") as fout:
        try:
            for obj in iter_jsonl(manifest_path):
                item = parse_item(obj)
                if item is None:
                    continue

                ap = Path(item.audio_path)
                if not ap.exists():
                    if strict_exists:
                        raise FileNotFoundError(f"Missing audio file: {ap}")
                    else:
                        continue

                if sample_in_shard >= samples_per_shard:
                    open_next_shard()

                key = f"{global_sample_id:09d}"
                audio_name = f"{key}.{audio_ext}"
                meta_name = f"{key}.json"

                audio_bytes = ap.read_bytes()

                meta = {
                    "text": item.text,
                    "language": item.language,
                    "duration": item.duration,
                    "audio_filepath": str(ap),
                }
                meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

                _tar_add_bytes(tar, audio_name, audio_bytes)
                _tar_add_bytes(tar, meta_name, meta_bytes)

                rec = {
                    "__key__": key,
                    "shard": tar_path.name,
                    "audio_key": audio_name,
                    "meta_key": meta_name,
                    "text": item.text,
                    "language": item.language,
                    "duration": item.duration,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                sample_in_shard += 1
                global_sample_id += 1
        except StopIteration:
            pass
        finally:
            tar.close()

    return manifest_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Input JSONL manifest (audio_filepath, text, ...).",
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for shards.")
    p.add_argument("--samples_per_shard", type=int, default=5000)
    p.add_argument("--shard_prefix", type=str, default="shard")
    p.add_argument("--start_shard", type=int, default=0)
    p.add_argument("--max_shards", type=int, default=None)
    p.add_argument("--audio_ext", type=str, default="wav")
    p.add_argument(
        "--no_strict_exists",
        action="store_true",
        help="Skip missing audio files instead of failing.",
    )
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest_out = make_shards(
        manifest_path=manifest_path,
        out_dir=out_dir,
        samples_per_shard=int(args.samples_per_shard),
        shard_prefix=str(args.shard_prefix),
        start_shard=int(args.start_shard),
        max_shards=None if args.max_shards is None else int(args.max_shards),
        audio_ext=str(args.audio_ext).lstrip("."),
        strict_exists=not bool(args.no_strict_exists),
    )
    logger.info("Wrote shards to: {}", out_dir)
    logger.info("Wrote sharded manifest to: {}", manifest_out)


if __name__ == "__main__":
    main()
