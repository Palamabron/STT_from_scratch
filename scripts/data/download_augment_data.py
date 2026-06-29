"""Download MUSAN (OpenSLR 17) and simulated RIRs (OpenSLR 28) for augmentation banks."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from loguru import logger

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIR_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


def _is_safe_archive_member(member_path: Path, dest_dir: Path) -> bool:
    resolved = (dest_dir / member_path).resolve()
    return resolved.is_relative_to(dest_dir.resolve())


def _extract_tar(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_root = out_dir.resolve()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe tar member path: {member.name}")
            if not _is_safe_archive_member(member_path, dest_root):
                raise ValueError(f"Tar member escapes destination: {member.name}")
            archive.extract(member, out_dir)


def _extract_zip(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_root = out_dir.resolve()
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe zip member path: {member.filename}")
            if not _is_safe_archive_member(member_path, dest_root):
                raise ValueError(f"Zip member escapes destination: {member.filename}")
            archive.extract(member, out_dir)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: {}", dest)
        return
    logger.info("Downloading {} -> {}", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dest",
        type=str,
        default="data/external",
        help="Root directory for extracted datasets",
    )
    ap.add_argument("--skip-musan", action="store_true")
    ap.add_argument("--skip-rir", action="store_true")
    args = ap.parse_args()

    root = Path(args.dest)
    cache = root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    musan_root = root / "musan"
    rir_root = root / "rirs"

    if not args.skip_musan:
        if not musan_root.exists():
            musan_archive = cache / "musan.tar.gz"
            _download(MUSAN_URL, musan_archive)
            _extract_tar(musan_archive, root)
            logger.info("MUSAN extracted to {}", musan_root)
        else:
            logger.info("MUSAN already present at {}", musan_root)

    if not args.skip_rir:
        if not rir_root.exists():
            rir_archive = cache / "rirs_noises.zip"
            _download(RIR_URL, rir_archive)
            _extract_zip(rir_archive, root)
            # OpenSLR 28 unpacks as RIRS_NOISES/ or rirs_noises/ — normalize to rirs/
            for candidate in (root / "rirs_noises", root / "RIRS_NOISES"):
                if candidate.exists() and not rir_root.exists():
                    candidate.rename(rir_root)
                    break
            logger.info("RIR extracted to {}", rir_root)
        else:
            logger.info("RIR already present at {}", rir_root)

    logger.info("Done. Run: make build-augment-banks")


if __name__ == "__main__":
    main()
