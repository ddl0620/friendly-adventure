#!/usr/bin/env python3
"""
Collect benign Python package source files from a local directory of archives.

This version simply iterates over the tarball/whl files in `BenignSet`,
extracts every ``.py`` file it finds and writes a JSON list of records with
fields identical to ``malicious.json``.  There are no heuristics or filename
filters – the goal is to produce benign.json with the same structure as
collect_malicious.py's output.
"""

import io
import json
import logging
import os
import sys
import tarfile
import zipfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE   = Path("./benign.json")
REGISTRY_DIR  = Path("./BenignSet")   # directory containing benign tarballs
MAX_SOURCE_MB = 5                      # skip huge members for safety
PRIMARY_FILES = {"setup.py", "__init__.py"}   # preferred files; match malicious collector
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def iter_archive_members(data: bytes, source_label: str):
    """
    Yield (member_path, read_fn) for every file in an in-memory archive.
    Handles both tar.gz and zip/whl formats.
    """
    # tar.gz first
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if member.size > MAX_SOURCE_MB * 1024 * 1024:
                    continue

                def make_tar_reader(tf=tf, m=member):
                    def _read() -> str | None:
                        try:
                            fobj = tf.extractfile(m)
                            return fobj.read().decode("utf-8", errors="replace") if fobj else None
                        except Exception:
                            return None
                    return _read

                yield member.name, make_tar_reader()
        return
    except (tarfile.TarError, EOFError):
        pass

    # fallback to zip
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.file_size > MAX_SOURCE_MB * 1024 * 1024:
                    continue

                def make_zip_reader(zf=zf, info=info):
                    def _read() -> str | None:
                        try:
                            return zf.read(info.filename).decode("utf-8", errors="replace")
                        except Exception:
                            return None
                    return _read

                yield info.filename, make_zip_reader()
        return
    except zipfile.BadZipFile:
        pass

    log.debug("Unrecognised archive format: %s", source_label)


def process_archive_simple(archive_path: Path) -> list[dict]:
    """Extract Python files from the archive, preferring setup.py / __init__.py."""
    stem = archive_path.name
    if stem.endswith(".tar.gz"):
        stem = stem[: -len(".tar.gz")]
    elif stem.endswith(".whl"):
        stem = stem[: -len(".whl")]
    pkg_name, version = (stem.rsplit("-", 1) + [""])[:2]

    try:
        with open(archive_path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        log.warning("Failed to read %s: %s", archive_path, exc)
        return []

    primary: list[dict] = []
    fallback: list[dict] = []

    for member_path, read_fn in iter_archive_members(data, archive_path.name):
        if not member_path.lower().endswith(".py"):
            continue
        basename = os.path.basename(member_path).lower()
        src = read_fn()
        if src is None:
            continue
        record = {
            "package_name": pkg_name,
            "version":      version,
            "filename":     os.path.basename(member_path),
            "source_code":  src,
        }
        if basename in PRIMARY_FILES:
            primary.append(record)
        else:
            fallback.append(record)

    # Prefer setup.py / __init__.py; fall back to other .py files only when
    # none of the primary targets exist in this archive.
    return primary if primary else fallback


def main() -> None:
    if not REGISTRY_DIR.is_dir():
        log.error("Benign set directory not found: %s", REGISTRY_DIR.resolve())
        sys.exit(1)

    records: list[dict] = []
    archive_paths = sorted(REGISTRY_DIR.glob("*.tar.gz")) + sorted(REGISTRY_DIR.glob("*.whl"))

    log.info("Scanning %d archives in %s …", len(archive_paths), REGISTRY_DIR)

    for archive_path in archive_paths:
        hits = process_archive_simple(archive_path)
        if hits:
            records.extend(hits)

    log.info("Collected %d records", len(records))

    log.info("Writing %s …", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    log.info("Saved → %s  (%.1f MB)", OUTPUT_FILE, OUTPUT_FILE.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
