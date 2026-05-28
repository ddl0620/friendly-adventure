#!/usr/bin/env python3
"""
Collect malicious Python scripts from the pypi_malregistry dataset.

Targets files at the 1st level inside each package's tar.gz:
  - setup.py         (always collected)
  - pytoolib.py     (always collected)
  - __init__.py      (collected only if suspiciously long, >= INIT_MIN_LINES lines)

Output: malicious.json  —  list of { package_name, version, filename, source_code }
"""

import os
import re
import sys
import json
import tarfile
import zipfile
import io
import logging
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
REGISTRY_DIR   = Path("./pypi_malregistry")   # root of the cloned repo
OUTPUT_FILE    = Path("./malicious.json")
INIT_MIN_LINES = 50        # __init__.py threshold: flag if >= this many lines
TARGET_FILES   = {"setup.py", "pytoolib.py"}  # always collected at depth-1
MAX_SOURCE_MB  = 5         # skip files larger than this (safety cap)

# Packages with non-standard layout: skip setup.py, collect __init__.py at depth-2
# e.g.  ForgePy-1.0.3/ForgePy/__init__.py
FORGE_PACKAGES = {"forgepy", "forgepys", "forgyp", "forgyps"}  # lowercased for comparison
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────






def iter_archive_members(archive_path: Path):
    """
    Yield (member_path, read_fn) for every file in the archive,
    transparently handling both real .tar.gz and zip/.whl files
    (even when disguised with a .tar.gz extension).

    read_fn() → str | None  (decoded UTF-8 text, or None on error)
    """
    # ── Try as tar.gz first ───────────────────────────────────────────────────
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if member.size > MAX_SOURCE_MB * 1024 * 1024:
                    continue

                # Capture member name for the closure
                def make_tar_reader(tf=tf, m=member):
                    def _read() -> str | None:
                        try:
                            fobj = tf.extractfile(m)
                            return fobj.read().decode("utf-8", errors="replace") if fobj else None
                        except Exception:
                            return None
                    return _read

                yield member.name, make_tar_reader()
        return  # success — don't fall through to zip
    except (tarfile.TarError, EOFError):
        pass  # not a valid tar.gz — try zip below

    # ── Fallback: try as zip (covers .whl and mis-labelled archives) ──────────
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
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

    log.warning("Unrecognised archive format, skipping: %s", archive_path)


def process_archive(pkg_name: str, version: str, archive_path: Path) -> list[dict]:
    """
    Open a single archive and return a list of hit records.
    Each record: { package_name, version, filename, source_code }

    Special case — ForgePy / ForgePys / ForgyP / ForgyPs:
      - Skip setup.py entirely
      - Collect __init__.py at depth-2 (e.g. ForgePy-1.0.3/ForgePy/__init__.py)
        with no minimum line-count requirement
    """
    hits = []
    is_forge = pkg_name.lower() in FORGE_PACKAGES

    try:
        for member_path, read_fn in iter_archive_members(archive_path):
            path_parts    = member_path.strip("/").split("/")
            filename_lower = path_parts[-1].lower()
            depth          = len(path_parts)

            # ── Forge-package special rules ───────────────────────────────────
            if is_forge:
                # depth-2: <pkg-version>/<pkg-subdir>/__init__.py
                if depth == 3 and filename_lower == "__init__.py":
                    src = read_fn()
                    if src is not None:
                        hits.append({
                            "package_name": pkg_name,
                            "version":      version,
                            "filename":     path_parts[-1],
                            "source_code":  src,
                        })
                # skip everything else (including setup.py)
                continue

            # ── Default rules (depth-1 only) ──────────────────────────────────
            if depth != 2:
                continue

            if filename_lower in TARGET_FILES:
                src = read_fn()
                if src is not None:
                    hits.append({
                        "package_name": pkg_name,
                        "version":      version,
                        "filename":     path_parts[-1],
                        "source_code":  src,
                    })

            elif filename_lower == "__init__.py":
                src = read_fn()
                if src is not None:
                    line_count = src.count("\n") + 1
                    if line_count >= INIT_MIN_LINES:
                        log.debug(
                            "%s/%s  __init__.py flagged (%d lines)",
                            pkg_name, version, line_count,
                        )
                        hits.append({
                            "package_name": pkg_name,
                            "version":      version,
                            "filename":     path_parts[-1],
                            "source_code":  src,
                        })

    except OSError as exc:
        log.warning("Failed to open %s: %s", archive_path, exc)

    return hits


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not REGISTRY_DIR.is_dir():
        log.error("Registry directory not found: %s", REGISTRY_DIR.resolve())
        sys.exit(1)

    records: list[dict] = []
    pkg_count = ver_count = hit_count = 0

    # Directory layout:  REGISTRY_DIR / <package_name> / <version> / <archive>.tar.gz
    pkg_dirs = sorted(
        p for p in REGISTRY_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    total_pkgs = len(pkg_dirs)
    log.info("Scanning %d package directories under %s …", total_pkgs, REGISTRY_DIR)

    for idx, pkg_dir in enumerate(pkg_dirs, 1):
        pkg_name = pkg_dir.name
        pkg_count += 1

        if idx % 500 == 0:
            log.info(
                "Progress: %d / %d packages  |  %d versions scanned  |  %d hits so far",
                idx, total_pkgs, ver_count, hit_count,
            )

        for ver_dir in sorted(pkg_dir.iterdir()):
            if not ver_dir.is_dir():
                continue

            version = ver_dir.name
            ver_count += 1

            # Collect .tar.gz archives (some may actually be zip/whl in disguise)
            archives = list(ver_dir.glob("*.tar.gz")) + list(ver_dir.glob("*.whl"))

            for archive_path in archives:
                hits = process_archive(pkg_name, version, archive_path)
                if hits:
                    records.extend(hits)
                    hit_count += len(hits)

    log.info(
        "Done. Packages: %d  |  Versions: %d  |  Malicious files found: %d",
        pkg_count, ver_count, hit_count,
    )

    log.info("Writing %s …", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    log.info("Saved → %s  (%.1f MB)", OUTPUT_FILE, OUTPUT_FILE.stat().st_size / 1e6)


if __name__ == "__main__":
    main()