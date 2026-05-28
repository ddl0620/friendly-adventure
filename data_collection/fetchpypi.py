"""
fetch_benign_pypi.py

Downloads sdist (.tar.gz) packages from PyPI and saves them into
BenignSet/ so that collect_benign.py can process them.

Two sourcing strategies (both run by default):
  1. TOP-N     – the N most downloaded packages this month
                 (from hugovk.github.io/top-pypi-packages)
  2. RANDOM    – random walk of the PyPI /simple index for breadth

Why sdist, not wheel?
  Wheels often omit setup.py.  sdist always contains it.

Usage:
  python fetchpypi.py                  # both strategies (top 3000 + random 5000)
  python fetchpypi.py --top 2000       # only top-N
  python fetchpypi.py --random 3000    # only random
  python fetchpypi.py --workers 8      # parallel downloads

Output: BenignSet/<package>-<version>.tar.gz
"""

import argparse
import concurrent.futures
import json
import logging
import random
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent.resolve()
OUTPUT_DIR  = SCRIPT_DIR / "BenignSet"
OUTPUT_DIR.mkdir(exist_ok=True)

PYPI_SIMPLE   = "https://pypi.org/simple/"
PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
TOP_PACKAGES_URL = (
    "https://hugovk.github.io/top-pypi-packages/"
    "top-pypi-packages-30-days.min.json"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "benign-collector/1.0 (research)"})

TIMEOUT      = 20   # seconds per request
MAX_SDIST_MB = 5    # skip sdists larger than this (huge packages → slow)
RETRY_SLEEP  = 2    # seconds between retries


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_json(url: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                log.debug(f"Failed {url}: {e}")
    return None


def get_sdist_url(package: str) -> tuple[str, str] | None:
    """Return (filename, url) of the latest sdist for a package, or None."""
    data = get_json(PYPI_JSON_URL.format(package=package))
    if not data:
        return None
    urls = data.get("urls", [])
    for u in urls:
        if u.get("packagetype") == "sdist":
            size_mb = u.get("size", 0) / 1_048_576
            if size_mb <= MAX_SDIST_MB:
                return u["filename"], u["url"]
    # Check all versions if latest has no small sdist
    releases = data.get("releases", {})
    for version in sorted(releases.keys(), reverse=True)[:5]:
        for u in releases[version]:
            if u.get("packagetype") == "sdist":
                size_mb = u.get("size", 0) / 1_048_576
                if size_mb <= MAX_SDIST_MB:
                    return u["filename"], u["url"]
    return None


def download_sdist(package: str) -> bool:
    """Download the sdist for a package into OUTPUT_DIR. Return True on success."""
    result = get_sdist_url(package)
    if not result:
        return False
    filename, url = result

    dest = OUTPUT_DIR / filename
    if dest.exists():
        return True  # already have it

    try:
        r = SESSION.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        return True
    except Exception as e:
        log.debug(f"  Download failed {package}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ─────────────────────────────────────────────
# STRATEGY 1 – TOP-N MOST DOWNLOADED
# ─────────────────────────────────────────────

def get_top_packages(n: int) -> list[str]:
    log.info(f"Fetching top-{n} packages list ...")
    data = get_json(TOP_PACKAGES_URL)
    if not data:
        log.warning("Could not fetch top-packages list; falling back to hardcoded top-50.")
        return _hardcoded_top50()
    rows = data.get("rows", [])[:n]
    names = [r["project"] for r in rows]
    log.info(f"  Got {len(names)} package names.")
    return names


def _hardcoded_top50() -> list[str]:
    """Fallback when the top-packages URL is unreachable."""
    return [
        "boto3", "botocore", "urllib3", "requests", "certifi",
        "charset-normalizer", "idna", "setuptools", "pip", "wheel",
        "six", "python-dateutil", "s3transfer", "pyyaml", "packaging",
        "cryptography", "cffi", "pycparser", "numpy", "pandas",
        "click", "colorama", "tqdm", "attrs", "typing-extensions",
        "pytz", "jinja2", "markupsafe", "werkzeug", "flask",
        "django", "sqlalchemy", "pillow", "scipy", "matplotlib",
        "scikit-learn", "tensorflow", "torch", "transformers", "huggingface-hub",
        "fastapi", "uvicorn", "pydantic", "httpx", "aiohttp",
        "paramiko", "fabric", "invoke", "pytest", "coverage",
    ]


# ─────────────────────────────────────────────
# STRATEGY 2 – RANDOM WALK OF THE SIMPLE INDEX
# ─────────────────────────────────────────────

def get_all_package_names() -> list[str]:
    """Fetch the full PyPI simple index (~700k packages). Cached locally."""
    cache = SCRIPT_DIR / ".pypi_simple_cache.txt"
    if cache.exists() and cache.stat().st_mtime > time.time() - 86400:
        names = cache.read_text().splitlines()
        log.info(f"Loaded {len(names)} package names from cache.")
        return names

    log.info("Fetching PyPI simple index (this takes ~30s) ...")
    try:
        r = SESSION.get(PYPI_SIMPLE, timeout=60,
                        headers={"Accept": "application/vnd.pypi.simple.v1+json"})
        r.raise_for_status()
        data = r.json()
        names = [p["name"] for p in data.get("projects", [])]
    except Exception:
        # Fallback: parse HTML
        r = SESSION.get(PYPI_SIMPLE, timeout=60)
        import re
        names = re.findall(r'href="[^"]*">([^<]+)</a>', r.text)

    log.info(f"  Got {len(names)} packages from simple index.")
    cache.write_text("\n".join(names))
    return names


def get_random_packages(n: int) -> list[str]:
    all_names = get_all_package_names()
    sample = random.sample(all_names, min(n, len(all_names)))
    log.info(f"Sampled {len(sample)} random packages.")
    return sample


# ─────────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────────

def download_batch(packages: list[str], workers: int, label: str) -> int:
    """Download sdists for a list of packages. Return count of successes."""
    already    = sum(1 for p in OUTPUT_DIR.glob("*.tar.gz"))
    log.info(f"[{label}] Downloading {len(packages)} packages "
             f"(already have {already} in BenignSet) ...")
    success = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_sdist, p): p for p in packages}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            pkg = futures[fut]
            ok  = fut.result()
            if ok:
                success += 1
            if i % 100 == 0:
                log.info(f"  [{label}] {i}/{len(packages)} done, "
                         f"{success} saved so far ...")
    log.info(f"[{label}] Done. {success}/{len(packages)} downloaded.")
    return success


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download benign PyPI sdists")
    parser.add_argument("--top",     type=int, default=3000,
                        help="Download top-N most-downloaded packages (default 3000)")
    parser.add_argument("--random",  type=int, default=5000,
                        help="Download N random packages (default 5000)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Parallel download threads (default 12)")
    parser.add_argument("--no-top",    action="store_true")
    parser.add_argument("--no-random", action="store_true")
    args = parser.parse_args()

    total = 0

    if not args.no_top and args.top > 0:
        pkgs = get_top_packages(args.top)
        total += download_batch(pkgs, args.workers, "top")

    if not args.no_random and args.random > 0:
        pkgs = get_random_packages(args.random)
        total += download_batch(pkgs, args.workers, "random")

    log.info(f"\nAll done. {total} new sdists saved to {OUTPUT_DIR}")
    log.info("Now run:  python collect_benign.py")


if __name__ == "__main__":
    main()
