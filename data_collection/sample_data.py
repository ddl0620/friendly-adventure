#!/usr/bin/env python3
"""Utility for drawing a length-matched, balanced sample from a pair of JSON datasets.

The two input files are expected to be plain JSON arrays (i.e. ``[ {...}, ... ]``).
Each record must have a ``source_code`` field; length is measured in lines.

Records are grouped into length buckets. Within each bucket, we take
``min(benign_count, malicious_count)`` records from each class so that the
final dataset is both class-balanced AND length-distribution-matched.

If ``-n`` is supplied, it is treated as a *total* target across all buckets;
each bucket is allocated proportionally to how many matched pairs it contains.

The output is a single JSON array; every element gets a ``label`` field
with either ``"benign"`` or ``"malicious"``.

Example::

    ./sample_data.py -n 10000 \\
        --benign benign.json --malicious malicious.json \\
        --output sample.json

    # Custom bucket boundaries (line counts):
    ./sample_data.py --buckets 10 30 75 150 400 \\
        --benign benign.json --malicious malicious.json \\
        --output sample.json

If the input files are too large for simple parsing, install the dependency
with ``pip install ijson`` and the script will stream the JSON safely.
"""

import argparse
import json
import random
import sys
from collections import defaultdict

try:
    import ijson
except ImportError:
    ijson = None

# Default bucket boundaries (line counts, exclusive upper edges).
# A record with L lines lands in bucket i where BUCKET_EDGES[i-1] <= L < BUCKET_EDGES[i].
# Everything >= the last edge goes into one final open-ended bucket.
DEFAULT_BUCKET_EDGES = [20, 50, 100, 200, 500]


def iterate_json_array(path):
    """Yield items from a top-level JSON array contained in *path*.

    Uses ``ijson`` for streaming if available, otherwise loads the whole file.
    """
    if ijson:
        with open(path, "rb") as f:
            for item in ijson.items(f, "item"):
                yield item
    else:
        with open(path, "r") as f:
            data = json.load(f)
        for item in data:
            yield item


def line_count(record: dict) -> int:
    """Return the number of lines in record['source_code'], defaulting to 0."""
    src = record.get("source_code", "")
    return src.count("\n") + 1 if src else 0


def bucket_index(n_lines: int, edges: list[int]) -> int:
    """Return which bucket *n_lines* belongs to given the sorted *edges* list."""
    for i, edge in enumerate(edges):
        if n_lines < edge:
            return i
    return len(edges)  # open-ended final bucket


def bucket_label(idx: int, edges: list[int]) -> str:
    """Human-readable label for a bucket, e.g. '[50, 100)'."""
    if idx == 0:
        return f"[0, {edges[0]})"
    if idx == len(edges):
        return f"[{edges[-1]}, ∞)"
    return f"[{edges[idx - 1]}, {edges[idx]})"


def load_into_buckets(path: str, edges: list[int]) -> dict[int, list[dict]]:
    """Stream *path* and group records by length bucket. Returns {bucket_idx: [records]}."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for record in iterate_json_array(path):
        idx = bucket_index(line_count(record), edges)
        buckets[idx].append(record)
    return buckets


def sample_buckets(
    benign_buckets: dict[int, list[dict]],
    malicious_buckets: dict[int, list[dict]],
    edges: list[int],
    total_n: int | None,
) -> tuple[list[dict], list[dict]]:
    """
    For each bucket, draw min(benign, malicious) matched pairs.
    If *total_n* is set, scale each bucket proportionally so the grand total
    is at most *total_n* per class.

    Returns (benign_sample, malicious_sample).
    """
    all_bucket_ids = sorted(
        set(benign_buckets.keys()) | set(malicious_buckets.keys())
    )

    # How many matched pairs are available per bucket
    available: dict[int, int] = {
        b: min(len(benign_buckets.get(b, [])), len(malicious_buckets.get(b, [])))
        for b in all_bucket_ids
    }
    total_available = sum(available.values())

    # Print bucket stats
    print("\nBucket stats (before sampling):", file=sys.stderr)
    print(
        f"  {'Bucket':<16}  {'Benign':>8}  {'Malicious':>10}  {'Matched pairs':>14}",
        file=sys.stderr,
    )
    for b in all_bucket_ids:
        lbl = bucket_label(b, edges)
        nb = len(benign_buckets.get(b, []))
        nm = len(malicious_buckets.get(b, []))
        print(f"  {lbl:<16}  {nb:>8}  {nm:>10}  {available[b]:>14}", file=sys.stderr)
    print(f"  {'TOTAL':<16}  {'':>8}  {'':>10}  {total_available:>14}", file=sys.stderr)

    # Determine per-bucket quota
    if total_n is None:
        quota: dict[int, int] = available
    else:
        if total_n > total_available:
            print(
                f"warning: requested -n {total_n} exceeds available matched pairs "
                f"({total_available}); capping at {total_available}",
                file=sys.stderr,
            )
            total_n = total_available
        # Proportional allocation; give any remainder to the largest buckets
        quota = {}
        allocated = 0
        bucket_order = sorted(all_bucket_ids, key=lambda b: -available[b])
        remaining_n = total_n
        remaining_avail = total_available
        for b in bucket_order:
            if remaining_avail == 0:
                quota[b] = 0
            else:
                q = round(available[b] / remaining_avail * remaining_n)
                q = min(q, available[b])  # never exceed what's available
                quota[b] = q
                remaining_n -= q
                remaining_avail -= available[b]
            allocated += quota[b]

    # Sample within each bucket
    benign_out: list[dict] = []
    malicious_out: list[dict] = []

    print("\nSampling per bucket:", file=sys.stderr)
    for b in all_bucket_ids:
        q = quota.get(b, 0)
        if q == 0:
            continue
        b_sample = random.sample(benign_buckets.get(b, []), q)
        m_sample = random.sample(malicious_buckets.get(b, []), q)
        benign_out.extend(b_sample)
        malicious_out.extend(m_sample)
        print(
            f"  {bucket_label(b, edges):<16}  sampled {q} from each class",
            file=sys.stderr,
        )

    return benign_out, malicious_out


def main():
    parser = argparse.ArgumentParser(
        description="Sample a length-matched, balanced dataset from two JSON arrays."
    )
    parser.add_argument("--benign", default="benign.json", help="path to benign dataset")
    parser.add_argument(
        "--malicious", default="malicious.json", help="path to malicious dataset"
    )
    parser.add_argument(
        "-n",
        type=int,
        help="total number of samples to draw from EACH class (spread across buckets)",
    )
    parser.add_argument("--output", default="merged_dataset.json", help="output file")
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=DEFAULT_BUCKET_EDGES,
        metavar="EDGE",
        help=(
            "sorted line-count bucket boundaries "
            f"(default: {DEFAULT_BUCKET_EDGES})"
        ),
    )

    args = parser.parse_args()

    if args.n is not None and args.n <= 0:
        parser.error("-n must be a positive integer")

    edges = sorted(args.buckets)

    print(f"Loading benign records from {args.benign} …", file=sys.stderr)
    benign_buckets = load_into_buckets(args.benign, edges)
    total_benign = sum(len(v) for v in benign_buckets.values())
    print(f"  → {total_benign} records loaded", file=sys.stderr)

    print(f"Loading malicious records from {args.malicious} …", file=sys.stderr)
    malicious_buckets = load_into_buckets(args.malicious, edges)
    total_malicious = sum(len(v) for v in malicious_buckets.values())
    print(f"  → {total_malicious} records loaded", file=sys.stderr)

    benign_sample, malicious_sample = sample_buckets(
        benign_buckets, malicious_buckets, edges, args.n
    )

    combined = []
    for rec in benign_sample:
        rec["label"] = 0
        combined.append(rec)
    for rec in malicious_sample:
        rec["label"] = 1
        combined.append(rec)

    random.shuffle(combined)

    print(
        f"\nWriting {len(combined)} total records ({len(benign_sample)} per class) "
        f"to {args.output} …",
        file=sys.stderr,
    )
    with open(args.output, "w") as out:
        json.dump(combined, out, indent=2)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()