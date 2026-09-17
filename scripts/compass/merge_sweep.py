"""Merge sharded calibration measurements into one table, with a manifest.

A long calibration collected as several fresh processes is not the same thing
as one long process's calibration, and the merged table has to say so rather
than look like a sweep that finished. So this writes two files: the table the
oracle reads, and a manifest recording which shard and which pass each row came
from, which shards are missing, and the limitation.

Rows are labelled by their own `started_at` against the per-round intervals in
each shard's rounds file, so the label needs no cooperation from the engine.

    python scripts/compass/merge_sweep.py <shard-dir> --out <table.jsonl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LIMITATION = (
    "Collected as separately identified fresh processes, not as one sweep. "
    "Each shard paid its own process warmup, and any effect that depends on a "
    "single long process's history is not sampled -- including whatever ended "
    "the monolithic run in a device fault. This table is a calibration; it is "
    "not evidence that the monolithic fault is resolved."
)


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            # The sink is killed mid-write when a process ends; a torn final
            # line costs one step, not the shard.
            pass
    return out


def _label(rows: list[dict], rounds: list[dict]) -> dict:
    """Attribute each row to the round whose interval contains its start."""
    counts = {"warmup": 0, "steady": 0, "unattributed": 0}
    for row in rows:
        at = row.get("started_at")
        hit = None
        if at is not None:
            for r in rounds:
                if r["t0"] <= at <= r["t1"]:
                    hit = r
                    break
        if hit is None:
            counts["unattributed"] += 1
            row["sweep_pass"] = None
            row["sweep_round"] = None
        else:
            counts[hit["pass"]] += 1
            row["sweep_pass"] = hit["pass"]
            row["sweep_round"] = hit["round"]
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard_dir")
    ap.add_argument("--out", required=True, help="the merged table")
    ap.add_argument("--rank", default="", help="rank suffix, e.g. tp0")
    args = ap.parse_args(argv)

    root = Path(args.shard_dir)
    suffix = f".{args.rank}" if args.rank else ""
    shards, merged, expected = [], [], set()
    for rounds_path in sorted(root.glob("shard*.rounds.json")):
        index = rounds_path.name.split(".")[0]
        expected.add(index)
        table = root / f"{index}{suffix}.jsonl"
        record = {"shard": index, "rounds_file": rounds_path.name,
                  "table": table.name}
        if not table.exists():
            # A shard that failed had its rows renamed away; it is missing
            # here on purpose and is reported as missing, not skipped quietly.
            record["status"] = "missing"
            record["rows"] = 0
            shards.append(record)
            continue
        meta = json.loads(rounds_path.read_text())
        rows = _rows(table)
        record["status"] = "merged"
        record["rows"] = len(rows)
        record["shard_label"] = meta.get("shard")
        record["rounds"] = len(meta.get("rounds", []))
        record["passes"] = _label(rows, meta.get("rounds", []))
        merged.extend(rows)
        shards.append(record)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row) + "\n")

    manifest = {
        "table": out.name, "rows": len(merged),
        "shards": shards,
        "missing": sorted(s["shard"] for s in shards
                          if s["status"] == "missing"),
        "limitation": LIMITATION,
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"{len(merged)} rows from {len(shards)} shard(s) -> {out}")
    for s in shards:
        passes = s.get("passes") or {}
        print(f"  {s['shard']}: {s['status']}, {s['rows']} rows"
              + (f", warmup {passes.get('warmup', 0)} / steady "
                 f"{passes.get('steady', 0)} / unattributed "
                 f"{passes.get('unattributed', 0)}" if passes else ""))
    if manifest["missing"]:
        print(f"  MISSING: {', '.join(manifest['missing'])} -- the table is "
              f"incomplete and the manifest says which rounds are absent")
        return 1
    print(f"  manifest {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
