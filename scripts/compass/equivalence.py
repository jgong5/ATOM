"""Are two runs of the same target the same run?

Used for the CPU-only replay against the GPU-resident simulator: same target,
same oracle table, same workload, executed in a device-free container and in a
GPU one. The question is whether removing the devices changed anything the
engine decided or predicted.

The answer has two strengths and they are not interchangeable.

*Per-request identity* -- request i has the same TTFT, TPOT and latency on both
sides -- is the strong one, and it is the only one available when the requests
differ from each other.

*Equivalence up to permutation of interchangeable requests* is the weaker one.
Two requests are interchangeable when the **declared workload** gives them the
same arrival instant, input length and output length; nothing about the results
enters that. A workload of 64 identical requests is one such class, and a
permutation within it is the same workload presented in a different order.

The classes are computed from the declared workload before any result is read,
so a permutation claim cannot be manufactured after the fact by noticing that
the mismatched requests happen to have equal values. A heterogeneous workload
has classes of size one, permutation buys nothing there, and the tool says so
rather than falling back to comparing distributions.

    python scripts/compass/equivalence.py <a.json> <b.json> [--steps-a F --steps-b F]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Everything a step row records except the fields that are allowed to differ.
#: `tick` is the engine's step counter, which starts wherever the process's own
#: history left it; `started_at` and `seconds` are wall readings.
STEP_IGNORED = {"tick", "started_at", "seconds", "t"}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def classes(workload: list[dict]) -> dict[tuple, list[int]]:
    """Interchangeable requests, from the declaration alone.

    Computed before any result is looked at. That ordering is the point: a
    class discovered from the results would be a description of the mismatch
    rather than a property of the workload.
    """
    out = defaultdict(list)
    for i, row in enumerate(workload):
        out[(round(float(row.get("arrival_s") or 0.0), 9),
             int(row.get("input_tokens") or 0),
             int(row.get("output_tokens") or 0))].append(i)
    return dict(out)


def per_request(run: dict) -> dict[int, tuple]:
    """TTFT, TPOT and latency by request index, with the run epoch removed.

    Only the epoch: each side's own first arrival is subtracted, and nothing
    else is normalised.
    """
    records = {r.get("seq_id") or r.get("request_id"): r
               for r in (run.get("engine") or {}).get("requests") or []}
    rows = {}
    order = sorted(records.values(), key=lambda r: (r["arrive_time"],
                                                    str(r["request_id"])))
    for i, rec in enumerate(order):
        first, arrive, finish = (rec.get("first_token_time"),
                                 rec["arrive_time"], rec["finish_time"])
        rows[i] = (round((first - arrive), 9) if first else None,
                   round((finish - arrive), 9))
    return rows


def results_key(run: dict) -> list[tuple]:
    out = []
    for r in run.get("results") or []:
        usage = ((r.get("response") or {}).get("usage") or {})
        choices = ((r.get("response") or {}).get("choices") or [{}])
        out.append((r["index"], bool(r["ok"]),
                    usage.get("completion_tokens"),
                    choices[0].get("finish_reason")))
    return sorted(out)


def steps_of(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def compare_steps(a: list[dict], b: list[dict]) -> dict:
    same_len = len(a) == len(b)
    differing = 0
    for x, y in zip(a, b):
        if ({k: v for k, v in x.items() if k not in STEP_IGNORED}
                != {k: v for k, v in y.items() if k not in STEP_IGNORED}):
            differing += 1
    predicted = lambda rows: round(sum(  # noqa: E731
        float(r.get("predicted_seconds") or r.get("seconds") or 0.0)
        for r in rows), 9)
    return {"rows_a": len(a), "rows_b": len(b), "same_length": same_len,
            "decisions_differing": differing,
            "summed_seconds_a": predicted(a), "summed_seconds_b": predicted(b)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--steps-a")
    ap.add_argument("--steps-b")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    run_a, run_b = load(args.a), load(args.b)
    workload = run_a.get("workload") or []
    groups = classes(workload)
    sizes = Counter(len(v) for v in groups.values())
    homogeneous = len(groups) == 1 and len(workload) > 1

    print(f"workload: {len(workload)} requests in {len(groups)} declared "
          f"interchangeability class(es); sizes "
          + ", ".join(f"{n}x{size}" for size, n in sorted(sizes.items())))

    report = {"requests": len(workload), "classes": len(groups),
              "homogeneous": homogeneous}

    same_results = results_key(run_a) == results_key(run_b)
    print(f"per-request (index, ok, completion_tokens, finish_reason): "
          f"{'identical' if same_results else 'DIFFER'}")
    report["results_identical"] = same_results

    ra, rb = per_request(run_a), per_request(run_b)
    mismatched = sorted(i for i in set(ra) | set(rb) if ra.get(i) != rb.get(i))
    print(f"per-request timings, epoch-normalised: "
          f"{len(ra) - len(mismatched)}/{len(ra)} identical")
    report["timings_identical"] = not mismatched
    report["mismatched_indices"] = mismatched

    verdict = "per-request identity"
    if mismatched:
        # Can permutation account for it? Only within a class declared before
        # any of this was read, and only if the two sides hold the same
        # multiset of readings inside that class.
        by_class = {k: v for k, v in groups.items()
                    if any(i in v for i in mismatched)}
        accountable = all(
            Counter(ra.get(i) for i in members)
            == Counter(rb.get(i) for i in members)
            for members in by_class.values())
        if accountable and any(len(v) > 1 for v in by_class.values()):
            verdict = "equivalence up to permutation within declared classes"
            print(f"  every mismatch lies inside a declared class whose two "
                  f"sides hold the same multiset of readings: "
                  f"{len(by_class)} class(es) involved")
        else:
            verdict = "NOT EQUIVALENT"
            singleton = [k for k, v in by_class.items() if len(v) == 1]
            if singleton:
                print(f"  {len(singleton)} mismatch(es) are in classes of one "
                      f"request, where permutation is not available: these are "
                      f"genuine per-request differences")
            else:
                print("  the mismatched readings are not a permutation of each "
                      "other inside any declared class")

    if args.steps_a or args.steps_b:
        steps = compare_steps(steps_of(args.steps_a), steps_of(args.steps_b))
        report["steps"] = steps
        print(f"steps: {steps['rows_a']} vs {steps['rows_b']} rows, "
              f"{steps['decisions_differing']} differing in anything but "
              f"{sorted(STEP_IGNORED)}")
        print(f"summed step seconds: {steps['summed_seconds_a']} vs "
              f"{steps['summed_seconds_b']}")
        if not steps["same_length"] or steps["decisions_differing"]:
            verdict = "NOT EQUIVALENT"

    if not same_results:
        verdict = "NOT EQUIVALENT"
    report["verdict"] = verdict
    print(f"VERDICT: {verdict}")
    if verdict == "equivalence up to permutation within declared classes":
        print("  Per-request identity is not claimed. This label is only "
              "available because the workload declares interchangeable "
              "requests; it would not be available for a heterogeneous one.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if verdict != "NOT EQUIVALENT" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
