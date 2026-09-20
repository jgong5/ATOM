#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Diff two `t5_trace.py` records -- the second half of T5's check, which asks
for the captured structure at TP2 *diffed against* TP1, not summarised.

Every section names its members. A count with no list under it would be the
defect principle 7 exists to catch.

    t5_diff.py tp1.json tp2.json
"""

from __future__ import annotations

import json
import sys


def _fwd(rec: dict) -> dict:
    f = rec.get("forward") or {}
    # A refused run still carries everything it got to before the refusal.
    return f.get("inventory_partial") if not f.get("ok") else f


def _shape_key(o: dict) -> str:
    return f"{o['op']}({';'.join(','.join(s) for s in o['in'])})"


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    recs = []
    for p in argv:
        with open(p) as fh:
            recs.append(json.load(fh))
    a, b = recs
    la, lb = f"TP{a['tp']}", f"TP{b['tp']}"

    print("=== provenance ===")
    for lbl, r in ((la, a), (lb, b)):
        c = r["config"]
        print(f"{lbl}: stage={r['stage']} atom={r['atom_file']} torch={r['torch']}")
        print(
            f"     tp={c['tensor_parallel_size']} tp_world_size={c['tp_world_size']} "
            f"tp_group_world_size={c['tp_group_world_size']} "
            f"tp_group_physical={c['tp_group_physical']} "
            f"fake_eplb={c['fake_eplb']} dtype={c['torch_dtype']}"
        )
        fw = r.get("forward") or {}
        if fw and not fw.get("ok"):
            print(f"     REFUSED {fw['error_type']}: {fw['error']}")
            print(f"     at {fw['at']}")
            if fw.get("raised_at"):
                print(f"     raised at {fw['raised_at']}")

    print("\n=== parameter geometry ===")
    ga, gb = a["geometry"], b["geometry"]
    print(f"{la}: {ga['n_param_elements']:,} elements in {ga['n_params']} params")
    print(f"{lb}: {gb['n_param_elements']:,} elements in {gb['n_params']} params")
    only_a = sorted(set(ga["params"]) - set(gb["params"]))
    only_b = sorted(set(gb["params"]) - set(ga["params"]))
    print(f"params only in {la}: {len(only_a)}  only in {lb}: {len(only_b)}")
    for n in only_a[:20]:
        print(f"  only {la}: {n} {ga['params'][n][0]}")
    for n in only_b[:20]:
        print(f"  only {lb}: {n} {gb['params'][n][0]}")

    # Which parameters changed shape, grouped by the *pattern* of the change,
    # so 944 rows become a handful of named classes.
    classes: dict = {}
    unchanged = 0
    for n in sorted(set(ga["params"]) & set(gb["params"])):
        sa, sb = ga["params"][n][0], gb["params"][n][0]
        if sa == sb:
            unchanged += 1
            continue
        ratio = tuple(
            (int(x) // int(y)) if int(y) and int(x) % int(y) == 0 else f"{x}->{y}"
            for x, y in zip(sa, sb)
        )
        key = (len(sa), ratio)
        classes.setdefault(key, []).append((n, sa, sb))
    print(f"unchanged shape: {unchanged} params")
    for (ndim, ratio), members in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        ex = members[0]
        print(
            f"  {len(members):4d} params  ndim={ndim} per-dim factor {ratio}"
            f"   e.g. {ex[0]}: {ex[1]} -> {ex[2]}"
        )
    ea = sum(_numel(ga["params"][n][0]) for n in ga["params"])
    eb = sum(_numel(gb["params"][n][0]) for n in gb["params"])
    both = sorted(set(ga["params"]) & set(gb["params"]))
    same = sum(
        _numel(ga["params"][n][0])
        for n in both
        if ga["params"][n][0] == gb["params"][n][0]
    )
    print(
        f"  elements held identically at both widths: {same:,} "
        f"({100.0 * same / ea:.2f}% of {la})"
    )
    print(
        f"  elements that shard: {ea - same:,} -> {eb - same:,} "
        f"(factor {(ea - same) / max(eb - same, 1):.4f})"
    )

    # T68: buffers are not parameters, so `--load_dummy` and the meta wrapper
    # never touch them; a buffer is the likeliest thing to escape the mode.
    # Collecting them and then diffing only params is how that would go unseen.
    print("  buffers:")
    ba, bb = ga.get("buffers", {}), gb.get("buffers", {})
    print(
        f"    {la}: {len(ba)} on {ga.get('buffer_devices')}   "
        f"{lb}: {len(bb)} on {gb.get('buffer_devices')}"
    )
    for n in sorted(set(ba) | set(bb)):
        sa_, sb_ = ba.get(n), bb.get(n)
        if sa_ != sb_:
            print(f"    DIFFERS {n}: {la}={sa_} {lb}={sb_}")
    if ba == bb:
        print("    identical at both widths")

    print("\n=== operator inventory ===")
    ia, ib = _fwd(a), _fwd(b)
    if not ia or not ib:
        print("one side has no inventory; nothing to diff")
        return 1
    for lbl, r, i in ((la, a, ia), (lb, b, ib)):
        if r.get("diagnostic_inventory"):
            print(
                f"{lbl}: DIAGNOSTIC (--skip-triton). Raw @triton.jit launches "
                "were recorded and not run; this inventory is an enumeration, "
                "NOT a cost model input (TritonLaunchRecorder's own rule)."
            )
        entries = i.get("shape_entries")
        free = i.get("non_numeric_shape_entries")
        if entries is not None:
            print(
                f"{lbl}: {free} of {entries} shape entries are non-numeric -> "
                f"{'symbolic' if free else 'CONCRETE (`02` D10.1 T5 fallback)'}"
            )
    print(f"{la}: {ia['n_ops']} dispatches, {ia['n_distinct_ops']} distinct ops")
    print(f"{lb}: {ib['n_ops']} dispatches, {ib['n_distinct_ops']} distinct ops")
    ca, cb = ia["op_counts"], ib["op_counts"]
    for op in sorted(set(ca) | set(cb)):
        na, nb = ca.get(op, 0), cb.get(op, 0)
        flag = "  " if na == nb else ("<<" if na > nb else ">>")
        print(f"  {flag} {na:6d} {nb:6d}  {op}")

    print("\n=== shape signatures present at one width only ===")
    sa = {_shape_key(o) for o in ia["ops"]}
    sb = {_shape_key(o) for o in ib["ops"]}
    for k in sorted(sa - sb):
        print(f"  only {la}: {k}")
    for k in sorted(sb - sa):
        print(f"  only {lb}: {k}")

    print("\n=== raw triton launches (never dispatched; not above) ===")
    for lbl, r in ((la, a), (lb, b)):
        t = (r.get("forward") or {}).get("triton")
        if not t:
            print(f"{lbl}: not recorded")
            continue
        print(
            f"{lbl}: {t['n_triton_launches']} launches / "
            f"{t['n_distinct_triton_kernels']} kernels"
        )
        for k in t["triton_kernels"]:
            print(f"    {k['launches']:6d}  {k['kernel']}  {k['defined_at']}")
    return 0


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


if __name__ == "__main__":
    sys.exit(main())
