"""Emit a graph covering token counts nobody traced, so they can be priced.

A prefill step's cost is not smooth in its token count: the gemm whose M *is* the
token count halves between M=4376 and M=4500 when the library switches tile, so
4588 tokens cost less than 4376. Interpolating across that is wrong by 2x with
nothing to warn you, and there is no enumerable set of prefill shapes to measure
the way decode has capture rungs.

What there is, is a rule. The operator *list* does not depend on the token count
-- 319 operators and 19 kinds at every unchunked prefill size traced -- and the
per-token operators carry the count as a leading dimension. So a graph traced at
one size can be rewritten to any other, and the result priced exactly rather than
interpolated.

Rewritten, deliberately, is only the leading dimension that *equals* the traced
token count. A dimension that merely happens to be the same number is rewritten
too, which is why the tool prints what it changed: the check is that the
synthesised graph prices the same as a traced one at the same size, and
`--verify` against a real trace is how that was established.

Operators that read the forward context are left alone. Attention's cost depends
on `cu_seqlens_q` -- quadratically, since it is quadratic within a sequence --
and not on a token total, so rewriting its shapes would produce a number that
means nothing. It has to be traced or fitted, which is a separate job.

    python scripts/compass/synth_shapes.py traced.prefill.json \\
        --tokens 4376,4588,8192,16384 --out synth.json
"""

import argparse
import copy
import json
import sys


def _traced_tokens(graph: dict) -> int:
    shape = (graph.get("provenance") or {}).get("shape") or {}
    if shape.get("num_prefill_tokens"):
        return int(shape["num_prefill_tokens"])
    return int(sum(shape.get("num_scheduled_tokens") or [0]))


def _context_dependent(name: str) -> bool:
    try:
        from atom.compass.runtime import forward_ctx

        return forward_ctx.is_context_dependent(name)
    except Exception:  # noqa: BLE001 - the tool is useful without the engine
        return "attention" in name


def rewrite(graph: dict, tokens: int, target: int) -> tuple[list, int]:
    """Every operator, with leading dimensions of `tokens` changed to `target`."""
    ops, touched = [], 0
    for op in graph["ops"]:
        if _context_dependent(op["name"]):
            continue
        clone = copy.deepcopy(op)
        changed = False
        for group in ("input_shapes", "output_shapes"):
            shapes = []
            for shape in clone.get(group) or []:
                dims = list(shape)
                if dims and int(dims[0]) == tokens:
                    dims[0] = target
                    changed = True
                shapes.append(dims)
            clone[group] = shapes
        # Recorded contents describe the traced size and are meaningless at
        # another one; the signature keys on them, so they must go too.
        clone["int_values"] = []
        if changed:
            touched += 1
        ops.append(clone)
    return ops, touched


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph")
    p.add_argument("--tokens", required=True,
                   help="comma-separated token counts to cover")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.graph, encoding="utf-8") as fh:
        graph = json.load(fh)
    tokens = _traced_tokens(graph)
    if not tokens:
        print("the graph records no shape, so there is nothing to rewrite from",
              file=sys.stderr)
        return 2

    targets = [int(t) for t in args.tokens.split(",") if t.strip()]
    ops, seen = [], set()
    for target in targets:
        rewritten, touched = rewrite(graph, tokens, target)
        kept = 0
        for op in rewritten:
            key = json.dumps([op["name"], op["input_shapes"], op["dtypes"],
                              op.get("scalars")], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            ops.append(op)
            kept += 1
        print(f"  {tokens} -> {target}: {touched} operators rescaled, "
              f"{kept} new signatures")

    out = dict(graph)
    out["ops"] = ops
    out["provenance"] = dict(graph.get("provenance") or {},
                             synthesised_from=args.graph,
                             synthesised_tokens=targets)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"wrote {len(ops)} operators covering {len(targets)} sizes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
