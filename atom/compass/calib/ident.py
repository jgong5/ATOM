# SPDX-License-Identifier: MIT
"""Identifiability of a design's vectors, over its GPU-bound steps (#739 2.2).

Per vector (each decode rung, and prefill), the design matrix holds one row a
GPU-bound step: the costliest rank's value of each column (`columns.values`),
the row the step linear fit solves on. A step a host launch paced or exposed
is no design point: a fit cuts it (rule 2.4.2). For each:

- rank: the column-scaled matrix's rank, against its columns (I1);
- VIF: each column's variance inflation, 1 / (1 - R^2) of it on the others
  and the intercept (I2, at most `vif`), over the design's steps alike; the
  same weighted as the fit weighs them (the inverse squared price, for
  relative error) is reported beside it, `vif_weighted`, and enters G through
  the fit's information. With one DP rank, `graph_padding` and
  `row_padding` are collinear inside a rung by construction (rule 2.4.6;
  identical at a rung of one row count): their VIF, and a rank one short for
  them, are reported as failed and the vector is judged by G;
- G: the worst relative prediction error over the vector's reachable range
  (I3-I4): `g_z` standard errors of the price a law fitted on these steps
  gives at a vertex of the range (`vertices`), relative to the law's price
  there, under relative noise `noise[kind]` a step, independent between
  steps: `z sigma sqrt(v' (X' W X)^-1 v) / (v' beta)`, `W` the inverse
  squared price of each step (the fit minimises relative error); on a design
  short of full rank, with the pseudo-inverse where every vertex lies in the
  design's row space (its price estimable), else infinite. A prefill
  vertex whose device time its host launch would pace is left out (its price
  is the host's), and so counted apart;
- levels: the distinct values each column takes, at two significant digits
  (I5, at least `levels`);
- given a target workload's steps (a real or simulated record), the same
  relative error at each of its GPU-bound steps of the vector, reported, not
  gated: its mean (the target-weighted I criterion), 95th percentile and worst
  (infinite for a step moving a column the design holds constant).

All of it is variance under the stated form: a design passes G with any form
the columns fit, so form bias is what the record's residuals per segment
(`check.residuals`) and held-out grading show, not this.

The reachable range comes from the deployment alone (rule 2.1.2): a decode
rung's widest-rank rows (and, above one rank, a priced rank below the rung
that a peer holds), each row's context from `min_context` to the KV pool or
the model length, one long anchor among short rows, and the peers idle or
full; a prefill's tokens from `min_prompt` to the largest token-graph bucket
and the step budget, one row or many, no history or the longest the pool
holds, the peers idle or at the full budget, the prefill duty 0 or 1, the
step producing output or not (a middle chunk, producing none, is one row
filling the step budget).
"""

from __future__ import annotations

import math

import numpy as np

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import PACED_S
from atom.compass.calib import columns as cols
from atom.compass.calib.deploy import Config
from atom.compass.calib.engine import Law

#: The two decode columns one DP rank makes collinear inside a rung (rule 2.4.6).
RULE6 = ("graph_padding", "row_padding")


def variants(cfg: Config, kind: str) -> list[tuple[str, list[str]]]:
    """The column sets a `kind` vector may be fitted at, each with its label:
    its columns (`base`), plus each of `<kind>_variants`' additions in turn
    (the candidates the form-selection rule chooses among, #739 2.3; a decode
    rung's MoE column among them)."""
    base = list(cfg.columns[kind])
    return [
        ("+".join(add) or "base", base + [c for c in add if c not in base])
        for add in cfg.columns.get(f"{kind}_variants", [[]])
    ]


def decode_vertices(cfg: Config, rung: int) -> list[BatchView]:
    dep, lo_ctx = cfg.deployment, cfg.range["min_context"]
    q, dp = dep.query, dep.dp
    lo = dep.rows_of(rung).start
    out = []
    for n in sorted({lo, rung} | ({1} if dp > 1 else set())):
        long = min(dep.max_model_len, dep.kv_tokens // n)
        anchor = min(dep.max_model_len, dep.kv_tokens - (n - 1) * lo_ctx)
        profiles = {
            "short": [lo_ctx] * n,
            "long": [long] * n,
            "anchor": [anchor] + [lo_ctx] * (n - 1),
        }
        for ctxs in profiles.values():
            for full in (False, True) if dp > 1 else (False,):
                if n < lo and not full:
                    continue  # a rank below the rung needs a peer holding it
                peers = (dp - 1) * rung if full else 0
                dummies = 0 if full else dp - 1
                rows = tuple(RequestShape(q, max(c, q), True) for c in ctxs)
                out.append(
                    BatchView(
                        rows,
                        capture_rung=rung,
                        moe_rows=rung * dp * q if dp > 1 else None,
                        moe_tokens=(n + peers) * q if dp > 1 else None,
                        produces_output=True,
                        dp_rank=0 if dp > 1 else None,
                        group_dummies=dummies,
                        prefill_duty=0.0,
                        kda=dep.kda(rows),
                    )
                )
    return out


def prefill_vertices(cfg: Config) -> list[BatchView]:
    dep, rng = cfg.deployment, cfg.range
    budget, dp = dep.max_num_batched_tokens, dep.dp
    sizes = {rng["min_prompt"], budget}
    if dep.prefill_buckets:
        sizes.add(max(dep.prefill_buckets))
    out = []
    for t in sorted(sizes):
        for seqs in sorted(
            {1, max(1, min(dep.max_num_seqs, t // rng.get("min_seq_tokens", 128)))}
        ):
            q = t // seqs
            deep = max(0, min(dep.max_model_len, dep.kv_tokens // seqs) - q)
            for history in (0, deep):
                rows = tuple(RequestShape(q, q + history, False) for _ in range(seqs))
                own = q * seqs
                for full in (False, True) if dp > 1 else (False,):
                    peers = (dp - 1) * budget if full else 0
                    dummies = 0 if full else dp - 1
                    for duty in (0.0, 1.0):
                        # a step producing no output is a middle chunk, which
                        # the scheduler fills to the step budget, and only
                        # with one row: first come first served gives a
                        # second row only the budget the first leaves, which
                        # it fills by finishing
                        middle = t == budget and seqs == 1
                        for output in (False, True) if middle else (True,):
                            out.append(
                                BatchView(
                                    rows,
                                    moe_rows=own + peers if dp > 1 else None,
                                    moe_tokens=own + peers if dp > 1 else None,
                                    produces_output=output,
                                    dp_rank=0 if dp > 1 else None,
                                    group_dummies=dummies,
                                    group_prefills=dp > 1,
                                    group_max_tokens=(
                                        max(own, budget if full else 0)
                                        if dp > 1
                                        else None
                                    ),
                                    group_tail=output and dp > 1,
                                    prefill_duty=duty,
                                    kda=dep.kda(rows),
                                )
                            )
    return out


def _levels(column: np.ndarray) -> int:
    def two(x):
        if x == 0 or not math.isfinite(x):
            return x
        return round(x, 1 - math.floor(math.log10(abs(x))))

    return len({two(float(x)) for x in column})


def vif(X: np.ndarray, w: np.ndarray | None = None) -> list[float]:
    """Each non-intercept column's VIF (column 0 is the intercept), under row
    weights `w` (the fit's, for relative error: the inverse squared price):
    1 / (1 - R^2) of the column's weighted regression on the others; inf for a
    constant column or one the others hold exactly."""
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=float) / np.mean(w)
    root = np.sqrt(w)
    out = []
    for j in range(1, X.shape[1]):
        y = X[:, j]
        if np.ptp(y) == 0:
            out.append(math.inf)
            continue
        others = np.delete(X, j, axis=1)
        coef, *_ = np.linalg.lstsq(others * root[:, None], y * root, rcond=None)
        resid = y - others @ coef
        mean = float((w * y).sum() / w.sum())
        ss = float((w * (y - mean) ** 2).sum())
        r2 = 1.0 - float((w * resid**2).sum()) / ss
        out.append(math.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
    return out


def assess(names, X, y, V, vy, sigma: float, z: float, T=None, ty=None) -> dict:
    """Rank, VIF, G and levels of design `X` (rows priced `y`) over vertices
    `V` priced `vy`; see the module docstring. Given target steps `T` priced
    `ty`, also the same relative error at each of them (`target`: their mean,
    the target-weighted I criterion, its 95th percentile and its worst)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, p = X.shape if X.size else (0, len(names))
    out = {"columns": list(names), "steps": n}
    if n == 0:
        return out | {"rank": 0, "vif": {}, "g": math.inf, "levels": {}}
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    Xw = Xs / y[:, None]
    rank = int(np.linalg.matrix_rank(Xw))
    out["rank"] = rank
    out["vif"] = dict(zip(names[1:], vif(Xs)))
    out["vif_weighted"] = dict(zip(names[1:], vif(Xs, 1.0 / y**2)))
    out["levels"] = {c: _levels(X[:, j]) for j, c in enumerate(names) if j}
    if not len(V):
        return out | {"g": math.inf, "g_vertex": None}
    Vs = np.asarray(V, dtype=float) / scale
    inv = np.linalg.pinv(Xw.T @ Xw)
    if rank < p:
        # a vertex's price is estimable when it lies in the design's row space
        # (two columns the range holds equal, as one DP rank's paddings at a
        # rung of one row count, are fitted as their sum)
        off = Vs - Vs @ (np.linalg.pinv(Xw) @ Xw)
        lost = np.linalg.norm(off, axis=1) > 1e-6 * np.maximum(
            np.linalg.norm(Vs, axis=1), 1e-300
        )
        out["not_estimable"] = int(lost.sum())
        if lost.any():
            return out | {"g": math.inf, "g_vertex": None}
    if T is not None and len(T):
        Ts = np.asarray(T, dtype=float) / scale
        st = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Ts, inv, Ts), 0.0))
        st = z * sigma * st / np.asarray(ty, dtype=float)
        if rank < p:
            off = Ts - Ts @ (np.linalg.pinv(Xw) @ Xw)
            far = np.linalg.norm(off, axis=1) > 1e-6 * np.linalg.norm(Ts, axis=1)
            st[far] = math.inf
        out["target"] = {
            "steps": len(st),
            "i": float(np.mean(st)),
            "p95": float(np.percentile(st, 95)),
            "max": float(np.max(st)),
        }
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Vs, inv, Vs), 0.0)) / np.asarray(
        vy, dtype=float
    )
    worst = int(np.argmax(se))
    out["g"] = float(z * sigma * se[worst])
    out["g_vertex"] = {c: float(v) for c, v in zip(names, V[worst])}
    return out


def _verdict(cfg: Config, kind: str, got: dict) -> dict:
    acc = cfg.acceptance
    p = len(got["columns"])
    waived = (
        [c for c in RULE6 if c in got["columns"]]
        if cfg.deployment.dp == 1 and all(c in got["columns"] for c in RULE6)
        else []
    )
    checked = {c: v for c, v in got.get("vif", {}).items() if c not in waived}
    fails = []
    if got["rank"] < p and not (
        waived and got["rank"] == p - 1 and got["g"] < math.inf
    ):
        fails.append(f"rank {got['rank']} < {p}")
    if any(v > acc["vif"] for v in checked.values()):
        fails.append(
            "VIF "
            + ", ".join(f"{c} {v:.3g}" for c, v in checked.items() if v > acc["vif"])
        )
    if not got["g"] <= acc["g"][kind]:
        fails.append(f"G {got['g']:.3%} > {acc['g'][kind]:.0%}")
    if kind == "decode" and got["steps"] < acc["rung_steps"]:
        fails.append(f"{got['steps']} GPU-bound steps < {acc['rung_steps']}")
    few = [
        c
        for c, k in got.get("levels", {}).items()
        if k < acc["levels"] and c not in got.get("indicator", ())
    ]
    if few:
        fails.append(f"levels < {acc['levels']}: {', '.join(few)}")
    rule6 = None
    if waived:
        bad = {c: got["vif"].get(c, math.inf) for c in waived}
        rule6 = (
            "aliased (one DP rank, rule 2.4.6; estimability checked, judged by G): "
            + ", ".join(f"{c} VIF {v:.3g}" for c, v in bad.items())
        )
        if got["rank"] < p:
            rule6 += f"; rank {got['rank']} < {p}"
    return {"pass": not fails, "fails": fails, "rule6": rule6}


def vectors(cfg: Config, steps, law: Law, target=()) -> list[dict]:
    """Every vector's assessment on `steps` (`engine.Step`s, or the same shape
    from a record): per variant of its columns, with its verdict; given
    `target`, the steps of a target workload (a real or simulated record's,
    `check.record_steps`), each variant's error at its GPU-bound ones
    (`assess`'s `target`), reported beside the verdict, not gating it."""
    dep, acc = cfg.deployment, cfg.acceptance
    z = acc["g_z"]
    out = []
    groups = {("prefill", None): []}
    for r in dep.rungs:
        groups[("decode", r)] = []
    counted = {k: [0, 0] for k in groups}
    aimed = {k: [] for k in groups}
    for s in target:
        key = (s.kind, s.rung if s.kind == "decode" else None)
        if key in aimed and s.gpu_bound:
            aimed[key].append(s)
    for s in steps:
        key = (s.kind, s.rung if s.kind == "decode" else None)
        if key not in groups:
            continue
        counted[key][0] += 1
        if s.gpu_bound:
            counted[key][1] += 1
            groups[key].append(s)
    for (kind, rung), got in groups.items():
        vertices = (
            prefill_vertices(cfg) if kind == "prefill" else decode_vertices(cfg, rung)
        )
        priced, paced = [], 0
        for v in vertices:
            target, draft = law.price(v)
            if target <= 0:
                continue
            if kind == "prefill":
                host = "prefill" if v.produces_output else "middle"
                tokens = sum(r.query_tokens for r in v.requests)
                launch = law.host_s(host, tokens, len(v.requests), ("W_launch",))
                if target + draft <= launch + PACED_S:
                    paced += 1
                    continue
            priced.append((v, target))
        results = []
        for label, names in variants(cfg, kind):
            try:
                spread = cfg.columns.get("min_spread", 0.0)
                knot = cfg.columns.get("ctx_knot")
                X = np.array(
                    [cols.values(s.view, kind, names, dep, spread, knot) for s in got]
                )
                V = np.array(
                    [cols.values(v, kind, names, dep, spread, knot) for v, _ in priced]
                )
                aim = aimed[(kind, rung)]
                T = np.array(
                    [cols.values(s.view, kind, names, dep, spread, knot) for s in aim]
                )
                X, V, T = (a.reshape(-1, len(names)) for a in (X, V, T))
            except ValueError as e:
                results.append({"variant": label, "columns": names, "error": str(e)})
                continue
            # a column the whole reachable range holds at one value (and the
            # design with it) is no column at this vector: the fit names it and
            # gives it zero (`step_linear_fit`); one the range holds at 0 and 1
            # alone is an indicator, which takes no third level. A target step
            # that moves such a column lies off the design: its error is infinite
            both = np.vstack([V, X]) if len(X) else V
            keep, structural, indicator = [0], [], []
            for j in range(1, len(names)):
                if len(both) and np.ptp(both[:, j]) == 0:
                    structural.append(names[j])
                    continue
                keep.append(j)
                if len(V) and set(np.unique(V[:, j])) <= {0.0, 1.0}:
                    indicator.append(names[j])
            used = [names[j] for j in keep]
            gone = [j for j in range(1, len(names)) if j not in keep]
            if len(T) and gone and len(both):
                off = np.any(T[:, gone] != both[0, gone], axis=1)
            else:
                off = np.zeros(len(T), dtype=bool)
            y = [s.ranks[s.priced][2] for s in got]
            res = assess(
                used,
                X[:, keep],
                y,
                V[:, keep],
                [p for _, p in priced],
                cfg.noise[kind],
                z,
                T[:, keep] if len(T) else None,
                [s.ranks[s.priced][2] for s in aim],
            )
            if "target" in res and off.any():
                res["target"]["off_design"] = int(off.sum())
                res["target"]["max"] = math.inf
            res["variant"] = label
            res["structural"] = structural
            res["indicator"] = indicator
            res["provisional"] = cols.provisional(kind, used)
            res |= _verdict(cfg, kind, res)
            results.append(res)
        out.append(
            {
                "vector": "prefill" if kind == "prefill" else f"decode rung {rung}",
                "kind": kind,
                "rung": rung,
                "steps": counted[(kind, rung)][0],
                "gpu_bound": counted[(kind, rung)][1],
                "vertices": len(priced),
                "vertices_paced": paced,
                "variants": results,
                "pass": any(r.get("pass") for r in results),
            }
        )
    return out
