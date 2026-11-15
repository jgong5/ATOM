# SPDX-License-Identifier: MIT
"""A block drafter's passes under DP attention, fitted and priced per rank.

DeepSeek-V4.1-Flash with DSpark5 at DP 2: each rank's step is its target,
then on a decode its draft-KV pass and its block draft, on an output prefill
its draft-KV pass and its propose pass. The group's step is the max over its
ranks, and an idle rank runs a dummy forward that drafts but writes no draft
context. Each rank's record is written from known per-rank laws, every line of
a step labelled with the group's step, each pass the max over the ranks that
ran it, as a real DP group's ranks each measure it; rank 1 runs the dummy
forwards. The fitted law is read by `StepLinearLaw` and priced by
`StepLinearBackend.price`.
"""

import dataclasses
import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _record, traced

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear_fit import fit, label_records, mark_held_out
from atom.compass.backends.work import activated_experts
from atom.compass.validate.cost import cost_accuracy

DP, RUNG, K = 2, 4, 5
ROUTING = (64, 4)
#: Each rank's target decode at rung 4, and its block draft's, both over the
#: experts the group's real tokens reach.
TARGET = {"step": 8e-3, "context": 2e-9, "graph_padding": 1e-10, "moe_experts": 6e-5}
DRAFT = {"step": 7e-4, "context": 3e-10, "graph_padding": 0.0, "moe_experts": 2e-5}
#: Each rank's draft-KV pass on a decode, and its propose pass on an output
#: prefill, over `[1, seqs]`.
DRAFT_KV = {"step": 5e-5, "seqs": 2.4e-6}
PROPOSE = {"step": 7e-4, "seqs": 7e-5}
#: Each rank's target prefill, and its draft-KV pass on an output prefill.
PREFILL = {
    "step": 0.02,
    "tokens": 6e-5,
    "query_square": 2e-10,
    "query_cached": 3e-10,
    "moe_rows": 2e-5,
}
DRAFT_PREFILL = {"step": 4e-5, "tokens": 2e-8, "seqs": 2.5e-5}
FULL, HALF, SPARSE = (3, 5), (2, 4), (1, 3)
OPTIONS = {"form": "base", "dp": DP, "moe_routing": ROUTING}


def _decode(view: BatchView, law: dict) -> float:
    reached = view.moe_tokens - max(view.group_dummies - 1, 0)
    return (
        law["step"]
        + law["context"] * sum(r.context_tokens for r in view.decode)
        + law["graph_padding"] * view.graph_padding
        + law["moe_experts"] * activated_experts(*ROUTING, reached)
    )


def _prefill(view: BatchView, law: dict) -> float:
    q = np.array([r.query_tokens for r in view.prefill])
    n = np.array([r.context_tokens for r in view.prefill])
    x = {
        "step": 1,
        "tokens": q.sum(),
        "query_square": (q * q).sum(),
        "query_cached": (q * (n - q)).sum(),
        "moe_rows": view.moe_rows,
        "seqs": len(q),
    }
    return float(sum(c * x[f] for f, c in law.items()))


def _seqs(view: BatchView, law: dict) -> float:
    return law["step"] + law["seqs"] * len(view.requests)


def _groups(seed, n, others):
    """`n` steps over `DP` ranks, each its ranks' views and its labels, every
    pass's the max over the ranks that ran it. Every third step prefills on
    both ranks, each request finishing its prompt; the others decode at rung 4,
    rank `i % DP` the slowest with up to 4 rows of 6 queries, the other
    `others`' rows but no more, but every fifth, where rank 1 runs a dummy
    forward of one row. A step's slowest rank runs the most requests."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        slow = i % DP
        if i % 3 == 0:
            ranks, most = [], int(rng.integers(1, 4))
            for r in range(DP):
                lo, hi = (4096, 8192) if r == slow else (16, 2048)
                widths = most if r == slow else int(rng.integers(1, most + 1))
                rows = []
                for _ in range(widths):
                    q = int(rng.integers(lo, hi))
                    rows.append(RequestShape(q, q + int(rng.integers(0, 30000)), False))
                ranks.append(tuple(rows))
            g = sum(r.query_tokens for rows in ranks for r in rows)
            views = [
                BatchView(rows, moe_rows=g, moe_tokens=g, produces_output=True)
                for rows in ranks
            ]
            parts = [
                {
                    "target": _prefill(v, PREFILL),
                    "kv": _prefill(v, DRAFT_PREFILL),
                    "draft": _seqs(v, PROPOSE),
                }
                for v in views
            ]
        else:
            idle = i % 5 == 1
            slow = 0 if idle else slow
            contexts, most = [], int(rng.integers(1, RUNG + 1))
            for r in range(DP):
                if idle and r == 1:
                    contexts.append([K + 1])
                    continue
                low, high = others
                widths = most if r == slow else min(int(rng.integers(low, high)), most)
                span = (150_000, 200_000) if r == slow else (1, 40_000)
                contexts.append([int(rng.integers(*span)) for _ in range(widths)])
            tokens = (K + 1) * sum(map(len, contexts))
            views = [
                BatchView(
                    tuple(RequestShape(K + 1, c, True) for c in mine),
                    capture_rung=RUNG,
                    moe_rows=DP * RUNG * (K + 1),
                    moe_tokens=tokens,
                    produces_output=True,
                    dp_dummy=idle and r == 1,
                    group_dummies=int(idle),
                )
                for r, mine in enumerate(contexts)
            ]
            parts = [
                {"target": _decode(v, TARGET), "draft": _decode(v, DRAFT)}
                | ({} if v.dp_dummy else {"kv": _seqs(v, DRAFT_KV)})
                for v in views
            ]
        labels = {
            k: max(p[k] for p in parts if k in p) for k in ("target", "kv", "draft")
        }
        assert all(parts[slow][k] == label for k, label in labels.items()), i
        out.append(
            ([dataclasses.replace(v, dp_rank=r) for r, v in enumerate(views)], labels)
        )
    return out


def _records(path, groups):
    """Rank `r`'s record of `groups`, its labels stretched by `1 + 0.01 r`: on
    each step its target to `forward_done_event` less the draft-KV pass, that
    pass, which a dummy forward does not run, and the rest of its device time
    its draft; on a prefill the draft is its propose pass too."""
    path.mkdir()
    paths = []
    for r in range(DP):
        parts = [
            {
                k: round(s * (1 + 0.01 * r) * 1e9)
                for k, s in labels.items()
                if k != "kv" or not views[r].dp_dummy
            }
            for views, labels in groups
        ]
        steps = [
            (
                [(x.query_tokens, x.context_tokens, x.decode) for x in v[r].requests],
                v[r].capture_rung,
                sum(p.values()) * 1e-9,
            )
            for (v, _), p in zip(groups, parts)
        ]
        record = _record(path / f"dp{r}.jsonl", steps)
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        for line, (views, _), p in zip(lines, groups, parts):
            v = views[r]
            line.update(
                dp_rank=r,
                moe_rows=v.moe_rows,
                moe_tokens=v.moe_tokens,
                d_busy_ns=sum(p.values()),
                d_target_ns=p["target"],
            )
            if "kv" in p:
                line["d_draft_kv_ns"] = p["kv"]
            if v.prefill:
                line["d_propose_ns"] = p["draft"]
            if v.dp_dummy:
                line.update(is_dummy_run=True, batch=[])
        record.write_text("".join(json.dumps(traced(x)) + "\n" for x in lines))
        paths.append(record)
    return paths


def _cells(path):
    """Three runs, whose other rank decodes up to 3 or 4 rows, 2 or 3, and 1
    or 2: a law fitted on any two picks a decode rung's MoE column."""
    cells = {"full": FULL, "half": HALF, "sparse": SPARSE}
    return [
        record
        for seed, (name, others) in enumerate(cells.items())
        for record in _records(path / name, _groups(seed, 300, others))
    ]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """The law fitted on `_cells`, held out by run, and its samples."""
    root = tmp_path_factory.mktemp("cells")
    samples = label_records(_cells(root), DP)[0]
    law, report = fit(samples, OPTIONS)
    mark_held_out(law, report, samples, OPTIONS)
    return law, report, samples, root


def _priced(law, views):
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    return [backend.price(v) for v in views]


def test_each_draft_vector_is_fitted_as_a_dp_vector_on_each_rank_s_rows(fitted):
    """The draft law's decode rung adopts the experts its block draft reaches,
    as the target's does, and `propose` and `draft_kv` recover each rank's
    law from labels each the group's: the max over its ranks."""
    law, _, _, _ = fitted
    draft = law["draft"]
    assert draft["moe_routing"] == {"experts": ROUTING[0], "top_k": ROUTING[1]}
    for vector, expected in (
        (law["decode"]["4"], TARGET),
        (draft["decode"]["4"], DRAFT),
    ):
        assert vector["fit"]["moe_column"]["adopted"] == "moe_experts"
        assert vector["coefficients"] == pytest.approx(expected, rel=1e-5, abs=1e-12)
    for name, expected in (("propose", PROPOSE), ("draft_kv", DRAFT_KV)):
        assert draft[name]["coefficients"] == pytest.approx(expected, rel=1e-5)
        assert draft[name]["fit"]["ranks"] == 0
        assert draft[name]["fit"]["passes"]["stop"] == "settled"
    # Every per-rank row: each rank's label is the group's, so the slower
    # rank's coefficients do not fit the faster rank's rows.
    assert draft["draft_kv"]["fit"]["all_rows"]["mad"] > 1e-3


def test_every_draft_pass_of_a_dp_step_is_priced_at_its_own_rank_s_rows(fitted):
    """Each step's ranks priced apart: the max over them is the group's target
    with its draft-KV pass, and its draft, on every step of a held-out run."""
    law, _, _, _ = fitted
    for views, labels in _groups(2, 120, FULL):
        costs = _priced(law, views)
        target = labels["target"] + labels["kv"]
        assert max(c.target_seconds for c in costs) == pytest.approx(target, rel=1e-5)
        assert max(c.draft_seconds for c in costs) == pytest.approx(
            labels["draft"], rel=1e-5
        )


def test_a_dummy_forward_s_draft_prices_no_draft_kv_pass(fitted):
    """An idle rank's dummy forward drafts as its peers do and writes no draft
    context: on a decode its draft is the rung's draft vector and nothing more,
    on a prefill its propose pass."""
    law, _, _, _ = fitted
    decode = BatchView(
        (RequestShape(K + 1, K + 1, True),),
        capture_rung=RUNG,
        moe_rows=DP * RUNG * (K + 1),
        moe_tokens=5 * (K + 1),
        produces_output=True,
        group_dummies=1,
    )
    prefill = BatchView(
        (RequestShape(256, 256, False),),
        moe_rows=4096,
        moe_tokens=4096,
        produces_output=True,
        group_dummies=1,
    )
    for view in (decode, prefill):
        ran, idle = _priced(law, [view, dataclasses.replace(view, dp_dummy=True)])
        kv = [t.name for t in ran.terms if t.name.startswith("draft.")]
        assert kv and not [t for t in idle.terms if t.name.startswith("draft.")]
        assert idle.draft_seconds == ran.draft_seconds > 0
        drafted = sum(t.seconds for t in ran.terms if t.name in kv)
        assert idle.target_seconds == pytest.approx(ran.target_seconds - drafted)


def test_the_draft_law_s_vectors_are_held_out_by_run(fitted):
    law, report, _, root = fitted
    draft = law["draft"]
    held = {"prefill": draft["prefill"], "decode 4": draft["decode"]["4"]}
    held |= {name: draft[name] for name in ("propose", "draft_kv")}
    for name, vector in held.items():
        bands = vector["held_out"]["bands"]
        assert bands, name
        for band in bands.values():
            assert band["p10"] == pytest.approx(0.0, abs=1e-5), name
            assert band["p90"] == pytest.approx(0.0, abs=1e-5), name
    for run in report["held_out"].values():
        assert {"draft propose", "draft draft_kv", "draft decode rung 4"} <= set(run)

    # A held-out run is priced within the bands: its draft graded validated.
    _records(root / "held", _groups(4, 300, FULL))
    graded = cost_accuracy(root / "held", law)["validated"]
    for kind in ("prefill", "decode", "draft"):
        assert graded[kind]["on_totals"] == pytest.approx(0.0, abs=1e-5), kind
