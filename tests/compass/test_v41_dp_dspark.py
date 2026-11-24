# SPDX-License-Identifier: MIT
"""A block drafter's passes under DP attention, fitted and priced per rank.

DeepSeek-V4.1-Flash with DSpark5 at DP 2: each rank's step is its target,
then on a decode its draft-KV pass and its block draft, on an output prefill
its draft-KV pass and its propose pass. The group's step is the max over its
ranks, and an idle rank runs a dummy forward that drafts but writes no draft
context. Each rank's record is written from known per-rank laws: a pass that
waits on its group labelled with the group's, the max over the ranks that ran
it, as a real DP group's ranks each measure it, and a decode's draft-KV pass,
which writes the rank's own draft context, with the rank's own; rank 1 runs
the dummy forwards. The fitted law is read by `StepLinearLaw` and priced by
`StepLinearBackend.price`.
"""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import _engine_file, _record, traced

from atom.compass.artifacts import machine_platform
from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
    json_file,
    step_linear_fit,
)
from atom.compass.backends.step_linear_fit import (
    fit,
    label_records,
    main,
    mark_held_out,
)
from atom.compass.backends.work import activated_experts
from atom.compass.fit import steps
from atom.compass.spec import MachineSpec
from atom.compass.validate.cost import cost_accuracy

TREE = Path(__file__).resolve().parents[2]
#: `deepseek-ai/DeepSeek-V4.1-Flash`'s config.json: the target routes each row
#: to 6 of 384 experts, its DSpark drafter to 3 of its own 128.
CONFIG = TREE / "tests/models/deepseek_v41/fixtures/config.json"
DP, RUNG, K = 2, 4, 5
ROUTING, DRAFT_ROUTING = (384, 6), (128, 3)
#: Each rank's target decode at rung 4, and its block draft's, each over the
#: experts of its own routing the group's real tokens reach.
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
OPTIONS = {
    "form": "base",
    "dp": DP,
    "moe_routing": ROUTING,
    "draft_moe_routing": DRAFT_ROUTING,
}


def _decode(view: BatchView, law: dict, routing=ROUTING) -> float:
    reached = view.moe_tokens - max(view.group_dummies - 1, 0)
    return (
        law["step"]
        + law["context"] * sum(r.context_tokens for r in view.decode)
        + law["graph_padding"] * view.graph_padding
        + law["moe_experts"] * activated_experts(*routing, reached)
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
    """`n` steps over `DP` ranks, each its ranks' views and its labels: the
    target's and the draft's the max over the ranks, `kv` each rank's own
    draft-KV pass, None on a dummy forward. Every third step prefills on
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
                {
                    "target": _decode(v, TARGET),
                    "draft": _decode(v, DRAFT, DRAFT_ROUTING),
                }
                | ({} if v.dp_dummy else {"kv": _seqs(v, DRAFT_KV)})
                for v in views
            ]
        labels = {k: max(p[k] for p in parts) for k in ("target", "draft")}
        assert all(parts[slow][k] == label for k, label in labels.items()), i
        labels["kv"] = [p.get("kv") for p in parts]
        out.append(
            ([dataclasses.replace(v, dp_rank=r) for r, v in enumerate(views)], labels)
        )
    return out


def _records(path, groups, trace=True):
    """Rank `r`'s record of `groups`: on each step its target to
    `forward_done_event` less the draft-KV pass, that pass, which a dummy
    forward does not run, and the rest of its device time its draft; on a
    prefill the draft is its propose pass too. Each label its group's is
    stretched by `1 + 0.01 r`, as a rank measures a pass its group waits on.
    A decode's draft-KV pass is the rank's own; an output prefill's is the
    group's, as the draft law's prefill vector is fitted over the group.
    Unless not `trace`, each prefill also carries a trace run's busy time
    (`traced`)."""
    path.mkdir()
    paths = []
    for r in range(DP):
        parts = []
        for views, labels in groups:
            part = {k: labels[k] * (1 + 0.01 * r) for k in ("target", "draft")}
            if views[r].prefill:
                part["kv"] = max(labels["kv"]) * (1 + 0.01 * r)
            elif labels["kv"][r] is not None:
                part["kv"] = labels["kv"][r]
            parts.append({k: round(s * 1e9) for k, s in part.items()})
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
        # A plain record holds no trace run's busy time, `_record`'s included.
        label = (
            traced
            if trace
            else (lambda x: {k: v for k, v in x.items() if k[:2] != "b_"})
        )
        record.write_text("".join(json.dumps(label(x)) + "\n" for x in lines))
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


def test_each_draft_vector_is_fitted_on_the_labels_its_pass_is_measured_by(fitted):
    """The draft law's decode rung adopts the experts its block draft reaches,
    over the drafter's own routing, as the target's does over its; `propose`
    recovers each rank's law from labels each the group's, the max over its
    ranks, and `draft_kv` from each rank's own label, as at width 1."""
    law, _, samples, _ = fitted
    draft = law["draft"]
    assert draft["moe_routing"] == {"experts": 128, "top_k": 3}
    for vector, expected in (
        (law["decode"]["4"], TARGET),
        (draft["decode"]["4"], DRAFT),
    ):
        assert vector["fit"]["moe_column"]["adopted"] == "moe_experts"
        assert vector["coefficients"] == pytest.approx(expected, rel=1e-5, abs=1e-12)
    for name, expected in (("propose", PROPOSE), ("draft_kv", DRAFT_KV)):
        assert draft[name]["coefficients"] == pytest.approx(expected, rel=1e-5)
    assert draft["propose"]["fit"]["ranks"] == 0
    assert draft["propose"]["fit"]["passes"]["stop"] == "settled"
    # Every per-rank row: each rank's label is the group's, so the slower
    # rank's coefficients do not fit the faster rank's rows.
    assert draft["propose"]["fit"]["all_rows"]["mad"] > 1e-3
    # Every decode rank that ran the pass is a sample of its own, fitted to
    # within the labels' rounding to the nanosecond.
    ran = [s for s in samples if s.draft_kv_s is not None and not s.view.prefill]
    assert {s.rank for s in ran} == set(range(DP))
    assert draft["draft_kv"]["fit"]["samples"] == len(ran)
    assert draft["draft_kv"]["fit"]["mad"] < 1e-4


def test_every_draft_pass_of_a_dp_step_is_priced_at_its_own_rank_s_rows(fitted):
    """Each step's ranks priced apart, on every step of a held-out run: each
    rank's draft-KV pass is its own, and the max over the ranks is the group's
    target without it, and its draft."""
    law, _, _, _ = fitted
    for views, labels in _groups(2, 120, FULL):
        costs = _priced(law, views)
        kv = [
            sum(t.seconds for t in c.terms if t.name.startswith("draft."))
            for c in costs
        ]
        assert kv == pytest.approx([s or 0.0 for s in labels["kv"]], rel=1e-5)
        target = max(c.target_seconds - s for c, s in zip(costs, kv))
        assert target == pytest.approx(labels["target"], rel=1e-5)
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


def test_a_propose_step_a_rank_neither_labels_nor_idles_is_left_out(fitted):
    """A step one rank proposes on, where its peer gives no propose label and
    ran no dummy forward, has no group label the vector can price as the max
    over its ranks: it is left out and counted, and the rest fit as before."""
    _, _, samples, _ = fitted
    peer = next(s for s in samples if s.rank == 1 and s.propose_s is not None)
    planted = [
        dataclasses.replace(s, propose_s=None) if s is peer else s for s in samples
    ]
    propose = fit(planted, OPTIONS)[0]["draft"]["propose"]
    assert propose["fit"]["ranks"] == 1
    assert propose["coefficients"] == pytest.approx(PROPOSE, rel=1e-5)


def test_the_figures_command_folds_the_draft_law_over_the_drafter_s_routing(
    fitted, tmp_path, monkeypatch
):
    """`figures` on the law restates the figures the fit wrote: each fold's
    draft decode rung adopts the drafter's MoE column, over the routing the
    law's draft states."""
    law, _, _, root = fitted
    records = []
    for name in ("full", "half", "sparse"):
        (tmp_path / name).mkdir()
        for r in range(DP):
            record = tmp_path / name / f"dp{r}.jsonl"
            shutil.copyfile(root / name / record.name, record)
            _engine_file(
                record, {"data_parallel_size": DP, "enable_dp_attention": True}
            )
            records.append(record)
    file = tmp_path / "law.json"
    json_file.write_json(law, file)
    written = json.loads(file.read_text())
    file.write_text(json.dumps({**written, "figures": {}}))
    folds = []

    def kept(*args, **kwargs):
        for fold in fold_laws(*args, **kwargs):
            folds.append(fold[2])
            yield fold

    fold_laws = step_linear_fit._fold_laws
    monkeypatch.setattr(step_linear_fit, "_fold_laws", kept)
    main(["figures", *map(str, records), "--law", str(file)])

    assert len(folds) == 3
    for fold in folds:
        rung = fold["draft"]["decode"]["4"]
        assert rung["fit"]["moe_column"]["adopted"] == "moe_experts"
    assert json.loads(file.read_text()) == written
    for run in written["figures"].values():
        assert run["draft"]["on_totals"] == pytest.approx(0.0, abs=1e-5)


def test_fit_steps_takes_the_drafter_s_routing_from_the_served_model(fitted, tmp_path):
    """`compass fit steps --model` reads V4.1-Flash's target and DSpark routing
    off its config.json, and the draft law fitted at DP 2 states the
    drafter's."""
    _, _, samples, _ = fitted
    # `compass fit steps` labels each sample with the traces its client served.
    samples = [
        dataclasses.replace(s, provenance={**s.provenance, "traces": []})
        for s in samples
    ]
    (tmp_path / "config.json").write_bytes(CONFIG.read_bytes())
    digest = "sha256:" + hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    document = json.loads(
        (TREE / "tests/compass/machines/mi308x-hjbog39.json").read_text()
    )
    config = {
        "engine_config": {"data_parallel_size": DP},
        "model": digest,
        "platform": machine_platform(MachineSpec.from_mapping(document)),
    }
    args = SimpleNamespace(
        form="base",
        seed=0,
        moe_routing=None,
        model=str(tmp_path),
        cold_peer_rows=False,
        peer_share=None,
        keep_peer_tail=None,
        ladder_run=None,
    )
    law = steps._step_linear(samples, config, args)[1]
    assert law["moe_routing"] == {"experts": 384, "top_k": 6}
    assert law["draft"]["moe_routing"] == {"experts": 128, "top_k": 3}
    assert law["draft"]["decode"]["4"]["fit"]["moe_column"]["adopted"] == "moe_experts"


@pytest.mark.parametrize("label", ["record", None])
def test_one_plain_dp_record_fits_its_prefill_only_from_its_own_windows(
    tmp_path, monkeypatch, label
):
    """The budget of one real run: a single DP record with no trace run. Under
    `COMPASS_PREFILL_LABEL=record` each prefill step is labelled from its own
    device window, so the prefill vector and the draft's prefill passes fit
    and price the record's own prefills as their per-rank laws do; unset, every
    prefill is cut as untraced and the prefill vector stays unfitted."""
    if label:
        monkeypatch.setenv("COMPASS_PREFILL_LABEL", label)
    else:
        monkeypatch.delenv("COMPASS_PREFILL_LABEL", raising=False)
    groups = _groups(0, 300, FULL)
    samples = label_records(_records(tmp_path / "plain", groups, trace=False), DP)[0]

    law, _ = fit(samples, OPTIONS)

    if not label:
        assert "prefill" in law["unfitted"]
        return
    assert "prefill" not in law["unfitted"] and not law["draft"]["unfitted"]
    views = [v for v, _ in groups if any(x.prefill for x in v)]
    assert views
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    for v in views:
        rank = next(r for r, x in enumerate(v) if x.prefill and not x.dp_dummy)
        want = _prefill(v[rank], PREFILL) * (1 + 0.01 * rank)
        assert backend.price(v[rank]).seconds == pytest.approx(want, rel=0.1)
