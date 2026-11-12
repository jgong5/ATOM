# SPDX-License-Identifier: MIT
"""MiniMax-M3 with its EAGLE3 drafter: recorded, fitted, priced and run.

EAGLE3 drafts serially: its `propose` runs `K` draft forwards, the first over
the step's whole token stream, each later one a row a request. A real record
times each forward apart (`d_draft_passes_ns`), the step linear fit prices one
draft forward per pass at its rows (`forward`), the rest of the propose
(`propose`) and, on a middle chunk, the draft-KV pass (the draft law's
prefill vector), each held out by run.

The scenario run needs a driver and two model directories, each holding a
config and the target's a tokenizer: `ATOM_COMPASS_M3_MODEL`, MiniMax-M3, and
`ATOM_COMPASS_M3_EAGLE3`, its EAGLE3 drafter; no weight is read. The rest run
anywhere.
"""

import copy
import json
import os

import numpy as np
import pytest
import torch
from test_backend_step_linear_fit import _price, _record, _steps, traced
from test_parity_device_window import KV, PROPOSE, _recorded
from test_parity_record import _config, _drive

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_labels import DRAFT_PASSES, label
from atom.compass.backends.step_linear import serial_passes
from atom.compass.backends.step_linear_fit import fit, label_records, mark_held_out
from atom.compass.parity import ENV, read

#: The draft forwards each step's propose runs, `--num-speculative-tokens`.
K = 3
#: The draft forward's price, per pass at its rows; the rest of the propose,
#: per request; the draft-KV pass of a middle chunk.
FORWARD = {"step": 4e-5, "tokens": 2e-7, "query_square": 2e-7, "query_cached": 5e-12}
REST = {"step": 1e-5, "seqs": 3e-6}
DRAFT_KV = {"step": 3e-4, "tokens": 4e-6, "query_square": 2e-11, "query_cached": 6e-12}


def _dot(coefficients: dict, x) -> float:
    return float(np.dot(list(coefficients.values()), x))


def _eagle_steps(seed):
    """`_steps`' steps, each decode row verifying `K` drafts, its price redone."""
    out = []
    for rows, rung, _ in _steps(seed):
        if rows[0][2]:
            rows = [(K + 1, max(n, K + 1), True) for _, n, _ in rows]
        out.append((rows, rung, _price(rows, rung)))
    return out


def _eagle_record(path, steps):
    """`steps` as an EAGLE3 record: a decode drafts its passes at `FORWARD`,
    padded to its rung, then the rest of its propose at `REST`; every third
    prefill is a middle chunk, whose draft is its draft-KV pass at `DRAFT_KV`."""
    _record(path, steps)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for i, (line, (rows, rung, seconds)) in enumerate(zip(lines, steps)):
        target = round(seconds * 1e9)
        view = BatchView(tuple(RequestShape(*r) for r in rows), draft_bs=rung)
        if view.decode:
            line["draft_bs"] = rung
            passes = [round(_dot(FORWARD, x) * 1e9) for x in serial_passes(view, K)]
            rest = round(_dot(REST, (1, len(rows))) * 1e9)
            line[DRAFT_PASSES] = passes
            draft = sum(passes) + rest
        elif i % 3 == 0:
            line["produces_output"] = False
            draft = round(_dot(DRAFT_KV, serial_passes(view, 1)[0]) * 1e9)
        else:  # an output prefill's draft, not fitted under a serial drafter
            draft = target
        line["d_target_ns"], line["d_busy_ns"] = target, target + draft
    path.write_text("".join(json.dumps(traced(x)) + "\n" for x in lines))
    return path


def _law(tmp_path, seeds=(1,)):
    records = [
        _eagle_record(tmp_path / f"dp0-{s}.jsonl", _eagle_steps(s)) for s in seeds
    ]
    samples = label_records(records)[0]
    law, report = fit(samples)
    if len(seeds) > 1:
        mark_held_out(law, report, samples)
    return law, report


def test_a_record_s_draft_passes_fit_the_forward_and_propose_vectors(tmp_path):
    draft = _law(tmp_path)[0]["draft"]

    assert draft["forward"]["passes"] == K
    # Each pass is recorded in whole nanoseconds.
    assert draft["forward"]["coefficients"] == pytest.approx(FORWARD, rel=1e-3)
    assert draft["propose"]["coefficients"] == pytest.approx(REST, rel=1e-3)
    # The draft law's prefill vector is the middle chunks' draft-KV pass; it
    # has no vector of its own for a decode's whole draft.
    kv = {k: v for k, v in draft["prefill"]["coefficients"].items() if k in DRAFT_KV}
    assert kv == pytest.approx(DRAFT_KV, rel=1e-3)
    assert draft["decode"] == {} and "draft_kv" not in draft


def test_each_draft_pass_is_priced_at_its_rows(tmp_path):
    backend = StepLinearBackend(StepLinearLaw(_law(tmp_path)[0]))
    rows = (RequestShape(K + 1, 9000, True), RequestShape(K + 1, 300, True))

    # Two requests decoding at rung 4: the first pass runs their 8 verify
    # rows, each later one a row a request padded to the rung.
    cost = backend.price(BatchView(rows, capture_rung=4, draft_bs=4))
    tokens = {t.name: t.seconds for t in cost.draft_terms if t.name.endswith("tokens")}
    assert tokens == pytest.approx(
        {
            f"draft.pass{i}.tokens": FORWARD["tokens"] * n
            for i, n in enumerate((8, 4, 4))
        },
        rel=1e-3,
    )
    planted = _dot(FORWARD, (1, 8, 2 * 16, 4 * (8996 + 296)))
    planted += 2 * _dot(FORWARD, (1, 4, 2, 8999 + 299)) + _dot(REST, (1, 2))
    assert cost.draft_seconds == pytest.approx(planted, rel=1e-4)

    # An output prefill drafts the same way, its first pass over its chunk.
    chunk = (RequestShape(2048, 6000, False),)
    out = backend.price(BatchView(chunk, produces_output=True))
    first = [t for t in out.draft_terms if t.name == "draft.pass0.tokens"]
    assert first[0].seconds == pytest.approx(FORWARD["tokens"] * 2048, rel=1e-3)
    assert {t.name.split(".")[1] for t in out.draft_terms} == {
        "pass0",
        "pass1",
        "pass2",
        "propose",
    }
    # A middle chunk's draft is its draft-KV pass.
    middle = backend.price(BatchView(chunk, produces_output=False))
    assert {t.name.split(".")[1] for t in middle.draft_terms} == {"prefill"}
    x = (1, 2048, 2048**2, 2048 * (6000 - 2048))
    assert middle.draft_seconds == pytest.approx(_dot(DRAFT_KV, x), rel=1e-4)


def test_each_draft_vector_is_held_out_by_run(tmp_path):
    law, report = _law(tmp_path, seeds=(1, 2))
    draft = law["draft"]

    bands = {
        key: draft[key]["held_out"]["bands"]
        for key in ("forward", "propose", "prefill")
    }
    assert set(bands["forward"]) == {"pass 0", "pass 1", "pass 2"}
    assert set(bands["propose"]) == {"propose"}
    assert all(b.startswith("prefill tokens") for b in bands["prefill"])
    for band in (b for vector in bands.values() for b in vector.values()):
        assert band["steps"] > 0
        assert band["p10"] == pytest.approx(0.0, abs=1e-4)
        assert band["p90"] == pytest.approx(0.0, abs=1e-4)
    for run in report["held_out"].values():
        assert {"draft pass 0", "draft pass 2", "draft propose"} <= set(run)

    # A held-out pass is validated; the law's terms say so.
    backend = StepLinearBackend(StepLinearLaw(law))
    steps = [s for s in _eagle_steps(1) if s[0][0][2] and s[1] == 4]
    rows, rung, _ = steps[0]
    view = BatchView(tuple(RequestShape(*r) for r in rows), rung, draft_bs=rung)
    terms = backend.price(view).draft_terms
    assert {t.name.split(".")[1] for t in terms} == {
        "pass0",
        "pass1",
        "pass2",
        "propose",
    }
    assert all(t.provenance.validated for t in terms)


def test_a_record_times_each_serial_draft_pass(tmp_path, monkeypatch):
    """A serial drafter's propose points its context at each pass it runs; the
    recording runner records a device event there, and each line of a forward
    that proposes carries each pass's device time, the last to the propose's
    end. A decode's label carries them where its draft label stands."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1], drafter="eagle")
    base = recorded.__mro__[2]

    class _Serial(base):
        def __init__(self, rank, config):
            super().__init__(rank, config)
            self.drafter.is_block_drafter = False
            self.drafter._publish_draft_shape = lambda *a, **k: None

        def propose_draft_token_ids(self, batch, align_only=False):
            for work in ("kv", "propose", "propose"):  # KV, then PROPOSE twice
                getattr(self.drafter, "_publish_draft_shape", lambda: None)()
                self.queued.append(work)

    class _Eagle(recorded, _Serial):
        pass

    device.tick = 10**12
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = _Eagle(0, _config())
    for b in batches:
        runner.forward(b)
    runner.exit()

    lines = read(tmp_path / "real")[0]
    outputs = [b.produces_output() for b in batches]
    assert any(outputs) and not all(outputs)
    for line, output in zip(lines, outputs):
        if output:
            assert line[DRAFT_PASSES] == [KV, PROPOSE, PROPOSE]
            assert line["d_propose_ns"] == KV + 2 * PROPOSE
        else:
            assert DRAFT_PASSES not in line
    samples = label(tmp_path / "real" / "dp0.jsonl")[0]
    decodes = [s for s in samples if s.view.decode and s.draft_s is not None]
    assert decodes
    for s in decodes:
        assert s.passes_s == pytest.approx((KV * 1e-9, PROPOSE * 1e-9, PROPOSE * 1e-9))
    assert all(s.passes_s is None for s in samples if s.view.prefill)


# --- with a driver -----------------------------------------------------------

M3 = os.environ.get("ATOM_COMPASS_M3_MODEL")
EAGLE3 = os.environ.get("ATOM_COMPASS_M3_EAGLE3")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """MiniMax-M3 at TP4 with its EAGLE3 drafter, the stage's draft width and
    forced acceptance, served twice under `test_spec_scenario`'s stub."""
    if not (torch.cuda.is_available() and M3 and EAGLE3):
        pytest.skip(
            "a run needs a driver, ATOM_COMPASS_M3_MODEL and ATOM_COMPASS_M3_EAGLE3"
        )
    import test_dp_slice
    from test_spec_scenario import MACHINE, STUB

    flags = ["-tp", "4", "--trust-remote-code", "--block-size", "128"]
    flags += ["--kv_cache_dtype", "fp8", "--index-cache-dtype", "fp8"]
    flags += ["--method", "eagle3", "--draft-model", EAGLE3]
    flags += ["--num-speculative-tokens", str(K)]
    flags += ["--spec-decode-acceptance-length", "2.78"]
    root = tmp_path_factory.mktemp("eagle3")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_dp_slice, "MODEL", M3)
        mp.setattr(test_dp_slice, "CONSTANT", STUB)
        return [
            test_dp_slice.run_dp_slice(
                root / f"seed{s}", s, flags, dp=1, machine=copy.deepcopy(MACHINE)
            )
            for s in ("1", "2")
        ]


def test_m3_with_its_eagle3_drafter_meets_every_scenario_check(runs):
    from test_dp_slice import MAX_TOKENS

    from atom.compass.detect.scenario import check_run

    print(f"\n{runs[0]['table']}")
    failed = check_run(
        [r["out"] for r in runs],
        MAX_TOKENS,
        [line for _, line in runs[0]["events"]],
        {"engine": runs[0]["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)
    # Every verify drafts K tokens a row, each line the draft graph's batch.
    decodes = [s for s in runs[0]["parity"][0] if all(r[2] for r in s["rows"])]
    assert decodes and all(r[0] == K + 1 for s in decodes for r in s["rows"])
    assert runs[0]["summary"]["configuration"]["step_stub"]["draft_step"] > 0
