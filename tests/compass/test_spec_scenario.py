# SPDX-License-Identifier: MIT
"""A speculative run, MTP and DSpark, held to every scenario check: each served
twice, under two `PYTHONHASHSEED`, by `test_dp_slice`'s four requests under a
constant step stub that prices a draft on every forward, and again priced by
the step roofline alone (`08` D120 point 3), every verify step with a draft
term. Under the roofline DSpark is priced on an MI355X's spec, whose FP4 peak
its experts need and MI308X's spec does not state, and MTP on node 18's, its
tokenizer charged by `QWEN3_8_TOKENIZER`, assumed.

The acceptance is declared, so the drafts a step accepts are drawn from the
runner's seeded stream. The runs need a driver, as in `test_vertical_slice`,
and a model directory holding a config and a tokenizer, whose config declares
the drafter: `ATOM_COMPASS_MTP_MODEL` for MTP, `ATOM_COMPASS_DSPARK_MODEL` for
DSpark; no weight is read. The rest run anywhere.
"""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_dp_slice
import torch
from test_dp_slice import (  # noqa: F401
    DP_MACHINE,
    MAX_TOKENS,
    device_classes_put_back,
    run_dp_slice,
)
from test_served_tokenizer import _LoadedFrom, _model_dir
from test_vertical_slice import CONSTANT, _run_file

from atom.compass import run as compass_run
from atom.compass.backends.json_file import read_jsonl
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.detect.scenario import check_run, expected_timeline
from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.compass.spec.rules import FingerprintMismatch

DRAFT_S = 2.0**-10
STUB = CONSTANT | {"draft_step": DRAFT_S}
ACCEPTANCE = ["--spec-decode-acceptance-rate", "0.6"]
#: Each method's model variable and server flags, after the recipes'.
METHODS = {
    "mtp": ("ATOM_COMPASS_MTP_MODEL", ["--method", "mtp"], "1"),
    "dspark": (
        "ATOM_COMPASS_DSPARK_MODEL",
        ["-tp", "2", "--trust-remote-code", "--method", "dspark"],
        "5",
    ),
}
#: The slice's machine.
MACHINE = copy.deepcopy(DP_MACHINE)
MACHINES = Path(__file__).parent / "machines"
#: MTP's tokenizer charge, assumed for Qwen3.5-0.8B, since no MI308X node can
#: record it now: the ``host.tokenizer`` term of
#: `scripts/compass/laws/qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json.xz`
#: as 4c53e5fe2 (#3932) fitted it, by `compass fit host` (36b46d7be8b9) on one
#: record on hjbog-srdc-39, an MI308X node (tokenizer.jsonl:90140d7c474d); this
#: branch's copy of that file predates the fit. Its fingerprint stays Qwen3.8's
#: tokenizer.json, the file the timed calls loaded (`05`), so the MTP run warns
#: `FingerprintMismatch`, naming it and `QWEN3_5_JSON`.
QWEN3_8_TOKENIZER = {
    "backend": "fast",
    "fingerprint": "sha256:"
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "encode": {"fixed_s": 0.0018664796139258922, "tokens_per_s": 668474.2547418053},
    "decode": {"fixed_s": 1.1371071666609296e-05, "tokens_per_s": 887576.7512085318},
}
#: Qwen3.5's tokenizer.json, the 0.8B and 9B snapshots' on g08.
QWEN3_5_JSON = "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
#: Each method's machine under the step roofline, and its run file's more keys.
ROOFLINE = {
    "mtp": ("mi308x-hjbog18.json", {"host": {"tokenizer": QWEN3_8_TOKENIZER}}),
    "dspark": ("mi355x-gpu40.json", {}),
}


@pytest.fixture(scope="module", params=sorted(METHODS))
def runs(request, tmp_path_factory):
    name, flags, drafts = METHODS[request.param]
    model = os.environ.get(name)
    if not (torch.cuda.is_available() and model):
        pytest.skip(f"a run needs a driver and {name} naming a model directory")
    flags = [*flags, "--num-speculative-tokens", drafts, *ACCEPTANCE]
    root = tmp_path_factory.mktemp(request.param)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_dp_slice, "MODEL", model)
        mp.setattr(test_dp_slice, "CONSTANT", STUB)
        return [
            run_dp_slice(root / f"seed{s}", s, flags, dp=1, machine=MACHINE)
            for s in ("1", "2")
        ]


def test_a_speculative_run_meets_every_scenario_check(runs):
    print(f"\n{runs[0]['table']}")
    failed = check_run(
        [r["out"] for r in runs],
        MAX_TOKENS,
        [line for _, line in runs[0]["events"]],
        {"engine": runs[0]["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)


def roofline_run_file(out: Path, more: dict, **keys) -> Path:
    """`_run_file` with `more`'s keys and no step stub: the step roofline alone
    prices every step (`08` D120 point 3)."""
    path = _run_file(out, **keys, **more)
    run = json.loads(path.read_text())
    del run["step_stub"]
    path.write_text(json.dumps(run | {"step_roofline": True}))
    return path


@pytest.fixture(scope="module", params=sorted(METHODS))
def roofline(request, tmp_path_factory):
    """The method's name and its two seeds' runs under the step roofline."""
    name, flags, drafts = METHODS[request.param]
    model = os.environ.get(name)
    if not (torch.cuda.is_available() and model):
        pytest.skip(f"a run needs a driver and {name} naming a model directory")
    flags = [*flags, "--num-speculative-tokens", drafts, *ACCEPTANCE]
    spec, more = ROOFLINE[request.param]
    machine = json.loads((MACHINES / spec).read_text())
    root = tmp_path_factory.mktemp(f"{request.param}_roofline")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_dp_slice, "MODEL", model)
        mp.setattr(
            test_dp_slice,
            "_run_file",
            lambda out, **keys: roofline_run_file(out, more, **keys),
        )
        return request.param, [
            run_dp_slice(root / f"seed{s}", s, flags, dp=1, machine=machine)
            for s in ("1", "2")
        ]


def verify_prices(record: Path) -> list[tuple[float, float, list[str]]]:
    """Each verify step's target and draft price and draft terms, from DP rank
    0's cost lines in the step record `record` (`_group_step_seconds`): a
    step whose every row decodes."""
    steps = {s["step"]: s for s in read_jsonl(record / "dp0.jsonl")}
    _, *costs = read_jsonl(record / "dp0.cost.jsonl")
    return [
        (
            c["device_s"],
            c["draft_s"],
            [t["name"] for t in c["terms"][c["target_terms"] :]],
        )
        for c in costs
        if (rows := steps[c["step"]]["rows"]) and all(d for *_, d in rows)
    ]


def test_a_speculative_run_meets_every_scenario_check_under_the_step_roofline(
    roofline,
):
    """Every check but the hand-computed timeline, which only a constant stub's
    run has; each verify step prices a draft term. A server charging an
    assumed tokenizer term warns, naming its file and the one it loaded."""
    method, runs = roofline
    summary = runs[0]["summary"]["schedule"]
    print(f"\n{summary['seconds_by_method']}\n{runs[0]['table']}")
    failed = check_run(
        [r["out"] for r in runs],
        MAX_TOKENS,
        [line for _, line in runs[0]["events"]],
    )
    assert not failed, "\n".join(failed)
    verify = verify_prices(runs[0]["out"] / "parity")
    assert verify and all(d > 0 and names for _, d, names in verify), verify
    assert all(n.startswith("draft roofline.") for *_, names in verify for n in names)
    target, draft = (sum(v) / len(verify) for v in list(zip(*verify))[:2])
    print(
        f"{method}: {len(verify)} verify steps, mean target {target!r} s, "
        f"mean draft {draft!r} s a step"
    )
    if term := ROOFLINE[method][1].get("host", {}).get("tokenizer"):
        for run in runs:
            log = (run["out"] / "server.log").read_text().splitlines()
            warned = [line for line in log if FingerprintMismatch.__name__ in line]
            print("\n".join(warned))
            assert any(
                term["fingerprint"] in line and QWEN3_5_JSON in line for line in warned
            ), warned


# --- without a driver ---------------------------------------------------------


def test_the_mtp_run_files_tokenizer_term_warns_on_another_tokenizer_json(
    monkeypatch, tmp_path
):
    """The MTP run file under the roofline passes the run file's checks with
    no step stub, and its tokenizer term, Qwen3.8's, loaded on another
    tokenizer.json, warns naming both."""
    run = roofline_run_file(tmp_path, ROOFLINE["mtp"][1], step_stub=CONSTANT)
    monkeypatch.setenv(compass_run.ENV, str(run))
    checked = compass_run.spec()
    assert "step_stub" not in checked and checked["step_roofline"] is True
    assert checked["host"]["tokenizer"] == QWEN3_8_TOKENIZER
    model, loaded = _model_dir(tmp_path)
    with pytest.warns(FingerprintMismatch) as caught:
        compass_run.tokenizer(_LoadedFrom(model), None)
    message = str(caught[0].message)
    assert QWEN3_8_TOKENIZER["fingerprint"] in message and loaded in message


def test_a_stub_with_a_draft_intercept_prices_a_draft_and_stays_constant():
    view = BatchView((RequestShape(2, 100, True),))
    cost = StepStubBackend(Coefficients(**STUB)).price(view)
    assert (cost.target_seconds, cost.draft_seconds) == (STUB["decode_step"], DRAFT_S)
    assert Coefficients(**STUB).shape_blind
    assert not StepStubBackend().price(view).draft_terms


def test_the_recurrence_charges_each_draft_as_the_timeline_expects():
    """A prefill of 4 s and two decodes of 1 s, each with a 2 s draft: the
    first decode waits for the prefill's target, the second for the first
    decode's, which ran after the prefill's draft."""
    steps = [[[8, 8, False]], [[1, 9, True]], [[1, 10, True]]]
    record = {0: [{"rows": rows, "produces_output": True} for rows in steps]}
    want = [(0, 0), (0, 4), (4, 7)]
    assert expected_timeline(record, 4.0, 1.0, want, draft_s=2.0) == want
    clock, got = [0.0], []
    rec = HostDeviceRecurrence({}, (), lambda: clock[0], mtp=True)
    for rows in steps:
        batch = SimpleNamespace(
            total_seqs_num_prefill=int(not rows[0][2]),
            total_tokens_num=rows[0][0],
            total_seqs_num=1,
            produces_output=lambda: True,
        )
        t = rec.charge(batch, 1.0 if rows[0][2] else 4.0, 2.0)
        got.append((clock[0], clock[0] + t))
        clock[0] += t
    assert got == want
