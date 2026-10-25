# SPDX-License-Identifier: MIT
"""A speculative run, MTP and DSpark, held to every scenario check: each served
twice, under two `PYTHONHASHSEED`, by `test_dp_slice`'s four requests under a
constant step stub that prices a draft on every forward.

The acceptance is declared, so the drafts a step accepts are drawn from the
runner's seeded stream. The runs need a driver, as in `test_vertical_slice`,
and a model directory holding a config and a tokenizer, whose config declares
the drafter: `ATOM_COMPASS_MTP_MODEL` for MTP, `ATOM_COMPASS_DSPARK_MODEL` for
DSpark; no weight is read. The rest run anywhere.
"""

import copy
import os
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
from test_memory_readings import TOKENIZER
from test_vertical_slice import CONSTANT

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.detect.scenario import check_run, expected_timeline
from atom.compass.runner.overrides import HostDeviceRecurrence

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
#: The slice's machine, with a declared tokenizer for DeepSeek's.
MACHINE = copy.deepcopy(DP_MACHINE)
MACHINE["host"]["tokenizers"].append(
    TOKENIZER
    | {
        "id": "deepseek-v4-129k-bpe",
        "vocab_size": 129280,
        "applies_to": ["DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"],
    }
)


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


# --- without a driver ---------------------------------------------------------


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
