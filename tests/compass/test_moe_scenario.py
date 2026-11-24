# SPDX-License-Identifier: MIT
"""EP on a model with experts, at TP 2 and under DP-attention, two-batch
overlap, and DP without DP-attention, each served twice under two
`PYTHONHASHSEED` by `test_dp_slice`'s four requests and held to every scenario
check: at level 0 under its constant step stub, at level 1 under the step
roofline alone on `ROOFLINE_MACHINE` (`08` D120).

The expert forms serve Qwen3-30B-A3B's config, 128 experts, 8 a token:
``-tp 2 --enable-expert-parallel``, each TP rank holding half the experts and
no all-to-all; the same at DP-attention width 2, where MoRI's dispatch and
combine carry each token to the rank that holds its experts; and that with
``--enable-tbo``, which splits a prefill into two micro-batches. TBO splits a
prefill of `ATOM_TBO_PREFILL_MIN_TOKENS` tokens or more, 8192 by default, so
the TBO runs set it to 0: the four prompts are a few tokens each. Pure DP
serves the slice model at ``-dp 2``.

The runs need a driver, as in `test_vertical_slice`, and a model directory
holding a config and a tokenizer: `ATOM_COMPASS_MOE_MODEL`, Qwen3-30B-A3B's
(`qwen3_30b_a3b_config.json`, Qwen3-30B-A3B-Instruct-2507's, beside its
tokenizer), and `ATOM_COMPASS_SLICE_MODEL`; no weight is read. The rest run
anywhere: Qwen3-30B-A3B's meta build at each expert form, walked as
`run.runner` walks it, every module classified and its steps priced by the
step roofline on `ROOFLINE_MACHINE`.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest
import test_dp_slice
import torch
from capabilities import ROOFLINE, ROOFLINE_MACHINE, capability
from test_dp_slice import (  # noqa: F401
    DP_MACHINE,
    MAX_TOKENS,
    device_classes_put_back,
    run_dp_slice,
)
from test_vertical_slice import CONSTANT, TREE

from atom.compass.detect.scenario import check_run

MOE = "ATOM_COMPASS_MOE_MODEL"
SLICE = "ATOM_COMPASS_SLICE_MODEL"
DPA_EP = ["-tp", "2", "--enable-dp-attention", "--enable-expert-parallel"]
#: Each form: its model's variable, the server's flags, its DP ranks, and the
#: environment its runs add.
FORMS = {
    "ep": (MOE, ["-tp", "2", "--enable-expert-parallel"], 1, {}),
    "ep_dp": (MOE, DPA_EP, 2, {}),
    "tbo": (MOE, [*DPA_EP, "--enable-tbo"], 2, {"ATOM_TBO_PREFILL_MIN_TOKENS": "0"}),
    "pure_dp": (SLICE, ["-dp", "2"], 2, {}),
}
#: Each level's backend and the machine it prices on.
LEVELS = {0: (CONSTANT, DP_MACHINE), 1: (ROOFLINE, ROOFLINE_MACHINE)}


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((form, level), marks=capability(form, level), id=f"{form}-{level}")
        for form in FORMS
        for level in LEVELS
    ],
)
def runs(request, tmp_path_factory):
    """The form, the level and the form's two runs at that level."""
    form, level = request.param
    name, flags, dp, env = FORMS[form]
    model = os.environ.get(name)
    if not (torch.cuda.is_available() and model):
        pytest.skip(f"a run needs a driver and {name} naming a model directory")
    backend, machine = LEVELS[level]
    root = tmp_path_factory.mktemp(f"{form}-{level}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_dp_slice, "MODEL", model)
        for key, value in env.items():
            mp.setenv(key, value)
        done = [
            run_dp_slice(
                root / f"seed{s}", s, flags, dp, machine=machine, backend=backend
            )
            for s in ("1", "2")
        ]
    return form, level, done


def _terms(run) -> set[str]:
    """The names of the terms every rank's cost lines price in `run`."""
    names = set()
    for path in sorted((run["out"] / "parity").glob("dp*.cost.jsonl")):
        _, *lines = map(json.loads, path.read_text().splitlines())
        names |= {t["name"] for line in lines for t in line["terms"]}
    return names


def test_each_form_meets_every_scenario_check(runs):
    """At level 1, every check but the hand-computed timeline, which only a
    constant stub's run has."""
    form, level, done = runs
    run = done[0]
    print(
        f"\n{form} at level {level}: {run['summary']['schedule']['seconds_by_method']}"
    )
    print(run["table"])
    records = {"engine": run["out"] / "parity"} if level == 0 else None
    failed = check_run(
        [r["out"] for r in done],
        MAX_TOKENS,
        [line for _, line in run["events"]],
        records,
    )
    assert not failed, "\n".join(failed)


def test_each_form_runs_as_its_flags_say(runs):
    """Every DP rank serves a request; under TBO each rank's prefill is split;
    under the roofline the routed experts are priced, and under TBO the
    overlap of the two micro-batches."""
    form, level, done = runs
    steps = done[0]["parity"]
    assert sorted(steps) == list(range(FORMS[form][2]))
    assert all(any(s["batch"] for s in steps[rank]) for rank in steps)
    if form == "tbo":
        assert all(any(s.get("tbo_tokens") for s in steps[rank]) for rank in steps)
    if level == 1:
        terms = _terms(done[0])
        assert ("roofline.moe" in terms) == (form != "pure_dp"), sorted(terms)
        assert ("roofline.tbo" in terms) == (form == "tbo"), sorted(terms)


# --- without a driver ---------------------------------------------------------

CONFIG = pathlib.Path(__file__).with_name("qwen3_30b_a3b_config.json")

#: Builds the config under argv's root on the meta device of argv's machine's
#: arch, with argv's engine fields, as `run.runner` builds it; walks it, and
#: prints each module and the step roofline's price of each of argv's views, a
#: refusal's reason in place of a refused one's terms.
BUILD = """
import json, pathlib, sys, torch
root, machine = pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text())
fields, views = json.loads(sys.argv[3]), json.loads(sys.argv[4])
from atom.compass.capture import driverless
driverless.install(str(root), machine["device"]["arch"])
driverless.group(fields.get("tensor_parallel_size", 1), 1)
from atom.config import Config, set_current_atom_config
config = Config(model=str(root / "model"), load_dummy="empty", enforce_eager=True, **fields)
set_current_atom_config(config)
from atom.model_engine.model_runner import support_model_arch_dict
from atom.utils import resolve_obj_by_qualname
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.run.modules import drafter, walk
from atom.compass.runner.overrides import FactoriesOnMeta, process_weights_after_loading
from atom.compass.spec import MachineSpec
torch.set_default_dtype(config.torch_dtype)
with FactoriesOnMeta():
    model = support_model_arch_dict[config.hf_config.architectures[0]]
    tree = resolve_obj_by_qualname(model)(config)
    process_weights_after_loading(tree)
modules = walk(tree, drafter(config))
roofline = StepRooflineBackend(MachineSpec.from_mapping(machine), modules)
def priced(rows, view):
    if "tbo_tokens" in view:
        view["tbo_tokens"] = tuple(map(tuple, view["tbo_tokens"]))
    try:
        cost = roofline.price(BatchView(tuple(RequestShape(*r) for r in rows), **view))
    except CostRefused as refused:
        return str(refused)
    return {t.name: t.seconds for t in cost.terms}
print(json.dumps({
    "modules": [[m.name, m.kind, list(m.dims), m.fmt, m.ranks] for m in modules],
    "priced": {name: priced(*v) for name, v in views.items()},
}))
"""

PREFILL = [[512, 512, False]]
DECODE = [[1, 600 + i, True] for i in range(4)]
#: One prefill of 512 tokens and a decode of four rows, on one rank.
ONE_RANK = {"prefill": [PREFILL, {}], "decode": [DECODE, {}]}


def _dp_rank_0(rows, moe_rows, **view):
    """`rows` on rank 0 of a DP group of two, the group's gather carrying
    `moe_rows` real rows."""
    view |= {"moe_rows": moe_rows, "moe_tokens": moe_rows, "dp_rank": 0}
    return [rows, view | {"group_prefills": not rows[0][2]}]


#: The same on rank 0 of a DP group of two whose other rank schedules half as
#: much; and the prefill split into two micro-batches, half of each rank's
#: tokens in each.
DP_RANK = {
    "prefill": _dp_rank_0(PREFILL, 768),
    "decode": _dp_rank_0(DECODE, 6),
    "split": _dp_rank_0(PREFILL, 768, tbo_tokens=[[256, 128], [256, 128]]),
}


def walked(config: dict, fields: dict, views: dict, tmp_path) -> dict:
    """`config`'s meta build at the engine `fields`, walked and priced on
    `ROOFLINE_MACHINE` in a process of its own, since the driverless
    declarations are process-global: each module, as ``[name, kind, dims,
    fmt, ranks]``, and each of `views`' price by term, or its refusal."""
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(config))
    (tmp_path / "machine.json").write_text(json.dumps(ROOFLINE_MACHINE))
    args = [str(tmp_path), str(tmp_path / "machine.json")]
    done = subprocess.run(
        [sys.executable, "-c", BUILD, *args, json.dumps(fields), json.dumps(views)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.splitlines()[-1])


#: Each expert form's engine fields, its DP ranks and the views it prices. A
#: DP-attention width of 2 is declared to the meta build of one rank as the
#: width its experts are sharded over.
EXPERT_FORMS = {
    "ep": ({"tensor_parallel_size": 2}, 1, ONE_RANK),
    "ep_dp": ({"enable_dp_attention": True, "dp_logical_size": 2}, 2, DP_RANK),
}


@pytest.mark.parametrize("form", sorted(EXPERT_FORMS))
def test_each_expert_form_classifies_every_module_and_prices_its_steps(form, tmp_path):
    """Every module of Qwen3-30B-A3B's build at the form is classified; each
    of its 48 MoE layers holds 64 of the 128 experts, with 768 intermediate
    rows an expert and 8 experts a token, over an expert group of 2; at TP 2
    its output is all-reduced over the TP ranks, and under DP-attention it
    exchanges with the other DP rank. The roofline prices each view with no
    refusal, the routed experts among its terms, and a split prefill as the
    overlap of its micro-batches (`tbo`)."""
    fields, dp, views = EXPERT_FORMS[form]
    fields = {**fields, "enable_expert_parallel": True}
    out = walked(json.loads(CONFIG.read_text()), fields, views, tmp_path)
    print(f"\n{form}: {out['priced']}")
    modules = out["modules"]

    assert [m[0] for m in modules if m[1] == "unclassified"] == []
    moe = [m for m in modules if m[1] == "moe"]
    assert len(moe) == 48
    assert {tuple(m[2][:7]) for m in moe} == {(128, 64, 8, 768, 2048, 2, dp)}
    assert {m[4] for m in moe} == {2}
    for name, terms in out["priced"].items():
        assert isinstance(terms, dict), f"{name}: {terms}"
        kind = "roofline.tbo" if name == "split" else "roofline.moe"
        assert terms[kind] > 0, (name, terms)
    priced = out["priced"]
    assert sum(priced["prefill"].values()) > sum(priced["decode"].values())
