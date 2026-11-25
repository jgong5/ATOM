# SPDX-License-Identifier: MIT
"""A pipeline stage is priced by its own part of the step linear law and of
the host laws: `compass fit steps` and `fit host` write one entry a pipeline,
a part per stage under ``stages``, and `run.runner` builds each stage's
`StepLinearBackend` and `HostDeviceRecurrence` from the part at its rank. The
backend states its stage, which `_group_step_seconds` admits as it admits a
step stub's `stage_share`; only the last stage holds a draft law.

The served run is `test_pp_drafter`'s 1P1D, its PP2 x TP1 prefill drafting,
priced by a planted two-part law and host entry in place of the step stub,
each side's entry keyed as its server resolves its config; it needs atomesh on
PATH and no driver.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_pd_prefill_pp
import test_vertical_slice
from test_backend_step_linear import law
from test_dp_step_max import prefill, runner
from test_filler_tokens import model_dir  # noqa: F401  a fixture
from test_memory_readings import DOCUMENT, META_8B
from test_pd_slice import publish_pd_entries
from test_pp_drafter import DRAFT, QWEN, STAGES
from test_run_step_grid import deployment
from test_speculative_step import _speculative
from test_vertical_slice import MAX_TOKENS, TREE, _run_file, publish_entry_on_arch

from atom.compass import run as compass_run
from atom.compass.artifacts import Kind, keyed_platform, machine_platform
from atom.compass.backends.launch_features import GRAPHS
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_linear import StepLinearBackend, StepLinearLaw
from atom.compass.backends.step_stub import Coefficients
from atom.compass.run.step_grid import keyed
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    MTP_SEGMENT,
    PP_RECV_SEGMENT,
    RunnerRefusal,
    _group_step_seconds,
    _installed_backend,
)
from atom.compass.spec import MachineSpec

#: Each stage's constant prefill price, and the last stage's draft's.
PREFILL_S, DRAFT_S = (2.0**-9, 2.0**-8), 2.0**-10
#: Each stage's host `E`, and the last stage's `W_recv`.
E_S, RECV_S = (2.0**-12, 2.0**-11), 2.0**-13


def _part(prefill_s: float, draft_s: float = 0.0) -> dict:
    """A stage's part of a law, as `compass fit steps` writes one: a constant
    prefill vector of `prefill_s`, no decode vector, and with `draft_s` a draft
    law of it."""
    part = law(Coefficients.constant(prefill_s, 0.0)) | {"decode": {}}
    del part["platform"]
    if draft_s:
        part["draft"] = _part(draft_s)
    return part


def _host(e_s: float, recv_s: float = 0.0) -> dict:
    """A stage's part of host laws: every kind's `E` costs `e_s`, its
    `W_recv` `recv_s` where given, and every other segment nothing."""
    laws = {s: [0.0, 0.0, 0.0] for s in (*HOST_SEGMENTS, MTP_SEGMENT)}
    laws["E"] = [e_s, 0.0, 0.0]
    if recv_s:
        laws[PP_RECV_SEGMENT] = [recv_s, 0.0, 0.0]
    return {"laws": {kind: laws for kind in FORWARD_KINDS}}


PARTS = [_part(PREFILL_S[0]), _part(PREFILL_S[1], DRAFT_S)]
#: A prefill of eight tokens, as `prefill(8)` schedules it.
EIGHT = BatchView((RequestShape(8, 8, False),))
HOST_PARTS = [_host(E_S[0]), _host(E_S[1], RECV_S)]


def _platform(kind: Kind, machine: dict) -> dict:
    return keyed_platform(kind, machine_platform(MachineSpec.from_mapping(machine)))


def _staged(machine=DOCUMENT, parts=PARTS) -> dict:
    return {
        "provenance": "planted, a part per stage",
        "platform": _platform(Kind.STEP_LINEAR_LAW, machine),
        "stages": parts,
    }


def _deployment(tmp_path, pp: int, rank: int):
    stage = SimpleNamespace(
        data_parallel_size=1, data_parallel_rank=0, pipeline_parallel_rank=rank
    )
    return deployment(tmp_path, pipeline_parallel_size=pp, parallel_config=stage)


def _installed(monkeypatch, tmp_path, rank, pp=2, step_linear=None, host=None):
    """Stage `rank` of `pp` after `run.runner` installed it from a run file
    holding the planted two-part law, or `step_linear`, and host laws `host`,
    the planted parts by default, each at this deployment's key."""
    r = _deployment(tmp_path, pp, rank)
    key = compass_run.host_key(r.config) | _platform(Kind.HOST_LAWS, DOCUMENT)
    run = json.loads(_run_file(tmp_path).read_text())
    del run["step_stub"]
    step_linear = step_linear or _staged()
    run["step_linear"] = step_linear | {"engine_config": keyed(r.config)}
    run["host"] = {"key": key, **(host or {"stages": HOST_PARTS})}
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    np.zeros(1).tofile(tmp_path / "now.bin")
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run.runner(r, META_8B)
    return r


def test_each_stage_installs_its_own_part_of_the_law_and_of_the_host_laws(
    monkeypatch, tmp_path
):
    for rank in range(2):
        r = _installed(monkeypatch, tmp_path, rank)
        backend = _installed_backend(r)
        assert backend.stage == (rank, 2)
        assert backend.price(EIGHT).target_seconds == PREFILL_S[rank]
        assert (backend.law.draft is not None) == (rank == 1)
        assert r.compass_host.laws == HOST_PARTS[rank]["laws"]


@pytest.mark.parametrize(
    "pp, step_linear, host, refusal",
    [
        (2, law(), None, "step linear law holds no part per stage"),
        (2, None, HOST_PARTS[0], "host laws holds no part per stage"),
        (
            2,
            _staged(parts=PARTS * 2),
            None,
            "holds 4 stages' parts, and this run has 2",
        ),
        (1, None, None, "holds 2 stages' parts, and this run has 1 pipeline stage,"),
    ],
    ids=["law", "host", "four-parts", "one-stage"],
)
def test_an_entry_without_a_part_per_stage_is_refused_by_name(
    monkeypatch, tmp_path, pp, step_linear, host, refusal
):
    with pytest.raises(ValueError, match=refusal):
        _installed(monkeypatch, tmp_path, 0, pp, step_linear, host)


def _stage(rank: int, backend):
    """Stage `rank` of two, drafting at a declared acceptance, priced by `backend`."""
    stage = _speculative(runner(backend))
    stage.config.pipeline_parallel_size = 2
    stage.config.parallel_config.pipeline_parallel_rank = rank
    return stage


def test_each_stage_charges_its_own_part_and_the_last_alone_its_draft():
    stages = [
        _stage(k, StepLinearBackend(StepLinearLaw(p), stage=(k, 2)))
        for k, p in enumerate(PARTS)
    ]
    seconds = [
        _group_step_seconds(s, prefill(8), _installed_backend(s)) for s in stages
    ]
    assert seconds == [PREFILL_S[0], PREFILL_S[1] + DRAFT_S]
    assert [s.refused_commands() for s in stages] == [(), ()]


def test_a_backend_pricing_another_stage_is_refused_naming_both():
    stage = _stage(1, StepLinearBackend(StepLinearLaw(PARTS[0]), stage=(0, 2)))
    named = "pipeline stage 1 of 2, and StepLinearBackend prices stage 0 of 2"
    with pytest.raises(RunnerRefusal, match=named):
        _group_step_seconds(stage, prefill(8), _installed_backend(stage))


@pytest.mark.parametrize(
    "host, refusal",
    [
        ({"stages": HOST_PARTS}, None),
        ({"stages": [{**HOST_PARTS[0], "skip": ["E"]}]}, "host's stages is"),
        ({**HOST_PARTS[0], "stages": HOST_PARTS}, "host's stages is"),
        ({"stages": [{"laws": {"decod": {}}}]}, "host is {'laws': {'decod'"),
    ],
    ids=["staged", "part-holds-skip", "laws-beside", "part-checked"],
)
def test_a_run_file_reads_a_pipeline_s_host_entry_and_refuses_a_misshapen_one(
    monkeypatch, tmp_path, host, refusal
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, host=host)))
    if refusal is None:
        assert compass_run.spec()["host"] == host
        return
    with pytest.raises(ValueError, match=refusal):
        compass_run.spec()


def test_a_pipeline_s_host_laws_under_another_key_are_refused(tmp_path):
    """As `compass advice` holds an entry it finds to this configuration."""
    r = _deployment(tmp_path, 2, 0)
    platform = _platform(Kind.HOST_LAWS, DOCUMENT)
    key = compass_run.host_key(r.config) | platform
    key["engine"] = key["engine"] | {"tensor_parallel_size": 2}
    host = {"key": key, "stages": HOST_PARTS}
    with pytest.raises(ValueError, match="host laws were fitted under another"):
        compass_run.refuse_unfitted_host(host, r.config, platform)


#: A staged law whose parts state the buckets, and its engine config none.
BUCKETED = {
    "engine_config": {"env": {}},
    "stages": [{"prefill": {"buckets": [8, 16]}}] * 2,
}
#: Staged host laws whose key states the prefill bucket 8 alone.
ONE_BUCKET = {
    "key": {"engine": {GRAPHS: {"architecture": None, "token": {"prefill": [8]}}}},
    "stages": HOST_PARTS,
}


@pytest.mark.parametrize(
    "entry, refusal",
    [
        ({"step_linear": BUCKETED}, None),
        ({"host": ONE_BUCKET}, r"key states the prefill token-graph buckets \[8\]"),
    ],
    ids=["law", "host"],
)
def test_a_stage_holds_its_own_part_to_the_buckets_its_runner_records(entry, refusal):
    stage = SimpleNamespace(pipeline_parallel_rank=1)
    config = SimpleNamespace(pipeline_parallel_size=2, parallel_config=stage)
    if refusal is None:
        compass_run.refuse_other_buckets(config, (8, 16), entry)
        return
    with pytest.raises(ValueError, match=refusal):
        compass_run.refuse_other_buckets(config, (8, 16), entry)


# --- the served run -----------------------------------------------------------

#: Each side's host key as its server resolves its config, printed: in a
#: process whose aiter answers the machine's arch, under the side's
#: environment, as `publish_entry_on_arch` resolves it for the readings.
HOST_KEY = """
import json, os, sys
from unittest import mock
from atom.compass.capture.driverless import import_aiter_on
import_aiter_on(sys.argv[1])
from atom.compass import run
from atom.compass.drive import run as drive_run
argv, env = map(json.loads, sys.argv[2:])
with mock.patch.dict(os.environ, env, clear=True):
    print(json.dumps(run.host_key(drive_run._config(argv))))
"""


def _host_key(machine: dict, argv: list, env: dict) -> dict:
    """The host key of the side `pd_sim.sh` starts with `argv` under `env`."""
    done = subprocess.run(
        [sys.executable, "-c", HOST_KEY, machine["device"]["arch"]]
        + [json.dumps(argv), json.dumps(env)],
        env=dict(os.environ, PYTHONPATH=str(TREE)),
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    key = json.loads(done.stdout.splitlines()[-1])
    return key | _platform(Kind.HOST_LAWS, machine)


def _planted(store, machine, env) -> None:
    """Each side's readings published, as `test_pp_drafter` publishes them;
    then, in the run file beside `store`, in place of its step stub, a law and
    host laws for each side at its key: the prefill's in a part per stage, the
    decode's one-stage, constant, its draft at `DRAFT_S`, and no host law."""
    keys = []

    def publish(store, machine, argv, env):
        publish_entry_on_arch(store, machine, argv, env)
        keys.append(_host_key(machine, argv, env))

    publish_pd_entries(store, machine, env, publish=publish)
    prefill_key, decode_key = keys
    decode = law(Coefficients.constant(PREFILL_S[0], 2.0**-11))
    decode["draft"] = law(Coefficients.constant(DRAFT_S, DRAFT_S))
    platform = _platform(Kind.STEP_LINEAR_LAW, machine)
    path = Path(store).parent / "run.json"
    run = json.loads(path.read_text())
    del run["step_stub"]
    run["step_linear"] = [
        _staged(machine) | {"engine_config": prefill_key["engine"]},
        decode | {"platform": platform, "engine_config": decode_key["engine"]},
    ]
    run["host"] = [{"key": prefill_key, "stages": HOST_PARTS}, {"key": decode_key}]
    path.write_text(json.dumps(run))


@pytest.fixture(scope="module")
def served(model_dir, tmp_path_factory):  # noqa: F811
    """One request through the run: its usage, and each prefill stage's cost
    lines, by stage."""
    if not shutil.which("atomesh"):
        pytest.skip("a run needs atomesh on PATH")
    (model_dir / "config.json").write_text(json.dumps(QWEN))
    draft = tmp_path_factory.mktemp("draft")
    (draft / "config.json").write_text(json.dumps(DRAFT))
    root = tmp_path_factory.mktemp("pp_stage_laws")
    spec = f"--method dspark --draft-model {draft} --num-speculative-tokens 3"
    with pytest.MonkeyPatch.context() as mp:
        for module in (test_pd_prefill_pp, test_vertical_slice):
            mp.setattr(module, "MODEL", str(model_dir))
        mp.setattr(test_pd_prefill_pp, "publish_pd_entries", _planted)
        mp.setenv("ATOM_COMPASS_PARITY_RECORD", str(root / "parity"))
        mp.setenv(
            "SERVER_ARGS",
            f"--enforce-eager --max-model-len 2048 {spec} "
            "--spec-decode-acceptance-rate 0.6",
        )
        ran = test_pd_prefill_pp.run_pd_pp(root / "seed1", "1", 2)
    (usage,) = [json.loads(e[6:])["usage"] for _, e in ran["events"] if '"usage"' in e]
    lines = []
    for name in STAGES:
        path = root / "parity" / "prefill" / f"{name}.cost.jsonl"
        lines.append([json.loads(s) for s in path.read_text().splitlines()[1:]])
    return usage, lines


def test_a_pipelined_prefill_is_priced_by_each_stage_s_own_parts(served):
    usage, lines = served
    shown = ("step", "device_s", "draft_s", "E", PP_RECV_SEGMENT, "refused")
    for name, rows in zip(STAGES, lines):
        print(f"\n{name}:")
        for row in rows:
            print(f"  {json.dumps({k: row[k] for k in shown})}")
    assert usage["completion_tokens"] == MAX_TOKENS
    for k, rows in enumerate(lines):
        assert rows and all(row["refused"] is None for row in rows)
        assert {row["device_s"] for row in rows} == {PREFILL_S[k]}
        assert {row["E"] for row in rows} == {E_S[k]}
        assert {row[PP_RECV_SEGMENT] for row in rows} == {RECV_S * k}
    assert {row["draft_s"] for row in lines[0]} == {0.0}
    assert {row["draft_s"] for row in lines[1]} == {DRAFT_S}
