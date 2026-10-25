# SPDX-License-Identifier: MIT
"""A simulated 1P1D run whose prefill side is a pipeline.

The run file's ``prefill_pipeline_parallel_size`` makes each prefill stage the
engine LP ``engine-P-pp<k>``. Decode sends its write request to the head,
which prices every stage's write: decode is ready at ``max_k c_k + L`` and
the head is done at ``max_k c_k + 2L``, after Mooncake's release.

The run tests start `pd_sim.sh` at prefill ``-pp 2`` and at ``-pp 1`` and
skip as `test_pd_slice`'s do; the rest run anywhere.
"""

import dataclasses
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_kv_remote_prefill
from test_dp_slice import DP_MACHINE, _deployment
from test_kv_remote_prefill import (
    SEND,
    L,
    R,
    Script,
    admitted,
    connector,
    engine_step,
    free_blocks,
    prefilled,
    scheduler_with,
)
from test_kv_simulated_connector import PEAKS, TICK, geometry, model_for  # noqa: F401
from test_pd_slice import NEEDS_A_PD_RUN, PD, PORTS, PdTraffic, publish_pd_entries
from test_pp_kv_status import _head
from test_vertical_slice import CONSTANT, MAX_TOKENS, MODEL, TREE, WALL_S, _run_file

from atom.compass import run as compass_run
from atom.compass.backends import KvGeometry
from atom.compass.clock import LpId, prefill_decode_table
from atom.compass.detect.determinism import StepTable
from atom.compass.detect.scenario import check_run, pipeline_timeline
from atom.compass.drive.serve import _free_ports
from atom.compass.kv import Scope, TransferModel
from atom.compass.runner.pp_send import StageSends
from atom.compass.spec import MachineSpec
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.models.utils import get_pp_indices
from atom.utils.clock import LPRuntime

KV = "engine-D->engine-P-pp0:kv_write_req"


def _run(tmp_path, pp: int, **more) -> dict:
    run = json.loads(_run_file(tmp_path, machine=DP_MACHINE, **PD, **more).read_text())
    return run | ({compass_run.PREFILL_PP: pp} if pp > 1 else {})


# --- without a driver ---------------------------------------------------------


def test_the_pp_head_hands_the_scheduler_every_poll_empty_included():
    proc = _head(pp_size=2, local_outputs=[KVConnectorOutput()])
    proc._poll_kv_transfer_progress()
    assert [out.is_empty() for out in proc.scheduler.outputs] == [True]


@pytest.mark.parametrize("pp", [2, 4])
def test_each_prefill_stage_is_an_engine_lp_and_decode_writes_to_the_head(tmp_path, pp):
    run = _run(tmp_path, pp)
    table = compass_run.channel_table(run)
    stages = [f"engine-P-pp{k}" for k in range(pp)]
    servers = ["frontend-P", "frontend-D", "engine-D", *stages]
    assert {lp.name for lp in table.registry.ids()} == {"traffic", *servers}
    assert sorted(compass_run._hand_ins(run)) == sorted(servers)
    names = {c.name for lp in table.registry.ids() for c in table.channels_from(lp)}
    assert {n for n in names if ":kv_write_req" in n} == {KV}
    assert f"{stages[-1]}->engine-P-pp0:tokens" in names
    for k, stage in enumerate(stages):
        assert f"frontend-P->{stage}:request#dp{k}" in names
        if k:
            assert f"engine-P-pp0->{stage}:meta#pp{k}" in names
            assert f"{stages[k - 1]}->{stage}:pp_data#dp0" in names
            assert f"{stage}->{stages[k - 1]}:pp_ack#dp0" in names
    assert table.distance(LpId("engine-D"), LpId(stages[0])) == PD["kv_write_req_s"]


@pytest.mark.parametrize(
    "more, refusal",
    [
        ({"pipeline_parallel_size": 2}, "names pipeline_parallel_size"),
        ({"prefill_data_parallel_size": 2}, "at DP width 2: ATOM runs no pipeline"),
        ({"lmcache": {}}, "lmcache_mp does not run under PP"),
    ],
)
def test_what_a_prefill_pipeline_does_not_run_is_refused_by_name(
    tmp_path, more, refusal
):
    with pytest.raises(ValueError, match=refusal):
        compass_run.channel_table(_run(tmp_path, 2, **more))


def test_a_prefill_width_in_a_run_with_no_sides_is_refused_by_name(tmp_path):
    run = json.loads(_run_file(tmp_path, **{compass_run.PREFILL_PP: 2}).read_text())
    with pytest.raises(ValueError, match=f"names {compass_run.PREFILL_PP}"):
        compass_run.channel_table(run)


@pytest.mark.parametrize(
    "role, pp, refusal",
    [
        ("kv_consumer", 2, "a decode side of 2 pipeline stages"),
        ("kv_producer", 4, f"run file's {compass_run.PREFILL_PP} is 2"),
    ],
)
def test_each_side_is_held_to_its_pipeline_width(
    monkeypatch, tmp_path, role, pp, refusal
):
    deployment = _deployment()
    deployment.pipeline_parallel_size = pp
    deployment.kv_transfer_config = {"kv_connector": "compass", "kv_role": role}
    run = tmp_path / "pp.json"
    run.write_text(json.dumps(_run(tmp_path, 2)))
    monkeypatch.setenv(compass_run.CLOCK_ENV, "tcp://127.0.0.1:1")
    monkeypatch.setenv(compass_run.ENV, str(run))
    with pytest.raises(ValueError, match=refusal):
        compass_run.frontend(deployment)


def test_a_stage_sends_on_its_own_pipelines_channels(tmp_path):
    table = compass_run.channel_table(_run(tmp_path, 2))
    sends = StageSends(LPRuntime(LpId("engine-P-pp1"), table, None), None, 1, 2, {})
    assert sends.data_in == "engine-P-pp0->engine-P-pp1:pp_data#dp0"
    assert sends.ack_out == "engine-P-pp1->engine-P-pp0:pp_ack#dp0"


def test_the_pipeline_check_reads_a_prefill_pipeline(tmp_path):
    run = _run(tmp_path, 2)
    table = StepTable(json.dumps(run))
    table.record("engine-P-pp0", 1.0, 1.0, "TAR")
    meta = "engine-P-pp0->engine-P-pp1:meta#pp1"
    table.record("engine-P-pp1", 0.5, 1.0 + run["ipc_s"], "release", meta, 0)
    data = "engine-P-pp0->engine-P-pp1:pp_data#dp0"
    assert pipeline_timeline(table.text()) == [
        f"engine-P-pp0 never sent forward 0's output on {data}"
    ]


def _pd_pp(geometry, monkeypatch, pp: int):  # noqa: F811
    """A prefill head of `pp` stages, the first stage's write the largest, and a
    decode engine, on one write-request pair."""
    base = model_for(geometry, PEAKS[0])
    models = tuple(
        dataclasses.replace(base, bytes_per_block=base.bytes_per_block * (pp - k))
        for k in range(pp)
    )
    address = f"inproc://kv-pp{pp}-{id(models)}"
    monkeypatch.setattr(test_kv_remote_prefill, "KV", KV)
    table = prefill_decode_table(
        admission_path="serving",
        ipc_s=L,
        stream_s=L,
        router_s=L,
        kv_write_req_s=L,
        prefill_pp=pp,
    )
    rt_p = LPRuntime(LpId("engine-P-pp0"), table, Script())
    rt_d = LPRuntime(LpId("engine-D"), table, Script())
    rt_p.start_run()
    rt_d.start_run()
    producer = {"kv_role": "kv_producer", "pipeline_parallel_size": pp}
    p = connector(rt_p, monkeypatch, models, address=address, **producer)
    d = connector(rt_d, monkeypatch, models, kv_role="kv_consumer", address=address)
    stage = SimpleNamespace(data_parallel_rank=0, pipeline_parallel_rank=1)
    rt_1 = LPRuntime(LpId("engine-P-pp1"), table, Script())
    inert = connector(rt_1, monkeypatch, models, parallel_config=stage, **producer)
    assert inert._socks == {} and not inert.has_pending_work()
    return SimpleNamespace(
        models=models, rt_p=rt_p, rt_d=rt_d, p=scheduler_with(p), d=scheduler_with(d)
    )


def _after_step(engine, rt, t, holds) -> bool:
    engine_step(engine, rt, t)
    engine.schedule()
    return holds()


@pytest.mark.parametrize("pp", [2, 4])
def test_decode_is_ready_after_the_latest_stage_write_and_the_head_frees_after_the_release(
    geometry, monkeypatch, seq_factory, pp  # noqa: F811
):
    pd = _pd_pp(geometry, monkeypatch, pp)
    total = free_blocks(pd.p)
    (x,) = prefilled(pd, 1, R)
    assert x.kv_transfer_params_output["remote_pp_size"] == pp
    (y,) = admitted(pd, [x], seq_factory, SEND)
    a = SEND + L
    c = a + pd.models[0].duration_s(len(y.block_table))

    engine_step(pd.p, pd.rt_p, a, (0, a))
    parked = lambda: x.id in pd.p.deferred_free_blocks
    times = (c + 2 * L - TICK, c + 2 * L, c + 2 * L + TICK)
    assert [_after_step(pd.p, pd.rt_p, t, parked) for t in times] == [1, 1, 0]
    assert free_blocks(pd.p) == total
    waiting = lambda: y.status.name == "WAITING_FOR_REMOTE_KVS"
    times = (c + L - TICK, c + L, c + L + TICK)
    assert [_after_step(pd.d, pd.rt_d, t, waiting) for t in times] == [1, 1, 0]


def test_a_request_prefilled_at_another_pipeline_width_is_refused_by_name(
    geometry, monkeypatch, seq_factory  # noqa: F811
):
    pd = _pd_pp(geometry, monkeypatch, 2)
    (x,) = prefilled(pd, 1, R)
    x.kv_transfer_params_output["remote_pp_size"] = 4
    with pytest.raises(ValueError, match="over 4 pipeline stages, and this conn"):
        admitted(pd, [x], seq_factory, SEND)


# --- with a driver ------------------------------------------------------------


def run_pd_pp(out: Path, seed: str, pp: int) -> dict:
    """One run under `pd_sim.sh` with a prefill side of `pp` stages."""
    out.mkdir()
    ports = dict(zip(PORTS, _free_ports(len(PORTS))))
    more = {compass_run.PREFILL_PP: pp} if pp > 1 else {}
    run_file = _run_file(
        out,
        clock_endpoint=f"tcp://127.0.0.1:{ports['CLOCK_PORT']}",
        step_stub=CONSTANT,
        machine=DP_MACHINE,
        **PD,
        **more,
    )
    env = dict(
        os.environ,
        PYTHONHASHSEED=seed,
        PYTHONPATH=str(TREE),
        AITER_LOG_LEVEL="WARNING",
        MODEL=MODEL,
        PREFILL_ARGS=f"-pp {pp}" if pp > 1 else "",
        **{name: str(port) for name, port in ports.items()},
    )
    publish_pd_entries(out / "store", DP_MACHINE, env)
    log = out / "pd_sim.log"
    with open(log, "w") as f:
        sim = subprocess.Popen(
            ["bash", str(TREE / "scripts/compass/pd_sim.sh"), str(run_file)],
            cwd=TREE,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    watchdog = threading.Timer(WALL_S, os.killpg, (sim.pid, signal.SIGKILL))
    watchdog.start()
    try:
        deadline = time.monotonic() + WALL_S
        while "pd_sim: ready" not in log.read_text():
            assert sim.poll() is None, f"pd_sim.sh exited:\n{log.read_text()}"
            assert time.monotonic() < deadline, f"no router in {WALL_S} wall seconds"
            time.sleep(0.5)
        run = json.loads(run_file.read_text())
        traffic = PdTraffic(run, ports["ROUTER_PORT"], sim, ports["PREFILL_PORT"])
        traffic.run()
        code = sim.wait(timeout=120)
    finally:
        watchdog.cancel()
        if sim.poll() is None:
            os.killpg(sim.pid, signal.SIGKILL)
    assert code == 0, f"pd_sim.sh exited {code}:\n{log.read_text()[-4000:]}"
    table = (out / compass_run.STEP_TABLE_FILE).read_text()
    return {"out": out, "run": run, "table": table, "events": traffic.events}


@pytest.fixture(scope="module", params=[1, 2], ids=["pp1", "pp2"])
def pipeline(request, tmp_path_factory):
    pp, root = request.param, tmp_path_factory.mktemp(f"pp{request.param}")
    return pp, [run_pd_pp(root / f"seed{s}", s, pp) for s in ("1", "2")]


@NEEDS_A_PD_RUN
def test_a_prefill_pipeline_meets_every_scenario_check(pipeline):
    _, runs = pipeline
    print(f"\n{runs[0]['table']}")
    taken = [line for _, line in runs[0]["events"]]
    failed = check_run([r["out"] for r in runs], [MAX_TOKENS], taken)
    assert not failed, "\n".join(failed)


def _write_end(run: dict, a: float, prompt_tokens: int, pp: int) -> list[float]:
    """Each prefill stage's write-done of a request whose write request arrives
    at `a`, by hand: its first write on its link, one block per 16 tokens."""
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(MODEL)
    blocks, spec, ends = (
        -(-prompt_tokens // 16),
        MachineSpec.from_mapping(run["machine"]),
        [],
    )
    for k in range(pp):
        span = get_pp_indices(hf.num_hidden_layers, k, pp)
        block = KvGeometry.from_hf_config(
            hf, block_size=16, layer_range=span, kv_dtype="bf16"
        )
        m = TransferModel.from_spec(spec, block, Scope(run["kv_link"]))
        moved = blocks * block.bytes_per_block / m.bandwidth_bytes_per_s
        ends.append(a + m.setup_s + m.latency_s + moved)
    return ends


@NEEDS_A_PD_RUN
def test_the_write_ends_where_both_ends_compute_it(pipeline):
    """Decode's first forward starts at the first or second poll at or after
    ``max_k c_k + L``; the head frees the request's blocks, and so leaves its
    loop for good, at the first or second poll at or after ``max_k c_k + 2L``,
    or ``c`` at one stage."""
    pp, runs = pipeline
    rows = [r.split() for r in runs[0]["table"].splitlines()[1:]]
    head, kv = ("engine-P-pp0", KV) if pp > 1 else ("engine-P", KV.replace("-pp0", ""))
    (write,) = [r for r in rows if r[3] == "release" and r[4] == kv]
    a, L = float(write[2]), PD["kv_write_req_s"]
    (usage,) = [
        json.loads(e[6:])["usage"] for _, e in runs[0]["events"] if '"usage"' in e
    ]
    c = max(_write_end(runs[0]["run"], a, usage["prompt_tokens"], pp))
    ready, done = c + L, c + (2 * L if pp > 1 else 0.0)

    def polls(lp, start, until):
        return {
            float(r[2]) for r in rows if r[0] == lp and start <= float(r[2]) <= until
        }

    forward = min(float(r[1]) for r in rows if r[0] == "engine-D" and r[3] == "TAR")
    (freed,) = [float(r[1]) for r in rows if r[0] == head and r[2] == "inf"]
    print(
        f"\npp{pp}: write request at {a!r}, write end {c!r}, decode ready {ready!r} "
        f"first forward {forward!r}, prefill done {done!r} freed {freed!r}"
    )
    assert 1 <= len(polls("engine-D", ready, forward)) <= 2
    assert 1 <= len(polls(head, done, freed)) <= 2
