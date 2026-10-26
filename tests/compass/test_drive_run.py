# SPDX-License-Identifier: MIT
"""`compass run`: one simulated instance's server, its coverage check before
the server starts, and what run.json records for grading once it stops.

Steps are ATOM's `Scheduler`'s, recorded as `test_step_grid_plan.record`
records them and laid out as `record` writes a record directory; a population
is published from one by `trace_steps.publish`. Bundles are planned and
measured from those steps and published as `test_step_grid_plan.loaded`
publishes them, and ATOM's arguments resolve that test's deployment. The
server is `test_drive_trace_steps`'s planted one, which also copies a run
summary into its out_dir; the summary after a refused step is the one
`test_refused_step_continues.py` produces.
"""

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_drive_trace_steps
import torch
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_capture_v4_pro import CONFIG as V4_PRO
from test_dp_step_max import runner
from test_drive_runfile import build, price_list_key, published
from test_drive_trace_steps import IDENTITY
from test_memory_readings import ASSUMED, DOCUMENT, READINGS
from test_op_graph import BENT, GEMM, g, listed, pt
from test_op_work_v4 import MORI
from test_refused_step_continues import REFUSED, drive, finish, no_eager_rung
from test_refused_step_continues import run_file as refusing_run_file
from test_run_graph_pool import _installed
from test_run_step_grid import KEY, deployment
from test_step_grid_plan import (
    ENGINE,
    LADDER,
    MIXED,
    SHORT,
    loaded,
    located,
    measured,
    plan,
    record,
    refusing,
)
from test_work import SPEC

from atom.compass import run as compass_run
from atom.compass.artifacts import Key, Kind
from atom.compass.backends import op_graph
from atom.compass.backends.cost import CostTerm
from atom.compass.backends.op_graph import LISTED, OpGraphBackend
from atom.compass.backends.provenance import Method, Provenance
from atom.compass.backends.resolver import CostSource
from atom.compass.backends.step_stub import Coefficients
from atom.compass.backends.work import Work
from atom.compass.capture.forward import structure
from atom.compass.clock import RefusalTally
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve, trace_steps

#: `test_drive_trace_steps`'s planted server, which first copies the summary
#: `PLANTED_SUMMARY` names as client run 0's into the out_dir of the run file it
#: is handed.
PLANTED = (
    "import json, os, shutil, sys\n"
    'run = json.load(open(sys.argv[sys.argv.index("--compass-run") + 1]))\n'
    'first = os.path.join(run["out_dir"], "client-0")\n'
    "os.makedirs(first)\n"
    'shutil.copy(os.environ["PLANTED_SUMMARY"], first)\n'
    + test_drive_trace_steps.PLANTED
)


def recorded(tmp_path, monkeypatch, lengths, **drive):
    """A record directory, as `record` lays one out, of prompts of `lengths`;
    `drive` goes to `record`."""
    directory = tmp_path / ("record-" + "-".join(map(str, lengths)))
    directory.mkdir()
    record(tmp_path, monkeypatch, *lengths, **drive).rename(directory / serve.STEPS)
    (directory / serve.RUN_JSON).write_text(json.dumps(IDENTITY))
    return directory


def bundle(tmp_path, monkeypatch, member, **config):
    """`member`'s token and loaded backend, with ATOM's arguments resolving the
    deployment it was published under, changed by `config`."""
    backend = loaded(tmp_path, member)
    bounds = {k: ENGINE[k] for k in ("max_num_seqs", "max_num_batched_tokens")}
    r = deployment(tmp_path, **{**bounds, **config}, capture_sizes=LADDER)
    monkeypatch.setattr(drive_run, "_config", lambda atom_args: r.config)
    return Key.of(Kind.STEP_GRID_BUNDLE, **KEY).token(), backend


def stub(tmp_path):
    path = tmp_path / "stub.json"
    path.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    return str(path)


def flags(tmp_path, *backend, atom=()):
    """`compass run`'s arguments, the suite's machine and the store with
    `backend`'s flags, out to ``out``, serving the model in `tmp_path`."""
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    common = ["--machine", str(tmp_path / "machine.json")]
    common += ["--store", str(tmp_path / "store"), "--out", str(tmp_path / "out")]
    return [*backend, *common, "--", "--model", str(tmp_path), *atom]


def refused_summary(directory, monkeypatch):
    """The run summary written after a refused step, in `directory`."""
    directory.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, "all_reduce", None)
        run = refusing_run_file(directory, patch)
        r = runner(no_eager_rung())
        drive(r)
        finish(patch, directory, run, r)
    return directory / compass_run.SUMMARY_FILE


def plant(tmp_path, monkeypatch, steps, summary) -> None:
    """Serve from the planted server, which writes the record `steps` and
    client run 0's summary `summary`."""
    (tmp_path / "planted.py").write_text(PLANTED)
    monkeypatch.setattr(serve, "SERVER", (sys.executable, str(tmp_path / "planted.py")))
    monkeypatch.setattr(serve, "identity", lambda *args: IDENTITY)
    monkeypatch.setenv("PLANTED_RECORD", str(steps))
    monkeypatch.setenv("PLANTED_SUMMARY", str(summary))


def unstarted(*args, **kwargs):
    raise AssertionError("a server started")


def test_steps_the_grid_cannot_answer_start_the_run_and_are_counted_by_name(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, MIXED)
    planned = plan(directory / serve.STEPS)
    points = located(tmp_path, planned)
    member = measured(planned, points, drop=("decode", points["decode"][0]))
    token, backend = bundle(tmp_path, monkeypatch, member)
    (tmp_path / "authority").mkdir()
    summary = tmp_path / "authority" / compass_run.SUMMARY_FILE
    summary.write_text(json.dumps({"schedule": {"refusals": RefusalTally().record()}}))
    plant(tmp_path, monkeypatch, directory / serve.STEPS, summary)

    drive_run.main(flags(tmp_path, "--step-grid", token, "--coverage", str(directory)))

    lines = [json.loads(x) for x in (directory / "steps/dp0.jsonl").open()]
    named = [(f"{directory} dp0 step {line['step']}", line) for line in lines]
    declined = refusing(backend, named)
    assert 0 < len(declined) < len(named)
    coverage = json.loads((tmp_path / "out" / serve.RUN_JSON).read_text())["coverage"]
    assert [d["step"] for d in coverage["declined"]] == declined
    assert coverage["fallback"] == len(declined)
    assert coverage["covered"] == len(named) - len(declined)


def test_a_bundle_measured_under_another_engine_config_refuses_before_any_server(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, MIXED)
    planned = plan(directory / serve.STEPS)
    member = measured(planned, located(tmp_path, planned))
    token, _ = bundle(tmp_path, monkeypatch, member, kv_cache_dtype="fp8")
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError, match="kv_cache_dtype 'bf16' in the bundle"):
        drive_run.main(
            flags(tmp_path, "--step-grid", token, "--coverage", str(directory))
        )


def test_an_input_recorded_at_another_token_budget_refuses_naming_that_field(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, MIXED, budget=2048)
    # Planned for this run's 8192 tokens, the bundle covers the input's steps.
    target = {"max_num_batched_tokens": 8192, "max_model_len": 16384}
    planned = plan(directory / serve.STEPS, engine=target)
    member = measured(planned, located(tmp_path, planned))
    token, _ = bundle(tmp_path, monkeypatch, member, max_num_batched_tokens=8192)
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError) as refused:
        drive_run.main(
            flags(tmp_path, "--step-grid", token, "--coverage", str(directory))
        )

    assert str(refused.value) == (
        f"--coverage: {directory} dp0 ran under another engine than ATOM's "
        "arguments resolve: max_num_batched_tokens 2048 there, 8192 here"
    )


def test_a_population_on_another_ladder_and_tp_width_refuses_naming_both(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        trace_steps, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT)
    )
    served = recorded(tmp_path, monkeypatch, SHORT, ladder=[1, 2, 4])
    # A TP-2 rank's engine file, which this suite's runner does not write.
    engine = served / serve.STEPS / "dp0.engine.json"
    stated = json.loads(engine.read_text())
    engine.write_text(json.dumps(stated | {"tensor_parallel_size": 2}))
    run = build(tmp_path, "--step-stub", stub(tmp_path))
    population = trace_steps.publish(served, run, "compass trace steps").token()
    # Planned from the population's steps, the bundle covers them.
    planned = plan(served / serve.STEPS)
    member = measured(planned, located(tmp_path, planned))
    token, _ = bundle(tmp_path, monkeypatch, member)
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError, match="arguments resolve: ladder") as refused:
        drive_run.main(flags(tmp_path, "--step-grid", token, "--coverage", population))

    assert str(refused.value).endswith(
        f"ladder [1, 2, 4] there, {LADDER} here; tensor_parallel_size 2 there, 1 here"
    )


@pytest.mark.parametrize("backend", ["--step-stub", "--step-linear"])
def test_coverage_with_a_backend_that_has_no_check_refuses_naming_it(tmp_path, backend):
    argv = flags(tmp_path, backend, "k", "--coverage", str(tmp_path))

    with pytest.raises(ValueError, match=f"^--coverage: {backend} has no coverage"):
        drive_run.main(argv)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "role, side", [("kv_producer", "prefill"), ("kv_consumer", "decode")]
)
def test_a_prefill_or_a_decode_instance_refuses_before_its_server_starts(
    tmp_path, role, side
):
    kv = json.dumps({"kv_connector": "compass", "kv_role": role})
    argv = flags(tmp_path, "--step-stub", stub(tmp_path))

    with pytest.raises(ValueError, match=f"makes this a {side} instance"):
        drive_run.main([*argv, "--kv-transfer-config", kv])
    assert not (tmp_path / "out").exists()


def test_run_and_trace_steps_refuse_a_decode_instance_with_one_message(tmp_path):
    kv = json.dumps({"kv_connector": "compass", "kv_role": "kv_consumer"})
    atom = ["--kv-transfer-config", kv]
    argv = flags(tmp_path, "--step-stub", stub(tmp_path), atom=atom)

    with pytest.raises(ValueError) as ran:
        drive_run.main(argv)
    with pytest.raises(ValueError) as traced:
        trace_steps.main(argv)
    assert str(traced.value) == str(ran.value)
    assert "makes this a decode instance" in str(ran.value)


def test_run_json_records_the_summarys_refusals_the_roofline_and_the_backend(
    tmp_path, monkeypatch
):
    summary = refused_summary(tmp_path / "authority", monkeypatch)
    directory = recorded(tmp_path, monkeypatch, MIXED)
    plant(tmp_path, monkeypatch, directory / serve.STEPS, summary)

    drive_run.main(flags(tmp_path, "--step-stub", stub(tmp_path), "--step-roofline"))

    out = tmp_path / "out"
    written = json.loads((out / serve.RUN_JSON).read_text())
    handed = json.loads((out / serve.RUN_FILE).read_text())
    refusals = json.loads(summary.read_text())["schedule"]["refusals"]
    assert written["refusals"] == refusals
    assert [reason for reason, _ in refusals["reasons"]] == [REFUSED]
    assert written["run_file"] == handed and handed["out_dir"] == str(out)
    assert written["step_roofline"] is handed["step_roofline"] is True
    assert written["skip"] == [] and written["coverage"] is None
    provenance = handed["backend"] | {"memory_readings_assumed": []}
    assert written["provenance"] == provenance
    assert written["host_key"] == IDENTITY["host_key"]


def test_a_covered_run_starts_and_records_its_check_over_a_record_and_a_population(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        trace_steps, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT)
    )
    directory = recorded(tmp_path, monkeypatch, MIXED)
    served = recorded(tmp_path, monkeypatch, SHORT)
    run = build(tmp_path, "--step-stub", stub(tmp_path))
    population = trace_steps.publish(served, run, "compass trace steps").token()
    planned = plan(directory / serve.STEPS, served / serve.STEPS)
    token, _ = bundle(
        tmp_path, monkeypatch, measured(planned, located(tmp_path, planned))
    )
    (tmp_path / "authority").mkdir()
    summary = tmp_path / "authority" / compass_run.SUMMARY_FILE
    summary.write_text(json.dumps({"schedule": {"refusals": RefusalTally().record()}}))
    plant(tmp_path, monkeypatch, directory / serve.STEPS, summary)

    inputs = [str(directory), population]
    drive_run.main(flags(tmp_path, "--step-grid", token, "--coverage", *inputs))

    out = tmp_path / "out"
    written = json.loads((out / serve.RUN_JSON).read_text())
    coverage = written["coverage"]
    assert coverage["inputs"] == inputs
    assert coverage["covered"] == coverage["steps"] == 15
    assert coverage["declined"] == [] and written["step_roofline"] is False
    handed = json.loads((out / serve.RUN_FILE).read_text())
    provenance = handed["backend"] | {"memory_readings_assumed": []}
    assert written["provenance"] == provenance
    assert handed["step_grid"]["grid"] == KEY["grid"]


#: The planted server, which also hands in the engine's memory as
#: `PLANTED_HAND_IN` holds it.
HANDING_IN = (
    "import json, os, shutil, sys\n"
    'run = json.load(open(sys.argv[sys.argv.index("--compass-run") + 1]))\n'
    'shutil.copy(os.environ["PLANTED_HAND_IN"], run["out_dir"])\n'
) + PLANTED


def test_run_json_names_the_assumed_memory_readings_its_engine_read(
    tmp_path, monkeypatch
):
    # The engine's hand-in, as a worker installed from an assumed entry writes it.
    worker = tmp_path / "worker"
    worker.mkdir()
    readings = dict(READINGS[1], assumed=ASSUMED)
    memory = _installed(monkeypatch, worker, eager=True, readings=readings)
    monkeypatch.delenv(compass_run.ENV)
    run = {"out_dir": str(worker)}
    compass_run._hand_in(run, "engine", [], memory=memory.compass_memory)
    (tmp_path / "authority").mkdir()
    summary = tmp_path / "authority" / compass_run.SUMMARY_FILE
    summary.write_text(json.dumps({"schedule": {"refusals": RefusalTally().record()}}))
    steps = recorded(tmp_path, monkeypatch, SHORT) / serve.STEPS
    plant(tmp_path, monkeypatch, steps, summary)
    (tmp_path / "planted.py").write_text(HANDING_IN)
    hand_in = worker / compass_run.LP_FILE.format("engine")
    monkeypatch.setenv("PLANTED_HAND_IN", str(hand_in))

    drive_run.main(flags(tmp_path, "--step-stub", stub(tmp_path)))

    out = tmp_path / "out"
    written = json.loads((out / serve.RUN_JSON).read_text())
    handed = json.loads((out / serve.RUN_FILE).read_text())
    provenance = handed["backend"] | {"memory_readings_assumed": [ASSUMED]}
    assert written["provenance"] == provenance


#: The planted server, which also writes the arguments it was started with to
#: the file `PLANTED_ARGV` names.
SEEING = (
    "import json, os, sys\n"
    'json.dump(sys.argv[1:], open(os.environ["PLANTED_ARGV"], "w"))\n'
) + PLANTED


def deployed(tmp_path, monkeypatch, summary):
    """A deployment's directory as `deploy.sh` leaves it before its instances
    start: its run file, with both engines' hand-ins beside it, each from an
    entry assumed as its side names; returns the run file's path."""
    d = tmp_path / "deployment"
    d.mkdir()
    run = build(tmp_path, "--step-stub", stub(tmp_path))
    run |= {"out_dir": str(d), "clock_endpoint": "tcp://127.0.0.1:1"}
    (d / serve.RUN_FILE).write_text(json.dumps(run))
    for side in ("P", "D"):
        hand_in = {"refusals": [], "memory": {"assumed": f"assumed on {side}"}}
        (d / compass_run.LP_FILE.format(f"engine-{side}")).write_text(
            json.dumps(hand_in)
        )
    steps = recorded(tmp_path, monkeypatch, SHORT) / serve.STEPS
    plant(tmp_path, monkeypatch, steps, summary)
    (tmp_path / "planted.py").write_text(SEEING)
    monkeypatch.setenv("PLANTED_ARGV", str(tmp_path / "argv.json"))
    return d / serve.RUN_FILE


def test_a_joined_instance_joins_its_deployment_s_authority_and_grades_from_there(
    tmp_path, monkeypatch
):
    summary = refused_summary(tmp_path / "authority", monkeypatch)
    joined = deployed(tmp_path, monkeypatch, summary)
    out = tmp_path / "prefill"
    kv = json.dumps({"kv_connector": "compass", "kv_role": "kv_producer"})

    drive_run.main(
        ["--join", str(joined), "--out", str(out), "--", "--model", str(tmp_path)]
        + ["--kv-transfer-config", kv]
    )

    run = json.loads(joined.read_text())
    seen = json.loads((tmp_path / "argv.json").read_text())
    handed = seen[seen.index("--compass-run") + 1]
    assert handed == str(out / serve.RUN_FILE)
    assert json.loads(Path(handed).read_text()) == run
    assert seen[seen.index("--compass-clock-endpoint") + 1] == run["clock_endpoint"]
    ready = json.loads((out / serve.READY_FILE).read_text())
    assert ready["clock_endpoint"] == run["clock_endpoint"]
    written = json.loads((out / serve.RUN_JSON).read_text())
    refusals = json.loads(summary.read_text())["schedule"]["refusals"]
    assert written["refusals"] == refusals
    assert written["run_file"] == run
    provenance = run["backend"] | {"memory_readings_assumed": ["assumed on P"]}
    assert written["provenance"] == provenance


def test_a_joined_instance_refuses_a_run_file_flag_and_a_run_file_naming_no_authority(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(serve, "serve", unstarted)
    run = build(tmp_path, "--step-stub", stub(tmp_path))
    (tmp_path / "run_file.json").write_text(json.dumps(run))
    join = ["--join", str(tmp_path / "run_file.json"), "--out", str(tmp_path / "out")]
    atom = ["--", "--model", str(tmp_path)]

    with pytest.raises(SystemExit):
        drive_run.main([*join, "--step-roofline", *atom])
    with pytest.raises(ValueError, match="names no clock_endpoint and out_dir"):
        drive_run.main([*join, *atom])
    assert not (tmp_path / "out").exists()


def step_line(*rows, rung=None):
    """A step-record line of `rows`, each ``(query, context, decode)``."""
    return {"rows": [list(r) for r in rows], "rung": rung, "produces_output": True}


def decode_gemm(phase, *shape):
    """A decode step's graph, one GEMM; a prefill has none."""
    return {"seq": [g(2, 64, 64)]} if phase == "decode" else None


def test_op_coverage_counts_covered_steps_by_phase_and_why_each_kind_left_one_out():
    backend = OpGraphBackend(SPEC, decode_gemm, kv="bf16", price_list=BENT)
    steps = [
        step_line(*[(1, 2000, True)] * 2, rung=2),  # on a measured point
        step_line(*[(1, 1500, True)] * 2, rung=2),  # along the line the fit declined
        step_line(*[(1, 1000, True)] * 4, rung=4),  # past the measured widths
        step_line((8, 8, False)),  # no graph at its shape
    ]

    report = drive_run.op_coverage(backend, [(str(i), s) for i, s in enumerate(steps)])

    assert report == {
        "steps": 4,
        "covered": 1,
        "share": 0.25,
        "by phase": {
            "decode": {"steps": 3, "covered": 1},
            "prefill": {"steps": 1, "covered": 0},
        },
        "left out, by leaf kind": {
            GEMM: {
                "the fit declined context at B = #: context = # reads +#% off its "
                "neighbours": 1,
                "B = # is outside the measured # to #": 1,
            },
            "step refused": {
                "no op_graph holds a prefill step of # tokens over # sequences; "
                "compass trace ops captures one": 1
            },
        },
    }


def test_op_coverage_counts_a_step_once_per_kind_however_many_whys_it_has():
    def two_gemms(phase, *shape):
        return {"seq": [g(2, 64, 64), g(2, 128, 64)]}

    backend = OpGraphBackend(SPEC, two_gemms, kv="bf16", price_list=BENT)
    step = step_line(*[(1, 1500, True)] * 2, rung=2)

    report = drive_run.op_coverage(backend, [("0", step)])

    assert report["left out, by leaf kind"] == {
        GEMM: {"no point of its signature was measured": 1}
    }


def op_graph_run(tmp_path, monkeypatch, steps, price_list):
    """Serve from the planted server, which writes the record `steps`, with
    ATOM's arguments resolving the deployment `bundle` does and the store
    holding `price_list`'s params at its run's key when one is given, each
    graph `decode_gemm`."""
    bounds = {k: ENGINE[k] for k in ("max_num_seqs", "max_num_batched_tokens")}
    r = deployment(tmp_path, **bounds, capture_sizes=LADDER)
    monkeypatch.setattr(drive_run, "_config", lambda atom_args: r.config)
    monkeypatch.setattr(compass_run, "_op_graphs", lambda *args: decode_gemm)
    if price_list:
        key = price_list_key(tmp_path)
        published(tmp_path, r, Kind.PRICE_LIST, key, price_list.params)
    (tmp_path / "authority").mkdir()
    summary = tmp_path / "authority" / compass_run.SUMMARY_FILE
    summary.write_text(json.dumps({"schedule": {"refusals": RefusalTally().record()}}))
    plant(tmp_path, monkeypatch, steps, summary)


def test_an_op_graph_run_records_its_price_list_s_coverage_in_run_json(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, MIXED)
    lines = [json.loads(x) for x in (directory / "steps/dp0.jsonl").open()]
    lines = [line for line in lines if not line["is_dummy_run"]]
    decode = [line for line in lines if all(row[2] for row in line["rows"])]
    # The GEMM measured at each decode step's width and longest row.
    widths = {
        (line["rung"] or len(line["rows"]), max(row[1] for row in line["rows"]))
        for line in decode
    }
    points = [pt(g(2, 64, 64), f"decode B={b}", 1e3, c) for b, c in widths]
    op_graph_run(tmp_path, monkeypatch, directory / serve.STEPS, listed(*points))

    drive_run.main(flags(tmp_path, "--op-graph", "--coverage", str(directory)))

    coverage = json.loads((tmp_path / "out" / serve.RUN_JSON).read_text())["coverage"]
    left_out = coverage.pop("left out, by leaf kind")
    prefill = len(lines) - len(decode)
    assert 0 < len(decode) < len(lines)
    assert coverage == {
        "inputs": [str(directory)],
        "steps": len(lines),
        "covered": len(decode),
        "share": len(decode) / len(lines),
        "by phase": {
            "decode": {"steps": len(decode), "covered": len(decode)},
            "prefill": {"steps": prefill, "covered": 0},
        },
    }
    assert left_out.keys() == {"step refused"}
    assert sum(left_out["step refused"].values()) == prefill


def test_coverage_of_an_op_graph_run_with_no_price_list_refuses_before_any_server(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, SHORT)
    op_graph_run(tmp_path, monkeypatch, directory / serve.STEPS, None)
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError, match="^--coverage: the store holds no price_list"):
        drive_run.main(flags(tmp_path, "--op-graph", "--coverage", str(directory)))


class Asked(CostSource):
    """A price list that prices every leaf, keeping the kinds it was asked for."""

    name = LISTED

    def __init__(self):
        self.kinds = set()

    def price(self, request):
        self.kinds.add(kind := request[0]["op"])
        return CostTerm(f"op.{kind}", 1e-6, Provenance(Method.MEASURED))


def test_coverage_of_a_v4_pro_ep8_run_asks_its_price_list_at_atom_s_ep_width(
    tmp_path, monkeypatch
):
    # The nightly DPA-TP8 EP8 deployment, its server's own flags among ATOM's
    # arguments. ATOM's MoE spans its eight DP ranks, as `compass run` reads
    # off the model it builds.
    (tmp_path / "v4").mkdir()
    (tmp_path / "v4" / "config.json").write_bytes(V4_PRO.read_bytes())
    atom = ["--model", str(tmp_path / "v4"), "-tp", "8", "--kv_cache_dtype", "fp8"]
    atom += ["--enable-dp-attention", "--enable-expert-parallel"]
    atom += ["--server-port", "8000"]
    directory = recorded(tmp_path, monkeypatch, SHORT)
    bounds = {k: ENGINE[k] for k in ("max_num_seqs", "max_num_batched_tokens")}
    r = deployment(
        tmp_path, **bounds, capture_sizes=LADDER, enable_expert_parallel=True
    )
    monkeypatch.setattr(drive_run, "_config", lambda atom_args: r.config)
    kinds = ("aiter.fused_moe_.default", MORI + "dispatch", MORI + "combine")
    region = {"seq": [{"op": kind, "args": {}} for kind in kinds]}
    monkeypatch.setattr(compass_run, "_op_graphs", lambda *args: lambda *s: region)
    asked = Asked()
    monkeypatch.setattr(compass_run, "_price_list", lambda *args: asked)
    widths = set()

    def work(leaf, view, tp, kv, ep):
        widths.add((leaf["op"], ep))
        return Work()

    monkeypatch.setattr(op_graph, "work", work)
    run = {"store": str(tmp_path / "store"), "machine": DOCUMENT}
    run |= {compass_run.OP_GRAPH: True, compass_run.PRICE_LIST: {}}

    report = drive_run.check([str(directory)], run, atom)

    assert report["covered"] == report["steps"] > 0
    assert asked.kinds == set(kinds)
    assert widths == {(kind, 8) for kind in kinds}


@pytest.mark.parametrize("layout", [["-tp", "8"], ["-tp", "4", "-dp", "2"]])
def test_coverage_of_a_v4_pro_dpa_run_reads_graphs_at_the_widths_its_ranks_run(
    tmp_path, monkeypatch, layout
):
    # The nightly DPA-TP8 EP8 deployment, and a DP 2 x TP 4 one of the same
    # eight ranks. Its record states each rank's TP 1, as `CoreManager`
    # rewrites it; ATOM's MoE spans all eight.
    (tmp_path / "v4").mkdir()
    (tmp_path / "v4" / "config.json").write_bytes(V4_PRO.read_bytes())
    atom = ["--model", str(tmp_path / "v4"), *layout, "--kv_cache_dtype", "fp8"]
    atom += ["--index-cache-dtype", "fp4"]  # what V4 resolves on gfx950
    atom += ["--enable-dp-attention", "--enable-expert-parallel"]
    atom += ["--max-num-seqs", "8", "--max-num-batched-tokens", "64"]
    atom += ["--max-model-len", "2048", "--cudagraph-capture-sizes", str(LADDER)]
    directory = recorded(tmp_path, monkeypatch, SHORT)
    # Reading the quantization imports aiter, which a process with no driver
    # cannot do; the build that reads the EP width does it in its own process.
    quant = SimpleNamespace(compute_hash=lambda: "fp8", exclude_layers=[])
    monkeypatch.setattr("atom.config.QuantizationConfig", lambda *args: quant)
    kinds = ("aiter.fused_moe_.default", MORI + "dispatch", MORI + "combine")
    region = {"seq": [{"op": kind, "args": {}} for kind in kinds]}
    structures = []

    def graphs(root, config, arch):
        structures.append(structure(config, "decode", True, arch))
        return lambda *shape: region

    monkeypatch.setattr(compass_run, "_op_graphs", graphs)
    asked = Asked()
    monkeypatch.setattr(compass_run, "_price_list", lambda *args: asked)
    widths = set()

    def work(leaf, view, tp, kv, ep):
        widths.add((leaf["op"], tp, ep))
        return Work()

    monkeypatch.setattr(op_graph, "work", work)
    run = {"store": str(tmp_path / "store"), "machine": DOCUMENT}
    run |= {compass_run.OP_GRAPH: True, compass_run.PRICE_LIST: {}}

    report = drive_run.check([str(directory)], run, atom)

    assert report["covered"] == report["steps"] > 0
    assert asked.kinds == set(kinds)
    assert widths == {(kind, 1, 8) for kind in kinds}
    assert [s.endswith(" tp=1 dp=8 ep=8 decode eager") for s in structures] == [True]
