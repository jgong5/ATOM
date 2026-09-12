"""What the cc-traces validator refuses, and why each refusal exists.

Every case here is a cell that would otherwise read as a result: the artifacts
are present, the client reported no failure, and the numbers would print. The
validator's job is to be the thing that says which of them are the experiment
`CC_TRACES_PROTOCOL.md` registers, so each test builds a passing cell and breaks
exactly one property of it.

The behaviour under test is the verdict, not the text of the message.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate = _load("cc_traces_validate")


def _supplied(seconds: float, *, source: str = "test-fixture", within=None) -> dict:
    """A `compass.costs/3` supplied term.

    Supplied durations are read from somewhere else, so the record carries the
    artifact they came from and the measured window they happen inside. The
    fixtures default to a term that overlaps nothing, because most of these
    tests are about the ratio rather than the containment.
    """
    return {"seconds": float(seconds), "source": source, "within": within}


ROWS = [
    {"arrival_s": 0.0, "input_tokens": 512, "output_tokens": 8},
    {"arrival_s": 1.0, "input_tokens": 1024, "output_tokens": 16},
    {"arrival_s": 2.5, "input_tokens": 2048, "output_tokens": 32},
]

PRICES_SHA = "a" * 64
SWEEP_SHA = "e" * 64
CODE_SHA = "f" * 64
#: The replay target the deployment was sized from. Its own digest, because
#: what sized a deployment is declared and checked like any other measured
#: input -- see `check_capacity_provenance`.
TARGET_SHA = "b" * 64
TARGET_CAPTURE_SHA = "c" * 64


#: The preset the passing cell prices its regions from. A real built-in, so
#: the snapshot in the record and the declaration in the registry are both
#: taken from the same code a served run would select.
CELL_REGIONS = "source-27b-tp1-conc-v2"


def region_snapshot_of(name):
    """The real snapshot of a built-in preset, as a run publishes it."""
    from atom.compass.runtime.source_oracle import region_snapshot

    from atom.compass.core.cost.regions import region_model

    return region_snapshot(name, region_model(name))


def region_artifact(name=CELL_REGIONS):
    """Declare the preset a run selected, by the digest over its own values.

    No `contents`: a region preset is code, snapshotted where it was selected,
    and an entry claiming files for it would describe bytes that never
    existed.
    """
    return {
        "sha256": region_snapshot_of(name)["sha256"],
        "kind": "region_model",
        "measured_at_tp": 1,
        "produced_by": "regions.py",
        "workload_sha256": None,
        "sources": [{"path": "/m/cap_subspan.json", "sha256": SWEEP_SHA}],
        "code": {"atom/compass/core/cost/regions.py": CODE_SHA},
    }


def capacity_artifact():
    """The declaration of what sized the deployment.

    Kept beside whatever a test is varying, because what sized the deployment
    is checked like any other measured input and almost no test here is about
    it. A test that *is* about it overrides this entry by name.
    """
    return {
        "sha256": TARGET_SHA,
        "kind": "derived_graph",
        "measured_at_tp": 1,
        "produced_by": "replay_target_out",
        "workload_sha256": None,
        # Enumerated, like every other artifact the server reads. The file is
        # named by its basename, which is what the validator compares against
        # what the rank reports having opened.
        "contents": {"target.json": TARGET_SHA},
        "sources": [
            {"path": "/m/target_capture.json", "sha256": TARGET_CAPTURE_SHA}
        ],
        "code": {"atom/compass/replay/runner.py": CODE_SHA},
    }
TABLE_SHA = "b" * 64
WORKLOAD_ROW_KEYS = ("arrival_s", "input_tokens", "output_tokens")

#: What the passing cell's modelled side told the acceptance factory: the whole
#: model, priced completely, at the width the cell is for. Anything less is a
#: different experiment, so the fixture that is supposed to pass states it all.
CELL_FACTORY_OPTIONS = {
    "tp": 2,
    "require_complete": "true",
    "head": "true",
    "regions": CELL_REGIONS,
    "derive": "true",
    "model": "Qwen/Qwen3.8-27B",
    "block_size": 16,
    "max_model_len": 262144,
    "price": "/x/prices.json",
}


PROC_HOST = "cell-host"
PROC_BOOT = "8b0a7a2c-5d31-4f6e-9a11-77c0d2e4b900"
KEEP = object()


def _identity(pid=4242, *, ppid=4240, ticks=132307571, host=PROC_HOST, boot=PROC_BOOT):
    """What a server says about itself, in the shape `/proc` gives it."""
    return {
        "pid": pid,
        "ppid": ppid,
        "host": host,
        "boot_id": boot,
        "start_ticks": ticks,
        "ticks_per_second": 100,
    }


def _device_reading(when, *, nodes=None, handles=(), process=None):
    """One reading as the predictor's own process takes it about itself."""
    return {
        "when": when,
        "process": process or _identity(pid=4243, ppid=4242),
        "namespaces": {"mnt": "mnt:[4026532281]", "pid": "pid:[4026532282]",
                       "net": "net:[4026531840]", "user": "user:[4026531837]",
                       "cgroup": "cgroup:[4026531835]"},
        "device_cgroup": ["0::/"],
        "device_nodes": nodes or {n: False for n in validate.DEVICE_NODES},
        "own_driver_handles": list(handles),
        "device_free": not any((nodes or {}).values()) and not handles,
        # Recorded and not counted. A replay interpreter answers hardware
        # queries from the captured target, so this describes the deployment
        # being modelled. The fixture carries a nonzero count on purpose, to
        # pin that it decides nothing.
        "reported_by_runtime": {"device_count": 4, "cuda_available": True,
                                "bootstrap_installed": True},
        "runtime_note": "recorded, not counted",
    }


def _device_freedom(**over):
    """Both readings, from one process, as the manifest carries them."""
    return {"launch": _device_reading("launch", **over),
            "readback": _device_reading("readback", **over)}


def _server(
    tp,
    *,
    mode,
    virtual,
    options=None,
    digests=None,
    files=None,
    process=KEEP,
    oracle="Oracle",
):
    served = _identity() if process is KEEP else process
    provenance = {
        "server_revision": "abc123",
        "server_code_sha256": "c" * 64,
        "model": "Qwen/Qwen3.8-27B",
        "model_revision": "rev1",
        "tensor_parallel_size": tp,
        "max_model_len": 262144,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 32,
        "compass": {
            "enabled": True,
            "mode": mode,
            "oracle": oracle,
            "oracle_options": options or {},
            "oracle_option_sha256": digests or {},
            "oracle_option_files": files or {},
            "virtual_clock": virtual,
            "admission_seconds": 0.0,
            # What the ranks recorded as they read, and what the predicting
            # process observed about itself. Only the modelled side is held to
            # the device reading; the real side carries one because the same
            # runner produces both records.
            "loaded_inputs": {
                "ranks": [
                    {
                        "rank_coords": {},
                        "inputs": [
                            {
                                "role": "runtime.replay_target",
                                "requested": "/x/target.json",
                                "path": "/x/target.json",
                                "rank_own": False,
                                "sha256": TARGET_SHA,
                                "size": 128,
                                "rank_coords": {},
                            }
                        ],
                        "rolled_sha256": "0" * 64,
                        # The record the capacity selector publishes: the kind
                        # it chose, whether the engine ran on it, what it
                        # refers to and the lineage behind it. The real side
                        # is sized by the device it ran on; the modelled side
                        # is sized from the capture of one, which is the whole
                        # capability.
                        "budget_source": {
                            "kind": "device-measured"
                            if mode == "measure"
                            else "captured",
                            "served": True,
                            "hardware_reference": "MI308X",
                            "lineage": ["/x/target.json"],
                            "deployment": {"num_kvcache_blocks": 4096},
                        },
                        # The coefficients this run priced preparation and
                        # postprocess from, snapshotted by value where they
                        # were selected. From the real preset, so the record
                        # and the declaration are both the code a served run
                        # would have selected.
                        "regions": region_snapshot_of(CELL_REGIONS),
                        "device_freedom": _device_freedom(),
                    }
                ]
            },
        },
    }
    if served is not None:
        provenance["server_process"] = served
    return provenance


def _journal(cell_dir, side, repeats, *, executions=None, seconds=10.0):
    """The run journal `cc_traces_run.py` leaves beside the artifacts.

    Each execution carries its repeat number and the replay's own stopwatch,
    the way `SideRun` writes them: those are what a cost record is bound to.
    """
    if executions is None:
        executions = [
            {
                "execution_id": f"cx-{side}{index}",
                "purpose": "acceptance",
                "repeat": index + 1,
                "replay": {"seconds": seconds},
                "server_process": {
                    "said": _identity(),
                    "observed": {
                        "launched_pid": 4240,
                        "host": PROC_HOST,
                        "boot_id": PROC_BOOT,
                        "start_ticks": 132307571,
                        "ancestry": [4242, 4240, 1],
                        "alive_at_provenance": True,
                    },
                    "verified": True,
                },
                # What the harness read out of `/proc` about the process that
                # predicted. Its own reading, not the worker's: the worker's
                # account lives in the run manifest, and the two agreeing is
                # the whole point of taking both.
                "predictor_process": {
                    "observed": [
                        {
                            "rank": 0,
                            "said_pid": 4243,
                            "launched_pid": 4240,
                            "host": PROC_HOST,
                            "boot_id": PROC_BOOT,
                            "start_ticks": 132307571,
                            "ancestry": [4243, 4242, 4240, 1],
                        }
                    ],
                    "verified": True,
                },
            }
            for index in range(repeats)
        ]
    blob = {
        "schema": "compass.execution/1",
        "cell": str(cell_dir),
        "side": side,
        "purpose": "acceptance",
        "executions": executions,
        "refused": False,
        "failures": [],
        "steps": [],
    }
    path = Path(cell_dir) / f"run.{side}.json"
    path.write_text(json.dumps(blob))
    return path, blob


def _side(
    tmp_path,
    name,
    rows,
    *,
    tp,
    paced,
    mode,
    virtual,
    origin=0.0,
    drift=0.0,
    prepare=True,
    options=None,
    digests=None,
    files=None,
    completion=None,
    prompt=None,
    ttft=0.5,
    service=1.0,
    records=None,
    process=KEEP,
    oracle="Oracle",
):
    """One saved run, in the shape `replay.py` writes."""
    results, engine_records = [], []
    for i, row in enumerate(rows):
        arrive = origin + row["arrival_s"] + (drift if i == len(rows) - 1 else 0.0)
        first = arrive + ttft
        finish = first + service
        produced = row["output_tokens"] if completion is None else completion
        results.append(
            {
                "index": i,
                "ok": True,
                "response": {
                    "id": f"req-{i}",
                    "usage": {
                        "prompt_tokens": (
                            row["input_tokens"] if prompt is None else prompt
                        ),
                        "completion_tokens": produced,
                    },
                },
            }
        )
        engine_records.append(
            {
                "request_id": f"req-{i}",
                "arrive_time": arrive,
                "first_token_time": first,
                "finish_time": finish,
                "ttft": first - arrive,
                "latency": finish - arrive,
            }
        )
    manifest = {
        "paced": paced,
        "time_scale": 1.0,
        "requests": len(rows),
        "trace_sha256": "d" * 64,
        "model": "Qwen/Qwen3.8-27B",
        "failed": 0,
        "prompt_lengths": "passed",
        "server_revision": "abc123",
        "server_code_sha256": "c" * 64,
        "model_revision": "rev1",
        # What the engine answered when the run was drained. A predictor
        # answers False -- its barrier was reached and held. A wall-clock
        # server has no barrier to reach, and answers nothing, which is why
        # the passing fixture leaves the real side unread.
        "arrival_barrier_timed_out": None if paced else False,
        "arrival_barrier": (
            {"timed_out": None, "why": "the barrier was never reached"}
            if paced
            else {"timed_out": False, "ranks": [{"timed_out": False}]}
        ),
        "server": _server(
            tp,
            mode=mode,
            virtual=virtual,
            options=options,
            digests=digests,
            files=files,
            process=process,
            oracle=oracle,
        ),
        "prepare": (
            {
                "requested": 3,
                "returned": 3,
                "drained": True,
                "store_empty_after_drain": True,
                "boundary_engine_time": max(0.0, origin - 1.0),
            }
            if prepare
            else None
        ),
    }
    blob = {
        "run": manifest,
        # The stamp the harness puts inside every artifact. An acceptance cell
        # has to state its purpose, so the passing fixture states it.
        "execution": {"purpose": "acceptance"},
        "workload": [{k: r[k] for k in WORKLOAD_ROW_KEYS} for r in rows],
        "results": results,
        "engine": {
            "clock": "wall" if not virtual else "virtual",
            "records": records if records is not None else engine_records,
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(blob))
    return path, blob


def _write(path, blob):
    """Rewrite a saved run, keeping the cell's device-free evidence current.

    The validator refuses evidence whose covered digests no longer match, which
    is a real property (an artifact edited after the probe is not the artifact
    the probe saw) and has its own test. Every other test here changes a run to
    break something else, so the coverage is re-taken for them.
    """
    path.write_text(json.dumps(blob))
    evidence_path = path.parent / validate.GPU_FREE_EVIDENCE
    if evidence_path.exists() and path.name.startswith("modelled"):
        evidence = json.loads(evidence_path.read_text())
        if path.name in (evidence.get("covers") or {}):
            evidence["covers"][path.name] = validate._digest(path)
            evidence_path.write_text(json.dumps(evidence))


def _gpu_free(cell_dir, **overrides):
    """The device-free observation a cell is expected to carry.

    Written as if the probe had run inside the modelled side's own container:
    no device node, no open driver handle, and a torch that enumerates nothing.
    """
    covers = {
        p.name: validate._digest(p) for p in validate._runs(Path(cell_dir), "modelled")
    }
    evidence = {
        "probe_version": validate.GPU_FREE_PROBE,
        "taken_in": {"hostname": "cpu-only", "cwd": str(cell_dir)},
        "device_nodes": {node: False for node in validate.DEVICE_NODES},
        "driver_handles": [],
        "masking_vars": {},
        "torch": {
            "imported": True,
            "version": "2.7.0",
            "cuda_available": False,
            "device_count": 0,
        },
        "covers": covers,
    }
    evidence.update(overrides)
    (Path(cell_dir) / validate.GPU_FREE_EVIDENCE).write_text(json.dumps(evidence))
    return evidence


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """A cell that passes, and the pieces to break."""
    registered = tmp_path / "cc_traces_long.jsonl"
    registered.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    monkeypatch.setitem(validate.WORKLOADS, "long", registered)

    cell_dir = tmp_path / "tp2_long"
    cell_dir.mkdir()
    _side(
        cell_dir,
        "real.r1.json",
        ROWS,
        tp=2,
        paced=True,
        mode="measure",
        virtual=False,
        origin=10.0,
    )
    _side(
        cell_dir,
        "modelled.r1.json",
        ROWS,
        tp=2,
        paced=False,
        mode="predict",
        virtual=True,
        origin=0.0,
        prepare=False,
        oracle=validate.SOURCE_FACTORY,
        options=dict(CELL_FACTORY_OPTIONS),
        digests={"price": PRICES_SHA},
        files={"price": {"prices.json": PRICES_SHA}},
    )
    _journal(cell_dir, "real", 1)
    _journal(cell_dir, "modelled", 1)
    (cell_dir / "cc_traces_protocol.json").write_text(
        json.dumps(
            {
                "matches_registration": True,
                "workloads": {"cc_traces_long": validate._digest(registered)},
            }
        )
    )
    (cell_dir / "costs.json").write_text(
        json.dumps(
            {
                **{t: 10.0 for t in validate.MEASURED_COST_TERMS},
                **{t: _supplied(10.0) for t in validate.SUPPLIED_COST_TERMS},
                "cost_schema": validate.COSTS_SCHEMA,
                "execution_clocks": {"real": "wall", "modelled": "wall"},
            }
        )
    )
    (cell_dir / "isolation.json").write_text(
        json.dumps({"verdict": "clean", "isolated": True})
    )
    _gpu_free(cell_dir)
    (cell_dir / "registry.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "sha256": PRICES_SHA,
                        "kind": "standalone_primitive",
                        "measured_at_tp": 2,
                        "produced_by": "primitives.py",
                        "workload_sha256": None,
                        "contents": {"prices.json": PRICES_SHA},
                        "sources": [
                            {"path": "/m/primitive_sweep.json", "sha256": SWEEP_SHA}
                        ],
                        "code": {"scripts/compass/primitives.py": CODE_SHA},
                    },
                    # What sized the deployment. Declared like any other
                    # measured input, because it is one: it decides how many
                    # requests fit, which decides the schedule.
                    capacity_artifact(),
                    region_artifact(),
                ]
            }
        )
    )
    return cell_dir


def run(cell_dir, **kwargs) -> int:
    argv = [
        "cell",
        str(cell_dir),
        "--class",
        kwargs.pop("klass", "long"),
        "--tp",
        str(kwargs.pop("tp", 2)),
        "--repeats",
        "1",
        "--calibration-registry",
        str(cell_dir / "registry.json"),
    ]
    for key, value in kwargs.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return validate.main(argv)


def verdict(cell_dir) -> dict:
    return json.loads((cell_dir / "cc_traces_cell.json").read_text())


class TestAPassingCell:
    def test_a_well_formed_cell_passes(self, cell):
        assert run(cell) == 0
        assert verdict(cell)["passed"] is True

    def test_the_verdict_records_what_it_judged(self, cell):
        run(cell)
        saved = verdict(cell)
        assert saved["class"] == "long" and saved["tp"] == 2
        assert saved["repeats"] == 1
        assert "throughput_tok_s" in saved["metrics"]

    def test_a_missing_cell_is_refused_not_created(self, tmp_path):
        missing = tmp_path / "nothing"
        assert (
            validate.main(["cell", str(missing), "--class", "long", "--tp", "1"]) == 1
        )
        assert not missing.exists()

    def test_fewer_repeats_than_registered_is_refused(self, cell):
        argv = [
            "cell",
            str(cell),
            "--class",
            "long",
            "--tp",
            "2",
            "--calibration-registry",
            str(cell / "registry.json"),
        ]
        assert validate.main(argv) == 1  # default --repeats is 3
        assert any("repeats" in f for f in verdict(cell)["failures"])


class TestTheWorkloadThatRan:
    def test_a_truncated_workload_is_refused(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["workload"] = blob["workload"][:2]
        blob["results"] = blob["results"][:2]
        blob["engine"]["records"] = blob["engine"]["records"][:2]
        blob["run"]["requests"] = 2
        _write(path, blob)
        assert run(cell) == 1

    def test_a_rescaled_arrival_process_is_refused(self, cell):
        """A workload compressed at send time is a different experiment."""
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["time_scale"] = 40.0
        _write(path, blob)
        assert run(cell) == 1
        assert any("time_scale" in f for f in verdict(cell)["failures"])

    def test_a_substituted_easier_request_is_refused(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["workload"][2]["input_tokens"] = 64
        _write(path, blob)
        assert run(cell) == 1

    def test_a_cell_stamped_against_another_workload_is_refused(self, cell):
        (cell / "cc_traces_protocol.json").write_text(
            json.dumps(
                {
                    "matches_registration": True,
                    "workloads": {"cc_traces_long": "9" * 64},
                }
            )
        )
        assert run(cell) == 1

    def test_an_unstamped_cell_is_refused(self, cell):
        (cell / "cc_traces_protocol.json").unlink()
        assert run(cell) == 1


class TestClocksAndArrivals:
    def test_an_infinite_timestamp_is_refused(self, cell):
        """`inf <= inf <= inf` is an ordering, and `inf - inf` is `nan`.

        Both of the checks that came before this one pass on such a run, and
        its percentage errors would be computed from an infinite window.
        """
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        record = blob["engine"]["records"][1]
        record["finish_time"] = math.inf
        record["latency"] = math.inf
        _write(path, blob)
        assert run(cell) == 1
        assert any("finite" in f for f in verdict(cell)["failures"])

    def test_a_declared_arrival_the_engine_did_not_honour_is_refused(self, cell):
        """The modelled side is declared, so its stamps must be exact."""
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        for record in blob["engine"]["records"]:
            record["arrive_time"] = 0.0  # everything at the epoch
            record["first_token_time"] = 0.5
            record["finish_time"] = 1.5
            record["ttft"], record["latency"] = 0.5, 1.5
        _write(path, blob)
        assert run(cell) == 1

    def test_a_paced_arrival_is_allowed_its_jitter(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["engine"]["records"][2]["arrive_time"] += 0.4
        blob["engine"]["records"][2]["first_token_time"] += 0.4
        blob["engine"]["records"][2]["finish_time"] += 0.4
        _write(path, blob)
        assert run(cell) == 0

    def test_a_paced_arrival_that_missed_its_slot_is_refused(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        for field in ("arrive_time", "first_token_time", "finish_time"):
            blob["engine"]["records"][2][field] += 30.0
        _write(path, blob)
        assert run(cell) == 1

    def test_a_timed_out_arrival_barrier_is_refused(self, cell):
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["arrival_barrier_timed_out"] = True
        _write(path, blob)
        assert run(cell) == 1

    def test_an_unread_barrier_on_the_modelled_side_is_refused(self, cell):
        """Unknown is not a pass. The engine answered neither way, so nothing
        says virtual time stayed behind the last arrival -- which is the only
        thing that makes these latencies mean anything."""
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["arrival_barrier_timed_out"] = None
        blob["run"]["arrival_barrier"] = {
            "timed_out": None,
            "why": "TimeoutError: no response from rank 0",
        }
        _write(path, blob)
        assert run(cell) == 1

    def test_a_modelled_run_that_never_recorded_a_barrier_is_refused(self, cell):
        """An artifact written before the field existed reads as unknown, not
        as a pass: it is exactly the run this check was added for."""
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"].pop("arrival_barrier_timed_out")
        blob["run"].pop("arrival_barrier")
        _write(path, blob)
        assert run(cell) == 1

    def test_the_refusal_says_why_the_barrier_could_not_be_read(self, cell, capsys):
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["arrival_barrier"] = {
            "timed_out": None,
            "why": "the engine is not initialised",
        }
        blob["run"]["arrival_barrier_timed_out"] = None
        _write(path, blob)
        assert run(cell) == 1
        assert "the engine is not initialised" in capsys.readouterr().out

    def test_an_unread_barrier_on_the_real_side_is_allowed(self, cell):
        """A wall-clock server never waits for a declared workload, so it has
        no barrier state to report and its absence is not a defect."""
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["run"].pop("arrival_barrier_timed_out")
        blob["run"].pop("arrival_barrier")
        _write(path, blob)
        assert run(cell) == 0

    def test_a_barrier_that_held_is_what_lets_the_cell_pass(self, cell):
        """The passing fixture passes *because* the reading is False, not
        because nobody looked: flipping it to unknown refuses the same cell."""
        assert run(cell) == 0
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        assert blob["run"]["arrival_barrier_timed_out"] is False
        blob["run"]["arrival_barrier_timed_out"] = None
        _write(path, blob)
        assert run(cell) == 1


class TestWhatTheServerActuallyServed:
    def test_a_missing_engine_record_is_refused(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["engine"]["records"] = blob["engine"]["records"][:2]
        _write(path, blob)
        assert run(cell) == 1

    def test_a_response_without_usage_is_refused(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["results"][1]["response"]["usage"].pop("completion_tokens")
        _write(path, blob)
        assert run(cell) == 1

    def test_a_prompt_of_the_wrong_length_is_refused(self, cell):
        """`--check-lengths` is the client's claim; this reads the server's."""
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["results"][1]["response"]["usage"]["prompt_tokens"] = 99
        _write(path, blob)
        assert run(cell) == 1
        assert any("prompt tokens" in f for f in verdict(cell)["failures"])

    def test_sides_that_produced_different_token_counts_are_refused(self, cell):
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["results"][0]["response"]["usage"]["completion_tokens"] = 4
        _write(path, blob)
        assert run(cell) == 1

    def test_a_short_generation_is_reported_without_failing_the_cell(self, cell):
        """A stop token is legitimate; silently comparing throughput is not."""
        for name in ("real.r1.json", "modelled.r1.json"):
            path = cell / name
            blob = json.loads(path.read_text())
            blob["results"][2]["response"]["usage"]["completion_tokens"] = 20
            _write(path, blob)
        assert run(cell) == 0
        assert any("requested output tokens" in n for n in verdict(cell)["notes"])

    def test_a_cell_served_at_another_width_is_refused(self, cell):
        assert run(cell, tp=4) == 1

    def test_prefix_caching_left_on_is_refused(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["server"]["enable_prefix_caching"] = True
        _write(path, blob)
        assert run(cell) == 1


class TestTheTwoSidesRoles:
    def test_an_unprepared_real_side_is_refused(self, cell):
        path = cell / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["prepare"] = None
        _write(path, blob)
        assert run(cell) == 1

    def test_a_prepared_predictor_is_refused(self, cell):
        """Preparation does not move the epoch a declared arrival is stamped
        against, so its whole duration lands inside every measured TTFT."""
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["prepare"] = {
            "drained": True,
            "requested": 3,
            "boundary_engine_time": 0.0,
        }
        _write(path, blob)
        assert run(cell) == 1

    def test_a_paced_predictor_is_refused(self, cell):
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["paced"] = True
        _write(path, blob)
        assert run(cell) == 1

    def test_a_predictor_charged_a_warmup_constant_is_refused(self, cell):
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["server"]["compass"]["oracle_options"]["warmup_seconds"] = 6.7
        _write(path, blob)
        assert run(cell) == 1

    def test_a_predictor_on_the_wall_clock_is_refused(self, cell):
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["server"]["compass"]["virtual_clock"] = False
        _write(path, blob)
        assert run(cell) == 1


class TestCalibrationLeakage:
    """The bet is capture at TP=1 and derive. A cell that read a measurement of
    the width it is predicting has not tested it."""

    def _registry(self, cell, artifacts):
        """Write a registry, filling in the provenance every entry must carry.

        The leakage tests are about one property each, so the parts this file
        checks elsewhere -- what the server loaded, what produced it -- are
        supplied unless the case under test overrides them.
        """
        filled = []
        for entry in artifacts:
            entry = dict(entry)
            entry.setdefault("contents", {"prices.json": entry.get("sha256")})
            entry.setdefault(
                "sources", [{"path": "/m/sweep.json", "sha256": SWEEP_SHA}]
            )
            entry.setdefault("code", {"scripts/compass/primitives.py": CODE_SHA})
            filled.append(entry)
        if not any(e.get("sha256") == TARGET_SHA for e in filled):
            filled.append(capacity_artifact())
        if not any(e.get("kind") == "region_model" for e in filled):
            filled.append(region_artifact())
        (cell / "registry.json").write_text(json.dumps({"artifacts": filled}))

    def test_a_standalone_primitive_at_this_width_is_allowed(self, cell):
        """A collective's price is a property of the rank count, and one rank
        cannot produce it. Measured on its own, outside the target engine."""
        assert run(cell) == 0

    def test_an_undeclared_artifact_is_refused(self, cell):
        self._registry(cell, [])
        assert run(cell) == 1
        assert any("does not declare" in f for f in verdict(cell)["failures"])

    def test_no_registry_at_all_is_refused(self, cell):
        argv = ["cell", str(cell), "--class", "long", "--tp", "2", "--repeats", "1"]
        assert validate.main(argv) == 1

    def test_a_source_calibration_measured_at_the_predicted_width_is_refused(
        self, cell
    ):
        self._registry(
            cell,
            [{"sha256": PRICES_SHA, "kind": "source_calibration", "measured_at_tp": 2}],
        )
        assert run(cell) == 1

    def test_a_source_calibration_from_the_source_width_is_allowed(self, cell):
        self._registry(
            cell,
            [{"sha256": PRICES_SHA, "kind": "source_calibration", "measured_at_tp": 1}],
        )
        assert run(cell) == 0

    def test_an_artifact_fitted_to_the_acceptance_workload_is_refused(self, cell):
        self._registry(
            cell,
            [
                {
                    "sha256": PRICES_SHA,
                    "kind": "source_calibration",
                    "measured_at_tp": 1,
                    "workload_sha256": validate._digest(validate.WORKLOADS["long"]),
                }
            ],
        )
        assert run(cell) == 1

    def test_an_artifact_from_the_target_engine_is_refused(self, cell):
        self._registry(
            cell,
            [
                {
                    "sha256": PRICES_SHA,
                    "kind": "source_calibration",
                    "measured_at_tp": 1,
                    "from_target_engine": True,
                }
            ],
        )
        assert run(cell) == 1

    def test_reading_this_cell_s_own_step_table_is_refused(self, cell):
        """The one that needs no declaration to catch: the file is right there."""
        steps = cell / "real_steps.jsonl"
        steps.write_text('{"step": 0}\n')
        sha = validate._digest(steps)
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["server"]["compass"]["oracle_option_sha256"] = {"table": sha}
        blob["run"]["server"]["compass"]["oracle_options"] = {"table": "steps"}
        _write(path, blob)
        self._registry(
            cell, [{"sha256": sha, "kind": "source_calibration", "measured_at_tp": 1}]
        )
        assert run(cell) == 1

    def test_an_unknown_kind_is_refused(self, cell):
        self._registry(
            cell, [{"sha256": PRICES_SHA, "kind": "whatever", "measured_at_tp": 1}]
        )
        assert run(cell) == 1


class TestCostAndSpeedup:
    def test_missing_cost_terms_are_refused(self, cell):
        (cell / "costs.json").write_text(json.dumps({"execution_real": 100.0}))
        assert run(cell) == 1

    def test_the_gate_reads_the_replay_ratio_not_the_amortised_one(self):
        """The registered gate is the PoC's own, and acquisition sits outside it.

        Folding capture and calibration into the gate would be a different,
        stronger claim than the one registered; reporting them only as a ratio
        that hides them would be a weaker one. Both are reported, and the pass
        criterion is the replay ratio.
        """
        costs = {
            "capture": _supplied(600.0),
            "calibration": _supplied(600.0),
            "derivation": _supplied(0.0),
            "startup_real": 0.0,
            "startup_modelled": 0.0,
            "execution_real": 300.0,
            "execution_modelled": 10.0,
            "load": _supplied(0.0),
            "execution_clocks": {"real": "wall", "modelled": "wall"},
        }
        one = validate._speedup(costs, reuse_cells=1)
        assert one["replay_ratio"] == pytest.approx(30.0)
        assert one["meets_gate"] is True
        # 300 / (10 + 1200) -- the capture is not free the first time it is used
        assert one["amortised_ratio"] == pytest.approx(300.0 / 1210.0)
        assert one["acquisition_s"] == pytest.approx(1200.0)
        assert one["amortised_ratio"] < one["replay_ratio"]
        six = validate._speedup(costs, reuse_cells=6)
        assert six["amortised_ratio"] == pytest.approx(300.0 / 210.0)
        assert six["meets_gate"] is True

    def test_deriving_this_candidate_is_charged_to_the_prediction(self):
        """Per-candidate derivation is part of asking the question, so it is
        inside the gate's denominator rather than beside it."""
        costs = {
            "capture": _supplied(0.0),
            "calibration": _supplied(0.0),
            "derivation": _supplied(50.0),
            "startup_real": 0.0,
            "startup_modelled": 0.0,
            "execution_real": 300.0,
            "execution_modelled": 10.0,
            "load": _supplied(0.0),
            "execution_clocks": {"real": "wall", "modelled": "wall"},
        }
        got = validate._speedup(costs, reuse_cells=1)
        assert got["replay_ratio"] == pytest.approx(300.0 / 60.0)
        assert got["derivation_included_s"] == pytest.approx(50.0)
        assert got["meets_gate"] is True
        costs["derivation"] = _supplied(500.0)
        assert validate._speedup(costs, reuse_cells=1)["meets_gate"] is False

    def test_the_break_even_count_is_reported(self):
        costs = {
            "capture": _supplied(600.0),
            "calibration": _supplied(0.0),
            "derivation": _supplied(0.0),
            "startup_real": 0.0,
            "startup_modelled": 0.0,
            "execution_real": 310.0,
            "execution_modelled": 10.0,
            "load": _supplied(0.0),
            "execution_clocks": {"real": "wall", "modelled": "wall"},
        }
        got = validate._speedup(costs, reuse_cells=2)
        assert got["break_even_cells"] == 2  # 600 acquisition, 300 saved a cell

    def test_a_load_inside_the_startup_is_not_charged_beside_it(self):
        """`CC_TRACES_PROTOCOL.md` §5 defines `load` as the weight load and
        graph capture *inside that startup*, so `startup_real` already contains
        it. The startup-inclusive total used to add it again, which made the
        real side look 200 s more expensive than the clock that measured it."""
        contained = self._wall_costs(
            startup_real=300.0,
            startup_modelled=10.0,
            load=_supplied(200.0, source="server.log", within="startup_real"),
        )
        got = validate._speedup(contained, reuse_cells=1)
        # (300 execution + 300 startup) / (10 execution + 10 startup), not 800/20
        assert got["startup_inclusive_ratio"] == pytest.approx(600.0 / 20.0)

        beside = self._wall_costs(
            startup_real=300.0,
            startup_modelled=10.0,
            load=_supplied(200.0, source="server.log", within=None),
        )
        # A load that declares it overlaps nothing is still a real cost.
        assert validate._speedup(beside, reuse_cells=1)[
            "startup_inclusive_ratio"
        ] == pytest.approx(800.0 / 20.0)

    def test_a_derivation_inside_the_modelled_startup_is_not_doubled(self):
        """The oracle is built in `_init_compass_state`, before `/health`
        answers, so a derivation declared inside `startup_modelled` is already
        in that startup. It stays in the gate denominator either way -- that is
        the per-candidate question -- but the end-to-end total counts it once."""
        costs = self._wall_costs(
            startup_real=0.0,
            startup_modelled=40.0,
            derivation=_supplied(
                30.0, source="startup.json", within="startup_modelled"
            ),
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["derivation_included_s"] == pytest.approx(30.0)
        assert got["replay_ratio"] == pytest.approx(300.0 / 40.0)
        assert got["startup_inclusive_ratio"] == pytest.approx(300.0 / 50.0)

    def test_derivation_inside_the_served_window_is_not_added_to_it(self):
        """A structure first seen mid-schedule is derived while the server is
        serving, so its seconds are already in `execution_modelled`. Adding the
        whole derivation term beside that window charges the mid-schedule part
        twice and makes the replay look slower than it was."""
        costs = self._wall_costs(
            execution_real=300.0,
            execution_modelled=50.0,
            derivation=_supplied(
                20.0, source="derivations.jsonl", within="execution_modelled"
            ),
        )
        got = validate._speedup(costs, reuse_cells=1)
        # 300 / 50, not 300 / 70: the 20 s is inside the 50.
        assert got["replay_ratio"] == pytest.approx(6.0)
        assert got["derivation_included_s"] == pytest.approx(20.0)
        assert got["derivation_inside_execution_s"] == pytest.approx(20.0)
        assert got["derivation_added_to_gate_s"] == pytest.approx(0.0)
        assert got["meets_gate"] is True
        # And the criterion itself did not move.
        assert validate.SPEEDUP_MIN == 5.0

    def test_a_derivation_that_straddles_phases_is_counted_once_each_way(self):
        """Oracle construction happens during startup and a first-seen
        structure is derived mid-schedule, so one run's derivation is two
        durations with two containers. The gate adds the part the served
        window does not already hold, and adds it exactly once."""
        costs = self._wall_costs(
            execution_real=300.0,
            execution_modelled=50.0,
            startup_modelled=40.0,
            derivation=[
                _supplied(30.0, source="derivations.jsonl", within="startup_modelled"),
                _supplied(
                    20.0, source="derivations.jsonl", within="execution_modelled"
                ),
            ],
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["derivation_included_s"] == pytest.approx(50.0)
        assert got["derivation_inside_execution_s"] == pytest.approx(20.0)
        assert got["derivation_added_to_gate_s"] == pytest.approx(30.0)
        # 300 / (50 + 30). The 20 is in the 50 already; the 30 is not.
        assert got["replay_ratio"] == pytest.approx(300.0 / 80.0)
        # The startup part is in `startup_modelled`, so the end-to-end total
        # does not add it a second time either.
        assert got["startup_inclusive_ratio"] == pytest.approx(300.0 / 90.0)

    def test_a_part_naming_a_window_this_cell_does_not_measure_is_refused(self, cell):
        """`within` has to name a window whose seconds exist, or the
        subtraction it licenses is against nothing."""
        costs = json.loads((cell / "costs.json").read_text())
        costs["derivation"] = {
            "seconds": 30.0,
            "source": "derivations.jsonl",
            "within": "warmup",
        }
        (cell / "costs.json").write_text(json.dumps(costs))
        assert run(cell) == 1

    def test_every_part_of_a_split_term_is_checked(self, cell):
        """A term written as parts is only as good as its worst part."""
        costs = json.loads((cell / "costs.json").read_text())
        costs["derivation"] = [
            _supplied(30.0, within="startup_modelled"),
            {"seconds": 20.0, "source": "derivations.jsonl"},  # no `within`
        ]
        (cell / "costs.json").write_text(json.dumps(costs))
        assert run(cell) == 1

    def test_a_bare_supplied_number_is_refused_not_reinterpreted(self, cell):
        """A version 2 record never said what contained its supplied terms.
        Reading one now means guessing the containment that was the bug."""
        costs = json.loads((cell / "costs.json").read_text())
        costs["load"] = 200.0
        costs["cost_schema"] = validate.COSTS_SCHEMA_V2
        (cell / "costs.json").write_text(json.dumps(costs))
        assert run(cell) == 1

    def test_a_supplied_second_without_its_artifact_is_refused(self, cell):
        costs = json.loads((cell / "costs.json").read_text())
        costs["calibration"] = {"seconds": 600.0, "source": "", "within": None}
        (cell / "costs.json").write_text(json.dumps(costs))
        assert run(cell) == 1

    def test_an_unstated_container_is_not_read_as_no_container(self, cell):
        """Silence is the defect, not a claim of independence."""
        costs = json.loads((cell / "costs.json").read_text())
        costs["load"] = {"seconds": 200.0, "source": "server.log"}
        (cell / "costs.json").write_text(json.dumps(costs))
        assert run(cell) == 1

    def test_a_duration_nobody_recorded_is_missing_not_zero(self, cell):
        """Dropping the term is how a never-measured duration used to become a
        free one. The cell is refused so the next run instruments it."""
        costs = json.loads((cell / "costs.json").read_text())
        del costs["derivation"]
        (cell / "costs.json").write_text(json.dumps(costs))
        assert run(cell) == 1

    def _wall_costs(self, **over):
        costs = {
            "capture": _supplied(0.0),
            "calibration": _supplied(0.0),
            "derivation": _supplied(0.0),
            "startup_real": 0.0,
            "startup_modelled": 0.0,
            "execution_real": 300.0,
            "execution_modelled": 10.0,
            "load": _supplied(0.0),
            "execution_clocks": {"real": "wall", "modelled": "wall"},
        }
        costs.update(over)
        return costs

    def test_a_virtual_execution_term_is_not_a_runtime_cost(self):
        """The engine's own window is what the prediction says the workload
        would take. Dividing the real side by it reports how fast the machine
        being modelled is, not how fast modelling it was."""
        costs = self._wall_costs(
            execution_clocks={"real": "wall", "modelled": "virtual"}
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert got["replay_ratio"] is None
        assert "wall-clock" in got["reason"]
        assert "modelled='virtual'" in got["reason"]

    def test_a_cost_record_that_names_no_clock_is_not_read_as_wall(self):
        """Silence is the old record, which called a virtual window
        `execution_modelled` and said nothing. It is not evidence of a wall."""
        costs = self._wall_costs()
        del costs["execution_clocks"]
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert "real=None" in got["reason"] and "modelled=None" in got["reason"]

    def test_the_served_windows_are_reported_beside_the_gate_not_inside_it(self):
        """A 98.87 s prediction against a 3.53 s replay is the TP1 diagnostic.
        The gate has to divide the 3.53, and the 98.87 has to stay visible."""
        costs = self._wall_costs(
            execution_real=60.0,
            execution_modelled=3.53,
            served_window_modelled=98.87,
            served_window_real=59.0,
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["replay_ratio"] == pytest.approx(60.0 / 3.53)
        assert got["meets_gate"] is True
        # The gate did not touch the virtual number even though it is bigger
        # and would have made every ratio here look better.
        assert 60.0 / 98.87 < validate.SPEEDUP_MIN

    def test_a_cell_whose_costs_name_a_virtual_execution_is_refused(self, cell):
        costs = json.loads((cell / "costs.json").read_text())
        costs["execution_clocks"] = {"real": "wall", "modelled": "virtual"}
        (cell / "costs.json").write_text(json.dumps(costs))
        assert run(cell) == 1

    def test_an_unwatched_cell_is_refused(self, cell):
        (cell / "isolation.json").unlink()
        assert run(cell) == 1

    def test_a_contaminated_cell_is_refused(self, cell):
        (cell / "isolation.json").write_text(
            json.dumps({"verdict": "own_contaminated", "isolated": False})
        )
        assert run(cell) == 1

    def test_a_noisy_node_is_a_note_not_a_refusal(self, cell):
        (cell / "isolation.json").write_text(
            json.dumps({"verdict": "node_busy", "isolated": False})
        )
        assert run(cell) == 0
        assert verdict(cell)["notes"]


class TestTheDeviceFreeProof:
    """The speedup claim is a *GPU-free* replay, so the absence has to be real.

    An empty `HIP_VISIBLE_DEVICES` says what a library will enumerate. It does
    not say the process could not have opened `/dev/kfd` itself, and a
    prediction that quietly used a device is not the thing being sold.
    """

    def test_a_cell_with_no_probe_at_all_is_refused(self, cell):
        (cell / validate.GPU_FREE_EVIDENCE).unlink()
        assert run(cell) == 1
        assert any("no gpu_free.json" in f for f in verdict(cell)["failures"])

    def test_a_reachable_device_node_is_refused(self, cell):
        nodes = {node: False for node in validate.DEVICE_NODES}
        nodes["/dev/kfd"] = True
        _gpu_free(cell, device_nodes=nodes)
        assert run(cell) == 1
        assert any("/dev/kfd" in f for f in verdict(cell)["failures"])

    def test_masking_alone_is_not_device_freedom(self, cell):
        """The case this check exists for: devices present, env emptied."""
        nodes = {node: False for node in validate.DEVICE_NODES}
        nodes["/dev/kfd"] = True
        nodes["/dev/dri"] = True
        _gpu_free(cell, device_nodes=nodes, masking_vars={"HIP_VISIBLE_DEVICES": ""})
        assert run(cell) == 1
        assert any("masking" in f for f in verdict(cell)["failures"])

    def test_an_open_driver_handle_is_refused(self, cell):
        _gpu_free(
            cell,
            driver_handles=[{"pid": "41", "fd": "7", "target": "/dev/dri/renderD128"}],
        )
        assert run(cell) == 1
        assert any("driver handle" in f for f in verdict(cell)["failures"])

    def test_a_runtime_that_still_sees_a_device_is_refused(self, cell):
        _gpu_free(
            cell,
            torch={
                "imported": True,
                "version": "2.7.0",
                "cuda_available": True,
                "device_count": 1,
            },
        )
        assert run(cell) == 1
        assert any("torch reports" in f for f in verdict(cell)["failures"])

    def test_a_probe_that_never_asked_the_runtime_is_refused(self, cell):
        _gpu_free(cell, torch={"imported": False, "error": "ImportError: no torch"})
        assert run(cell) == 1

    def test_a_probe_covering_nothing_is_refused(self, cell):
        _gpu_free(cell, covers={})
        assert run(cell) == 1

    def test_a_modelled_run_edited_after_the_probe_is_refused(self, cell):
        """The probe saw a file. This is about whether it is still that file."""
        evidence = json.loads((cell / validate.GPU_FREE_EVIDENCE).read_text())
        evidence["covers"]["modelled.r1.json"] = "9" * 64
        (cell / validate.GPU_FREE_EVIDENCE).write_text(json.dumps(evidence))
        assert run(cell) == 1
        assert any("has changed since" in f for f in verdict(cell)["failures"])

    def test_an_older_probe_is_refused_rather_than_read(self, cell):
        _gpu_free(cell, probe_version=validate.GPU_FREE_PROBE - 1)
        assert run(cell) == 1

    def test_the_probe_records_what_it_looked_at(self, monkeypatch):
        """`observe_device_freedom` is the thing run on the node; what it
        asserts about itself is what the checker later reads.

        The runtime view is stubbed: importing torch here would say more about
        this test runner than about the probe, and takes tens of seconds.
        """
        monkeypatch.setattr(
            validate, "_torch_view", lambda: {"imported": True, "device_count": 0}
        )
        monkeypatch.setattr(validate, "_driver_handles", list)
        seen = validate.observe_device_freedom({"modelled.r1.json": "0" * 64})
        assert seen["probe_version"] == validate.GPU_FREE_PROBE
        assert set(seen["device_nodes"]) == set(validate.DEVICE_NODES)
        assert isinstance(seen["driver_handles"], list)
        assert "masking" in seen["means"]

    def test_the_probe_refuses_a_cell_with_no_modelled_runs(self, tmp_path):
        empty = tmp_path / "cell"
        empty.mkdir()
        assert validate.main(["gpu-free", str(empty)]) == 1


class TestCalibrationProvenanceIsTransitive:
    """A digest names a file. It does not say what is inside it, or what made it.

    Two of the oracle's options are paths, one of which stands for several
    files; the server reports the members it actually read. So the registry has
    to enumerate the same members, name the measurements the artifact came
    from, and pin the code that derived it -- and every digest reachable that
    way is checked for the leakage the top-level digest is checked for.
    """

    def _registry(self, cell, artifacts):
        held = list(artifacts)
        if not any(e.get("sha256") == TARGET_SHA for e in held):
            held.append(capacity_artifact())
        if not any(e.get("kind") == "region_model" for e in held):
            held.append(region_artifact())
        (cell / "registry.json").write_text(json.dumps({"artifacts": held}))

    def _entry(self, **overrides):
        entry = {
            "sha256": PRICES_SHA,
            "kind": "standalone_primitive",
            "measured_at_tp": 2,
            "contents": {"prices.json": PRICES_SHA},
            "sources": [{"path": "/m/sweep.json", "sha256": SWEEP_SHA}],
            "code": {"scripts/compass/primitives.py": CODE_SHA},
        }
        entry.update(overrides)
        return entry

    def test_a_declaration_with_no_sources_is_refused(self, cell):
        self._registry(cell, [self._entry(sources=[])])
        assert run(cell) == 1
        assert any("declares no sources" in f for f in verdict(cell)["failures"])

    def test_a_declaration_with_no_code_hashes_is_refused(self, cell):
        self._registry(cell, [self._entry(code={})])
        assert run(cell) == 1
        assert any("no code digests" in f for f in verdict(cell)["failures"])

    def test_a_source_named_without_a_digest_is_refused(self, cell):
        self._registry(cell, [self._entry(sources=[{"path": "/m/sweep.json"}])])
        assert run(cell) == 1
        assert any("carries no sha256" in f for f in verdict(cell)["failures"])

    def test_a_loaded_file_the_registry_does_not_declare_is_refused(self, cell):
        """The option stands for a directory of files; the digest hides them."""
        self._modelled_files(
            cell, {"prices.json": PRICES_SHA, "prices.tp2.json": "7" * 64}
        )
        self._registry(cell, [self._entry()])
        assert run(cell) == 1
        assert any("does not declare" in f for f in verdict(cell)["failures"])

    def test_a_declared_set_that_matches_what_was_read_is_allowed(self, cell):
        members = {"prices.json": PRICES_SHA, "prices.tp2.json": "7" * 64}
        rolled = validate._rolled_digest(members)
        self._modelled_files(cell, members, digest=rolled)
        self._registry(cell, [self._entry(sha256=rolled, contents=dict(members))])
        assert run(cell) == 0

    def test_a_declared_set_that_does_not_roll_up_is_refused(self, cell):
        """The server's digest over several files is reproducible; if the
        declared members do not produce it, the declaration is of something
        else."""
        members = {"prices.json": PRICES_SHA, "prices.tp2.json": "7" * 64}
        self._modelled_files(cell, members, digest="8" * 64)
        self._registry(cell, [self._entry(sha256="8" * 64, contents=dict(members))])
        assert run(cell) == 1
        assert any("do not roll up" in f for f in verdict(cell)["failures"])

    def test_a_member_file_that_differs_is_refused(self, cell):
        self._registry(cell, [self._entry(contents={"prices.json": "7" * 64})])
        assert run(cell) == 1
        assert any("differs between" in f for f in verdict(cell)["failures"])

    def test_a_source_that_is_this_cell_s_own_step_table_is_refused(self, cell):
        """Reached in two hops rather than one, and just as much the run being
        predicted."""
        steps = cell / "real_steps.jsonl"
        steps.write_text('{"step": 1}\n')
        self._registry(
            cell,
            [
                self._entry(
                    sources=[{"path": str(steps), "sha256": validate._digest(steps)}]
                )
            ],
        )
        assert run(cell) == 1
        assert any("reached" in f for f in verdict(cell)["failures"])

    def test_a_source_that_is_the_acceptance_workload_is_refused(self, cell):
        workload_sha = validate._digest(validate.WORKLOADS["long"])
        self._registry(
            cell,
            [
                self._entry(
                    sources=[{"path": "cc_traces_long.jsonl", "sha256": workload_sha}]
                )
            ],
        )
        assert run(cell) == 1
        assert any("acceptance workload itself" in f for f in verdict(cell)["failures"])

    def test_an_overhead_constant_with_no_value_is_refused(self, cell):
        self._registry(cell, [self._entry(kind="overhead_constant", measured_at_tp=1)])
        assert run(cell) == 1
        assert any("no value written down" in f for f in verdict(cell)["failures"])

    def test_an_overhead_constant_that_states_its_value_is_allowed(self, cell):
        self._registry(
            cell,
            [self._entry(kind="overhead_constant", measured_at_tp=1, value=0.0012)],
        )
        assert run(cell) == 0

    def _modelled_files(self, cell, members, digest=None):
        """Say the modelled server loaded these files for its `price` option."""
        path = cell / "modelled.r1.json"
        blob = json.loads(path.read_text())
        compass = blob["run"]["server"]["compass"]
        compass["oracle_option_files"] = {"price": dict(members)}
        if digest is not None:
            compass["oracle_option_sha256"] = {"price": digest}
        _write(path, blob)


def _cell_verdict(
    name,
    klass,
    real,
    modelled,
    spread=0.01,
    *,
    tp=1,
    modelled_spread=None,
    metrics=("throughput_tok_s",),
    speedup=10.0,
):
    """A passed cell, for the ranking gate.

    Both sides carry a range, not just a centre. The real one is what says
    whether the hardware separated two configurations; the modelled one is what
    says whether the model claims a separation of its own, and the protocol's
    §6 fidelity rule is a comparison of the two. A fixture that carried only
    the real spread could not express an invented separation at all.
    """
    if modelled_spread is None:
        modelled_spread = spread
    block = {
        "real": [real],
        "modelled": [modelled],
        "real_centre": real,
        "modelled_centre": modelled,
        "real_range": [real * (1 - spread), real * (1 + spread)],
        "modelled_range": [
            modelled * (1 - modelled_spread),
            modelled * (1 + modelled_spread),
        ],
        "error_pct": 0.0,
        "tolerance_pct": 10.0,
        "within_tolerance": True,
    }
    return {
        "cell": name,
        "class": klass,
        "tp": tp,
        "passed": True,
        "metrics": {metric: dict(block) for metric in metrics},
        "speedup": {"replay_ratio": speedup, "meets_gate": speedup >= 5.0},
    }


#: Every objective the matrix ranks, which is what a cell has to carry.
ALL_METRICS = tuple(validate.OBJECTIVES)


def _six(tmp_path, over=None):
    """The six cells of the registered matrix, written out and passing.

    `over` is a mapping keyed by `(tp, class)` that replaces that cell's
    verdict, so a test can break exactly one of the six and leave the rest a
    matrix.
    """
    over = over or {}
    dirs = []
    for tp in (1, 2, 4):
        for klass in ("short", "long"):
            name = f"tp{tp}_{klass}"
            where = tmp_path / name
            where.mkdir(parents=True, exist_ok=True)
            blob = over.get(
                (tp, klass),
                _cell_verdict(
                    name,
                    klass,
                    100.0 * tp,
                    100.0 * tp,
                    tp=tp,
                    metrics=ALL_METRICS,
                ),
            )
            (where / "cc_traces_cell.json").write_text(json.dumps(blob))
            dirs.append(str(where))
    return dirs


class TestTheRankingGate:
    def test_spearman_is_tie_aware(self):
        assert validate.spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
        assert validate.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
        # two of the three modelled values tie: average ranks, not an
        # arbitrary order that would read as agreement or disagreement
        assert validate.spearman([1, 2, 3], [1, 2, 2]) == pytest.approx(0.866, abs=1e-3)

    def test_top1_agreement_and_zero_regret(self):
        cells = [
            _cell_verdict("tp1", "long", 100.0, 98.0),
            _cell_verdict("tp2", "long", 150.0, 149.0),
            _cell_verdict("tp4", "long", 130.0, 131.0),
        ]
        out = validate._decide(cells, "throughput_tok_s", "max")
        assert out["real_best"] == "tp2" and out["modelled_top1"] == "tp2"
        assert out["top1_agrees"] is True
        assert out["regret_pct"] == pytest.approx(0.0)
        assert out["rho_meets_gate"] is True

    def test_choosing_the_wrong_config_costs_regret(self):
        """Regret is positive when the model chose badly, in either direction.

        Throughput is maximised and TTFT minimised, and a regret carrying the
        metric's direction in its sign is read wrongly the first time it is
        read. Both say "13.3 % worse" as `+13.3`.
        """
        cells = [
            _cell_verdict("tp1", "long", 100.0, 90.0),
            _cell_verdict("tp2", "long", 150.0, 120.0),
            _cell_verdict("tp4", "long", 130.0, 140.0),
        ]
        out = validate._decide(cells, "throughput_tok_s", "max")
        assert out["modelled_top1"] == "tp4" and out["top1_agrees"] is False
        # the chosen configuration delivers 130 where the best delivers 150
        assert out["regret_pct"] == pytest.approx(13.333, abs=1e-2)

    def test_regret_is_positive_for_a_minimised_metric_too(self):
        cells = [
            _cell_verdict("tp1", "long", 1.0, 3.0),
            _cell_verdict("tp2", "long", 2.0, 1.0),
        ]
        out = validate._decide(cells, "throughput_tok_s", "min")
        assert out["real_best"] == "tp1" and out["modelled_top1"] == "tp2"
        assert out["regret_pct"] == pytest.approx(100.0)

    def test_a_tie_the_hardware_shows_may_be_broken_either_way(self):
        """The short class is arrival-bound: all three widths deliver the
        arrival rate. A model that picks any of them has not disagreed."""
        cells = [
            _cell_verdict("tp1", "short", 100.0, 100.5, spread=0.05),
            _cell_verdict("tp2", "short", 101.0, 99.0, spread=0.05),
            _cell_verdict("tp4", "short", 100.5, 101.5, spread=0.05),
        ]
        out = validate._decide(cells, "throughput_tok_s", "max")
        assert len(out["real_tied_with_best"]) == 3
        assert out["top1_agrees"] is True

    def test_a_matrix_missing_a_cell_reports_no_ranking(self, tmp_path):
        first = tmp_path / "a"
        first.mkdir()
        (first / "cc_traces_cell.json").write_text(
            json.dumps(_cell_verdict("a", "long", 100.0, 100.0))
        )
        out = tmp_path / "verdict.json"
        assert (
            validate.main(
                ["matrix", str(first), str(tmp_path / "missing"), "--out", str(out)]
            )
            == 1
        )
        assert json.loads(out.read_text())["refused"]

    def test_a_matrix_will_not_rank_around_a_failed_cell(self, tmp_path):
        dirs = []
        for name, passed in (("a", True), ("b", False)):
            where = tmp_path / name
            where.mkdir()
            blob = _cell_verdict(name, "long", 100.0, 100.0)
            blob["passed"] = passed
            blob["failures"] = [] if passed else ["something"]
            (where / "cc_traces_cell.json").write_text(json.dumps(blob))
            dirs.append(str(where))
        assert validate.main(["matrix"] + dirs) == 1


class TestTheArtifactsMustSayWhoServed:
    """A live refusal nobody can re-run is not evidence after the fact.

    `cc_traces_run.py` checks, while the repeat is running, that the process
    answering `/compass/provenance` is the one it launched. That check ends
    with the process. These tests are about the other reader: someone holding
    only the directory, months later, asking the same question of the files.
    """

    def _journal_of(self, cell_dir, side):
        return json.loads((Path(cell_dir) / f"run.{side}.json").read_text())

    def _rewrite(self, cell_dir, side, journal):
        (Path(cell_dir) / f"run.{side}.json").write_text(json.dumps(journal))

    def test_a_cell_carrying_its_journals_passes(self, cell):
        assert run(cell) == 0

    def test_a_cell_with_no_journal_cannot_say_who_served(self, cell):
        (Path(cell) / "run.real.json").unlink()
        assert run(cell) == 1

    def test_a_journal_recording_no_executions_is_refused(self, cell):
        journal = self._journal_of(cell, "modelled")
        journal["executions"] = []
        self._rewrite(cell, "modelled", journal)
        assert run(cell) == 1

    def test_an_execution_with_no_process_evidence_is_refused(self, cell):
        journal = self._journal_of(cell, "real")
        journal["executions"][0].pop("server_process")
        self._rewrite(cell, "real", journal)
        assert run(cell) == 1

    def test_half_a_record_is_refused(self, cell):
        journal = self._journal_of(cell, "real")
        journal["executions"][0]["server_process"] = {"verified": True}
        self._rewrite(cell, "real", journal)
        assert run(cell) == 1

    def test_an_unverified_repeat_is_refused(self, cell):
        journal = self._journal_of(cell, "modelled")
        journal["executions"][0]["server_process"]["verified"] = False
        self._rewrite(cell, "modelled", journal)
        assert run(cell) == 1

    def test_a_verdict_its_own_evidence_contradicts_is_refused(self, cell):
        """The stale server, seen from the files.

        Same tree, same flags, same port, so every configuration field in the
        provenance matches. The pid that answered is simply not one this
        repeat started, and the journal nonetheless says verified.
        """
        journal = self._journal_of(cell, "real")
        journal["executions"][0]["server_process"]["said"]["pid"] = 9999
        journal["executions"][0]["server_process"]["observed"]["ancestry"] = [1]
        self._rewrite(cell, "real", journal)
        failures = self._fail(cell)
        assert any("descendant" in reason for reason in failures)
        assert any("does not support that" in reason for reason in failures)

    def test_a_reused_pid_is_caught_by_its_start_tick(self, cell):
        journal = self._journal_of(cell, "real")
        journal["executions"][0]["server_process"]["said"]["start_ticks"] = 7
        self._rewrite(cell, "real", journal)
        assert any("start tick" in r for r in self._fail(cell))

    def test_an_answer_from_another_host_is_refused(self, cell):
        journal = self._journal_of(cell, "modelled")
        journal["executions"][0]["server_process"]["said"]["host"] = "elsewhere"
        self._rewrite(cell, "modelled", journal)
        assert any("answered from" in r for r in self._fail(cell))

    def test_a_domain_suffix_is_not_a_different_host(self, cell):
        journal = self._journal_of(cell, "modelled")
        said = journal["executions"][0]["server_process"]["said"]
        said["host"] = f"{PROC_HOST}.example.com"
        self._rewrite(cell, "modelled", journal)
        assert run(cell) == 0

    def test_an_answer_from_before_the_reboot_is_refused(self, cell):
        journal = self._journal_of(cell, "real")
        journal["executions"][0]["server_process"]["said"]["boot_id"] = "0" * 36
        self._rewrite(cell, "real", journal)
        assert any("different boot" in r for r in self._fail(cell))

    def test_a_process_gone_by_provenance_time_is_refused(self, cell):
        journal = self._journal_of(cell, "modelled")
        journal["executions"][0]["server_process"]["observed"][
            "alive_at_provenance"
        ] = False
        self._rewrite(cell, "modelled", journal)
        assert any("not alive" in r for r in self._fail(cell))

    def test_a_forked_child_answering_is_accepted(self, cell):
        """Ancestry, not equality: the engine may answer from a child."""
        journal = self._journal_of(cell, "real")
        assert (
            4240 in journal["executions"][0]["server_process"]["observed"]["ancestry"]
        )
        assert run(cell) == 0

    def test_fewer_executions_than_repeats_is_refused(self, cell):
        journal = self._journal_of(cell, "real")
        journal["executions"] = journal["executions"] * 2
        self._rewrite(cell, "real", journal)
        assert any("cannot be matched up" in r for r in self._fail(cell))

    def test_a_replay_artifact_silent_about_the_server_is_refused(self, cell):
        """The startup check covers startup only, unless the run says more."""
        path = Path(cell) / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["server"].pop("server_process")
        _write(path, blob)
        assert any("covers only startup" in r for r in self._fail(cell))

    def test_a_different_process_serving_the_replay_is_refused(self, cell):
        """Startup and service are different moments.

        The harness fetched provenance when the server came up; the replay
        client fetched it again, from its own process, during the run these
        numbers come from. If the two name different processes, the repeat was
        verified against a server that did not serve it.
        """
        path = Path(cell) / "real.r1.json"
        blob = json.loads(path.read_text())
        blob["run"]["server"]["server_process"]["start_ticks"] = 555
        _write(path, blob)
        assert any(
            "different process than the one verified" in r for r in self._fail(cell)
        )

    def _fail(self, cell_dir):
        assert run(cell_dir) == 1
        verdict = json.loads((Path(cell_dir) / "cc_traces_cell.json").read_text())
        return verdict["failures"]


class TestADiagnosticIsNotACellHoweverItIsNamed:
    """The same harness, the same file names, a different question.

    A plumbing diagnostic runs `cc_traces_run.py` for real and writes
    `modelled.r1.json`, `run.modelled.json`, an execution record -- everything
    a cell has. What it does not have is a real side, three repeats, or a
    predictor that was not fitted to the run it is predicting. Nothing about
    the directory says so, and renaming it says even less, so the purpose
    travels in the evidence.
    """

    def _mark(self, cell_dir):
        (Path(cell_dir) / "DIAGNOSTIC.json").write_text(
            json.dumps({"not_acceptance": True})
        )

    def test_the_marker_file_refuses_the_cell(self, cell):
        self._mark(cell)
        assert any("DIAGNOSTIC.json is present" in r for r in self._fail(cell))

    def test_an_acceptance_named_directory_is_still_refused(self, cell, tmp_path):
        """The directory is named exactly what the plan names a cell."""
        assert Path(cell).name == "tp2_long"
        self._mark(cell)
        assert run(cell) == 1

    def test_a_journal_run_for_something_else_refuses(self, cell):
        path = Path(cell) / "run.modelled.json"
        journal = json.loads(path.read_text())
        journal["purpose"] = "diagnostic"
        path.write_text(json.dumps(journal))
        assert any("not acceptance" in r for r in self._fail(cell))

    def test_one_diagnostic_execution_refuses_the_whole_cell(self, cell):
        path = Path(cell) / "run.real.json"
        journal = json.loads(path.read_text())
        journal["executions"][0]["purpose"] = "diagnostic"
        path.write_text(json.dumps(journal))
        assert any("run for 'diagnostic'" in r for r in self._fail(cell))

    def test_an_artifact_carries_its_purpose_through_a_copy(self, cell):
        """The point of the stamp being inside the file.

        Copying a diagnostic's artifact into a cell directory leaves the
        journal, the marker and the path behind. The stamp comes with it.
        """
        path = Path(cell) / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob["execution"] = {"purpose": "diagnostic"}
        _write(path, blob)
        assert any(
            "copied here rather than produced here" in r for r in self._fail(cell)
        )

    def test_evidence_that_says_nothing_is_not_acceptance_evidence(self, cell):
        """Silence is not a weak yes.

        Reading a missing purpose as acceptance made the artifact that never
        declared anything the one that passed unquestioned -- which is exactly
        the artifact a diagnostic becomes once its journal is dropped.
        """
        path = Path(cell) / "run.real.json"
        journal = json.loads(path.read_text())
        journal.pop("purpose", None)
        for execution in journal["executions"]:
            execution.pop("purpose", None)
        path.write_text(json.dumps(journal))
        reasons = self._fail(cell)
        assert any("does not say what it was run for" in r for r in reasons)
        assert any("cannot be counted as acceptance" in r for r in reasons)

    def test_an_artifact_with_no_stamp_is_refused(self, cell):
        path = Path(cell) / "modelled.r1.json"
        blob = json.loads(path.read_text())
        blob.pop("execution", None)
        _write(path, blob)
        assert any("does not say what the run was for" in r for r in self._fail(cell))

    def test_a_stored_verdict_with_no_purpose_keeps_its_history(self, cell, capsys):
        """The exception: verdicts predate the field and are not re-graded.

        One cell is not a matrix, so this cannot assert acceptance -- what it
        asserts is that nothing is said about purpose. A missing field is not
        a diagnostic marker.
        """
        blob = {"cell": str(cell), "class": "long", "tp": 2, "passed": True}
        (Path(cell) / "cc_traces_cell.json").write_text(json.dumps(blob))
        validate.main(["matrix", str(cell)])
        out = capsys.readouterr().out
        assert "not acceptance" not in out
        assert "DIAGNOSTIC.json is present" not in out

    def test_a_passing_verdict_beside_a_marker_is_refused_by_the_matrix(
        self, cell, capsys
    ):
        """A verdict is a file, and files travel."""
        blob = {"cell": str(cell), "class": "long", "tp": 2, "passed": True}
        (Path(cell) / "cc_traces_cell.json").write_text(json.dumps(blob))
        self._mark(cell)
        assert validate.main(["matrix", str(cell)]) == 1
        assert "DIAGNOSTIC.json is present" in capsys.readouterr().out

    def test_a_verdict_written_for_a_diagnostic_is_refused_by_the_matrix(
        self, cell, capsys
    ):
        blob = {
            "cell": str(cell),
            "class": "long",
            "tp": 2,
            "passed": True,
            "purpose": "diagnostic",
        }
        (Path(cell) / "cc_traces_cell.json").write_text(json.dumps(blob))
        assert validate.main(["matrix", str(cell)]) == 1
        assert "not acceptance" in capsys.readouterr().out

    def test_the_verdict_says_what_it_was_computed_for(self, cell):
        assert run(cell) == 0
        blob = json.loads((Path(cell) / "cc_traces_cell.json").read_text())
        assert blob["purpose"] == "acceptance"

    def _fail(self, cell_dir):
        assert run(cell_dir) == 1
        verdict = json.loads((Path(cell_dir) / "cc_traces_cell.json").read_text())
        return verdict["failures"]


# --------------------------------------------------------------------------
# the source factory, as it was actually invoked


#: A configuration of the integrated factory that this validator has nothing
#: to say against: every option is one the factory takes, the four that decide
#: coverage are stated, and the seeded-template branch is consistent.
GOOD_FACTORY = {
    "tp": 2,
    "require_complete": "true",
    "head": "true",
    "regions": "source-27b-tp2",
    "derive": "false",
    "template": "body.json",
    "head_template": "head.json",
    "price": "prices.json:graph.json",
    "seconds_per_launch": 0.0,
}


def _factory(options, *, tp=2, served_tp=None):
    """A modelled side that named the source factory, in manifest shape."""
    server = _server(
        served_tp if served_tp is not None else tp, mode="predict", virtual=True
    )
    server["compass"]["oracle"] = validate.SOURCE_FACTORY
    server["compass"]["oracle_options"] = dict(options)
    return SimpleNamespace(manifest={"server": server})


class TestTheSourceFactoryWasGivenAWholeModel:
    """Option pass-through, checked against what the factory does with it.

    The factory has a working default for everything, so a run that states
    almost nothing still starts, still answers, and still produces numbers that
    print. What it does not do is predict the whole model -- and the record of
    such a run is indistinguishable from a complete one unless the options are
    read the way the factory reads them.
    """

    def _reasons(self, options, **kwargs):
        return validate.check_source_factory(_factory(options, **kwargs), 2, "repeat 0")

    def test_the_known_good_configuration_passes(self):
        assert self._reasons(GOOD_FACTORY) == []

    def test_another_oracle_is_refused_rather_than_skipped(self):
        """The binding: a different predictor is a different experiment.

        It is not a weaker acceptance run. Returning no reasons here would let
        any oracle at all past the one check that reads what the predictor was
        actually given.
        """
        side = _factory(GOOD_FACTORY)
        side.manifest["server"]["compass"]["oracle"] = "atom.compass.core.cost.x.Y"
        reasons = validate.check_source_factory(side, 2, "repeat 0")
        assert any("not a cc-traces acceptance cell" in r for r in reasons)

    def test_a_near_miss_qualname_is_a_different_predictor(self):
        side = _factory(GOOD_FACTORY)
        side.manifest["server"]["compass"][
            "oracle"
        ] = "source_oracle.source_cost_oracle"
        reasons = validate.check_source_factory(side, 2, "repeat 0")
        assert any("not a cc-traces acceptance cell" in r for r in reasons)

    def test_naming_no_oracle_at_all_is_refused(self):
        side = _factory(GOOD_FACTORY)
        side.manifest["server"]["compass"].pop("oracle")
        assert validate.check_source_factory(side, 2, "repeat 0") != []

    def test_a_region_profile_of_none_is_not_a_profile(self):
        assert any(
            "not attributed to any region model" in r
            for r in self._reasons({**GOOD_FACTORY, "regions": "none"})
        )

    def test_regions_switched_off_is_not_a_profile(self):
        assert any(
            "not attributed to any region model" in r
            for r in self._reasons({**GOOD_FACTORY, "regions": "false"})
        )

    def test_a_named_profile_is_accepted(self):
        assert self._reasons({**GOOD_FACTORY, "regions": "source-27b-tp4"}) == []

    def test_the_implemented_qualname_is_the_one_checked(self):
        assert (
            validate.SOURCE_FACTORY
            == "atom.compass.runtime.source_oracle.source_cost_oracle"
        )

    def test_the_option_list_is_exactly_what_the_factory_takes(self):
        assert set(validate.SOURCE_FACTORY_OPTIONS) == {
            "model",
            "tp",
            "device",
            "replay_target",
            "block_size",
            "max_model_len",
            "position_rows",
            "block_policy",
            "cudagraph_mode",
            "price",
            "template",
            "head_template",
            "head",
            "regions",
            "seconds_per_launch",
            "require_complete",
            "carry_allocation",
            "derive",
            "interpolate",
        }

    def test_an_option_the_factory_does_not_take_is_refused(self):
        reasons = self._reasons({**GOOD_FACTORY, "require_complete_": "true"})
        assert any("takes no such option" in r for r in reasons)

    def test_every_documented_option_passes_through_unremarked(self):
        """None of the nineteen is itself a complaint."""
        options = {
            **GOOD_FACTORY,
            "model": "/models/qwen3-27b",
            "device": "meta",
            "replay_target": "target.json",
            "block_size": 16,
            "max_model_len": 262144,
            "position_rows": 1,
            "block_policy": "rounds",
            "cudagraph_mode": "FULL",
            "carry_allocation": "false",
            "derive": "true",
            "interpolate": 2.0,
        }
        assert set(options) == set(validate.SOURCE_FACTORY_OPTIONS)
        assert self._reasons(options) == []

    def test_fitted_prices_are_allowed_when_the_density_is_stated(self):
        """A fit inside its own declared support is a prediction, not a gap.

        The coverage record counts it apart from a measurement, so a cell that
        used one stays legible as what it is. Refusing it outright would make
        the only truthful way to report interpolation also the only way to
        fail.
        """
        assert self._reasons({**GOOD_FACTORY, "interpolate": 2.0}) == []

    def test_a_density_the_factory_cannot_read_is_refused(self):
        reasons = self._reasons({**GOOD_FACTORY, "interpolate": "maybe"})
        assert any("not a sampling density" in r for r in reasons)

    def test_a_ratio_under_one_is_refused_here_as_it_is_there(self):
        reasons = self._reasons({**GOOD_FACTORY, "interpolate": 0.5})
        assert any("not a sampling density" in r for r in reasons)

    def test_taking_the_providers_default_does_not_state_the_support(self):
        """`true` runs; what it does not do is say how wide a gap was crossed.

        The manifest keeps the option as written, so the record would move if
        the provider's default ever did, and nothing in the file would say so.
        """
        reasons = self._reasons({**GOOD_FACTORY, "interpolate": "true"})
        assert any("does not state the support" in r for r in reasons)

    def test_leaving_it_out_is_exact_prices_and_no_complaint(self):
        assert "interpolate" not in GOOD_FACTORY
        assert self._reasons(GOOD_FACTORY) == []

    def test_incomplete_pricing_is_refused(self):
        reasons = self._reasons({**GOOD_FACTORY, "require_complete": "false"})
        assert any("complete-only" in r for r in reasons)

    def test_a_defaulted_require_complete_is_still_refused_as_unstated(self):
        """Its default is on, so the run would be complete -- but nobody said so."""
        options = {k: v for k, v in GOOD_FACTORY.items() if k != "require_complete"}
        reasons = self._reasons(options)
        assert any("left require_complete to the factory default" in r for r in reasons)

    def test_the_head_region_must_be_priced(self):
        reasons = self._reasons({**GOOD_FACTORY, "head": "false"})
        assert any("head is off" in r for r in reasons)

    def test_a_defaulted_head_is_off_and_unstated(self):
        options = {k: v for k, v in GOOD_FACTORY.items() if k != "head"}
        reasons = self._reasons(options)
        assert any("left head to the factory default" in r for r in reasons)
        assert any("head is off" in r for r in reasons)

    def test_the_region_model_must_be_named(self):
        options = {k: v for k, v in GOOD_FACTORY.items() if k != "regions"}
        assert any("left regions" in r for r in self._reasons(options))

    def test_an_unmeasured_carried_allocation_cannot_be_graded(self):
        reasons = self._reasons({**GOOD_FACTORY, "carry_allocation": "1"})
        assert any("unmeasured" in r for r in reasons)

    def test_a_flag_the_factory_would_refuse_is_not_read_as_false(self):
        reasons = self._reasons({**GOOD_FACTORY, "head": "on-ish"})
        assert any("not a boolean" in r for r in reasons)

    def test_a_numeric_flag_is_read_the_way_arg_utils_delivers_it(self):
        """`head=1` reaches the server as an int, not the string it was typed as."""
        assert self._reasons({**GOOD_FACTORY, "head": 1}) == []

    def test_the_width_must_be_the_width_of_the_cell(self):
        reasons = self._reasons({**GOOD_FACTORY, "tp": 1})
        assert any("built for another width" in r for r in reasons)

    def test_the_width_must_be_the_width_the_server_was_launched_at(self):
        reasons = self._reasons({**GOOD_FACTORY, "tp": 2}, served_tp=4)
        assert any("tensor_parallel_size=4" in r for r in reasons)

    def test_a_defaulted_width_is_refused_rather_than_assumed(self):
        options = {k: v for k, v in GOOD_FACTORY.items() if k != "tp"}
        assert any("left tp" in r for r in self._reasons(options))

    def test_derivation_on_needs_what_the_tracer_needs(self):
        options = {k: v for k, v in GOOD_FACTORY.items() if k != "derive"}
        options["derive"] = "true"
        reasons = self._reasons(options)
        assert any("derive is on and model" in r for r in reasons)
        assert any("derive is on and block_size" in r for r in reasons)
        assert any("derive is on and max_model_len" in r for r in reasons)

    def test_derivation_on_with_the_tracer_inputs_is_accepted(self):
        options = {
            **GOOD_FACTORY,
            "derive": "true",
            "model": "/models/qwen3-27b",
            "block_size": 16,
            "max_model_len": 262144,
        }
        assert self._reasons(options) == []

    def test_derivation_off_with_no_template_prices_nothing(self):
        options = {k: v for k, v in GOOD_FACTORY.items() if k != "template"}
        reasons = self._reasons(options)
        assert any("refused for want of a graph" in r for r in reasons)

    def test_derivation_off_with_no_head_template_prices_no_head(self):
        options = {k: v for k, v in GOOD_FACTORY.items() if k != "head_template"}
        reasons = self._reasons(options)
        assert any("neither a graph nor a deriver" in r for r in reasons)

    def test_an_empty_option_is_not_a_stated_one(self):
        """`regions=` parses, arrives as an empty string, and sets nothing."""
        assert any(
            "left regions" in r for r in self._reasons({**GOOD_FACTORY, "regions": ""})
        )


class TestTheRegisteredWorkloadIsNotInTheCheckout:
    """The `.jsonl` are reproduced from the corpus, so absence must stop a run.

    They used to be committed, which made a clean checkout look like a complete
    acceptance environment. They are a slice of a licensed 568 MB corpus and
    the manifest beside them is the registration, so the file is now emitted by
    `cc_traces_workload.py emit`. The failure mode that costs something is not
    the missing file -- it is a run that reacts to a missing file by going
    ahead with anything else.
    """

    def _registered(self, tmp_path, klass="short", rows=("a", "b")):
        path = tmp_path / f"cc_traces_{klass}.jsonl"
        path.write_text("".join(json.dumps({"n": r}) + "\n" for r in rows))
        manifest = validate.workload_manifest(path)
        manifest.write_text(json.dumps({
            "class": klass,
            "corpus": {"dataset": "semianalysisai/cc-traces-weka-062126-256k",
                       "file": "traces.jsonl", "sha256": "e39cd2ff" + "0" * 56},
            "emitted_at": "2026-09-11T00:00:00Z",
            "sha256": validate._digest(path),
            "file": path.name,
        }))
        return path, manifest

    def test_the_registered_bytes_are_read_when_they_are_there(
            self, tmp_path, monkeypatch):
        path, _ = self._registered(tmp_path)
        monkeypatch.setitem(validate.WORKLOADS, "short", path)
        assert validate.registered_workload("short") == path
        assert [r["n"] for r in validate.registered_rows("short")] == ["a", "b"]

    def test_an_absent_workload_says_how_to_reproduce_it(
            self, tmp_path, monkeypatch):
        path, _ = self._registered(tmp_path)
        path.unlink()
        monkeypatch.setitem(validate.WORKLOADS, "short", path)
        with pytest.raises(SystemExit) as raised:
            validate.registered_workload("short")
        said = str(raised.value)
        # The command, the class, the corpus and the emission time: enough to
        # produce the same bytes without reading the validator.
        assert "cc_traces_workload.py emit" in said
        assert "--class short" in said
        assert "2026-09-11T00:00:00Z" in said
        assert "semianalysisai/cc-traces-weka-062126-256k" in said

    def test_nothing_is_synthesised_in_place_of_the_absent_file(
            self, tmp_path, monkeypatch):
        path, _ = self._registered(tmp_path)
        path.unlink()
        monkeypatch.setitem(validate.WORKLOADS, "short", path)
        with pytest.raises(SystemExit):
            validate.registered_rows("short")
        assert not path.exists()

    def test_with_neither_file_nor_manifest_it_still_refuses(
            self, tmp_path, monkeypatch):
        path, manifest = self._registered(tmp_path)
        path.unlink()
        manifest.unlink()
        monkeypatch.setitem(validate.WORKLOADS, "short", path)
        with pytest.raises(SystemExit) as raised:
            validate.registered_workload("short")
        assert "Refusing to guess one" in str(raised.value)

    def test_a_file_that_is_not_the_registered_one_is_refused(
            self, tmp_path, monkeypatch):
        path, _ = self._registered(tmp_path)
        path.write_text(json.dumps({"n": "a"}) + "\n")
        monkeypatch.setitem(validate.WORKLOADS, "short", path)
        with pytest.raises(SystemExit) as raised:
            validate.registered_workload("short")
        assert "not the registered short workload" in str(raised.value)

    def test_a_substituted_workload_with_no_manifest_is_left_alone(
            self, tmp_path, monkeypatch):
        # What the validator's own fixtures do: supply rows directly. There is
        # no registration beside them to check against, and what the run read
        # is digested into its record either way.
        path = tmp_path / "cc_traces_short.jsonl"
        path.write_text(json.dumps({"n": "a"}) + "\n")
        monkeypatch.setitem(validate.WORKLOADS, "short", path)
        assert validate.registered_workload("short") == path


class TestTheMatrixIsAWholeMatrix:
    """A decision over part of the matrix is not the registered decision.

    `CC_TRACES_PROTOCOL.md` §3 names six cells -- TP ∈ {1,2,4} × {short, long}
    -- and §6's gates are about the ranking *across* them. One cell cannot
    disagree with itself about which width is best, so a matrix assembled from
    fewer than six has nothing to rank and must not report a pass.
    """

    def test_a_single_cell_is_not_a_matrix(self, tmp_path, capsys):
        """The defect: every per-metric result was skipped for having fewer
        than two comparable cells, and `all()` over nothing is True."""
        where = tmp_path / "tp1_long"
        where.mkdir()
        (where / "cc_traces_cell.json").write_text(
            json.dumps(
                _cell_verdict(
                    "tp1_long", "long", 100.0, 100.0, tp=1, metrics=ALL_METRICS
                )
            )
        )
        assert validate.main(["matrix", str(where)]) == 1
        assert "PASS" not in capsys.readouterr().out

    def test_the_six_registered_cells_pass(self, tmp_path, capsys):
        assert validate.main(["matrix"] + _six(tmp_path)) == 0
        assert "MATRIX PASS" in capsys.readouterr().out

    def test_a_matrix_missing_one_cell_is_refused(self, tmp_path, capsys):
        dirs = _six(tmp_path)
        assert validate.main(["matrix"] + dirs[:-1]) == 1
        out = capsys.readouterr().out
        assert "tp4" in out and "long" in out

    def test_a_repeated_cell_does_not_stand_in_for_a_missing_one(
        self, tmp_path, capsys
    ):
        """Six directories, five configurations. The count is right and the
        matrix is not."""
        dirs = _six(tmp_path)
        twice = tmp_path / "tp1_short_again"
        twice.mkdir()
        (twice / "cc_traces_cell.json").write_text(
            (tmp_path / "tp1_short" / "cc_traces_cell.json").read_text()
        )
        assert validate.main(["matrix"] + dirs + [str(twice)]) == 1
        assert "twice" in capsys.readouterr().out

    def test_a_cell_outside_the_registered_matrix_is_refused(self, tmp_path, capsys):
        where = tmp_path / "tp8_long"
        where.mkdir()
        (where / "cc_traces_cell.json").write_text(
            json.dumps(
                _cell_verdict(
                    "tp8_long", "long", 100.0, 100.0, tp=8, metrics=ALL_METRICS
                )
            )
        )
        assert validate.main(["matrix"] + _six(tmp_path) + [str(where)]) == 1
        assert "tp=8" in capsys.readouterr().out

    def test_a_cell_missing_an_objective_is_refused_not_skipped(self, tmp_path, capsys):
        """The metric a cell does not carry is the one nobody ranked."""
        short = _cell_verdict(
            "tp2_short", "short", 200.0, 200.0, tp=2, metrics=ALL_METRICS
        )
        del short["metrics"]["ttft"]
        dirs = _six(tmp_path, {(2, "short"): short})
        assert validate.main(["matrix"] + dirs) == 1
        assert "ttft" in capsys.readouterr().out


class TestTheSpeedupGateIsPartOfTheVerdict:
    """§6 registers the ≥5× replay speedup as a gate, not as a footnote.

    An accurate model is not the claim being accepted. The claim is an
    accurate model that answers faster than the hardware does, and a matrix
    that ranks perfectly at 2× has measured something valid and failed the
    acceptance it was run for.
    """

    def _slow(self, tmp_path, ratio=2.0):
        slow = _cell_verdict(
            "tp4_long",
            "long",
            400.0,
            400.0,
            tp=4,
            metrics=ALL_METRICS,
            speedup=ratio,
        )
        return _six(tmp_path, {(4, "long"): slow})

    def test_an_accurate_matrix_that_replays_too_slowly_is_not_accepted(
        self, tmp_path, capsys
    ):
        assert validate.main(["matrix"] + self._slow(tmp_path)) == 1
        out = capsys.readouterr().out
        assert "MATRIX FAIL" in out

    def test_the_failing_measurement_is_still_reported(self, tmp_path):
        out = tmp_path / "verdict.json"
        validate.main(["matrix"] + self._slow(tmp_path) + ["--out", str(out)])
        report = json.loads(out.read_text())
        # Valid, measured, and reported -- and not accepted. The distinction
        # is the point: a refused cell has no numbers to read, a failed gate
        # has numbers that say why it failed.
        assert len(report["cells_used"]) == 6
        assert not report["refused"]
        assert report["speedup"]["tp4_long"]["replay_ratio"] == 2.0
        assert report["gates"]["speedup"] is False
        assert report["accepted"] is False

    def test_a_cell_whose_gate_was_never_computed_is_not_accepted(self, tmp_path):
        unknown = _cell_verdict(
            "tp1_short", "short", 100.0, 100.0, tp=1, metrics=ALL_METRICS
        )
        unknown["speedup"] = {
            "replay_ratio": None,
            "meets_gate": None,
            "reason": "costs.json carries no execution terms",
        }
        dirs = _six(tmp_path, {(1, "short"): unknown})
        assert validate.main(["matrix"] + dirs) == 1


class TestTiesAndSeparationsAreBothGated:
    """§6: a model that invents a separation the hardware does not show fails,
    and so does one that flattens a separation the hardware does show.

    Both sides are read as spreads. Two cells are separated on a side when
    that side's repeats do not overlap, and tied when they do -- the same rule
    for the model as for the hardware, because the question is whether the
    model claims a difference the hardware has, not whether its centres happen
    to sort the same way.
    """

    def _pair(self, real_spread, modelled_spread, modelled=(100.0, 101.0)):
        return [
            _cell_verdict(
                "tp1_long",
                "long",
                100.0,
                modelled[0],
                real_spread,
                tp=1,
                modelled_spread=modelled_spread,
            ),
            _cell_verdict(
                "tp2_long",
                "long",
                101.0,
                modelled[1],
                real_spread,
                tp=2,
                modelled_spread=modelled_spread,
            ),
        ]

    def test_an_invented_separation_fails(self):
        """The hardware's repeats overlap; the model's do not. The model is
        claiming a difference nothing measured."""
        cells = self._pair(real_spread=0.05, modelled_spread=0.0001)
        out = validate._decide(cells, "throughput_tok_s", "max")
        assert out["separation_faithful"] is False
        assert any("invent" in r for r in out["separation_failures"])

    def test_a_flattened_separation_fails(self):
        """The hardware separates them; the model's spread swallows it."""
        cells = self._pair(real_spread=0.0001, modelled_spread=0.05)
        out = validate._decide(cells, "throughput_tok_s", "max")
        assert out["separation_faithful"] is False
        assert any("flatten" in r for r in out["separation_failures"])

    def test_agreeing_about_a_tie_is_faithful(self):
        out = validate._decide(
            self._pair(real_spread=0.05, modelled_spread=0.05),
            "throughput_tok_s",
            "max",
        )
        assert out["separation_faithful"] is True
        assert out["separation_failures"] == []

    def test_agreeing_about_a_separation_is_faithful(self):
        out = validate._decide(
            self._pair(real_spread=0.0001, modelled_spread=0.0001),
            "throughput_tok_s",
            "max",
        )
        assert out["separation_faithful"] is True

    def test_a_verdict_with_no_modelled_spread_cannot_be_judged(self):
        cells = self._pair(0.01, 0.01)
        for cell in cells:
            del cell["metrics"]["throughput_tok_s"]["modelled_range"]
            cell["metrics"]["throughput_tok_s"]["modelled"] = []
        out = validate._decide(cells, "throughput_tok_s", "max")
        assert out["separation_faithful"] is False
        assert any("spread" in r for r in out["separation_failures"])

    def test_the_matrix_reads_it(self, tmp_path, capsys):
        """Separation is a gate in its own right.

        The three short cells order the same way on both sides and are inside
        tolerance, so ranking, tolerance and speedup all hold. What fails is
        that the hardware's repeats overlap and the model's do not: the model
        claims three distinct configurations where the machine shows one. A
        matrix that did not read separation would accept this.
        """
        short = {
            (tp, "short"): _cell_verdict(
                f"tp{tp}_short",
                "short",
                centre,
                centre,
                0.05,
                tp=tp,
                modelled_spread=0.0001,
                metrics=ALL_METRICS,
            )
            for tp, centre in ((1, 100.0), (2, 101.0), (4, 102.0))
        }
        dirs = _six(tmp_path, short)
        report = tmp_path / "matrix.json"
        assert validate.main(["matrix", "--out", str(report)] + dirs) == 1
        assert "MATRIX FAIL" in capsys.readouterr().out
        gates = json.loads(report.read_text())["gates"]
        assert gates["separation"] is False
        assert gates["ranking"] is True
        assert gates["tolerance"] is True
        assert gates["speedup"] is True
        assert gates["decided"] is True


class TestTheGateDividesLikeForLike:
    """Execution is one repeat's seconds; derivation must be too.

    `execution_modelled` is the median over the cell's repeats -- what one
    replay cost. A derivation journal covers every repeat, so summing it and
    adding it to a single repeat's execution divides a per-repeat numerator by
    an all-repeats denominator, and the ratio is wrong by the repeat count.
    """

    def _costs(
        self,
        repeats=3,
        execution=10.0,
        derivation=10.0,
        real=120.0,
        within="startup_modelled",
        attribute=True,
    ):
        parts = []
        for index in range(1, repeats + 1):
            part = {
                "seconds": derivation,
                "source": "derivations.jsonl",
                "within": within,
            }
            if attribute:
                part["repeat"] = index
            parts.append(part)
        return {
            "capture": _supplied(0.0),
            "calibration": _supplied(0.0),
            "derivation": parts,
            "load": _supplied(0.0),
            "startup_real": 0.0,
            "startup_modelled": 0.0,
            "execution_real": real,
            "execution_modelled": execution,
            "execution_clocks": {"real": "wall", "modelled": "wall"},
            "repeats": {"real": repeats, "modelled": repeats},
            "execution_by_repeat": {
                "real": {str(i): real for i in range(1, repeats + 1)},
                "modelled": {str(i): execution for i in range(1, repeats + 1)},
            },
            "startup_by_repeat": {
                "real": {str(i): 0.0 for i in range(1, repeats + 1)},
                "modelled": {str(i): 0.0 for i in range(1, repeats + 1)},
            },
        }

    def test_one_repeats_execution_is_divided_by_one_repeats_derivation(self):
        """Three repeats, each 10 s of replay and 10 s of derivation, against
        120 s of real serving. One question costs 20 s, so the gate is 6×."""
        got = validate._speedup(self._costs(), reuse_cells=1)
        assert got["replay_ratio"] == pytest.approx(6.0)
        assert got["meets_gate"] is True
        # The whole term is still reported; only the denominator is per-repeat.
        assert got["derivation_included_s"] == pytest.approx(30.0)
        assert got["derivation_per_repeat_s"] == pytest.approx(10.0)

    def test_derivation_inside_the_replay_window_is_still_counted_once(self):
        """Containment does not change: seconds the execution window already
        holds are not added to it again, per repeat as before."""
        got = validate._speedup(self._costs(within="execution_modelled"), reuse_cells=1)
        assert got["replay_ratio"] == pytest.approx(12.0)
        assert got["derivation_added_to_gate_s"] == pytest.approx(0.0)
        assert got["derivation_inside_execution_s"] == pytest.approx(30.0)

    def test_seconds_belonging_to_no_repeat_are_refused(self):
        """A journal row outside every measured window has no repeat to be
        divided by, and guessing one is what this replaces."""
        got = validate._speedup(self._costs(attribute=False), reuse_cells=1)
        assert got["meets_gate"] is None
        assert "repeat" in got["reason"]

    def test_a_multi_repeat_record_with_no_attribution_at_all_is_refused(self):
        costs = self._costs()
        del costs["execution_by_repeat"]
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert "repeat" in got["reason"]

    def test_a_cost_that_happens_once_per_repeat_may_say_so(self):
        """A supplied term nobody journalled can still be placed: `each` says
        these seconds are spent in every repeat, which is a claim the record
        carries rather than an allocation the reader invents."""
        costs = self._costs(attribute=False)
        costs["derivation"] = [
            {
                "seconds": 10.0,
                "source": "startup.json",
                "within": "startup_modelled",
                "repeat": "each",
            }
        ]
        got = validate._speedup(costs, reuse_cells=1)
        assert got["replay_ratio"] == pytest.approx(6.0)


def _median_of(by_repeat: dict) -> float:
    """The registered quantile over a per-repeat map, as the merge takes it."""
    values = sorted(v for v in by_repeat.values() if isinstance(v, (int, float)))
    return values[min(len(values) - 1, int(0.5 * len(values)))]


class TestEveryRepeatIsAccountedFor:
    """A per-repeat denominator is only per-repeat if the repeats are all there.

    Reading the gate off `execution_by_repeat` made the map the authority on
    which repeats exist, and nothing checked the map against what the record
    says it ran. A side that reports three repeats and lists two, that lists
    one twice under two spellings, or that lists one with no duration, still
    produced a ratio -- over the repeats that happened to be written down.
    And a derivation charged to a repeat outside the map was neither added to
    a repeat nor reported as belonging to none: it was silently dropped, which
    reads as a faster replay than was measured.
    """

    def _costs(
        self,
        *,
        repeats=3,
        modelled_by_repeat=None,
        real_by_repeat=None,
        derivation=None,
        real=120.0,
    ):
        both = {str(i): 10.0 for i in range(1, repeats + 1)}
        if derivation is None:
            derivation = [
                {
                    "seconds": 10.0,
                    "source": "derivations.jsonl",
                    "within": "startup_modelled",
                    "repeat": i,
                }
                for i in range(1, repeats + 1)
            ]
        modelled = both if modelled_by_repeat is None else modelled_by_repeat
        return {
            "capture": _supplied(0.0),
            "calibration": _supplied(0.0),
            "derivation": derivation,
            "load": _supplied(0.0),
            "startup_real": 0.0,
            "startup_modelled": 0.0,
            "execution_real": real,
            "execution_modelled": _median_of(modelled),
            "execution_clocks": {"real": "wall", "modelled": "wall"},
            "repeats": {"real": repeats, "modelled": repeats},
            "execution_by_repeat": {
                "real": (
                    {str(i): real for i in range(1, repeats + 1)}
                    if real_by_repeat is None
                    else real_by_repeat
                ),
                "modelled": modelled,
            },
            "startup_by_repeat": {
                "real": {str(i): 0.0 for i in range(1, repeats + 1)},
                "modelled": {str(i): 0.0 for i in range(1, repeats + 1)},
            },
        }

    def test_a_derivation_charged_to_a_repeat_that_does_not_exist_is_refused(self):
        """Three repeats, and 100 s of derivation booked to a fourth. It is
        not in the map, so it was added to nothing and the ratio read 12x."""
        costs = self._costs(
            derivation=[
                {
                    "seconds": 100.0,
                    "source": "derivations.jsonl",
                    "within": None,
                    "repeat": 4,
                }
            ]
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert got["replay_ratio"] is None
        assert "repeat 4" in got["reason"]

    def test_a_derivation_belonging_to_no_repeat_is_refused_wherever_it_sits(self):
        costs = self._costs(
            derivation=[
                {
                    "seconds": 100.0,
                    "source": "derivations.jsonl",
                    "within": "execution_modelled",
                }
            ]
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert "repeat" in got["reason"]

    def test_a_side_that_does_not_list_every_repeat_is_refused(self):
        costs = self._costs(modelled_by_repeat={"1": 10.0, "2": 10.0})
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert "3" in got["reason"]

    def test_a_repeat_listed_twice_under_two_spellings_is_refused(self):
        """`1` and `01` are one repeat written twice, and the second silently
        replaced the first -- a three-repeat record covering two."""
        costs = self._costs(modelled_by_repeat={"1": 10.0, "01": 30.0, "2": 10.0})
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert "twice" in got["reason"]

    def test_a_repeat_with_no_duration_is_refused(self):
        costs = self._costs(
            modelled_by_repeat={"1": 10.0, "2": None, "3": 10.0},
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert "2" in got["reason"]

    def test_a_repeat_named_by_something_that_is_not_a_number_is_refused(self):
        costs = self._costs(modelled_by_repeat={"1": 10.0, "two": 10.0, "3": 10.0})
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None

    def test_the_real_side_must_cover_its_repeats_too(self):
        costs = self._costs(real_by_repeat={"1": 120.0})
        got = validate._speedup(costs, reuse_cells=1)
        assert got["meets_gate"] is None
        assert "real" in got["reason"]

    def test_repeats_of_different_lengths_take_the_registered_median(self):
        """Ten, twenty and thirty seconds of replay, each deriving for ten.
        One question costs 20, 30 or 40 s; the registered convention takes the
        middle one, so 120 s of real serving is 4x and not an average of
        ratios."""
        costs = self._costs(
            modelled_by_repeat={"1": 10.0, "2": 20.0, "3": 30.0},
        )
        got = validate._speedup(costs, reuse_cells=1)
        assert got["predict_once_s"] == pytest.approx(30.0)
        assert got["replay_ratio"] == pytest.approx(4.0)
        assert got["meets_gate"] is False

    def _twenty_each(self, **over):
        return self._costs(
            real_by_repeat={"1": 20.0, "2": 20.0, "3": 20.0},
            modelled_by_repeat={"1": 20.0, "2": 20.0, "3": 20.0},
            derivation=[],
            **over,
        )

    def test_the_numerator_is_the_median_of_the_records(self):
        """Twenty seconds a repeat on both sides, and no stated totals at all.
        The records are the measurement, so the ratio is 1x."""
        costs = self._twenty_each()
        del costs["execution_real"], costs["execution_modelled"]
        got = validate._speedup(costs, reuse_cells=1)
        assert got["replay_ratio"] == pytest.approx(1.0)
        assert got["meets_gate"] is False

    def test_a_stale_total_beside_the_records_is_never_the_numerator(self):
        """Three real repeats of twenty seconds, and a 120-second
        `execution_real` left over from a longer run. That total read 6x. It
        is not preferred and it is not quietly dropped either: which of the
        two is from another run is not something this can decide."""
        got = validate._speedup(self._twenty_each(real=120.0), reuse_cells=1)
        assert got["replay_ratio"] is None
        assert got["meets_gate"] is None
        assert "execution_real" in got["reason"]

    def test_a_complete_record_still_passes(self):
        got = validate._speedup(self._costs(), reuse_cells=1)
        assert got["replay_ratio"] == pytest.approx(6.0)
        assert got["meets_gate"] is True


class TestAnUndecidedToleranceIsNotAPass:
    """`within_tolerance` has three readings and only one of them is a pass.

    `_across_repeats` writes `None` when the real centre is zero or the metric
    carries no registered tolerance -- an error it could not compute. Reading
    `is not False` turned that into a pass, so a cell nobody could grade
    counted as a cell inside tolerance.
    """

    def _pair(self, value, drop=False):
        cells = [
            _cell_verdict(f"tp{tp}", "long", 100.0 * tp, 100.0 * tp, tp=tp)
            for tp in (1, 2)
        ]
        block = cells[0]["metrics"]["throughput_tok_s"]
        if drop:
            del block["within_tolerance"]
        else:
            block["within_tolerance"] = value
        return cells

    def test_an_ungraded_cell_is_not_within_tolerance(self):
        out = validate._decide(self._pair(None), "throughput_tok_s", "max")
        assert out["within_tolerance"] is False
        assert out["tolerance_undecided"] == ["tp1"]

    def test_a_cell_with_no_reading_at_all_is_not_within_tolerance(self):
        out = validate._decide(self._pair(None, drop=True), "throughput_tok_s", "max")
        assert out["within_tolerance"] is False
        assert out["tolerance_undecided"] == ["tp1"]

    def test_a_graded_failure_stays_a_failure_and_is_not_undecided(self):
        out = validate._decide(self._pair(False), "throughput_tok_s", "max")
        assert out["within_tolerance"] is False
        assert out["tolerance_undecided"] == []

    def test_a_graded_pass_passes(self):
        out = validate._decide(self._pair(True), "throughput_tok_s", "max")
        assert out["within_tolerance"] is True
        assert out["tolerance_undecided"] == []

    def test_a_zero_real_centre_is_where_the_ungraded_reading_comes_from(self):
        reports = [
            {"metrics": {"throughput_tok_s": {"real": 0.0, "modelled": 5.0}}},
            {"metrics": {"throughput_tok_s": {"real": 0.0, "modelled": 6.0}}},
        ]
        block = validate._across_repeats(reports)["throughput_tok_s"]
        assert block["within_tolerance"] is None
        assert block["error_pct"] is None

    def test_the_matrix_does_not_accept_an_ungraded_cell(self, tmp_path, capsys):
        ungraded = _cell_verdict("tp4_long", "long", 400.0, 400.0, tp=4)
        for name in ALL_METRICS:
            block = dict(ungraded["metrics"]["throughput_tok_s"])
            block["within_tolerance"] = None
            ungraded["metrics"][name] = block
        dirs = _six(tmp_path, {(4, "long"): ungraded})
        report = tmp_path / "matrix.json"
        assert validate.main(["matrix", "--out", str(report)] + dirs) == 1
        assert "MATRIX FAIL" in capsys.readouterr().out
        assert json.loads(report.read_text())["gates"]["tolerance"] is False


class TestTheCostRecordIsBoundToTheRunsItPriced:
    """A cost record is about particular executions, and has to name them.

    `repeats` and `execution_by_repeat` were checked against each other and
    against nothing else, so a record could be internally perfect and about a
    different run of this cell: three saved repeats priced by a one-repeat
    record, or a per-execution map beside an aggregate left over from an
    earlier, longer run. Both read as a faster replay than the cell measured.
    """

    def _three(self, cell_dir, costs, seconds=20.0):
        for side in ("real", "modelled"):
            base = json.loads((Path(cell_dir) / f"{side}.r1.json").read_text())
            for index in (2, 3):
                _write(Path(cell_dir) / f"{side}.r{index}.json", base)
            _journal(cell_dir, side, 3, seconds=seconds)
        _gpu_free(cell_dir)
        (Path(cell_dir) / "costs.json").write_text(json.dumps(costs))
        return cell_dir

    def _costs(self, *, coverage=(1, 2, 3), execution=20.0, aggregate=None):
        blob = {
            **{t: 10.0 for t in validate.MEASURED_COST_TERMS},
            **{t: _supplied(10.0) for t in validate.SUPPLIED_COST_TERMS},
            "cost_schema": validate.COSTS_SCHEMA,
            "execution_clocks": {"real": "wall", "modelled": "wall"},
        }
        if coverage:
            blob["repeats"] = {"real": len(coverage), "modelled": len(coverage)}
            blob["execution_by_repeat"] = {
                side: {str(i): execution for i in coverage}
                for side in ("real", "modelled")
            }
            blob["startup_by_repeat"] = {
                side: {str(i): 1.0 for i in coverage} for side in ("real", "modelled")
            }
            for side in ("real", "modelled"):
                blob[f"execution_{side}"] = (
                    aggregate if aggregate is not None else execution
                )
        return blob

    def _run(self, cell_dir):
        return validate.main(
            [
                "cell",
                str(cell_dir),
                "--class",
                "long",
                "--tp",
                "2",
                "--repeats",
                "3",
                "--calibration-registry",
                str(Path(cell_dir) / "registry.json"),
            ]
        )

    def _failures(self, cell_dir):
        self._run(cell_dir)
        return verdict(cell_dir)["failures"]

    def test_a_matching_record_passes(self, cell):
        assert self._run(self._three(cell, self._costs())) == 0

    def test_three_runs_priced_by_a_one_repeat_record_are_refused(self, cell):
        """The defect: the record is self-consistent and about one execution
        of a cell that ran three."""
        self._three(cell, self._costs(coverage=(1,)))
        assert any("saved" in f for f in self._failures(cell))

    def test_a_record_naming_repeats_the_cell_did_not_save_is_refused(self, cell):
        self._three(cell, self._costs(coverage=(1, 2, 7)))
        assert any("7" in f for f in self._failures(cell))

    def test_an_aggregate_only_record_cannot_price_a_multi_repeat_cell(self, cell):
        """Aggregate-only is the one-shot diagnostic shape and stays supported
        there; it cannot stand in for the executions an acceptance cell ran."""
        self._three(cell, self._costs(coverage=()))
        assert any("per-execution" in f for f in self._failures(cell))

    def test_a_stale_aggregate_beside_the_records_is_refused(self, cell):
        """Twenty seconds a repeat, and a 120-second total left over from a
        run that is not this one."""
        self._three(cell, self._costs(execution=20.0, aggregate=120.0))
        assert any("execution_real" in f for f in self._failures(cell))

    def test_one_repeat_may_still_be_priced_in_aggregate(self, cell):
        """The one-shot diagnostic route: one execution, one number, and no
        per-execution record to match it against."""
        assert run(cell) == 0


class TestTheCostRecordIsBoundToWhatTheRunsMeasured:
    """Matching repeat numbers is not matching executions.

    A rerun of a cell writes `run.<side>.json` and `costs.<side>.json` again
    and leaves the merged `costs.json` from the previous run in place until
    someone merges again. Same cell, same three repeats, same numbers 1..3 --
    so every identity check passes while the seconds belong to the run before
    this one. The modelled side got ten times slower and the record still says
    ten seconds a repeat: a 12x replay over a 1.2x measurement.

    The journal is what the repeats were timed by. The cost record is a
    summary of it, and a summary that disagrees with the measurement it
    summarises is not this cell's price.
    """

    def _cell(self, cell_dir, *, real, modelled, priced_real, priced_modelled):
        for side in ("real", "modelled"):
            base = json.loads((Path(cell_dir) / f"{side}.r1.json").read_text())
            for index in (2, 3):
                _write(Path(cell_dir) / f"{side}.r{index}.json", base)
        _journal(cell_dir, "real", 3, seconds=real)
        _journal(cell_dir, "modelled", 3, seconds=modelled)
        _gpu_free(cell_dir)
        costs = {
            **{t: 10.0 for t in validate.MEASURED_COST_TERMS},
            **{t: _supplied(10.0) for t in validate.SUPPLIED_COST_TERMS},
            "cost_schema": validate.COSTS_SCHEMA,
            "execution_clocks": {"real": "wall", "modelled": "wall"},
            "derivation": _supplied(0.0, within="execution_modelled"),
            "repeats": {"real": 3, "modelled": 3},
            "execution_by_repeat": {
                "real": {str(i): priced_real for i in (1, 2, 3)},
                "modelled": {str(i): priced_modelled for i in (1, 2, 3)},
            },
            "startup_by_repeat": {
                side: {str(i): 1.0 for i in (1, 2, 3)} for side in ("real", "modelled")
            },
            "execution_real": priced_real,
            "execution_modelled": priced_modelled,
        }
        (Path(cell_dir) / "costs.json").write_text(json.dumps(costs))
        return cell_dir

    def _failures(self, cell_dir):
        validate.main(
            [
                "cell",
                str(cell_dir),
                "--class",
                "long",
                "--tp",
                "2",
                "--repeats",
                "3",
                "--calibration-registry",
                str(Path(cell_dir) / "registry.json"),
            ]
        )
        return verdict(cell_dir)["failures"]

    def test_a_stale_merge_of_the_previous_run_is_refused(self, cell):
        """The defect: this run replayed in 100s a repeat, the merged record
        still carries the 10s of the run before it, and 120/10 reads 12x."""
        self._cell(
            cell,
            real=120.0,
            modelled=100.0,
            priced_real=120.0,
            priced_modelled=10.0,
        )
        failures = self._failures(cell)
        assert any("100" in f and "10" in f for f in failures)

    def test_a_record_of_what_the_journal_timed_passes(self, cell):
        assert (
            self._failures(
                self._cell(
                    cell,
                    real=120.0,
                    modelled=100.0,
                    priced_real=120.0,
                    priced_modelled=100.0,
                )
            )
            == []
        )

    def test_a_one_shot_aggregate_is_bound_to_its_one_execution_too(self, cell):
        """One repeat, one number, and the same stale-merge failure: the
        journal timed ten seconds and the record prices a hundred."""
        (Path(cell) / "costs.json").write_text(
            json.dumps(
                {
                    **{t: 10.0 for t in validate.MEASURED_COST_TERMS},
                    **{t: _supplied(10.0) for t in validate.SUPPLIED_COST_TERMS},
                    "cost_schema": validate.COSTS_SCHEMA,
                    "execution_clocks": {"real": "wall", "modelled": "wall"},
                    "execution_real": 100.0,
                }
            )
        )
        run(cell)
        assert any("100" in f for f in verdict(cell)["failures"])

    def test_an_execution_the_journal_never_timed_cannot_be_priced(self, cell):
        """A repeat with no replay duration is a repeat nobody measured; the
        seconds beside it came from somewhere else."""
        self._cell(
            cell,
            real=120.0,
            modelled=100.0,
            priced_real=120.0,
            priced_modelled=100.0,
        )
        blob = json.loads((Path(cell) / "run.modelled.json").read_text())
        blob["executions"][1]["replay"] = None
        (Path(cell) / "run.modelled.json").write_text(json.dumps(blob))
        assert any("timed" in f for f in self._failures(cell))
