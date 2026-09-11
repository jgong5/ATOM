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
TABLE_SHA = "b" * 64
WORKLOAD_ROW_KEYS = ("arrival_s", "input_tokens", "output_tokens")

#: What the passing cell's modelled side told the acceptance factory: the whole
#: model, priced completely, at the width the cell is for. Anything less is a
#: different experiment, so the fixture that is supposed to pass states it all.
CELL_FACTORY_OPTIONS = {
    "tp": 2,
    "require_complete": "true",
    "head": "true",
    "regions": "source-27b-tp2",
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
        },
    }
    if served is not None:
        provenance["server_process"] = served
    return provenance


def _journal(cell_dir, side, repeats, *, executions=None):
    """The run journal `cc_traces_run.py` leaves beside the artifacts."""
    if executions is None:
        executions = [
            {
                "execution_id": f"cx-{side}{index}",
                "purpose": "acceptance",
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
                    }
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
        (cell / "registry.json").write_text(json.dumps({"artifacts": artifacts}))

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


def _cell_verdict(name, klass, real, modelled, spread=0.01):
    """A passed cell carrying one metric, for the ranking gate."""
    return {
        "cell": name,
        "class": klass,
        "passed": True,
        "metrics": {
            "throughput_tok_s": {
                "real": [real],
                "modelled": [modelled],
                "real_centre": real,
                "modelled_centre": modelled,
                "real_range": [real * (1 - spread), real * (1 + spread)],
                "error_pct": 0.0,
                "tolerance_pct": 10.0,
                "within_tolerance": True,
            }
        },
    }


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
        """The exception: verdicts predate the field and are not re-graded."""
        blob = {"cell": str(cell), "class": "long", "tp": 2, "passed": True}
        (Path(cell) / "cc_traces_cell.json").write_text(json.dumps(blob))
        assert validate.main(["matrix", str(cell)]) == 0
        capsys.readouterr()

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
