"""CPU integration through actual AIPerf/core records into proper comparison."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("aiperf.common.clock")
from aiperf.common.models import RawRecordInfo
if "end_perf_ns" not in RawRecordInfo.model_fields:
    pytest.skip("requires pinned optional raw export patch", allow_module_level=True)

from atom.compass.core.proper_replay import phase_accounting, profiling_wall_window, validate_records
from atom.compass.replay.aiperf_records import export_controlled_records, normalize_records
from atom.compass.replay.aiperf_runner import run_controlled_replay
from atom.utils.clock import VirtualClock, set_clock
from .test_controlled_engine import clock, make_core
from .test_aiperf_controlled import _dataset_and_config, _tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/compass"))
spec = importlib.util.spec_from_file_location("proper_pair_integration", ROOT / "scripts/compass/cc_traces_proper.py")
paired = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = paired
spec.loader.exec_module(paired)


def test_actual_controlled_records_normalize_and_compare_different_dynamic_counts(make_core, tmp_path):
    from atom.compass.replay.bootstrap import install
    install("gfx942", source="synthetic controlled-record comparator fixture")
    blobs = []
    for side, seconds in (("real", 10.), ("modelled", 20.)):
        set_clock(VirtualClock(epoch=100.))
        directory = tmp_path / side
        directory.mkdir()
        fixture = make_core(seconds=seconds)
        fixture.scheduler.config.hf_config.model_type = "llama"
        fixture.scheduler.config.hf_config.vocab_size = 128
        store, metadata, config = _dataset_and_config(
            directory, branch=True, start_ratio=0., grace_period=1000.)
        config.benchmark_id = "00000000-0000-4000-8000-000000000123"
        from aiperf.common.enums import ExportLevel
        config.output.export_level = ExportLevel.RAW
        config.output.export_http_trace = True
        tokenizer = _tokenizer()
        try:
            result = run_controlled_replay(
                core=fixture.core, tokenizer=tokenizer, user_config=config,
                dataset_metadata=metadata, dataset_client_metadata=store.get_client_metadata())
        finally:
            asyncio.run(store.stop())
        raw = export_controlled_records(result.records)
        consumed = {row["api_request_id"]: {"input_tokens": row["prompt_tokens"],
                    "prompt_token_sha256": row["prompt_token_sha256"]} for row in result.dispatches}
        caps = {("root", index): 2 for index in range(3)}
        caps.update({("left", 0): 2, ("right", 0): 2})
        records = normalize_records(raw, user_config=config, tokenizer=tokenizer, model_path="fixture",
                                    consumed=consumed, expected_caps=caps)
        phase = phase_accounting([m.model_dump(mode="json") for m in result.messages])
        validate_records(records, phase)
        wall = profiling_wall_window(result.wall_phase_events)
        assert 0 < wall["seconds"] < phase["observed_duration_seconds"]
        assert not any(result.cleanup.values())
        blobs.append({"side": side, "server": {}, "engine": {"requests": []},
                      "records": records, "phase": phase})
    real, modelled = blobs
    assert len(real["records"]) != len(modelled["records"])
    identities = paired.pair_input_observations(real["records"], modelled["records"])
    assert identities["shared_marker_conversation_turns"] > 0
    report = paired.compare_dynamic(real, modelled)
    assert report["metrics"]["ttft"]["real"]["n"] == len(real["records"])
    assert report["metrics"]["ttft"]["modelled"]["n"] == len(modelled["records"])
    assert report["metrics"]["ttft"]["error_pct"]["median"] > 0
    changed = [dict(row, cache_bust_marker="different-" + row["cache_bust_marker"])
               for row in modelled["records"]]
    with pytest.raises(ValueError, match="no shared"):
        paired.pair_input_observations(real["records"], changed)


def test_declared_acceptance_requires_protocol_repeats_before_launch(tmp_path):
    prepared = {"config": {"scenario": "inferencex-agentx-mvp", "benchmark_id": "benchmark",
        "loadgen": {"concurrency": 1, "benchmark_duration": 900},
        "input": {"random_seed": 42},
        "endpoint": {"type": "chat", "streaming": True, "use_server_token_count": True}},
        "source": {"sha256": "source"}, "metadata": {},
        "conversations": [{"context_mode": "deltas_with_responses"}]}
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(prepared))
    import hashlib
    plan = {"schema": "compass.aiperf_proper_pair/1", "purpose": "acceptance", "repeats": 1,
            "prepared": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "native_engine_args": [], "modelled_engine_args": [], "model": "fixture", "cache_policy": {},
            "record_export": {"export_level": "raw", "export_http_trace": True}}
    pin = tmp_path / "plan.json"
    pin.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="protocol repeat"):
        paired.load_case(str(pin), hashlib.sha256(pin.read_bytes()).hexdigest(), "aiperf_proper_fixture")
    plan["repeats"] = 3
    pin.write_text(json.dumps(plan))
    case = paired.load_case(str(pin), hashlib.sha256(pin.read_bytes()).hexdigest(), "aiperf_proper_fixture")
    assert case["purpose"] == "acceptance" and case["registered_acceptance_cell"]
    built = paired.build_steps(case, tmp_path / "cell", port=8850, engine_port=8860, advisory=False)
    modelled = [step for step in built["steps"] if step["role"] == "proper_session"]
    assert [step["repeat"] for step in modelled] == [1, 2, 3]
    assert len({step["command"][step["command"].index("--out")+1] for step in modelled}) == 3


def test_throughput_uses_full_serving_phase_and_only_completed_outputs():
    def artifact(side, origin, duration, tokens, request_duration):
        start = int((origin + 10) * 1e9)
        successful = {"start_ns": start, "first_visible_ns": start + 10**9,
            "end_ns": start + int(request_duration * 1e9), "output_tokens": tokens,
            "input_tokens": 20, "cancelled": False, "error": None}
        cancelled = dict(successful, output_tokens=999, cancelled=True)
        failed = dict(successful, output_tokens=777, error={"type": "RequestError"})
        return {"side": side, "server": {}, "engine": {"requests": []},
            "records": [successful, cancelled, failed],
            "phase": {"origin_ns": int(origin * 1e9),
                      "completed_ns": int((origin + duration) * 1e9)},
            "execution_wall_window": {"seconds": .1}}

    real = artifact("real", 100., 40., 10, 10.)
    modelled = artifact("modelled", 200., 20., 12, 5.)
    metrics = paired.compare_dynamic(real, modelled)["metrics"]
    throughput = metrics["throughput_tok_s"]
    assert throughput["real"] == .25
    assert throughput["modelled"] == .6
    assert throughput["real_window_s"] == 40.
    assert throughput["modelled_window_s"] == 20.
    assert throughput["real_output_tokens"] == 10
    assert throughput["modelled_output_tokens"] == 12
    assert throughput["error_pct"] == pytest.approx(140.)
    assert metrics["ttft"]["real"]["n"] == metrics["ttft"]["modelled"]["n"] == 1


@pytest.mark.parametrize("phase", [{}, {"origin_ns": 1, "completed_ns": 1}])
def test_throughput_refuses_missing_or_empty_phase(phase):
    row = {"start_ns": 10**9, "first_visible_ns": 2 * 10**9,
           "end_ns": 3 * 10**9, "output_tokens": 2, "input_tokens": 1,
           "cancelled": False, "error": None}
    blob = {"side": "real", "server": {}, "engine": {"requests": []},
            "records": [row], "phase": phase}
    with pytest.raises(ValueError, match="profiling-phase endpoints"):
        paired.compare_dynamic(blob, dict(blob, side="modelled"))
