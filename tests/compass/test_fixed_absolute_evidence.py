"""Final evidence binds complete roots and independently replays source queues."""

import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

from .test_fixed_absolute import bundle
from .test_fixed_absolute_scheduler import configured, finish_step
from atom.model_engine.engine_utility import EngineUtilityHandler


def completed_evidence(tmp_path):
    data = bundle(root_times=(0.,), child_times=(), clients=2)
    with configured(tmp_path, data) as (scheduler, clock, sequences, plan):
        scheduler.extend(list(reversed(sequences)))
        while not scheduler._release_calendar.state.done:
            batch, selected = scheduler.schedule()
            finish_step(scheduler, clock, batch, selected)
        output = queue.Queue()
        runner = SimpleNamespace(call_func=lambda *args, **kwargs: {"inputs": []})
        EngineUtilityHandler(runner, output, scheduler=scheduler)._handle_get_compass_inputs({})
        core = output.get_nowait()[1]["result"]["core_inputs"]
        compass = scheduler.config.compass_config
        server = {"compass": {
            "fixed_absolute_plan": compass.fixed_absolute_plan,
            "fixed_absolute_plan_sha256": compass.fixed_absolute_plan_sha256,
            "request_readiness_profile": compass.request_readiness_profile,
            "loaded_inputs": {"ranks": [{"core_inputs": core}]}}}
        records, results = [], []
        for index, (row, seq) in enumerate(zip(plan.rows, sequences)):
            records.append({"request_id": f"r{index}", "seq_id": str(seq.id),
                "arrive_time": seq.arrive_time, "finish_time": seq.finish_time,
                "shared_preprocessing": {"input_tokens": row["input_tokens"],
                                          "prompt_token_sha256": row["prompt_token_sha256"]}})
            results.append({"index": index, "ok": True, "response": {"id": f"r{index}",
                "usage": {"prompt_tokens": row["input_tokens"], "completion_tokens": row["output_tokens"]}}})
        return plan, server, {"clock": "virtual", "requests": records}, results


@pytest.mark.parametrize("damage", [None, "source_read", "missing_completion", "wrong_sequence",
                                    "reset_queue", "queue_drain", "root_drain", "duplicate_result"])
def test_final_evidence_cannot_hide_missing_leaves_or_queue_resets(tmp_path, damage):
    plan, server, engine, results = completed_evidence(tmp_path)
    core = server["compass"]["loaded_inputs"]["ranks"][0]["core_inputs"]
    calendar = core["release_calendar"]
    if damage == "source_read":
        core["inputs"] = [r for r in core["inputs"] if r["role"] != "runtime.fixed_absolute.source_root"]
    elif damage == "missing_completion":
        calendar["completions"].pop()
    elif damage == "wrong_sequence":
        calendar["releases"][0]["seq_id"] = "another sequence"
    elif damage == "reset_queue":
        # Make all duplicated ready fields agree with a per-request reset;
        # source-law re-derivation must still detect the missing receiver wait.
        service = core["request_readiness"]["causal_releases"][1]
        service["ready_at"] = 1000.3
        service["source_service_started_at"] = 1000.
        calendar["releases"][1].update(ready_at=1000.3, source_service_started_at=1000.)
    elif damage == "queue_drain":
        core["request_readiness"]["causal_queue"]["writer_available_at"] += 1.
    elif damage == "root_drain":
        calendar["root_completed_at"].pop("root1")
    elif damage == "duplicate_result":
        results[1]["index"] = 0
    errors = plan.observation_errors(server, engine, results)
    assert bool(errors) == (damage is not None), errors
    if damage == "reset_queue":
        assert any("re-derivation" in error for error in errors)


def test_final_validation_rechecks_original_source_bytes(tmp_path):
    plan, server, engine, results = completed_evidence(tmp_path)
    path = Path(plan.roots[0]["source"]["path"])
    data = json.loads(path.read_text())
    data["requests"][0]["t"] = 1.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source-root bytes"):
        plan.observation_errors(server, engine, results)
