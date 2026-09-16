"""Native dispatch initialization needs observed classes and a final empty reset."""
from copy import deepcopy
import hashlib
import json

import pytest

from atom.compass.core.cache_boundary import FLUSH_SCHEMA, RESET_SCHEMA
from atom.compass.prefix_workload import token_digest
from atom.compass.replay import native_preparation as preparation

MODEL = "Qwen/Qwen3.8-27B"


def native_scope():
    layers, views = {}, {}
    for layer in (i for i in range(64) if i % 4 != 3):
        layers[str(layer)] = dict(impl="atom.model_ops.attention_gdn.GatedDeltaNet",
            impl_attrs=dict(layer_num=layer, num_v_heads=48, num_k_heads=16, head_k_dim=128, head_v_dim=128))
        views["layer_" + str(layer)] = {"k": {"dtype": "torch.bfloat16"},
            "v": {"dtype": "torch.bfloat16", "shape": [32, 48, 128, 128]}}
    return dict(body_flags={"FLA_GDN_FIX_BT": False, "USE_DEFAULT_FLA_NORM": 0}, gdn_dispatch={"is_amd": True},
        record=dict(config={"kv_cache_dtype": "bf16", "kv_cache_block_size": 16}, layers=layers, kv_views=views))


def native_row(tokens, request="1", *, dummy=False):
    return dict(req_ids=[request], num_scheduled_tokens=[tokens], context_lens=[tokens],
        num_prefill_tokens=tokens, topology={"tp": 1}, rank_coords={"tp": 0}, compiled=True, capture_bucket=None,
        decision={"allocation": dict(source="ScheduledBatch", is_dummy_run=dummy, block_tables=[[1, 2, 3, 4, 5]],
            num_prefill_seqs=1, state_rows=[0], state_slots=[1], state_fork_srcs=[-1])})


def boundary(reset=False):
    snapshot = dict(quiescence=dict(idle=True, state_readers=0, state_deferred_readers=0, reasons=[]),
                    indexes={"kv": 0, "state": 0}, counters={"requests": 3})
    worker = dict(acknowledged=True, kind="device_synchronize", workers_completed=1,
                  core_timeline_drained=True, measurement_journal={"pending_steps_after": 0})
    rank = dict(acknowledged=True, before=deepcopy(snapshot), after=deepcopy(snapshot), worker_barrier=worker,
                counter_delta={"requests": 0})
    if reset:
        rank["before"]["indexes"] = {"kv": 4, "state": 2}
    return dict(schema=RESET_SCHEMA if reset else FLUSH_SCHEMA, acknowledged=True, ranks=[rank])


def test_keys_follow_tile_classes_and_short_fallback_not_exact_trace_lengths():
    rows = [native_row(1), native_row(17), native_row(48), native_row(128)]
    result = preparation.gdn_dispatch_coverage(rows, native_scope=native_scope(), model=MODEL, require_complete=True)
    assert not result["missing"]
    assert result["observed"]["output_bt16"]["actual_tokens"] == 1
    assert result["observed"]["output_bt32"]["actual_tokens"] == 17
    assert result["observed"]["output_bt64"]["actual_tokens"] == 48
    assert result["scope"]["arbitrary_first_use_covered"] is False


def test_short_bt64_does_not_cover_the_amd_fused_branch():
    rows = [native_row(16), native_row(32), native_row(48)]
    result = preparation.gdn_dispatch_coverage(rows, native_scope=native_scope(), model=MODEL)
    assert result["missing"] == ["amd_fused_ge64"]
    with pytest.raises(ValueError, match="amd_fused_ge64"):
        preparation.gdn_dispatch_coverage(rows, native_scope=native_scope(), model=MODEL, require_complete=True)


@pytest.mark.parametrize("damage", ["dummy", "missing_dummy_fact", "empty_tables", "no_request", "decode", "tp2"])
def test_unexecuted_or_out_of_scope_rows_do_not_warm_a_class(damage):
    row = native_row(32)
    allocation = row["decision"]["allocation"]
    if damage == "dummy":
        allocation["is_dummy_run"] = True
    elif damage == "missing_dummy_fact":
        del allocation["is_dummy_run"]
    elif damage == "empty_tables":
        allocation["block_tables"] = []
    elif damage == "no_request":
        row["req_ids"] = []
    elif damage == "decode":
        row["num_prefill_tokens"] = 0
    else:
        row["topology"] = {"tp": 2}
    result = preparation.gdn_dispatch_coverage([row], native_scope=native_scope(), model=MODEL)
    assert result["observed"] == {}
    with pytest.raises(ValueError, match="no actual GDN dispatch evidence"):
        preparation.gdn_dispatch_coverage([row], native_scope=native_scope(), model=MODEL, require_complete=True)


@pytest.mark.parametrize("damage", ["fixed_tile", "dtype", "geometry", "backend"])
def test_declared_runtime_scope_is_required(damage):
    scope = native_scope()
    if damage == "fixed_tile":
        scope["body_flags"]["FLA_GDN_FIX_BT"] = True
    elif damage == "dtype":
        scope["record"]["kv_views"]["layer_0"]["v"]["dtype"] = "torch.float16"
    elif damage == "backend":
        scope["gdn_dispatch"]["is_amd"] = False
    else:
        scope["record"]["layers"]["0"]["impl_attrs"]["num_v_heads"] = 24
    with pytest.raises(ValueError, match="GDN dispatch preparation"):
        preparation.gdn_dispatch_coverage([native_row(32)], native_scope=scope, model=MODEL)


@pytest.mark.parametrize("tokens", [16, 32, 64])
def test_fixed_prompts_reach_the_class_through_native_checkpoint_scheduling(tokens):
    from atom.compass.core.resolved_runtime import native_batch_allocation
    from atom.model_engine.sequence import Sequence
    from .test_cache_boundary import engine_with_cache

    engine, _ = engine_with_cache(warm=False)
    prompt = preparation.gdn_dispatch_prompt(tokens)
    sequence = Sequence(prompt, 16, has_per_req_cache=True)
    engine.scheduler.add(sequence)
    batch, _ = engine.scheduler.schedule()
    assert list(batch.num_scheduled_tokens) == [tokens]
    allocation = native_batch_allocation(batch)
    assert allocation["source"] == "ScheduledBatch" and allocation["is_dummy_run"] is False
    assert allocation["block_tables"]
    assert len(prompt) == tokens + 1 and prompt == preparation.gdn_dispatch_prompt(tokens)


class Replay:
    def __init__(self, journal, *, initial=(16, 64), damage=None):
        self.journal, self.initial, self.damage = journal, initial, damage
        self.calls, self.records = [], []
        self.next_id = 0

    async def _submit_requests(self, base, payloads, arrivals, **kwargs):
        assert len(payloads) == 1 and kwargs["streaming"] is True
        self.next_id += 1
        request = str(self.next_id)
        body = json.loads(payloads[0])
        extra = kwargs["endpoint"] == "/v1/completions"
        tokens = len(body["prompt"]) - 1 if extra else None
        self.calls.append(("dispatch", tokens) if extra else ("chat",))
        classes = [tokens] if extra else self.initial
        if extra and self.damage == "unobserved":
            classes = [16]
        with self.journal.open("a") as stream:
            for value in classes:
                stream.write(json.dumps(native_row(value, request, dummy=self.damage == "dummy")) + "\n")
            # An unrelated row cannot be used to satisfy the missing class.
            stream.write(json.dumps(native_row(32, "unrelated")) + "\n")
        consumed = dict(input_tokens=len(body["prompt"]), prompt_token_sha256=token_digest(body["prompt"])) if extra else {}
        if extra and self.damage == "wrong_tokens":
            consumed["prompt_token_sha256"] = "wrong"
        self.records.append(dict(seq_id=request, request_id="response-" + request, shared_preprocessing=consumed))
        response = dict(id="response-" + request,
                        usage=dict(prompt_tokens=len(body["prompt"]) if extra else 20, completion_tokens=2))
        return [dict(ok=True, response=response)], {"submitted": 1}

    def _flush_measurements(self, *args):
        self.calls.append(("flush",))
        return boundary()

    def _drain_records(self, *args):
        self.calls.append(("drain",))
        records, self.records = self.records, []
        return dict(requests=records, admissions=[], active_streams=0, active_api_requests=0)

    def _reset_prefix_cache(self, *args):
        self.calls.append(("reset",))
        result = boundary(reset=True)
        if self.damage == "nonempty_reset":
            result["ranks"][0]["after"]["indexes"]["state"] = 1
        return result


def run_preparation(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(preparation, "marked_payloads", lambda *_: [{"model": MODEL, "messages": []}])
    journal = tmp_path / "steps.jsonl"
    journal.write_text(json.dumps(native_row(32, "stale")) + "\n")
    replay = Replay(journal, **kwargs)
    result = preparation.prepare_native("http://owned", None, None, replay, tmp_path,
        step_journal=journal, native_scope=native_scope(), model=MODEL)
    return result, replay


def test_only_missing_classes_run_before_flush_drain_and_empty_reset(tmp_path, monkeypatch):
    cache = tmp_path / "compiled-cache"
    cache.write_text("retained compiled bytes")
    result, replay = run_preparation(tmp_path, monkeypatch)
    assert [call for call in replay.calls if call[0] == "dispatch"] == [("dispatch", 32)]
    assert replay.calls[-4:] == [("flush",), ("drain",), ("drain",), ("reset",)]
    assert result["dispatch_coverage_before"]["missing"] == ["output_bt32"]
    assert result["dispatch_coverage"]["missing"] == []
    assert result["dispatch_requests"][0]["input_tokens"] == 33
    assert result["initialization_regime"] == "declared_gdn_dispatch_classes_warmed"
    assert result["setup_elapsed_seconds"] >= 0 and result["outside_profile"]
    assert result["first_use_latency_fitted"] is False
    assert cache.read_text() == "retained compiled bytes"
    raw = replay.journal.read_bytes()
    start = result["step_journal_start_offset"]
    assert start > 0 and result["step_journal_end_offset"] == len(raw)
    assert result["step_journal_region_sha256"] == hashlib.sha256(raw[start:]).hexdigest()


def test_no_extra_requests_when_existing_native_preparation_covers_classes(tmp_path, monkeypatch):
    result, replay = run_preparation(tmp_path, monkeypatch, initial=(16, 32, 64))
    assert not result["dispatch_requests"]
    assert not any(call[0] == "dispatch" for call in replay.calls)
    assert replay.calls[-1] == ("reset",)


def test_short_bt64_still_requires_only_the_missing_t64_representative(tmp_path, monkeypatch):
    result, replay = run_preparation(tmp_path, monkeypatch, initial=(16, 32, 48))
    assert result["dispatch_coverage_before"]["missing"] == ["amd_fused_ge64"]
    assert [call for call in replay.calls if call[0] == "dispatch"] == [("dispatch", 64)]
    assert not result["dispatch_coverage"]["missing"]


@pytest.mark.parametrize("damage,match", [
    ("unobserved", "did not execute"), ("dummy", "did not execute"),
    ("wrong_tokens", "consumed token"), ("nonempty_reset", "empty KV and state"),
])
def test_requested_prompts_and_unacknowledged_resets_are_not_coverage(tmp_path, monkeypatch, damage, match):
    with pytest.raises(ValueError, match=match):
        run_preparation(tmp_path, monkeypatch, damage=damage)
    assert not (tmp_path / "preparation.json").exists()
