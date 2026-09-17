"""Native coverage retains workload facts and isolates all timing observations."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def driver():
    scripts = ROOT / "scripts" / "compass"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("native_coverage_test", scripts / "aiperf_native_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coverage_projection_rejects_timing_fields_at_every_level():
    m = driver()
    record = {key: None for key in m.REQUEST_FACTS}
    record.update(request_id="client", response_id="engine", engine_seq_id="native-seq", seconds=999)
    allocation = {"source": "ScheduledBatch", "block_tables": [[2, 5]], "cached_tokens": [16],
                  "state_rows": [0], "state_slots": [1], "state_fork_srcs": [-1],
                  "num_prefill_seqs": 1, "seconds": 888,
                  "state_maintenance": {"relocations": [], "checkpoint_stores": 0,
                                        "checkpoint_restores": 0, "seconds": 777}}
    decision = {key: 0 for key in m.DECISION_FACTS}
    decision.update(allocation=allocation, seconds=666)
    step = {key: None for key in m.STEP_FACTS}
    step.update(req_ids=["native-seq"], decision=decision, seconds=555, spans={"seconds": 444})
    cache = {"ranks": [{"indexes": {"kv": 0, "state": 0}, "quiescence": {"idle": True},
                       "cache_statistics": dict.fromkeys(("requests", "cached_tokens", "compressed_tokens",
                           "wanted_tokens", "reusable_tokens", "full_tokens"), 0), "seconds": 333}]}
    profile = dict.fromkeys(("scenario", "clients", "seed", "benchmark_id", "context_mode",
                            "source_sha256", "dataset_sha256", "config_sha256"))
    result = m.coverage_facts([record], [step], cache, cache, {"counts": {}}, profile, "plan")
    encoded = json.dumps(result)
    assert "seconds" not in encoded and "spans" not in encoded
    assert result["scheduled_steps"][0]["allocation"]["block_tables"] == [[2, 5]]
    step["req_ids"] = ["unattributed"]
    with pytest.raises(ValueError, match="attributed"):
        m.coverage_facts([record], [step], cache, cache, {"counts": {}}, profile, "plan")


def test_only_audited_inactive_dataset_metadata_differences_are_allowed():
    pytest.importorskip("aiperf")
    from aiperf.common.models import DatasetMetadata
    from atom.compass.replay.aiperf_profile import compare_native_metadata
    frozen = DatasetMetadata(conversations=[], has_timing_data=True, sampling_strategy="sequential")
    actual = frozen.model_dump(mode="json")
    actual.update(has_timing_data=False, default_context_mode="deltas_with_responses")
    conversations = [SimpleNamespace(context_mode="deltas_with_responses")]
    result = compare_native_metadata(frozen, actual, conversations)
    assert result["effective_equal"]
    assert set(result["inactive_raw_differences"]) == {"has_timing_data", "default_context_mode"}
    actual["sampling_strategy"] = "shuffle"
    with pytest.raises(ValueError, match="active profile"):
        compare_native_metadata(frozen, actual, conversations)


def test_native_allocation_is_copied_before_batch_changes():
    from atom.compass.core.resolved_runtime import native_batch_allocation
    batch = SimpleNamespace(block_tables=[[2, 5]], num_cached_tokens=[16], state_rows=[0],
                            state_slots_committed=[7], state_fork_srcs=[-1],
                            total_seqs_num_prefill=1, state_maintenance_ops=None)
    result = native_batch_allocation(batch)
    batch.block_tables[0][0] = 999
    batch.state_slots_committed[0] = 999
    assert result["block_tables"] == [[2, 5]]
    assert result["state_slots"] == [7]


def test_backend_scope_and_resolved_body_flags_must_match(tmp_path):
    m = driver()
    expected = {"unified": {"attention_backend": [["backend", "AiterBackend"]],
        "kv_cache_layout": [["k", [["shape", [131072, 4, 16]], ["stride", [64, 16, 1]], ["dtype", "bf16"]]]]},
                "gdn": {"gdn_decode_lossy_fast": False}}
    path = tmp_path / "scope.json"
    path.write_text(json.dumps({"attention_scope": expected}))
    cache_paths = {key: "/owned/cache/" + key.lower() for key in (
        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR")}
    native = {"declaration": {"scopes": expected}, "body_flags": {"FLA_GDN_FIX_BT": False},
              "generated_cache_paths": cache_paths}
    plan = {"request_scope": m.pin(path), "backend_body_flags": dict(native["body_flags"]),
            "environment": {"ATOM_COMPILE_CACHE_ROOT": "/owned/cache", **cache_paths}}
    provenance = {"worker_runtime": [{"native_attention": native,
                    "configuration": {"compilation_cache_dir": "/owned/cache/native-hash"}}]}
    native["declaration"]["scopes"] = json.loads(json.dumps(expected))
    native["declaration"]["scopes"]["unified"]["kv_cache_layout"][0][1][0][1][0] = 112760
    result = m.check_native_scope(plan, provenance)
    assert result["native"] is native
    assert result["kv_capacity_differences"][0]["native_capacity"] == 112760
    native["declaration"]["scopes"]["unified"]["kv_cache_layout"][0][1][1][1][0] = 999
    with pytest.raises(ValueError, match="trailing geometry/strides"):
        m.check_native_scope(plan, provenance)
    native["declaration"]["scopes"]["unified"]["kv_cache_layout"][0][1][1][1][0] = 64
    native["body_flags"]["FLA_GDN_FIX_BT"] = True
    with pytest.raises(ValueError, match="resolved FLA"):
        m.check_native_scope(plan, provenance)
    native["body_flags"]["FLA_GDN_FIX_BT"] = False
    native["generated_cache_paths"]["TRITON_CACHE_DIR"] = "/shared/default"
    with pytest.raises(ValueError, match="generated cache path"):
        m.check_native_scope(plan, provenance)


def test_closeout_waits_for_abort_then_stable_empty_native_drain(tmp_path):
    m = driver()
    journal = tmp_path / "steps.jsonl"
    journal.write_text("native steps")
    clock = [0.0]
    class Replay:
        polls = 0
        flushes = 0
        cache_queries = 0
        def _prefix_cache_snapshot(self, *args):
            self.cache_queries += 1
            # A late native control item can invalidate an earlier quiet read.
            idle = self.polls >= 1 and self.cache_queries != 7
            return {"ranks": [{"quiescence": {"idle": idle}}]}
        def _drain_records(self, *args):
            self.polls += 1
            return {"active_streams": 1 if self.polls == 1 else 0,
                    "active_api_requests": 1 if self.polls == 1 else 0,
                    "admissions": [{"request_id": "cancelled"}] if self.polls == 1 else [],
                    "requests": [{"request_id": "late-finish"}] if self.polls == 2 else []}
        def _flush_measurements(self, *args):
            self.flushes += 1
            return {"acknowledged": True}
    replay, evidence = Replay(), {}
    _, engine, cache = m.collect_native_closeout("", replay, journal, evidence=evidence,
        reserve_seconds=1, now=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0]+n))
    assert engine["admissions"] == [{"request_id": "cancelled"}]
    assert engine["requests"] == [{"request_id": "late-finish"}]
    assert evidence["stable"] and cache["ranks"][0]["quiescence"]["idle"]
    assert replay.polls >= 7 and replay.flushes >= 6


def test_closeout_refuses_nonquiescent_native_stream(tmp_path):
    m = driver()
    clock = [0.0]
    replay = SimpleNamespace(
        _prefix_cache_snapshot=lambda *args: {"ranks": [{"quiescence": {"idle": False}}]},
        _drain_records=lambda *args: {"active_streams": 1, "active_api_requests": 1})
    with pytest.raises(TimeoutError, match="stable flushed"):
        m.collect_native_closeout("", replay, tmp_path / "unused", evidence={},
            reserve_seconds=.1, now=lambda: clock[0],
            sleep=lambda n: clock.__setitem__(0, clock[0]+n))
