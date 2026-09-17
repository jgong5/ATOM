"""Pinned prefix rows survive diagnostic identity and the replay boundary."""

import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.prefix_workload import token_digest


def test_module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


test_module.__test__ = False
fixture = test_module("prefix_diag_fixture", "test_corpus_diagnostic_run.py")
codec_tests = test_module("prefix_codec_fixture", "test_prefix_workload.py")
diagnostic = fixture.diagnostic
replay = codec_tests.replay


def prefix_case(tmp_path):
    codec, tokenizer = codec_tests.encoding(tmp_path)
    case = fixture.fixture_case(tmp_path, count=2)
    manifest = json.loads(case[1].read_text())
    rows = []
    for i, source in enumerate(manifest["provenance"]):
        row = dict(source, corpus_line_1based=1, client_index=0,
                   hash_id_scope="local", hash_ids=[10, 11 + i],
                   agent_id=None, subagent_type=None)
        row["prompt_token_sha256"] = token_digest(codec.tokens(row))
        rows.append(row)
    case[0].write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest.update(provenance=rows, sha256=fixture.digest(case[0]),
                    prompt_encoding={"path": codec.path, "sha256": codec.sha256},
                    cache_policy=cache_on_policy())
    case[1].write_text(json.dumps(manifest))
    return case, codec, tokenizer, rows


def test_encoding_policy_and_row_digests_travel_with_case(tmp_path):
    case, codec, _, rows = prefix_case(tmp_path)
    identity = fixture.load(case)
    assert identity["prompt_encoding"]["sha256"] == codec.sha256
    assert identity["prompt_token_sha256"] == [r["prompt_token_sha256"] for r in rows]
    portable = diagnostic.identity(identity)
    assert portable["cache_policy"] == cache_on_policy()
    assert portable["prompt_encoding"] == {"sha256": codec.sha256}
    assert "path" not in portable["prompt_encoding"]
    blob = {"run": {"requests": 2, "prompt_encoding": {
        "corpus_encoding": codec.evidence(identity["prompt_token_sha256"], "measured")}},
        "results": [{}, {}], "workload": rows}
    diagnostic.check_result(blob, identity)
    blob["run"]["prompt_encoding"]["corpus_encoding"]["phase"] = "warmup"
    with pytest.raises(ValueError, match="attest"):
        diagnostic.check_result(blob, identity)


@pytest.mark.parametrize("change", ["missing_encoding", "policy", "row_digest", "source_blocks"])
def test_inconsistent_prefix_manifest_is_refused(tmp_path, change):
    case, _, _, _ = prefix_case(tmp_path)
    manifest = json.loads(case[1].read_text())
    if change == "missing_encoding":
        del manifest["prompt_encoding"]
    elif change == "policy":
        manifest["cache_policy"]["enable_prefix_caching"] = False
    elif change == "row_digest":
        manifest["provenance"][0]["prompt_token_sha256"] = "0" * 64
    else:
        manifest["provenance"][0]["hash_ids"][1] = 99
    case[1].write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        fixture.load(case)


def replay_fixture(tmp_path, monkeypatch, *, reset_failure=False):
    case, codec, tokenizer, rows = prefix_case(tmp_path)
    events = []
    monkeypatch.setattr(replay, "_load_prompt_tokenizer", lambda *_: tokenizer)
    monkeypatch.setattr(replay, "_clock_of", lambda *_: "wall")
    monkeypatch.setattr(replay, "_prepare", lambda *_: events.append("prepare") or {
        "drained": True, "records": [], "boundary_engine_time": 1.0})
    def reset(*_):
        events.append("reset")
        if reset_failure:
            raise ValueError("busy native reader")
        return {"schema": "compass.cache_reset/1", "acknowledged": True}
    monkeypatch.setattr(replay, "_reset_prefix_cache", reset)
    monkeypatch.setattr(replay, "_prefix_cache_snapshot", lambda *_: events.append("end") or {
        "schema": "compass.cache_snapshot/1", "ranks": [{"counters": {}}]})
    async def submit(_base, payloads, arrivals, **_kwargs):
        events.append("measured")
        bodies = [json.loads(x) for x in payloads]
        assert bodies[0]["prompt"][:64] == bodies[1]["prompt"][:64]
        return [{"index": i, "ok": True, "response": {
            "id": str(i), "choices": [{"finish_reason": "length"}],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": body["max_tokens"]}}}
            for i, body in enumerate(bodies)], {"pacing_started_at": time.time()}
    monkeypatch.setattr(replay, "_submit_requests", submit)
    monkeypatch.setattr(replay, "_send", lambda *_: {"arrival_barrier": {"timed_out": False}})
    monkeypatch.setattr(replay.urllib.request, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(OSError("stub")))
    output = tmp_path / "replayed.json"
    args = ["--port", "8500", "--model", codec_tests.MODEL, "--trace", str(case[0]), "--num-requests", "0",
            "--prompt-encoding", codec.path, "--prompt-encoding-sha256", codec.sha256,
            "--prepare", "1", "--out", str(output), "--check-lengths"]
    return args, output, events, codec, rows


def test_replay_resets_after_warmup_before_measured_registration(tmp_path, monkeypatch):
    args, output, events, codec, rows = replay_fixture(tmp_path, monkeypatch)
    assert replay.main(args) == 0
    assert events == ["prepare", "reset", "measured", "end"]
    blob = json.loads(output.read_text())
    assert blob["workload"] == rows
    manifest = blob["run"]
    assert manifest["cache_boundary"]["acknowledged"] is True
    assert manifest["cache_state_after"]["schema"] == "compass.cache_snapshot/1"
    assert manifest["prompt_encoding"]["corpus_encoding"] == codec.evidence(
        [row["prompt_token_sha256"] for row in rows], "measured")


def test_failed_reset_sends_no_measured_requests(tmp_path, monkeypatch):
    args, output, events, _, _ = replay_fixture(tmp_path, monkeypatch, reset_failure=True)
    assert replay.main(args) == 3
    assert events == ["prepare", "reset"]
    assert not output.exists()


def test_modelled_client_requires_a_fresh_virtual_boundary(monkeypatch):
    snapshot = {"indexes": {"kv": 0, "state": 0}, "counters": {"hits": 0},
                "quiescence": {"idle": True, "reasons": [], "state_readers": 0,
                               "state_deferred_readers": 0}}
    barrier = {"acknowledged": True, "kind": "modelled_no_device", "workers_completed": 1,
               "core_timeline_drained": True, "core_timeline_advance_seconds": 0,
               "core_virtual_elapsed_seconds": 0, "retained_output_requests": 0}
    receipt = {"schema": "compass.cache_reset/1", "acknowledged": True,
               "ranks": [{"acknowledged": True, "before": snapshot, "after": snapshot,
                          "worker_barrier": barrier, "counter_delta": {"hits": 0}}]}
    monkeypatch.setattr(replay, "_clock_of", lambda *_: "virtual")
    monkeypatch.setattr(replay, "_send", lambda *_: receipt)
    assert replay._reset_prefix_cache("stub", 1) == receipt
    barrier["core_virtual_elapsed_seconds"] = 1.0
    with pytest.raises(ValueError, match="fresh server"):
        replay._reset_prefix_cache("stub", 1)
