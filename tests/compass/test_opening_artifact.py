"""The client writes usable evidence only for a complete, pinned opening."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.replay_plan import OpeningPlan

fixture_spec = importlib.util.spec_from_file_location(
    "opening_artifact_fixture", Path(__file__).with_name("test_opening_release.py"))
fixture = importlib.util.module_from_spec(fixture_spec)
sys.modules[fixture_spec.name] = fixture
fixture_spec.loader.exec_module(fixture)
opening_fixture = fixture.opening_fixture


spec = importlib.util.spec_from_file_location(
    "opening_artifact_replay", Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.mark.parametrize("damage", [None, "wrong_tokens", "early_release", "unused_calendar"])
def test_client_artifact_binds_payload_cache_and_release_evidence(tmp_path, monkeypatch, damage):
    path, digest, _, rows = opening_fixture(tmp_path, 1.)
    clock = "virtual" if damage == "unused_calendar" else "wall"
    events = []
    monkeypatch.setattr(replay, "_clock_of", lambda *_: clock)
    monkeypatch.setattr(replay, "_load_prompt_tokenizer", lambda *_: SimpleNamespace(init_kwargs={}))
    monkeypatch.setattr(OpeningPlan, "verify_tokenizer", lambda *_: None)
    monkeypatch.setattr(replay, "_reset_prefix_cache", lambda *_: events.append("reset") or {
        "ranks": [{"after": {"policy": cache_on_policy()}}], "acknowledged": True})
    monkeypatch.setattr(replay, "_prefix_cache_snapshot", lambda *_: events.append("snapshot") or {
        "schema": "compass.cache_snapshot/1", "ranks": [{"policy": cache_on_policy()}]})
    records = [{"request_id": f"r{i}", "seq_id": str(i), "shared_preprocessing": {
        "prompt_token_sha256": row["prompt_token_sha256"], "input_tokens": row["input_tokens"]}}
        for i, row in enumerate(rows)]
    if damage == "wrong_tokens":
        records[0]["shared_preprocessing"]["prompt_token_sha256"] = "0" * 64

    async def submit(_base, payloads, arrivals, **kwargs):
        events.append("submit")
        assert kwargs["endpoint"] == "/v1/chat/completions" and kwargs["streaming"]
        assert kwargs["response_gated"] == (clock == "wall")
        assert arrivals == [0., 1.]
        assert all(json.loads(body)["ignore_eos"] for body in payloads)
        return [{"index": i, "ok": True, "response": {
            "id": f"r{i}", "choices": [{"finish_reason": "length"}],
            "usage": {"prompt_tokens": row["input_tokens"], "completion_tokens": row["output_tokens"]}},
            "send_timing": {"request_started_offset_s": (1.5 if damage == "early_release" else 2.1) if i else 0.,
                            "finished_offset_s": 4. if i else 2.}}
            for i, row in enumerate(rows)], {"pacing_started_at": 1.}

    monkeypatch.setattr(replay, "_submit_requests", submit)
    monkeypatch.setattr(replay, "_send", lambda *_: {
        "clock": clock, "requests": records, "arrival_barrier": {"timed_out": False}})
    provenance = {"server_revision": "fixture-committed-source", "server_code_sha256": "a" * 64,
                  "compass": {"opening_plan_sha256": digest, "loaded_inputs": {"ranks": []}}}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(provenance).encode()

    monkeypatch.setattr(replay.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    out = tmp_path / "result.json"
    code = replay.main(["--port", "1", "--model", "fixture", "--opening-plan", str(path),
                        "--opening-plan-sha256", digest, "--out", str(out)])
    result = json.loads(out.read_text())
    assert events == ["reset", "submit", "snapshot"]
    assert result["run"]["aiperf_opening"]["input"]["sha256"] == digest
    assert result["run"]["server_revision"] == "fixture-committed-source"
    assert result["run"]["prompt_encoding"]["kind"] == "chat_messages"
    assert result["run"]["complete"] == (damage is None)
    assert code == (0 if damage is None else replay.INCOMPLETE_EXIT)
