"""The CLI preserves the corrected profile and final evidence refusal."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.fixed_absolute import FixedAbsolutePlan
from .test_fixed_absolute_evidence import completed_evidence


spec = importlib.util.spec_from_file_location(
    "fixed_absolute_artifact_replay", Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.mark.parametrize("damage", [None, "source_read"])
def test_client_cli_keeps_source_bound_profile_and_refuses_invalid_evidence(tmp_path, monkeypatch, damage):
    plan, server, engine, results = completed_evidence(tmp_path)
    for result in results:
        result["response"]["choices"] = [{"finish_reason": "length"}]
        result["send_timing"] = {"request_started_offset_s": 0., "finished_offset_s": 1.}
    if damage == "source_read":
        core = server["compass"]["loaded_inputs"]["ranks"][0]["core_inputs"]
        core["inputs"] = [r for r in core["inputs"] if r["role"] != "runtime.fixed_absolute.source_root"]
    server.update(server_revision="source", server_code_sha256="a" * 64)
    monkeypatch.setattr(replay, "_clock_of", lambda *_: "virtual")
    monkeypatch.setattr(replay, "_load_prompt_tokenizer", lambda *_: SimpleNamespace(init_kwargs={}))
    monkeypatch.setattr(FixedAbsolutePlan, "verify_tokenizer", lambda *_: None)
    monkeypatch.setattr(replay, "_reset_prefix_cache", lambda *_: {
        "ranks": [{"after": {"policy": cache_on_policy()}}], "acknowledged": True})
    monkeypatch.setattr(replay, "_prefix_cache_snapshot", lambda *_: {
        "schema": "compass.cache_snapshot/1", "ranks": [{"policy": cache_on_policy()}]})
    monkeypatch.setattr(replay, "_send", lambda *_: engine)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(server).encode()

    monkeypatch.setattr(replay.urllib.request, "urlopen", lambda *_a, **_kw: Response())

    async def submit(_base, payloads, arrivals, **kwargs):
        assert kwargs["fixed_absolute_plan"].loaded_input.sha256 == plan.loaded_input.sha256
        assert kwargs["pace"] is False and kwargs["streaming"] is True
        assert "response_gated" not in kwargs
        assert all(json.loads(body)["compass_workload_size"] == 2 for body in payloads)
        return results, {"pacing_started_at": 1.}

    monkeypatch.setattr(replay, "_submit_requests", submit)
    out = tmp_path / "result.json"
    code = replay.main(["--port", "1", "--model", "fixture", "--fixed-absolute-plan", plan.loaded_input.requested,
        "--fixed-absolute-plan-sha256", plan.loaded_input.sha256, "--out", str(out)])
    blob = json.loads(out.read_text())
    assert blob["run"]["fixed_absolute"]["input"]["sha256"] == plan.loaded_input.sha256
    assert "aiperf_opening" not in blob["run"]
    assert blob["run"]["trace_sha256"] == plan.loaded_input.sha256
    assert blob["run"]["prompt_lengths"] == "passed"
    assert blob["run"]["complete"] is (damage is None)
    assert code == (0 if damage is None else replay.INCOMPLETE_EXIT)
