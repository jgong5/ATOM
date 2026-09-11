"""The cc-traces acceptance workload is selected by a rule, and the rule holds.

What these cover is not that the selector runs. It is the properties the
protocol leans on when it calls the result held out and source-timed: that the
development session cannot come back, that a `subagent` summary is never served
as a request, that a source length or output is never edited to make a session
usable, that arrivals inside a session are the source's raw intervals, that the
one constructed quantity -- the short class's simultaneous session start -- is
declared, and that an edit to either the file or the rule is visible afterwards.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workload = _load("cc_traces_workload")

BLOCK = 64


def _request(t, tokens, produced, kind="s", blocks=None):
    """One corpus request row. `in` is tokens; `hash_ids` is one id per block."""
    # Corpus `in` values are block-aligned; the fixture aligns so the selector's
    # own check passes, and the mismatch case below breaks it on purpose.
    tokens = (tokens // BLOCK) * BLOCK
    n = blocks if blocks is not None else tokens // BLOCK
    return {
        "t": t,
        "type": kind,
        "model": "claude-opus-4-8",
        "in": tokens,
        "out": produced,
        "hash_ids": list(range(n)),
        "api_time": 1.0,
        "ttft": 0.5,
    }


def _session(session_id, turns, extra=()):
    return {
        "id": session_id,
        "block_size": BLOCK,
        "hash_id_scope": "local",
        "models": ["claude-opus-4-8"],
        "requests": [_request(t, i, o) for t, i, o in turns] + list(extra),
    }


def _subagent(t, nested=()):
    """A `subagent` row: a summary with no `in`/`out`, wrapping real requests.

    In the corpus the nested requests are on the session's own clock -- the
    wrapper's `t` equals its first nested request's `t` -- and are not repeated
    at the top level. So they are load that nothing else in the session records.
    """
    rows = [_request(t, tokens, produced, kind="n") for t, tokens, produced in nested]
    return {
        "t": t,
        "type": "subagent",
        "agent_id": "subagent_001",
        "subagent_type": "Subagent",
        "duration_ms": 1000,
        "total_tokens": 331009,
        "status": "completed",
        "requests": rows,
        "models": ["claude-opus-4-8"],
    }


def _write(tmp_path, lines, monkeypatch):
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    monkeypatch.setitem(workload.CORPUS, "sha256", workload.digest_file(path))
    return path


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A small corpus shaped like the real one, plus the traps it contains."""
    lines = [
        # the development session, which must never be selected
        _session("dev", [(0.0, 512, 10), (1.0, 40_000, 20), (2.0, 80_000, 30)]),
        # a long candidate whose window spans hours: eligible only if idle time
        # were compressed, which the rule does not do
        _session(
            "long-idle",
            [(0.0, 512, 10)]
            + [(3600.0 * i, 60_000 + 1_000 * i, 100) for i in range(1, 20)],
        ),
        # a long candidate that would fit, but a delegated agent is running
        # requests of its own inside the window: concurrent load this workload
        # does not replay, so the window is not what it appears to be
        _session(
            "long-nested",
            [(0.0, 512, 10)]
            + [(10.0 * i, 60_000 + 1_000 * i, 100) for i in range(1, 20)],
            extra=[_subagent(35.0, nested=[(35.0, 30_000, 50), (44.0, 31_000, 60)])],
        ),
        # a long candidate that fits in its own time, whose delegated agent runs
        # long after the window has been served
        _session(
            "long-a",
            [(0.0, 512, 10)]
            + [(10.0 * i, 60_000 + 1_000 * i, 100) for i in range(1, 20)],
            extra=[_subagent(5000.0, nested=[(5000.0, 30_000, 50)])],
        ),
        # short openings: one small turn, then the conversation grows
        _session("short-a", [(0.0, 256, 5), (0.5, 640, 6), (3600.0, 90_000, 40)]),
        # a session whose opening produced nothing: no TPOT, and raising the
        # zero would be an edit to a source output length
        _session("short-zero-out", [(0.0, 320, 0), (1.0, 448, 7)]),
        _session("short-c", [(0.0, 448, 8), (200.0, 70_000, 50)]),
        # an opening that is served while a delegated agent is mid-request
        _session(
            "short-busy",
            [(0.0, 448, 8), (300.0, 70_000, 50)],
            extra=[_subagent(0.5, nested=[(0.5, 30_000, 50)])],
        ),
        _session("short-d", [(0.0, 384, 9), (4.0, 512, 11), (50.0, 80_000, 60)]),
    ]
    monkeypatch.setattr(workload, "DEVELOPMENT_SESSIONS", ("dev",))
    # The registered short rule asks for 64 sessions and a corpus this size
    # cannot hold them. The rule's *shape* is what these tests are about, so the
    # count is scaled down and every other constant is left as registered.
    monkeypatch.setitem(workload.RULES, "short", workload.ShortRule(sessions=6))
    monkeypatch.setitem(
        workload.RULES, "long", workload.LongRule(volume_band=(1_000_000, 2_500_000))
    )
    return _write(tmp_path, lines, monkeypatch)


class TestWhatARowIs:
    def test_a_subagent_summary_is_never_served(self, corpus):
        """It has no `in` and no `out`; reading it as a request invents a row."""
        text, manifest = workload.build(str(corpus), "long")
        rows = [json.loads(line) for line in text.splitlines()]
        assert len(rows) == workload.RULES["long"].window
        assert all(r["input_tokens"] > 0 for r in rows)
        assert manifest["scanned_subagent_wrapper_rows"] >= 1
        assert manifest["scanned_nested_requests"] >= 1

    def test_a_length_the_corpus_contradicts_makes_the_session_ineligible(
        self, tmp_path, monkeypatch
    ):
        """`in` must equal `len(hash_ids) * block_size`; it is checked, not assumed."""
        bad = _session("short-bad", [(0.0, 448, 8)])
        bad["requests"][0]["hash_ids"] = [0, 1]  # claims 128 tokens
        good = [_session(f"ok-{k}", [(0.0, 448, 8)]) for k in range(3)]
        path = _write(tmp_path, [bad] + good, monkeypatch)
        monkeypatch.setattr(workload, "DEVELOPMENT_SESSIONS", ())
        monkeypatch.setitem(workload.RULES, "short", workload.ShortRule(sessions=3))
        _text, manifest = workload.build(str(path), "short")
        assert "short-bad" not in [s["id"] for s in manifest["sessions"]]
        with pytest.raises(workload.Ineligible):
            workload.checked_tokens(bad["requests"][0], BLOCK)

    def test_the_development_session_is_never_selected(self, corpus):
        for klass in ("long", "short"):
            _text, manifest = workload.build(str(corpus), klass)
            assert "dev" not in [s["id"] for s in manifest["sessions"]]


class TestDelegatedAgentLoadIsNotDropped:
    """`subagent` wrappers are summaries, but the requests inside them are real.

    They are 39822 actual inference calls across the corpus, on the session's
    own clock, not repeated at top level. Omitting the wrapper row is right;
    omitting them silently would understate the load a selected segment was
    served under. So a scope that overlaps one is refused, and what was
    selected carries the proof that it overlaps none.
    """

    def test_nested_timestamps_are_read_on_the_session_clock(self, corpus):
        """The wrapper's `t` equals its first nested `t` -- session-relative."""
        wrapper = _subagent(35.0, nested=[(35.0, 30_000, 50), (44.0, 31_000, 60)])
        nested = wrapper["requests"]
        assert nested[0]["t"] == wrapper["t"]
        assert len(workload.nested_in_scope(nested, 0.0, 191.0)) == 2
        # Read as parent-relative (0.0 and 9.0 after the wrapper) they would
        # land in a different place entirely; this is the reading being fixed.
        assert workload.nested_in_scope(nested, 100.0, 191.0) == []

    def test_a_window_with_delegated_requests_inside_it_is_refused(self, corpus):
        """`long-nested` fits every other criterion and comes first in order."""
        _text, manifest = workload.build(str(corpus), "long")
        assert manifest["sessions"][0]["id"] == "long-a"
        why = {c["id"]: c["rejected"] for c in manifest["sessions_rejected"]}
        assert "subagent requests run inside this window" in why["long-nested"]

    def test_the_selected_window_carries_the_proof_it_overlaps_none(self, corpus):
        _text, manifest = workload.build(str(corpus), "long")
        chosen = manifest["sessions"][0]
        assert chosen["nested_requests_in_session"] >= 1
        assert chosen["nested_requests_in_scope"] == 0
        assert manifest["selection_nested_requests_in_scope"] == 0
        assert manifest["selection_nested_requests_in_selected_sessions"] >= 1
        assert manifest["scanned_nested_requests"] >= 1

    def test_an_opening_served_alongside_one_is_refused(self, corpus):
        """`short-busy` opens fine, but a delegated call is in flight with it."""
        _text, manifest = workload.build(str(corpus), "short")
        assert "short-busy" not in [s["id"] for s in manifest["sessions"]]
        why = {c["id"]: c["rejected"] for c in manifest["sessions_rejected"]}
        assert "while this opening is being served" in why["short-busy"]

    def test_a_later_delegated_call_does_not_disqualify_a_clear_opening(self, corpus):
        """Scope is the selected segment, not the whole session."""
        _text, manifest = workload.build(str(corpus), "short")
        ids = [s["id"] for s in manifest["sessions"]]
        assert "long-a" in ids  # its subagent runs at t=5000, long after turn 0
        assert manifest["selection_nested_requests_in_scope"] == 0


class TestLongIsSourcePaced:
    def test_the_window_is_taken_whole_and_in_order(self, corpus):
        text, manifest = workload.build(str(corpus), "long")
        rows = [json.loads(line) for line in text.splitlines()]
        assert len(manifest["sessions"]) == 1
        assert manifest["sessions"][0]["id"] == "long-a"
        assert [r["request_index"] for r in rows] == list(range(20))

    def test_arrivals_are_the_source_intervals_with_one_origin_shift(self, corpus):
        text, manifest = workload.build(str(corpus), "long")
        rows = [json.loads(line) for line in text.splitlines()]
        assert all(
            r["arrival_s"] == pytest.approx(r["source_t_s"] + r["origin_shift_s"])
            for r in rows
        )
        assert len({r["origin_shift_s"] for r in rows}) == 1
        assert manifest["gaps_clipped"] == 0
        assert rows[1]["arrival_s"] == pytest.approx(10.0)
        assert rows[-1]["arrival_s"] == pytest.approx(190.0)

    def test_an_idle_session_is_rejected_rather_than_compressed(self, corpus):
        """`long-idle` comes first in the corpus and is the denser session's rival.

        The only way to take it would be to shorten its gaps. It is passed over
        instead, and the manifest says why in those words.
        """
        _text, manifest = workload.build(str(corpus), "long")
        assert manifest["sessions"][0]["id"] == "long-a"
        why = {c["id"]: c["rejected"] for c in manifest["sessions_rejected"]}
        assert "spans" in why["long-idle"]

    def test_a_session_with_a_silent_turn_is_not_patched_up(
        self, tmp_path, monkeypatch
    ):
        """out=0 has no TPOT. The session goes, the output length stays."""
        turns = [(10.0 * i, 60_000, 100) for i in range(20)]
        silent = _session("silent", [(0.0, 60_000, 0)] + turns[1:])
        whole = _session("whole", turns)
        path = _write(tmp_path, [silent, whole], monkeypatch)
        monkeypatch.setattr(workload, "DEVELOPMENT_SESSIONS", ())
        _text, manifest = workload.build(str(path), "long")
        assert manifest["sessions"][0]["id"] == "whole"
        assert manifest["outputs_altered"] == 0


class TestShortIsADeclaredCohort:
    def test_only_leading_runs_are_taken(self, corpus):
        """A short turn an hour into a session is not an opening."""
        text, _ = workload.build(str(corpus), "short")
        rows = [json.loads(line) for line in text.splitlines()]
        assert rows
        for row in rows:
            assert row["request_index"] <= 1

    def test_every_session_starts_at_zero_and_says_so(self, corpus):
        text, manifest = workload.build(str(corpus), "short")
        rows = [json.loads(line) for line in text.splitlines()]
        firsts = {}
        for row in rows:
            firsts.setdefault(row["session"], row["arrival_s"])
        assert set(firsts.values()) == {0.0}
        assert manifest["arrivals"]["across_sessions"].startswith("constructed")
        assert manifest["rule"]["alignment"] == "session_start_at_zero"

    def test_within_a_session_the_source_intervals_survive(self, corpus):
        """`short-d` opens with two turns four seconds apart; four it stays."""
        text, _ = workload.build(str(corpus), "short")
        rows = [json.loads(line) for line in text.splitlines()]
        mine = sorted(
            (r for r in rows if r["session"] == "short-d"),
            key=lambda r: r["request_index"],
        )
        assert len(mine) == 2
        assert mine[1]["arrival_s"] - mine[0]["arrival_s"] == pytest.approx(4.0)

    def test_a_session_whose_opening_is_silent_contributes_nothing(self, corpus):
        _text, manifest = workload.build(str(corpus), "short")
        assert "short-zero-out" not in [s["id"] for s in manifest["sessions"]]
        assert manifest["outputs_altered"] == 0

    def test_output_lengths_are_the_source_s(self, corpus):
        for klass in ("long", "short"):
            text, _ = workload.build(str(corpus), klass)
            rows = [json.loads(line) for line in text.splitlines()]
            assert all(r["output_tokens"] >= 1 for r in rows)


class TestReproducibility:
    def test_the_same_corpus_gives_the_same_bytes(self, corpus):
        first, manifest = workload.build(str(corpus), "long")
        second, again = workload.build(str(corpus), "long")
        assert first == second
        assert manifest["sha256"] == again["sha256"]

    def test_emit_refuses_a_corpus_that_is_not_the_registered_one(
        self, corpus, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(workload.CORPUS, "sha256", "0" * 64)
        rc = workload.main(
            [
                "emit",
                "--class",
                "long",
                "--corpus",
                str(corpus),
                "--out",
                str(tmp_path / "w.jsonl"),
                "--manifest",
                str(tmp_path / "w.manifest.json"),
                "--at",
                "2026-09-11T00:00:00Z",
            ]
        )
        assert rc == 2
        assert not (tmp_path / "w.jsonl").exists()

    def test_verify_catches_an_edited_workload(self, corpus, tmp_path):
        out, man = tmp_path / "w.jsonl", tmp_path / "w.manifest.json"
        assert (
            workload.main(
                [
                    "emit",
                    "--class",
                    "long",
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(out),
                    "--manifest",
                    str(man),
                    "--at",
                    "2026-09-11T00:00:00Z",
                ]
            )
            == 0
        )
        assert (
            workload.main(["verify", "--manifest", str(man), "--corpus", str(corpus)])
            == 0
        )
        rows = out.read_text().splitlines()
        edited = json.loads(rows[0])
        edited["input_tokens"] = 64  # a cheaper request than registered
        out.write_text("\n".join([json.dumps(edited)] + rows[1:]) + "\n")
        assert workload.main(["verify", "--manifest", str(man)]) == 1

    def test_verify_catches_an_edited_rule(self, corpus, tmp_path):
        """Re-emission is what catches a change to the selection itself.

        The file's digest still matches the manifest -- nobody touched the
        file. What changed is what the rule would produce now, and only running
        it again can see that.
        """
        out, man = tmp_path / "w.jsonl", tmp_path / "w.manifest.json"
        workload.main(
            [
                "emit",
                "--class",
                "long",
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--manifest",
                str(man),
                "--at",
                "2026-09-11T00:00:00Z",
            ]
        )
        manifest = json.loads(man.read_text())
        manifest["rule"]["window"] = 5
        man.write_text(json.dumps(manifest))
        assert workload.main(["verify", "--manifest", str(man)]) == 1

    def test_verify_catches_a_restated_arrival_model(self, corpus, tmp_path):
        """Calling a constructed alignment `source_paced` is the claim to catch."""
        out, man = tmp_path / "w.jsonl", tmp_path / "w.manifest.json"
        workload.main(
            [
                "emit",
                "--class",
                "short",
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--manifest",
                str(man),
                "--at",
                "2026-09-11T00:00:00Z",
            ]
        )
        manifest = json.loads(man.read_text())
        manifest["arrivals"]["model"] = "source_paced_open_loop"
        man.write_text(json.dumps(manifest))
        assert workload.main(["verify", "--manifest", str(man)]) == 1


def _workload_bytes(klass: str):
    """The registered workload file, or a skip that says how to get it.

    The `.jsonl` are not in the repository: they are a slice of a licensed
    568 MB corpus, reproduced by `emit` from the manifest that is committed
    beside them. These checks are about those exact bytes, so in a checkout
    without them there is nothing to check rather than something that failed.
    A run that needs them refuses instead -- see `registered_workload` in
    `cc_traces_validate.py`.
    """
    path = ROOT / "atom" / "compass" / f"cc_traces_{klass}.jsonl"
    if not path.exists():
        pytest.skip(
            f"{path.name} is not in this checkout; reproduce it with "
            f"`cc_traces_workload.py emit --class {klass}` against the corpus "
            f"named in cc_traces_{klass}.manifest.json")
    return path


class TestRegisteredArtifacts:
    """The files the protocol names are the ones the rule produces."""

    @pytest.mark.parametrize("klass", ["long", "short"])
    def test_the_registered_workload_matches_its_manifest(self, klass):
        here = ROOT / "atom" / "compass"
        manifest = json.loads((here / f"cc_traces_{klass}.manifest.json").read_text())
        assert (
            workload.digest_file(_workload_bytes(klass))
            == manifest["sha256"]
        )
        assert manifest["corpus"]["sha256"] == workload.CORPUS["sha256"]
        assert manifest["rule"]["kind"] == klass
        assert manifest["generator_sha256"] == workload.digest_file(
            ROOT / "scripts" / "compass" / "cc_traces_workload.py"
        )

    @pytest.mark.parametrize("klass", ["long", "short"])
    def test_the_registered_workload_is_servable(self, klass):
        """Every registered request has a length the engine can be asked for."""
        rows = [
            json.loads(line)
            for line in _workload_bytes(klass).read_text().splitlines()
            if line.strip()
        ]
        assert rows
        assert all(r["input_tokens"] >= workload.MIN_INPUT_TOKENS for r in rows)
        assert all(r["input_tokens"] <= workload.CONTEXT_TOKENS for r in rows)
        assert all(r["input_tokens"] == r["input_blocks"] * 64 for r in rows)
        assert all(r["output_tokens"] >= 1 for r in rows)
        assert all(
            a["arrival_s"] <= b["arrival_s"] for a, b in itertools.pairwise(rows)
        )

    @pytest.mark.parametrize("klass", ["long", "short"])
    def test_nothing_in_the_registered_workload_was_rewritten(self, klass):
        manifest = json.loads(
            (ROOT / "atom" / "compass" / f"cc_traces_{klass}.manifest.json").read_text()
        )
        assert manifest["gaps_clipped"] == 0
        assert manifest["outputs_altered"] == 0
        rows = [
            json.loads(line)
            for line in _workload_bytes(klass).read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            assert row["arrival_s"] == pytest.approx(
                row["source_t_s"] + row["origin_shift_s"]
            )
