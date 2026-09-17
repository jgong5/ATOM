"""The corpus reader must see the whole corpus, and say what it left out.

Two failures here are silent rather than loud, which is why they get tests
rather than a comment. A reader that iterates a session's `requests` and takes
what it finds drops 58% of the 256k corpus -- the sub-agent leaves -- and still
writes a well-formed trace that replays cleanly. And `hash_id_scope` is
"local", so merging two sessions' ids without namespacing them invents prefix
reuse neither session had; the run does not fail, it just reports a cache hit
rate nothing earned.
"""
import importlib.util
import json
from pathlib import Path

import pytest

CORPUS = Path.home() / ".cache/huggingface/cc-traces-256k/traces.jsonl"


def _module():
    spec = importlib.util.spec_from_file_location(
        "cc_traces_mod",
        Path(__file__).resolve().parents[2] / "scripts/compass/cc_traces.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaf(t, n_in, out=8, ids=None, **extra):
    ids = list(range(n_in // 64)) if ids is None else ids
    row = {"t": t, "model": "m", "in": n_in, "out": out, "hash_ids": ids,
           "api_time": 1.0, "type": "s"}
    row.update(extra)
    return row


def _session(requests, sid="abc"):
    return {"id": sid, "models": ["m"], "block_size": 64,
            "hash_id_scope": "local", "requests": requests}


def _write(tmp_path, sessions):
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in sessions))
    return str(path)


class TestSubagentsAreRequestsToo:
    """The 58% that a flat read loses."""

    def test_a_wrapper_contributes_its_leaves_and_not_itself(self, tmp_path):
        mod = _module()
        wrapper = {"t": 10.0, "type": "subagent", "agent_id": "a",
                   "requests": [_leaf(10.0, 640), _leaf(11.0, 1280)]}
        path = _write(tmp_path, [_session([_leaf(0.0, 640), wrapper])])
        rows, stats = mod.extract(path)
        assert stats["leaves_in_corpus"] == 3
        assert len(rows) == 3
        assert [r["input_tokens"] for r in rows] == [640, 640, 1280]

    def test_nested_arrivals_are_kept_as_the_session_clock_gives_them(self, tmp_path):
        """Nested `t` is absolute, not an offset from the wrapper -- checked on
        all 1,697 wrappers in the 256k corpus, none below its wrapper's `t`."""
        mod = _module()
        wrapper = {"t": 10.0, "type": "subagent", "requests": [_leaf(112.0, 640)]}
        rows, _ = mod.extract(_write(tmp_path, [_session([wrapper])]))
        assert rows[0]["arrival_s"] == 112.0

    def test_a_sub_agent_fan_out_survives_the_sort(self, tmp_path):
        """Several requests in the server at once is the thing a sub-agent
        contributes that a length histogram cannot; they must stay coincident."""
        mod = _module()
        wrapper = {"t": 5.0, "type": "subagent",
                   "requests": [_leaf(5.0, 640), _leaf(5.0, 704), _leaf(5.0, 768)]}
        rows, _ = mod.extract(_write(tmp_path, [_session([wrapper])]))
        assert [r["arrival_s"] for r in rows] == [5.0, 5.0, 5.0]


class TestSessionsDoNotShareIds:
    def test_each_session_gets_its_own_index(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640)], "a"),
                                 _session([_leaf(0.0, 640)], "b")])
        rows, _ = mod.extract(path)
        assert {r["session"] for r in rows} == {0, 1}
        assert {r["session_id"] for r in rows} == {"a", "b"}

    def test_reuse_is_counted_per_session_not_globally(self, tmp_path):
        """Both sessions use ids 0..9. That is ten blocks twice, not ten
        blocks reused -- the mistake this guards is worth 50% of a hit rate."""
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640)], "a"),
                                 _session([_leaf(0.0, 640)], "b")])
        rows, _ = mod.extract(path)
        assert mod.summarise(rows)["reusable_blocks"] == 0

    def test_a_repeated_id_inside_one_session_is_reuse(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640),
                                           _leaf(1.0, 1280)], "a")])
        rows, _ = mod.extract(path)
        # Second request re-sends blocks 0..9 and adds 10..19.
        assert mod.summarise(rows)["reusable_blocks"] == 10
        assert mod.summarise(rows)["reusable_tokens"] == 640


class TestItRefusesCorpusItCannotReplay:
    def test_a_different_block_size_is_refused(self, tmp_path):
        mod = _module()
        s = _session([_leaf(0.0, 640)])
        s["block_size"] = 32
        with pytest.raises(ValueError, match="block_size"):
            mod.extract(_write(tmp_path, [s]))

    def test_a_global_hash_id_scope_is_refused(self, tmp_path):
        """Namespacing per session would be wrong for a global scope: it would
        split reuse the corpus recorded, silently."""
        mod = _module()
        s = _session([_leaf(0.0, 640)])
        s["hash_id_scope"] = "global"
        with pytest.raises(ValueError, match="hash_id_scope"):
            mod.extract(_write(tmp_path, [s]))


class TestEveryDropIsCounted:
    def test_short_requests_are_dropped_and_counted(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 64), _leaf(1.0, 640)])])
        rows, stats = mod.extract(path)
        assert len(rows) == 1 and stats["dropped_short"] == 1

    def test_keeping_them_is_one_flag(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 64), _leaf(1.0, 640)])])
        rows, stats = mod.extract(path, min_input_tokens=0)
        assert len(rows) == 2 and stats["dropped_short"] == 0

    def test_a_request_asking_for_no_output_is_dropped(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640, out=0)])])
        rows, stats = mod.extract(path)
        assert rows == [] and stats["dropped_no_output"] == 1

    def test_over_the_context_limit_is_dropped(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640, out=100000)])])
        rows, stats = mod.extract(path, max_total_tokens=1024)
        assert rows == [] and stats["dropped_total_tokens"] == 1

    def test_max_requests_truncates_by_arrival_and_says_so(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(2.0, 640), _leaf(0.0, 704),
                                           _leaf(1.0, 768)])])
        rows, stats = mod.extract(path, max_requests=2)
        assert [r["arrival_s"] for r in rows] == [0.0, 1.0]
        assert stats["dropped_by_max_requests"] == 1


class TestSessionSelection:
    def test_peak_filter_keeps_whole_sessions(self, tmp_path):
        """Whole, because cutting a session apart removes the turns that would
        have hit the cache and lowers reuse for reasons that are this filter."""
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640), _leaf(1.0, 1280)], "a"),
                                 _session([_leaf(0.0, 640)], "b")])
        rows, _ = mod.extract(path, session_min_peak_tokens=1280)
        assert {r["session_id"] for r in rows} == {"a"}
        assert len(rows) == 2

    def test_span_filter_bounds_a_paced_run(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640), _leaf(9999.0, 640)], "a"),
                                 _session([_leaf(0.0, 640), _leaf(10.0, 640)], "b")])
        rows, _ = mod.extract(path, max_session_span_s=100.0)
        assert {r["session_id"] for r in rows} == {"b"}

    def test_offset_staggers_sessions_that_all_start_at_zero(self, tmp_path):
        """Every corpus session's clock starts at t=0, so the default overlays
        them. That is a choice about concurrency and it has to be visible."""
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640)], "a"),
                                 _session([_leaf(0.0, 640)], "b")])
        flat, _ = mod.extract(path)
        assert [r["arrival_s"] for r in flat] == [0.0, 0.0]
        spread, _ = mod.extract(path, session_offset_s=60.0)
        assert [r["arrival_s"] for r in spread] == [0.0, 60.0]


class TestTheRowsAreWhatReplayReads:
    def test_it_carries_the_ground_truth_replay_would_otherwise_lose(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(0.0, 640, ttft=0.9)])])
        rows, _ = mod.extract(path)
        assert rows[0]["api_time_s"] == 1.0
        assert rows[0]["source_ttft_s"] == 0.9
        assert rows[0]["hash_ids"] == list(range(10))

    def test_rows_come_out_in_arrival_order(self, tmp_path):
        mod = _module()
        path = _write(tmp_path, [_session([_leaf(5.0, 640), _leaf(1.0, 640)], "a"),
                                 _session([_leaf(3.0, 640)], "b")])
        rows, _ = mod.extract(path)
        assert [r["arrival_s"] for r in rows] == [1.0, 3.0, 5.0]


@pytest.mark.skipif(not CORPUS.exists(), reason="256k corpus not present")
class TestAgainstTheRealCorpus:
    """Numbers measured off `cc-traces-weka-062126-256k` itself.

    These are the check that the recursion is finding everything: a flat read
    gives 28,444, and 28,444 rows is a perfectly valid-looking trace.
    """

    def test_it_finds_every_leaf(self):
        mod = _module()
        _, stats = mod.extract(str(CORPUS), session_min_peak_tokens=10 ** 9)
        assert stats["leaves_in_corpus"] == 68266
        assert stats["sessions_in_corpus"] == 393


def _replay():
    spec = importlib.util.spec_from_file_location(
        "replay_cctr",
        Path(__file__).resolve().parents[2] / "scripts/compass/replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay_workload(module, rows, scale=1.0):
    import argparse
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        trace = Path(d) / "t.jsonl"
        trace.write_text("\n".join(json.dumps(r) for r in rows))
        args = argparse.Namespace(trace=str(trace), num_requests=0,
                                  time_scale=scale, input_tokens=8,
                                  output_tokens=1, rate=0.0, seed=0)
        return module._workload(args)


class TestReplayKeepsWhatTheTraceRecorded:
    """`_workload` used to project every row down to three fields.

    A trace could carry `hash_ids`, `session` and `api_time_s` and the run would
    discard all three between reading the file and sending the request, so the
    artifact showed no sign they had ever been there. Nothing failed; the run
    was simply a different workload from the one on disk.
    """

    ROW = {"arrival_s": 0.0, "input_tokens": 640, "output_tokens": 8,
           "hash_ids": list(range(10)), "session": 3, "api_time_s": 1.5,
           "source_ttft_s": 0.4}

    def test_the_sharing_survives_the_read(self):
        row = _replay_workload(_replay(), [self.ROW])[0]
        assert row["hash_ids"] == list(range(10))
        assert row["session"] == 3

    def test_the_recorded_duration_survives_the_read(self):
        row = _replay_workload(_replay(), [self.ROW])[0]
        assert row["api_time_s"] == 1.5
        assert row["source_ttft_s"] == 0.4

    def test_a_trace_without_them_still_reads(self):
        row = _replay_workload(_replay(), [{"arrival_s": 0.0,
                                            "input_tokens": 64,
                                            "output_tokens": 4}])[0]
        assert "hash_ids" not in row and row["input_tokens"] == 64


class TestReplayBuildsTheRightKindOfPrompt:
    """Two opposite constructions, and the row decides which."""

    def test_rows_with_hash_ids_share_their_leading_blocks(self):
        module = _replay()
        a = module._prompt({"input_tokens": 192, "hash_ids": [1, 2, 3],
                            "session": 0}, 0).split()
        b = module._prompt({"input_tokens": 192, "hash_ids": [1, 2, 9],
                            "session": 0}, 1).split()
        assert a[:128] == b[:128]
        assert a[128:144] != b[128:144]

    def test_the_index_does_not_break_the_sharing(self):
        """`prompt_of_tokens` varies the opening by request index on purpose.
        If that leaked into this path, two turns of one session would never
        share a block and the replay would look like a cache that never hits."""
        module = _replay()
        row = {"input_tokens": 128, "hash_ids": [4, 5], "session": 2}
        assert module._prompt(row, 0) == module._prompt(row, 77)

    def test_rows_without_hash_ids_still_defeat_the_cache(self):
        module = _replay()
        a = module._prompt({"input_tokens": 64}, 0)
        b = module._prompt({"input_tokens": 64}, 1)
        assert a != b and len(a.split()) == len(b.split()) == 64

    def test_two_sessions_with_the_same_ids_do_not_share(self):
        module = _replay()
        a = module._prompt({"input_tokens": 128, "hash_ids": [1, 2],
                            "session": 0}, 0)
        b = module._prompt({"input_tokens": 128, "hash_ids": [1, 2],
                            "session": 1}, 0)
        assert a != b


class TestAnOutputLengthNotACeiling:
    def test_the_flag_exists_and_sets_ignore_eos(self):
        """`max_tokens` is a ceiling. A row asking for 376 tokens that stops at
        3 on an EOS the recorded run never hit is a shorter workload reported
        as a successful replay."""
        source = (Path(__file__).resolve().parents[2]
                  / "scripts/compass/replay.py").read_text()
        assert '"--ignore-eos"' in source
        assert 'body["ignore_eos"] = True' in source

    def test_the_artifact_records_whether_it_was_used(self):
        source = (Path(__file__).resolve().parents[2]
                  / "scripts/compass/replay.py").read_text()
        assert '"ignore_eos": bool(args.ignore_eos)' in source
        assert '"rows_with_hash_ids"' in source


class TestTheSessionCapKeepsAPrefixOfTheConversation:
    """A cap that sampled turns instead of taking the first N would keep the
    length histogram and destroy the reuse, which is most of this corpus's
    prefill work. So it must be a prefix, and it must say what it cut."""

    def _corpus(self, tmp_path, turns):
        mod = _module()
        reqs = [{"type": "s", "t": float(i), "in": 640 * (i + 1), "out": 8,
                 "api_time": 1.0, "hash_ids": list(range(10 * (i + 1)))}
                for i in range(turns)]
        path = tmp_path / "corpus.jsonl"
        path.write_text(json.dumps({"id": "s0", "block_size": 64,
                                    "hash_id_scope": "local",
                                    "requests": reqs}) + "\n")
        return mod, path

    def test_it_keeps_the_earliest_turns(self, tmp_path):
        mod, path = self._corpus(tmp_path, 10)
        rows, stats = mod.extract(str(path), max_requests_per_session=3)
        assert [r["arrival_s"] for r in rows] == [0.0, 1.0, 2.0]
        assert stats["dropped_past_session_cap"] == 7

    def test_an_uncapped_session_is_untouched(self, tmp_path):
        mod, path = self._corpus(tmp_path, 3)
        rows, stats = mod.extract(str(path), max_requests_per_session=10)
        assert len(rows) == 3
        assert stats["dropped_past_session_cap"] == 0

    def test_the_prefix_still_shares_its_blocks(self, tmp_path):
        mod, path = self._corpus(tmp_path, 10)
        rows, _ = mod.extract(str(path), max_requests_per_session=3)
        # Turn k re-sends turn k-1's blocks. Sampling rather than prefixing
        # would leave rows whose leading ids no longer overlap.
        assert mod.summarise(rows)["reusable_blocks"] == 10 + 20
