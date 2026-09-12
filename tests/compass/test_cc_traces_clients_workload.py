"""The clients workloads are selected by a rule, and the rule holds.

The first registration's two classes refuse every window that overlaps a
delegated agent. These two exist to replay exactly that concurrency, so what
has to hold here is different, and it is what the protocol's §1B leans on: that
an episode is a complete busy period taken whole, that an episode holding one
request this engine cannot serve is refused entire rather than repaired by
dropping the request, that a `subagent` summary is ancestry and never load,
that every eligible descendant is offered and counted, that the client counts
are nested so a wider cell is the narrower one plus more roots, that each
client's arrivals are the source's own intervals with one declared origin
shift, and that the refusal census stays a census -- stable categories, with
the measurements that tripped them travelling separately.
"""

from __future__ import annotations

import importlib.util
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


clients = _load("cc_traces_clients_workload")

BLOCK = 64


def _request(t, tokens, produced, kind="s", api_time=1.0, blocks=None):
    """One corpus request row. `in` is tokens; `hash_ids` is one id per block."""
    tokens = (tokens // BLOCK) * BLOCK
    n = blocks if blocks is not None else tokens // BLOCK
    return {
        "t": t,
        "type": kind,
        "model": "claude-opus-4-8",
        "in": tokens,
        "out": produced,
        "hash_ids": list(range(n)),
        "api_time": api_time,
        "ttft": 0.5,
    }


def _subagent(t, nested, agent_id="subagent_001"):
    """A `subagent` row: a summary with no `in`/`out`, wrapping real requests.

    The wrapper carries no length and no output, so it is ancestry. Its nested
    rows are on the root's own clock and appear nowhere else in the session,
    which is why dropping them would drop load the source really offered.
    """
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Subagent",
        "duration_ms": 1000,
        "total_tokens": 331009,
        "status": "completed",
        "models": ["claude-opus-4-8"],
        "requests": [
            _request(at, tokens, produced, kind="n", api_time=api)
            for at, tokens, produced, api in nested
        ],
    }


def _session(session_id, rows):
    return {
        "id": session_id,
        "block_size": BLOCK,
        "hash_id_scope": "local",
        "models": ["claude-opus-4-8"],
        "requests": rows,
    }


def _rows(text):
    return [json.loads(line) for line in text.splitlines()]


def _identities(text):
    return [(r["session"], r["json_path"]) for r in _rows(text)]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A small corpus holding one of every refusal, then the roots that pass.

    Order matters twice over. The pool is filled in corpus order and scanning
    stops when it is full, so everything a test wants to see refused is placed
    ahead of the roots that qualify.
    """
    lines = [
        # the development session, in the shape that would otherwise qualify
        _session(
            "dev",
            [
                _request(0.0, 40960, 100, api_time=10.0),
                _subagent(2.0, [(2.0, 32768, 50, 1.0)]),
            ],
        ),
        # two overlapping root turns and no delegate: concurrency, but not the
        # kind this matrix is about
        _session(
            "no-subagent",
            [
                _request(0.0, 40960, 100, api_time=10.0),
                _request(2.0, 40960, 100, api_time=1.0),
            ],
        ),
        # a delegate, but the source never has two of these in flight: the
        # nested call starts exactly as the root turn ends
        _session(
            "peak-one",
            [
                _request(0.0, 40960, 100, api_time=10.0),
                _subagent(10.0, [(10.0, 40960, 50, 1.0)]),
            ],
        ),
        # one busy period, spread over more wall clock than the class replays
        _session(
            "spans",
            [
                _request(0.0, 40960, 100, api_time=100.0),
                _subagent(70.0, [(70.0, 40960, 50, 1.0)]),
            ],
        ),
        # inside the span and genuinely concurrent, and more prefill than one
        # root is allowed to offer
        _session(
            "too-much",
            [
                _request(0.0, 199936, 100, api_time=30.0),
                _subagent(1.0, [(1.0, 199936, 50, 5.0), (2.0, 199936, 50, 5.0)]),
            ],
        ),
        # concurrent, bounded, and not a large-prompt episode
        _session(
            "not-large",
            [
                _request(0.0, 19968, 100, api_time=10.0),
                _subagent(2.0, [(2.0, 19968, 50, 1.0)]),
            ],
        ),
        # a root whose first busy period holds a request that produced nothing.
        # The rest of that episode is perfectly servable, which is the trap:
        # the episode goes whole, and the root contributes its next one.
        _session(
            "late",
            [
                _request(0.0, 40960, 100, api_time=10.0),
                _subagent(2.0, [(2.0, 32768, 0, 1.0)]),
                _request(1000.0, 40960, 100, api_time=10.0),
                _subagent(1002.0, [(1002.0, 32768, 50, 1.0)], agent_id="sub_002"),
            ],
        ),
        _session(
            "big-a",
            [
                _request(0.0, 40960, 100, api_time=10.0),
                _subagent(3.0, [(3.0, 49152, 60, 2.0)]),
            ],
        ),
        # short: in this corpus as in the real one, a concurrent short episode
        # is siblings inside one branch with no root turn in it
        _session("short-a", [_subagent(0.0, [(0.0, 2944, 10, 1.0),
                                             (0.5, 3840, 12, 1.0)])]),
        _session("short-b", [_subagent(0.0, [(0.0, 1024, 8, 1.0),
                                             (0.3, 2048, 9, 1.0)])]),
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    monkeypatch.setitem(clients.CORPUS, "sha256", clients.digest_file(path))
    monkeypatch.setattr(clients, "DEVELOPMENT_SESSIONS", ("dev",))
    # The registered pool is eight roots and a corpus this size cannot hold
    # eight of each class. The *shape* is what these tests are about, so the
    # pool and the counts are scaled down and every rule constant is left as
    # registered.
    monkeypatch.setattr(clients, "CLIENT_COUNTS", (1, 2))
    monkeypatch.setattr(clients, "POOL_SIZE", 2)
    return str(path)


class TestWhatAnEpisodeIs:
    def test_a_subagent_summary_is_never_served(self, corpus):
        text, manifest = clients.build(corpus, "clients_large", 2)
        for row in _rows(text):
            assert row["actor"] in ("root", "subagent")
            # A wrapper's own path is /requests/<i>; a nested request's is
            # /requests/<i>/requests/<j>. No served row ever has the former.
            depth = row["json_path"].count("/requests/")
            assert depth == (2 if row["actor"] == "subagent" else 1)
            assert row["input_tokens"] > 0 and row["output_tokens"] > 0
        assert manifest["descendant_requests"] > 0

    def test_an_episode_holding_an_unservable_request_is_refused_whole(self, corpus):
        """The defect this forbids: dropping the silent request and keeping its
        neighbours, which splices a burst the source never served."""
        text, manifest = clients.build(corpus, "clients_large", 1)
        rows = _rows(text)
        assert {r["session"] for r in rows} == {"late"}
        # every row comes from the second busy period, including the servable
        # root turn that shared an episode with the silent one
        assert all(r["source_t_s"] >= 1000.0 for r in rows)
        assert "/requests/0" not in [r["json_path"] for r in rows]
        assert manifest["episodes_rejected"]["a request produced no output tokens"] == 1

    def test_the_development_session_is_never_selected(self, corpus):
        for klass in ("clients_large", "clients_short"):
            _text, manifest = clients.build(corpus, klass, 2)
            assert "dev" not in {e["id"] for e in manifest["roots_pool"]}
            assert manifest["episodes_rejected"]["a development session"] == 1

    def test_one_root_contributes_one_contiguous_burst(self, corpus):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        ids = [e["id"] for e in manifest["roots_used"]]
        assert ids == ["late", "big-a"]
        assert len(set(ids)) == len(ids)


class TestTheRefusalCensus:
    """A census of what was refused, not a transcript of each refusal."""

    def test_every_category_is_a_category(self, corpus):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        rule = clients.RULES["clients_large"]
        assert set(manifest["episodes_rejected"]) == {
            "a development session",
            "a request produced no output tokens",
            "no subagent request in the episode",
            f"the source never has {rule.min_peak} of these in flight",
            f"arrivals span more than {rule.max_span_s}s",
            f"more than {rule.max_total_input} input tokens in the episode",
            (
                f"no prompt reaches {rule.min_large_input} tokens, so this is "
                f"not a large-prompt episode"
            ),
        }

    def test_the_measurement_travels_beside_the_category(self, corpus):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        ranges = manifest["episodes_rejected_value_range"]
        rule = clients.RULES["clients_large"]
        assert ranges["no subagent request in the episode"] == {"min": 0, "max": 0}
        assert ranges[f"the source never has {rule.min_peak} of these in flight"] == {
            "min": 1,
            "max": 1,
        }
        assert ranges[f"arrivals span more than {rule.max_span_s}s"] == {
            "min": 70.0,
            "max": 70.0,
        }
        over = ranges[f"more than {rule.max_total_input} input tokens in the episode"]
        assert over["min"] == over["max"] > rule.max_total_input

    def test_the_census_says_how_far_it_looked(self, corpus):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        assert "not of the whole corpus" in manifest["episodes_rejected_scope"]
        assert manifest["scanned_sessions"] >= len(manifest["episodes_rejected"])


class TestTheClientCountsAreNested:
    def test_a_narrower_cell_is_a_prefix_of_a_wider_one(self, corpus):
        narrow, _ = clients.build(corpus, "clients_large", 1)
        wide, _ = clients.build(corpus, "clients_large", 2)
        narrow_ids, wide_ids = _identities(narrow), _identities(wide)
        assert set(narrow_ids) < set(wide_ids)
        assert len(narrow_ids) == len(set(narrow_ids))

    def test_the_shared_roots_are_replayed_identically(self, corpus):
        """Same requests, same arrivals, same client index -- so a difference
        between two client counts is the added load and nothing else."""
        narrow, _ = clients.build(corpus, "clients_large", 1)
        wide, _ = clients.build(corpus, "clients_large", 2)
        kept = {(r["session"], r["json_path"]): r for r in _rows(wide)}
        for row in _rows(narrow):
            assert kept[(row["session"], row["json_path"])] == row

    def test_the_pool_is_the_same_whatever_the_cell_takes_from_it(self, corpus):
        _n, narrow = clients.build(corpus, "clients_large", 1)
        _w, wide = clients.build(corpus, "clients_large", 2)
        assert narrow["roots_pool"] == wide["roots_pool"]
        assert narrow["roots_used"] == wide["roots_used"][:1]
        assert "subset of the next" in narrow["nested_pool"]


class TestArrivals:
    def test_every_client_starts_at_zero_and_says_so(self, corpus):
        text, manifest = clients.build(corpus, "clients_large", 2)
        rows = _rows(text)
        for index in range(2):
            mine = [r for r in rows if r["client_index"] == index]
            assert mine and min(r["arrival_s"] for r in mine) == 0.0
        assert "first arrival at 0.0" in manifest["arrivals"]["across_roots"]
        assert "construction and not a chronology" in (
            manifest["arrivals"]["across_roots"]
        )

    def test_within_a_client_the_source_intervals_survive(self, corpus):
        text, _manifest = clients.build(corpus, "clients_large", 2)
        for index in range(2):
            mine = sorted(
                (r for r in _rows(text) if r["client_index"] == index),
                key=lambda r: r["source_t_s"],
            )
            shift = mine[0]["source_t_s"]
            for row in mine:
                assert row["arrival_s"] == pytest.approx(row["source_t_s"] - shift)
                assert row["origin_shift_s"] == pytest.approx(shift)

    def test_nothing_is_clipped_compressed_or_serialised(self, corpus):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        assert manifest["gaps_clipped"] == 0
        assert manifest["outputs_altered"] == 0
        assert manifest["subagents_pruned"] == 0
        assert manifest["requests_serialised"] == 0

    def test_no_completion_dependency_is_assumed(self, corpus):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        assert "none assumed" in manifest["arrivals"]["causality"]
        assert manifest["arrivals"]["model"] == "source_paced_open_loop"


class TestEveryDescendantIsOffered:
    def test_the_counts_add_up_to_the_file(self, corpus):
        text, manifest = clients.build(corpus, "clients_large", 2)
        rows = _rows(text)
        assert manifest["requests"] == len(rows)
        assert manifest["descendant_requests"] == sum(
            1 for r in rows if r["actor"] == "subagent"
        )
        assert manifest["root_requests"] == sum(
            1 for r in rows if r["actor"] == "root"
        )
        assert (
            manifest["root_requests"] + manifest["descendant_requests"]
            == manifest["requests"]
        )

    def test_a_short_episode_is_all_delegates_and_says_so(self, corpus):
        text, manifest = clients.build(corpus, "clients_short", 2)
        rows = _rows(text)
        assert rows and all(r["actor"] == "subagent" for r in rows)
        assert all(r["input_tokens"] <= clients.RULES["clients_short"].max_input
                   for r in rows)
        assert manifest["source_peak_overlap"] >= 2

    def test_the_source_overlap_is_reported_as_evidence_not_as_a_bound(self, corpus):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        assert manifest["source_peak_overlap"] >= 2
        assert "not a bound" in manifest["source_peak_overlap_means"]


class TestProvenance:
    def test_a_row_can_be_traced_back_without_the_corpus(self, corpus):
        text, _manifest = clients.build(corpus, "clients_large", 2)
        for row in _rows(text):
            assert row["session"] and row["corpus_line_1based"] >= 1
            assert row["json_path"].startswith("/requests/")
            assert row["source_model"] == "claude-opus-4-8"
            if row["actor"] == "subagent":
                assert row["agent_id"] and row["subagent_type"]

    def test_the_label_that_served_it_is_not_the_label_it_is_replayed_as(
        self, corpus
    ):
        _text, manifest = clients.build(corpus, "clients_large", 2)
        assert manifest["target_model"] == clients.TARGET_MODEL
        assert "claude-opus-4-8" in manifest["source_models"]
        assert "No descendant is dropped" in manifest["source_models_declared"]

    def test_the_same_corpus_gives_the_same_bytes(self, corpus):
        first, one = clients.build(corpus, "clients_large", 2)
        second, two = clients.build(corpus, "clients_large", 2)
        assert first == second
        assert one["sha256"] == two["sha256"]


def _registered(klass: str, count: int):
    """The registered workload file, or a skip that says how to get it.

    The `.jsonl` are not in the repository: they are a slice of a licensed
    568 MB corpus, reproduced by `emit` from the manifest committed beside
    them. In a checkout without them there is nothing to check rather than
    something that failed.
    """
    here = ROOT / "atom" / "compass"
    path = here / f"cc_traces_{klass}_c{count}.jsonl"
    if not path.exists():
        pytest.skip(
            f"{path.name} is not in this checkout; reproduce it with "
            f"`cc_traces_clients_workload.py emit --class {klass} "
            f"--clients {count}` against the corpus named in its manifest"
        )
    return path, json.loads(
        (here / f"cc_traces_{klass}_c{count}.manifest.json").read_text()
    )


CELLS = [(klass, count) for klass in clients.RULES for count in (1, 2, 4, 8)]


class TestRegisteredArtifacts:
    """The eight files the protocol names are the ones the rule produces."""

    @pytest.mark.parametrize("klass,count", CELLS)
    def test_the_registered_workload_matches_its_manifest(self, klass, count):
        path, manifest = _registered(klass, count)
        assert clients.digest_file(path) == manifest["sha256"]
        assert manifest["corpus"]["sha256"] == clients.CORPUS["sha256"]
        assert manifest["class"] == klass and manifest["clients"] == count
        assert manifest["generator_sha256"] == clients.digest_file(
            ROOT / "scripts" / "compass" / "cc_traces_clients_workload.py"
        )

    @pytest.mark.parametrize("klass", sorted(clients.RULES))
    def test_the_registered_client_counts_are_nested(self, klass):
        """The property on the emitted bytes, not on a re-run of the rule."""
        previous = None
        for count in clients.CLIENT_COUNTS:
            path, _manifest = _registered(klass, count)
            here = _identities(path.read_text())
            assert len(here) == len(set(here))
            if previous is not None:
                assert set(previous) < set(here)
            previous = here
