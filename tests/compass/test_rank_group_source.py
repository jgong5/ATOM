"""Pricing every rank of a group from every rank's own tables.

`rank_aggregation="slowest"` exists because a TP group's step ends when its
slowest rank ends and the ranks are not interchangeable -- the TP4 head
measurements run 13.510 / 16.158 / 13.459 / 13.452 ms. It priced each rank by
moving `StepShape.rank_coords` and asking one `LibraryCostOracle`, and that
oracle does not select prices by rank: it looks a signature up in the library
it holds, which was loaded once, at the executor's own coordinates. So every
rank was priced from rank 0's tables, and the rank-1 outlier that motivates the
whole policy could not appear in the answer. The spread read as zero.

A stub oracle returning rank-dependent numbers cannot catch this -- it has the
per-rank behaviour built in, which is the thing that was missing. So these
build the real composition from real per-rank files on disk and drive it
through the real `_estimate_over_ranks`.
"""

import json
import types

from atom.compass.config import CompassConfig
from atom.compass.core.cost.base import StepShape
from atom.compass.runtime.source_oracle import source_cost_oracle


def _build_group(**kwargs):
    """`build_source_group`, imported at call time rather than at import time.

    The behaviour these tests are about is reachable through
    `source_cost_oracle`, which is what a served run names and which existed
    before this fix. So the module must still import against a tree without
    the fix, and those tests must fail on their *assertions* there. A module
    that could not be imported at all would prove only that a new name is new.
    """
    from atom.compass.runtime.source_oracle import build_source_group

    return build_source_group(**kwargs)

#: The documented TP4 head spread, in seconds. Rank 1 is the outlier.
SECONDS_BY_RANK = {0: 0.013510, 1: 0.016158, 2: 0.013459, 3: 0.013452}

WIDTH = 4
QUERIES = [1, 1]
CONTEXTS = [128, 130]


def _ops():
    return [{"name": "mm", "input_shapes": [[2, 64], [64, 64]],
             "dtypes": ["bfloat16"]}]


def _template(tmp_path, rank):
    """This rank's own graph, keyed by the coordinates its provenance names."""
    path = tmp_path / f"graph.tp{rank}.json"
    path.write_text(json.dumps({
        "ops": _ops(),
        "key": {"topology": [["tp", WIDTH]], "rank_coords": [["tp", rank]]},
        "provenance": {
            "batch_spec": {"kind": "decode", "query_lens": QUERIES,
                           "context_lens": CONTEXTS},
            "execution": {"capture_bucket": None},
        },
    }), encoding="utf-8")
    return str(path)


def _prices(tmp_path, rank, seconds):
    """This rank's own price list, priced at this rank's own number."""
    from atom.compass.runtime.microbench import signature_of

    path = tmp_path / f"prices.tp{rank}.json"
    path.write_text(json.dumps({
        "provenance": {"topology": {"tp": WIDTH},
                       "registration": "unregistered"},
        "prices": {signature_of(op): {"name": op["name"], "seconds": seconds,
                                      "occurrences": 1,
                                      "kernels": {"k0": seconds}}
                   for op in _ops()},
        "unpriced": {},
    }), encoding="utf-8")
    return str(path)


def _group(tmp_path, seconds_by_rank=None):
    """Every rank's files on disk, and the options a served run would carry."""
    seconds_by_rank = seconds_by_rank or SECONDS_BY_RANK
    for rank, seconds in seconds_by_rank.items():
        _prices(tmp_path, rank, seconds)
        _template(tmp_path, rank)
    return {"price": str(tmp_path / "prices.json"),
            "template": str(tmp_path / "graph.json"),
            "tp": WIDTH, "derive": 0, "require_complete": 0,
            "regions": "none", "rank_coords": {"tp": 0}}


def _shape(rank):
    return StepShape(num_scheduled_tokens=tuple(QUERIES),
                     context_lens=tuple(CONTEXTS),
                     num_prefill_tokens=0,
                     topology={"tp": WIDTH}, rank_coords={"tp": rank},
                     capture_bucket=None, compiled=None,
                     produces_output=True)


def _allocation_record():
    """One step's assignment, in the shape the scheduler hands the runner."""
    from atom.compass.runtime.templates import NativeStepAllocation

    return NativeStepAllocation(
        rows=list(zip(QUERIES, CONTEXTS)),
        block_tables=[[0, 1], [2, 3]],
        state_slots=None, num_prefill_seqs=0, source="test")


def _runner(policy, oracle):
    """The mixin itself, as `test_rank_aggregation` drives it.

    Not `CompassModelRunner`: that inherits `ModelRunner`, which imports
    `aiter`, which resolves the chip at import. Nothing here is
    device-dependent.
    """
    from atom.compass.runtime.predict import CompassPredictMixin

    stub = CompassPredictMixin.__new__(CompassPredictMixin)
    stub.__dict__["_compass_config_cache"] = CompassConfig(
        enabled=True, mode="predict", rank_aggregation=policy)
    stub.config = types.SimpleNamespace()
    stub._oracle = oracle
    return stub


class TestEveryRankIsPricedFromItsOwnTables:

    def test_the_group_step_is_the_slowest_ranks_own_price(self, tmp_path):
        """The number the policy exists to produce. One oracle holding rank
        0's library answers 13.510 ms four times; the group's step is rank 1's
        16.158 ms."""
        oracle = source_cost_oracle(**_group(tmp_path))

        cost, record = _runner("slowest", oracle)._estimate_over_ranks(
            _shape(0))

        assert record["slowest_rank"] == 1
        assert cost.seconds == SECONDS_BY_RANK[1]

    def test_each_rank_reports_its_own_number(self, tmp_path):
        oracle = source_cost_oracle(**_group(tmp_path))

        _, record = _runner("slowest", oracle)._estimate_over_ranks(_shape(0))

        assert record["seconds_by_rank"] == {
            str(rank): seconds for rank, seconds in SECONDS_BY_RANK.items()}

    def test_the_spread_is_not_zero(self, tmp_path):
        """What the defect looked like from outside: four identical rank
        timings and a spread of zero, on a group whose ranks differ by 20%."""
        oracle = source_cost_oracle(**_group(tmp_path))

        _, record = _runner("slowest", oracle)._estimate_over_ranks(_shape(0))

        assert record["spread_seconds"] > 0
        assert record["spread_seconds"] == (
            max(SECONDS_BY_RANK.values()) - min(SECONDS_BY_RANK.values()))

    def test_every_rank_was_actually_selected(self, tmp_path):
        """Asking one oracle four times and asking four oracles once are
        indistinguishable in the cost record unless selection is recorded."""
        oracle = source_cost_oracle(**_group(tmp_path))

        _runner("slowest", oracle)._estimate_over_ranks(_shape(0))

        assert sorted(oracle.selected_ranks) == list(range(WIDTH))


class TestTheRecordNamesEveryRanksFiles:

    def test_every_rank_retains_the_identity_of_what_it_read(self, tmp_path):
        """Not only the executor's own rank. The manifest is the evidence for
        a group's prediction, so it has to hold the group's inputs.

        Templates only, for now. Each rank does load its own price file -- the
        cost assertions above are what prove it -- but `PriceLibrary` does not
        yet report the identity of what it read, so there is nothing here to
        assert it against. This grows an `oracle.price` arm per rank when that
        reader lands; it is not a statement that prices are exempt.
        """
        import hashlib

        options = _group(tmp_path)
        group = _build_group(**options)

        templates = {i.path: i for i in group.loaded_inputs
                     if i.role == "oracle.template"}
        for rank in range(WIDTH):
            path = str(tmp_path / f"graph.tp{rank}.json")
            assert path in templates, f"rank {rank} is missing from the record"
            with open(path, "rb") as handle:
                assert templates[path].sha256 == hashlib.sha256(
                    handle.read()).hexdigest()
            assert templates[path].rank_own is True
            assert templates[path].rank_coords == (("tp", rank),)

    def test_the_stem_every_rank_asked_for_is_the_option_s(self, tmp_path):
        options = _group(tmp_path)
        group = _build_group(**options)

        asked = {i.requested for i in group.loaded_inputs
                 if i.role == "oracle.template"}
        assert asked == {options["template"]}

    def test_replacing_a_ranks_file_afterwards_does_not_move_its_identity(
            self, tmp_path):
        options = _group(tmp_path)
        group = _build_group(**options)
        before = {i.path: i.sha256 for i in group.loaded_inputs}

        _template(tmp_path, 2)  # rewritten, same name
        (tmp_path / "graph.tp2.json").write_text("{}", encoding="utf-8")

        after = {i.path: i.sha256 for i in group.loaded_inputs}
        assert after == before


class TestTheGroupPaysForOneDerivation:

    def test_every_rank_gets_a_composition(self, tmp_path):
        group = _build_group(**_group(tmp_path))

        assert set(group.by_rank) == set(range(WIDTH))

    def test_the_model_is_traced_once_and_the_derivers_are_shared(
            self, tmp_path, monkeypatch):
        """With derivation actually on, which is the only setting where there
        is a build to count.

        `ModelTracer.build` is spied rather than run: it loads a model, and
        the claim under test is about how many times it is called and which
        deriver each rank view ends up holding. Four builds of a 27B model to
        read four small JSON files would be the wrong trade, and would put the
        build cost in the record four times.
        """
        from atom.compass.runtime import tracer as tracer_module

        builds = []

        class _Tracer:
            pass

        class _Deriver:
            def __init__(self, tracer, region=None, **kwargs):
                self.tracer = tracer
                self.region = region

        def _build(model_path, tp, device="meta", **kwargs):
            builds.append((model_path, tp))
            return _Tracer()

        monkeypatch.setattr(tracer_module.ModelTracer, "build",
                            staticmethod(_build))
        monkeypatch.setattr(tracer_module, "ShapeDeriver", _Deriver)

        options = dict(_group(tmp_path))
        for rank in range(WIDTH):
            _template(tmp_path, rank)
        options.update(derive=1, head=True, model="/models/stub",
                       block_size=16, max_model_len=4096,
                       head_template=options["template"])
        group = _build_group(**options)

        assert len(builds) == 1, builds
        body = {c.deriver for c in group.by_rank.values()}
        assert len(body) == 1 and None not in body
        heads = {c.head_graphs._derive for c in group.by_rank.values()}
        assert len(heads) == 1 and None not in heads
        assert next(iter(body)).region == "body"
        assert next(iter(heads)).region == "head"

    def test_the_build_cost_is_counted_once(self, tmp_path, monkeypatch):
        """A record whose whole purpose is to be accountable must not report
        one build four times."""
        from atom.compass.runtime import tracer as tracer_module

        class _Deriver:
            def __init__(self, tracer, region=None, **kwargs):
                self.region = region

        monkeypatch.setattr(tracer_module.ModelTracer, "build",
                            staticmethod(lambda *a, **k: object()))
        monkeypatch.setattr(tracer_module, "ShapeDeriver", _Deriver)

        options = dict(_group(tmp_path))
        options.update(derive=1, model="/models/stub", block_size=16,
                       max_model_len=4096)
        group = _build_group(**options)

        assert group.build_seconds == group.by_rank[0].build_seconds
        for rank in range(1, WIDTH):
            assert group.by_rank[rank].build_seconds == 0.0


class TestTheServedSurfaceIsUnchanged:

    def test_the_allocation_the_runner_offers_reaches_every_rank(self, tmp_path):
        """The offer the runner actually makes, not an assignment to the
        wrapper.

        `_offer_native_allocation` reads the provider off the oracle and calls
        `offer()` on *the provider*, which lives inside each rank's
        `TemplateGraphs`. So a group whose ranks hold their own providers has
        three of them that are never offered anything, and every rank above
        the representative refuses the step for want of an allocation. This
        fails if only the representative is reached.
        """


        options = dict(_group(tmp_path))
        options.update(allocation="native", block_size=16, max_model_len=4096)
        group = _build_group(**options)

        providers = {rank: composition.allocation
                     for rank, composition in group.by_rank.items()}
        assert len(providers) == WIDTH
        # One object, so the runner's single `offer()` is the group's.
        assert len({id(p) for p in providers.values()}) == 1
        assert group.oracle.native_allocation is providers[0]

        group.oracle.native_allocation.offer(_allocation_record())
        for rank in range(WIDTH):
            assert group.by_rank[rank].allocation.offered == 1

    def test_describe_and_the_reporting_surface_still_answer(self, tmp_path):
        oracle = source_cost_oracle(**_group(tmp_path))

        assert isinstance(oracle.describe(), str)
        # Reached through delegation, as every report reads it.
        assert oracle.library.sources

    def test_one_rank_is_still_one_oracle(self, tmp_path):
        """At TP1 there is nothing to select between, and wrapping would add a
        layer that only obscures the composition."""
        _prices(tmp_path, 0, 0.001)
        _template(tmp_path, 0)
        group = _build_group(
            price=str(tmp_path / "prices.tp0.json"),
            template=str(tmp_path / "graph.tp0.json"),
            tp=1, derive=0, require_complete=0, regions="none")

        assert group.by_rank == {}
        assert not hasattr(group.oracle, "selected_ranks")
