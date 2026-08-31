"""Aligning a derived graph into a captured one.

A capture is not a derivation plus noise: it holds the runner's own work as
well as the model's. The question a capture answers is therefore containment
rather than equality, and these tests pin the distinction down — including the
ways containment can be reported as holding when it does not.
"""

import pytest

from atom.compass.core.diff import align_graphs
from atom.compass.core.graph import GraphKey, OpGraph, OpSpec


def op(name, shape=(1, 8), dtype="bfloat16", group=None):
    return OpSpec(
        name=name,
        input_shapes=(tuple(shape),),
        output_shapes=(tuple(shape),),
        dtypes=(dtype,),
        group=group,
    )


def graph(*ops):
    g = OpGraph()
    for o in ops:
        g.add(o)
    return g


def test_capture_containing_the_body_in_order_is_contained():
    derived = graph(op("aten::embedding"), op("aiter::gemm"), op("aiter::rmsnorm"))
    captured = graph(
        op("aten::slice"),                 # runner: batch metadata
        op("aten::embedding"),
        op("aiter::gemm"),
        op("aiter::rmsnorm"),
        op("aiter::gemm", shape=(1, 99)),  # runner: the LM head
        op("aten::exponential_"),          # runner: sampling
    )
    result = align_graphs(derived, captured)
    assert result.contained
    assert result.matched == 3
    assert not result.unmatched
    assert result.extra_counts() == {
        "aten::slice": 1, "aiter::gemm": 1, "aten::exponential_": 1,
    }


def test_a_missing_body_operator_is_reported():
    derived = graph(op("aiter::gemm"), op("aiter::attention"), op("aiter::gemm"))
    captured = graph(op("aiter::gemm"), op("aiter::gemm"))
    result = align_graphs(derived, captured)
    assert not result.contained
    # Matching stalls at the absent operator rather than skipping past it, so
    # everything from there on is reported unfound. That is the conservative
    # reading: the second gemm might be the body's, or might be the LM head,
    # and once the sequence has diverged there is no way to tell.
    assert [o.name for _, o in result.unmatched] == ["aiter::attention", "aiter::gemm"]
    assert result.matched == 1


def test_reordering_is_not_containment():
    derived = graph(op("a"), op("b"))
    captured = graph(op("b"), op("a"))
    assert not align_graphs(derived, captured).contained


def test_a_dtype_difference_alone_breaks_the_match():
    """The int32/int64 case that made a correct derivation look wrong.

    Worth a test of its own: the shapes agree, the order agrees, and the only
    disagreement is a dtype — which is exactly the shape of a real regression
    too, so the check must not soften it.
    """
    derived = graph(op("aten::embedding", dtype="int32"))
    captured = graph(op("aten::embedding", dtype="int64"))
    assert not align_graphs(derived, captured).contained
    assert align_graphs(derived, captured, compare_dtypes=False).contained


def test_group_membership_is_part_of_the_signature():
    """A collective on the wrong group is a different operation."""
    derived = graph(op("aiter::all_reduce", group="tp"))
    captured = graph(op("aiter::all_reduce", group="ep"))
    assert not align_graphs(derived, captured).contained


def test_an_empty_derivation_is_not_evidence():
    """Vacuous containment must not read as validation.

    Every operator of an empty graph is trivially present in any capture. If
    that counted, a derivation that silently produced nothing would report the
    same success as one that reproduced the model exactly.
    """
    result = align_graphs(OpGraph(), graph(op("aiter::gemm")))
    assert not result.contained
    assert result.matched == 0


def test_provenance_survives_a_round_trip(tmp_path):
    g = graph(op("aiter::gemm"))
    g.key = GraphKey.of("m", {"tp": 2}, {"tp": 0}, (1,))
    g.provenance = {"source": "capture", "device": "cuda", "compilation_level": 0}
    path = tmp_path / "g.json"
    g.save(path)
    assert OpGraph.load(path).provenance == g.provenance


def test_version_1_graphs_still_load(tmp_path):
    """Graphs written before provenance existed remain readable."""
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "version": 1,
        "key": None,
        "ops": [{"name": "aiter::gemm", "input_shapes": [[1, 8]],
                 "output_shapes": [[1, 8]], "dtypes": ["bfloat16"], "group": None}],
    }))
    loaded = OpGraph.load(path)
    assert len(loaded) == 1
    assert loaded.provenance == {}


def test_an_unknown_version_is_refused(tmp_path):
    import json

    path = tmp_path / "future.json"
    path.write_text(json.dumps({"version": 99, "ops": []}))
    with pytest.raises(ValueError):
        OpGraph.load(path)


class TestRankPath:
    """Each rank's graph gets its own file.

    Not cosmetic: at TP=2 both ranks trace and both write, and a shared path
    means they overwrite each other. The survivor names no rank, so it cannot
    be attributed to one — and it is the graphs *differing* between ranks that
    carries the information about how the model is sharded.
    """

    from atom.compass.runtime.runner import CompassModelRunner

    rank_path = staticmethod(CompassModelRunner._rank_path)

    def test_rank_is_in_the_name(self):
        assert self.rank_path("/out/g.json", {"tp": 1}) == "/out/g.tp1.json"

    def test_ranks_do_not_collide(self):
        paths = {self.rank_path("g.json", {"tp": r}) for r in range(4)}
        assert len(paths) == 4

    def test_every_group_a_rank_belongs_to_is_named(self):
        assert self.rank_path("g.json", {"tp": 1, "dp": 3}) == "g.dp3-tp1.json"

    def test_a_path_without_an_extension_still_works(self):
        assert self.rank_path("graph", {"tp": 0}) == "graph.tp0"


class TestGroupResolution:
    """Naming the communication group a collective ran on.

    The op graph carries no notion of any parallel strategy: a collective names
    its group, and the shapes around it do the rest. That only works if the name
    is actually resolved — a graph recording every collective identically cannot
    tell an all-reduce over tensor ranks from one over expert ranks, which is
    exactly the distinction the representation exists to keep.
    """

    from atom.compass.runtime.meta import AMBIGUOUS_GROUP, _resolve_group

    resolve = staticmethod(_resolve_group)

    def test_local_computation_names_no_group(self):
        assert self.resolve("aiter::gemm_a16w16", {"tp": 2}) is None

    def test_the_only_group_of_size_above_one_is_the_answer(self):
        assert self.resolve("aiter::all_reduce_", {"tp": 2, "dp": 1}) == "tp"

    def test_a_second_group_makes_it_ambiguous_rather_than_guessed(self):
        got = self.resolve("aiter::all_reduce_", {"tp": 2, "ep": 4})
        assert got == self.AMBIGUOUS_GROUP
        assert got != "tp"

    def test_a_collective_at_world_size_one_is_still_a_collective(self):
        """It ran; it just had nobody to talk to. Recording it as local compute
        would lose that the model calls it at all."""
        assert self.resolve("aiter::all_reduce_", {"tp": 1}) == self.AMBIGUOUS_GROUP

    def test_every_collective_kind_is_recognised(self):
        for name in ("all_gather", "reduce_scatter", "broadcast", "all_to_all"):
            assert self.resolve(f"c10d::{name}_", {"tp": 2}) == "tp"


def test_skipped_cudagraph_capture_still_honours_its_contract():
    """A do-nothing override must still return what the caller unpacks.

    `engine_core` calls `capture_cudagraph` across the worker boundary with
    `wait_out=True` and unpacks three values. Returning None killed the worker
    on an unpacking error while the parent waited for a reply, so the run hung
    on a shared-memory broadcast — a symptom naming neither CUDA graphs nor
    Compass. It only reproduced without `--enforce-eager`, which is the default,
    so every gate run had been in a configuration nobody deploys.
    """
    from atom.compass.runtime.runner import CompassModelRunner

    cost, sizes, pool_bytes = CompassModelRunner.capture_cudagraph(object())
    assert cost == 0.0
    assert list(sizes) == []
    assert pool_bytes == 0
