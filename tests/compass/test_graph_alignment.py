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


def test_a_recorded_forward_context_survives_a_round_trip(tmp_path):
    """Attention cannot be replayed without it, and it lives only here.

    Its arguments cannot carry it: `torch.compile` constant-folds every one that
    is not a tensor, so an operator given its metadata as arguments records the
    values the *warmup* forward had.
    """
    recorded = (("context_lens", [315, 315]), ("max_seqlen_q", 1),
                ("block_tables_shape", [2, 2560]))
    g = graph(OpSpec(name="aiter::unified_attention_with_output_base",
                     context=recorded))
    path = tmp_path / "g.json"
    g.save(path)
    assert OpGraph.load(path).ops[0].context == recorded


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


def _stub_runner(mode):
    """A runner with settings but no engine behind it.

    `_compass_config` resolves lazily from `self.config`, so seeding the cache
    directly is what lets these run without standing up a model.
    """
    from atom.compass.config import CompassConfig
    from atom.compass.runtime.runner import CompassModelRunner

    stub = CompassModelRunner.__new__(CompassModelRunner)
    stub.__dict__["_compass_config_cache"] = CompassConfig(
        enabled=True, mode=mode,
        graph_out="g.json" if mode == "trace" else None,
        measure_out="t.jsonl" if mode == "measure" else None,
    )
    return stub


class TestCudagraphCapturePerMode:
    """Whether to capture CUDA graphs depends on what the mode is doing.

    Getting this wrong is expensive in a way that does not announce itself.
    Skipping capture in `measure` fitted the oracle to eager execution while the
    deployment replayed a graph -- 28.78 ms against 3.24 ms on Qwen3-0.6B
    decode, an 8.9x error that reached the end-to-end comparison as +800% TPOT.
    """

    def test_measure_captures_for_real(self, monkeypatch):
        """Timings must come from the path production actually runs."""
        from atom.model_engine.model_runner import ModelRunner

        called = []
        monkeypatch.setattr(
            ModelRunner, "capture_cudagraph",
            lambda self: (called.append(True), (1.0, [8], 42))[1],
        )
        assert _stub_runner("measure").capture_cudagraph() == (1.0, [8], 42)
        assert called

    def test_predict_and_trace_skip_but_return_the_triple(self):
        """A skip still has to honour the contract.

        `engine_core` calls this across the worker boundary with wait_out=True
        and unpacks three values. Returning None killed the worker mid-reply and
        hung the parent on a broadcast that never arrived -- a symptom naming
        neither CUDA graphs nor Compass.
        """
        for mode in ("predict", "trace"):
            cost, sizes, pool = _stub_runner(mode).capture_cudagraph()
            assert (cost, list(sizes), pool) == (0.0, [], 0)

    def test_trace_runs_a_real_forward_and_predict_does_not(self):
        assert _stub_runner("trace")._runs_real_forward
        assert _stub_runner("measure")._runs_real_forward
        assert not _stub_runner("predict")._runs_real_forward


def test_warmup_batches_are_run_but_not_counted(monkeypatch):
    """Warmup drives dummy batches through `forward`.

    They must run -- they are what autotunes Triton and settles the allocator --
    but they are not steps a deployment performs. Counted, they would spend the
    trace budget on a dummy shape and put dummy rows in the table a cost model
    is fitted to.
    """
    from atom.model_engine.model_runner import ModelRunner

    sentinel = object()
    monkeypatch.setattr(ModelRunner, "forward", lambda self, batch: sentinel)

    for mode in ("measure", "trace"):
        stub = _stub_runner(mode)

        def _fail(batch):
            raise AssertionError(f"{mode}: a dummy batch reached the recording path")

        stub._forward_measured = _fail
        stub._forward_traced = _fail

        class DummyBatch:
            is_dummy_run = True

        from atom.compass.runtime.runner import CompassModelRunner

        assert CompassModelRunner.forward(stub, DummyBatch()) is sentinel


def test_a_real_batch_still_reaches_the_recording_path():
    """The guard must key on the dummy flag, not disable recording outright."""
    from atom.compass.runtime.runner import CompassModelRunner

    reached = []
    stub = _stub_runner("measure")
    stub._forward_measured = lambda batch: reached.append(batch) or "out"

    class RealBatch:
        is_dummy_run = False

    assert CompassModelRunner.forward(stub, RealBatch()) == "out"
    assert reached
