"""Rebuilding a call from what the graph recorded.

A kernel is priced by calling it, and calling it means putting its arguments
back in the order it wants them. The tracer records tensors as an ordered list
of shapes and everything else as (name, value), a positional argument being
named by its index -- so the two have to be interleaved again. Getting this
wrong is silent: the call raises, the signature is reported unpriced, and the
price list quietly covers less than it should. That is how 113 of 330 operators
went missing before scalars were recorded at all.
"""

import pytest

from atom.compass.runtime.microbench import _rebuild_args, signature_of


def _op(shapes=(), dtypes=(), scalars=()):
    return {"name": "aiter::x", "input_shapes": list(shapes),
            "dtypes": list(dtypes), "scalars": [list(s) for s in scalars]}


class TestRebuildingACall:
    def test_tensors_only(self):
        args, kw = _rebuild_args(_op(), ["A", "B"])
        assert args == ["A", "B"]
        assert kw == {}

    def test_a_trailing_scalar(self):
        """rmsnorm2d_fwd_(out, input, weight, eps) -- eps is position 3."""
        args, kw = _rebuild_args(_op(scalars=[("#3", 1e-6)]), ["out", "in", "w"])
        assert args == ["out", "in", "w", 1e-6]

    def test_a_scalar_in_the_middle(self):
        args, kw = _rebuild_args(_op(scalars=[("#1", 0.5)]), ["A", "B"])
        assert args == ["A", 0.5, "B"], "the tensor after it must shift right"

    def test_keywords_stay_keywords(self):
        args, kw = _rebuild_args(
            _op(scalars=[("#2", 8), ("causal", True)]), ["q", "k"])
        assert args == ["q", "k", 8]
        assert kw == {"causal": True}

    def test_a_list_valued_scalar(self):
        args, _ = _rebuild_args(_op(scalars=[("#1", [1, 2, 3])]), ["A"])
        assert args == ["A", [1, 2, 3]]

    def test_no_scalars_recorded_at_all(self):
        """Graphs written before scalars existed still price what they can."""
        op = {"name": "aiter::x", "input_shapes": [], "dtypes": []}
        args, kw = _rebuild_args(op, ["A"])
        assert args == ["A"] and kw == {}


class TestSignature:
    def test_shape_is_part_of_the_price(self):
        """A kernel does not have one cost; it has one per shape it runs at."""
        a = signature_of({"name": "aiter::gemm", "input_shapes": [[4, 1024]],
                          "dtypes": ["bfloat16"]})
        b = signature_of({"name": "aiter::gemm", "input_shapes": [[8, 1024]],
                          "dtypes": ["bfloat16"]})
        assert a != b

    def test_dtype_is_too(self):
        a = signature_of({"name": "aiter::gemm", "input_shapes": [[4, 1024]],
                          "dtypes": ["bfloat16"]})
        b = signature_of({"name": "aiter::gemm", "input_shapes": [[4, 1024]],
                          "dtypes": ["float16"]})
        assert a != b


class TestScalarCapture:
    """What the tracer keeps, and what it refuses to keep."""

    def _scalars(self, args, kwargs=None):
        from atom.compass.runtime.meta import _scalars_of

        return _scalars_of(args, kwargs or {})

    def test_positions_are_recorded(self):
        assert self._scalars(("t", 1e-6)) == (("#0", "t"), ("#1", 1e-6))

    def test_keywords_keep_their_names(self):
        assert self._scalars((), {"eps": 1e-6}) == (("eps", 1e-6),)

    def test_unreplayable_values_are_dropped_not_stringified(self):
        """A value that cannot be passed back must not look like it can."""
        class Odd:
            pass

        assert self._scalars((Odd(),)) == ()


class TestPricingIsNotInsideAForward:
    """Nothing priced here is inside a live forward.

    `capture_cudagraph` leaves its last rung's context installed, and an
    operator that reads its metadata from the ambient context rather than its
    arguments will use it. Attention did: it walked the leftover 16384-token
    sequence whatever it was handed, priced at 163.7us against a true 23.0us,
    and was invariant to every argument because none was read. The reset is per
    signature, not once, because an operator that rebuilds the context from its
    arguments leaves that context behind for whatever is priced next.
    """

    def test_the_context_is_reset_before_each_signature(self, tmp_path,
                                                        monkeypatch):
        import json

        import atom.utils.forward_context as forward_context
        from atom.compass.runtime import microbench

        graph = {"ops": [
            {"name": "aiter::a", "input_shapes": [], "dtypes": [], "scalars": []},
            {"name": "aiter::b", "input_shapes": [], "dtypes": [], "scalars": []},
        ]}
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(graph))

        resets: list[int] = []
        monkeypatch.setattr(forward_context, "reset_forward_context",
                            lambda: resets.append(1))
        # Unresolvable, so each signature stops right after its reset.
        monkeypatch.setattr(microbench, "_resolve", lambda name: None)

        microbench.price_graph(str(path))

        assert len(resets) == 2, "once per signature, before it is priced"


class TestOperatorsThatReadAmbientState:
    """Attention takes its metadata from a forward context, not its arguments.

    Giving it those arguments was tried and does not survive `torch.compile`:
    the tensor reads become graph inputs, but an `int` is constant-folded and an
    argument that was `None` when the graph compiled is baked in as `None`. So
    the context is recorded beside the operator instead, and is the only place
    the difference between a 40-token and a 4000-token decode is written down.
    """

    def test_the_context_is_part_of_the_key(self):
        short = _op(shapes=[(4, 2048)], dtypes=["bfloat16"])
        short["context"] = [["context_lens", [40, 40]]]
        long = dict(short, context=[["context_lens", [4000, 4000]]])
        assert signature_of(short) != signature_of(long), (
            "same shapes, different amounts of KV walked")

    def test_block_table_contents_are_not(self):
        """They decide which blocks are walked, not how many."""
        a = _op(shapes=[(4, 2048)], dtypes=["bfloat16"])
        a["context"] = [["context_lens", [40]], ["block_tables", [1, 2, 3]]]
        b = dict(a, context=[["context_lens", [40]], ["block_tables", [7, 8, 9]]])
        assert signature_of(a) == signature_of(b)

    def test_one_without_a_recorded_context_is_not_priced(self, tmp_path,
                                                          monkeypatch):
        """Rather than priced against whatever capture left installed."""
        import json

        import atom.utils.forward_context as forward_context
        from atom.compass.runtime import microbench

        name = "aiter::unified_attention_with_output_base"
        path = tmp_path / "graph.json"
        path.write_text(json.dumps({"ops": [
            {"name": name, "input_shapes": [], "dtypes": [], "scalars": []},
        ]}))
        monkeypatch.setattr(forward_context, "reset_forward_context", lambda: None)
        monkeypatch.setattr(microbench, "_resolve", lambda n: object())

        result = microbench.price_graph(str(path))

        assert not result["prices"]
        assert "recorded none" in next(iter(result["unpriced"].values()))

    def test_kv_regions_are_rotated_only_when_capturing(self, tmp_path,
                                                        monkeypatch):
        """A loop cannot rotate them: it installs one context and calls in it.

        Only the captured path gives each call its own region, which is what
        stops the whole working set staying resident across the batch.
        """
        import json

        import atom.utils.forward_context as forward_context
        from atom.compass.runtime import forward_ctx, microbench

        path = tmp_path / "graph.json"
        path.write_text(json.dumps({"ops": [{
            "name": "aiter::unified_attention_with_output_base",
            "input_shapes": [], "dtypes": [], "scalars": [],
            "context": [["context_lens", [8]]]}]}))
        monkeypatch.setattr(forward_context, "reset_forward_context", lambda: None)
        monkeypatch.setattr(microbench, "_resolve", lambda n: object())

        asked = []

        def fake_install(name, recorded, variants=1):
            asked.append(variants)
            return []

        monkeypatch.setattr(forward_ctx, "install", fake_install)

        microbench.price_graph(str(path), cache="graph")
        microbench.price_graph(str(path), cache="hot")

        assert asked == [microbench.KV_VARIANTS, 1]


class TestOperatorsThatCannotBeCaptured:
    """Some operators cannot go into a CUDA graph at all.

    Chunked-prefill attention gathers cached and new KV with a
    `repeat_interleave` whose output size is only known on the device, so it
    synchronises, and a synchronise inside a capture is an error. Refusing to
    price those left the whole of chunked prefill unpriced; they are timed
    back-to-back instead, which is a different kind of number and so is
    recorded as one.
    """

    def test_a_capture_failure_is_told_from_a_broken_operator(self):
        from atom.compass.runtime.microbench import _uncapturable

        assert _uncapturable(RuntimeError(
            "HIP error: operation not permitted when stream is capturing"))
        assert _uncapturable(RuntimeError("cudaErrorStreamCaptureUnsupported"))

    def test_an_ordinary_failure_still_leaves_the_operator_unpriced(self):
        """A fallback that swallowed real errors would price nonsense."""
        from atom.compass.runtime.microbench import _uncapturable

        assert not _uncapturable(RuntimeError("out of memory"))
        assert not _uncapturable(TypeError(
            "empty() received an invalid combination of arguments"))


class TestKernelsThatAreNotTorchOperators:
    """A raw `@triton.jit` kernel cannot be found through `torch.ops`.

    Recording its name is not enough to call it back, and a launch grid is not
    an argument but decides how much work runs. Both are recorded so the kernel
    can be relaunched; without either it stays unpriced rather than being priced
    over a guessed grid.
    """

    def test_the_grid_is_part_of_the_key(self):
        small = _op(shapes=[(4096, 8)], dtypes=["bfloat16"])
        small["launch"] = [["grid", [64, 8]], ["origin", "m:k"]]
        large = dict(small, launch=[["grid", [6594, 8]], ["origin", "m:k"]])
        assert signature_of(small) != signature_of(large), (
            "same arguments, different amounts of work")

    def test_where_it_came_from_is_not(self):
        """Origin says how to import the kernel, not what it costs."""
        a = _op(shapes=[(4096, 8)], dtypes=["bfloat16"])
        a["launch"] = [["grid", [64, 8]], ["origin", "one:k"]]
        b = dict(a, launch=[["grid", [64, 8]], ["origin", "other:k"]])
        assert signature_of(a) == signature_of(b)

    def test_a_kernel_with_no_origin_is_not_resolved(self):
        """Inductor generates into a module that does not outlive the process."""
        from atom.compass.runtime.microbench import _resolve_triton

        assert _resolve_triton({"launch": [["grid", [64]], ["origin", ""]]}) is None
        assert _resolve_triton({"launch": []}) is None

    def test_nor_is_one_whose_grid_never_resolved(self):
        """A guessed grid would price a different amount of work than ran."""
        from atom.compass.runtime.microbench import _resolve_triton

        assert _resolve_triton({
            "launch": [["grid", ["<unresolved>"]], ["origin", "m:k"]]}) is None


class TestGeneratedKernels:
    """Inductor's kernels have no importable name, only a file.

    The autotuner keeps the path it generated into, the file is named by a hash
    of the code it holds, and it defines the kernel at module level under the
    name inductor calls it. So the file is the origin -- for as long as the
    codecache survives, which is why a missing one leaves the operator unpriced
    rather than priced against something else.
    """

    def test_a_missing_codecache_leaves_it_unpriced(self, tmp_path):
        from atom.compass.runtime.microbench import _resolve_triton

        gone = str(tmp_path / "never-generated.py")
        assert _resolve_triton({
            "launch": [["grid", []], ["origin", f"path:{gone}::k"]]}) is None

    def test_a_generated_origin_is_not_refused_for_having_no_grid(self, tmp_path):
        """It computes its own grid from its arguments, unlike a hand-written
        kernel, so an empty grid is expected rather than unresolved."""
        from atom.compass.runtime import microbench

        seen = []
        original = microbench._resolve_generated
        microbench._resolve_generated = seen.append
        try:
            microbench._resolve_triton({
                "launch": [["grid", []], ["origin", "path:/x/y.py::k"]]})
        finally:
            microbench._resolve_generated = original
        assert seen == ["path:/x/y.py::k"]
        assert microbench._resolve_generated is original


class TestGeneratedKernelsUnderParallelism:
    """Inductor caches per rank, and the bench graph is the union of all ranks.

    Loading another rank's generated module bound the process to a device it
    could not use, and every later allocation died with "invalid device
    ordinal" far from the cause. Each rank prices its own.
    """

    def test_another_ranks_module_is_not_this_process_to_load(self):
        from atom.compass.runtime.microbench import _is_this_rank

        assert _is_this_rank("/cache/abc/rank_0/backbone/x.py")
        assert not _is_this_rank("/cache/abc/rank_1/backbone/x.py")

    def test_a_path_naming_no_rank_is_left_alone(self):
        from atom.compass.runtime.microbench import _is_this_rank

        assert _is_this_rank("/tmp/torchinductor_root/qk/cqkf626.py")

    def test_generated_loading_is_off_under_parallelism(self, monkeypatch):
        """A fault kills the run; an unpriced operator does not.

        Asked of the process group rather than of `WORLD_SIZE`, which the
        engine never sets -- the gate that read it never once fired.
        """
        from atom.compass.runtime import microbench

        monkeypatch.setattr(microbench, "LOAD_GENERATED", None)
        monkeypatch.setattr(microbench, "under_parallelism", lambda: True)
        assert not microbench._load_generated()
        assert microbench._breakdown_over() > 0
        monkeypatch.setattr(microbench, "under_parallelism", lambda: False)
        assert microbench._load_generated()
        assert microbench._breakdown_over() == 0

    def test_an_explicit_setting_still_wins(self, monkeypatch):
        from atom.compass.runtime import microbench

        monkeypatch.setattr(microbench, "under_parallelism", lambda: True)
        monkeypatch.setattr(microbench, "LOAD_GENERATED", "1")
        assert microbench._load_generated()


class TestBreakdownsUnderParallelism:
    """Taking a breakdown runs the operator twice more.

    For a collective those are two calls its peers do not make, so one rank
    pricing a signature another skipped leaves them waiting for each other.
    That was read as "breakdowns deadlock under parallelism" and turned off for
    every operator, removing the per-kernel comparison from the configurations
    that most needed it. Only collectives need excluding.
    """

    def test_a_collective_is_excluded(self):
        from atom.compass.runtime.microbench import _is_collective_op

        assert _is_collective_op({"name": "c10d::allreduce_", "group": "tp"})
        assert _is_collective_op({"name": "c10d::allreduce_"})
        assert _is_collective_op({"name": "aiter::fused_all_reduce"})

    def test_a_local_operator_is_not(self):
        from atom.compass.runtime.microbench import _is_collective_op

        assert not _is_collective_op({"name": "aten::mm", "group": None})
        assert not _is_collective_op({"name": "aiter::fmha_fwd"})


class TestOneBreakdownPerOperator:
    """A threshold on the signature is blind to an operator that fragments.

    `aiter::linear_attention_with_output_base` arrives as 48 signatures of
    0.093ms, one per layer, because its signature carries per-layer state. Each
    falls under the 1ms bar that decides whether a breakdown is worth a profiler
    session, so the family got none -- while being the third largest thing in
    the step at 4.47ms. That is most of the quarter of kernel time that could
    not be audited against a price.

    The rule added for it: a signature also earns a breakdown by being the first
    of an operator nobody has taken apart yet. One per family is enough to name
    an operator's kernels, and it bounds the sessions at the number of distinct
    operator names.
    """

    def test_a_fragment_of_an_uncovered_operator_is_taken(self):
        from atom.compass.runtime.microbench import _wants_breakdown

        assert _wants_breakdown(23.2e-6, 4, "aiter::linear_attention", set())

    def test_later_fragments_of_the_same_operator_are_not(self, monkeypatch):
        """Under parallelism, where sessions are scarce and the bar is 1ms.

        Without the parallel context the bar is zero and every signature earns a
        breakdown anyway, so this says nothing unless the context is set.
        """
        from atom.compass.runtime import microbench

        monkeypatch.setattr(microbench, "under_parallelism", lambda: True)
        assert not microbench._wants_breakdown(
            23.2e-6, 4, "aiter::linear_attention", {"aiter::linear_attention"})

    def test_at_tp1_every_signature_is_taken_apart(self, monkeypatch):
        """The bar is zero off parallelism, and the family rule changes nothing.

        Sessions are only scarce where they fault, so nothing is skipped here
        and the fragmenting operator was never invisible at TP=1.
        """
        from atom.compass.runtime import microbench

        monkeypatch.setattr(microbench, "under_parallelism", lambda: False)
        assert microbench._wants_breakdown(
            23.2e-6, 4, "aiter::linear_attention", {"aiter::linear_attention"})

    def test_a_big_signature_is_taken_however_covered_its_family(self, monkeypatch):
        """Two shapes of one gemm launch different kernels.

        So the family rule must not stop the second being taken apart on its
        own merit.
        """
        from atom.compass.runtime import microbench

        monkeypatch.setattr(microbench, "under_parallelism", lambda: True)
        assert microbench._wants_breakdown(42e-6, 256, "aiter::gemm_a16w16",
                                           {"aiter::gemm_a16w16"})

    def test_without_a_family_it_is_the_threshold_alone(self, monkeypatch):
        """The old behaviour, for callers that pass no family."""
        from atom.compass.runtime import microbench

        monkeypatch.setattr(microbench, "under_parallelism", lambda: True)
        assert not microbench._wants_breakdown(23.2e-6, 4, None, None)
        assert microbench._wants_breakdown(42e-6, 256, None, None)


class TestArgumentLayout:
    """A view argument is not addressable as a dense tensor of its shape.

    The case that forced this: a fused norm in the 27B prefill graph takes
    `in_ptr2` as a `[16, 48, 128]` view whose row stride is 16480, not
    48 * 128 = 6144. The kernel indexes `r0_1 + 128*x2 + 16480*x3` over
    x3 < 16 and x2 < 48, so it reads element 253,343 -- while a stand-in built
    from the shape alone holds 98,304. That read is 310 KB past the end of the
    buffer, and it kills the process with

        Memory access fault by GPU node-2 ... Reason: Unknown.

    which no per-signature try/except can catch. It took a five-round bisect to
    name, because nothing in the recorded operator looks wrong: arity, dtypes
    and both scalars are correct.
    """

    def test_a_view_reaches_past_its_own_element_count(self):
        from atom.compass.runtime.microbench import _base_extent

        assert _base_extent((16, 48, 128), (16480, 128, 1)) == 253344
        assert 16 * 48 * 128 == 98304, "which is what a dense stand-in holds"

    def test_a_dense_argument_needs_none_of_this(self):
        from atom.compass.runtime.microbench import _base_extent

        assert _base_extent((16, 48, 128), (6144, 128, 1)) is None
        assert _base_extent((128,), (1,)) is None

    def test_an_offset_is_part_of_the_extent(self):
        from atom.compass.runtime.microbench import _base_extent

        assert _base_extent((16, 48, 128), (6144, 128, 1), 100) == 98404

    def test_an_unrecorded_layout_falls_back_to_dense(self):
        """A graph captured before layout was recorded prices as it did."""
        from atom.compass.runtime.microbench import _base_extent

        assert _base_extent((16, 48, 128), None) is None
        assert _base_extent((16, 48, 128), ()) is None
        assert _base_extent((16, 48, 128), (128, 1)) is None

    def test_a_layout_reaching_before_the_start_is_refused(self):
        """Rather than allocating a buffer the offset cannot sit inside."""
        from atom.compass.runtime.microbench import _base_extent

        assert _base_extent((4,), (-1,), 0) is None
        assert _base_extent((4,), (-1,), 3) == 4

    def test_a_strided_call_does_not_inherit_a_dense_price(self):
        op = {"name": "inductor::fused", "input_shapes": [[16, 48, 128]],
              "dtypes": ["bfloat16"]}
        dense = signature_of(op)
        assert signature_of({**op, "input_strides": [[6144, 128, 1]]}) == dense, (
            "a dense layout keys the same as a graph that recorded none, so "
            "prices measured before layout was recorded still answer")
        assert signature_of({**op, "input_strides": [[16480, 128, 1]]}) != dense


class TestHowManyRepetitionsAPriceIsWorth:
    """A fixed iteration count prices a microsecond kernel well and a
    multi-second one absurdly. Attention at 16384 queries over 229376 of KV
    runs about 0.7s a launch, so the old fixed 2000 was 23 minutes for one
    number, and a graph holding 16 such signatures took six hours.
    """

    def _probed(self, monkeypatch, seconds):
        from atom.compass.runtime import microbench as mb

        monkeypatch.setattr(mb, "_time_over",
                            lambda *a, **k: (seconds, seconds))
        return mb._iters_for(None, [], 2000, 20)

    def test_a_microsecond_kernel_keeps_the_ceiling(self, monkeypatch):
        assert self._probed(monkeypatch, 5e-6) == 2000

    def test_a_slow_kernel_is_cut_to_the_budget(self, monkeypatch):
        from atom.compass.runtime.microbench import ITER_BUDGET_SECONDS

        assert self._probed(monkeypatch, 0.01) == int(ITER_BUDGET_SECONDS / 0.01)

    def test_the_floor_wins_over_the_budget(self, monkeypatch):
        """The 27B's worst attention shape: 16384 queries over 229376 of KV at
        about 0.7s a launch. The budget alone would ask for 4 repetitions; the
        floor holds it at MIN_ITERS, so the signature costs ~14s instead of the
        23 minutes a fixed 2000 cost."""
        from atom.compass.runtime.microbench import MIN_ITERS

        assert self._probed(monkeypatch, 0.7) == MIN_ITERS

    def test_a_very_slow_kernel_still_gets_a_usable_mean(self, monkeypatch):
        from atom.compass.runtime.microbench import MIN_ITERS

        assert self._probed(monkeypatch, 60.0) == MIN_ITERS

    def test_the_probe_never_asks_for_more_than_it_was_given(self, monkeypatch):
        """The caller's count is a ceiling, so a cheap kernel is timed exactly
        as it was before the probe existed."""
        from atom.compass.runtime import microbench as mb

        monkeypatch.setattr(mb, "_time_over", lambda *a, **k: (1e-9, 1e-9))
        assert mb._iters_for(None, [], 50, 20) == 50

    def test_a_failed_probe_leaves_the_count_alone(self, monkeypatch):
        """Not a guess. The timing call that follows raises the same way and is
        reported against the signature."""
        from atom.compass.runtime import microbench as mb

        def boom(*a, **k):
            raise RuntimeError("no")

        monkeypatch.setattr(mb, "_time_over", boom)
        assert mb._iters_for(None, [], 2000, 20) == 2000

    def test_a_zero_reading_is_not_divided_by(self, monkeypatch):
        assert self._probed(monkeypatch, 0.0) == 2000


class TestLayersAreOneMeasurement:
    """Sixteen full-attention and forty-eight GDN layers each produced a
    distinct signature, and the only field that differed was a string scalar
    naming the module the kernel was called from. That is 64 of a prefill
    graph's 151 signatures measured to learn the same number 64 times.

    The grouping is done beside the key, not in it. These tests pin both
    halves: that layers group, and that `signature_of` is untouched -- because
    changing it would retire every price list already measured.
    """

    def test_two_layers_are_one_class(self):
        from atom.compass.runtime.microbench import _layer_class
        a = _op(scalars=[("#5", "language_model.model.layers.3.self_attn")])
        b = _op(scalars=[("#5", "language_model.model.layers.7.self_attn")])
        assert _layer_class(a) == _layer_class(b)

    def test_but_they_are_still_two_signatures(self):
        a = _op(scalars=[("#5", "language_model.model.layers.3.self_attn")])
        b = _op(scalars=[("#5", "language_model.model.layers.7.self_attn")])
        assert signature_of(a) != signature_of(b), (
            "the key must not change, or every existing price list stops "
            "answering")

    def test_a_cost_bearing_scalar_is_not_dropped(self):
        """The grouping is by layer index, not by "has a dot in it"."""
        from atom.compass.runtime.microbench import _layer_class
        a = _op(scalars=[("#2", 1), ("dtype", "torch.bfloat16")])
        b = _op(scalars=[("#2", 16384), ("dtype", "torch.bfloat16")])
        assert _layer_class(a) != _layer_class(b)

    def test_a_dotted_string_without_a_layer_index_stays(self):
        from atom.compass.runtime.microbench import _is_module_path
        assert not _is_module_path("torch.bfloat16")
        assert not _is_module_path("causal")
        assert _is_module_path("language_model.model.layers.0.linear_attn")

    def test_operators_that_differ_elsewhere_do_not_group(self):
        from atom.compass.runtime.microbench import _layer_class
        a = _op(shapes=[(16, 10240)],
                scalars=[("#5", "model.layers.3.self_attn")])
        b = _op(shapes=[(32, 10240)],
                scalars=[("#5", "model.layers.7.self_attn")])
        assert _layer_class(a) != _layer_class(b)
