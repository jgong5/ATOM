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

from atom.compass.runtime.microbench import (
    ArgumentStructureRefusal,
    _group_for_schema,
    _index_output_shape,
    _rebuild_args,
    signature_of,
)


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


class TestAStrideIntoMemoryTheGraphNeverSaw:
    """A Triton kernel's tensor arguments may be views, and a view is not a shape.

    `_fused_qk_norm_single_kernel` takes q as a view into the fused qkv buffer.
    The graph records q's shape -- [4, 24, 256] at TP=1 -- and, separately, the
    plain integer 14336 that is the *buffer's* row stride. Rebuilt dense at the
    recorded shape and launched with the recorded stride, row 3 addresses
    element 49151 of 24576. On the device that is not an exception that leaves
    one signature unpriced; it is a memory access fault that kills the process
    with 36 signatures priced and no record of which one was in hand.
    """

    def _kernel(self, scalars):
        op = _op(shapes=[(4, 24, 256), (4, 4, 256)],
                 dtypes=["bfloat16", "bfloat16"], scalars=scalars)
        op["name"] = "triton::k"
        return op

    def test_a_stride_larger_than_every_row_is_refused(self):
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        found = _stride_past_its_tensors(self._kernel([("#9", 14336)]))
        assert found == ("#9", 14336, 6144)

    def test_a_contiguous_stride_is_not(self):
        """`q_out_stride0` is 6144 -- q's own row -- and the output is dense."""
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        assert _stride_past_its_tensors(self._kernel([("#11", 6144)])) is None

    def test_nor_are_the_extents_beside_it(self):
        """num_tokens=4, head_dim=256, num_q_heads=24: all inside a row."""
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = self._kernel([("#7", 4), ("#8", 256), ("#13", 24)])
        assert _stride_past_its_tensors(op) is None

    def test_a_constexpr_is_not_applied_to_a_pointer(self):
        """It is compiled into the kernel. `BLOCKS_PER_TILE=4096` is a tile size.

        That kernel prices correctly today, and reading its constexpr as a
        stride would refuse a price that works.
        """
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = self._kernel([("BLOCKS_PER_TILE", 4096), ("XBLOCK", 99999)])
        assert _stride_past_its_tensors(op) is None

    def test_a_single_row_cannot_be_walked_off(self):
        """With one row there is no second row for a stride to reach."""
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = _op(shapes=[(1, 8)], dtypes=["bfloat16"], scalars=[("#2", 4096)])
        op["name"] = "triton::k"
        assert _stride_past_its_tensors(op) is None

    def test_so_is_a_kernel_of_flat_tensors(self):
        """A 1-D argument has no row extent, so no integer contradicts it."""
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = _op(shapes=[(4097,), (5,)], dtypes=["int32", "int32"],
                 scalars=[("#5", 0)])
        op["name"] = "triton::k"
        assert _stride_past_its_tensors(op) is None

    def test_it_measures_against_the_widest_argument(self):
        """A stride belongs to one tensor, and the others are narrower.

        Against the narrowest, q's own dense stride would read as a fault: k's
        row is 1024 and q's is 6144.
        """
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        assert _stride_past_its_tensors(self._kernel([("#11", 6144)])) is None
        assert _stride_past_its_tensors(self._kernel([("#9", 6145)]))


class TestRebuildingAViewIntoItsBuffer:
    """With layout recorded, q comes back as a window on the fused qkv buffer.

    Without it, q came back dense at [4, 24, 256] and the kernel was still
    launched with stride0 = 14336, which is why pricing refuses such an operator
    rather than faulting the device. What is tested here is the arithmetic of
    the reconstruction and its refusals, not the launch: `_make_tensor`
    allocates on the device, so it is replaced with a host allocator.
    """

    def _op_with(self, layouts, shapes=((4, 24, 256), (4, 4, 256))):
        op = _op(shapes=list(shapes), dtypes=["bfloat16"] * len(shapes))
        op["layouts"] = layouts
        return op

    def _build(self, monkeypatch, op):
        """Allocated on meta: a 16k prefill chunk's fused buffer is 470 MB, and
        what is under test is the addressing, not the bytes."""
        import torch

        from atom.compass.runtime import microbench

        monkeypatch.setattr(
            microbench, "_make_tensor",
            lambda shape, dtype, values=None, span=None: torch.empty(
                tuple(int(d) for d in shape), dtype=getattr(torch, dtype),
                device="meta"))
        return microbench._operand_tensors(op, {}, {})

    #: q = fused[:, :24, :], k = fused[:, 24:28, :] over a [4, 56, 256] buffer.
    QKV = [[0, [[14336, 256, 1], 0, 57344, 0]],
           [1, [[14336, 256, 1], 6144, 57344, 0]]]

    def test_the_view_has_the_recorded_stride(self, monkeypatch):
        tensors = self._build(monkeypatch, self._op_with(self.QKV))
        assert tuple(tensors[0].stride()) == (14336, 256, 1)
        assert tuple(tensors[0].shape) == (4, 24, 256)

    def test_both_views_share_one_storage(self, monkeypatch):
        """Two windows on one buffer, as the projection wrote them."""
        tensors = self._build(monkeypatch, self._op_with(self.QKV))
        assert (tensors[0].untyped_storage()._cdata
                == tensors[1].untyped_storage()._cdata)
        assert tensors[1].storage_offset() == 6144

    def test_an_argument_without_layout_is_still_dense(self, monkeypatch):
        op = self._op_with([[0, [[14336, 256, 1], 0, 57344, 0]]])
        tensors = self._build(monkeypatch, op)
        assert tensors[1].is_contiguous()

    def test_a_view_that_leaves_its_storage_is_refused(self, monkeypatch):
        """The check that keeps a bad layout from becoming a memory fault."""
        op = self._op_with([[0, [[14336, 256, 1], 0, 24576, 0]]])
        assert self._build(monkeypatch, op) is None

    def test_so_is_a_stride_of_the_wrong_rank(self, monkeypatch):
        op = self._op_with([[0, [[14336, 256], 0, 57344, 0]]])
        assert self._build(monkeypatch, op) is None

    def test_so_is_a_shared_storage_of_two_dtypes(self, monkeypatch):
        """One allocation cannot be rebuilt as two element sizes at once."""
        op = self._op_with(self.QKV)
        op["dtypes"] = ["bfloat16", "float32"]
        assert self._build(monkeypatch, op) is None

    def test_so_is_an_owner_that_is_not_itself_recorded(self, monkeypatch):
        op = self._op_with([[1, [[14336, 256, 1], 6144, 57344, 0]]])
        assert self._build(monkeypatch, op) is None

    def test_a_prefill_chunks_views_rebuild_the_same_way(self, monkeypatch):
        """The shape that actually goes unpriced today: 16384 tokens, not 4.

        The stride is the same 14336 either way -- it is a property of the
        projection, not of the chunk -- so only the storage grows, and the last
        row of q must still land inside it.
        """
        fused = 16384 * 56 * 256
        op = self._op_with(
            [[0, [[14336, 256, 1], 0, fused, 0]],
             [1, [[14336, 256, 1], 6144, fused, 0]]],
            shapes=((16384, 24, 256), (16384, 4, 256)))
        tensors = self._build(monkeypatch, op)
        assert tuple(tensors[0].stride()) == (14336, 256, 1)
        assert (tensors[0].untyped_storage()._cdata
                == tensors[1].untyped_storage()._cdata)
        last = (16383 * 14336) + (23 * 256) + 255
        assert last < fused, "q's final element stays inside the buffer"


#: `_fused_qk_norm_single_kernel`, as it declares itself in layernorm.py. The
#: first six positions are the tensors; the guard only ever looks past them.
QK_NORM_PARAMS = [
    [0, "q_ptr"], [1, "k_ptr"], [2, "q_out_ptr"], [3, "k_out_ptr"],
    [4, "q_weight_ptr"], [5, "k_weight_ptr"],
    [6, "eps"], [7, "num_tokens"], [8, "head_dim"],
    [9, "q_in_stride0"], [10, "k_in_stride0"],
    [11, "q_out_stride0"], [12, "k_out_stride0"],
    [13, "num_q_heads"], [14, "num_k_heads"],
]


def _qk_norm(tokens, layouts=(), params=QK_NORM_PARAMS):
    """The recorded signature at a given chunk size, both norms alike.

    At decode the refusal fires on `q_in_stride0 = 14336`; at a 16k prefill
    chunk it fired first on `num_tokens = 16384`, and the two must be told
    apart by what the kernel calls them, not by which is bigger.
    """
    op = _op(shapes=[(tokens, 24, 256), (tokens, 4, 256),
                     (tokens, 24, 256), (tokens, 4, 256), (256,), (256,)],
             dtypes=["bfloat16"] * 6,
             scalars=[("#6", 1e-6), ("#7", tokens), ("#8", 256),
                      ("#9", 14336), ("#10", 14336), ("#11", 6144),
                      ("#12", 1024), ("#13", 24), ("#14", 4)])
    op["name"] = "triton::_fused_qk_norm_single_kernel"
    op["layouts"] = list(layouts)
    op["param_names"] = list(params)
    return op


#: q and k as the fused projection hands them over, at that chunk size.
def _qkv_layout(tokens):
    fused = tokens * 56 * 256
    return [[0, [[14336, 256, 1], 0, fused, 0]],
            [1, [[14336, 256, 1], 6144, fused, 0]]]


class TestTellingAStrideFromACountByItsDeclaredName:
    """The size test cannot do it, and the kernel's own signature can.

    At batch 4 the refusal fires on `q_in_stride0 = 14336`, which is real. At a
    16k prefill chunk it fires first on `num_tokens = 16384`, which addresses
    nothing -- so the whole family went unpriced at every long-domain shape for
    a number that was never a stride. Both are exercised here; passing the
    decode case alone would not show the difference.
    """

    def test_the_decode_norm_refuses_its_unrecorded_stride(self):
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        assert _stride_past_its_tensors(_qk_norm(4)) == ("#9", 14336, 6144)

    def test_and_prices_once_the_layout_is_recorded(self):
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = _qk_norm(4, layouts=_qkv_layout(4))
        assert _stride_past_its_tensors(op) is None

    def test_the_prefill_norm_does_not_refuse_its_token_count(self):
        """16384 is `num_tokens`. The kernel says so; its size says nothing."""
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = _qk_norm(16384, layouts=_qkv_layout(16384))
        assert _stride_past_its_tensors(op) is None

    def test_but_still_refuses_the_stride_when_the_layout_is_absent(self):
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        assert _stride_past_its_tensors(_qk_norm(16384)) == ("#9", 14336, 6144)

    def test_an_undeclared_stride_is_refused_whatever_the_layout_says(self):
        """A layout for q does not vouch for a stride belonging to nothing."""
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = _qk_norm(4, layouts=_qkv_layout(4))
        op["scalars"] = [["#9", 28672]]
        assert _stride_past_its_tensors(op) == ("#9", 28672, 6144)

    def test_a_graph_without_names_keeps_the_old_size_test(self):
        """Including its over-refusal: the prefill token count is refused.

        An artifact written before names were recorded has nothing better to go
        on, and relaxing it on the strength of a *later* graph's evidence would
        price it against a rebuild it never described.
        """
        from atom.compass.runtime.microbench import _stride_past_its_tensors

        op = _qk_norm(16384, params=[])
        assert _stride_past_its_tensors(op) == ("#7", 16384, 6144)


class _Type:
    """A schema argument type, which is only ever read as its text."""

    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


class _Schema:
    def __init__(self, *types, name=None, overload_name=None):
        self.arguments = [type("Arg", (), {"type": _Type(t)})() for t in types]
        if name is not None:
            self.name = name
        if overload_name is not None:
            self.overload_name = overload_name


class _Op:
    """An operator that carries a schema, as a torch OpOverload does."""

    def __init__(self, *types, name=None, overload_name=None):
        self._schema = _Schema(*types, name=name, overload_name=overload_name)


def _index_op(self_shape, out_shape):
    """A recorded `aten::index.Tensor`, as `_build_arg_sets` would see it."""
    return {"name": "aten::index.Tensor",
            "scalars": [],
            "output_shapes": [list(out_shape)] if out_shape else [],
            "input_shapes": [list(self_shape)]}


class TestATensorListArgument:
    """`aten::index.Tensor(Tensor self, Tensor?[] indices)`.

    The graph records tensors as one flat list, so the index tensor arrives
    where a container belongs. Handed a bare tensor, torch iterates it: a
    24-element index becomes 24 indices into a 2-D tensor and the call raises
    `IndexError: too many indices`, which is why every prefill head's region
    went unpriced while its gemm priced fine.

    A plain `Tensor[]` has no empty slots, so the recorded tensors fill it in
    order and there is nothing to settle. `Tensor?[]` is the hard case and has
    its own tests below.
    """

    def test_the_trailing_tensors_become_one_list(self):
        fn = _Op("Tensor[]", "int")
        assert _group_for_schema(fn, ["a", "b"], {}) == [["a", "b"]]

    def test_a_list_after_a_tensor_takes_what_is_left(self):
        fn = _Op("Tensor", "Tensor[]")
        assert _group_for_schema(fn, ["s", "i", "j"], {}) == ["s", ["i", "j"]]

    def test_the_list_survives_rebuilding_with_scalars(self):
        fn = _Op("Tensor", "Tensor[]")
        args, _ = _rebuild_args(_op(), _group_for_schema(fn, ["s", "i"], {}))
        assert args == ["s", ["i"]], "the list must stay one argument"

    def test_an_operator_without_a_list_is_untouched(self):
        fn = _Op("Tensor", "Tensor", "float")
        assert _group_for_schema(fn, ["a", "b"], {}) == ["a", "b"]

    def test_no_schema_at_all_is_untouched(self):
        assert _group_for_schema(object(), ["a", "b"], {}) == ["a", "b"]


class TestWhenTheSchemaLeavesAChoice:
    """Refuse rather than reconstruct something no one can check."""

    def test_two_tensor_lists_are_refused(self):
        fn = _Op("Tensor[]", "Tensor[]")
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, ["a", "b", "c"], {})
        assert "cannot be divided" in str(exc.value)

    def test_an_optional_tensor_beside_a_list_is_refused(self):
        """The graph does not record which optional arguments were present."""
        fn = _Op("Tensor", "Tensor?", "Tensor?[]")
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, ["self", "idx"], {})
        assert "optional" in str(exc.value)

    def test_too_few_tensors_for_the_schema_is_refused(self):
        fn = _Op("Tensor", "Tensor", "Tensor[]")
        with pytest.raises(ArgumentStructureRefusal):
            _group_for_schema(fn, ["only_one"], {})


class TestAnOptionalListOnAnUnmodelledOperator:
    """The placement rule below is advanced indexing's, not a general law.

    Recovering where a `None` sat needs the operator's own shape rule. One is
    written out here, for `aten::index.Tensor`. Applying it to the next
    operator that happens to declare a `Tensor?[]` would be assuming its
    semantics; those are refused, and stay refused, until someone models them.
    """

    def test_an_unnamed_operator_with_an_optional_list_is_refused(self):
        fn = _Op("Tensor", "Tensor?[]")
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, ["self", "idx"], _index_op((8, 8), (3, 8)))
        assert "not an operator whose placement of None is modelled" in str(
            exc.value)

    def test_a_different_named_operator_is_refused_by_name(self):
        fn = _Op("Tensor", "Tensor?[]",
                 name="aten::_unsafe_index", overload_name="Tensor")
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, ["self", "idx"], _index_op((8, 8), (3, 8)))
        assert "aten::_unsafe_index.Tensor" in str(exc.value)


class TestNonePlaceholdersInATensorList:
    """`x[:, idx]` records its indices as `[None, idx]` -- minus the None.

    Only the tensors are recorded, so one index tensor is consistent with
    indexing dimension 0 and with indexing dimension 1, and something has to
    say which. That something is arithmetic: each candidate placement's output
    shape is predicted from the recorded shapes and kept only if exactly one
    candidate reproduces the recorded output. Nothing is launched to find out.
    """

    @staticmethod
    def _idx(torch, n, dtype=None):
        return torch.arange(n, dtype=dtype or torch.int64)

    def test_the_prefill_head_gather_places_its_index_on_dim_0(self):
        """The real case: last token of each request out of the hidden rows."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        hidden = torch.zeros(1536, 5120, dtype=torch.bfloat16)
        idx = self._idx(torch, 3)
        grouped = _group_for_schema(fn, [hidden, idx],
                                    _index_op((1536, 5120), (3, 5120)))
        assert grouped[1] == [idx], "no placeholder in front of it"
        args, kw = _rebuild_args(_index_op((1536, 5120), (3, 5120)), grouped)
        assert tuple(fn(*args, **kw).shape) == (3, 5120)

    def test_a_column_gather_recovers_the_leading_placeholder(self):
        """`x[:, idx]` -- the same one recorded tensor, the other axis."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        hidden = torch.zeros(1536, 5120, dtype=torch.bfloat16)
        idx = self._idx(torch, 3)
        grouped = _group_for_schema(fn, [hidden, idx],
                                    _index_op((1536, 5120), (1536, 3)))
        assert grouped[1][0] is None and grouped[1][1] is idx
        args, kw = _rebuild_args(_index_op((1536, 5120), (1536, 3)), grouped)
        assert tuple(fn(*args, **kw).shape) == (1536, 3)

    def test_two_placements_of_the_same_shape_are_refused(self):
        """A square tensor and a full-length index: the counterexample.

        Both placements return `[8, 8]`, so replaying one and checking the
        shape afterwards cannot tell them apart -- it would confirm a guess.
        This asserts the ambiguity is real by running both, then asserts the
        module refuses without running either.
        """
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        square = torch.zeros(8, 8, dtype=torch.bfloat16)
        idx = self._idx(torch, 8)
        rows = fn(square, [idx])
        cols = fn(square, [None, idx])
        assert tuple(rows.shape) == tuple(cols.shape) == (8, 8), (
            "the premise: the output shape does not distinguish them")

        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, [square, idx], _index_op((8, 8), (8, 8)))
        assert "2 placements" in str(exc.value)
        assert "does not say which axes" in str(exc.value)

    def test_a_full_width_list_has_no_slot_for_a_placeholder(self):
        """Every axis indexed: the reconstruction is forced by the count."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        t = torch.zeros(4, 5, dtype=torch.bfloat16)
        i, j = self._idx(torch, 3), self._idx(torch, 3)
        grouped = _group_for_schema(fn, [t, i, j], _index_op((4, 5), (3,)))
        assert grouped[1] == [i, j]

    def test_a_record_with_no_output_shape_is_refused(self):
        """An older graph cannot settle a placement, so it does not get one."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, [torch.zeros(4, 5), self._idx(torch, 3)],
                              _index_op((4, 5), None))
        assert "cannot be settled" in str(exc.value)

    def test_an_output_no_placement_reaches_is_refused(self):
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, [torch.zeros(4, 5), self._idx(torch, 3)],
                              _index_op((4, 5), (7, 7)))
        assert "no placement" in str(exc.value)

    def test_a_boolean_mask_is_refused(self):
        """A mask selects a data-dependent count; a shape cannot pin it down."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        mask = torch.zeros(4, dtype=torch.bool)
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, [torch.zeros(4, 5), mask],
                              _index_op((4, 5), (0, 5)))
        assert "boolean mask" in str(exc.value)

    def test_an_index_that_is_not_one_dimensional_is_refused(self):
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, [torch.zeros(4, 5),
                                   torch.zeros(2, 2, dtype=torch.int64)],
                              _index_op((4, 5), (2, 2, 5)))
        assert "one-dimensional" in str(exc.value)

    def test_more_indices_than_dimensions_is_refused(self):
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        with pytest.raises(ArgumentStructureRefusal) as exc:
            _group_for_schema(fn, [torch.zeros(4), self._idx(torch, 2),
                                   self._idx(torch, 2)],
                              _index_op((4,), (2,)))
        assert "1-dimensional tensor" in str(exc.value)


class TestThePredictedShapeIsTorchsShape:
    """The predictor stands in for the operator, so hold it to the operator.

    These compare the arithmetic against what torch actually returns, on the
    CPU. They are evidence for the rule, not the gate the runtime uses: the
    runtime never calls the operator to decide how to call the operator.
    """

    def test_each_axis_of_a_matrix(self):
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        t = torch.zeros(6, 7)
        idx = torch.arange(3)
        assert _index_output_shape((6, 7), (0,), (3,)) == tuple(
            fn(t, [idx]).shape)
        assert _index_output_shape((6, 7), (1,), (3,)) == tuple(
            fn(t, [None, idx]).shape)

    def test_adjacent_axes_keep_the_broadcast_in_place(self):
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        t = torch.zeros(2, 3, 4)
        idx = torch.arange(5) % 2
        assert _index_output_shape((2, 3, 4), (0, 1), (5,)) == tuple(
            fn(t, [idx, idx]).shape)

    def test_separated_axes_move_the_broadcast_to_the_front(self):
        """The rule that makes a non-adjacent placement distinguishable."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        t = torch.zeros(2, 3, 4)
        idx = torch.arange(5) % 2
        assert _index_output_shape((2, 3, 4), (0, 2), (5,)) == tuple(
            fn(t, [idx, None, idx]).shape)

    def test_a_separated_placement_is_recovered_from_its_output(self):
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        t = torch.zeros(2, 3, 4)
        idx = torch.arange(5) % 2
        grouped = _group_for_schema(fn, [t, idx, idx],
                                    _index_op((2, 3, 4), (5, 3)))
        assert grouped[1][1] is None, "the middle axis was not indexed"
        args, kw = _rebuild_args(_index_op((2, 3, 4), (5, 3)), grouped)
        assert tuple(fn(*args, **kw).shape) == (5, 3)


class TestAgainstTheOperatorsOwnSchema:
    """Read the real schemas, not a remembered spelling of them.

    This torch prints `index.Tensor`'s indices as `List[Optional[Tensor]]`; the
    schema language spells the same type `Tensor?[]`. A first cut of this fix
    matched the spelling, classified the argument as "not a list", and left the
    bug exactly where it was -- passing its own hand-written tests. These call
    torch, on the CPU, so a spelling change fails here instead of in a GPU run.
    """

    def test_the_optional_list_is_seen_as_one(self):
        torch = pytest.importorskip("torch")
        from atom.compass.runtime.microbench import (
            _MAYBE_TENSOR_LIST,
            _ONE_TENSOR,
            _schema_kinds,
        )

        fn = torch.ops.aten.index.Tensor
        assert _schema_kinds(fn) == [_ONE_TENSOR, _MAYBE_TENSOR_LIST]

    def test_a_list_of_int_is_not_a_list_of_tensor(self):
        """convolution's stride is `List[int]` and must stay one argument."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.convolution.default
        assert _group_for_schema(fn, ["in", "w"], {}) == ["in", "w"]

    def test_a_plain_gemm_is_untouched(self):
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.addmm.default
        assert _group_for_schema(fn, ["c", "a", "b"], {}) == ["c", "a", "b"]

    def test_ungrouped_the_recorded_prefill_call_still_raises(self):
        """The bug this fixes, kept as the reason the grouping is there."""
        torch = pytest.importorskip("torch")
        fn = torch.ops.aten.index.Tensor
        hidden = torch.zeros(1536, 8, dtype=torch.bfloat16)
        idx = torch.arange(3)
        with pytest.raises(IndexError):
            fn(hidden, idx)
        assert tuple(fn(hidden, [idx]).shape) == (3, 8)
