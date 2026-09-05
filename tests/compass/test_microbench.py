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
