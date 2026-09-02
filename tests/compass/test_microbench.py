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
