"""Recording where a tensor argument sat inside its allocation.

A shape says how big a tensor is. It cannot say the tensor is a *view* into
something larger, and for a Triton kernel -- which takes pointers and strides as
separate plain integers -- that difference decides whether a rebuilt call reads
its own memory or walks off the end of the device. `_layouts_of` records the
difference; `OpGraph` carries it; `microbench` rebuilds from it.

The fused QKV projection is the case that forced this: it writes one
``[N, 56, 256]`` buffer and hands the qk-norm kernel ``q`` as ``[N, 24, 256]``
with the *buffer's* row stride, 14336.
"""

import pytest

torch = pytest.importorskip("torch")

from atom.compass.core.graph import OpGraph, OpSpec
from atom.compass.runtime.meta import _layouts_of


def _qkv(tokens=4, dev="meta"):
    """q and k as the projection hands them over: two windows on one buffer."""
    fused = torch.empty((tokens, 56, 256), dtype=torch.bfloat16, device=dev)
    q = fused[:, :24, :]
    k = fused[:, 24:28, :]
    return fused, q, k


class TestWhatALayoutRecords:
    def test_a_view_carries_the_buffers_stride_not_its_own(self):
        _fused, q, _k = _qkv()
        (pos, (stride, offset, elements, owner)), = _layouts_of([q])
        assert pos == 0
        assert stride == (14336, 256, 1), "q's rows are 14336 apart, not 6144"
        assert offset == 0
        assert elements == 4 * 56 * 256
        assert owner == 0

    def test_two_windows_on_one_buffer_name_the_same_owner(self):
        """Otherwise they rebuild as two buffers -- twice the traffic."""
        _fused, q, k = _qkv()
        layouts = dict(_layouts_of([q, k]))
        assert layouts[0][3] == 0
        assert layouts[1][3] == 0, "k's storage is q's storage"
        assert layouts[1][1] == 24 * 256, "k starts where q's heads end"

    def test_a_plain_tensor_gets_no_entry(self):
        """A contiguous tensor alone in its storage is fully described by shape."""
        t = torch.empty((8, 16), dtype=torch.bfloat16, device="meta")
        assert _layouts_of([t]) == ()

    def test_a_non_tensor_argument_is_skipped(self):
        assert _layouts_of([None, 4, "grid"]) == ()

    def test_it_works_on_meta_where_derivation_runs(self):
        """Unlike `_int_values_of` and `_int_ranges_of`, which read contents."""
        _fused, q, _k = _qkv(dev="meta")
        assert q.device.type == "meta"
        assert _layouts_of([q])


class TestALayoutSurvivesTheArtifact:
    def test_it_round_trips_through_the_graph_file(self):
        _fused, q, k = _qkv()
        graph = OpGraph()
        graph.add(OpSpec(name="triton::qk_norm",
                         input_shapes=((4, 24, 256), (4, 4, 256)),
                         output_shapes=(),
                         dtypes=("bfloat16", "bfloat16"),
                         layouts=_layouts_of([q, k])))
        back = OpGraph.from_dict(graph.to_dict())
        assert back.ops[0].layouts == graph.ops[0].layouts

    def test_the_declared_parameter_names_round_trip_too(self):
        from atom.compass.runtime.triton_trace import _param_names

        class Jit:
            arg_names = ["q_ptr", "k_ptr", "eps", "num_tokens", "q_in_stride0"]

        names = _param_names(Jit(), [None] * 5)
        graph = OpGraph()
        graph.add(OpSpec(name="triton::qk_norm", input_shapes=((4, 24, 256),),
                         output_shapes=(), dtypes=("bfloat16",),
                         param_names=names))
        back = OpGraph.from_dict(graph.to_dict())
        assert back.ops[0].param_names == names
        assert dict(names)[4] == "q_in_stride0"

    def test_a_kernel_that_declares_nothing_records_nothing(self):
        """Refused later, rather than guessed at here."""
        from atom.compass.runtime.triton_trace import _param_names

        assert _param_names(object(), [1, 2]) == ()

    def test_names_stop_at_the_arguments_actually_passed(self):
        from atom.compass.runtime.triton_trace import _param_names

        class Jit:
            arg_names = ["a", "b", "c", "BLOCK"]

        assert _param_names(Jit(), [None, None]) == ((0, "a"), (1, "b"))

    def test_a_graph_written_before_layouts_reads_back_empty(self):
        """Not an error: an older artifact is a graph about shapes."""
        graph = OpGraph()
        graph.add(OpSpec(name="aiter::x", input_shapes=((4, 8),),
                         output_shapes=(), dtypes=("bfloat16",)))
        data = graph.to_dict()
        for op in data["ops"]:
            op.pop("layouts")
        assert OpGraph.from_dict(data).ops[0].layouts == ()
