"""What the dispatch tracer records about where tensors come from and go.

Shapes alone do not make a graph walkable for liveness: two tensors of the same
shape are the same entry. What makes it walkable is `inputs_from` and
`output_aliases`, and both are keyed on storage identity -- so these tests are
about that key, on the device derivation actually runs on.

Meta is that device, and a meta tensor has no address: `data_ptr()` is 0 for
every storage ever made, and it does not raise. The trace that came of keying
on the address alone was not missing anything a reader could notice; it was
wrong in a way that reads as a very tidy graph, every input produced by the
operator immediately before it and every output an alias. The CPU trace of the
same function is the control: the two should agree about provenance, because
provenance is a property of the function, not of where its tensors live.
"""

import pytest

torch = pytest.importorskip("torch")

from atom.compass.core.graph import OpSpec  # noqa: E402
from atom.compass.runtime.meta import MetaOpTracer, _storage_of  # noqa: E402


def _tensor(device):
    return torch.zeros(4, 4, device=device)


@torch.library.custom_op("compass_test::fill_", mutates_args={"out"})
def _fill(out: torch.Tensor, x: torch.Tensor) -> None:
    """An operator that writes into a destination and returns nothing.

    The shape of an aiter out-kernel in three lines: what it produced is the
    buffer it was handed, a dispatch tracer sees that only in the arguments,
    and the destination is the first of them.
    """
    out.copy_(x)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_two_storages_are_two_keys_and_a_view_is_one(device):
    a, b = _tensor(device), _tensor(device)
    assert _storage_of(a) is not None
    assert _storage_of(a) != _storage_of(b)
    # A reshape allocates nothing. Counting it as its own storage would invent
    # activation memory that never existed, which is the reason the key is a
    # storage and not a tensor.
    assert _storage_of(a.view(2, 8)) == _storage_of(a)


def _trace(device):
    """A function with one of each provenance, traced on `device`.

    It has a residual and a second weight on purpose. A straight chain cannot
    tell a correct producer map from a collapsed one: where every operator
    reads only what the one before it wrote, "the operator before" is the right
    answer by accident. What separates them is a read of something older -- the
    residual -- and a read of something this step never produced at all -- the
    second weight, reached after the first operator has run.
    """
    tracer = MetaOpTracer()
    w1 = torch.ones(4, 4, device=device)     # a weight: from before the step
    w2 = torch.ones(4, 4, device=device)     # a second one, read later
    x = torch.zeros(4, 4, device=device)     # the step's input: likewise
    with tracer:
        h = torch.mm(x, w1)                  # fresh allocation, from nothing
        y = torch.relu(h)                    # fresh allocation, from mm
        z = torch.mm(y, w2)                  # reads a weight, not an activation
        out = z + h                          # the residual: reads h again
        out.add_(1.0)                        # in place: aliases its input
    return tracer, [op.name for op in tracer.graph.ops]


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_provenance_is_recorded_operator_by_operator(device):
    tracer, names = _trace(device)
    assert names == ["aten::mm", "aten::relu", "aten::mm",
                     "aten::add.Tensor", "aten::add_.Tensor"]
    ops = tracer.graph.ops

    # The weight and the step input were not produced by this step.
    assert ops[0].inputs_from == (-1, -1)
    # relu reads what mm wrote, and nothing else.
    assert ops[1].inputs_from == (0,)
    # The second mm reads relu's output and a weight. Under one collapsed key
    # the weight reads as relu's output too, and a weight counted as an
    # activation is activation memory that is freed when it is not.
    assert ops[2].inputs_from == (1, -1)
    # The residual reaches back past the operator before it.
    assert ops[3].inputs_from == (2, 0)
    assert ops[4].inputs_from == (3,)

    # Four fresh allocations and one write into an existing buffer. Reading
    # these wrongly is the difference between a live set that holds four
    # tensors and one that holds none.
    assert [op.output_aliases for op in ops] == [
        (None,), (None,), (None,), (None,), (3,)]


def test_meta_records_the_same_provenance_as_cpu():
    """The control, stated as one comparison rather than two sets of numbers.

    Before `_storage_of` had a key for a tensor with no address, this is the
    assertion that failed: on meta every `inputs_from` was the operator before
    and every `output_aliases` was an alias, because every storage was one
    storage.
    """
    on_cpu, cpu_names = _trace("cpu")
    on_meta, meta_names = _trace("meta")
    assert meta_names == cpu_names
    assert ([op.inputs_from for op in on_meta.graph.ops]
            == [op.inputs_from for op in on_cpu.graph.ops])
    assert ([op.output_aliases for op in on_meta.graph.ops]
            == [op.output_aliases for op in on_cpu.graph.ops])


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_a_destination_the_trace_never_saw_is_recorded_as_this_step_s(device):
    """The out-variant recovery, which needs `_seen` to be a real set.

    An operator that returns no tensor writes into the destination it was
    handed. Where that buffer was allocated inside a wrapper the tracer cannot
    see into, it is real memory belonging to this step and nothing else in the
    graph says so. With every storage collapsed to one key, `_seen` saturates
    on the first tensor and this never fires again.
    """
    tracer = MetaOpTracer()
    x = torch.ones(4, 4, device=device)
    # Allocated where the tracer cannot see it, which is the case the recovery
    # is for: `torch.empty` inside a wrapper that is itself a custom operator
    # never reaches a dispatch tracer.
    out = torch.empty(4, 4, device=device)
    with tracer:
        torch.ops.compass_test.fill_(out, x)
    named = [op.name for op in tracer.graph.ops]
    assert named == ["compass_test::fill_"]
    op = tracer.graph.ops[0]
    # The operator returned nothing, so the destination is what it produced.
    assert op.output_shapes == ((4, 4),)
    assert op.output_aliases == (None,)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_a_destination_the_trace_has_seen_belongs_to_whoever_wrote_it(device):
    """The other half: seeing it twice must not count it twice."""
    tracer = MetaOpTracer()
    x = torch.ones(4, 4, device=device)
    with tracer:
        out = torch.zeros(4, 4, device=device)
        torch.ops.compass_test.fill_(out, x)
    named = [op.name for op in tracer.graph.ops]
    assert named[-1] == "compass_test::fill_"
    # `zeros` allocated it and is recorded as having done so; the fill claims
    # no output of its own, because claiming one would double the buffer.
    assert tracer.graph.ops[-1].output_shapes == ()


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_deaths_are_stamped_onto_the_operator_whose_output_died(device):
    """The tracer watched every tensor go; nothing wrote it down.

    Only the capture path stamped, and every template on disk came down the
    derivation path, so every derived graph reached the memory walk with no
    `dies_at` -- where the walk falls back to a last-read rule and returns a
    number either way.
    """
    tracer = MetaOpTracer()
    w = torch.ones(4, 4, device=device)
    x = torch.zeros(4, 4, device=device)
    with tracer:
        h = torch.mm(x, w)
        y = torch.relu(h)
        del h                    # dies while relu is the operator in progress
        z = torch.mm(y, w)
        del y                    # dies while the second mm is
    assert tracer.deaths, "the finalizers did not fire"
    assert tracer.stamp_deaths() == 2
    ops = tracer.graph.ops
    assert ops[0].dies_at == (1,)
    assert ops[1].dies_at == (2,)
    # `z` is still held, so its producer carries no death and the walk holds
    # it to the end of the step -- which is what it did.
    assert not ops[2].dies_at
    del z


def test_a_synthesized_operator_is_given_the_producer_map_s_answer():
    """What `record_collectives` needs, and added straight to a graph loses.

    A collective under simulated tensor parallelism never dispatches, so its
    operator is synthesized. Appended to the graph alone it reads nothing --
    its input looks unread by anything in the step -- and produces nothing the
    map knows about, so the next operator is recorded as reading the tensor the
    collective was handed instead of the one it returned.
    """
    tracer = MetaOpTracer()
    x = torch.zeros(4, 4, device="meta")
    w = torch.ones(4, 4, device="meta")
    with tracer:
        h = torch.mm(x, w)
        out = torch.empty(4, 4, device="meta")
        tracer.note_operator(
            OpSpec(name="aiter::all_reduce_", input_shapes=((4, 4),),
                   output_shapes=((4, 4),), dtypes=("bfloat16",),
                   group="tp", output_aliases=(None,)),
            inputs=(h,), outputs=(out,))
        torch.relu(out)
    names = [op.name for op in tracer.graph.ops]
    index = names.index("aiter::all_reduce_")
    collective = tracer.graph.ops[index]
    assert collective.inputs_from == (0,)          # the gemm, not nothing
    assert collective.output_aliases == (None,)    # out of place, verified
    assert names[-1] == "aten::relu"
    assert tracer.graph.ops[-1].inputs_from == (index,)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_a_reused_key_is_not_credited_to_the_tensor_that_died(device):
    """`_died` forgets the key, so the next tensor at it is a new tensor."""
    tracer = MetaOpTracer()
    tracer._producers[("storage", 7)] = 3
    tracer._seen.add(("storage", 7))
    tracer._died(3, 0, ("storage", 7))
    assert ("storage", 7) not in tracer._producers
    assert ("storage", 7) not in tracer._seen
    # A key held by somebody else is left alone.
    tracer._producers[("storage", 9)] = 5
    tracer._died(3, 0, ("storage", 9))
    assert tracer._producers[("storage", 9)] == 5
