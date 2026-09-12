"""What a derived graph says about communication.

The failure these guard against is not a crash. It is a TP4 graph that traces
cleanly, prices completely, and reports tensor parallelism as free -- because
simulated TP either passed the collective through or reimplemented it locally,
and the tracer dutifully recorded the local copy.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from atom.compass.core.graph import OpGraph  # noqa: E402
from atom.compass.runtime.derive import (  # noqa: E402
    head_gather_opspec, record_collectives)


class FakeGroup:
    """Enough of a GroupCoordinator for the wrapper to bind against.

    ``all_reduce`` carries the real one's extra arguments because the wrapper
    binds its signature to recover their defaults, and a fake that omits them
    would pass a test the real group fails.
    """

    unique_name = "tp:0"
    simulated_tp_physical_world_size = 1

    def __init__(self, world_size: int = 4):
        self.world_size = world_size
        self.gather_calls = 0

    def all_reduce(self, input_, ca_use_new: bool = True,
                   ca_fp8_quant: bool = False, prefill_support: bool = False):
        return input_

    def all_gather(self, input_, use_custom: bool = False, dim: int = -1):
        # The local reimplementation. If a test sees this run, the graph it
        # produced holds a copy where the deployment has a transfer.
        self.gather_calls += 1
        shape = list(input_.shape)
        shape[dim % len(shape)] *= self.world_size
        return torch.zeros(shape, dtype=input_.dtype)


@pytest.fixture()
def group(monkeypatch):
    # `import aiter.dist.parallel_state as ps` does not work: `aiter.dist` is
    # bound to `torch.distributed` in aiter's __init__, which shadows the
    # subpackage for the dotted form. importlib resolves the real module, which
    # is also how `from aiter.dist.parallel_state import ...` succeeds inside
    # derive.py.
    import importlib

    ps = importlib.import_module("aiter.dist.parallel_state")

    g = FakeGroup()
    monkeypatch.setattr(ps, "get_tp_group", lambda: g, raising=False)
    return g


class TestTheGatherIsATransferNotACopy:
    def test_the_head_gather_is_recorded_as_a_collective(self, group):
        graph = OpGraph()
        logits = torch.zeros(1, 124160, dtype=torch.bfloat16)
        with record_collectives(graph):
            group.all_gather(logits, use_custom=True, dim=-1)

        gathers = [op for op in graph.ops
                   if "all_gather" in op.name]
        assert len(gathers) == 1, [op.name for op in graph.ops]
        op = gathers[0]
        assert op.name == "aiter::all_gather_unreg"
        assert op.input_shapes == ((1, 124160),)
        assert op.output_shapes == ((1, 496640),)
        assert op.dtypes == ("bfloat16",)
        assert op.group == "tp"

    def test_the_local_reimplementation_does_not_run(self, group):
        with record_collectives(OpGraph()):
            group.all_gather(torch.zeros(1, 8, dtype=torch.bfloat16))
        assert group.gather_calls == 0

    def test_the_returned_shape_is_the_deployments(self, group):
        with record_collectives(OpGraph()):
            out = group.all_gather(torch.zeros(2, 16, dtype=torch.bfloat16))
        assert tuple(out.shape) == (2, 64)
        assert out.dtype == torch.bfloat16

    def test_a_leading_axis_gather_grows_the_leading_axis(self, group):
        graph = OpGraph()
        with record_collectives(graph):
            group.all_gather(torch.zeros(3, 5, dtype=torch.bfloat16), dim=0)
        op = next(o for o in graph.ops if "all_gather" in o.name)
        assert op.output_shapes == ((12, 5),)
        assert ("#5", 0) in op.scalars

    def test_the_group_is_restored_on_exit(self, group):
        # Asserted by behaviour, not by identity: `group.all_gather` is a bound
        # method and every attribute access builds a new object, so `is` would
        # fail on a group that was restored perfectly. What matters is that the
        # call reaches the group's own implementation again -- a derivation must
        # not leave the process with a recorder wired into it.
        graph = OpGraph()
        with record_collectives(graph):
            group.all_gather(torch.zeros(1, 8, dtype=torch.bfloat16))
        assert group.gather_calls == 0
        group.all_gather(torch.zeros(1, 8, dtype=torch.bfloat16))
        assert group.gather_calls == 1
        assert len([o for o in graph.ops if "all_gather" in o.name]) == 1

    def test_the_width_comes_from_the_group_not_the_caller(self, group):
        # Simulated TP sets `world_size` to the logical width; a derivation of
        # TP2 and one of TP4 differ only there, and the gathered axis has to
        # follow it or both graphs claim the same amount of traffic.
        group.world_size = 2
        graph = OpGraph()
        with record_collectives(graph):
            group.all_gather(torch.zeros(1, 100, dtype=torch.bfloat16))
        op = next(o for o in graph.ops if "all_gather" in o.name)
        assert op.output_shapes == ((1, 200),)


class TestTheReduceIsStillRecorded:
    def test_all_reduce_keeps_its_arguments(self, group):
        graph = OpGraph()
        with record_collectives(graph):
            group.all_reduce(torch.zeros(4, 5120, dtype=torch.bfloat16))
        op = next(o for o in graph.ops if "all_reduce" in o.name)
        assert op.name == "aiter::all_reduce_"
        # Bound from the real signature's defaults rather than omitted: a
        # derived signature missing them never matches a captured one.
        assert ("#2", True) in op.scalars
        assert ("#3", False) in op.scalars

    def test_both_collectives_survive_one_context(self, group):
        graph = OpGraph()
        with record_collectives(graph):
            group.all_reduce(torch.zeros(4, 5120, dtype=torch.bfloat16))
            group.all_gather(torch.zeros(1, 64, dtype=torch.bfloat16))
        names = [op.name for op in graph.ops]
        assert "aiter::all_reduce_" in names
        assert "aiter::all_gather_unreg" in names

    def test_the_fake_matches_the_real_signature(self):
        # Guards the fake itself: if GroupCoordinator.all_reduce grows or loses
        # an argument, this fake stops standing in for it and the tests above
        # start passing for the wrong reason.
        ps = pytest.importorskip("aiter.dist.parallel_state")
        real = set(inspect.signature(ps.GroupCoordinator.all_reduce).parameters)
        fake = set(inspect.signature(FakeGroup.all_reduce).parameters)
        assert real <= fake, real - fake


class TestTheRecorderAndThePricerAgreeOnTheSignature:
    """The gather's price comes from a different process than its record.

    A derivation records the operator into a graph; a two-rank harness performs
    the real collective and writes a price under its signature
    (`agent_scratch/g4/ag_probe.py`, STAGE=verify). Those two never run
    together, so nothing at runtime would notice them disagreeing -- and the
    symptom of disagreement is not an error but a lookup miss, which prices
    tensor-parallel communication at zero. One builder, checked here.
    """

    def test_the_recorded_op_is_the_helper_s_op(self, group):
        graph = OpGraph()
        logits = torch.zeros(20, 124160, dtype=torch.bfloat16)
        with record_collectives(graph):
            group.all_gather(logits, use_custom=True, dim=-1)
        recorded = next(o for o in graph.ops if "all_gather" in o.name)
        assert recorded == head_gather_opspec(group, logits, -1)

    def test_the_signature_survives_the_round_trip_to_a_price_key(self, group):
        from atom.compass.runtime.microbench import signature_of

        logits = torch.zeros(20, 124160, dtype=torch.bfloat16)
        graph = OpGraph()
        graph.add(head_gather_opspec(group, logits, -1))
        blob = OpGraph.from_dict(graph.to_dict())
        assert (signature_of(graph.to_dict()["ops"][0])
                == signature_of(blob.to_dict()["ops"][0]))

    def test_the_width_is_in_the_signature(self, group):
        # Two widths gather different amounts over different numbers of links.
        # A signature that did not separate them would let a TP4 measurement
        # answer a TP2 lookup.
        from atom.compass.runtime.microbench import signature_of

        logits = torch.zeros(20, 124160, dtype=torch.bfloat16)
        # Explicitly, not by relying on the fixture's default: the input shape
        # is the same at both widths -- each rank holds one shard either way --
        # so the width reaches the signature only through `context`, and a test
        # that silently compared one width against itself would pass whether or
        # not it did.
        group.world_size = 2
        two = OpGraph()
        two.add(head_gather_opspec(group, logits, -1))
        group.world_size = 4
        four = OpGraph()
        four.add(head_gather_opspec(group, logits, -1))
        assert (signature_of(two.to_dict()["ops"][0])
                != signature_of(four.to_dict()["ops"][0]))


class TestAnUnrebuildableCallIsNotRebuilt:
    """The gather cannot be priced by reconstruction, and says so.

    A two-rank probe through the real ``GroupCoordinator`` confirmed the tuple
    ``all_gather_unreg(_fa, inp, reg_buffer, out, reg_bytes, dim)``. Two of
    those six are a live communicator handle and a live IPC pool address --
    process state, not step state -- so no artifact can carry them and a
    rebuilt call would be inventing pointers. It does not reliably fault on an
    invented pointer; it reads whatever is mapped. So the refusal has to happen
    before anything is resolved or allocated, and it has to survive the
    artifact.
    """

    def test_the_derived_gather_says_its_tuple_is_live_state(self, group):
        graph = OpGraph()
        with record_collectives(graph):
            group.all_gather(torch.zeros(1, 64, dtype=torch.bfloat16))
        op = next(o for o in graph.ops if "all_gather" in o.name)
        assert op.abi == "live-state"

    def test_the_reduce_does_not(self, group):
        # Its tuple was read off a real dispatcher record, and none of its
        # arguments is a pointer.
        graph = OpGraph()
        with record_collectives(graph):
            group.all_reduce(torch.zeros(4, 5120, dtype=torch.bfloat16))
        op = next(o for o in graph.ops if "all_reduce" in o.name)
        assert op.abi == ""

    def test_it_survives_the_artifact(self, group):
        # A mark that is lost on the way to disk protects the deriving process
        # and nothing else; pricing reads the JSON.
        graph = OpGraph()
        with record_collectives(graph):
            group.all_gather(torch.zeros(1, 64, dtype=torch.bfloat16))
        back = OpGraph.from_dict(graph.to_dict())
        op = next(o for o in back.ops if "all_gather" in o.name)
        assert op.abi == "live-state"

    def test_pricing_refuses_it_before_resolving_anything(self, tmp_path,
                                                          monkeypatch):
        import json

        import atom.utils.forward_context as forward_context
        from atom.compass.runtime import microbench

        path = tmp_path / "graph.json"
        path.write_text(json.dumps({"ops": [{
            "name": "aiter::all_gather_unreg",
            "input_shapes": [[1, 124160]], "output_shapes": [[1, 248320]],
            "dtypes": ["bfloat16"], "scalars": [], "group": "tp",
            "abi": "live-state"}]}))
        monkeypatch.setattr(forward_context, "reset_forward_context",
                            lambda: None)

        def explode(_name):
            raise AssertionError("resolved an operator it cannot rebuild")

        monkeypatch.setattr(microbench, "_resolve", explode)

        result = microbench.price_graph(str(path))

        assert not result["prices"]
        assert "live communicator handle" in next(
            iter(result["unpriced"].values()))

    def test_an_inferred_tuple_is_refused_too(self, tmp_path, monkeypatch):
        # The other reason an operator can carry: a tuple read off a
        # declaration and never checked. Distinct refusal, same guard.
        import json

        import atom.utils.forward_context as forward_context
        from atom.compass.runtime import microbench

        path = tmp_path / "graph.json"
        path.write_text(json.dumps({"ops": [{
            "name": "aiter::some_future_collective",
            "input_shapes": [[1, 8]], "dtypes": ["bfloat16"], "scalars": [],
            "abi": "unverified"}]}))
        monkeypatch.setattr(forward_context, "reset_forward_context",
                            lambda: None)

        result = microbench.price_graph(str(path))

        assert "inferred from a declaration" in next(
            iter(result["unpriced"].values()))

    def test_a_recorded_tuple_is_still_priced(self, tmp_path, monkeypatch):
        # The guard is about provenance, not about collectives: an all-reduce
        # whose tuple came from a capture must still reach the bench, or this
        # refusal silently drops every priced collective in the library.
        import json

        import atom.utils.forward_context as forward_context
        from atom.compass.runtime import microbench

        path = tmp_path / "graph.json"
        path.write_text(json.dumps({"ops": [{
            "name": "aiter::all_reduce_", "input_shapes": [[4, 5120]],
            "dtypes": ["bfloat16"], "scalars": [], "group": "tp"}]}))
        monkeypatch.setattr(forward_context, "reset_forward_context",
                            lambda: None)

        reached = []
        monkeypatch.setattr(microbench, "_resolve",
                            lambda n: reached.append(n))

        microbench.price_graph(str(path))

        assert reached == ["aiter::all_reduce_"]
