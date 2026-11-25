# SPDX-License-Identifier: MIT
"""An `op_graph` is keyed by the ATOM source and torch it was traced under.

A graph a capture published under another ATOM source or torch than a run's is
a miss there: the run traces the shape and publishes its own graph beside it,
and reads that one; a reader that traces nothing, `measure ops` or a run given
no ATOM arguments, names the field that differs. Under the same readings each
reads the graph it finds, and the run traces nothing.

No device: a run's lazy traces take a stand-in tracer, in a thread of this
process, whose one leaf names the trace.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_op_graph_arch import ENGINE

from atom.compass import capture
from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactRefusal, ArtifactStore, Reading
from atom.compass.bench import ops
from atom.compass.capture.forward import decode, structure
from atom.compass.capture.graph import Traced
from atom.compass.ir import Graph, Op

ARCH = "gfx942"
#: Each field a graph's key holds beside its structure and shape, with the
#: axis it reads and another reading of that axis.
OTHER = {
    "source_root": ("atom_src", Reading("atom_py", "0" * 64)),
    "torch": ("torch", Reading.stated("2.0.0+other")),
}


@pytest.fixture
def config(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        **ENGINE,
    )


def tracer(config, leaf):
    """A stand-in for `forward.Tracer`: each decode shape it is given as one
    graph of `config` on `ARCH` holding the one leaf named `leaf`."""
    at = structure(config, "decode", True, ARCH)
    model = compass_run._file_digest(config.model, "config.json")

    def trace(shapes, unsplit):
        graph = Graph(Traced(at, str(shapes[0])), Op(leaf, "opaque_leaf", (), ()))
        return [(graph, model, {"arch": ARCH})]

    return SimpleNamespace(trace=trace)


def published(monkeypatch, store, config, leaf, field=None):
    """The key `trace ops` publishes `config`'s decode B=2 graph of one leaf
    `leaf` at, in a process reading this one's ATOM source and torch, or with
    `field`'s axis read as `OTHER` reads it."""
    [(graph, model, declared)] = tracer(config, leaf).trace([decode(2)], [])
    under = capture.traced_under()
    if field:
        axis, reading = OTHER[field]
        under = {**under, axis: reading}
    with monkeypatch.context() as m:
        m.setattr(capture, "traced_under", lambda: under)
        return capture.publish(store, graph, model, declared, 1, [], "trace ops")


def lookup(monkeypatch, store, config, leaf):
    """A run's lookup over `store`, given ATOM's arguments, each lazy trace
    tracing one leaf `leaf`; and its counts."""
    stand_in = tracer(config, leaf)
    monkeypatch.setattr(compass_run, "_tracer", lambda *args: (None, stand_in))
    monkeypatch.setattr(compass_run, "_trace_process", lambda: ThreadPoolExecutor(1))
    lazy = {"lazy_traces": 0, "lazy_trace_wall_seconds": 0}
    args = ["--model", config.model]
    return compass_run._op_graphs(str(store.root), config, ARCH, args, lazy), lazy


@pytest.mark.parametrize("field", OTHER)
def test_a_run_traces_again_a_graph_traced_under_another_and_publishes_it_beside(
    monkeypatch, tmp_path, config, field
):
    store = ArtifactStore(tmp_path / "store")
    other = published(monkeypatch, store, config, "aiter.other", field)
    run, lazy = lookup(monkeypatch, store, config, "aiter.here")

    assert run("decode", 2, 2)["op"] == "aiter.here"
    assert lazy["lazy_traces"] == 1
    here = capture.graph_key(other.values["structure"], "decode B=2")
    assert [k for k in here.values if here.values[k] != other.values[k]] == [field]
    assert store.read(other).key == other and store.read(here).key == here


@pytest.mark.parametrize("field", OTHER)
def test_a_reader_that_traces_nothing_names_the_field_another_s_graph_differs_in(
    monkeypatch, tmp_path, config, field
):
    store = ArtifactStore(tmp_path / "store")
    other = published(monkeypatch, store, config, "aiter.other", field)
    here = capture.graph_key(other.values["structure"], "decode B=2")
    named = (
        f"differs in {field} {other.values[field]!r} in the graph and "
        f"{here.values[field]!r} here"
    )

    with pytest.raises(ArtifactRefusal) as bench:
        ops.graph(store, config, decode(2), ARCH)
    assert named in str(bench.value)
    assert named in compass_run._op_graphs(str(store.root), config, ARCH)(
        "decode", 2, 2
    )


def test_each_reader_reads_a_graph_traced_under_its_own_readings(
    monkeypatch, tmp_path, config
):
    store = ArtifactStore(tmp_path / "store")
    published(monkeypatch, store, config, "aiter.held")
    run, lazy = lookup(monkeypatch, store, config, "aiter.here")

    assert run("decode", 2, 2)["op"] == "aiter.held"
    assert lazy["lazy_traces"] == 0
    assert ops.graph(store, config, decode(2), ARCH)["region"]["op"] == "aiter.held"
