# SPDX-License-Identifier: MIT
"""A DP prefill's graph holds G, the rows its MoE gathers, as one symbol: a
run's lookup traces each (T, B) once, at both ends of G's range, and puts each
step's G into that graph, byte for byte the graph a trace at that G gives
(`forward.Tracer.symbolic`, `peers.at`). Where the two ends differ but for
each exchange's `sizes`, every G of the (T, B) is traced as it is, named in
the run summary; a G the graph's hints or guards do not admit is traced too.
The run summary counts the lookups a graph holding G served, and the wall
every lookup took.
"""

import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_capture_v4_pro import DPA_TP8, EP8, MORI, trace
from test_op_graph import _counts, _run_config, _spawned, in_thread
from test_op_graph_arch import ENGINE

from atom.compass import ir
from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.capture import forward, peers
from atom.compass.capture.graph import Traced, to_json
from atom.compass.run import _op_graphs
from atom.compass.runner.overrides import PricedSteps

DP, T = 8, 64
#: A prefill's gathered rows: off their default below and above it, and at it.
GATHERED = (100, DP * T, 3000)
#: The rows a sampling prefill's LM head pads its one sequence to.
WIDE = 2
#: The top of G's range at T: each peer at ATOM's default batch of 16384.
TOP = T + (DP - 1) * 16384
HELD = f"prefill T={T} B=1 G=G"


def shape(g: int, width: int | None = None) -> str:
    """The shape a prefill of T tokens over one sequence gathering `g` rows,
    its LM head padded to `width`, is published at."""
    return (
        f"prefill T={T} B=1" + f" W={width}" * bool(width) + f" G={g}" * (g != DP * T)
    )


def _record(root):
    """A record directory of six steps: rank 0 prefills T tokens, its MoE
    gathering each of `GATHERED` in turn, then each again with its LM head
    padded to `WIDE`, its peers idle."""
    (root / "steps").mkdir(parents=True)
    (root / "run.json").write_text("{}")
    steps = [(g, width) for width in (None, WIDE) for g in GATHERED]
    for rank in range(DP):
        lines = [
            {"step": n, "is_dummy_run": rank > 0, "rung": None, "moe_rows": g}
            | {"rows": [[T, T, False]] if rank == 0 else [[1, 1, False]]}
            | {"target_bs": width}
            for n, (g, width) in enumerate(steps, 1)
        ]
        text = "".join(json.dumps(line) + "\n" for line in lines)
        (root / "steps" / f"dp{rank}.jsonl").write_text(text)
    return root


def _published(root, done) -> dict:
    """Each graph `trace ops` published in `root`'s store, by its shape and
    whether its step produces output."""
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, done.stderr[-4000:]
    store, rank, graphs = ArtifactStore(str(root / "store")), Topology().ranks()[0], {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        graphs[graph["shape"], not graph["structure"].endswith("no output")] = graph
    return graphs


@pytest.mark.parametrize("ep", [False, True], ids=["gather", "mori"])
def test_a_dp_prefill_s_graph_holds_g_and_each_step_s_g_puts_it_byte_for_byte(
    ep, tmp_path_factory, monkeypatch
):
    root, ahead = tmp_path_factory.mktemp("traced"), tmp_path_factory.mktemp("ahead")
    deployed = f"{DPA_TP8}{EP8 * ep}"
    population = f"--population {_record(root / 'record')}{deployed}"
    traced = _published(root, trace(root, population))
    # Traced ahead: (T, B) at its default G, then with G a symbol.
    held = _published(ahead, trace(ahead, f"--prefill {T} 1 --symbolic{deployed}"))
    chunks = (False, True)
    assert held.keys() == {(s, o) for s in (shape(DP * T), HELD) for o in chunks}
    config = _run_config(held[HELD, True]["structure"], root / "model")
    args = ["--model", config.model, "-tp", "8", *deployed.split()[1:]]
    spawned, lazy = _spawned(monkeypatch), _counts()
    store = tmp_path_factory.mktemp("store")
    lookups = (
        _op_graphs(str(store), config, "gfx950", args, lazy),
        _op_graphs(str(ahead / "store"), config, "gfx950"),
    )

    for g in GATHERED:
        for output in (False, True):
            want = json.dumps(traced[shape(g), output]["region"])
            assert [json.dumps(at("prefill", T, 1, output, g)) for at in lookups] == [
                want
            ] * 2, g
    # One lazy trace, both ends of G's range, answered all six.
    assert (len(spawned), lazy["lazy_traces"]) == (1, 1)
    assert "lazy_per_g" not in lazy

    # A prefill whose LM head pads holds G too, keyed by its width: one more.
    lookup = lookups[0]
    for g in GATHERED:
        want = json.dumps(traced[shape(g, WIDE), True]["region"])
        assert json.dumps(lookup("prefill", T, 1, True, g, False, WIDE)) == want, g
    assert lazy["lazy_traces"] == 2 and "lazy_per_g" not in lazy

    # Past the range's top, where MORI's trim guards G too, the step is traced.
    exchange = f"{MORI}.dispatch" if ep else "atom.model_ops.moe:all_gatherv"
    past = lookup("prefill", T, 1, True, TOP + 1)
    assert lazy["lazy_traces"] == 3
    assert {op["out"][0][0] for op in _ops(past) if op["op"] == exchange} == {TOP + 1}


def _ops(region):
    """Every op of `region`, each `Repeat` body once."""
    if "op" in region:
        return [region]
    inner = region.get("seq") or region.get("par") or [region.get("repeat")]
    return [op for item in inner for op in _ops(item)]


class _Tracer:
    """A tracer whose DP prefill is one gather and one expert leaf at the rows
    it gathers, G where traced `symbolic` with `guards` installed on it; with
    `apart`, the expert also takes the gathered rows' count as ATOM's numpy
    holds it, the hint and not G. Its symbolic step is `Tracer.symbolic`'s."""

    symbolic = forward.Tracer.symbolic
    runner, drafts = "built", 0

    def __init__(self, config, guards=(), apart=False):
        self.config = SimpleNamespace(
            parallel_config=config.parallel_config, max_num_batched_tokens=16384
        )
        self.at, self.guards, self.apart, self.traced = config, guards, apart, []

    def trace(self, shapes, unsplit, symbolic=False, warm=True):
        (shape,) = shapes
        self.traced.append((shape.gathered, symbolic))
        g = shape.gathered or DP * shape.tokens
        rows = ir.G if symbolic else g
        x = ir.Tensor((shape.tokens, 8), "torch.bfloat16", (8, 1), "activation")
        sizes = tuple(peers.split(g, shape.tokens, DP))
        gather = ir.Op(
            peers.SIZED[0], "declared", (("x", x), ("sizes", sizes)), ((rows, 8),)
        )
        held = ir.Tensor((rows, 8), "torch.bfloat16", (8, 1), "activation")
        count = (("n", g),) if self.apart else ()
        expert = ir.Op(
            "aiter.fused_moe_", "opaque_leaf", (("x", held), *count), ((rows, 8),)
        )
        for output in (False, True):
            at = forward.structure(self.at, "prefill", True, "gfx950", output, False)
            text = forward.symbolic_shape(shape) if symbolic else str(shape)
            traced = Traced(at, text, self.guards if symbolic else ())
            yield ir.Graph(traced, ir.Seq((gather, expert))), self.at.model, {}


@pytest.fixture
def lookup(monkeypatch, tmp_path):
    """The run's lookup of a DPA-TP8 config through a dirty tree, whose lazy
    traces answer it unpublished, by a `_Tracer` built from the fixture's
    parameters, and its counts."""
    from test_artifact_store import AITER_ROOT, ATOM_ROOT

    from atom.compass import artifacts

    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    config = SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=DP),
        enable_expert_parallel=False,
        enforce_eager=True,
        **ENGINE,
    )
    dirty = (replace(ATOM_ROOT, dirty=1), AITER_ROOT)
    monkeypatch.setattr(artifacts, "roots_for", lambda modules: dirty)
    in_thread(monkeypatch)

    def made(**kwargs):
        tracer, lazy = _Tracer(config, **kwargs), _counts()
        monkeypatch.setattr(compass_run, "_tracer", lambda *args: (None, tracer))
        args = ["--model", config.model]
        return (
            _op_graphs(str(tmp_path / "store"), config, "gfx950", args, lazy),
            lazy,
            tracer,
        )

    return made


def _concrete(tracer, g, output=True):
    """The region `tracer` traces at `g`, as a lookup returns it."""
    shape = forward.prefill(T, 1, g).at(DP)
    graph, *_ = list(tracer.trace([shape], [shape]))[output]
    return json.loads(to_json(graph))["region"]


def test_a_g_its_guards_refuse_is_traced_and_one_they_admit_is_put(lookup):
    graphs, lazy, tracer = lookup(guards=("G <= 600",))

    admitted, refused = (graphs("prefill", T, 1, True, g) for g in (100, 3000))

    # Both ends traced once with G a symbol, then G=3000 as it is.
    assert tracer.traced == [(TOP, True), (T + 1, True), (3000, False)]
    assert admitted == _concrete(tracer, 100) and refused == _concrete(tracer, 3000)
    assert lazy["lazy_traces"] == 2 and "lazy_per_g" not in lazy


def test_ends_that_differ_past_sizes_trace_each_g_and_name_the_shape(lookup):
    graphs, lazy, tracer = lookup(apart=True)

    final = {g: graphs("prefill", T, 1, True, g) for g in (100, 3000)}

    assert tracer.traced == [(TOP, True), (T + 1, True), (100, False), (3000, False)]
    assert final == {g: _concrete(tracer, g) for g in (100, 3000)}
    # The lower end's count against the upper end's put there.
    why = f"its middle chunk at G={T + 1}: aiter.fused_moe_ at args/n: {T + 1}"
    assert list(lazy["lazy_per_g"]) == [f"{HELD}, {why} against {TOP}"]
    assert lazy["lazy_traces"] == 3


def test_the_run_summary_names_each_prefill_whose_g_were_each_traced(
    monkeypatch, tmp_path
):
    run = {"admission_s": 0.54e-3, "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(tmp_path))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    authority = compass_run._RecordingAuthority(run)
    why = f"{HELD}, its middle chunk at G=65: aiter.fused_moe_ at args/n: 65 against {TOP}"
    worker = SimpleNamespace(compass_lazy={**_counts(), "lazy_per_g": {why: 1}})
    compass_run._hand_in(run, "engine", [], priced=PricedSteps.priced_steps(worker))
    compass_run._hand_in(run, "frontend", [])
    compass_run._write_summary(run, authority, ended=True)

    cost = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())["cost"]
    assert cost["lazy_per_g"] == {why: 1}


def test_the_run_summary_counts_the_lookups_a_held_graph_served(
    lookup, monkeypatch, tmp_path
):
    graphs, lazy, tracer = lookup(guards=("G <= 600",))

    # G=100 and 200 are put into the graph that holds G; G=3000 is traced.
    for g in (100, 200, 3000):
        graphs("prefill", T, 1, True, g)

    assert tracer.traced == [(TOP, True), (T + 1, True), (3000, False)]
    assert lazy["held_lookups"] == 2
    assert lazy["lookup_wall_seconds"] >= lazy["lazy_trace_wall_seconds"] > 0
    run = {"admission_s": 0.54e-3, "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(tmp_path))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    authority = compass_run._RecordingAuthority(run)
    worker = SimpleNamespace(compass_lazy=lazy)
    compass_run._hand_in(run, "engine", [], priced=PricedSteps.priced_steps(worker))
    compass_run._hand_in(run, "frontend", [])
    compass_run._write_summary(run, authority, ended=True)

    cost = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())["cost"]
    assert cost["held_lookups"] == 2
    assert cost["lookup_wall_seconds"] == lazy["lookup_wall_seconds"]


def test_symbolic_is_refused_without_dp_before_any_trace(tmp_path):
    done = trace(tmp_path, f"--prefill {T} 1 --symbolic -- --kv_cache_dtype fp8")

    assert done.returncode != 0 and "only a DP prefill" in done.stderr
    assert "published" not in done.stdout
