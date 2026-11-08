# SPDX-License-Identifier: MIT
"""Grading an op-graph deployment whose two sides' servers ran under different
performance variables: each side's real steps are priced by the graphs keyed
under the variables its engine file states, whatever the grading process holds.

The cell is `test_validate_main`'s deployment with its simulated run made
`compass run --op-graph`'s, priced by `run.op_graph` over a store of graphs
published as `compass trace ops` keys them in a process holding one side's
variables. No device: the run file's machine spec states the arch.
"""

import json
from types import SimpleNamespace

from test_artifact_store import (
    CONDITIONS,
    GATES,
    clean_checkout,  # noqa: F401  autouse
    stanza,
)
from test_memory_readings import DOCUMENT
from test_op_graph import g as gemm_leaf
from test_op_graph_arch import ENGINE, RANK
from test_validate_main import _as_recorded, _deploy, _graded, _rows
from test_work import CURVE, FLOOR

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.artifacts import ArtifactStore, Key, Kind, Topology, member_name
from atom.compass.backends.work import gemm, price
from atom.compass.capture.forward import decode, prefill, structure
from atom.compass.drive import run as drive_run
from atom.compass.run import OP_GRAPH
from atom.compass.run.step_grid import BUCKETS
from atom.compass.spec import MachineSpec

#: Each side's performance variables, the prefill server's two more than the decode's.
VARIABLES = {
    "prefill": {
        "ATOM_MOE_GU_ITLV": "1",
        "ATOM_NUMA_BIND": "1",
        "GPU_MAX_HW_QUEUES": "5",
    },
    "decode": {"ATOM_MOE_GU_ITLV": "1"},
}
#: Each step shape of the cell's records: its warmup's, its prefill chunks' and
#: its decode rows' at rung 1.
SHAPES = [prefill(t, 1) for t in (1, *range(100, 1000, 90))] + [decode(1)]
#: The one shape no graph under the decode side's variables holds.
UNTRACED = prefill(190, 1)
#: The run file's machine spec, with what a GEMM's analytical price reads.
MACHINE = json.loads(json.dumps(DOCUMENT))
MACHINE["device"]["kernel_floor_s"] = FLOOR
MACHINE["device"]["memory"]["bandwidth_curve"] = CURVE
#: Every graph's region: a GEMM repeated to about 30 ms on MACHINE, a step of
#: the cell's records within the factor past which a step is left out as warm.
GEMM_S = price(gemm(4096, 4096, 4096), MachineSpec.from_mapping(MACHINE))
REGION = {
    "repeat": gemm_leaf(4096, 4096, 4096),
    "count": int(0.03 / GEMM_S) + 1,
    "index": "l0",
    "start": 0,
    "step": 1,
}


def _variables(monkeypatch, stated: dict) -> None:
    """This process's performance variables: `stated` and no other."""
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)
    for name, value in stated.items():
        monkeypatch.setenv(name, value)


def _traced(monkeypatch, store, config, stated: dict, shapes) -> None:
    """A graph of `REGION` at each of `shapes`, published at the key `compass
    trace ops` gives it in a process whose performance variables are `stated`."""
    graph = json.dumps({"region": REGION}).encode()
    with monkeypatch.context() as m:
        _variables(m, stated)
        for shape in shapes:
            at = structure(config, shape.phase, True, MACHINE["device"]["arch"])
            store.publish(
                Key.of(Kind.OP_GRAPH, structure=at, shape=str(shape)),
                provenance=stanza("compass trace ops"),
                topology=Topology(),
                conditions=CONDITIONS,
                gates=GATES,
                members={member_name("graph", RANK, "json"): graph},
            )


def test_each_side_reads_the_graphs_traced_under_the_variables_its_engine_file_states(
    capsys, tmp_path, monkeypatch
):
    cell = tmp_path / "c4"
    _rows(capsys, tmp_path)
    _as_recorded(cell)
    _deploy(cell)
    sim = (cell / "step_linear").rename(cell / OP_GRAPH)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    config = SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        **ENGINE,
    )
    store = ArtifactStore(str(tmp_path / "store"))
    run_file = {
        OP_GRAPH: True,
        "store": str(store.root),
        "machine": MACHINE,
        "host": {"laws": {}},
    }
    for side, stated in VARIABLES.items():
        shapes = [s for s in SHAPES if side == "prefill" or s != UNTRACED]
        _traced(monkeypatch, store, config, stated, shapes)
        # Each run's side, real or simulated, ran under its side's variables.
        for engine in cell.glob(f"*/{side}/steps/dp0.engine.json"):
            engine.write_text(
                json.dumps(json.loads(engine.read_text()) | {"env": stated})
            )
        run = json.loads((sim / side / "run.json").read_text())
        run["argv"] = ["-c", "run", "--out", str(sim / side), "--", "--model", "m"]
        (sim / side / "run.json").write_text(json.dumps(run | {"run_file": run_file}))
    deployed = json.loads((sim / "run.json").read_text())
    (sim / "run.json").write_text(json.dumps(deployed | {"run_file": run_file}))
    monkeypatch.setattr(drive_run, "_config", lambda args: config)
    # The grading process holds neither side's variables.
    _variables(monkeypatch, {})

    rows = _graded(capsys, cell)

    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    for real in ("real0", "real1", "real2"):
        refused = {
            side: {
                metric: value
                for (result, metric), (value, _) in rows.items()
                if result == f"cost {real} {side}" and metric.startswith("refused")
            }
            for side in VARIABLES
        }
        assert refused["prefill"] == {}
        # Only the decode side's variables lack a graph, and the store's one graph
        # of that shape is the one traced under the prefill side's.
        (value,) = refused["decode"].values()
        assert set(refused["decode"]) == {"refused prefill tokens 0-512"}
        assert value.startswith("1 steps")
        assert "a prefill step of 190 tokens over 1 sequences" in value
        for name in ("ATOM_NUMA_BIND", "GPU_MAX_HW_QUEUES"):
            stated = VARIABLES["prefill"][name]
            assert f"env {name} '{stated}' in the graph and None here" in value
        # Each side priced its other steps; the spec is assumed, so its prices declared.
        for side in VARIABLES:
            for phase in ("prefill", "decode"):
                value, _ = rows[f"cost {real} {side}", f"{phase} declared"]
                assert int(value.split()[0]) > 0
