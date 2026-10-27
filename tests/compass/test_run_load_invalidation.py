# SPDX-License-Identifier: MIT
"""A run loads each `step_linear_law` and `op_graph` it reads through the
invalidation matrix, and each refuses on its row's axes, naming the axis.

A law is loaded as `drive.runfile` builds the run file, before a server starts,
under the readings a run on the suite's machine document takes, with the
engine config its key states. A graph is loaded as `run._op_graphs` looks it
up, under this process's `graph_readings` and the engine config it records,
the tracer's own arguments, which its key's structure names.
"""

import json
import re
from types import SimpleNamespace

import pytest
from test_artifact_invalidation import STANZA, moved
from test_backend_step_linear import law
from test_drive_runfile import LAWS, build, host_laws, install
from test_op_graph_arch import ENGINE, run_readings
from test_run_step_grid import NEW_KINDS, deployment, readings

from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    GateState,
    Key,
    Kind,
    Row,
    Topology,
    axes_of,
    member_name,
)
from atom.compass.backends import StepLinearBackend
from atom.compass.capture.forward import structure
from atom.compass.fit.publish import RANK
from atom.compass.run import _op_graphs
from atom.compass.run.step_grid import keyed

ARCH = "gfx942"
REGION = {"op": "aiter.leaf"}


def fitted(tmp_path, r, conditions):
    """A step linear law of `r`'s engine config, handed off as `fit` hands one
    off, under `conditions`."""
    engine = json.dumps(keyed(r.config), sort_keys=True)
    key = Key.of(
        Kind.STEP_LINEAR_LAW,
        **{**NEW_KINDS[Kind.STEP_LINEAR_LAW], "engine_config": engine},
    )
    return ArtifactStore(tmp_path / "store").publish(
        key,
        provenance=STANZA,
        topology=Topology(),
        conditions=conditions,
        gates=GateState.of(),
        members={member_name("params", RANK, "json"): json.dumps(law()).encode()},
    )


@pytest.mark.parametrize("axis", axes_of(Row.STEP_LINEAR_LAW), ids=str)
def test_a_law_fitted_under_other_readings_refuses_its_run_file_naming_the_axis(
    tmp_path, axis
):
    r = deployment(tmp_path)
    token = fitted(tmp_path, r, moved(readings(r), axis)).key.token()

    named = re.escape(f"`{Row.STEP_LINEAR_LAW}` x `{axis}`")
    with pytest.raises(ArtifactRefusal, match=named):
        build(tmp_path, "--step-linear", token)


def test_a_law_fitted_on_another_host_cpu_builds_its_run_file(tmp_path):
    r = deployment(tmp_path)
    entry = fitted(tmp_path, r, moved(readings(r), Axis.HOST_CPU))

    run = build(tmp_path, "--step-linear", entry.key.token())

    assert run["backend"]["digest"] == entry.digest


def test_a_law_and_host_laws_fitted_under_the_run_s_readings_install(
    monkeypatch, tmp_path
):
    r = deployment(tmp_path)
    law_token = fitted(tmp_path, r, readings(r)).key.token()
    host_token = host_laws(tmp_path, r).key.token()

    run = build(tmp_path, "--step-linear", law_token, "--host-laws", host_token)
    install(tmp_path, monkeypatch, run, r)

    assert isinstance(r.compass_backend, StepLinearBackend)
    assert r.compass_host.laws == LAWS


def config(tmp_path):
    """A stand-in config of a model whose config.json is `{}`."""
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


def traced(tmp_path, c, conditions):
    """A run's lookup over a store holding `c`'s `decode B=2` graph, keyed as a
    capture keys it and handed off under `conditions`."""
    store = ArtifactStore(tmp_path / "store")
    at = structure(c, "decode", True, ARCH)
    store.publish(
        Key.of(Kind.OP_GRAPH, structure=at, shape="decode B=2"),
        provenance=STANZA,
        topology=Topology(),
        conditions=conditions,
        gates=GateState.of(),
        members={
            member_name("graph", RANK, "json"): json.dumps({"region": REGION}).encode()
        },
    )
    return _op_graphs(str(store.root), c, ARCH)


#: The axes of the op_graph row a run reads; the key's structure names the
#: engine config.
READ = [a for a in axes_of(Row.OP_GRAPH) if a is not Axis.ENGINE_CONFIG]


@pytest.mark.parametrize("axis", READ, ids=str)
def test_a_graph_traced_under_other_readings_refuses_its_run_naming_the_axis(
    tmp_path, axis
):
    c = config(tmp_path)
    lookup = traced(tmp_path, c, moved(run_readings(c.model), axis))

    named = re.escape(f"`{Row.OP_GRAPH}` x `{axis}`")
    with pytest.raises(ArtifactRefusal, match=named):
        lookup("decode", 2, 2)


def test_a_graph_traced_in_this_process_loads_whatever_arguments_traced_it(tmp_path):
    c = config(tmp_path)
    here = run_readings(c.model)

    assert traced(tmp_path, c, here)("decode", 2, 2) == REGION
