# SPDX-License-Identifier: MIT
"""The machine spec a command composes from `--device`'s datasheet, the
`platform_readings` entry the store holds for it and, across nodes, a network
file: a folded entry's rates over the datasheet's peaks, an assumed entry's
ratios as they stand; the run file's `machine` block of each committed one-node
spec, from that spec's readings, equal to the spec; the stack the run's entries
state picking the readings, and a network file read only across nodes; and a
fit of step measurements alone keyed by the platform their keys state."""

import argparse
import dataclasses
import json

import platforms
import pytest
from test_fit_device import (  # noqa: F401  the nodes fixture
    BF16,
    CURVE,
    LINK,
    fit,
    nodes,
)
from test_fit_samples import RUNS, measured, record
from test_measure_device import (  # noqa: F401  the node fixture
    PLATFORM,
    STACK,
    node,
)
from test_memory_readings import DOCUMENT
from test_probe_device import SCRIPTS

from atom.compass.artifacts import ArtifactStore
from atom.compass.artifacts.keys import NO_PTL
from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import runfile
from atom.compass.fit import publish
from atom.compass.fit.device import compose
from atom.compass.fit.samples import load
from atom.compass.spec import MachineSpec, SpecRefusal
from atom.compass.spec.layers import datasheet

#: Each committed spec of one node's readings.
ONE_NODE = sorted(p.stem for p in (SCRIPTS / "machines").glob("*.json"))
#: What leaves the spec: the document's header and the scale-out network, which
#: only a run across nodes reads.
MOVED = ("provenance.", "name", "interconnect.inter_node.")
AT_A = {
    "software_stack": "rocm 7.2.4 / aiter f4e7c7509 / rccl 2.22.3",
    "torch": "2.10.0",
}
AT_B = {
    "software_stack": "rocm 7.2.4 / aiter 9205e1059 / rccl 2.22.3",
    "torch": "2.10.0",
}
B = "rocm 7.2.4 / aiter 9205e1059 / rccl 2.22.3 / torch 2.10.0"


def parsed(tmp_path, *flags):
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    return parser.parse_args([*flags, "--store", str(tmp_path / "store")])


def flat(value):
    """A spec value as pytest.approx takes it: a curve's points in one list."""
    if isinstance(value, (list, tuple)):
        return [x for point in value for x in point]
    return value


def model(tmp_path):
    return ["--model", str(tmp_path)]


def stub(tmp_path) -> list[str]:
    (tmp_path / "stub.json").write_text(json.dumps(dataclasses.asdict(Coefficients())))
    return ["--step-stub", str(tmp_path / "stub.json")]


def law(tmp_path, name: str, at: dict, host=False) -> str:
    """A law file whose step law, or with `host` its host laws, `at` keys."""
    platform = {"device": "MI355X gfx950", "ptl": "disabled", **at}
    body = {"host": {"key": platform}} if host else {"platform": platform}
    (tmp_path / name).write_text(json.dumps(body))
    return str(tmp_path / name)


def test_a_folded_entry_composes_each_rate_over_its_datasheet_peak(
    nodes, tmp_path  # noqa: F811
):
    token, _ = nodes("crs005", "0,1")  # no MXFP4 GEMM runs there
    entry = fit(tmp_path, token)

    machine = compose("mi355x", ArtifactStore(tmp_path / "store"), "disabled")

    sheet = datasheet("mi355x").values
    bf16 = sheet["device.compute.bf16_flops"]
    assert machine.value("device.compute.derate.bf16") == entry[BF16] / bf16
    band = machine.origin("device.compute.derate.bf16")
    assert (band["p10"], band["p90"]) == tuple(
        entry["provenance"]["origins"][BF16][end] / bf16 for end in ("p10", "p90")
    )
    link = entry[LINK] / sheet["interconnect.intra_node.link_bandwidth_bytes_per_s"]
    assert machine.value("interconnect.intra_node.derate") == link
    bandwidth = sheet["device.memory.bandwidth_bytes_per_s"]
    assert machine.value("device.memory.derate") == entry[CURVE][-1][1] / bandwidth
    with pytest.raises(SpecRefusal, match="fp4_flops"):
        machine.value("device.compute.fp4_flops")
    pinned = {c: machine.value(f"device.software_pinned_to.{c}") for c in STACK}
    assert pinned == STACK and machine.value("device.ptl") == PLATFORM["ptl"]
    assert not [p for p in machine.values if p.startswith(MOVED[2:])]


def test_an_assumed_entry_composes_its_ratios_on_its_own_datasheet(
    nodes, tmp_path  # noqa: F811
):
    token, _ = nodes("mi355-gpu-40", "0,1")
    source = fit(tmp_path, token)
    fit(
        tmp_path,
        "--device",
        "mi455x-b0",
        "--assume-from",
        "MI355X",
        "--ptl",
        "disabled",
    )

    machine = compose("mi455x-b0", ArtifactStore(tmp_path / "store"), "disabled")

    mi355x, b0 = datasheet("mi355x").values, datasheet("mi455x-b0").values
    ratio = source[BF16] / mi355x["device.compute.bf16_flops"]
    assert machine.value("device.compute.derate.bf16") == ratio
    assert machine.value("device.compute.bf16_flops") == b0["device.compute.bf16_flops"]
    bandwidth = "device.memory.bandwidth_bytes_per_s"
    curve = [r / mi355x[bandwidth] * b0[bandwidth] for _, r in source[CURVE]]
    assert [r for _, r in machine.value(CURVE)] == pytest.approx(curve)
    assert machine.origin("device.compute.derate.bf16") == {"origin": "assumed"}


@pytest.mark.parametrize("name", ONE_NODE)
def test_a_run_file_s_machine_block_equals_its_one_node_spec(tmp_path, name):
    document = json.loads((SCRIPTS / "machines" / f"{name}.json").read_text())
    flags = platforms.flags(tmp_path / "store", document, tmp_path)

    run = runfile.build(parsed(tmp_path, *stub(tmp_path), *flags), model(tmp_path))

    got = MachineSpec.from_mapping(run["machine"]).values
    want = MachineSpec.from_mapping(document).values
    want = {"device.ptl": NO_PTL} | {p: v for p, v in want.items()}
    assert not [p for p in got if p.startswith(MOVED[2:])]
    kept = [p for p in want if not p.startswith(MOVED)]
    got = {p: v for p, v in got.items() if not p.startswith(MOVED)}
    assert sorted(got) == sorted(kept)
    for path in kept:
        assert flat(got[path]) == pytest.approx(flat(want[path]), rel=1e-15), path


def test_the_run_reads_the_readings_at_the_stack_its_entries_state(tmp_path):
    store = tmp_path / "store"
    flags = platforms.flags(store, DOCUMENT, tmp_path)
    platforms.flags(store, DOCUMENT, tmp_path, at=AT_B)
    stated = law(tmp_path, "b.json", AT_B)

    run = runfile.build(
        parsed(tmp_path, "--step-linear", stated, *flags), model(tmp_path)
    )
    assert run["machine"]["device"]["software_pinned_to"]["aiter"] == "9205e1059"

    with pytest.raises(ValueError) as refused:
        runfile.build(parsed(tmp_path, *stub(tmp_path), *flags), model(tmp_path))
    assert "holds 2 platform_readings entries for MI355X gfx950 at PTL disabled" in (
        str(refused.value)
    )
    assert f"; {B} as " in str(refused.value) and "--stack names one" in str(
        refused.value
    )
    picked = parsed(tmp_path, *stub(tmp_path), *flags, "--stack", B)
    run = runfile.build(picked, model(tmp_path))
    assert run["machine"]["device"]["software_pinned_to"]["aiter"] == "9205e1059"


def test_entries_that_state_two_stacks_refuse_naming_each(tmp_path):
    flags = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    step, host = law(tmp_path, "a.json", AT_A), law(tmp_path, "b.json", AT_B, True)

    with pytest.raises(ValueError) as refused:
        runfile.spec(
            parsed(tmp_path, "--step-linear", step, "--host-laws", host, *flags),
            ArtifactStore(tmp_path / "store"),
        )
    assert str(refused.value) == (
        "the entries the run prices with state 2 stacks, and one run reads one "
        f"platform: {step}: rocm 7.2.4 / aiter f4e7c7509 / rccl 2.22.3 / torch "
        f"2.10.0; {host}: {B}"
    )
    with pytest.raises(ValueError, match=f"--stack: {B}"):
        runfile.spec(
            parsed(tmp_path, "--step-linear", step, "--stack", B, *flags),
            ArtifactStore(tmp_path / "store"),
        )


def test_only_a_run_across_nodes_reads_a_network_file(tmp_path):
    flags = [*stub(tmp_path), *platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)]
    net = platforms.network(DOCUMENT, tmp_path)
    one_node = runfile.build(parsed(tmp_path, *flags), model(tmp_path))
    assert "inter_node" not in one_node["machine"]["interconnect"]

    with pytest.raises(ValueError, match="a run on one node reads no network"):
        runfile.build(parsed(tmp_path, *flags, *net), model(tmp_path))
    pd = model(tmp_path), model(tmp_path)
    with pytest.raises(ValueError, match="reads a network file and names none"):
        runfile.prefill_decode(parsed(tmp_path, *flags), *pd, "inter_node")

    run = runfile.prefill_decode(parsed(tmp_path, *flags, *net), *pd, "inter_node")
    inter = run["machine"]["interconnect"]["inter_node"]
    assert inter == DOCUMENT["interconnect"]["inter_node"]


def test_step_measurements_fit_on_their_keys_platform_and_two_refuse(
    tmp_path, monkeypatch
):
    config = load({record(tmp_path, monkeypatch): RUNS})[1]
    store = tmp_path / "store"
    one = measured(store, config)
    two = measured(store, config | {"stack": {**config["stack"], "aiter": "9205e1059"}})

    assert publish._keyed([one]) == publish.platform(config)
    with pytest.raises(ValueError, match="the step measurements ran on several"):
        publish._keyed([one, two])
