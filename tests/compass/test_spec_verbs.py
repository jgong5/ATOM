# SPDX-License-Identifier: MIT
"""The verbs over the spec: combining fragments, checking one, reading one.

The fragments below are the ones the probes of each hardware probe class would emit --
a single-card engine start and a fabric measurement -- so a test that combines them combines
the same partial documents a real run would, and a test that breaks one breaks
exactly what it says it does.

The numbers are the measured ones wherever a measurement exists, because two of
these checks are judgements about size: the cross-rank spread accepted is the
one the hardware really shows, and the spread refused is the one a neighbour
really produced. Round numbers would make both tests vacuous.
"""

import copy
import importlib.util
import json
import pathlib
import re

import pytest

from atom.compass import spec as spec_package
from atom.compass.spec import (
    Fragment,
    MachineSpec,
    SpecRefusal,
    across_ranks,
    explain,
    merge,
    validate,
)
from atom.compass.spec.fields import BY_PATH
from atom.compass.spec.rules import Rule

MACHINE = "mi355x-8gpu-2node"
STACK = {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.22.3", "torch": "2.10.0"}
MIB = 1024**2

#: Probe class 1, one card: the device, its capacity, rates and stack.
CLASS1 = {
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {
            "capacity_bytes": 288.0e9,
            "bandwidth_bytes_per_s": 8.0e12,
            "derate": 0.85,
        },
        "compute": {
            "bf16_flops": 2.5e15,
            "fp8_flops": 5.0e15,
            "derate": {"bf16": 0.70, "fp8": 0.70},
        },
        "software_pinned_to": dict(STACK),
    }
}

#: Probe classes 2 and 3: collectives in real groups, then the same across two nodes.
LINKS = {
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
            "domain_devices": 8,
            "link_bandwidth_bytes_per_s": 1.0e12,
            "link_latency_s": 2.0e-6,
            "derate": 0.80,
        },
        "inter_node": {
            "link_bandwidth_bytes_per_s": 5.0e10,
            "link_latency_s": 5.0e-6,
            "derate": 0.80,
        },
    }
}


def fragment(
    source,
    body=None,
    *,
    machine=MACHINE,
    origin="probed",
    authored_by="a person",
    date="2026-09-18",
    **stanza,
):
    """One probe's output: what it measured, and where it was measured."""
    document = {
        "schema_version": 1,
        "name": machine,
        "provenance": dict(
            {"authored_by": authored_by, "date": date, "origin": origin}, **stanza
        ),
    }
    document.update(copy.deepcopy(body or {}))
    return Fragment.from_mapping(document, source)


def fragments(**edits):
    """The fragments a full spec is built from, with one body replaced."""
    bodies = {"class1": CLASS1, "links": LINKS}
    bodies.update(edits)
    return [fragment(source, body) for source, body in bodies.items()]


def merged(**edits):
    return merge(fragments(**edits))


def resolved(**edits):
    return MachineSpec.from_mapping(merged(**edits).document)


def without(body, *path):
    """A copy of a fragment body with one dotted field taken out of it."""
    fresh = copy.deepcopy(body)
    node = fresh
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return fresh


# --- merge: the conflict check, which is the point of the verb ----------------


def test_two_hosts_in_one_spec_are_refused_and_both_provenances_are_named():
    # A device and a fabric measured on two machines contradict nothing in
    # their shape: no field of one overlaps a field of the other. Only the
    # stanzas differ, so only the stanzas can catch it.
    here = fragment("device-a", CLASS1, machine="node-18", authored_by="ana")
    elsewhere = fragment(
        "links-b", LINKS, machine="node-22", authored_by="bo", date="2026-09-19"
    )
    with pytest.raises(SpecRefusal) as refused:
        merge([here, elsewhere])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    for named in ("device-a", "links-b", "node-18", "node-22", "ana", "bo"):
        assert named in message, f"{named} is not named in: {message}"


def test_the_refusal_says_the_fragments_are_authored_for_different_machines():
    # `Fragment.machine` is the `name` field, which is the machine the spec is
    # being authored for. No field records where a probe ran, so the refusal
    # says what the fragments claim and not where their numbers came from.
    with pytest.raises(SpecRefusal) as refused:
        merge(
            [fragment("a", CLASS1, machine="node-18"), fragment("b", machine="node-22")]
        )
    message = str(refused.value)
    assert refused.value.what.endswith(
        "are authored for different machines, 'node-18' and 'node-22'"
    )
    assert "measured on" not in message


def test_the_same_two_fragments_merge_when_the_machine_agrees():
    # The control for
    # `test_two_hosts_in_one_spec_are_refused_and_both_provenances_are_named`:
    # the same two measurements, the same two authors and dates, differing only
    # in the machine they name.
    here = fragment("device-a", CLASS1, machine="node-18", authored_by="ana")
    also_here = fragment(
        "links-b", LINKS, machine="node-18", authored_by="bo", date="2026-09-19"
    )
    combination = merge([here, also_here])
    assert combination.document["interconnect"] == LINKS["interconnect"]
    assert combination.document["name"] == "node-18"
    assert combination.document["provenance"]["fragments"] == ["device-a", "links-b"]


def test_a_merged_document_merged_again_keeps_the_names_that_built_it():
    # Incremental authoring: merge what has been measured, save it, merge next
    # week's probe into the saved document. A merged document is a legal
    # fragment, so this works -- and the block whose whole job is to say where
    # the numbers came from must not lose the fragments of the first round.
    first = merge([fragment("class1", CLASS1), fragment("links", LINKS)])
    again = merge(
        [
            Fragment.from_mapping(first.document, "machine.yaml"),
            fragment("ptl", {"device": {"ptl": "disabled"}}),
        ]
    )
    assert again.document["provenance"]["fragments"] == [
        "class1",
        "links",
        "machine.yaml",
        "ptl",
    ]


def test_one_machine_named_twice_with_two_answers_is_refused():
    smaller = copy.deepcopy(CLASS1)
    smaller["device"]["memory"]["capacity_bytes"] = 256.0e9
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", CLASS1), fragment("second", smaller)])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "device.memory.capacity_bytes" in message
    assert "288000000000.0" in message and "256000000000.0" in message
    assert "first" in message and "second" in message


def test_two_fragments_that_agree_are_not_a_conflict():
    combination = merge([fragment("first", CLASS1), fragment("second", CLASS1)])
    assert combination.sources["device.name"] == ("first", "second")


def test_two_fragments_pinned_to_different_stacks_are_refused_as_a_stack_conflict():
    moved = copy.deepcopy(CLASS1)
    moved["device"]["software_pinned_to"]["rocm"] = "7.3.0"
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("class1", CLASS1), fragment("class1b", moved)])
    assert refused.value.rule is Rule.PINNED_STACK
    assert "fragments pinned to different stacks" in refused.value.remedy


def test_the_merged_document_reads_as_a_spec_and_echoes_back():
    spec = resolved()
    assert MachineSpec.from_mapping(spec.echo()).digest() == spec.digest()


def test_the_merged_provenance_names_every_fragment_and_its_origin():
    combination = merge(
        [
            fragment("class1", CLASS1),
            fragment("links", LINKS, origin="datasheet", date="2026-09-20"),
            fragment(
                "ptl", {"device": {"ptl": "disabled"}}, authored_by="another person"
            ),
        ]
    )
    provenance = combination.document["provenance"]
    assert provenance["fragments"] == ["class1", "links", "ptl"]
    assert provenance["origin"] == "mixed"
    assert provenance["date"] == "2026-09-20"
    assert provenance["authored_by"] == "a person, another person"


def test_one_origin_is_kept_when_every_fragment_used_it():
    assert merged().document["provenance"]["origin"] == "probed"


@pytest.mark.parametrize(
    "path",
    [
        "schema_version",
        "name",
        "provenance.authored_by",
        "provenance.date",
        "provenance.origin",
    ],
)
def test_a_fragment_that_does_not_state_where_it_came_from_is_refused(path):
    document = {
        "schema_version": 1,
        "name": MACHINE,
        "provenance": {
            "authored_by": "a person",
            "date": "2026-09-18",
            "origin": "probed",
        },
    }
    document.update(copy.deepcopy(CLASS1))
    *blocks, leaf = path.split(".")
    node = document
    for block in blocks:
        node = node[block]
    del node[leaf]
    with pytest.raises(SpecRefusal) as refused:
        Fragment.from_mapping(document, "nameless")
    assert refused.value.rule is Rule.ONE_MACHINE
    assert path in str(refused.value)


def test_merging_nothing_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        merge([])
    assert refused.value.rule is Rule.ONE_MACHINE


def test_a_fragment_is_read_against_the_same_closed_schema():
    with pytest.raises(SpecRefusal) as refused:
        fragment("deployment", {"device": {"tensor_parallel_size": 8}})
    assert refused.value.rule is Rule.SEPARATION
    assert "EngineArgs.tensor_parallel_size" in str(refused.value)


def test_a_fragment_that_is_not_a_mapping_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        Fragment.from_mapping(["schema_version", 1], "a list")
    assert refused.value.rule is Rule.SHAPE


# --- validate: every reason at once, and the refusal list --------------------


def test_a_complete_spec_validates_and_resolves():
    checked = validate(merged().document)
    assert checked.ok, [str(refusal) for refusal in checked.refusals]
    assert str(checked) == "ok: 0 refusal(s)"


def test_every_missing_field_is_reported_and_not_only_the_first():
    thin = without(CLASS1, "device", "arch")
    thin = without(thin, "device", "memory", "capacity_bytes")
    checked = validate(merge(fragments(class1=thin)).document)
    missing = {
        refusal.what.split("`")[1]
        for refusal in checked.refusals
        if "`" in refusal.what
    }
    assert {"device.arch", "device.memory.capacity_bytes"} <= missing
    assert not checked.ok


def test_a_spec_peak_with_no_derate_beside_it_is_refused():
    thin = without(CLASS1, "device", "compute", "derate", "bf16")
    checked = validate(merge(fragments(class1=thin)).document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.DERATE]


def test_an_unknown_key_is_refused_where_it_sits():
    checked = validate(dict(merged().document, gpu_memory_utilization=0.9))
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SEPARATION]


def test_a_field_stated_twice_is_reported_beside_the_rest_rather_than_raised():
    # The reader refuses a document that states one field under two spellings.
    # The check walks the same document, so it reports that refusal with the
    # others and keeps walking: raised out of the walk instead, it would take
    # every other result of the run with it, the unknown key beside it included.
    document = dict(merged().document, gpu_memory_utilization=0.9)
    assert "link_latency_s" in document["interconnect"]["inter_node"]
    document["interconnect.inter_node.link_latency_s"] = 1
    checked = validate(document)
    rules = sorted(refusal.rule.name for refusal in checked.refusals)
    assert rules == ["SEPARATION", "SHAPE"]
    assert any("stated twice" in refusal.what for refusal in checked.refusals)


def test_one_mistyped_key_does_not_take_the_rest_of_the_document_with_it():
    # The refusal a mistyped key earns is about that key; the field it was
    # meant to state is reported missing beside it.
    document = merged().document
    memory = document["device"]["memory"]
    memory["capacity_byte"] = memory.pop("capacity_bytes")
    checked = validate(document)
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.SEPARATION) == 1
    assert rules.count(Rule.SHAPE) == 1


def test_a_block_holding_a_scalar_does_not_take_the_rest_with_it_either():
    # The other way the walk refuses: a block written as a number. The fields
    # under it are gone, and everything beside it is still checked.
    document = merged().document
    document["device"]["memory"] = 288.0e9
    checked = validate(document)
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.DERATE) == 1


def test_a_value_of_the_wrong_shape_is_reported():
    document = merged().document
    document["device"]["count_per_node"] = True
    checked = validate(document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]


# --- explain: the basis of a number ------------------------------------------


def test_every_term_a_resolved_spec_holds_can_be_explained():
    # The exit criterion, and it has to assert what is in the basis rather than
    # that there is one: a path that is in `spec.values` cannot come back empty,
    # so reachability alone survives every wrong value this module could report.
    # Each row states the value the document holds, and carries a derate exactly
    # when the field it came from is a spec peak.
    spec = resolved()
    for path in sorted(spec.values):
        basis = explain(spec, path)
        assert basis.contributions, path
        assert basis.digest == spec.digest()
        field = BY_PATH[path]
        for row in basis.contributions:
            assert row.value == spec.values[path], path
            assert (row.derate is not None) == field.peak, path


def test_a_spec_peak_is_reported_with_its_derate_and_the_derated_value():
    basis = explain(resolved(), "device.compute.bf16_flops")
    (row,) = basis.contributions
    assert (row.value, row.derate) == (2.5e15, 0.70)
    assert row.effective == pytest.approx(1.75e15)
    assert "x 0.7" in str(row)


def test_a_number_that_is_not_a_spec_peak_carries_no_derate():
    (row,) = explain(resolved(), "device.memory.capacity_bytes").contributions
    assert row.derate is None and row.effective == row.value


def test_a_block_explains_to_every_field_under_it():
    term = "interconnect.inter_node"
    paths = {row.path for row in explain(resolved(), term).contributions}
    assert paths == {
        f"{term}.link_bandwidth_bytes_per_s",
        f"{term}.link_latency_s",
        f"{term}.derate",
    }


def test_a_basis_names_the_fragment_each_field_came_from():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    basis = explain(spec, "device.memory.capacity_bytes", origin=combination)
    supplied = {row.path: row.supplied_by for row in basis.contributions}
    assert supplied["device.memory.capacity_bytes"] == ("class1",)
    links = explain(spec, "interconnect.intra_node", origin=combination)
    supplied = {row.path: row.supplied_by for row in links.contributions}
    assert supplied["interconnect.intra_node.link_latency_s"] == ("links",)


def test_without_a_merge_a_basis_still_reads_but_names_no_fragment():
    basis = explain(resolved(), "device.memory.capacity_bytes")
    assert basis.contributions[0].supplied_by == ()


def test_a_term_the_schema_does_not_know_is_asked_for_again_by_path():
    # An unknown term and an optional field the document left out both come
    # back empty however they are spelled, so both are asked for again.
    assert "device.clock_ceiling" not in BY_PATH
    assert not BY_PATH["provenance.notes"].required
    for term in ("device.clock_ceiling", "provenance.notes"):
        with pytest.raises(SpecRefusal) as refused:
            explain(resolved(), term)
        assert refused.value.rule is Rule.ADDRESSING
        assert refused.value.what == f"`{term}` names no field this spec states"
        assert "ask again by the dotted path" in refused.value.remedy


def test_a_block_explains_without_an_optional_field_the_document_left_out():
    spec = resolved()
    assert "provenance.notes" not in spec.values
    paths = {row.path for row in explain(spec, "provenance").contributions}
    assert "provenance.authored_by" in paths
    assert "provenance.notes" not in paths


def test_the_basis_prints_its_spec_and_one_line_per_field():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    term = "interconnect.inter_node"
    printed = str(explain(spec, term, origin=combination)).splitlines()
    assert printed[0].endswith(spec.digest())
    assert printed[2].strip().startswith(f"{term}.link_latency_s = 5e-06")
    assert printed[2].endswith("[links]  probed")


# --- one reading per rank ----------------------------------------------------


@pytest.mark.parametrize(
    "tp_width,reading,spread",
    [
        (1, 926 * MIB, 0),
        (2, 6906 * MIB, 0),
        (4, 7266 * MIB, 192 * MIB),
        (8, 10704 * MIB, 640 * MIB),
    ],
)
def test_the_spread_this_hardware_really_shows_is_not_an_error(
    tp_width, reading, spread
):
    across_ranks("non_torch", [reading] * (tp_width - 1) + [reading + spread])


def test_a_neighbour_holding_fifty_times_the_rank_is_refused():
    # The case this exists for: a rank had reserved 2.94 GB and the card read
    # 152.01 GB, and six runs died on a negative memory budget before anything
    # compared the ranks.
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [2.94e9, 2.94e9, 2.94e9, 152.01e9])
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "2940000000.0" in refused.value.what
    assert "152010000000.0" in refused.value.what


def test_a_reading_no_spec_could_hold_is_refused_where_the_ranks_are():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [0, 970.0e6])
    assert refused.value.rule is Rule.SHAPE


# --- what the package reaches -------------------------------------------------


def test_the_verbs_are_reachable_by_name():
    for name in ("merge", "validate", "explain", "across_ranks"):
        assert name in spec_package.__all__
        assert getattr(spec_package, name, None) is not None


SPEC_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _node18_script():
    path = SPEC_ROOT / "scripts" / "compass" / "mi308x_spec.py"
    loaded = importlib.util.spec_from_file_location("mi308x_spec", path)
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    return module


def test_the_node_18_fragments_validate_and_explain_splits_measured_from_declared():
    script = _node18_script()
    merged = merge(script.device_fragments())
    assert validate(merged.document).ok
    measured, declared = script.report(merged).split("assumed:")
    for field in ("device.memory.capacity_bytes", "device.ptl"):
        assert field in measured and field not in declared
    for field in ("interconnect.inter_node.derate", "inter_node.link_latency_s"):
        assert field in declared and field not in measured


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_the_checked_in_mi308x_specs_hold_their_start(monkeypatch, name):
    monkeypatch.setenv("COMPASS_MACHINE", name)
    script = _node18_script()
    path = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    values = MachineSpec.from_mapping(json.loads(path.read_text())).values
    for fragment in script.device_fragments():
        for key, value in fragment.values.items():
            if not key.startswith("provenance."):
                assert values[key] == value, key


MI355X = ["mi355x-gpu45", "mi355x-crs005", "mi355x-gpu40"]


def _mi355x_spec(monkeypatch, name):
    """The MI355X script for machine `name` and its checked-in document, read."""
    monkeypatch.setenv("COMPASS_MACHINE", name)
    path = SPEC_ROOT / "scripts" / "compass" / "mi355x_spec.py"
    loaded = importlib.util.spec_from_file_location("mi355x_spec", path)
    script = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(script)
    assert script.NAME == name
    document = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    return script, MachineSpec.from_mapping(json.loads(document.read_text()))


@pytest.mark.parametrize("name", ["mi355x-crs013", "mi308x-hjbog18"])
def test_the_mi355x_spec_refuses_a_machine_it_has_no_tables_for(monkeypatch, name):
    with pytest.raises(SystemExit, match=f"COMPASS_MACHINE={name}: "):
        _mi355x_spec(monkeypatch, name)


@pytest.mark.parametrize("name", MI355X)
def test_the_checked_in_mi355x_spec_holds_its_start(monkeypatch, name):
    script, spec = _mi355x_spec(monkeypatch, name)
    for fragment in script.device_fragments():
        for key, value in fragment.values.items():
            if not key.startswith("provenance."):
                assert spec.values[key] == value, key


@pytest.mark.parametrize("name", MI355X)
def test_the_mi355x_spec_explains_what_the_start_measured_apart_from_the_declared(
    monkeypatch, name
):
    script, spec = _mi355x_spec(monkeypatch, name)
    origins = {}
    for block in ("device", "interconnect"):
        for row in explain(spec, block).contributions:
            origins.setdefault(row.origin["origin"], set()).add(row.path)
    assert set(origins) == {"probed", "datasheet", "assumed"}
    assert "device.memory.capacity_bytes" in origins["probed"]
    assert origins["datasheet"] == {
        "device.memory.bandwidth_bytes_per_s",
        "device.compute.bf16_flops",
        "device.compute.fp8_flops",
        "device.compute.fp4_flops",
        "interconnect.intra_node.domain_devices",
        "interconnect.intra_node.link_bandwidth_bytes_per_s",
    }
    record = spec.origin("device.memory.capacity_bytes")["record"]
    assert record == f"sha256:{script.RECORD}"


@pytest.mark.parametrize("name", MI355X)
def test_the_mi355x_spec_rebuilds_byte_for_byte(monkeypatch, tmp_path, name):
    script, _ = _mi355x_spec(monkeypatch, name)
    document = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    out = tmp_path / "rebuilt.json"
    assert script.main(["--out", str(out)]) == 0
    assert out.read_bytes() == document.read_bytes()


def test_the_gpu40_spec_pins_nightly_20261026_with_the_recipes_aiter(monkeypatch):
    script, spec = _mi355x_spec(monkeypatch, "mi355x-gpu40")
    document = SPEC_ROOT / "scripts" / "compass" / "machines" / "mi355x-gpu40.json"
    assert validate(json.loads(document.read_text())).ok
    recipe = SPEC_ROOT / "scripts" / "compass" / "aiter_c90842add0.sh"
    patched = re.search(r"^PATCHED=([0-9a-f]{40})$", recipe.read_text(), re.MULTILINE)
    pinned = {
        k: spec.value(f"device.software_pinned_to.{k}")
        for k in ("rocm", "aiter", "rccl", "torch")
    }
    assert pinned == {
        "rocm": "7.2.4",
        "aiter": patched.group(1)[:9],
        "rccl": "2.27.7",
        "torch": "2.10.0+rocm7.2.4.git3d3aa833",
    }
    assert spec.value("device.ptl") == "disabled"
    # The stack is the patched container's start's; the device is still the
    # device start's, which ran the image's aiter.
    stack = spec.origin("device.software_pinned_to.aiter")["record"]
    assert stack == f"sha256:{script.STACK[-1]}"
    assert spec.origin("device.name")["record"] == f"sha256:{script.RECORD}"


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_the_mi308x_specs_rebuild_byte_for_byte(monkeypatch, tmp_path, name):
    monkeypatch.setenv("COMPASS_MACHINE", name)
    script = _node18_script()
    document = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    out = tmp_path / "rebuilt.json"
    assert script.main(["--out", str(out)]) == 0
    assert out.read_bytes() == document.read_bytes()
