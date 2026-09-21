# SPDX-License-Identifier: MIT
"""The three verbs over the spec: combining fragments, checking one, reading one.

The fragments below are the ones the probes of each hardware tier would emit --
a tokenizer and an interprocess measurement that need no device, a single-card
engine start, a multi-rank one, and a fabric measurement -- so a test that
combines them combines the same partial documents a real run would, and a test
that breaks one breaks exactly what it says it does.

The numbers are the measured ones wherever a measurement exists, because two of
these checks are judgements about size: the cross-rank spread accepted is the
one the hardware really shows, and the spread refused is the one a neighbour
really produced. Round numbers would make both tests vacuous.
"""

import copy

import pytest

from atom.compass import spec as spec_package
from atom.compass.spec import (
    QUANTITIES,
    Fragment,
    MachineSpec,
    Rule,
    SpecRefusal,
    StackMismatch,
    across_ranks,
    explain,
    merge,
    validate,
)
from atom.compass.spec.fields import BY_PATH, Kind
from atom.compass.spec.tokenizers import ENTRY_FIELDS
from atom.compass.spec.validate import STACK as STACK_ASKED
from atom.compass.spec.validate import TRANSFERS as TRANSFERS_ASKED
from atom.compass.spec.validate import WIDTHS as WIDTHS_ASKED

MACHINE = "mi355x-8gpu-2node"
#: A tokenizer entry's fields by name, to say which of its rates is a peak.
ENTRY_RATES = {field.path: field for field in ENTRY_FIELDS}
STACK = {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.22.3"}
MIB = 1024**2

TOKENIZER = {
    "id": "qwen3-151k-bpe",
    "backend": "fast",
    "vocab_size": 151936,
    "fingerprint": "sha256:" + "a" * 64,
    "applies_to": ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"],
    "encode_fixed_s": 3.0e-4,
    "encode_tokens_per_s": 2.0e6,
    "decode_fixed_s": 1.5e-4,
    "decode_tokens_per_s": 3.0e6,
    "derate": 0.85,
}

#: Tier 0, no device: the tokenizer sweep and the interprocess round trips.
TIER0 = {
    "host": {
        "cpu": {"cores_physical": 96, "cores_logical": 192},
        "tokenizers": [TOKENIZER],
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "admission_fixed_s": 9.0e-3,
    }
}

#: Tier 1, one card: the five readings and the single-width graph pool.
TIER1 = {
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {
            "capacity_bytes": 288.0e9,
            "bandwidth_bytes_per_s": 8.0e12,
            "derate": 0.85,
        },
        "compute": {"bf16_flops": 2.5e15, "fp8_flops": 5.0e15, "derate": 0.70},
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {1: 970.0e6},
            "allocator_retained_after_load_bytes": {1: 1.1e6},
            "persistent_forward_buffer_bytes": 124.0e6,
            "cudagraph_pool": {
                "w1_base_bytes": 95.5e6,
                "w1_bytes_per_captured_token": 0.318e6,
            },
        },
        "software_pinned_to": dict(STACK),
    }
}

#: Tier 2, N cards: one engine start per width, and the flat pool above one.
TIER2 = {
    "device": {
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {2: 7.2e9, 4: 7.6e9, 8: 11.2e9},
            "allocator_retained_after_load_bytes": {2: 2.17e9, 4: 2.17e9, 8: 2.17e9},
            "cudagraph_pool": {"w_gt1_flat_bytes": 109.0e6},
        }
    }
}

#: Tier 2 and 3: collectives in real groups, then the same across two nodes.
LINKS = {
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
            "link_bandwidth_bytes_per_s": 1.0e12,
            "link_latency_s": 2.0e-6,
            "derate": 0.80,
        },
        "inter_node": {
            "link_bandwidth_bytes_per_s": 5.0e10,
            "link_latency_s": 5.0e-6,
            "derate": 0.80,
        },
        "router_relay_s": 1.5e-3,
    }
}


def fragment(
    source,
    body=None,
    *,
    machine=MACHINE,
    method="probed",
    authored_by="a person",
    date="2026-09-18",
    **stanza,
):
    """One probe's output: what it measured, and where it was measured."""
    document = {
        "schema_version": 1,
        "name": machine,
        "provenance": dict(
            {"authored_by": authored_by, "date": date, "method": method}, **stanza
        ),
    }
    document.update(copy.deepcopy(body or {}))
    return Fragment.from_mapping(document, source)


def fragments(**edits):
    """The five fragments a full spec is built from, with one body replaced."""
    bodies = {
        "tier0": TIER0,
        "tier1": TIER1,
        "tier2": TIER2,
        "links": LINKS,
    }
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
    # The named result. Two tokenizers measured on two machines contradict
    # nothing in their shape: different identities, different entries, no field
    # of one overlapping a field of the other. Only the stanzas differ, so only
    # the stanzas can catch it.
    here = fragment("tokenizer-a", TIER0, machine="node-18", authored_by="ana")
    elsewhere = fragment(
        "tokenizer-b",
        {"host": {"tokenizers": [dict(TOKENIZER, id="llama-128k-bpe")]}},
        machine="node-22",
        authored_by="bo",
        date="2026-09-19",
    )
    with pytest.raises(SpecRefusal) as refused:
        merge([here, elsewhere])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    for named in ("tokenizer-a", "tokenizer-b", "node-18", "node-22", "ana", "bo"):
        assert named in message, f"{named} is not named in: {message}"


def test_the_refusal_says_the_fragments_are_authored_for_different_machines():
    # `Fragment.machine` is the `name` field, which is the machine the spec is
    # being authored for. No field records where a probe ran, so the refusal
    # says what the fragments claim and not where their numbers came from.
    with pytest.raises(SpecRefusal) as refused:
        merge(
            [fragment("a", TIER0, machine="node-18"), fragment("b", machine="node-22")]
        )
    message = str(refused.value)
    assert "are authored for different machines" in message
    assert "measured on" not in message


def test_the_same_two_tokenizers_merge_when_the_machine_agrees():
    # The control for the result above: the same two measurements, the same two
    # authors and dates, differing only in the machine they name.
    here = fragment("tokenizer-a", TIER0, machine="node-18", authored_by="ana")
    also_here = fragment(
        "tokenizer-b",
        {"host": {"tokenizers": [dict(TOKENIZER, id="llama-128k-bpe")]}},
        machine="node-18",
        authored_by="bo",
        date="2026-09-19",
    )
    combination = merge([here, also_here])
    measured = [entry["id"] for entry in combination.document["host"]["tokenizers"]]
    assert measured == ["qwen3-151k-bpe", "llama-128k-bpe"]
    assert combination.document["name"] == "node-18"
    assert combination.document["provenance"]["fragments"] == [
        "tokenizer-a",
        "tokenizer-b",
    ]


def test_a_merged_document_merged_again_keeps_the_names_that_built_it():
    # Incremental authoring: merge what has been measured, save it, merge next
    # week's probe into the saved document. A merged document is a legal
    # fragment, so this works -- and the block whose whole job is to say where
    # the numbers came from must not lose the fragments of the first round.
    first = merge([fragment("tier0", TIER0), fragment("tier1", TIER1)])
    again = merge(
        [
            Fragment.from_mapping(first.document, "machine.yaml"),
            fragment("tier2", TIER2),
            fragment("links", LINKS),
        ]
    )
    assert again.document["provenance"]["fragments"] == [
        "tier0",
        "tier1",
        "machine.yaml",
        "tier2",
        "links",
    ]


def test_one_machine_named_twice_with_two_answers_is_refused():
    other_cpu = {
        "host": dict(TIER0["host"], cpu={"cores_physical": 128, "cores_logical": 256})
    }
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", TIER0), fragment("second", other_cpu)])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "host.cpu.cores_physical" in message
    assert "96" in message and "128" in message
    assert "first" in message and "second" in message


def test_two_fragments_that_agree_are_not_a_conflict():
    combination = merge([fragment("first", TIER0), fragment("second", TIER0)])
    assert combination.sources["host.cpu.cores_physical"] == ("first", "second")


def test_a_width_table_is_combined_across_the_probes_that_measured_it():
    reserve = resolved().value(
        "device.runtime_constants.driver_and_collective_reserve_bytes"
    )
    assert sorted(reserve) == [1, 2, 4, 8]
    assert reserve[1] == 970.0e6 and reserve[8] == 11.2e9


def test_one_width_measured_twice_differently_is_refused():
    disagrees = copy.deepcopy(TIER2)
    disagrees["device"]["runtime_constants"]["driver_and_collective_reserve_bytes"][
        1
    ] = 980.0e6
    with pytest.raises(SpecRefusal) as refused:
        merged(tier2=disagrees)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "driver_and_collective_reserve_bytes[1]" in str(refused.value)


def test_the_merged_document_reads_as_a_spec_and_echoes_back():
    spec = resolved()
    assert MachineSpec.from_mapping(spec.echo()).digest() == spec.digest()


def test_the_merged_provenance_names_every_fragment_and_its_method():
    combination = merge(
        [
            fragment("tier0", TIER0),
            fragment("tier1", TIER1, method="datasheet", date="2026-09-20"),
            fragment("tier2", TIER2),
            fragment("links", LINKS, authored_by="another person"),
        ]
    )
    provenance = combination.document["provenance"]
    assert provenance["fragments"] == ["tier0", "tier1", "tier2", "links"]
    assert provenance["method"] == "mixed"
    assert provenance["date"] == "2026-09-20"
    assert provenance["authored_by"] == "a person, another person"


def test_one_method_is_kept_when_every_fragment_used_it():
    assert merged().document["provenance"]["method"] == "probed"


@pytest.mark.parametrize(
    "path",
    [
        "schema_version",
        "name",
        "provenance.authored_by",
        "provenance.date",
        "provenance.method",
    ],
)
def test_a_fragment_that_does_not_state_where_it_came_from_is_refused(path):
    document = {
        "schema_version": 1,
        "name": MACHINE,
        "provenance": {
            "authored_by": "a person",
            "date": "2026-09-18",
            "method": "probed",
        },
    }
    document.update(copy.deepcopy(TIER0))
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


# --- merge: the tokenizer table, keyed on the identity a person chose ---------


def test_one_id_over_two_files_is_refused_with_both_fingerprints():
    revised = {
        "host": {"tokenizers": [dict(TOKENIZER, fingerprint="sha256:" + "b" * 64)]}
    }
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", TIER0), fragment("second", revised)])
    assert refused.value.rule is Rule.TOKENIZER_IDENTITY
    assert "a" * 64 in str(refused.value) and "b" * 64 in str(refused.value)


def test_one_tokenizer_measured_on_both_backends_is_not_a_conflict():
    slow = {
        "host": {
            "tokenizers": [dict(TOKENIZER, backend="slow", encode_tokens_per_s=1.6e5)]
        }
    }
    combination = merge([fragment("fast", TIER0), fragment("slow", slow)])
    backends = [
        entry["backend"] for entry in combination.document["host"]["tokenizers"]
    ]
    assert backends == ["fast", "slow"]


def test_the_same_entry_measured_twice_differently_is_refused():
    faster = {"host": {"tokenizers": [dict(TOKENIZER, encode_tokens_per_s=9.0e6)]}}
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", TIER0), fragment("second", faster)])
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "host.tokenizers[qwen3-151k-bpe fast]" in str(refused.value)


def test_a_tokenizer_entry_with_no_id_combines_with_nothing():
    nameless = {"host": {"tokenizers": [without(TOKENIZER, "id")]}}
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("nameless", nameless)])
    assert refused.value.rule is Rule.TOKENIZER_IDENTITY


# --- validate: every reason at once, and the refusal list --------------------


def test_a_complete_spec_validates_and_resolves():
    checked = validate(merged(), tp_widths=(1, 2, 4, 8), observed_stack=STACK)
    assert checked.ok, [str(refusal) for refusal in checked.refusals]
    assert checked.spec.value("device.name") == "MI355X"
    assert checked.stack_differences == ()


def test_every_missing_field_is_reported_and_not_only_the_first():
    thin = without(TIER1, "device", "arch")
    thin = without(thin, "device", "memory", "capacity_bytes")
    checked = validate(merge(fragments(tier1=thin)).document)
    missing = {
        refusal.what.split("`")[1]
        for refusal in checked.refusals
        if "`" in refusal.what
    }
    assert {"device.arch", "device.memory.capacity_bytes"} <= missing
    assert not checked.ok and checked.spec is None


def test_a_missing_runtime_constant_has_no_default():
    thin = without(
        TIER1, "device", "runtime_constants", "persistent_forward_buffer_bytes"
    )
    checked = validate(merge(fragments(tier1=thin)).document)
    rules = {refusal.rule for refusal in checked.refusals}
    assert rules == {Rule.NO_DEFAULTS}
    assert "persistent_forward_buffer_bytes" in str(checked.refusals[0])


def test_a_spec_peak_with_no_derate_beside_it_is_refused():
    thin = without(TIER1, "device", "compute", "derate")
    checked = validate(merge(fragments(tier1=thin)).document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.DERATE]


def test_a_width_the_deployment_will_use_and_nobody_measured_is_refused():
    checked = validate(merged(), tp_widths=(1, 16))
    assert len(checked.refusals) == 2
    for refusal in checked.refusals:
        assert refusal.rule is Rule.NO_DEFAULTS
        assert "16" in refusal.what
    assert "[1, 2, 4, 8]" in checked.refusals[0].what


def test_the_widths_that_were_measured_are_not_refused():
    assert validate(merged(), tp_widths=(1, 2, 4, 8)).ok


def test_a_stack_that_has_moved_warns_and_names_both_versions():
    with pytest.warns(StackMismatch, match="7.2.4"):
        checked = validate(merged(), observed_stack=dict(STACK, rocm="7.3.0"))
    assert checked.ok
    assert checked.stack_differences == (("rocm", "7.2.4", "7.3.0"),)


def test_a_stack_that_has_moved_is_refused_under_the_strict_flag():
    with pytest.warns(StackMismatch):
        checked = validate(
            merged(), observed_stack=dict(STACK, rocm="7.3.0"), strict=True
        )
    assert not checked.ok
    assert checked.refusals[0].rule is Rule.PINNED_STACK
    assert "7.3.0" in checked.refusals[0].what


def test_a_stack_that_matches_is_silent():
    assert validate(merged(), observed_stack=STACK, strict=True).ok


def test_a_stack_difference_is_rendered_by_the_run_that_refused_nothing():
    # Without the strict flag nothing is refused, so the difference is the only
    # thing this check found. It is on the dataclass either way; the string is
    # where a reader of the summary looks for what the check did.
    with pytest.warns(StackMismatch):
        checked = validate(
            merged(),
            tp_widths=(1, 2, 4, 8),
            observed_stack=dict(STACK, rocm="7.3.0"),
        )
    assert checked.ok
    assert str(checked) == (
        "ok: 0 refusal(s)\n"
        "  stack moved: the constants were measured against rocm '7.2.4' "
        "(now '7.3.0')"
    )


def test_a_transfer_keeps_the_source_stack_out_of_this_machines_pin():
    # The constants came from another spec; the stack that spec was pinned to is
    # provenance about them, not a claim about this host's stack.
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK, rocm="7.0.2")
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    assert combination.document["device"]["software_pinned_to"]["rocm"] == "7.2.4"
    checked = validate(combination)
    assert not checked.ok
    assert checked.refusals[0].rule is Rule.PINNED_STACK
    assert "mi300x-8gpu" in checked.refusals[0].what
    assert "7.0.2" in checked.refusals[0].what


def test_a_transfer_that_names_no_stack_at_all_is_refused():
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", TIER2, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    checked = validate(combination)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.PINNED_STACK]
    assert "without saying which stack" in checked.refusals[0].what


def test_a_transfer_from_a_spec_pinned_to_this_stack_validates():
    carried = dict(copy.deepcopy(TIER2))
    carried["device"]["software_pinned_to"] = dict(STACK)
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    assert validate(combination).ok


def test_a_clear_check_says_which_conditions_it_could_not_ask():
    # Three of the five conditions are opt-in, and a check that is silent about
    # what it declined to ask is a clear with no content behind it.
    bare = validate(merged().document)
    assert bare.ok
    assert [condition.split(" -- ")[0] for condition in bare.not_asked] == [
        WIDTHS_ASKED,
        STACK_ASKED,
        TRANSFERS_ASKED,
    ]
    assert "`tp_widths=`" in bare.not_asked[0]
    asked = validate(
        merged(), tp_widths=(1, 2, 4, 8), observed_stack=STACK, strict=True
    )
    assert asked.ok and asked.not_asked == ()


def test_the_transfer_condition_names_itself_as_unaskable_of_a_document():
    # The same spec is refused as a `Merge` and clear as the document it makes,
    # because the source's pin is deliberately in no field of the document. The
    # verb the design writes takes a file, so the document must say as much.
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK, rocm="7.0.2")
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    assert not validate(combination).ok
    as_document = validate(combination.document)
    assert as_document.ok
    assert any(
        condition.startswith(TRANSFERS_ASKED) for condition in as_document.not_asked
    )
    assert "in no field" in str(as_document)


def test_an_incomplete_document_says_no_consistency_question_was_asked():
    # The phase-one refusal here is a derate an author types in at a desk; the
    # width table behind it is present and schema-valid, and the width nobody
    # measured costs an eight-GPU reservation. Phase two does not run, so the
    # result has to say that rather than let the cheap refusal stand alone.
    thin = without(TIER1, "device", "memory", "derate")
    checked = validate(
        merge(fragments(tier1=thin)).document,
        tp_widths=(1, 2, 4, 8, 16),
        observed_stack=dict(STACK, rocm="7.3.0"),
        strict=True,
    )
    assert [refusal.rule for refusal in checked.refusals] == [Rule.DERATE]
    assert len(checked.not_asked) == 3
    for condition in checked.not_asked:
        assert "not a complete spec" in condition
    assert "hiding a more expensive one" in str(checked)


def test_a_document_that_is_not_a_mapping_is_refused():
    checked = validate("a filename, not a document")
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]


def test_an_unknown_key_is_refused_where_it_sits():
    checked = validate(dict(merged().document, gpu_memory_utilization=0.9))
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SEPARATION]


def test_a_value_of_the_wrong_shape_is_reported():
    document = merged().document
    document["host"]["cpu"]["cores_physical"] = True
    checked = validate(document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]


def test_two_tokenizer_entries_claiming_one_architecture_are_refused():
    twice = {
        "host": dict(
            TIER0["host"],
            tokenizers=[
                TOKENIZER,
                dict(TOKENIZER, id="other", fingerprint="sha256:" + "c" * 64),
            ],
        )
    }
    document = merge(fragments(tier0=twice)).document
    checked = validate(document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.TOKENIZER_IDENTITY]


def test_raise_first_gives_the_spec_or_the_first_refusal():
    assert validate(merged()).raise_first().value("device.arch") == "gfx950"
    with pytest.raises(SpecRefusal):
        validate(without(merged().document, "device", "arch")).raise_first()


def test_the_refusal_list_is_the_one_the_design_asks_for():
    # Each condition, with the rule that declines it, in one place: a reader
    # comparing this against the list it implements has one thing to read.
    thin = without(TIER1, "device", "arch")
    no_constant = without(
        TIER1, "device", "runtime_constants", "persistent_forward_buffer_bytes"
    )
    no_derate = without(TIER1, "device", "memory", "derate")
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK, rocm="7.0.2")
    transferred = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    with pytest.warns(StackMismatch):
        moved = validate(
            merged(), observed_stack=dict(STACK, rocm="7.3.0"), strict=True
        )
    conditions = {
        "a missing required field": (
            validate(merge(fragments(tier1=thin)).document),
            Rule.SHAPE,
        ),
        "an unmeasured width the deployment uses": (
            validate(merged(), tp_widths=(16,)),
            Rule.NO_DEFAULTS,
        ),
        "a runtime constant with no default": (
            validate(merge(fragments(tier1=no_constant)).document),
            Rule.NO_DEFAULTS,
        ),
        "a derate missing beside a spec peak": (
            validate(merge(fragments(tier1=no_derate)).document),
            Rule.DERATE,
        ),
        "a stack that no longer matches": (moved, Rule.PINNED_STACK),
        "a transfer out of a differently pinned spec": (
            validate(transferred),
            Rule.PINNED_STACK,
        ),
    }
    for condition, (checked, rule) in conditions.items():
        assert not checked.ok, condition
        assert rule in {refusal.rule for refusal in checked.refusals}, condition


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
        if field.kind is Kind.WIDTH_TABLE:
            assert {row.path for row in basis.contributions} == {
                f"{path}[{width}]" for width in spec.values[path]
            }, path
        if field.kind is Kind.TOKENIZERS:
            assert {row.path.split("].")[0] + "]" for row in basis.contributions} == {
                f"{path}[{entry['id']} {entry['backend']}]"
                for entry in spec.values[path]
            }, path
        for row in basis.contributions:
            if field.kind is Kind.TOKENIZERS:
                rate = ENTRY_RATES[row.path.rsplit(".", 1)[-1]]
                assert (row.derate is not None) == rate.peak, row.path
                continue
            if field.kind is not Kind.WIDTH_TABLE:
                assert row.value == spec.values[path], path
            assert (row.derate is not None) == field.peak, path


def test_a_tokenizer_row_names_the_fragment_that_supplied_the_entry():
    # `merge` labels an entry with the raw backend string and `explain` rebuilds
    # the label from the parsed `Backend`; nothing else asserts that the two
    # still agree, and a row whose source is silently `()` looks identical to a
    # row explained without an origin.
    origin = merged()
    basis = explain(
        MachineSpec.from_mapping(origin.document), "host.tokenizers", origin=origin
    )
    assert basis.contributions
    for row in basis.contributions:
        assert row.supplied_by == ("tier0",), row.path


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
    paths = {row.path for row in explain(resolved(), "host.ipc").contributions}
    assert paths == {"host.ipc.zmq_roundtrip_s", "host.ipc.shm_broadcast_s"}


def test_a_quantity_names_the_fields_it_is_built_out_of():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    basis = explain(spec, "kv_blocks", tp_width=8, origin=combination)
    paths = [row.path for row in basis.contributions]
    assert "device.memory.capacity_bytes" in paths
    assert "device.runtime_constants.driver_and_collective_reserve_bytes[8]" in paths
    assert len(paths) == len(set(paths))


def test_a_quantity_names_the_fragment_each_field_came_from():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    basis = explain(spec, "kv_blocks", tp_width=8, origin=combination)
    supplied = {row.path: row.supplied_by for row in basis.contributions}
    assert supplied["device.memory.capacity_bytes"] == ("tier1",)
    assert supplied[
        "device.runtime_constants.driver_and_collective_reserve_bytes[8]"
    ] == ("tier2",)


def test_without_a_merge_a_basis_still_reads_but_names_no_fragment():
    basis = explain(resolved(), "device.memory.capacity_bytes")
    assert basis.contributions[0].supplied_by == ()


def test_a_width_keyed_constant_explains_at_every_measured_width():
    basis = explain(
        resolved(), "device.runtime_constants.driver_and_collective_reserve_bytes"
    )
    assert [row.path.rsplit("[", 1)[-1] for row in basis.contributions] == [
        "1]",
        "2]",
        "4]",
        "8]",
    ]


def test_explaining_at_a_width_nobody_measured_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        explain(resolved(), "kv_blocks", tp_width=16)
    assert refused.value.rule is Rule.NO_DEFAULTS


def test_tokenizer_rates_explain_per_entry_and_per_rate():
    basis = explain(resolved(), "host.tokenizers")
    paths = [row.path for row in basis.contributions]
    assert paths == [
        "host.tokenizers[qwen3-151k-bpe fast].encode_fixed_s",
        "host.tokenizers[qwen3-151k-bpe fast].encode_tokens_per_s",
        "host.tokenizers[qwen3-151k-bpe fast].decode_fixed_s",
        "host.tokenizers[qwen3-151k-bpe fast].decode_tokens_per_s",
    ]
    rates = {row.path.rsplit(".", 1)[-1]: row for row in basis.contributions}
    assert rates["encode_tokens_per_s"].derate == 0.85
    assert rates["encode_fixed_s"].derate is None


def test_every_quantity_is_built_out_of_fields_this_schema_really_has():
    for term, paths in QUANTITIES.items():
        for path in paths:
            assert path in BY_PATH, f"{term} names {path}, which is not a field"


def test_a_term_this_spec_carries_nothing_for_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        explain(resolved(), "device.clock_ceiling")
    assert refused.value.rule is Rule.SHAPE
    assert "kv_blocks" in refused.value.remedy


def test_the_basis_prints_its_spec_and_one_line_per_field():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    printed = str(explain(spec, "host.ipc", origin=combination)).splitlines()
    assert printed[0].endswith(spec.digest())
    assert printed[1].strip().startswith("host.ipc.zmq_roundtrip_s = 5e-05")
    assert printed[1].endswith("[tier0]")


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
    readings = [reading] * (tp_width - 1) + [reading + spread]
    measured = across_ranks("non_torch", tp_width, readings)
    assert measured.minimum == reading
    assert measured.spread == spread
    assert measured.relative == spread / reading


def test_the_reading_taken_is_the_smallest_whatever_order_the_ranks_came_in():
    high = across_ranks(
        "non_torch", 4, [7458 * MIB, 7266 * MIB, 7266 * MIB, 7266 * MIB]
    )
    assert high.minimum == 7266 * MIB


def test_a_neighbour_holding_fifty_times_the_rank_is_refused():
    # The case this exists for: a rank had reserved 2.94 GB and the card read
    # 152.01 GB, and six runs died on a negative memory budget before anything
    # compared the ranks.
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", 4, [2.94e9, 2.94e9, 2.94e9, 152.01e9])
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "2940000000.0" in refused.value.what
    assert "152010000000.0" in refused.value.what


def test_the_spread_is_reported_and_not_folded_into_the_reading():
    measured = across_ranks("non_torch", 8, [10704 * MIB] * 7 + [11344 * MIB])
    assert "spread" in str(measured) and "6.0%" in str(measured)


def test_a_rank_that_was_not_read_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", 8, [10704 * MIB] * 7)
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "7 reading(s) for the 8 ranks" in refused.value.what


def test_a_reading_no_spec_could_hold_is_refused_where_the_ranks_are():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", 2, [0, 970.0e6])
    assert refused.value.rule is Rule.SHAPE


def test_how_quiet_the_machine_has_to_be_is_the_callers_judgement():
    readings = [10704 * MIB] * 7 + [11344 * MIB]
    assert across_ranks("non_torch", 8, readings, limit=0.06).relative < 0.06
    with pytest.raises(SpecRefusal):
        across_ranks("non_torch", 8, readings, limit=0.01)


# --- what the package reaches -------------------------------------------------


def test_the_three_verbs_are_reachable_by_name():
    for name in ("merge", "validate", "explain", "across_ranks"):
        assert name in spec_package.__all__
        assert getattr(spec_package, name, None) is not None
