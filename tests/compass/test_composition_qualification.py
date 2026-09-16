"""Independent heldout qualification does not rewrite primitive source status."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from atom.compass.core.cost.composition_qualification import (
    SCHEMA, geometry, observed_components, predictor_identity, validate,
)
from atom.compass.core.loaded_input import load_json
from .test_native_ap_work import candidate, fresh_descriptor


@pytest.fixture
def bundle(candidate, tmp_path):
    regions, _ = candidate
    pins = {}

    def write(name, value):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        pins[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return pins[name]

    descriptor = fresh_descriptor([15], [222640], [13916], True)
    descriptor["req_ids"] = [1]
    row = dict(chain_id="native-chain", chain_step=0, repetition=0, role="source",
        normal_return=True, descriptor=descriptor,
        forward_context_after_return=dict(previous_sampled_ids=dict(shape=[1])))
    sources = []
    for repetition in range(6):
        source = copy.deepcopy(row)
        source["repetition"] = repetition
        source["descriptor"]["req_ids"] = [1 + repetition]
        sources.append(source)
    write("source", dict(rows=sources))
    write("model", {})
    write("scope", {})
    inputs = [load_json(pins[name]["path"], role=role)[1] for name, role in (
        ("source", "oracle.native_ap_regions.work.source"),
        ("model", "oracle.native_ap_regions.work.model"), ("scope", "oracle.attention_scope"))]
    terms = observed_components(regions, row)
    forward = 1. + sum(terms.values())
    write("predictions", dict(complete=True, refused=0, rows=[dict(chain_id=row["chain_id"],
        chain_step=0, geometry=geometry(descriptor), seconds=dict(body=1., forward=forward,
        prepare=terms["<prepare>"], postprocess=terms["<postprocess>"]))]))
    options = dict(region_overlay="retained.json", region_overlay_sha256="retained", regions="existing")
    oracle = SimpleNamespace(compass_loaded_inputs=inputs, seconds_per_launch=0)
    identity = predictor_identity(oracle, options, pins["predictions"])
    write("identity", identity)
    write("freeze", dict(frozen_before_heldout_warmups=True, source_refitted=False,
                         predictor_identity=pins["identity"], source_model=pins["model"]))
    heldouts = []
    for repetition in range(6):
        heldout = copy.deepcopy(row)
        heldout.update(role="heldout", repetition=repetition,
            seconds=dict(prepare=terms["<prepare>"], postprocess=terms["<postprocess>"], run_model=1., forward=forward))
        heldout["descriptor"]["req_ids"] = [10 + repetition]
        heldouts.append(heldout)
    write("heldout", dict(rows=heldouts))
    write("complete", dict(success=True, engine_closed=True))
    write("closeout", dict(exit_code=0, cleanup=dict(verified=True, writers_released=True),
        collection=dict(copy_complete=True), terminal=dict(unprofiled_control={
            "native/NATIVE_COMPLETE.json": dict(sha256=pins["complete"]["sha256"])})))
    receipt = dict(schema=SCHEMA, passed=True, relative_limit=.10, source_refitted=False,
        predictor_identity=pins["identity"], predictor_freeze=pins["freeze"], heldout=pins["heldout"],
        native_complete=pins["complete"], copy_closeout=pins["closeout"])
    write("receipt", receipt)
    return dict(regions=regions, inputs=inputs, options=options, pins=pins, write=write,
                receipt=receipt, identity=identity, heldouts=heldouts)


def qualify(bundle):
    pin = bundle["pins"]["receipt"]
    return validate(pin["path"], pin["sha256"], regions=bundle["regions"],
                    inputs=bundle["inputs"], options=bundle["options"])


def test_frozen_composition_passes_without_qualifying_raw_sources(bundle):
    verdict, inputs = qualify(bundle)
    assert verdict["passed"] and verdict["independent_prefill_steps"] == 1
    assert bundle["regions"].source_qualified is False
    assert all(item.role.startswith("validation.forward_composition.") for item in inputs)


def test_changed_book_or_scalar_refuses(bundle):
    bundle["options"]["derive"] = False
    with pytest.raises(ValueError, match="complete predictor"):
        qualify(bundle)


def test_changed_input_bytes_refuse(bundle):
    path = bundle["pins"]["heldout"]["path"]
    with open(path, "a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="input changed"):
        qualify(bundle)


@pytest.mark.parametrize("mutation", ["shared-request", "bad-forward", "missing-repeat"])
def test_no_heldout_leakage_omission_or_failed_gate(bundle, mutation):
    rows = bundle["heldouts"]
    if mutation == "shared-request":
        rows[0]["descriptor"]["req_ids"] = [1]
    elif mutation == "missing-repeat":
        rows.pop()
    else:
        for row in rows:
            row["seconds"]["forward"] += 1.
            row["seconds"]["run_model"] += 1.
    bundle["receipt"]["heldout"] = bundle["write"]("heldout", dict(rows=rows))
    bundle["write"]("receipt", bundle["receipt"])
    with pytest.raises(ValueError):
        qualify(bundle)


def test_acceptance_preflight_requires_pinned_composition_then_full_reader(bundle):
    from atom.compass.replay.aiperf_profile import _check_source_options

    options = dict(diagnostic_reference_handoff="original-references.json", diagnostic_only=False)
    with pytest.raises(ValueError, match="composition qualification"):
        _check_source_options({"purpose": "acceptance"}, options)
    pin = bundle["pins"]["receipt"]
    options.update(composition_qualification=pin["path"], composition_qualification_sha256=pin["sha256"])
    _check_source_options({"purpose": "acceptance"}, options)
    assert qualify(bundle)[0]["passed"]
    options["composition_qualification_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        _check_source_options({"purpose": "acceptance"}, options)
