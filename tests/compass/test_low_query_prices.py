"""Bounded source transfer preserves shared controls and failed measurements."""

import hashlib
import json

import pytest

from atom.compass.core.cost.low_query import LowQueryPrices, GEMM, GDN_LAYERS, MHA_LAYERS
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.runtime.microbench import signature_of
from .test_cached_q16_prices import gdn, mha, gather


def make_bundle(directory, scope_sha="a" * 64, *, failed=False):
    directory.mkdir(exist_ok=True)

    def write(name, data):
        path = directory / name
        path.write_text(json.dumps(data, sort_keys=True))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    cases, points, predictions, checks, ops = [], {}, {}, [], {}
    shapes = [(q, 5120, k) for q in range(9, 16) for k in (6144, 17408)]
    shapes += [(m, n, k) for m in (1744, 5456)
               for n, k in ((16480, 5120), (5120, 6144), (34816, 5120),
                            (5120, 17408), (14336, 5120)) if m != 5456 or n != 16480]

    def reference(name, family, op, **coordinates):
        signature = signature_of(op)
        cases.append(dict(cell_id=name, phase="reference", family=family,
                          signature=signature, observed_cache="over" if family == "mha" else "graph",
                          kv_regions=8 if family == "mha" else 1, arg_sets=64, **coordinates))
        seconds = 1e-4 + coordinates.get("cached_prefix", 0) * 1e-9
        points[name] = dict(signature=signature, seconds=seconds, source_qualified=True)
        ops[name] = op

    for q in range(1, 16):
        reference(f"gdn{q}", "gdn", gdn(query=q), q=q)
        for prefix in (32752, 65520):
            reference(f"mha{q}_{prefix}", "mha", mha(prefix + q, query=q), q=q, cached_prefix=prefix)
        op = gather(q - 1)
        op["input_shapes"][0][0] = q
        reference(f"gather{q}", "gather", op, q=q)
    for m, n, k in shapes:
        op = {"name": GEMM, "input_shapes": [[m, k], [n, k]],
              "dtypes": ["bfloat16", "bfloat16"], "layouts": [], "scalars": [],
              "output_shapes": [[m, n]], "output_dtypes": ["bfloat16"]}
        reference(f"gemm{m}_{n}_{k}", "gemm", op, M=m, N=n, K=k)

    def control(name, family, sources, **coordinates):
        cases.append(dict(cell_id=name, phase="heldout", family=family, **coordinates))
        predictions[name] = dict(sources=[dict(reference_cell_id=s) for s in sources], source_qualified=True)
        checks.append(dict(cell_id=name, group=name, error_gate_pass=True,
                           kernel_identity_pass=True, source_qualified=True))

    for q in range(1, 16):
        for layer in GDN_LAYERS[1:] if q == 1 else (32, 62):
            control(f"gdn{q}_layer{layer}", "gdn", [f"gdn{q}"], q=q)
        for layer in (3, *MHA_LAYERS[1:]) if q == 1 else (3, 31, 63):
            control(f"mha{q}_layer{layer}", "mha", [f"mha{q}_32752", f"mha{q}_65520"], q=q)
    for q in (1, 8, 9, 15):
        control(f"gather{q}_relocate", "gather", [f"gather{q}"], q=q)
    for m, n, k in shapes:
        control(f"gemm{m}_{n}_{k}_control", "gemm", [f"gemm{m}_{n}_{k}"], M=m, N=n, K=k)
    selected = {c["cell_id"] for c in cases if c["phase"] == "reference" and
                (c.get("q") == 14 or c.get("M") in (14, 1744, 5456))}
    required = {c["cell_id"] for c in cases if c["phase"] == "heldout" and
                ((c["family"] in ("gdn", "mha") and c["q"] in (1, 14)) or c["family"] == "gather"
                 or c.get("M") in (14, 1744, 5456))}
    assert len(points) == 83 and len(checks) == 160 and len(selected) == 15 and len(required) == 83
    if failed:
        check = next(c for c in checks if c["cell_id"] == "gdn1_layer62")
        readings = [1.06e-4, 1e-4, 1e-4]
        check.update(source_qualified=False, observed=dict(
            source_qualified=False, qualification="unqualified_pending_spread_review",
            seconds=1e-4, all_three=readings, range_over_median=(max(readings) - min(readings)) / 1e-4))
    plan = write("PLAN.json", dict(cases=cases, backend_flags={"use_triton": True}))
    freeze = write("FREEZE.json", dict(plan=plan, heldout_timings_read=False,
                                       reference_points=points, predictions=predictions))
    verdict = write("VERDICT.json", dict(plan=plan, prediction_freeze=freeze,
        source_qualified=False, all_error_gates_pass=True, checks=checks,
        groups=[dict(group=c["group"], **{"pass": True}) for c in checks]))
    families = {"gdn": {"abi": "gdn"}, "mha": {"abi": "mha"}}
    layers = {str(i): "mha" if i % 4 == 3 else "gdn" for i in range(64)}
    preflight = write("PREFLIGHT.json", dict(plan=plan,
        family_abi=dict(flags={"use_triton": True}, families=families,
                        all_layers={key: families[family] for key, family in layers.items()}),
        native=dict(layers={key: dict(family=family) for key, family in layers.items()})))
    entries = []
    for family in ("gdn", "mha", "gather", "gemm"):
        chosen = [c for c in cases if c["cell_id"] in selected and c["family"] == family]
        graph = {"ops": [ops[c["cell_id"]] for c in chosen]}
        prices = {"provenance": {"topology": {"tp": 1}}, "prices": {}, "unpriced": {}}
        for c in chosen:
            prices["prices"][c["signature"]] = dict(name=ops[c["cell_id"]]["name"],
                seconds=points[c["cell_id"]]["seconds"], kernels={}, cache=c["observed_cache"],
                kv_regions=c["kv_regions"], arg_sets=c["arg_sets"])
        entries.append(dict(family=family, cells=[c["cell_id"] for c in chosen],
                            graph=write(f"{family}.graph.json", graph),
                            price=write(f"{family}.price.json", prices)))
    handoff = dict(schema="compass.low_q_reference_export/1", purpose="diagnostic", source_qualified=not failed,
        all_error_gates_pass=True, campaign_source_qualified=False, campaign_all_error_gates_pass=True,
        fit_inputs_are_references_only=True, heldout_timings_used_as_fit_inputs=False,
        scope=dict(model="Qwen/Qwen3.8-27B", topology={"tp": 1}, dtype="bfloat16", num_sequences=1,
                   queries=[14], cached_prefix=[32752, 65520], gdn_layers=list(GDN_LAYERS),
                   mha_layers=list(MHA_LAYERS), gdn_alias="fork", request_scope_sha256=scope_sha,
                   gemm_shapes=[list(s) for s in shapes if s[0] in (14, 1744, 5456)]),
        selected_reference_cells=sorted(selected), required_control_cells=sorted(required),
        failed_spread_controls=["gdn1_layer62"] if failed else [],
        evidence=dict(plan=plan, freeze=freeze, verdict=verdict, preflight=preflight), entries=entries)
    pin = write("HANDOFF.json", handoff)
    return pin["path"], pin["sha256"]


def test_bounded_layer_and_prefix_transfer_does_not_fit_baseline(tmp_path):
    base = PriceLibrary()
    library = LowQueryPrices(base, *make_bundle(tmp_path), deployment_scope_sha256="a" * 64)
    for layer in GDN_LAYERS:
        assert library.lookup(gdn(layer=layer, query=14, read=2, write=3))[0]["seconds"] == 1e-4
    for layer in MHA_LAYERS:
        for prefix in (32752, 49136, 65520):
            price, _ = library.lookup(mha(prefix + 14, layer, 14))
            assert price["seconds"] == pytest.approx(1e-4 + prefix * 1e-9)
    assert not base._prices and len(library.loaded_inputs) == 13
    assert not library.campaign_source_qualified and library.source_qualified


@pytest.mark.parametrize("op,topology", [
    (gdn(query=13), None), (gdn(query=16), None), (gdn(query=14, read=0, write=0), None),
    (gdn(query=14, read=-1), None), (gdn(query=14, layer=3), None),
    (mha(32765, query=14), None), (mha(65550, query=14), None),
    (mha(49150, query=14), {"tp": 2}),
    (dict(mha(49150, query=14), output_dtypes=["float16"]), None),
    (dict(mha(49150, query=14), abi="different"), None),
])
def test_outside_exact_domain_delegates_to_unchanged_base(tmp_path, op, topology):
    base = PriceLibrary()
    library = LowQueryPrices(base, *make_bundle(tmp_path), deployment_scope_sha256="a" * 64)
    assert library.lookup(op, topology) == base.lookup(op, topology)


def test_failed_shared_control_requires_explicit_diagnostic_selection(tmp_path):
    bundle = make_bundle(tmp_path, failed=True)
    with pytest.raises(ValueError):
        LowQueryPrices(PriceLibrary(), *bundle, deployment_scope_sha256="a" * 64)
    with pytest.raises(ValueError, match="diagnostic_only"):
        LowQueryPrices(PriceLibrary(), *bundle, deployment_scope_sha256="a" * 64, allow_failed_spread=True)
    library = LowQueryPrices(PriceLibrary(), *bundle, deployment_scope_sha256="a" * 64,
                             allow_failed_spread=True, diagnostic_only=True)
    assert not library.source_qualified and library.failed_spread_controls == ("gdn1_layer62",)
    assert library.lookup(gdn(query=14))[0]["seconds"] == 1e-4


@pytest.mark.parametrize("damage", ["error", "kernel", "reference", "membership", "undeclared", "scope"])
def test_diagnostic_opt_in_never_bypasses_other_requirements(tmp_path, damage):
    path, _ = make_bundle(tmp_path, failed=True)
    handoff = json.loads((tmp_path / "HANDOFF.json").read_text())
    role = "freeze" if damage == "reference" else "verdict"
    file = tmp_path / ("FREEZE.json" if role == "freeze" else "VERDICT.json")
    data = json.loads(file.read_text())
    if damage in ("error", "kernel"):
        next(c for c in data["checks"] if c["cell_id"] == "gdn1_layer62")[
            "error_gate_pass" if damage == "error" else "kernel_identity_pass"] = False
    elif damage == "reference":
        data["reference_points"]["gdn1"]["source_qualified"] = False
    elif damage == "membership":
        handoff["required_control_cells"].remove("gdn1_layer62")
    elif damage == "undeclared":
        handoff["failed_spread_controls"] = []
    else:
        handoff["scope"]["request_scope_sha256"] = "b" * 64
    file.write_text(json.dumps(data, sort_keys=True))
    handoff["evidence"][role]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    (tmp_path / "HANDOFF.json").write_text(json.dumps(handoff))
    sha = hashlib.sha256((tmp_path / "HANDOFF.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        LowQueryPrices(PriceLibrary(), path, sha, deployment_scope_sha256="a" * 64,
                       allow_failed_spread=True, diagnostic_only=True)
