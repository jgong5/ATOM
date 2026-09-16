"""Whole-group primitive evidence composes without refitting heldout observations."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from atom.compass.core.cache_policy import cache_on_policy
from atom.compass.core.cost.cached_q16 import CachedQ16Prices
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.core.cost.low_query import GEMM
from atom.compass.core.cost.reached_primitive_evidence import EVENTS, Evidence, _phase
from atom.compass.core.cost.reached_primitives import ReachedPrimitivePrices, work_identity
from atom.compass.runtime.microbench import signature_of
from .test_cached_q16_prices import bundle as q16_bundle, gather, gdn, mha
from .test_prepared_plan import add_book, prepared_graph

SCOPE = "a" * 64


class Artifacts:
    """Reseal fixture dependencies so negative tests reach semantic validation."""

    def __init__(self, directory):
        self.directory = directory
        self.data, self.pins = OrderedDict(), {}

    def add(self, name, value):
        self.data[name] = value
        self.pins[name] = {"path": str(self.directory / name), "sha256": ""}
        return self.pins[name]

    def seal(self):
        for name, value in self.data.items():
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True))
            self.pins[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def auxiliary_operators():
    # Operator names/geometry from immutable rpa1 target_240 and target_262.
    # Durations in the campaign fixture remain synthetic protocol test data.
    embedding = dict(name="aten::embedding", input_shapes=[[248320, 5120], [23]],
        dtypes=["bfloat16", "int32"], output_shapes=[[23, 5120]], output_dtypes=["bfloat16"],
        layouts=[], scalars=[], context=[], group=None)
    mrope = dict(name="triton::_mrope_qk_tiled_kernel",
        input_shapes=[[1057, 6144], [1057, 1024], [1057, 6144], [1057, 1024],
                      [3, 1057], [262144, 1, 1, 32], [262144, 1, 1, 32]],
        dtypes=["bfloat16"] * 4 + ["int64", "bfloat16", "bfloat16"],
        output_shapes=[], output_dtypes=[], layouts=[], context=[], group=None,
        launch=[["grid", [67, 28]], ["origin", "atom.model_ops.triton_mrope:_mrope_qk_tiled_kernel"]],
        scalars=[["#7", 6144], ["#8", 1024], ["#9", 6144], ["#10", 1024], ["#11", 1057],
                 ["#12", 32], ["#13", 32], ["#14", 1057], ["#15", 24], ["#16", 4], ["#17", 256],
                 ["#18", 64], ["#19", 32], ["#20", 11], ["#21", 10], ["#22", 16], ["#23", 256],
                 ["num_stages", 1], ["num_warps", 8]])
    return embedding, mrope


def make_domain(store, *, include_auxiliary=False):
    gemm = {"name": GEMM, "input_shapes": [[32, 5120], [6144, 5120]],
            "dtypes": ["bfloat16", "bfloat16"], "layouts": [], "scalars": [],
            "output_shapes": [[32, 6144]], "output_dtypes": ["bfloat16"]}
    specs = [
        ("gemm_ref", "gemm", "reference", gemm, 2., []),
        ("gemm_control", "gemm", "heldout", gemm, 2.04, [("gemm_ref", 1.)]),
        ("gdn_ref", "gdn", "reference", gdn(), 1., []),
        ("gdn_layer1", "gdn", "heldout", gdn(layer=1), 1.01, [("gdn_ref", 1.)]),
        ("gdn_layer2", "gdn", "heldout", gdn(layer=2), .99, [("gdn_ref", 1.)]),
        ("mha_low", "mha", "reference", mha(32768), 2., []),
        ("mha_high", "mha", "reference", mha(98304), 4., []),
        ("mha_middle3", "mha", "heldout", mha(65536), 3.03, [("mha_low", .5), ("mha_high", .5)]),
        ("mha_middle7", "mha", "heldout", mha(65536, layer=7), 2.97, [("mha_low", .5), ("mha_high", .5)]),
        ("gather_ref", "gather", "reference", gather(), 1., []),
        ("gather_failed", "gather", "heldout", gather(), 1.3, [("gather_ref", 1.)]),
    ]
    if include_auxiliary:
        embedding, mrope = auxiliary_operators()
        specs.extend([
            ("embedding_ref", "embedding", "reference", embedding, 1.25, []),
            ("embedding_control", "embedding", "heldout", embedding, 1.2625, [("embedding_ref", 1.)]),
            ("mrope_ref", "mrope", "reference", mrope, .25, []),
            ("mrope_control", "mrope", "heldout", mrope, .2525, [("mrope_ref", 1.)]),
        ])
    cases, ops, durations = [], {}, {}
    for name, family, phase, op, seconds, sources in specs:
        graph = store.add("graphs/" + name + ".json", {"ops": [op]})
        cases.append(dict(cell_id=name, family=family, phase=phase, graph=graph,
            signature=signature_of(op), score_group=family, repetitions=3,
            graph_batch=8, warmup=2, iters=5, only=op["name"], requested_cache="graph",
            observed_cache="over" if family == "mha" else "graph", arg_sets=64,
            kv_regions=8 if family == "mha" else 1,
            frozen_prediction_sources=[dict(reference_cell_id=n, weight=w) for n, w in sources]))
        ops[name], durations[name] = op, seconds
    domain = store.add("DOMAIN.json", {"schema": "compass.reached_primitive_manifest/1", "cases": cases})
    return SimpleNamespace(cases=cases, ops=ops, durations=durations, pin=domain)


def make_campaign(store, domain, label="gpu3", groups=("gemm", "gdn", "mha", "gather"), selected=None):
    def add(name, value):
        return store.add(label + "/" + name, value)

    cases = [case for case in domain.cases if case["score_group"] in groups]
    orders = {phase + "_order_by_repeat": [[case["cell_id"] for case in cases if case["phase"] == phase]
                                          for _ in range(3)] for phase in ("reference", "heldout")}
    manifest = add("MANIFEST.json", dict(schema="compass.reached_primitive_manifest/1", cases=cases, **orders))
    collector = add("collector.py", "fixture collector bytes")
    flags = {"use_triton": True}
    policy = cache_on_policy()
    plan_data = dict(schema="compass.reached_primitive_executable/2", design=manifest, cases=cases,
        code={"collector": collector}, engine_args=dict(model="Qwen/Qwen3.8-27B", tensor_parallel_size=1,
            pipeline_parallel_size=1, kv_cache_dtype="bf16"), backend_flags=flags, cache_policy=policy,
        dispatch={"case_to_probe": {case["cell_id"]: case["cell_id"] for case in cases},
                  "probes": [{"cell_id": case["cell_id"]} for case in cases]},
        heldout_order_sha256=hashlib.sha256(json.dumps(orders["heldout_order_by_repeat"], sort_keys=True).encode()).hexdigest(),
        **orders)
    plan = add("PLAN.json", plan_data)
    runtime = dict(physical_uuid="factual-" + label, backend_flags=flags)
    profiles = {case["cell_id"]: [["same_" + case["family"] + "_kernel", 8]] for case in cases}
    dispatch = add("DISPATCH.json", dict(plan=plan, complete=True, dispatch_qualified=True,
        timings_used_as_prices=False, runtime_identity=runtime, inputs=[],
        cells={case["cell_id"]: dict(graph=case["graph"], kernel_profile=profiles[case["cell_id"]]) for case in cases}))
    families = {"gdn": {"abi": "gdn"}, "mha": {"abi": "mha"}}
    layers = {str(i): "mha" if i % 4 == 3 else "gdn" for i in range(64)}
    phases, preflights, points, raw_pins = {}, {}, {}, {}
    for phase in ("reference", "heldout"):
        preflights[phase] = add(phase + "/PREFLIGHT.json", dict(plan=plan, phase=phase, collector=collector,
            profiler_tool_loaded=False, dispatch=dispatch, runtime_identity=dict(runtime),
            boundary=dict(policy=policy, quiescence={"idle": True}),
            family_abi=dict(flags=flags, families=families,
                all_layers={key: families[family] for key, family in layers.items()}),
            native=dict(layers={key: dict(family=family) for key, family in layers.items()})))
        records = []
        for repeat in range(1, 4):
            for ordinal, case in enumerate(case for case in cases if case["phase"] == phase):
                name, seconds = case["cell_id"], domain.durations[case["cell_id"]]
                raw = add(f"{phase}/{name}.r{repeat}.json", dict(unpriced={}, provenance=dict(
                    topology={"tp": 1}, observed_group_width=1, cache="graph", iters=case["iters"],
                    graph=case["graph"]["path"], only=case["only"]), prices={case["signature"]: dict(
                        name=domain.ops[name]["name"], seconds=seconds, kernels={},
                        cache=case["observed_cache"], arg_sets=64, kv_regions=case["kv_regions"])}))
                raw_pins.setdefault(name, []).append(raw)
                records.append(dict(cell_id=name, repeat=repeat, raw=raw, profiled=False, errors=[],
                    seed=(314159 if phase == "reference" else 271828) + repeat * 1000 + ordinal,
                    treatment={field: case[field] for field in ("graph_batch", "warmup", "iters", "requested_cache",
                        "observed_cache", "arg_sets", "kv_regions")},
                    settings=dict(GRAPH_BATCH=8, KV_VARIANTS=8, REPLAY_INT_VALUES=False,
                        SYNTH_INT_RANGES=True, PRICE_KERNELS=False, PROFILE_MATCH="")))
                points[name] = dict(seconds=seconds, all_three=[seconds] * 3, range_over_median=0.,
                    source_qualified=True, signature=case["signature"],
                    kernel_profiles=[profiles[name]] * 3 if case["family"] == "gemm" else [[], [], []])
        phases[phase] = add(phase + "/PHASE_RESULT.json", dict(plan=plan, phase=phase, complete=True,
            collector=collector, target_timings_used=False, profiler_tool_loaded=False, source_only=True,
            candidate_activated=False, dispatch=dispatch, records=records))
    predictions = {}
    for case in cases:
        if case["phase"] != "heldout":
            continue
        weights = case["frozen_prediction_sources"]
        predictions[case["cell_id"]] = dict(signature=case["signature"], family=case["family"],
            seconds=sum(item["weight"] * points[item["reference_cell_id"]]["seconds"] for item in weights),
            sources=weights, source_qualified=True, group=case["score_group"],
            limit=.121 if case["family"] == "gather" else .10,
            kernel_profiles=points[weights[0]["reference_cell_id"]]["kernel_profiles"] if case["family"] == "gemm" else [])
    freeze = add("FREEZE.json", dict(schema="compass.low_q_reference_freeze/1", plan=plan,
        source_qualified=True, heldout_timings_read=False, target_timings_used=False, candidate_activated=False,
        reference_evidence=dict(phase_result=phases["reference"], dispatch=dispatch,
            raw_prices=[r["raw"] for r in store.data[label + "/reference/PHASE_RESULT.json"]["records"]]),
        reference_points={case["cell_id"]: points[case["cell_id"]] for case in cases if case["phase"] == "reference"},
        predictions=predictions, reuse={}))
    checks, totals = [], []
    for case in cases:
        if case["phase"] != "heldout":
            continue
        name = case["cell_id"]
        prediction, actual = predictions[name], points[name]
        error = abs(actual["seconds"] - prediction["seconds"]) / prediction["seconds"]
        checks.append(dict(cell_id=name, signature=case["signature"], prediction=prediction["seconds"],
            observed=actual, relative_error=error, limit=prediction["limit"], error_gate_pass=error <= prediction["limit"],
            kernel_identity_pass=True, source_qualified=True, family=case["family"], group=case["score_group"]))
    for group in groups:
        members = [row for row in checks if row["group"] == group]
        predicted = sum(row["prediction"] for row in members)
        observed = sum(row["observed"]["seconds"] for row in members)
        error = abs(observed - predicted) / predicted
        totals.append(dict(group=group, predicted_seconds=predicted, observed_seconds=observed,
            relative_error=error, **{"pass": error <= members[0]["limit"]}))
    verdict = add("VERDICT.json", dict(schema="compass.low_q_source_verdict/1", plan=plan,
        prediction_freeze=freeze, source_qualified="gather" not in groups, checks=checks, groups=totals,
        target_timings_used=False, candidate_activated=False,
        heldout_evidence=dict(phase_result=phases["heldout"], dispatch=dispatch,
            raw_prices=[r["raw"] for r in store.data[label + "/heldout/PHASE_RESULT.json"]["records"]])))
    execution = add("EXECUTION.json", dict(science_plan=plan))
    owner = dict(writers_released=True)
    copy = dict(copy_complete=True, owned_writers_released_before_collection=True)
    terminal = add("EXIT.json", dict(plan_sha256="", exit_code=1 if "gather" in groups else 0,
                                    cleanup=owner, collection=copy))
    owner_pin, copy_pin = add("OWNER.json", owner), add("COPY.json", copy)
    failure = add("ORIGINAL_FAILURE.json", dict(source_qualified=False, original_campaign_failed=True))
    payloads = [dict(phase=phases["reference"]), dict(reference_phase=phases["reference"], freeze=freeze),
        dict(freeze=freeze, heldout_order_sha256=plan_data["heldout_order_sha256"]), {},
        dict(phase=phases["heldout"]), dict(verdict=verdict)]
    events = []
    for index, event in enumerate(EVENTS, 1):
        if index == 4:
            payloads[index - 1]["release"] = events[2]
        events.append(add(f"EVENT{index}.json", dict(schema="compass.low_q_boundary_event/1", event=event,
            index=index, plan=plan, previous=events[-1] if events else None, payload=payloads[index - 1])))
    selected = list(selected if selected is not None else (group for group in groups if group != "gather"))
    needed = {item["reference_cell_id"] for case in cases if case["phase"] == "heldout" and
              case["score_group"] in selected for item in case["frozen_prediction_sources"]}
    handoff = add("HANDOFF.json", dict(schema="compass.reached_primitive_reference_export/1",
        source_qualified=True, fit_inputs_are_references_only=True, heldout_timings_used_as_fit_inputs=False,
        candidate_activated=False, selected_groups=selected,
        scope=dict(model="Qwen/Qwen3.8-27B", topology={"tp": 1}, dtype="bfloat16", request_scope_sha256=SCOPE),
        entries=[dict(reference_cell_id=case["cell_id"], graph=case["graph"], price=raw_pins[case["cell_id"]][1])
                 for case in cases if case["cell_id"] in needed], boundary_events=events,
        evidence=dict(domain_manifest=domain.pin, plan=plan, manifest=manifest, freeze=freeze, verdict=verdict,
            reference_plan=plan, reference_phase=phases["reference"], reference_preflight=preflights["reference"],
            heldout_phase=phases["heldout"], heldout_preflight=preflights["heldout"], dispatch=dispatch,
            execution_plan=execution, terminal=terminal, owner_closeout=owner_pin, copy_closeout=copy_pin,
            original_failure=failure)))
    store.seal()
    store.data[label + "/EXIT.json"]["plan_sha256"] = execution["sha256"]
    store.seal()
    return handoff


def full_layer_labels(inventory):
    # Label forms and 48 GDN / 16 MHA inventory from the actual Q3056
    # reference PREFLIGHT. ABI payloads remain the protocol fixture values.
    return {f"language_model.model.layers.{index}."
            f"{'self_attn' if int(index) % 4 == 3 else 'linear_attn'}": value
            for index, value in inventory.items()}


@pytest.mark.parametrize("abi_full,native_full", [(True, True), (True, False), (False, True)])
def test_actual_preflight_layer_label_forms_reach_reference_precision(tmp_path, abi_full, native_full):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    make_campaign(store, domain)
    preflight = store.data["gpu3/reference/PREFLIGHT.json"]
    if abi_full:
        preflight["family_abi"]["all_layers"] = full_layer_labels(preflight["family_abi"]["all_layers"])
    if native_full:
        preflight["native"]["layers"] = full_layer_labels(preflight["native"]["layers"])
    # Preserve the observed target_096 triplet's precision outcome. No freeze
    # is consumed here: this exercises the actual reference-phase reader.
    values = [0.0031671087741851804, 0.002965928316116333, 0.0029762182235717775]
    for repeat, value in enumerate(values, 1):
        raw = store.data[f"gpu3/reference/gemm_ref.r{repeat}.json"]
        next(iter(raw["prices"].values()))["seconds"] = value
    store.seal()
    pins = store.data["gpu3/HANDOFF.json"]["evidence"]
    points, samples, _, _ = _phase(Evidence(tmp_path, 0), "reference", store.data["gpu3/PLAN.json"],
        pins["plan"], pins["reference_phase"], pins["reference_preflight"])
    assert points["gemm_ref"]["all_three"] == values
    assert points["gemm_ref"]["range_over_median"] == pytest.approx(0.06759600370547081)
    assert points["gemm_ref"]["source_qualified"] is False
    assert len(samples["gemm_ref"]) == 3


@pytest.mark.parametrize("inventory", ["abi", "native"])
@pytest.mark.parametrize("damage", ["missing", "duplicate", "ambiguous", "out_of_range"])
def test_layer_inventories_reject_missing_duplicate_or_ambiguous_indices(tmp_path, inventory, damage):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    handoff = make_campaign(store, domain)
    preflight = store.data["gpu3/reference/PREFLIGHT.json"]
    entries = preflight["family_abi"]["all_layers"] if inventory == "abi" else preflight["native"]["layers"]
    if damage == "missing":
        del entries["63"]
    elif damage == "duplicate":
        del entries["63"]  # Keep 64 entries while duplicating layer zero.
        entries["language_model.model.layers.0.linear_attn"] = deepcopy(entries["0"])
    elif damage == "ambiguous":
        entries["model.layers.0.sub.1.linear_attn"] = entries.pop("0")
    else:
        entries["64"] = entries.pop("63")
    store.seal()
    with pytest.raises(ValueError, match="layer"):
        ReachedPrimitivePrices(PriceLibrary(), [handoff], deployment_scope_sha256=SCOPE)


@pytest.mark.parametrize("damage", ["family", "abi"])
def test_full_layer_labels_preserve_family_and_abi_checks(tmp_path, damage):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    handoff = make_campaign(store, domain)
    preflight = store.data["gpu3/reference/PREFLIGHT.json"]
    preflight["family_abi"]["all_layers"] = full_layer_labels(preflight["family_abi"]["all_layers"])
    preflight["native"]["layers"] = full_layer_labels(preflight["native"]["layers"])
    label = "language_model.model.layers.0.linear_attn"
    if damage == "family":
        preflight["native"]["layers"][label]["family"] = "mha"
    else:
        preflight["family_abi"]["all_layers"][label] = {"abi": "changed"}
    store.seal()
    with pytest.raises(ValueError, match="not homogeneous"):
        ReachedPrimitivePrices(PriceLibrary(), [handoff], deployment_scope_sha256=SCOPE)


@pytest.fixture
def campaign(tmp_path):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    pin = make_campaign(store, domain)
    return store, domain, pin


def load(pin, base=None):
    return ReachedPrimitivePrices(base or PriceLibrary(), [pin], deployment_scope_sha256=SCOPE)


def test_whole_groups_use_frozen_prices_and_preserve_original_failure(campaign):
    _, domain, pin = campaign
    library = load(pin)
    assert library.selected_groups == ("gdn", "gemm", "mha")
    assert library.lookup(domain.ops["gemm_control"])[0]["seconds"] == 2.
    assert library.lookup(gdn(layer=62, read=4, write=8))[0]["seconds"] == 1.
    price = library.lookup(mha(65536, layer=63))[0]
    assert price["seconds"] == 3. and price["interpolated"] is True
    assert price["validation_campaign_physical_uuid"] == "factual-gpu3"
    assert library.lookup(gather())[0] is None
    assert library.campaigns[0]["terminal_exit_code"] == 1
    assert library.campaigns[0]["whole_campaign_requalified"] is False
    assert library.lookup(mha(32768))[0] is None  # A reference alone is not active.


def conditioned_campaign(tmp_path, *, n=5120, k=6144, m=3056):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    op = domain.ops["gemm_ref"]
    op.update(input_shapes=[[m, k], [n, k]], output_shapes=[[m, n]])
    for case in domain.cases:
        if case["family"] == "gemm":
            case.update(signature=signature_of(op), graph_batch=32, warmup=20, iters=256)
    pin = make_campaign(store, domain, groups=("gemm",))
    policy = dict(schema="compass.q3056_fixed_conditioning/1", eager_warmup_calls=20,
        base_graph_calls=32, conditioning_base_replays=13, conditioning_operator_calls=416,
        timed_base_replays=8, timed_operator_calls=256, fixed_idle_seconds=0, adaptive=False, clock_reads=0)
    for name in ("gpu3/PLAN.json", "gpu3/MANIFEST.json"):
        store.data[name]["conditioning_policy"] = dict(policy)
    for phase in ("reference", "heldout"):
        for row in store.data[f"gpu3/{phase}/PHASE_RESULT.json"]["records"]:
            row["settings"]["GRAPH_BATCH"] = 32
            row["treatment"]["conditioning_policy"] = dict(policy)
            name = f"gpu3/{phase}/{row['cell_id']}.r{row['repeat']}.json"
            store.data[name]["provenance"]["conditioning"] = dict(policy=dict(policy), timer_invocations=1, completed=True)
    return store, domain, pin


def reseal_conditioned_campaign(store):
    store.seal()
    store.data["gpu3/EXIT.json"]["plan_sha256"] = store.pins["gpu3/EXECUTION.json"]["sha256"]
    store.seal()


@pytest.mark.parametrize("n,k", [(5120,17408), (14336,5120), (16480,5120), (34816,5120), (5120,6144)])
def test_declared_conditioning_uses_geometry_without_campaign_ids_or_counts(tmp_path, n, k):
    store, domain, pin = conditioned_campaign(tmp_path, n=n, k=k)
    reseal_conditioned_campaign(store)
    library = load(pin)
    assert library.selected_groups == ("gemm",)
    assert library.lookup(domain.ops["gemm_control"])[0]["seconds"] == 2.


@pytest.mark.parametrize("damage", ["missing", "wrong", "incomplete", "plan", "manifest", "row", "geometry"])
def test_conditioning_requires_matching_declarations_geometry_and_completed_receipt(tmp_path, damage):
    store, _, pin = conditioned_campaign(tmp_path, m=464 if damage == "geometry" else 3056)
    raw = store.data["gpu3/heldout/gemm_control.r1.json"]
    if damage == "missing":
        del raw["provenance"]["conditioning"]
    elif damage == "wrong":
        raw["provenance"]["conditioning"]["policy"]["conditioning_base_replays"] = 1
    elif damage == "incomplete":
        raw["provenance"]["conditioning"]["completed"] = False
    elif damage in ("plan", "manifest"):
        del store.data[f"gpu3/{damage.upper()}.json"]["conditioning_policy"]
    elif damage == "row":
        del store.data["gpu3/heldout/PHASE_RESULT.json"]["records"][0]["treatment"]["conditioning_policy"]
    reseal_conditioned_campaign(store)
    with pytest.raises(ValueError, match="conditioning"):
        load(pin)


@pytest.mark.parametrize("selected", [("gemm",), ("embedding", "mrope")])
def test_full_domain_keeps_auxiliary_groups_distinct_from_selected_prices(tmp_path, selected):
    store = Artifacts(tmp_path)
    domain = make_domain(store, include_auxiliary=True)
    pin = make_campaign(store, domain, groups=("gemm", "gdn", "mha", "gather", "embedding", "mrope"),
                        selected=selected)
    base = PriceLibrary()
    embedding, mrope = auxiliary_operators()
    add_book(base, tmp_path, "legacy_auxiliary", [embedding, mrope], [9., 9.])
    library = load(pin, base)
    assert library.selected_groups == tuple(sorted(selected))
    if selected == ("gemm",):
        assert library.lookup(domain.ops["gemm_control"])[0]["seconds"] == 2.
        assert library.lookup(embedding)[0] is None
        assert library.lookup(mrope)[0] is None
    else:
        assert library.lookup(embedding)[0]["seconds"] == 1.25
        assert library.lookup(mrope)[0]["seconds"] == .25
        assert library.lookup(domain.ops["gemm_control"])[0] is None
        body, coverage, _ = library.body(prepared_graph([embedding, mrope]))
        assert body == 1.5 and coverage.complete


def test_two_disjoint_campaigns_keep_their_own_physical_uuid(tmp_path):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    first = make_campaign(store, domain, "gpu3", ("gemm", "gdn"))
    second = make_campaign(store, domain, "gpu2", ("mha", "gather"))
    library = ReachedPrimitivePrices(PriceLibrary(), json.dumps([first, second]), deployment_scope_sha256=SCOPE)
    assert library.lookup(gdn())[0]["validation_campaign_physical_uuid"] == "factual-gpu3"
    assert library.lookup(mha())[0]["validation_campaign_physical_uuid"] == "factual-gpu2"
    assert library.lookup(gather())[0] is None
    with pytest.raises(ValueError, match="overlap"):
        ReachedPrimitivePrices(PriceLibrary(), [first, first], deployment_scope_sha256=SCOPE)


def test_independent_partial_exports_have_no_fixed_campaign_count(tmp_path):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    pins = [make_campaign(store, domain, label=group, groups=(group,)) for group in ("gemm", "gdn", "mha")]
    library = ReachedPrimitivePrices(PriceLibrary(), pins, deployment_scope_sha256=SCOPE)
    assert library.selected_groups == ("gdn", "gemm", "mha")
    with pytest.raises(ValueError, match="nonempty"):
        ReachedPrimitivePrices(PriceLibrary(), [], deployment_scope_sha256=SCOPE)


def test_body_and_prepared_lookup_preserve_unrelated_legacy_prices(campaign, tmp_path):
    _, domain, pin = campaign
    ordinary = {"name": "ordinary", "input_shapes": [], "dtypes": [], "scalars": []}
    base = PriceLibrary()
    add_book(base, tmp_path, "legacy", [ordinary, gather()], [5., 7.])
    library = load(pin, base)
    graph = prepared_graph([ordinary, domain.ops["gemm_control"], gdn(layer=2), mha(layer=7)])
    result = library.body(dict(graph))
    assert result[0] == 11. and result[1].complete
    assert library.body(graph) == result == library.body(graph)
    assert library._prepared_plan_prices
    assert library.lookup(gather())[0] is None  # Failed recognized work cannot escape into legacy.
    assert not library.body(prepared_graph([gather()]))[1].complete


def test_joint_alias_refuses_generic_prices_but_keeps_qualified_q16(campaign, tmp_path):
    _, _, pin = campaign
    base = PriceLibrary()
    add_book(base, tmp_path, "unsafe_alias", [gdn()], [7.])
    assert base.lookup(gdn(read=0, write=0))[0]["seconds"] == 7.
    assert load(pin, base).lookup(gdn(read=0, write=0))[0] is None
    q16_path = tmp_path / "q16"
    q16_path.mkdir()
    qualified = CachedQ16Prices(base, *q16_bundle.__wrapped__(q16_path))
    library = load(pin, qualified)
    for read, write in ((0, 0), (5, 5)):
        assert library.lookup(gdn(read=read, write=write)) == qualified.lookup(gdn(read=read, write=write))
    assert library.lookup(gdn(read=5, write=6))[0]["seconds"] == 1.
    for read, write in ((32, 32), (-2, -2), (True, True)):
        assert library.lookup(gdn(read=read, write=write))[0] is None


@pytest.mark.parametrize("damage", ["missing-heldout", "failed-selected", "partial-group", "weights", "refit", "median",
    "treatment", "seed", "uuid", "no-verdict", "open-boundary", "source-bytes", "graph-bytes", "copy", "scope"])
def test_incomplete_or_changed_evidence_cannot_activate(campaign, damage):
    store, _, pin = campaign
    data = store.data
    handoff = data["gpu3/HANDOFF.json"]
    if damage == "missing-heldout":
        data["gpu3/heldout/PHASE_RESULT.json"]["complete"] = False
    elif damage == "failed-selected":
        handoff["selected_groups"].append("gather")
    elif damage == "partial-group":
        # The common domain still requires both GDN layer controls.
        for label in ("gpu3/PLAN.json", "gpu3/MANIFEST.json"):
            data[label]["cases"] = [case for case in data[label]["cases"] if case["cell_id"] != "gdn_layer2"]
    elif damage == "weights":
        data["gpu3/FREEZE.json"]["predictions"]["mha_middle3"]["sources"] = [
            dict(reference_cell_id="mha_low", weight=.25), dict(reference_cell_id="mha_high", weight=.75)]
    elif damage == "refit":
        data["gpu3/FREEZE.json"]["predictions"]["mha_middle3"]["seconds"] = 3.03
    elif damage == "median":
        data["gpu3/FREEZE.json"]["reference_points"]["gemm_ref"]["seconds"] = 2.04
    elif damage in ("treatment", "seed"):
        row = data["gpu3/heldout/PHASE_RESULT.json"]["records"][0]
        if damage == "seed":
            row["seed"] += 1
        else:
            row["settings"]["PRICE_KERNELS"] = True
    elif damage == "uuid":
        data["gpu3/heldout/PREFLIGHT.json"]["runtime_identity"]["physical_uuid"] = "another-device"
    elif damage == "no-verdict":
        del handoff["evidence"]["verdict"]
    elif damage == "open-boundary":
        handoff["boundary_events"].pop()
    elif damage in ("source-bytes", "graph-bytes"):
        target = handoff["entries"][0]["price" if damage == "source-bytes" else "graph"]["path"]
        from pathlib import Path
        with Path(target).open("a") as stream:
            stream.write(" ")
        with pytest.raises(ValueError, match="evidence changed"):
            load(pin)
        return
    elif damage == "copy":
        data["gpu3/COPY.json"]["copy_complete"] = False
    elif damage == "scope":
        handoff["scope"]["request_scope_sha256"] = "wrong-scope"
    store.seal()
    # Keep execution binding valid when PLAN changes, so it does not hide later gates.
    data["gpu3/EXIT.json"]["plan_sha256"] = store.pins["gpu3/EXECUTION.json"]["sha256"]
    store.seal()
    with pytest.raises(ValueError):
        load(pin)


def test_joint_alias_preserves_relationships_but_not_absolute_addresses():
    assert work_identity(gdn(read=0, write=1))[0] == work_identity(gdn(read=8, write=9))[0]
    assert work_identity(gdn(read=0, write=0))[0] != work_identity(gdn(read=0, write=1))[0]


def test_complete_subset_still_refuses_a_partial_original_group(tmp_path):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    subset = SimpleNamespace(**vars(domain))
    subset.cases = [case for case in domain.cases if case["cell_id"] != "gdn_layer2"]
    pin = make_campaign(store, subset, groups=("gdn",))
    with pytest.raises(ValueError, match="only part of a score group"):
        load(pin)


@pytest.mark.parametrize("damage", ["spread", "kernel"])
def test_independent_failed_control_disqualifies_its_whole_group(campaign, damage):
    store, _, pin = campaign
    data = store.data
    name = "gdn_layer1" if damage == "spread" else "gemm_control"
    check = next(row for row in data["gpu3/VERDICT.json"]["checks"] if row["cell_id"] == name)
    if damage == "spread":
        values = [.97, 1.01, 1.05]
        for repeat, seconds in enumerate(values, 1):
            price = data[f"gpu3/heldout/{name}.r{repeat}.json"]["prices"][check["signature"]]
            price["seconds"] = seconds
        check["observed"].update(all_three=values, range_over_median=(1.05 - .97) / 1.01, source_qualified=False)
        check["source_qualified"] = False
    else:
        changed = [["different_gemm_dispatch", 8]]
        data["gpu3/DISPATCH.json"]["cells"][name]["kernel_profile"] = changed
        check["observed"]["kernel_profiles"] = [changed] * 3
        check["kernel_identity_pass"] = False
    store.seal()
    with pytest.raises(ValueError, match="selects a failed or incomplete heldout group"):
        load(pin)


@pytest.mark.parametrize("mode", ["same_device", "cross_device", "unexported", "software", "refit", "dispatch_count"])
def test_continuation_reuses_original_reference_phase_and_original_seeds(tmp_path, mode):
    store = Artifacts(tmp_path)
    domain = make_domain(store)
    parent = make_campaign(store, domain)
    pin = make_campaign(store, domain, "subset", groups=("gdn", "mha"))
    data = store.data
    plan = data["subset/PLAN.json"]
    handoff = data["subset/HANDOFF.json"]
    parent_evidence = data["gpu3/HANDOFF.json"]["evidence"]
    for role in ("reference_plan", "reference_phase", "reference_preflight"):
        handoff["evidence"][role] = parent_evidence[role]
    freeze = data["subset/FREEZE.json"]
    freeze.update(original_plan=parent_evidence["plan"], whole_campaign_source_qualified=False)
    needed = set(freeze["reference_points"])
    freeze["reference_evidence"] = dict(phase_result=parent_evidence["reference_phase"],
        dispatch=parent_evidence["dispatch"], raw_prices=[row["raw"] for row in
            data["gpu3/reference/PHASE_RESULT.json"]["records"] if row["cell_id"] in needed])
    parent_entries = {entry["reference_cell_id"]: entry for entry in data["gpu3/HANDOFF.json"]["entries"]}
    handoff["entries"] = [parent_entries[name] for name in sorted(needed)]
    data["subset/EVENT1.json"]["payload"].update(reused=True, original_plan=parent_evidence["plan"],
        phase=parent_evidence["reference_phase"])
    data["subset/EVENT2.json"]["payload"]["reference_phase"] = parent_evidence["reference_phase"]
    ordinals = {str(repeat): {name: data["gpu3/PLAN.json"]["heldout_order_by_repeat"][repeat - 1].index(name)
        for name in order} for repeat, order in enumerate(plan["heldout_order_by_repeat"], 1)}
    plan["continuation"] = dict(original_plan=parent_evidence["plan"], seed_ordinals=ordinals)
    for row in data["subset/heldout/PHASE_RESULT.json"]["records"]:
        row["seed"] = 271828 + row["repeat"] * 1000 + ordinals[str(row["repeat"])][row["cell_id"]]
    runtime = data["gpu3/DISPATCH.json"]["runtime_identity"]
    data["subset/DISPATCH.json"]["runtime_identity"] = runtime
    data["subset/heldout/PREFLIGHT.json"]["runtime_identity"] = runtime
    store.seal()
    data["subset/EXIT.json"]["plan_sha256"] = store.pins["subset/EXECUTION.json"]["sha256"]
    store.seal()
    if mode != "same_device":
        previous_plan = store.add("previous/PLAN.json", deepcopy(plan))
        previous_freeze = store.add("previous/FREEZE.json", dict(deepcopy(freeze), plan=previous_plan))
        validation_runtime = dict(runtime, physical_uuid="factual-gpu2")
        physical = store.add("gpu2/IDENTITY.json", {"gpu_index": 2})
        prior_process = store.add("gpu2/PRIOR_PROCESS.json", {"runtime_identity": dict(validation_runtime)})
        assessment = store.add("gpu2/ASSESSMENT.json", {"assessment_only": True})
        contract = dict(schema="compass.historical_reference_validation/1", mode="historical_references_on_new_device",
            original_reference_plan=parent_evidence["plan"], original_prepared_plan=previous_plan,
            original_prepared_freeze=previous_freeze, validation_gpu_identity=physical,
            validation_runtime_evidence=prior_process, assessment=assessment,
            reference_runtime_identity=runtime, validation_runtime_identity=validation_runtime,
            reference_timing_calls=0, source_values_changed=False, device_correction_fitted=False,
            dispatch_calls=len(plan["dispatch"]["probes"]),
            heldout_timing_calls=sum(len(order) for order in plan["heldout_order_by_repeat"]))
        contract_pin = store.add("gpu2/CONTRACT.json", contract)
        plan["continuation"]["cross_device_validation"] = contract_pin
        plan["gpu_identity"] = physical
        freeze["cross_device_validation"] = contract_pin
        handoff["cross_device_validation"] = contract_pin
        data["subset/DISPATCH.json"]["runtime_identity"] = validation_runtime
        data["subset/heldout/PREFLIGHT.json"]["runtime_identity"] = validation_runtime
        if mode == "unexported":
            del handoff["cross_device_validation"]
        elif mode == "software":
            validation_runtime["torch_version"] = "different-build"
        elif mode == "refit":
            data["previous/FREEZE.json"]["predictions"]["gdn_layer1"]["seconds"] = 1.01
        elif mode == "dispatch_count":
            contract["dispatch_calls"] += 1
        store.seal()
        store.seal()  # Forward pins to the immutable historical snapshots are now resolved.
        data["subset/EXIT.json"]["plan_sha256"] = store.pins["subset/EXECUTION.json"]["sha256"]
        store.seal()
        if mode not in ("same_device", "cross_device"):
            with pytest.raises(ValueError, match="cross-device"):
                load(pin)
            return
    library = load(pin)
    assert library.selected_groups == ("gdn", "mha")
    assert library.campaigns[0]["reference_plan"] == parent_evidence["plan"]
    assert library.campaigns[0]["handoff"] != parent
    assert library.lookup(gdn())[0]["seconds"] == 1.
    assert library.lookup(domain.ops["gemm_control"])[0] is None
    if mode == "cross_device":
        assert library.campaigns[0]["historical_reference_physical_uuid"] == "factual-gpu3"
        assert library.campaigns[0]["validation"]["validation_physical_uuid"] == "factual-gpu2"
        assert library.lookup(gdn())[0]["validation_campaign_physical_uuid"] == "factual-gpu2"


@pytest.mark.parametrize("damage", [None, "missing_read", "unregistered", "aggregate", "unconfigured", "changed_scope"])
def test_factory_api_and_opening_validate_every_loaded_evidence(campaign, tmp_path, damage):
    from pathlib import Path
    from atom.compass.core.loaded_input import file_digests, manifest
    from atom.compass.runtime import cache_region_oracle as runtime
    from .test_native_prefill_provenance import grouped_inputs
    from .test_opening_harness import opening, validate, wrapper_evidence

    store, _, pin = campaign
    wrapper_path = tmp_path / "wrapper"
    wrapper_path.mkdir()
    compass, _ = wrapper_evidence.__wrapped__(wrapper_path)
    options = compass["oracle_options"]
    options.update(include_failed_outputless=False, include_failed_final=False, diagnostic_only=False)
    scope_sha = hashlib.sha256(Path(options["attention_scope"]).read_bytes()).hexdigest()
    store.data["gpu3/HANDOFF.json"]["scope"]["request_scope_sha256"] = scope_sha
    store.seal()
    options["reached_primitive_handoffs"] = json.dumps([pin])
    oracle = runtime.source_cost_oracle(**options)
    rank = manifest(oracle.compass_loaded_inputs)
    rank["regions"] = oracle.compass_region_snapshot
    compass["loaded_inputs"]["ranks"] = [rank]
    grouped = grouped_inputs(rank)
    expected = file_digests([row for row in rank["inputs"] if row["role"].startswith("oracle.reached_primitives.")])
    assert grouped["reached_primitive_handoffs"] == expected
    digests = {key: next(iter(value.values())) if len(value) == 1 else validate._rolled_digest(value)
               for key, value in grouped.items()}
    compass.update(oracle_option_files=grouped, oracle_option_sha256=digests)
    provenance = dict(kind="source_calibration", measured_at_tp=1, from_target_engine=False,
        sources=[dict(path="isolated-source", sha256="8" * 64)], code={"collector.py": "9" * 64})
    artifacts = [dict(provenance, sha256=row["sha256"], contents={Path(row["path"]).name: row["sha256"]})
                 for row in rank["inputs"]]
    by_digest = {}
    for row in rank["inputs"]:
        if row["role"].startswith("oracle.reached_primitives."):
            by_digest.setdefault(row["sha256"], []).append(row)
    for sha, rows in by_digest.items():
        contents = file_digests(rows)
        digest = sha if len(contents) == 1 else validate._rolled_digest(contents)
        artifacts.append(dict(provenance, sha256=digest, contents=contents))
    artifacts += [dict(provenance, sha256=digests[key], contents=value) for key, value in grouped.items()]
    artifacts.append(dict(provenance, kind="region_model", sha256=rank["regions"]["sha256"]))
    role = "oracle.reached_primitives.0.verdict"
    if damage == "missing_read":
        rank["inputs"] = [row for row in rank["inputs"] if row["role"] != role]
    elif damage == "unregistered":
        digest = next(row["sha256"] for row in rank["inputs"] if row["role"] == role)
        artifacts = [row for row in artifacts if row["sha256"] != digest]
    elif damage == "aggregate":
        grouped["reached_primitive_handoffs"].pop(next(iter(expected)))
    elif damage == "unconfigured":
        options.pop("reached_primitive_handoffs")
    elif damage == "changed_scope":
        next(row for row in rank["inputs"] if row["role"] == "oracle.attention_scope")["sha256"] = "f" * 64
    modelled = SimpleNamespace(manifest={"server": {"compass": compass, "tensor_parallel_size": 1}})
    bad, notes = opening.check_source_contract(modelled, {"artifacts": artifacts}, "7" * 64, {}, "reached fixture")
    if damage is None:
        assert bad == [] and notes == []
    else:
        assert bad, damage
