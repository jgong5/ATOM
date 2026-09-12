"""The twenty-four cells' configuration, against the prose that explains it.

`SOURCE_ONLY_SERVING.md` is where the option set was written down first, in
shell, for a person to copy. `cc_traces_registry.py` is the same set as data,
for the plan to read. Two copies of a configuration drift, and the way this one
would drift is the expensive way: an option dropped from the registry and still
in the document reads as configured and is not.

So the shared options are held against the document itself. The per-width ones
cannot be -- the document writes them with shell variables a reader expands --
so what is checked there is the key set and the properties that matter.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "atom" / "compass" / "SOURCE_ONLY_SERVING.md"


def _load(name: str):
    """A sibling script as a module, the way the other compass tests load them."""
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


registry = _load("cc_traces_registry")


def documented_options() -> set:
    """Every `--compass-oracle-option K=V` the document spells out."""
    text = DOC.read_text(encoding="utf-8")
    return set(re.findall(r"--compass-oracle-option\s+(\S+)", text))


def test_every_shared_option_is_the_one_the_document_explains():
    documented = documented_options()
    for key, value in registry.SHARED_OPTIONS:
        if "{root}" in value:
            # A path the document writes with its own shell variable; its key
            # is checked below, its value cannot be compared as text.
            continue
        assert f"{key}={value}" in documented, (
            f"{key}={value} is in the registry and not in SOURCE_ONLY_SERVING.md"
        )


def test_every_documented_key_is_in_the_registry():
    keys = {o.partition("=")[0] for o in documented_options()}
    registered = {k for k, _ in registry.SHARED_OPTIONS}
    registered |= {k for tp in registry.TPS
                   for k, _ in registry.per_width_options(tp)}
    assert keys - registered == set()


def test_acceptance_never_carries_the_template_s_own_allocation():
    # The refusal this registry exists for. `carry_allocation=1` prices a step
    # against another step's blocks and says it is unmeasured; a diagnostic
    # option that survived into a matrix run would not announce itself.
    assert "carry_allocation" in registry.INADMISSIBLE
    for tp in registry.TPS:
        options = registry.options(tp, "/nowhere")
        assert not any(o.startswith("carry_allocation=") for o in options)
        assert "allocation=native" in options


def test_an_inadmissible_option_is_reported_and_not_merely_absent(monkeypatch):
    monkeypatch.setattr(
        registry, "SHARED_OPTIONS",
        registry.SHARED_OPTIONS + (("carry_allocation", "1"),))
    report = registry.check("/nowhere")
    for cell in report["cells"]:
        assert cell["inadmissible_options"] == ["carry_allocation"]
        assert not cell["runnable"]


def test_all_rank_mode_is_selected_for_every_cell():
    # `rank0` is the parser default and prices the rank that calls itself.
    # Acceptance names the other one, and it is named here rather than left to
    # whoever types the server command.
    assert registry.RANK_AGGREGATION == "slowest"
    report = registry.check("/nowhere")
    assert len(report["cells"]) == 24
    assert {c["rank_aggregation"] for c in report["cells"]} == {"slowest"}


def test_both_all_reduce_regimes_are_loaded_at_the_wide_widths():
    # Same signature, two registration regimes, and the graph selects between
    # them: which one answers must not depend on load order.
    for tp in (2, 4):
        price = next(o for o in registry.options(tp, "/r")
                     if o.startswith("price="))
        assert "ar_capture.json" in price and "ar_plain.json" in price
    tp1 = next(o for o in registry.options(1, "/r") if o.startswith("price="))
    assert "ar_capture.json" not in tp1


def test_the_required_artifacts_are_read_out_of_the_options():
    artifacts = registry.required_artifacts(2, "/r")
    assert artifacts["template"] == "/r/serving/src_tp2/b27dec32.json"
    assert artifacts["head_template"] == "/r/serving/src_tp2/h27dec32.json"
    assert artifacts["replay_target"] == "/r/serving/src_tp2/target.tp2.r22.json"
    assert artifacts["memory_model"] == (
        "/r/memval/capture_replay/profile_r22/profile.tp2.json"
    )
    # Five price specs at a wide width; the two with a graph beside them
    # contribute both files.
    assert artifacts["price[0].graph"].endswith("b27dec32.json")
    assert "price[2].graph" not in artifacts


def test_each_width_replays_a_target_of_its_own_width():
    # The replay runner takes the block and state capacities straight out of
    # the target record and refuses one whose tensor_parallel_size is not the
    # width being replayed. Handing the captured TP=1 record to TP=2 and TP=4
    # -- which is what one --replay-target for the whole matrix does -- is
    # therefore not a mistake the run survives to report.
    assert registry.replay_target(1, "/r") == "/r/poc/g5_27b/target.json"
    assert registry.replay_target(2, "/r") == "/r/serving/src_tp2/target.tp2.r22.json"
    assert registry.replay_target(4, "/r") == "/r/serving/src_tp4/target.tp4.r22.json"
    assert len({registry.replay_target(tp, "/r") for tp in registry.TPS}) == 3


def test_the_derived_targets_come_from_the_same_profile_set_as_the_budget():
    # A derived record is derived from a profile, so the record and the profile
    # in one cell have to be the same set. The unsuffixed `target.tp{2,4}.json`
    # beside these were derived from the oldest `capture_replay/profile/` set;
    # naming them here while the pool is sized from r22 would mix two weight
    # terms in a single run -- the budget from one, the parallel contract and
    # the state-runtime wire form from the other. Both old records stay on
    # disk, which is why this has to be asserted rather than assumed.
    for tp in (2, 4):
        target = registry.replay_target(tp, "/r")
        assert target.endswith(f"/serving/src_tp{tp}/target.tp{tp}.r22.json")
        assert "r22" in registry.memory_model(tp, "/r")


def test_every_width_is_sized_by_the_analytical_profile_of_its_width():
    # TP=1 included. The acceptance asks whether the analytical memory model
    # holds across the three widths; a TP=1 replay sized from the captured
    # count answers a different question, and it is the question whose answer
    # is already known.
    for tp in registry.TPS:
        assert registry.memory_model(tp, "/r") == (
            f"/r/memval/capture_replay/profile_r22/profile.tp{tp}.json"
        )


def test_the_profile_is_a_required_artifact_at_every_width():
    # Read out of the same place the plan reads it, so a profile that is not
    # staged is an absence in the readiness report rather than a server that
    # starts and quietly sizes itself from a captured count.
    for tp in registry.TPS:
        assert registry.required_artifacts(tp, "/r")["memory_model"] == (
            registry.memory_model(tp, "/r")
        )


def test_a_root_with_nothing_in_it_is_every_cell_of_absences(tmp_path):
    report = registry.check(tmp_path)
    assert [c["cell"] for c in report["cells"]] == list(registry.CELLS)
    for cell in report["cells"]:
        assert not cell["runnable"]
        # Keyed by role and rank: at TP4 one missing template is four
        # absences, because four ranks each open a different file.
        expected = {f"{role}@tp{rank}"
                    for rank, roles in cell["resolution"].items()
                    for role in roles}
        assert set(cell["absent_artifacts"]) == expected


def test_a_root_holding_every_file_is_runnable(tmp_path):
    for tp in registry.TPS:
        for path in registry.required_artifacts(tp, tmp_path).values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{}")
    report = registry.check(tmp_path)
    assert all(c["runnable"] for c in report["cells"])
    # Runnable is about artifacts. It is not a claim that the cell is costed:
    # the four owed terms are still owed, and the report still says so.
    assert set(report["owed_costs"]) == set(registry.OWED_TERMS)


def test_the_costs_nothing_measures_stay_named_rather_than_zeroed():
    assert set(registry.OWED_TERMS) == {
        "capture", "calibration", "derivation", "load"}
    # Only one of the four is inside the gate's denominator. Deriving this
    # candidate's graphs is work asking the question costs; capture and
    # calibration are paid once, before any question is asked.
    in_gate = {k for k in registry.OWED_TERMS
               if registry.COST_TERMS[k]["in_gate"]}
    assert in_gate == {"derivation"}
    for name in registry.OWED_TERMS:
        assert registry.COST_TERMS[name]["from"]


def test_the_measured_terms_name_the_step_that_measures_them():
    measured = {k for k, v in registry.COST_TERMS.items()
                if v["state"] == "measured"}
    assert measured == {"startup_real", "startup_modelled",
                        "execution_real", "execution_modelled"}
    for name in measured:
        assert registry.COST_TERMS[name]["from"] == "cc_traces_run.py"


def test_the_report_reads_as_a_report(tmp_path):
    text = registry.render(registry.check(tmp_path))
    assert "clients_large c8 -- NOT rankable" in text
    assert "tp4_clients_large_c8" in text
    assert "NOT runnable" in text
    assert "costs no step of this harness measures" in text
    assert "ABSENT" in text


def test_the_cli_exits_non_zero_while_anything_is_absent(tmp_path, capsys):
    assert registry.main(["--root", str(tmp_path)]) == 1
    assert "NOT runnable" in capsys.readouterr().out


@pytest.mark.parametrize("tp", registry.TPS)
def test_the_width_reaches_its_own_files(tp):
    options = registry.options(tp, "/r")
    assert f"tp={tp}" in options
    paths = registry.required_artifacts(tp, "/r")
    # The templates are the width's own seeded pair, and they stay under its
    # own directory. The rest of the price list does not: TP1's book draws on
    # `pricing_coverage`, `g4/card` and `xacq/training` as well as `g4/src1`,
    # and requiring one directory would mean either dropping those from the
    # registry or filing them under a name that is not where they live.
    # At TP1 that directory is `g4/src2c`, the corrected decode-32 capture,
    # and not `g4/src1`: the templates are what the step binds, and src1's
    # body graph recorded an attention chain the deployment does not run.
    own = "g4/src2c" if tp == 1 else f"serving/src_tp{tp}"
    for role in ("template", "head_template"):
        assert own in paths[role], (role, paths[role])
    # What must hold for every path is the thing the marker was standing in
    # for: no width reads another width's files. That is the failure worth
    # catching -- a TP2 cell priced from TP4's measurements would run, and
    # report a number about a deployment nobody configured.
    foreign = [f"serving/src_tp{other}" for other in registry.TPS if other != tp]
    foreign += [f".tp{other}." for other in registry.TPS if other != tp]
    for role, path in paths.items():
        assert not any(mark in path for mark in foreign), (role, path)


# -- what a rank actually opens ------------------------------------------


def test_a_wide_width_resolves_each_rank_to_its_own_file(tmp_path):
    # The option names `b27dec32.json`; rank 2 reads `b27dec32.tp2.json`. A
    # registry that only checked the written name would report a staged
    # directory as complete without ever naming a file a rank opens.
    for rank in range(4):
        for stem in ("b27dec32", "h27dec32", "p27bdec32", "p27hdec32"):
            path = tmp_path / "serving" / "src_tp4" / f"{stem}.tp{rank}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

    found = registry.resolution(4, tmp_path)
    assert sorted(found) == [0, 1, 2, 3]
    assert found[2]["template"]["path"].endswith("b27dec32.tp2.json")
    assert found[2]["template"]["own"] is True
    assert found[2]["template"]["exists"] is True


def test_a_rank_reading_another_rank_s_file_is_named_not_counted_present(tmp_path):
    """The fallback is admissible and it is a claim, so it is reported.

    `resolve_rank_path` falls back to the unsuffixed file on purpose: the ranks
    of a symmetric group time within a fraction of a percent. But a run in
    which every rank silently read rank 0's artifacts looks exactly like one in
    which each read its own, and the TP4 rank-1 outlier is 20% off its peers.
    """
    wide = tmp_path / "serving" / "src_tp2"
    wide.mkdir(parents=True)
    for stem in ("b27dec32", "h27dec32", "p27bdec32", "p27hdec32"):
        (wide / f"{stem}.json").write_text("{}")
        (wide / f"{stem}.tp0.json").write_text("{}")
    for name in ("ar_capture", "ar_plain", "ag_prices"):
        (wide / f"{name}.json").write_text("{}")
    # The target derived at this width, and the profile it is sized from --
    # which lives with MEMORY's other profiles, not under the width's
    # directory.
    Path(registry.replay_target(2, tmp_path)).write_text("{}")
    profile = Path(registry.memory_model(2, tmp_path))
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("{}")

    cell = next(c for c in registry.check(tmp_path)["cells"]
                if c["cell"] == "tp2_clients_short_c1")
    assert cell["absent_artifacts"] == []
    assert cell["runnable"]
    # Rank 0 has its own of the four per-rank artifacts. The three all-reduce
    # and gather lists are shared on purpose and are written unsuffixed, so
    # they are "not this rank's" even at rank 0; the per-rank four are what
    # separate the two ranks here.
    per_rank = ("template", "head_template", "price[0]", "price[1]")
    assert not any(key.startswith(per_rank) and key.endswith("@tp0")
                   for key in cell["shared_artifacts"])
    assert "template@tp1" in cell["shared_artifacts"]
    assert "head_template@tp1" in cell["shared_artifacts"]


def test_the_report_says_which_rank_is_missing_which_file(tmp_path):
    text = registry.render(registry.check(tmp_path))
    assert "ABSENT  template@tp3" in text
    assert "b27dec32.tp3.json" in text


def test_the_target_and_the_profile_are_not_per_rank_artifacts(tmp_path):
    # They are per *width*: one record builds the Config and one profile sizes
    # the pool, and every rank of the group gets the same pair. Asking rank 3
    # for `target.tp4.r22.tp3.json` would invent a file nothing produces.
    found = registry.resolution(4, tmp_path)
    for rank in range(4):
        assert found[rank]["replay_target"]["path"].endswith("target.tp4.r22.json")
        assert found[rank]["replay_target"]["own"] is True
        assert found[rank]["memory_model"]["path"].endswith("profile.tp4.json")
        assert found[rank]["memory_model"]["own"] is True


def test_a_group_of_one_has_nobody_to_be_confused_with(tmp_path):
    # The TP=1 options name `*.tp1.r0.json`, which already carries the
    # coordinates a rank would append. Resolving them again asks for
    # `...tp1.r0.tp0.json`, misses, falls back, and reports six lines of
    # "not this rank's" about a width with one rank.
    # Staged from the registry's own list rather than a handful of names
    # written out here. The four seed files were the whole of TP1's book when
    # this was written; the book has since grown the prefill cells, the row
    # ladder and the cached-MHA inputs, and a fixture that lists files by hand
    # would report them absent and hide the resolution claim this is about.
    # The profile is included: TP=1 bootstraps its Config from the captured
    # record and still replays under the analytical profile.
    for path in registry.required_artifacts(1, tmp_path).values():
        staged = Path(path)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("{}")

    cell = registry.check(tmp_path)["cells"][0]
    assert cell["cell"] == "tp1_clients_short_c1"
    assert cell["absent_artifacts"] == []
    assert cell["shared_artifacts"] == []
    assert cell["runnable"]
    found = registry.resolution(1, tmp_path)[0]
    assert found["template"]["path"].endswith("b27dec32.tp1.r0.json")
    assert found["template"]["own"] is True


def test_a_collective_s_price_list_is_the_group_s_not_a_rank_s(tmp_path):
    # Nothing writes `ar_capture.tp2.json`: the all-reduce is measured on the
    # group, once. Reporting its one file as each rank's fallback would file
    # twelve claims at TP4 against files that are correct, and bury the one
    # per-rank fallback that is a claim.
    found = registry.resolution(4, tmp_path)
    for rank in range(4):
        for role in ("price[2].prices", "price[3].prices", "price[4].prices"):
            assert found[rank][role]["own"] is True
            assert f".tp{rank}.json" not in found[rank][role]["path"]
        # The body and head templates stay per-rank, which is the point.
        assert found[rank]["template"]["path"].endswith(f"b27dec32.tp{rank}.json")


# -- what a fully configured cell still refuses ---------------------------


def test_every_artifact_resolving_does_not_make_a_cell_ready(tmp_path):
    # The failure this exists to stop: a matrix launched because the registry
    # said "runnable", meeting the head's 32-row refusal on the first decode
    # step with 31 running requests, hours in.
    # `head_rows` is no longer every cell's: the source-width row ladder
    # measures 1..32 at TP1, which is every count `max_num_seqs=32` reaches,
    # so it closed there and stands at TP2/TP4. What must hold for all six is
    # the original claim -- resolving every artifact is not readiness -- and
    # each cell must name the refusals the registry charges it, no fewer.
    wide = set(registry.OPEN_REFUSALS["head_rows"]["cells"])
    assert wide and wide < set(registry.CELLS)
    for cell in registry.check(tmp_path)["cells"]:
        assert not cell["ready"]
        owed = {k for k, v in registry.OPEN_REFUSALS.items()
                if cell["cell"] in v["cells"]}
        assert owed <= set(cell["open_refusals"]), cell["cell"]
        assert owed
        assert ("head_rows" in cell["open_refusals"]) is (cell["cell"] in wide)


def test_the_refusal_met_first_is_named_as_such():
    first = [k for k, v in registry.OPEN_REFUSALS.items() if v["first_met"]]
    assert first == ["head_rows"]
    # Charged at TP2 and TP4 and nowhere else, because that is where the head
    # is still priced at 32 rows alone. A cell that met it first and was not
    # told would be a matrix launched on a refusal it hits on the first decode
    # step with 31 running requests, hours in.
    assert registry.OPEN_REFUSALS["head_rows"]["cells"] == registry.cells_at(2, 4)
    # Every cell of those two widths, at every offered load: the head is
    # priced at 32 rows whatever the client count is, so a refusal charged to
    # `tp2 c1` and not to `tp2 c8` would be a claim about the wrong axis.
    assert len(registry.OPEN_REFUSALS["head_rows"]["cells"]) == 16
    assert not any(c.startswith("tp1_")
                   for c in registry.OPEN_REFUSALS["head_rows"]["cells"])


def test_the_two_width_refusals_are_not_charged_to_tp1():
    # Derivation serving rank 0's shard to everybody, and the region model
    # being held out, are both TP>1 statements. Charging them to TP1 would
    # make the one width that could run today look as blocked as the others.
    for key in ("derivation_is_rank0_s", "region_model_held_out"):
        cells = registry.OPEN_REFUSALS[key]["cells"]
        assert not any(c.startswith("tp1_") for c in cells)
        assert set(cells) == set(registry.cells_at(2, 4))


def test_each_refusal_says_what_would_close_it(tmp_path):
    for key, term in registry.OPEN_REFUSALS.items():
        assert term["closes"], key
        assert set(term["cells"]) <= set(registry.CELLS), key
    text = registry.render(registry.check(tmp_path))
    assert "REFUSES (met first)  head_rows" in text
    # Against the registry's own wording, not a copy of it typed here: the
    # copy is what went stale when the ladder replaced the head price sweep,
    # and a report that prints a closing condition nobody is working on is
    # worse than one that prints none. Every refusal, not just the first.
    for key, term in registry.OPEN_REFUSALS.items():
        assert f"closed by: {term['closes']}" in text, key


def test_all_three_widths_are_sized_from_the_staged_r22_profiles():
    # The routing test for the profile set, not for the per-width suffix.
    # Three directories exist and all three stay on disk, because published
    # results cite their own set by path: `profile/`, `profile_r21/`, and
    # `profile_r22/`, whose weight term is ATOM's own build of the resolved
    # checkpoint rather than the checkpoint's safetensors headers. Only the
    # last is this plan's input, and a default that silently reverted to
    # either retired set would size all twenty-four cells from a weight term
    # nobody chose while every other artifact stayed current -- an absence
    # nothing in the readiness report would show, because all three resolve.
    for tp in registry.TPS:
        assert registry.memory_model(tp, "/r") == (
            f"/r/memval/capture_replay/profile_r22/profile.tp{tp}.json"
        )
        assert "/capture_replay/profile/" not in registry.memory_model(tp, "/r")
        assert "/profile_r21/" not in registry.memory_model(tp, "/r")
    assert len({registry.memory_model(tp, "/r") for tp in registry.TPS}) == 3


# -- the matrix this registry is of --------------------------------------


def _plan_cells():
    """The cells an ordinary default plan prints, with no flags but a root."""
    plan = _load("cc_traces_plan")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert plan.main(["--root", "/r"]) == 0
    return json.loads(buffer.getvalue())["cells"]


class TestTheRegistryIsOfTheMatrixThatRuns:
    """Twenty-four cells, named the same in all three modules.

    The registry described six `tp{1,2,4}-{short,long}` cells while the plan
    wrote twenty-four directories and the validator keyed twenty-four verdicts.
    Nothing failed: a readiness report simply answered about a matrix nobody
    was going to run, and eighteen cells had no readiness line at all.
    """

    def test_the_registry_names_twenty_four_cells(self):
        assert len(registry.CELLS) == 24
        assert len(set(registry.CELLS)) == 24
        assert registry.CELLS == tuple(
            registry.cell_id(tp, klass, clients)
            for tp in registry.TPS
            for klass in registry.CLASSES
            for clients in registry.CLIENTS
        )

    def test_every_cell_carries_its_client_count(self):
        report = registry.check("/nowhere")
        assert len(report["cells"]) == 24
        for cell in report["cells"]:
            assert cell["clients"] in registry.CLIENTS
            assert cell["cell"].endswith(f"_c{cell['clients']}")
            assert cell["class"] in registry.CLASSES

    def test_the_registry_and_the_plan_name_the_same_cells(self):
        # The plan's directory basename *is* the registry's cell id. Two names
        # for one cell is how a readiness line and an evidence directory stop
        # being about the same thing.
        planned = {Path(c["cell"]).name for c in _plan_cells()}
        assert planned == set(registry.CELLS)

    def test_the_registry_and_the_validator_register_the_same_cells(self):
        validate = _load("cc_traces_validate")
        keyed = {registry.cell_id(tp, klass, clients)
                 for tp, klass, clients in validate.REGISTERED_CELLS}
        assert keyed == set(registry.CELLS)

    def test_no_cell_directory_can_overwrite_another_client_count(self):
        # Each cell's evidence is written under its own directory, and the
        # client count is in the name. Without it, `tp2_clients_short` would be
        # four runs into one directory and the last one would own every file.
        cells = _plan_cells()
        assert len(cells) == 24
        assert len({c["cell"] for c in cells}) == 24
        for tp in registry.TPS:
            for klass in registry.CLASSES:
                same = [c["cell"] for c in cells
                        if c["tp"] == tp and c["class"] == klass]
                assert len(set(same)) == len(registry.CLIENTS) == len(same)
        # Two steps of one cell may write one file -- the sampler's baseline
        # and its window both append to that cell's `gpu.jsonl` -- so what is
        # checked is that no *cell* can write a file another cell wrote.
        produced = [
            {f"{c['cell']}/{name}"
             for step in c["steps"] for name in step["produces"]}
            for c in cells
        ]
        flat = [path for cell in produced for path in cell]
        assert len(flat) == len(set(flat))
        assert all(paths for paths in produced)

    def test_no_verdict_can_be_read_as_another_client_count(self):
        validate = _load("cc_traces_validate")
        refused = []
        keys = {
            validate._cell_key(
                {"cell": registry.cell_id(2, "clients_short", clients),
                 "tp": 2, "class": "clients_short", "clients": clients},
                "where", refused)
            for clients in registry.CLIENTS
        }
        assert refused == []
        assert len(keys) == len(registry.CLIENTS)
        # And one that carries no count is not quietly resolved to any of them.
        assert validate._cell_key(
            {"cell": "tp2-short", "tp": 2, "class": "short"},
            "where", refused) is None
        assert refused

    def test_a_group_is_one_class_at_one_offered_load(self):
        report = registry.check("/nowhere")
        assert len(report["groups"]) == 8
        for name, group in report["groups"].items():
            assert group["widths"] == list(registry.TPS)
            assert len(set(group["cells"])) == 3
            # Nothing is rankable under a root with no artifacts in it, and the
            # group says so rather than reporting a rank over what resolved.
            assert group["rankable"] is False
            assert group["not_runnable"] == group["cells"]
            assert all(c["group"] == name for c in report["cells"]
                       if c["cell"] in group["cells"])

    def test_a_group_missing_one_width_is_not_a_partial_rank(self, tmp_path):
        # Everything staged for TP1 and TP2 and nothing for TP4: eight groups
        # each holding two runnable cells and one that is not, and no group
        # rankable. Two thirds of a rank over three widths is not a rank.
        for tp in (1, 2):
            for path in registry.required_artifacts(tp, tmp_path).values():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("{}")
        report = registry.check(tmp_path)
        assert not any(g["rankable"] for g in report["groups"].values())
        for group in report["groups"].values():
            assert len(group["not_runnable"]) == 1
            assert group["not_runnable"][0].startswith("tp4_")

    def test_a_refusal_is_charged_by_width_not_by_offered_load(self):
        # The head is priced at 32 rows and nowhere else at TP>1 whatever the
        # client count is. A refusal that named c1 and not c8 would be a claim
        # about the axis that does not carry it.
        for term in registry.OPEN_REFUSALS.values():
            widths = {c.split("_")[0] for c in term["cells"]}
            for width in widths:
                charged = {c for c in term["cells"] if c.startswith(f"{width}_")}
                assert charged == {c for c in registry.CELLS
                                   if c.startswith(f"{width}_")}, term["what"]

    def test_the_configuration_is_the_width_s_and_the_load_does_not_move_it(self):
        # What the offered load changes is which requests arrive, not what
        # prices them. If that ever stops being true it is a finding, and it
        # should be one here rather than in a run.
        report = registry.check("/nowhere")
        for tp in registry.TPS:
            same = [c for c in report["cells"] if c["tp"] == tp]
            assert len(same) == 8
            assert len({json.dumps(c["oracle_options"]) for c in same}) == 1
            assert len({json.dumps(c["artifacts"]) for c in same}) == 1
            assert len({json.dumps(c["absent_artifacts"]) for c in same}) == 1

    def test_the_first_registration_is_archived_and_not_relabelled(self):
        # The `short`/`long` classes selected different requests under a
        # different rule. Renaming `tp2-short` into this matrix would put the
        # clients matrix's name on measurements taken under the other one.
        archived = registry.ARCHIVED_REGISTRATION
        assert archived["cells"] == (
            "tp1-short", "tp1-long", "tp2-short", "tp2-long",
            "tp4-short", "tp4-long")
        assert archived["classes"] == ("short", "long")
        assert archived["why_archived"]
        assert not set(archived["cells"]) & set(registry.CELLS)
        assert "short" not in registry.CLASSES and "long" not in registry.CLASSES
        report = registry.check("/nowhere")
        assert report["archived_registration"]["cells"] == list(archived["cells"])
        assert not any(c["cell"] in archived["cells"] for c in report["cells"])

    def test_the_report_says_the_archived_registration_is_archived(self, tmp_path):
        text = registry.render(registry.check(tmp_path))
        assert "archived and not relabelled" in text
        assert "tp2-short" in text
        assert registry.ARCHIVED_REGISTRATION["why_archived"] in text

    def test_the_report_is_readable_at_twenty_four_cells(self, tmp_path):
        # The artifacts are the width's, so the report states them three times
        # rather than twenty-four. A reader who cannot find the eight lines
        # that differ under two hundred that cannot does not read it.
        text = registry.render(registry.check(tmp_path))
        assert text.count("  oracle            ") == len(registry.TPS)
        for cell in registry.CELLS:
            assert cell in text
        for name in registry.check(tmp_path)["groups"]:
            assert f"## {name} -- " in text
