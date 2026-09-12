"""The six cells' configuration, against the prose that explains it.

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

import importlib.util
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
    assert len(report["cells"]) == 6
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
    assert artifacts["replay_target"] == "/r/serving/src_tp2/target.tp2.json"
    assert artifacts["memory_model"] == "/r/serving/src_tp2/profile.tp2.json"
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
    assert registry.replay_target(2, "/r") == "/r/serving/src_tp2/target.tp2.json"
    assert registry.replay_target(4, "/r") == "/r/serving/src_tp4/target.tp4.json"
    assert len({registry.replay_target(tp, "/r") for tp in registry.TPS}) == 3


def test_only_the_widths_without_a_captured_record_are_sized_by_a_profile():
    # TP=1 has a record of its own width and is sized by it; a wider width has
    # nothing else to be sized from, so the profile is what makes its capacity
    # attributable rather than assumed.
    assert registry.memory_model(1, "/r") is None
    assert registry.memory_model(2, "/r") == "/r/serving/src_tp2/profile.tp2.json"
    assert registry.memory_model(4, "/r") == "/r/serving/src_tp4/profile.tp4.json"


def test_a_width_that_needs_a_profile_reports_it_as_required():
    # Read out of the same place the plan reads it, so a profile that is not
    # staged is an absence in the readiness report rather than a server that
    # starts and sizes itself from the wrong width.
    assert "memory_model" not in registry.required_artifacts(1, "/r")
    for tp in (2, 4):
        assert registry.required_artifacts(tp, "/r")["memory_model"] == (
            f"/r/serving/src_tp{tp}/profile.tp{tp}.json"
        )


def test_a_root_with_nothing_in_it_is_six_cells_of_absences(tmp_path):
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
    assert "tp4-long -- NOT runnable" in text
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
    marker = "g4/src1" if tp == 1 else f"serving/src_tp{tp}"
    assert all(marker in p for role, p in paths.items()
               if role != "replay_target")


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
    # The width's own two, not the captured record: a wide width replays a
    # target derived at that width and sizes its pool from the profile it was
    # derived from.
    (wide / "target.tp2.json").write_text("{}")
    (wide / "profile.tp2.json").write_text("{}")

    cell = registry.check(tmp_path)["cells"][2]
    assert cell["cell"] == "tp2-short"
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
    # for `target.tp4.tp3.json` would invent a file nothing produces.
    found = registry.resolution(4, tmp_path)
    for rank in range(4):
        assert found[rank]["replay_target"]["path"].endswith("target.tp4.json")
        assert found[rank]["replay_target"]["own"] is True
        assert found[rank]["memory_model"]["path"].endswith("profile.tp4.json")
        assert found[rank]["memory_model"]["own"] is True


def test_a_group_of_one_has_nobody_to_be_confused_with(tmp_path):
    # The TP=1 options name `*.tp1.r0.json`, which already carries the
    # coordinates a rank would append. Resolving them again asks for
    # `...tp1.r0.tp0.json`, misses, falls back, and reports six lines of
    # "not this rank's" about a width with one rank.
    src = tmp_path / "g4" / "src1"
    src.mkdir(parents=True)
    for stem in ("b27dec32", "h27dec32", "p27bdec32", "p27hdec32"):
        (src / f"{stem}.tp1.r0.json").write_text("{}")
    (tmp_path / "poc" / "g5_27b").mkdir(parents=True)
    (tmp_path / "poc" / "g5_27b" / "target.json").write_text("{}")

    cell = registry.check(tmp_path)["cells"][0]
    assert cell["cell"] == "tp1-short"
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
    for cell in registry.check(tmp_path)["cells"]:
        assert not cell["ready"]
        assert "head_rows" in cell["open_refusals"]


def test_the_refusal_met_first_is_named_as_such():
    first = [k for k, v in registry.OPEN_REFUSALS.items() if v["first_met"]]
    assert first == ["head_rows"]
    assert registry.OPEN_REFUSALS["head_rows"]["cells"] == registry.CELLS


def test_the_two_width_refusals_are_not_charged_to_tp1():
    # Derivation serving rank 0's shard to everybody, and the region model
    # being held out, are both TP>1 statements. Charging them to TP1 would
    # make the one width that could run today look as blocked as the others.
    for key in ("derivation_is_rank0_s", "region_model_held_out"):
        cells = registry.OPEN_REFUSALS[key]["cells"]
        assert "tp1-short" not in cells and "tp1-long" not in cells
        assert set(cells) == {"tp2-short", "tp2-long", "tp4-short", "tp4-long"}


def test_each_refusal_says_what_would_close_it(tmp_path):
    for key, term in registry.OPEN_REFUSALS.items():
        assert term["closes"], key
        assert set(term["cells"]) <= set(registry.CELLS), key
    text = registry.render(registry.check(tmp_path))
    assert "REFUSES (met first)  head_rows" in text
    assert "closed by: a head price sweep" in text
