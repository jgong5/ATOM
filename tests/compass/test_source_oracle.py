"""The seam between the frozen composition and a command line.

What a served run can say is `KEY=VALUE`, once per key, with numbers converted
and everything else left a string. What `LibraryCostOracle` needs is three live
objects. These tests pin the translation between the two -- including the
places it refuses rather than guessing, because an option that quietly reads
false produces a prediction missing a whole region with nothing in the record
to say which one.
"""

import inspect
import json

import pytest

from atom.compass.core.cost.regions import (REGION_MODELS, SOURCE_27B_TP1,
                                            SOURCE_27B_TP1_CONC,
                                            SOURCE_27B_TP1_CONC_V2,
                                            region_model)
from atom.compass.runtime.source_oracle import (SourceComposition,
                                                _entries, _flag,
                                                build_source_oracle,
                                                price_specs,
                                                source_cost_oracle,
                                                template_shape)


def _price_file(tmp_path, name="prices.json"):
    """A library file with one priced signature, in the schema `add` reads."""
    path = tmp_path / name
    path.write_text(json.dumps({
        "provenance": {"topology": {"tp": 1}, "registration": "unregistered"},
        "prices": {"sig::x": {"seconds": 1.0e-5}},
        "unpriced": {},
    }), encoding="utf-8")
    return str(path)


def saved_graph(*, bucket=None, spec=True, topology=(("tp", 2),),
                rank_coords=(("tp", 1),)):
    """A graph as `graph.save` writes one: coordinate maps as pair lists."""
    provenance = {"execution": {"capture_bucket": bucket, "step_kind": "decode"}}
    if spec:
        provenance["batch_spec"] = {
            "kind": "decode", "query_lens": [1, 1, 1, 1],
            "context_lens": [1151] * 4, "block_size": 16}
    return {"key": {"topology": [list(p) for p in topology],
                    "rank_coords": [list(p) for p in rank_coords]},
            "ops": [], "provenance": provenance}


def _template_file(tmp_path, name="graph.json"):
    """A body graph carrying the batch spec that says what it is a template for."""
    path = tmp_path / name
    path.write_text(json.dumps({
        "ops": [],
        "key": {"topology": [["tp", 1]], "rank_coords": []},
        "provenance": {
            "batch_spec": {"kind": "decode",
                           "query_lens": [1, 1],
                           "context_lens": [1025, 1025]},
            "execution": {"capture_bucket": 2},
        },
    }), encoding="utf-8")
    return str(path)


class TestTheRegionRegistry:

    def test_every_published_profile_can_be_asked_for_by_name(self):
        assert region_model("source-27b-tp1") is SOURCE_27B_TP1
        assert region_model("source-27b-tp1-conc") is SOURCE_27B_TP1_CONC
        assert region_model("source-27b-tp1-conc-v2") is SOURCE_27B_TP1_CONC_V2

    def test_none_is_a_name_and_not_an_absence(self):
        """Body plus head with no runner term is a claim, so it is spelled.

        A caller that wants it says so and the report records `regions:
        "none"`; a caller that mistypes a name gets an error. Those have to be
        different, because they differ by tens of percent in the step.
        """
        assert region_model("none") is None
        assert "none" in REGION_MODELS

    def test_an_unknown_name_is_refused_with_the_known_ones_named(self):
        with pytest.raises(ValueError) as exc:
            region_model("source-27b-tp1-conc-v3")
        assert "source-27b-tp1-conc-v2" in str(exc.value)

    def test_the_script_reads_this_registry_rather_than_its_own(self):
        """`predict_step.py` used to hold a second copy, by import path.

        Two lists of region models is two chances to publish a profile in one
        and not the other, and the one that lags answers with an older
        calibration under a name the caller believes is current.
        """
        source = (open("scripts/compass/predict_step.py", encoding="utf-8")
                  .read())
        assert "from atom.compass.core.cost.regions import REGION_MODELS" in source
        assert "SOURCE_27B_TP1\")" not in source


class TestWhatACommandLineCanCarry:

    def test_a_repeatable_argument_arrives_as_one_comma_separated_string(self):
        assert _entries("a.json,b.json", "template") == ["a.json", "b.json"]
        assert _entries(" a.json , b.json ", "template") == ["a.json", "b.json"]

    def test_a_programmatic_caller_may_pass_a_real_list(self):
        assert _entries(["a.json", "b.json"], "template") == ["a.json", "b.json"]
        assert _entries((), "template") == []

    def test_nothing_and_the_empty_string_are_no_entries(self):
        assert _entries(None, "price") == []
        assert _entries("", "price") == []

    def test_a_flag_is_read_from_the_int_or_the_word(self):
        """`arg_utils` converts a numeric value, so both forms really arrive."""
        for yes in (True, 1, "1", "true", "TRUE", "yes", "on"):
            assert _flag(yes, "head") is True
        for no in (False, 0, "0", "false", "no", "off"):
            assert _flag(no, "head") is False

    def test_a_flag_that_is_not_one_is_refused(self):
        """Rather than read as false, which loses a region silently."""
        for bad in ("maybe", 2, 0.5, None, []):
            with pytest.raises(ValueError, match="head"):
                _flag(bad, "head")

    def test_price_entries_take_one_two_or_three_parts(self):
        assert price_specs(["p.json"]) == [("p.json", None)]
        assert price_specs(["p.json:g.json"]) == [("p.json", "g.json")]
        assert (price_specs(["p.json:g.json:unregistered"])
                == [("p.json", "g.json", "unregistered")])
        # An empty middle is "no graph, but a regime", not a path named "".
        assert price_specs(["p.json::plain"]) == [("p.json", None, "plain")]

    def test_a_fourth_part_is_refused(self):
        with pytest.raises(ValueError, match="at most"):
            price_specs(["p.json:g.json:unregistered:extra"])


class TestKeyingASavedGraph:
    """Moved here with `template_shape` itself, from the CLI it used to live in.

    A template keyed at a bucket nobody traced it at answers for a padded graph
    that does not exist, which is a wrong answer nothing downstream can see.
    """

    def test_saved_coordinate_pair_lists_are_read_back_as_maps(self):
        shape = template_shape(saved_graph())
        assert shape.topology == {"tp": 2}
        assert shape.rank_coords == {"tp": 1}

    def test_a_coordinate_map_saved_as_a_dict_also_reads(self):
        graph = saved_graph()
        graph["key"]["topology"] = {"tp": 2}
        assert template_shape(graph).topology == {"tp": 2}

    def test_a_graph_is_keyed_by_the_batch_it_was_traced_over(self):
        shape = template_shape(saved_graph())
        assert shape.num_scheduled_tokens == (1, 1, 1, 1)
        assert shape.context_lens == (1151,) * 4
        assert shape.num_prefill_tokens == 0

    def test_the_bucket_comes_from_execution_not_the_batch(self):
        """A batch spec describes requests; the bucket is how the step was run."""
        assert template_shape(saved_graph(bucket=32)).capture_bucket == 32

    def test_a_graph_traced_with_no_bucket_keys_as_none(self):
        """It is a template for the uncaptured structure, and must not claim one."""
        assert template_shape(saved_graph()).capture_bucket is None

    def test_a_graph_with_no_batch_spec_cannot_be_a_template(self):
        with pytest.raises(ValueError, match="batch_spec"):
            template_shape(saved_graph(spec=False))


class TestBuildingWithoutAModel:
    """Derivation needs a model; seeding templates does not.

    Every test here runs with `derive=False`, which is the whole reason the
    factory separates the two: a run whose templates already cover its shapes
    can be built on a machine with no model and no GPU.
    """

    def test_it_returns_the_library_oracle_the_frozen_path_uses(self, tmp_path):
        from atom.compass.core.cost.library import LibraryCostOracle

        oracle = source_cost_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0, regions="source-27b-tp1-conc-v2")
        assert isinstance(oracle, LibraryCostOracle)
        assert oracle.regions is SOURCE_27B_TP1_CONC_V2
        assert oracle.require_complete is True

    def test_the_seeded_template_is_keyed_and_findable(self, tmp_path):
        from atom.compass.core.cost.base import StepShape

        built = build_source_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0)
        shape = StepShape(num_scheduled_tokens=(1, 1),
                          context_lens=(1025, 1025), num_prefill_tokens=0,
                          topology={"tp": 1}, capture_bucket=2)
        assert built.body_graphs.graph_for(shape) is not None
        assert built.deriver is None
        assert built.build_seconds == 0.0

    def test_a_build_that_can_only_refuse_says_so_at_build_time(self, tmp_path):
        """Not as one identical refusal per shape, an hour later."""
        with pytest.raises(ValueError, match="derive is off"):
            source_cost_oracle(price=_price_file(tmp_path), derive=0)

    def test_deriving_without_a_model_is_refused_before_anything_loads(self):
        with pytest.raises(ValueError, match="model path"):
            source_cost_oracle(derive=1)

    def test_deriving_without_the_block_table_settings_is_refused(self):
        with pytest.raises(ValueError, match="block_size"):
            source_cost_oracle(derive=1, model="/models/x")

    def test_the_head_is_off_unless_asked_for(self, tmp_path):
        """Body-only is a different claim, and it is the default one."""
        built = build_source_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0)
        assert built.head_graphs is None
        assert built.oracle.head_graphs is None

    def test_the_head_takes_its_own_templates(self, tmp_path):
        built = build_source_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            head_template=_template_file(tmp_path, "hgraph.json"),
            head="true", derive=0)
        assert built.head_graphs is not None
        assert built.oracle.head_graphs is built.head_graphs

    def test_an_unmeasured_allocation_is_carried_only_when_asked_and_named(
            self, tmp_path):
        plain = build_source_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0)
        assert plain.allocation is None
        carried = build_source_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0, carry_allocation=1)
        assert carried.allocation is not None
        assert "carry_allocation" in carried.allocation.describe()

    def test_two_price_files_arrive_from_one_option(self, tmp_path):
        built = build_source_oracle(
            price=f"{_price_file(tmp_path)},{_price_file(tmp_path, 'p2.json')}",
            template=_template_file(tmp_path), derive=0)
        assert len(built.oracle.library.sources) == 2

    def test_allowing_an_incomplete_price_takes_saying_so(self, tmp_path):
        built = build_source_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0, require_complete=0)
        assert built.oracle.require_complete is False


class TestTheContractWithTheServedPath:

    def test_every_parameter_is_something_a_key_value_flag_can_carry(self):
        """The point of the factory: no default is a live object.

        `--compass-oracle-option KEY=VALUE` produces `str`, `int` or `float`.
        A parameter whose default is anything else is a parameter the CLI
        cannot supply, which is the gap this closes.
        """
        params = inspect.signature(build_source_oracle).parameters
        assert params, "the factory takes arguments"
        for name, param in params.items():
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
            assert isinstance(param.default, (str, int, float, bool, type(None))), name

    def test_the_entry_point_resolves_by_qualname(self):
        from atom.utils import resolve_obj_by_qualname

        resolved = resolve_obj_by_qualname(
            "atom.compass.runtime.source_oracle.source_cost_oracle")
        assert resolved is source_cost_oracle

    def test_the_runner_will_not_offer_it_a_rank_it_does_not_take(self):
        """`_build_oracle` passes `rank_coords` only to an oracle that names it.

        This one does not: a shape carries its own rank coordinates, and an
        unexpected keyword would fail the build at TP>1 only -- the one place
        it would be hardest to see.
        """
        assert "rank_coords" not in inspect.signature(
            source_cost_oracle).parameters

    def test_a_mistyped_option_is_an_error_and_not_a_default(self, tmp_path):
        with pytest.raises(TypeError):
            source_cost_oracle(price=_price_file(tmp_path), derive=0,
                               template=_template_file(tmp_path),
                               reqions="source-27b-tp1")

    def test_the_composition_names_what_a_report_has_to_state(self):
        assert SourceComposition._fields == (
            "oracle", "body_graphs", "head_graphs", "deriver", "build_seconds",
            "allocation")
