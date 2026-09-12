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
from types import SimpleNamespace

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

    def test_the_factory_names_the_rank_so_the_runner_offers_it_one(self):
        """`_build_oracle` passes `rank_coords` only to an oracle that names it.

        It decides by ``inspect.signature(...).parameters``, and a bare
        ``**kwargs`` names nothing -- so while this factory declared only
        ``**kwargs`` the injection never fired, and every rank of a TP>1 run
        built the same composition off rank 0's artifacts. The failure was
        silent: each rank read a file that existed and answered.
        """
        params = inspect.signature(source_cost_oracle).parameters
        assert "rank_coords" in params
        assert (params["rank_coords"].kind
                is inspect.Parameter.KEYWORD_ONLY)
        # And it reaches the builder, which is where it does the work.
        assert "rank_coords" in inspect.signature(build_source_oracle).parameters

    def test_a_mistyped_option_is_an_error_and_not_a_default(self, tmp_path):
        with pytest.raises(TypeError):
            source_cost_oracle(price=_price_file(tmp_path), derive=0,
                               template=_template_file(tmp_path),
                               reqions="source-27b-tp1")

    def test_the_composition_names_what_a_report_has_to_state(self):
        assert SourceComposition._fields == (
            "oracle", "body_graphs", "head_graphs", "deriver", "build_seconds",
            "allocation", "rank_coords", "rank_artifacts",
            "interpolation_limit", "loaded_inputs")


def _read_prices(oracle):
    """The price files this oracle's library actually opened.

    `library.sources` holds the paths as they were *asked for*, which since the
    loader took over resolution is the stem the option carried -- the same
    string at every rank. Which file a rank was served is in the record the
    reader took as it parsed the bytes, and that is what these tests are about.
    """
    return [loaded.path for loaded in oracle.library.loaded_inputs
            if loaded.role == "oracle.price"]


def _rank_price_file(tmp_path, name, signature):
    """A library file whose one signature says which file it came from."""
    path = tmp_path / name
    path.write_text(json.dumps({
        "provenance": {"topology": {"tp": 2}, "registration": "unregistered"},
        "prices": {signature: {"seconds": 1.0e-5}},
        "unpriced": {},
    }), encoding="utf-8")
    return str(path)


class _Runner:
    """The two attributes `_build_oracle` reads, and its real methods.

    Bound off `CompassPredictMixin` rather than reimplemented: the question
    these tests answer is what the *served* path does, and a stub that decided
    for itself whether to inject the rank would answer a different question.
    The mixin is the definition both real runners inherit -- see
    `test_both_runners_inherit_this_definition` -- and it is the half of the
    runner that is deliberately device-free, so this runs with no GPU.
    """

    from atom.compass.runtime.predict import CompassPredictMixin as _real

    _topology = _real._topology
    _rank_coords = _real._rank_coords
    _build_oracle = _real._build_oracle
    del _real

    def __init__(self, tp, rank):
        self.config = SimpleNamespace(tensor_parallel_size=tp,
                                      parallel_config=None)
        self.rank = rank


def _oracle_config(price, template):
    from atom.compass.config import CompassConfig

    return CompassConfig(
        enabled=True,
        oracle_qualname="atom.compass.runtime.source_oracle.source_cost_oracle",
        oracle_options={"price": price, "template": template, "derive": 0,
                        "require_complete": 0})


class TestTheRankTheServedPathActuallyBuildsWith:
    """Through `_build_oracle`, because that is where the gap was.

    The factory built correctly in isolation the whole time. What did not
    happen was the runner handing it a rank, and no test of the factory alone
    could see that.
    """

    def test_both_runners_inherit_this_definition(self):
        """So `_Runner` above is the served path and not a second copy of it.

        Checked by reading the class statements rather than by importing them:
        `CompassModelRunner` inherits `ModelRunner`, which imports `aiter`,
        which resolves the chip at import time. These tests run where there is
        no chip -- which is the whole point of the split predict.py describes.
        """
        for module, line in (
                ("atom/compass/runtime/runner.py",
                 "class CompassModelRunner(CompassPredictMixin, ModelRunner):"),
                ("atom/compass/replay/runner.py",
                 "class ReplayModelRunner(CompassPredictMixin):")):
            with open(module, encoding="utf-8") as fh:
                assert line in fh.read(), module

    def test_each_rank_loads_its_own_price_file(self, tmp_path):
        shared = _rank_price_file(tmp_path, "prices.json", "sig::shared")
        _rank_price_file(tmp_path, "prices.tp1.json", "sig::rank1")
        template = _template_file(tmp_path)
        config = _oracle_config(shared, template)

        at_zero = _Runner(tp=2, rank=0)._build_oracle(config)
        at_one = _Runner(tp=2, rank=1)._build_oracle(config)

        assert _read_prices(at_zero) == [shared]
        assert _read_prices(at_one) == [str(tmp_path / "prices.tp1.json")]

    def test_a_rank_with_no_file_of_its_own_reads_the_shared_one(self, tmp_path):
        """Which is correct in a symmetric group, and a different claim.

        `SourceComposition.rank_artifacts` is what makes the difference
        reportable; the oracle alone cannot say it.
        """
        shared = _rank_price_file(tmp_path, "prices.json", "sig::shared")
        template = _template_file(tmp_path)
        oracle = _Runner(tp=2, rank=3)._build_oracle(
            _oracle_config(shared, template))
        assert _read_prices(oracle) == [shared]

        built = build_source_oracle(price=shared, template=template, derive=0,
                                    require_complete=0,
                                    rank_coords={"tp": 3})
        assert built.rank_coords == {"tp": 3}
        assert built.rank_artifacts[shared]["rank_own"] is False

    def test_the_registration_regime_is_not_recorded_as_an_artifact(self,
                                                                    tmp_path):
        """``prices.json:graph.json:unregistered`` names two files, not three.

        The third field states how the collectives in that list were timed. It
        was being resolved as a path, so a served TP4 run recorded
        ``unregistered.tp0`` among the files rank 0 owned -- a rank-own claim
        about something that is not a file.
        """
        shared = _rank_price_file(tmp_path, "prices.json", "sig::shared")
        template = _template_file(tmp_path)
        built = build_source_oracle(
            price=f"{shared}:{template}:unregistered", template=template,
            derive=0, require_complete=0, rank_coords={"tp": 1})
        assert set(built.rank_artifacts) == {shared, template}

    def test_the_composition_records_the_rank_it_was_built_for(self, tmp_path):
        shared = _rank_price_file(tmp_path, "prices.json", "sig::shared")
        own = _rank_price_file(tmp_path, "prices.tp1.json", "sig::rank1")
        built = build_source_oracle(price=shared,
                                    template=_template_file(tmp_path),
                                    derive=0, require_complete=0,
                                    rank_coords={"tp": 1})
        assert built.rank_coords == {"tp": 1}
        assert built.rank_artifacts[shared] == {"resolved": own,
                                                "rank_own": True}

    def test_a_single_rank_run_asks_for_no_suffix_at_all(self, tmp_path):
        """TP1 is frozen. `_build_oracle` guards on the topology, not the rank.

        The decoy would be read if the guard moved, and TP1 results would stop
        being the results that were frozen.
        """
        shared = _rank_price_file(tmp_path, "prices.json", "sig::shared")
        _rank_price_file(tmp_path, "prices.tp0.json", "sig::decoy")
        oracle = _Runner(tp=1, rank=0)._build_oracle(
            _oracle_config(shared, _template_file(tmp_path)))
        assert _read_prices(oracle) == [shared]

    def test_rank_zero_of_a_wide_group_is_also_unchanged(self, tmp_path):
        """The injection fires, resolves to rank 0's own name, and that is the
        name every frozen TP2/TP4 artifact was written under."""
        shared = _rank_price_file(tmp_path, "prices.json", "sig::shared")
        _rank_price_file(tmp_path, "prices.tp0.json", "sig::rank0")
        oracle = _Runner(tp=2, rank=0)._build_oracle(
            _oracle_config(shared, _template_file(tmp_path)))
        assert _read_prices(oracle) == [str(tmp_path / "prices.tp0.json")]

    def test_an_explicit_option_is_not_overridden_by_the_runner(self, tmp_path):
        """`_build_oracle` only fills a rank the options did not already set."""
        shared = _rank_price_file(tmp_path, "prices.json", "sig::shared")
        _rank_price_file(tmp_path, "prices.tp1.json", "sig::rank1")
        config = _oracle_config(shared, _template_file(tmp_path))
        config.oracle_options["rank_coords"] = "tp:1"
        oracle = _Runner(tp=2, rank=0)._build_oracle(config)
        assert _read_prices(oracle) == [str(tmp_path / "prices.tp1.json")]

    def test_a_rank_one_shape_is_served_by_the_rank_zero_template(self, tmp_path):
        """End to end: the miss the rank injection would otherwise have caused.

        Every frozen template is keyed at rank 0, and `template_key` carries
        the rank -- so handing rank 1 its own coordinates turns every template
        hit into a refusal unless the representative fallback stands in.
        """
        from atom.compass.core.cost.base import StepShape

        # As a TP2 derivation writes one: the width it was derived for, and
        # rank 0, because that is the rank the simulated group reports.
        path = tmp_path / "body.tp2.json"
        path.write_text(json.dumps({
            "ops": [],
            "key": {"topology": [["tp", 2]], "rank_coords": [["tp", 0]]},
            "provenance": {
                "batch_spec": {"kind": "decode", "query_lens": [1, 1],
                               "context_lens": [1025, 1025]},
                "execution": {"capture_bucket": 2},
            },
        }), encoding="utf-8")
        built = build_source_oracle(price=_price_file(tmp_path),
                                    template=str(path), derive=0,
                                    require_complete=0, rank_coords={"tp": 1})
        at_one = StepShape(
            num_scheduled_tokens=(1, 1), context_lens=(4096, 4096),
            num_prefill_tokens=0, topology={"tp": 2}, rank_coords={"tp": 1},
            capture_bucket=2, compiled=None, produces_output=True)
        assert built.body_graphs.graph_for(at_one) is not None
        assert built.body_graphs.representative_hits == 1


class TestAskingForFittedPrices:
    """The family provider, reachable from the same command line.

    It is off unless asked for. An interpolated price is an answer about a row
    count nobody ran, and the difference between "we measured this" and "we
    fitted this" is the difference the whole coverage split exists to keep -- so
    a served run gets one only by saying so, and the ratio it says names the
    sampling density its own evidence supports.
    """

    @staticmethod
    def _built(tmp_path, **over):
        return build_source_oracle(
            price=f"{_price_file(tmp_path)}:{_template_file(tmp_path)}",
            template=_template_file(tmp_path), derive=0,
            regions="source-27b-tp1-conc-v2", **over)

    def test_prices_are_exact_unless_interpolation_is_asked_for(self, tmp_path):
        from atom.compass.core.cost.families import ParametricPriceLibrary

        library = self._built(tmp_path).oracle.library
        assert not isinstance(library, ParametricPriceLibrary)

    def test_asking_for_it_by_the_word_takes_the_declared_density(self, tmp_path):
        """`true` is the provider's own default, not a number restated here."""
        from atom.compass.core.cost.families import ParametricPriceLibrary

        library = self._built(tmp_path, interpolate="true").oracle.library
        assert isinstance(library, ParametricPriceLibrary)
        assert library.max_gap_ratio == ParametricPriceLibrary().max_gap_ratio

    def test_a_ratio_is_carried_through_as_given(self, tmp_path):
        # `arg_utils` turns `interpolate=1.5` on a command line into a float,
        # and a programmatic caller passes one; both reach the provider.
        assert self._built(tmp_path,
                           interpolate=1.5).oracle.library.max_gap_ratio == 1.5

    def test_the_number_one_means_on_in_either_spelling(self, tmp_path):
        """`interpolate=1` is how a command line says yes, and it arrives int.

        `arg_utils` converts every numeric option value before the oracle sees
        it, so the string `"1"` branch never fires for the spelling the
        registry actually uses. Read as a ratio instead, 1.0 is a bound no two
        distinct row counts can meet -- run 8 asked for interpolation this way
        and refused 2118 operators for a gap "wider than the declared max gap
        ratio 1.0".
        """
        from atom.compass.core.cost.families import ParametricPriceLibrary

        default = ParametricPriceLibrary().max_gap_ratio
        assert default > 1.0
        for spelling in (1, 1.0, "1"):
            library = self._built(tmp_path, interpolate=spelling).oracle.library
            assert library.max_gap_ratio == default

    def test_a_ratio_under_one_is_refused_rather_than_clamped(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            self._built(tmp_path, interpolate=0.5)
        assert "interpolate" in str(exc.value)

    def test_a_value_that_is_neither_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            self._built(tmp_path, interpolate="sometimes")

    def test_the_graph_reaches_the_provider_and_not_just_the_price_list(
            self, tmp_path):
        """Without the graph a family has no structure to read a feature off.

        The triple is split before the library is built, so a provider handed
        only the first part would load every file as exact-signature-only.
        What it says about the file is how the two cases are told apart: given
        the graph it names that graph and what the graph itself lacks, and
        given no graph it says there was none.

        The two are reported separately because they are not the same loss. A
        file with no graph is exact-key-only and nothing can change that. A
        file WITH a graph that states no width of its own -- no embedding and
        no `body_rows_traced`, which is every head region -- is still read
        operator by operator for the families that declare where their width
        lives, so it lands in `no_file_width` rather than `unbuildable`.
        """
        graph = _template_file(tmp_path)
        library = self._built(tmp_path, interpolate="true").oracle.library
        assert graph in "".join(library.no_file_width.values())
        assert graph not in "".join(library.unbuildable.values())
        bare = build_source_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0, regions="source-27b-tp1-conc-v2",
            interpolate="true").oracle.library
        assert "no graph supplied" in "".join(bare.unbuildable.values())

    def test_the_limit_recorded_is_the_one_the_provider_holds(self, tmp_path):
        """`interpolate=true` names no number, and the number is the check.

        A report that says "interpolation: true" says nothing a reader can
        verify and moves silently if the provider's default ever does.
        """
        from atom.compass.core.cost.families import ParametricPriceLibrary

        assert self._built(tmp_path).interpolation_limit is None
        asked = self._built(tmp_path, interpolate="true")
        assert asked.interpolation_limit == ParametricPriceLibrary().max_gap_ratio
        assert self._built(tmp_path, interpolate=1.5).interpolation_limit == 1.5
