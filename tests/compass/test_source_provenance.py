"""What the source composition loaded, as the readers that parsed it say.

The defect these are about: nothing downstream could say what a served run had
actually read. The record was re-derived from the option string by a later
reader, and the option string is a DSL over per-rank stems -- so the files it
names are not the files that were opened, and by the time anyone asked, the
bytes could have changed.

These drive the real factory and the real bootstrap. Nothing is faked except
the conditions the defects need: a second module object for the adoption path,
and `aiter` standing in for "already imported".
"""

import hashlib
import inspect
import json
import sys
import types

from atom.compass.runtime.source_oracle import build_source_oracle

from .test_source_oracle import _template_file


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _by_role(composition):
    found = {}
    for loaded in composition.loaded_inputs:
        found.setdefault(loaded.role, []).append(loaded)
    return found


class TestTheTemplateMembersAreEnumerated:
    """A comma-separated option is a list, and each member is an input.

    `template=a.json,b.json` names two files. Every reader that went looking
    for one file called `a.json,b.json` found nothing, reported no digest, and
    left a run that had loaded two real graphs indistinguishable from a run
    that had loaded none.
    """

    def test_both_members_of_a_comma_separated_option_are_recorded(self, tmp_path):
        first = _template_file(tmp_path, "graph.json")
        second = _template_file(tmp_path, "other.json")

        built = build_source_oracle(template=f"{first},{second}", derive=0)

        templates = _by_role(built)["oracle.template"]
        assert sorted(t.path for t in templates) == sorted([first, second])
        assert {t.sha256 for t in templates} == {_digest(first), _digest(second)}

    def test_a_head_template_is_its_own_role(self, tmp_path):
        """A body graph and a head graph are different inputs to the
        prediction, and a record that called both `template` could not say
        which region a given file priced."""
        body = _template_file(tmp_path, "body.json")
        head = _template_file(tmp_path, "head.json")

        built = build_source_oracle(template=body, head_template=head,
                                    head=True, derive=0)

        roles = _by_role(built)
        assert [t.path for t in roles["oracle.template"]] == [body]
        assert [t.path for t in roles["oracle.head_template"]] == [head]

    def test_replacing_a_template_after_the_build_does_not_move_the_record(
            self, tmp_path):
        """The reopening defect, at the level a consumer sees it.

        A digest taken later describes the file that is there later. This one
        is taken as the graph is parsed, so it goes on describing the graph
        the oracle is actually holding.
        """
        path = _template_file(tmp_path, "graph.json")
        built = build_source_oracle(template=path, derive=0)
        before = _by_role(built)["oracle.template"][0].sha256

        with open(path, encoding="utf-8") as handle:
            graph = json.load(handle)
        graph["provenance"]["execution"]["capture_bucket"] = 64
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(graph, handle)

        assert _by_role(built)["oracle.template"][0].sha256 == before
        assert _digest(path) != before


class TestTheServedSeamCarriesIt:
    """`source_cost_oracle` is what a served run names, and it returns the
    oracle alone. A record held only in the composition would describe the
    diagnostic path and never the path that actually has to be attributable --
    `CompassPredictMixin` is handed the oracle and nothing else."""

    def test_the_oracle_a_served_run_gets_carries_what_was_loaded(self, tmp_path):
        from atom.compass.runtime.source_oracle import source_cost_oracle

        path = _template_file(tmp_path, "graph.json")

        oracle = source_cost_oracle(template=path, derive=0)

        loaded = list(oracle.compass_loaded_inputs)
        assert [i.path for i in loaded] == [path]
        assert loaded[0].sha256 == _digest(path)

    def test_what_rides_on_the_oracle_cannot_be_edited(self, tmp_path):
        from atom.compass.runtime.source_oracle import source_cost_oracle

        oracle = source_cost_oracle(template=_template_file(tmp_path),
                                    derive=0)

        assert isinstance(oracle.compass_loaded_inputs, tuple)

    def test_it_is_the_same_record_the_composition_reports(self, tmp_path):
        from atom.compass.runtime.source_oracle import source_cost_oracle

        path = _template_file(tmp_path, "graph.json")
        built = build_source_oracle(template=path, derive=0)
        served = source_cost_oracle(template=path, derive=0)

        assert built.oracle.compass_loaded_inputs == built.loaded_inputs
        assert ([i.sha256 for i in served.compass_loaded_inputs]
                == [i.sha256 for i in built.loaded_inputs])


class TestThePerRankFileIsTheOneRecorded:

    def test_the_stem_is_kept_beside_the_file_that_was_served(self, tmp_path):
        """Both ends of the resolution. The stem is what the option carried;
        the suffixed file is what this rank read. A record holding only one of
        them cannot say whether rank 1 measured itself or borrowed rank 0."""
        mine = _template_file(tmp_path, "graph.tp1.json")
        stem = str(tmp_path / "graph.json")

        built = build_source_oracle(template=stem, derive=0,
                                    rank_coords="tp:1")

        loaded = _by_role(built)["oracle.template"][0]
        assert loaded.requested == stem
        assert loaded.path == mine
        assert loaded.rank_own is True
        assert loaded.rank_coords == (("tp", 1),)

    def test_falling_back_to_the_shared_file_is_recorded_as_a_fallback(
            self, tmp_path):
        shared = _template_file(tmp_path, "graph.json")

        built = build_source_oracle(template=shared, derive=0,
                                    rank_coords="tp:3")

        loaded = _by_role(built)["oracle.template"][0]
        assert loaded.path == shared
        assert loaded.rank_own is False


class TestTheReplayTargetsAreTwoInputs:
    """One process reads two targets, for two purposes, and kept one slot.

    `replay_server` reads the deployment's own target to bootstrap the
    interpreter. `ModelTracer.build` then reads whatever the factory's
    `replay_target=` option names, to answer the derivation's architecture
    query. Same function, different files, different meanings -- and the
    second call lands after `aiter` is imported, where `install` either
    returns early or adopts another module's state. Both of those paths lost a
    read.
    """

    @staticmethod
    def _clean(monkeypatch):
        from atom.compass.replay import bootstrap

        for name in list(sys.modules):
            if name == "jax" or name.startswith("jax."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.delitem(sys.modules, "aiter", raising=False)
        for finder in list(sys.meta_path):
            if type(finder).__name__ == "_ChipInfoFinder":
                sys.meta_path.remove(finder)
        monkeypatch.setattr(bootstrap, "_STATE",
                            {"installed": False, "arch": None,
                             "gpu_archs": None, "source": None, "reason": None,
                             "calls": 0, "chip_info_hook": False,
                             "chip_info_calls": 0, "redundant_installs": 0,
                             "inputs": []})
        monkeypatch.setattr(bootstrap, "_live_arch", lambda: None)
        monkeypatch.setattr(bootstrap, "_jax_installed", lambda: False)
        return bootstrap

    @staticmethod
    def _target(tmp_path, name, device_name):
        path = tmp_path / name
        path.write_text(json.dumps({
            "version": 1,
            "hardware": {"arch": "gfx942:sramecc+:xnack-",
                         "device_name": device_name},
        }), encoding="utf-8")
        return str(path)

    def test_a_second_target_read_after_install_keeps_both_identities(
            self, tmp_path, monkeypatch):
        """The arch is unchanged, so the second install is redundant and
        returns early -- but the file it read is a different file, and that
        read is the one the factory option is accountable for."""
        bootstrap = self._clean(monkeypatch)
        runtime = self._target(tmp_path, "runtime.json", "MI308X-a")
        oracle = self._target(tmp_path, "oracle.json", "MI308X-b")
        assert _digest(runtime) != _digest(oracle)

        bootstrap.install_from_target(runtime)
        monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
        state = bootstrap.install_from_target(oracle,
                                              role="oracle.replay_target")

        assert state["redundant_installs"] == 1, "the same arch, asked twice"
        by_role = {row["role"]: row for row in state["inputs"]}
        assert set(by_role) == {"bootstrap.replay_target",
                                "oracle.replay_target"}
        assert by_role["bootstrap.replay_target"]["sha256"] == _digest(runtime)
        assert by_role["oracle.replay_target"]["sha256"] == _digest(oracle)

    def test_adopting_another_modules_state_does_not_drop_this_ones_read(
            self, tmp_path, monkeypatch):
        """`_sitedir` loads this file by path, so the module that installed is
        not the module the tracer imports. Adoption reads that state back --
        and a plain overwrite replaced the read the tracer had just taken with
        the one the other copy held."""
        bootstrap = self._clean(monkeypatch)
        runtime = self._target(tmp_path, "runtime.json", "MI308X-a")
        oracle = self._target(tmp_path, "oracle.json", "MI308X-b")

        # The path-loaded copy, as `_process_state` finds it: same file, other
        # module object, already installed, holding its own read.
        other = types.ModuleType("atom_compass_replay_bootstrap")
        other.__file__ = bootstrap.__file__
        other._STATE = {"installed": True, "arch": "gfx942:sramecc+:xnack-",
                        "source": "sitedir", "reason": None, "calls": 0,
                        "chip_info_hook": True, "chip_info_calls": 0,
                        "redundant_installs": 0,
                        "inputs": [{"role": "bootstrap.replay_target",
                                    "requested": runtime, "path": runtime,
                                    "rank_own": False,
                                    "sha256": _digest(runtime),
                                    "size": 0, "rank_coords": {}}]}
        monkeypatch.setitem(sys.modules, other.__name__, other)
        monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))

        state = bootstrap.install_from_target(oracle,
                                              role="oracle.replay_target")

        assert state["adopted_from"] == "atom_compass_replay_bootstrap"
        by_role = {row["role"]: row for row in state["inputs"]}
        assert by_role["bootstrap.replay_target"]["path"] == runtime
        assert by_role["oracle.replay_target"]["sha256"] == _digest(oracle), (
            "the factory option must still carry the identity of the bytes it "
            "read, after adoption replaced everything else in the state")

    def test_the_bootstrap_read_is_not_offered_as_an_oracle_input(
            self, tmp_path, monkeypatch):
        """The composition collects `oracle.replay_target` only. The
        deployment's own target belongs to the runtime side of the record and
        must not be able to stand in for a factory option nobody passed."""
        bootstrap = self._clean(monkeypatch)
        runtime = self._target(tmp_path, "runtime.json", "MI308X-a")
        bootstrap.install_from_target(runtime)

        roles = {row["role"] for row in bootstrap.state()["inputs"]}
        assert roles == {"bootstrap.replay_target"}

    def test_state_hands_back_a_copy_of_the_reads(self, tmp_path, monkeypatch):
        bootstrap = self._clean(monkeypatch)
        bootstrap.install_from_target(
            self._target(tmp_path, "runtime.json", "MI308X-a"))

        held = bootstrap.state()["inputs"]
        held.append({"role": "made up"})

        assert len(bootstrap.state()["inputs"]) == 1


class TestThePriceReaderSeam:
    """Prices arrive once the library reports its own reads.

    Until then they are *absent* from the record rather than guessed at, which
    is the safe direction: a consumer reads an absent input as unrecorded,
    where a re-derived one would read as evidence.
    """

    def test_the_factory_takes_whatever_the_library_reports(self, tmp_path,
                                                            monkeypatch):
        from atom.compass.core.loaded_input import LoadedInput
        from atom.compass.runtime import source_oracle

        stub = LoadedInput(role="oracle.price", requested="prices.json",
                           path="prices.tp1.json", rank_own=True,
                           sha256="ab" * 32, size=17, rank_coords=(("tp", 1),))

        class _Library:
            loaded_inputs = (stub,)
            max_gap_ratio = None

        monkeypatch.setattr(source_oracle, "_price_library",
                            lambda *a, **k: _Library())

        built = build_source_oracle(template=_template_file(tmp_path), derive=0)

        assert stub in built.loaded_inputs

    def test_a_library_that_reports_nothing_contributes_nothing(self, tmp_path):
        built = build_source_oracle(template=_template_file(tmp_path), derive=0)
        assert not [i for i in built.loaded_inputs
                    if i.role.startswith("oracle.price")]
class TestResolutionHappensOnce:
    """Only the reader resolves, and it does so as it opens the file.

    That is what lets one record hold both ends: the stem the option carried,
    and the per-rank file this rank was served. A factory that resolved on the
    way in would hand the library a suffixed path it had no way to recognise
    as a rank's own, and every record would read `rank_own: false` under a
    name nothing asked for.
    """

    def test_the_library_takes_the_rank_and_resolves_for_itself(self):
        from atom.compass.core.cost.library import PriceLibrary

        assert "coords" in inspect.signature(PriceLibrary.add).parameters
        assert "coords" in inspect.signature(PriceLibrary.load).parameters

    def test_the_factory_hands_the_option_stem_through_unresolved(self, tmp_path):
        from atom.compass.runtime.source_oracle import build_source_group

        from .test_source_oracle import _price_file

        # The group builds every rank, so rank 0 needs something to read too:
        # the shared list, which is what it falls back to.
        stem = _price_file(tmp_path, "prices.json")
        _price_file(tmp_path, "prices.tp1.json")
        built = build_source_group(price=stem, tp=2, derive=0,
                                   require_complete=0, regions="none",
                                   template=_template_file(tmp_path),
                                   rank_coords={"tp": 1})

        mine = [i for i in built.loaded_inputs
                if i.role == "oracle.price" and i.rank_coords == (("tp", 1),)]
        assert mine, built.loaded_inputs
        assert mine[0].requested == stem
        assert mine[0].path.endswith("prices.tp1.json")
        assert mine[0].rank_own is True
