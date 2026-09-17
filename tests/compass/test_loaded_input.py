"""The helper's own guarantees, as distinct from what consumers do with it.

These are unit tests of one module and are not the evidence that any defect is
fixed -- each of those is a regression through the producer or validator that
had it, and lives with the change that fixes it. What is checked here is the
property every one of those depends on: that identity is taken from the bytes
that were parsed, at the moment they were parsed.
"""

import json

import pytest

from atom.compass.core import loaded_input


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


class TestIdentityIsOfTheBytesParsed:
    def test_the_payload_and_the_digest_come_from_one_read(self, tmp_path):
        import hashlib

        path = _write(tmp_path / "prices.json", {"prices": {"sig": 1}})
        payload, loaded = loaded_input.load_json(path, role="oracle.price")

        assert payload == {"prices": {"sig": 1}}
        with open(path, "rb") as handle:
            raw = handle.read()
        assert loaded.sha256 == hashlib.sha256(raw).hexdigest()
        assert loaded.size == len(raw)

    def test_replacing_the_file_afterwards_does_not_move_the_record(self, tmp_path):
        """The whole point. A digest re-derived from the path would follow the
        replacement; one taken at the read describes what was loaded."""
        path = _write(tmp_path / "prices.json", {"prices": {"sig": 1}})
        payload, loaded = loaded_input.load_json(path, role="oracle.price")
        before = loaded.sha256

        _write(tmp_path / "prices.json", {"prices": {"sig": 999}})

        assert payload == {"prices": {"sig": 1}}
        assert loaded.sha256 == before
        assert (
            loaded.sha256 != loaded_input.load_json(path, role="oracle.price")[1].sha256
        )

    def test_repointing_a_symlink_afterwards_does_not_move_the_record(self, tmp_path):
        real = _write(tmp_path / "real.json", {"prices": {"sig": 1}})
        other = _write(tmp_path / "other.json", {"prices": {"sig": 2}})
        link = tmp_path / "prices.json"
        link.symlink_to(real)

        _, loaded = loaded_input.load_json(str(link), role="oracle.price")
        before = loaded.sha256
        link.unlink()
        link.symlink_to(other)

        assert loaded.sha256 == before


class TestRankResolutionKeepsBothEnds:
    def test_this_ranks_own_file_is_recorded_as_its_own(self, tmp_path):
        _write(tmp_path / "prices.tp1.json", {"prices": {}})
        stem = str(tmp_path / "prices.json")

        _, loaded = loaded_input.load_json(stem, role="oracle.price", coords={"tp": 1})

        assert (
            loaded.requested == stem
        ), "the option's stem is the thing a report quotes"
        assert loaded.path.endswith("prices.tp1.json")
        assert loaded.rank_own is True
        assert loaded.rank_coords == (("tp", 1),)

    def test_the_shared_file_is_recorded_as_not_this_ranks(self, tmp_path):
        stem = _write(tmp_path / "prices.json", {"prices": {}})

        _, loaded = loaded_input.load_json(stem, role="oracle.price", coords={"tp": 3})

        assert loaded.path == stem
        assert loaded.rank_own is False

    def test_a_caller_that_resolved_first_would_have_lost_the_stem(self, tmp_path):
        """Why `load_json` takes the unresolved path: resolving twice reports a
        rank's own file as a shared one, under a name nothing asked for."""
        mine = _write(tmp_path / "prices.tp1.json", {"prices": {}})

        _, twice = loaded_input.load_json(mine, role="oracle.price", coords={"tp": 1})

        assert twice.rank_own is False
        assert twice.requested != str(tmp_path / "prices.json")


class TestRolesAreNamespacedAndOpen:
    def test_a_nested_role_is_carried_as_given(self, tmp_path):
        path = _write(tmp_path / "collective.json", {})
        _, loaded = loaded_input.load_json(path, role="runtime.memory_model.collective")
        assert loaded.role == "runtime.memory_model.collective"

    def test_the_same_bytes_in_two_roles_are_two_different_inputs(self, tmp_path):
        """`oracle.replay_target` and `runtime.replay_target` are read by
        different code for different purposes. One cannot stand in for the
        other, so a rolled digest must not collapse them."""
        path = _write(tmp_path / "target.json", {"version": 1})
        _, as_oracle = loaded_input.load_json(path, role="oracle.replay_target")
        _, as_runtime = loaded_input.load_json(path, role="runtime.replay_target")

        assert as_oracle.sha256 == as_runtime.sha256
        assert loaded_input.roll([as_oracle]) != loaded_input.roll([as_runtime])
        assert loaded_input.roll([as_oracle, as_runtime]) != loaded_input.roll(
            [as_oracle]
        )

    def test_an_input_with_no_role_is_refused(self, tmp_path):
        path = _write(tmp_path / "prices.json", {})
        with pytest.raises(ValueError, match="role"):
            loaded_input.load_json(path, role="")


class TestManifest:
    def test_it_holds_every_input_sorted_and_rolls_them(self, tmp_path):
        a = _write(tmp_path / "b.json", {"n": 1})
        b = _write(tmp_path / "a.json", {"n": 2})
        _, first = loaded_input.load_json(a, role="oracle.template")
        _, second = loaded_input.load_json(b, role="oracle.price")

        out = loaded_input.manifest([first, second], coords={"tp": 2})

        assert [row["role"] for row in out["inputs"]] == [
            "oracle.price",
            "oracle.template",
        ]
        assert out["rank_coords"] == {"tp": 2}
        assert out["rolled_sha256"] == loaded_input.roll([first, second])

    def test_a_changed_member_changes_the_roll(self, tmp_path):
        path = _write(tmp_path / "prices.json", {"n": 1})
        _, before = loaded_input.load_json(path, role="oracle.price")
        _write(tmp_path / "prices.json", {"n": 2})
        _, after = loaded_input.load_json(path, role="oracle.price")

        assert (
            loaded_input.manifest([before])["rolled_sha256"]
            != loaded_input.manifest([after])["rolled_sha256"]
        )

    def test_coordinates_are_taken_from_the_inputs_when_not_given(self, tmp_path):
        _write(tmp_path / "prices.tp1.json", {})
        _, loaded = loaded_input.load_json(
            str(tmp_path / "prices.json"), role="oracle.price", coords={"tp": 1}
        )
        assert loaded_input.manifest([loaded])["rank_coords"] == {"tp": 1}

    def test_a_rank_that_loaded_nothing_is_a_state_and_not_an_error(self):
        out = loaded_input.manifest([])
        assert out["inputs"] == []
        assert out["rank_coords"] == {}
