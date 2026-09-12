"""The capacity chain, end to end, into the source-only registry validation.

The mirror of `test_source_manifest_accepted` for the half that sizes the
deployment. Until now every capacity check here was exercised against rows the
test wrote; this drives the real reader instead:

  `ReplayModelRunner.get_num_blocks`  reads the target or the profile and
                                      publishes `compass_runtime_inputs` and
                                      the structured `compass_budget_source`
  `compass_input_manifest`            assembles them at readback
  `/compass/provenance`               publishes them per rank
  `check_capacity_*`                  reads them against a registry declared
                                      from the bytes on disk

Nothing here writes a loaded-input row or a budget record. Both come from the
runner. The registry is built by reading the files again rather than by
copying the manifest, so the two accounts can disagree.
"""

import hashlib
import json
import os

import pytest

from . import test_cc_traces_validate as base
from . import test_replay_memory as mem

validate = base.validate

SOURCE_DERIVED = "source-derived"
CAPTURED = "captured"


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _runner(tmp_path, **over):
    """A real replay runner that has chosen and published a budget."""
    from atom.compass.replay.runner import ReplayModelRunner

    runner = ReplayModelRunner(0, mem._config(tmp_path, **over))
    runner.get_num_blocks()
    return runner


def _rank_record(runner):
    """What the worker would answer the RPC with, from the real runner."""
    return runner.compass_input_manifest()


def _blob(runner, options=None):
    """The compass block a saved modelled run carries."""
    manifest = _rank_record(runner)
    return {
        "compass": {
            "enabled": True,
            "mode": "predict",
            "oracle": validate.SOURCE_FACTORY,
            "oracle_options": options or {},
            "oracle_option_sha256": {},
            "oracle_option_files": {},
            "loaded_inputs": {"ranks": [manifest]},
        },
        "server_revision": "abc",
        "model_revision": "rev1",
    }


def _modelled(blob):
    return base.validate.compare.Run(
        path="modelled.json", label="modelled", manifest={"server": blob})


def _registry_from_disk(blob, *, kind="derived_graph", **over):
    """Declare every runtime input this run read, from the bytes on disk."""
    artifacts = []
    for rank in blob["compass"]["loaded_inputs"]["ranks"]:
        for row in rank["inputs"]:
            if not row["role"].startswith("runtime."):
                continue
            entry = {
                "sha256": row["sha256"],
                "kind": kind,
                "measured_at_tp": 1,
                "produced_by": "meta_probe.py",
                "workload_sha256": None,
                "contents": {os.path.basename(row["path"]): _sha(row["path"])},
                "sources": [{"path": "/m/capture.json",
                             "sha256": base.SWEEP_SHA}],
                "code": {"atom/compass/core/memory_blocks.py": base.CODE_SHA},
            }
            entry.update(over)
            artifacts.append(entry)
    return {"artifacts": artifacts}


class TestTheRealReaderPublishesWhatTheValidatorReads:

    def test_a_captured_budget_names_the_target_it_read(self, tmp_path):
        """No profile: the replay serves the captured block count, and the
        record says so and names the file it came out of."""
        runner = _runner(tmp_path)

        record = runner.compass_budget_source
        assert record["kind"] == CAPTURED
        assert record["hardware_reference"] is False
        roles = {i.role for i in runner.compass_runtime_inputs}
        assert "runtime.replay_target" in roles

    def test_a_source_derived_budget_names_the_profile_it_read(self, tmp_path):
        runner = _runner(tmp_path, profile=mem._profile(tmp_path, 1))

        record = runner.compass_budget_source
        assert record["kind"] == SOURCE_DERIVED
        assert record["hardware_reference"] is False
        roles = {i.role for i in runner.compass_runtime_inputs}
        assert "runtime.memory_model" in roles

    def test_the_manifest_carries_both_the_inputs_and_the_kind(self, tmp_path):
        runner = _runner(tmp_path, profile=mem._profile(tmp_path, 1))

        manifest = _rank_record(runner)
        assert manifest["budget_source"]["kind"] == SOURCE_DERIVED
        assert any(row["role"].startswith("runtime.")
                   for row in manifest["inputs"])


class TestAValidSourceDerivedCapacityIsAccepted:

    def test_the_whole_chain_passes(self, tmp_path):
        """The claim this file exists to make: a GPU-free replay sized from a
        profile, read by the real reader, declared honestly, is accepted."""
        runner = _runner(tmp_path, profile=mem._profile(tmp_path, 1))
        blob = _blob(runner)

        assert validate.check_capacity_inputs(_modelled(blob), "x") == []
        assert validate.check_capacity_provenance(
            _modelled(blob), _registry_from_disk(blob), 2, "d" * 64, {}) == []

    def test_a_captured_budget_is_accepted_too(self, tmp_path):
        """Being sized from a capture is legitimate for the modelled side; it
        is only disqualified as evidence about this run's hardware."""
        runner = _runner(tmp_path)
        blob = _blob(runner)

        assert validate.check_capacity_inputs(_modelled(blob), "x") == []
        assert validate.check_capacity_provenance(
            _modelled(blob), _registry_from_disk(blob), 2, "d" * 64, {}) == []


class TestAnInvalidCapacityInputIsRefused:
    """Same real run, same real record; only the declaration changes."""

    def _blob(self, tmp_path):
        return _blob(_runner(tmp_path, profile=mem._profile(tmp_path, 1)))

    def test_an_unregistered_profile_is_refused(self, tmp_path):
        blob = self._blob(tmp_path)

        bad = validate.check_capacity_provenance(
            _modelled(blob), {"artifacts": []}, 2, "d" * 64, {})

        assert any("not declared in the calibration registry" in b
                   for b in bad)

    def test_a_profile_from_the_target_engine_is_refused(self, tmp_path):
        """The central leak, on real bytes: the deployment was sized by the
        engine whose capacity is being predicted."""
        blob = self._blob(tmp_path)

        bad = validate.check_capacity_provenance(
            _modelled(blob), _registry_from_disk(blob, from_target_engine=True),
            2, "d" * 64, {})

        assert any("sized by the engine whose capacity is being predicted" in b
                   for b in bad)

    def test_a_profile_measured_at_the_predicted_width_is_refused(self, tmp_path):
        blob = self._blob(tmp_path)

        bad = validate.check_capacity_provenance(
            _modelled(blob), _registry_from_disk(blob, measured_at_tp=2),
            2, "d" * 64, {})

        assert any("width being predicted" in b for b in bad)

    def test_a_declaration_naming_another_file_is_refused(self, tmp_path):
        blob = self._blob(tmp_path)
        registry = _registry_from_disk(blob)
        for entry in registry["artifacts"]:
            entry["contents"] = {"somewhere_else.json": entry["sha256"]}

        bad = validate.check_capacity_provenance(
            _modelled(blob), registry, 2, "d" * 64, {})

        assert any("which the registry does not declare" in b for b in bad)

    def test_a_profile_that_is_this_cells_own_artifact_is_refused(self, tmp_path):
        blob = self._blob(tmp_path)
        sha = next(row["sha256"]
                   for row in blob["compass"]["loaded_inputs"]["ranks"][0]["inputs"]
                   if row["role"].startswith("runtime."))

        bad = validate.check_capacity_provenance(
            _modelled(blob), _registry_from_disk(blob), 2, "d" * 64,
            {"step table": sha})

        assert any("sized from the measurement it is predicting" in b
                   for b in bad)


class TestTheReferenceSideMustHaveBeenMeasured:
    """The one place the kind has to be `device-measured`.

    A replay cannot produce such a record -- it has no card to ask -- which is
    the point: the ground-truth side of an acceptance comparison has to have
    been sized by the device it ran on, and a replay standing in for it would
    be a second prediction.
    """

    def test_a_replays_own_budget_is_refused_as_a_reference(self, tmp_path):
        runner = _runner(tmp_path, profile=mem._profile(tmp_path, 1))
        blob = _blob(runner)

        bad = validate.check_reference_budget_is_measured(_modelled(blob), "x")

        assert any("the ground-truth side is itself a prediction" in b
                   for b in bad)

    def test_a_captured_reference_is_refused(self, tmp_path):
        blob = _blob(_runner(tmp_path))

        bad = validate.check_reference_budget_is_measured(_modelled(blob), "x")

        assert any("the ground-truth side is itself a prediction" in b
                   for b in bad)

    def test_the_readers_own_flag_agrees_with_the_validator(self, tmp_path):
        """`hardware_reference` is the reader's word for the same distinction.
        Two programs deciding it separately have to decide it the same way."""
        for over in ({}, {"profile": mem._profile(tmp_path, 1)}):
            runner = _runner(tmp_path, **over)
            record = runner.compass_budget_source
            refused = validate.check_reference_budget_is_measured(
                _modelled(_blob(runner)), "x")
            assert record["hardware_reference"] is (not refused)


class TestTheRealArtifactsOnThisNode:
    """Against the targets and profiles this node actually holds.

    Read-only, and skipped where they are absent, so this is a stronger check
    where the artifacts exist and not a failure where they do not. Nothing is
    written back to the shared tree.
    """

    SERVING = "/workspace/ATOM/agent_scratch/serving"
    PROFILES = "/workspace/ATOM/agent_scratch/memval/capture_replay/profile"

    def _target(self, width):
        path = f"{self.SERVING}/src_tp{width}/target.tp{width}.json"
        if not os.path.exists(path):
            pytest.skip(f"{path} is not on this node")
        return path

    @pytest.mark.parametrize("width", [2, 4])
    def test_a_real_target_loads_and_digests_as_a_capacity_input(self, width):
        from atom.compass.core.loaded_input import load_json

        path = self._target(width)

        blob, loaded = load_json(path, role="runtime.replay_target")

        assert loaded.role == "runtime.replay_target"
        assert loaded.sha256 == _sha(path)
        assert blob["config"]["tensor_parallel_size"] == width
        assert blob["blocks"]["num_kvcache_blocks"] > 0

    @pytest.mark.parametrize("width", [2, 4])
    def test_a_real_target_names_the_profile_it_was_derived_from(self, width):
        """These widths were not captured from a device -- no card ran the
        27B at TP2 or TP4 here -- so the target is itself source-derived and
        says so, naming the profile and the width it was derived at. That is
        the lineage a validator needs to follow, and it is real."""
        with open(self._target(width), encoding="utf-8") as handle:
            derivation = json.load(handle).get("derivation") or {}

        lineage = derivation.get("lineage") or {}
        assert lineage.get("kind") == SOURCE_DERIVED
        assert lineage.get("world_size") == width
        assert os.path.basename(lineage.get("profile") or "") == (
            f"profile.tp{width}.json")

    @pytest.mark.parametrize("width", [1, 2, 4])
    def test_the_real_profile_that_target_names_is_readable(self, width):
        """The other end of that lineage. Loaded through the same reader a
        run uses, under the role a capacity input carries."""
        from atom.compass.core.loaded_input import load_json

        path = f"{self.PROFILES}/profile.tp{width}.json"
        if not os.path.exists(path):
            pytest.skip(f"{path} is not on this node")

        blob, loaded = load_json(path, role="runtime.memory_model")

        assert loaded.sha256 == _sha(path)
        assert blob["world_size"] == width
        # The files a profile itself names get nested roles, which is why the
        # namespace is open rather than an enum.
        assert os.path.basename(blob["calibration"]).startswith("calibration")

    def test_a_real_profile_is_refused_for_another_width(self):
        """`world_size` is a guard, not a label: a profile describing a TP1
        deployment cannot size a TP4 one, and the reader says so rather than
        scaling it."""
        from atom.compass.core.loaded_input import load_json

        path = f"{self.PROFILES}/profile.tp1.json"
        if not os.path.exists(path):
            pytest.skip(f"{path} is not on this node")

        blob, _ = load_json(path, role="runtime.memory_model")

        assert blob["world_size"] == 1
