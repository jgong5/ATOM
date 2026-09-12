"""The reference side of an acceptance comparison, on the path that produces it.

Every other test of the reference guard is a negative from a replay: a replay
has no card to ask, so it cannot produce a `device-measured` record and the
guard refuses it. That establishes the guard refuses, and nothing about whether
the positive case composes -- whether the native runner's own publication
actually produces a record the API can carry and the validator will accept.
Left there, the first time anyone found out would have been on hardware.

So this drives the native path, on CPU. `CompassModelRunner` imports AITER,
which resolves a chip at import time; the replay bootstrap answers that from a
captured architecture, which is what it is for, and no device, no model and no
kernel is involved either way. What is stubbed is the sensor boundary -- the
four device calls the budget arithmetic reads -- and nothing above it:
`_recorded_readings` chooses the branch, `_publish_budget_source` publishes,
and `budget_source` builds the record, all of them the real ones.

The refusal that matters here is the one the flags cannot express:
`mode="measure"` forces the wall clock and says nothing about memory, so a run
that looks measured in every other field can have been sized analytically. On
this path it publishes `source-derived`, and the reference guard refuses it.
"""

import json
import types

import pytest

from . import test_cc_traces_validate as base

validate = base.validate

#: What the card would have said. Plausible MI308X numbers; nothing here
#: depends on them being exact, only on their coming from this boundary.
TOTAL = 192 * 2**30
FREE = 150 * 2**30


@pytest.fixture(scope="module")
def native():
    """`atom.compass.runtime.runner`, imported with the arch answered.

    Skipped rather than failed where the bootstrap cannot answer: this is a
    CPU container by design, and a node that refuses the import has told us
    something about itself rather than about the code.
    """
    from atom.compass.replay import bootstrap

    try:
        if not bootstrap.state().get("installed"):
            bootstrap.install("gfx942:sramecc+:xnack-", source="test")
        from atom.compass.runtime import runner
    except Exception as exc:  # noqa: BLE001 - environment-dependent
        pytest.skip(f"the native runner is not importable here: {exc}")
    return runner


class _Sensors:
    """A runner stub that answers only at the boundary the readings come from.

    Everything the real class does above these -- choosing the branch,
    publishing the record, building it -- is bound off `CompassModelRunner`
    below and runs unmodified.
    """

    def __init__(self, native, *, modelled=None, recorded=None, lineage=None):
        self._modelled = modelled
        self._recorded = recorded
        self._modelled_lineage = lineage
        self._compass_runtime_inputs = []
        self.compass_budget_source = None
        self.substituted = None
        self.config = types.SimpleNamespace(
            model="Qwen/Qwen3.8-27B", tensor_parallel_size=1,
            gpu_memory_utilization=0.9, max_model_len=262144,
            max_num_seqs=32, max_num_batched_tokens=16384,
            kv_cache_block_size=16, enforce_eager=False,
            kv_cache_dtype="bf16")
        self._compass_config = types.SimpleNamespace(
            memory_model="", memory_in="", mode="measure",
            replay_target="", replay_target_out="")
        self._recorded_readings = types.MethodType(
            native.CompassModelRunner._recorded_readings.__wrapped__, self)
        self._publish_budget_source = types.MethodType(
            native.CompassModelRunner._publish_budget_source, self)

    # -- the sensor boundary -------------------------------------------
    def _modelled_readings(self):
        return self._modelled

    def _recorded_memory(self):
        return self._recorded

    def _memory_config(self):
        return {"model": self.config.model}

    def _expected_non_torch(self):
        return None

    def _substitute(self, readings):
        self.substituted = dict(readings)
        return {}

    def _restore(self, was):
        return None

    @property
    def compass_runtime_inputs(self):
        return tuple(self._compass_runtime_inputs)


def _sized(stub):
    """Run the branch, as `get_num_blocks` runs it, and fill the count in.

    The count is what the engine's own arithmetic returns from the readings;
    nothing here recomputes it, so a plausible number stands in for the
    arithmetic this test is not about.
    """
    import contextlib

    with contextlib.contextmanager(lambda: stub._recorded_readings({}))():
        pass
    if isinstance(stub.compass_budget_source, dict):
        stub.compass_budget_source["num_kvcache_blocks"] = 266835
    return stub.compass_budget_source


def _blob(record, *, mode="measure"):
    """A saved run's compass block, carrying the record the path produced."""
    return {
        "compass": {
            "enabled": True,
            "mode": mode,
            "oracle": validate.SOURCE_FACTORY,
            "oracle_options": {},
            "oracle_option_sha256": {},
            "oracle_option_files": {},
            "loaded_inputs": {"ranks": [{
                "rank_coords": {},
                "inputs": [],
                "rolled_sha256": "0" * 64,
                "budget_source": record,
            }]},
        },
        "server_revision": "abc",
        "model_revision": "rev1",
    }


def _run(blob):
    return base.validate.compare.Run(
        path="real.json", label="real", manifest={"server": blob})


class TestTheNativePathProducesAMeasuredRecord:

    def test_asking_the_card_publishes_device_measured(self, native):
        """No profile and no record: the branch falls through to the device,
        and says so rather than being read as replayed."""
        record = _sized(_Sensors(native))

        assert record["kind"] == "device-measured"
        assert record["hardware_reference"] is True
        assert record["served"] is True
        assert record["schema"]

    def test_the_record_carries_what_it_sized(self, native):
        record = _sized(_Sensors(native))

        assert record["num_kvcache_blocks"] == 266835
        assert record["deployment"]["max_num_seqs"] == 32

    def test_nothing_was_substituted_for_the_readings(self, native):
        """The device branch leaves the engine's four calls alone; that is
        what makes the record a measurement."""
        stub = _Sensors(native)
        _sized(stub)

        assert stub.substituted is None


class TestTheMeasuredRecordSatisfiesTheReferenceGuard:

    def test_it_passes_where_every_replay_record_is_refused(self, native):
        """The positive case, which until now nothing established: the record
        this path produces is one the guard accepts."""
        blob = _blob(_sized(_Sensors(native)))

        assert validate.check_reference_budget_is_measured(_run(blob), "x") == []

    def test_the_same_record_unserved_is_refused(self, native):
        """`kind` and `served` are different claims. A budget taken off the
        card and then not used describes some other deployment."""
        record = dict(_sized(_Sensors(native)), served=False)

        bad = validate.check_reference_budget_is_measured(_run(_blob(record)), "x")

        assert any("does not say it served" in b for b in bad)

    def test_it_survives_the_provenance_endpoint(self, monkeypatch, native):
        """Through the real endpoint, because the record has to reach the
        validator by that route on a real run."""
        import asyncio
        import importlib

        from atom.compass.config import CompassConfig

        try:
            api_server = importlib.import_module(
                "atom.entrypoints.openai.api_server")
        except Exception:  # noqa: BLE001 - environment-dependent
            pytest.skip("api_server import unavailable")

        record = _sized(_Sensors(native))
        manifest = {"rank_coords": {}, "inputs": [], "rolled_sha256": "0" * 64,
                    "budget_source": record}
        engine = types.SimpleNamespace(
            config=types.SimpleNamespace(
                compass_config=CompassConfig(
                    enabled=True, mode="measure",
                    measure_out="/tmp/steps.jsonl",
                    oracle_qualname=validate.SOURCE_FACTORY,
                    oracle_options={}),
                model_config=None, parallel_config=None),
            get_compass_inputs=lambda timeout=10.0: {"ranks": [manifest]})
        monkeypatch.setattr(api_server, "engine", engine)

        blob = asyncio.run(api_server.compass_provenance())

        published = blob["compass"]["loaded_inputs"]["ranks"][0]
        assert published["budget_source"]["kind"] == "device-measured"
        assert validate.check_reference_budget_is_measured(
            _run(blob), "x") == []


class TestAMeasureModeRunSizedAnalyticallyIsRefused:
    """The refusal the flags cannot express.

    `CompassConfig` forces the wall clock for `mode="measure"` and says
    nothing about memory, and `_recorded_readings` honours a profile in every
    mode. So a run whose mode, clock and timings all look like a measurement
    can have served an analytically derived capacity -- and as a *reference*
    that makes the comparison two models rather than a model and a machine.
    """

    def test_a_profile_in_measure_mode_publishes_source_derived(self, native):
        """The branch does not consult the mode, and this is why that is
        right: what it publishes is what it did."""
        stub = _Sensors(native, modelled={"total": TOTAL, "free": FREE,
                                          "peak_torch": 0, "non_torch": 0,
                                          "cudagraph_overhead": 0},
                        lineage={"profile": "/m/profile.tp1.json",
                                 "world_size": 1})
        record = _sized(stub)

        assert stub._compass_config.mode == "measure"
        assert record["kind"] == "source-derived"
        assert record["hardware_reference"] is False
        assert stub.substituted["total"] == TOTAL, "the readings were replaced"

    def test_that_record_is_refused_as_a_reference(self, native):
        stub = _Sensors(native, modelled={"total": TOTAL, "free": FREE,
                                          "peak_torch": 0, "non_torch": 0,
                                          "cudagraph_overhead": 0})
        blob = _blob(_sized(stub), mode="measure")

        bad = validate.check_reference_budget_is_measured(_run(blob), "x")

        assert any("the ground-truth side is itself a prediction" in b
                   for b in bad)

    def test_the_mode_alone_would_not_have_caught_it(self, native):
        """Everything a mode-based check could read still says measure."""
        stub = _Sensors(native, modelled={"total": TOTAL, "free": FREE,
                                          "peak_torch": 0, "non_torch": 0,
                                          "cudagraph_overhead": 0})
        blob = _blob(_sized(stub), mode="measure")

        assert blob["compass"]["mode"] == "measure"
        assert json.dumps(blob).count("source-derived") == 1, (
            "the kind is the only field that says so")
