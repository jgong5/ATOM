"""Whether the process that predicted could have reached a device.

The gate is a GPU-free replay, so the claim is about a process. What stood for
it was a probe that ran afterwards, in its own container, and hashed the
artifacts it found there -- a true statement about a container and a directory,
and its own output says so. A GPU-container replay followed by a CPU-container
probe of the same shared files satisfies it exactly, which is the hole these
close.

The probe is kept: it covers the container the artifacts sat in. What is
required as well is the predictor's own reading of itself, at launch and at
readback, bound to the server that answered the requests by two accounts that
no single submission produced.
"""

import json

from . import test_cc_traces_validate as base

cell = base.cell
run = base.run
verdict = base.verdict
validate = base.validate
_write = base._write
_device_freedom = base._device_freedom
_device_reading = base._device_reading
_identity = base._identity


def _with_device_record(cell_dir, record, *, same_machine=False):
    """Replace what the predicting process said about itself.

    ``same_machine`` puts the fixture server's host and boot onto the record.
    A record produced here names this machine; the fixture server is a
    fabricated `cell-host`, and the harness's own startup-versus-service check
    is pinned to that. So the two are made to agree about the thing that is
    not under test, by moving the side that is fabricated anyway -- the
    record's device readings, which *are* under test, are left exactly as the
    producer wrote them.
    """
    path = cell_dir / "modelled.r1.json"
    blob = json.loads(path.read_text())
    ranks = blob["run"]["server"]["compass"]["loaded_inputs"]["ranks"]
    if record is None:
        ranks[0].pop("device_freedom", None)
    else:
        if same_machine:
            served = blob["run"]["server"].get("server_process") or {}
            for reading in ("launch", "readback"):
                process = record[reading]["process"]
                process["host"] = served.get("host")
                process["boot_id"] = served.get("boot_id")
        ranks[0]["device_freedom"] = record
    _write(path, blob)


def _failures(cell_dir):
    return verdict(cell_dir)["failures"]


class TestAProbeOfTheArtifactsIsNotEnough:

    def test_a_run_with_no_reading_from_its_predictor_is_refused(self, cell):
        """The cell still carries a passing `gpu_free.json`. That probe ran in
        a device-free container over these very artifacts, and it is exactly
        what a CPU-container probe of a GPU-container replay would produce."""
        _with_device_record(cell, None)

        assert run(cell) == 1
        assert any("rests on a probe of the container the artifacts were "
                   "later found in" in f for f in _failures(cell))

    def test_the_probe_itself_is_still_required(self, cell):
        """Not replaced. It covers the container the artifacts sat in, which
        is a different question from which process predicted."""
        (cell / validate.GPU_FREE_EVIDENCE).unlink()

        assert run(cell) == 1
        assert any("gpu_free.json" in f for f in _failures(cell))


class TestTheReadingMustSayTheProcessWasDeviceFree:

    def test_a_device_node_at_launch_is_refused(self, cell):
        nodes = {n: False for n in validate.DEVICE_NODES}
        nodes["/dev/kfd"] = True
        _with_device_record(cell, {
            "launch": _device_reading("launch", nodes=nodes),
            "readback": _device_reading("readback")})

        assert run(cell) == 1
        assert any("could reach /dev/kfd at launch" in f
                   for f in _failures(cell))

    def test_a_device_node_appearing_later_is_refused(self, cell):
        """Why there are two readings. A node bind-mounted in after startup is
        invisible to a check that only looked once."""
        nodes = {n: False for n in validate.DEVICE_NODES}
        nodes["/dev/dri"] = True
        _with_device_record(cell, {
            "launch": _device_reading("launch"),
            "readback": _device_reading("readback", nodes=nodes)})

        assert run(cell) == 1
        assert any("could reach /dev/dri at readback" in f
                   for f in _failures(cell))

    def test_a_driver_handle_this_process_held_is_refused(self, cell):
        _with_device_record(cell, {
            "launch": _device_reading(
                "launch", handles=[{"fd": "7", "target": "/dev/kfd"}]),
            "readback": _device_reading("readback")})

        assert run(cell) == 1
        assert any("driver handle(s) open at launch" in f
                   for f in _failures(cell))

    def test_only_one_reading_is_refused(self, cell):
        _with_device_record(cell, {"launch": _device_reading("launch")})

        assert run(cell) == 1
        assert any("only one device reading" in f for f in _failures(cell))


class TestTheReadingIsBoundToTheProcessThatPredicted:

    def test_two_different_processes_are_refused(self, cell):
        """One process launched and another read back: only one of them
        predicted, and the record does not say which."""
        _with_device_record(cell, {
            "launch": _device_reading("launch",
                                      process=_identity(pid=4243, ticks=100)),
            "readback": _device_reading("readback",
                                        process=_identity(pid=9999, ticks=200))})

        assert run(cell) == 1
        assert any("two processes, and only one of them predicted" in f
                   for f in _failures(cell))

    def test_a_predictor_on_another_host_is_refused(self, cell):
        """The independent binding. The API server reads its own identity from
        its own /proc and reports `server_process`; the worker reads its own
        and reports it here. Neither is copied from the other, so a
        disagreement is two accounts disagreeing rather than one field
        contradicting itself."""
        elsewhere = _identity(pid=4243, host="another-host")
        _with_device_record(cell, {
            "launch": _device_reading("launch", process=elsewhere),
            "readback": _device_reading("readback", process=elsewhere)})

        assert run(cell) == 1
        assert any("somewhere other than where it was served" in f
                   for f in _failures(cell))

    def test_a_predictor_from_another_boot_is_refused(self, cell):
        other_boot = _identity(pid=4243,
                               boot="11111111-2222-3333-4444-555555555555")
        _with_device_record(cell, {
            "launch": _device_reading("launch", process=other_boot),
            "readback": _device_reading("readback", process=other_boot)})

        assert run(cell) == 1
        assert any("somewhere other than where it was served" in f
                   for f in _failures(cell))

    def test_a_reading_that_names_no_process_is_refused(self, cell):
        launch = _device_reading("launch")
        launch["process"] = {}
        _with_device_record(cell, {"launch": launch,
                                   "readback": _device_reading("readback")})

        assert run(cell) == 1
        assert any("only partly" in f for f in _failures(cell))


class TestAReadingThatOmitsSomethingIsNotAReadingThatFoundNothing:
    """Fail closed. An absent node list has nothing set to true, so "reports
    nothing" and "reports no devices" are the same answer to `any()` -- and
    they are opposite claims."""

    def test_a_partial_node_list_is_refused(self, cell):
        partial = _device_reading("launch")
        partial["device_nodes"] = {"/dev/kfd": False}
        _with_device_record(cell, {"launch": partial,
                                   "readback": _device_reading("readback")})

        assert run(cell) == 1
        assert any("does not report every device node" in f
                   for f in _failures(cell))

    def test_an_absent_node_list_is_refused(self, cell):
        blank = _device_reading("launch")
        blank.pop("device_nodes")
        _with_device_record(cell, {"launch": blank,
                                   "readback": _device_reading("readback")})

        assert run(cell) == 1
        assert any("does not report every device node" in f
                   for f in _failures(cell))

    def test_a_half_named_process_is_refused(self, cell):
        thin = _device_reading("launch")
        thin["process"] = dict(thin["process"], boot_id=None)
        _with_device_record(cell, {"launch": thin,
                                   "readback": _device_reading("readback")})

        assert run(cell) == 1
        assert any("only partly" in f for f in _failures(cell))

    def test_a_changed_mount_namespace_is_refused(self, cell):
        """The device readings are readings of a view of `/dev`. Two readings
        from different mount namespaces are readings of different machines."""
        moved = _device_reading("readback")
        moved["namespaces"] = dict(moved["namespaces"], mnt="mnt:[4026539999]")
        _with_device_record(cell, {"launch": _device_reading("launch"),
                                   "readback": moved})

        assert run(cell) == 1
        assert any("different views of the machine" in f
                   for f in _failures(cell))

    def test_an_unreported_namespace_is_refused(self, cell):
        blind = _device_reading("readback")
        blind["namespaces"] = {}
        _with_device_record(cell, {"launch": _device_reading("launch"),
                                   "readback": blind})

        assert run(cell) == 1
        assert any("does not report its mnt namespace" in f
                   for f in _failures(cell))


class TestTheRecordTheRealProducerWrites:
    """Through the actual producer, not a dictionary written by this test.

    `observe` reads this process's own `/proc`, and the mixin is what a served
    run calls. A regression built only from fabricated records would pin the
    validator against a shape nothing produces.
    """

    def _produced(self):
        import types

        from atom.compass.config import CompassConfig
        from atom.compass.runtime.predict import CompassPredictMixin

        stub = CompassPredictMixin.__new__(CompassPredictMixin)
        stub.__dict__["_compass_config_cache"] = CompassConfig(enabled=True)
        stub.config = types.SimpleNamespace()
        stub._build_oracle = lambda config: types.SimpleNamespace(
            compass_loaded_inputs=(), describe=lambda: "stub")
        stub._topology = lambda: {"tp": 1}
        stub._rank_coords = dict
        stub._warn_if_compiled = lambda: None
        stub._init_compass_state()
        return stub.compass_input_manifest()["device_freedom"]

    def test_what_it_writes_satisfies_the_validator(self, cell):
        """This test runs in the device-free container, so the real reading is
        a passing one and the cell passes with it in place."""
        _with_device_record(cell, self._produced(), same_machine=True)

        assert run(cell) == 0, _failures(cell)

    def test_it_names_the_process_that_took_it(self):
        import os

        produced = self._produced()

        assert produced["launch"]["process"]["pid"] == os.getpid()
        assert produced["readback"]["process"]["pid"] == os.getpid()
        assert produced["launch"]["process"]["boot_id"]

    def test_both_readings_come_from_one_process(self):
        from atom.compass.core import device_freedom

        produced = self._produced()

        assert device_freedom.same_process(produced["launch"],
                                           produced["readback"])

    def test_it_reports_every_node_the_validator_asks_about(self):
        produced = self._produced()

        assert set(produced["launch"]["device_nodes"]) == set(
            validate.DEVICE_NODES)

    def test_the_producer_and_the_validator_ask_about_the_same_nodes(self):
        """These two lists drifted by one entry while this was being written,
        and the reading that omitted `/dev/nvidia-uvm` passed as a reading
        that had found nothing. They are pinned equal rather than merged,
        because the validator must stay loadable without importing `atom`."""
        from atom.compass.core import device_freedom

        assert tuple(device_freedom.DEVICE_NODES) == tuple(
            validate.DEVICE_NODES)

    def test_a_gpu_container_run_is_not_rescued_by_a_later_cpu_probe(self, cell):
        """The hole, end to end.

        The cell keeps its passing `gpu_free.json` -- taken in a device-free
        container, over these artifacts, exactly as a CPU-container probe of a
        GPU-container replay would be. What changes is the one thing that
        probe could never have seen: the predictor's own reading says it could
        reach a device.
        """
        produced = self._produced()
        produced["launch"]["device_nodes"]["/dev/kfd"] = True
        produced["readback"]["device_nodes"]["/dev/kfd"] = True
        _with_device_record(cell, produced, same_machine=True)

        evidence = json.loads(
            (cell / validate.GPU_FREE_EVIDENCE).read_text())
        assert not any(evidence["device_nodes"].values()), (
            "the standalone probe still says the container was device-free")

        assert run(cell) == 1
        assert any("could reach /dev/kfd" in f for f in _failures(cell))


class TestTheRuntimeReadingDecidesNothing:
    """The fixture's readings already report four devices and cuda available.

    They are the replay interpreter's answers, and the replay bootstrap
    answers hardware queries from the captured target -- so they describe the
    deployment being modelled. The cell passes with them present, which is the
    assertion.
    """

    def test_a_cell_passes_with_a_nonzero_runtime_device_count(self, cell):
        record = _device_freedom()
        assert record["launch"]["reported_by_runtime"]["device_count"] == 4

        assert run(cell) == 0

    def test_the_reading_is_still_carried_for_a_reader(self, cell):
        run(cell)
        blob = json.loads((cell / "modelled.r1.json").read_text())
        ranks = blob["run"]["server"]["compass"]["loaded_inputs"]["ranks"]
        seen = ranks[0]["device_freedom"]["launch"]

        assert seen["reported_by_runtime"]["cuda_available"] is True
        assert "not counted" in seen["runtime_note"]
