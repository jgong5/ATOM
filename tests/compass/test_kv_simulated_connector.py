# SPDX-License-Identifier: MIT
"""The simulated KV connector: what it charges, and when it lets go.

Every timing claim below is made against the connector the engine's own
factory built, driven through the engine's own metadata objects, and read
through `get_finished` -- not against the arithmetic helper underneath it. The
helper is exact by construction; what is worth testing is that the announced
transfer, the priced one and the released one are the same transfer.

The clock is a list holding one number. That is the whole point: the connector
reads the time the harness hands it, and a test that can put the clock one
nanosecond before a deadline and see nothing released is a test that the
release turns on the deadline rather than on the call.

The machine spec is the complete document the spec tests own, so there is one
valid spec in the suite rather than two that can drift apart, and the two
bandwidths below are made by editing a copy of it -- which means they travel
through the schema's peak-and-derate rule the way a real one would.
"""

import copy
import json
import pathlib
from types import SimpleNamespace

import pytest
from conftest import atom_config_double
from test_spec_schema import DOCUMENT
from transformers import PretrainedConfig

import atom.compass.kv as compass_kv
from atom.compass.backends import KvGeometry
from atom.compass.kv import (
    CLOCK_KEY,
    TRANSFER_KEY,
    Scope,
    SimulatedKVConnector,
    SimulatedKVConnectorScheduler,
    TransferModel,
    UnboundSeam,
)
from atom.compass.spec import MachineSpec, SpecRefusal
from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput

CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
BLOCK_SIZE = 64

#: A nanosecond: the "one tick before the deadline" the release must not take.
TICK = 1.0e-9

#: A clock reading that is not zero, so an issue time cannot hide in the sum.
ISSUE_AT = 12.5

#: The two link speeds of the named result, as spec peaks. The document's own
#: derate turns each into what a transfer actually reaches.
PEAKS = (64.0e9, 400.0e9)

#: The two transfer sizes of the named result, in KV blocks.
BLOCK_COUNTS = (8, 512)


@pytest.fixture(scope="module")
def geometry():
    """One worker's KV block, for a real model's shape at one rank."""
    raw = json.loads(CONFIG_JSON.read_text())
    hf = PretrainedConfig.from_dict(raw["text_config"])
    return KvGeometry.from_hf_config(hf, block_size=BLOCK_SIZE)


def spec_with(peak_bytes_per_s, *, link="intra_node"):
    """The suite's machine spec with one link speed replaced."""
    document = copy.deepcopy(DOCUMENT)
    document["interconnect"][link]["link_bandwidth_bytes_per_s"] = peak_bytes_per_s
    return MachineSpec.from_mapping(document)


def model_for(geometry, peak, scope=Scope.INTRA_NODE):
    """The transfer model a spec at that link speed gives this geometry."""
    return TransferModel.from_spec(spec_with(peak), geometry, scope)


def connector(model, clock, *, role="worker", kv_role="kv_consumer"):
    """A connector built the way the engine builds one."""
    config = atom_config_double(
        kv_transfer_config={
            "kv_connector": "compass",
            "kv_role": kv_role,
            CLOCK_KEY: clock,
            TRANSFER_KEY: model,
        }
    )
    return KVConnectorFactory.create_connector(config, role=role)


def announce(req_id, blocks, *, direction="recv", computed=0):
    """The engine's own metadata for one transfer of *blocks* blocks."""
    meta = ConnectorMetadata()
    add = meta.add_new_req_to_recv if direction == "recv" else meta.add_new_req_to_save
    add(
        request_id=req_id,
        local_block_ids=list(range(blocks + computed)),
        kv_transfer_params={"num_computed_blocks": computed},
    )
    return meta


def test_the_factory_builds_both_halves_under_the_registered_name():
    """The connector is reached by name through the engine's registry."""
    assert KVConnectorFactory.canonical_name("compass") == "compass"
    model = TransferModel(
        latency_s=1.0e-6, bandwidth_bytes_per_s=1.0e9, bytes_per_block=1024
    )
    assert isinstance(connector(model, lambda: 0.0), SimulatedKVConnector)
    assert isinstance(
        connector(model, lambda: 0.0, role="scheduler"),
        SimulatedKVConnectorScheduler,
    )


@pytest.mark.parametrize("peak", PEAKS)
@pytest.mark.parametrize("blocks", BLOCK_COUNTS)
def test_a_transfer_is_released_when_the_clock_reaches_its_deadline(
    geometry, blocks, peak
):
    """The named result: four cells of predicted against observed release.

    The prediction is rebuilt here from the spec's own numbers rather than
    asked of the model, so agreement means the connector charged the latency,
    the derated bandwidth and the exact byte count it was given -- and one
    nanosecond earlier it charges all of them still.
    """
    spec = spec_with(peak)
    model = TransferModel.from_spec(spec, geometry, Scope.INTRA_NODE)
    predicted = ISSUE_AT + (
        spec.value("interconnect.intra_node.link_latency_s")
        + (blocks * geometry.bytes_per_block)
        / (peak * spec.value("interconnect.intra_node.derate"))
    )

    now = [ISSUE_AT]
    worker = connector(model, lambda: now[0])
    worker.start_load_kv(announce("r", blocks))

    now[0] = predicted - TICK
    assert worker.get_finished() == (set(), set()), "released one tick early"

    now[0] = predicted
    assert worker.get_finished() == (set(), {"r"})
    assert worker.get_finished() == (set(), set()), "released a second time"


def test_nothing_is_released_before_any_time_passes(geometry):
    """A transfer announced and polled at one instant is still in flight."""
    worker = connector(model_for(geometry, PEAKS[0]), lambda: ISSUE_AT)
    worker.start_load_kv(announce("r", 8))
    assert worker.get_finished() == (set(), set())


def test_release_does_not_depend_on_the_order_they_were_announced(geometry):
    """Two transfers at one instant release by deadline, not by sequence."""
    model = model_for(geometry, PEAKS[0])
    deadline = model.release_at(ISSUE_AT, 512)
    small, big = ("small", 8), ("big", 512)

    released = []
    now = [ISSUE_AT]
    for order in ((small, big), (big, small)):
        now[0] = ISSUE_AT
        worker = connector(model, lambda: now[0])
        for req_id, blocks in order:
            worker.start_load_kv(announce(req_id, blocks))
        now[0] = deadline
        sending, recving = worker.get_finished()
        assert sending == set()
        assert isinstance(recving, set)
        released.append(recving)

    assert released[0] == released[1] == {"small", "big"}


def test_the_producer_reports_sending_and_the_consumer_recving(geometry):
    """Which set a completion lands in is the connector's role, not its size."""
    model = model_for(geometry, PEAKS[0])
    deadline = model.release_at(ISSUE_AT, 8)

    now = [ISSUE_AT]
    producer = connector(model, lambda: now[0], kv_role="kv_producer")
    producer.start_load_kv(announce("r", 8, direction="save"))
    producer.start_load_kv(announce("ignored", 8))
    now[0] = deadline
    assert producer.get_finished() == ({"r"}, set())

    later = [ISSUE_AT]
    consumer = connector(model, lambda: later[0])
    consumer.start_load_kv(announce("r", 8))
    consumer.start_load_kv(announce("ignored", 8, direction="save"))
    later[0] = deadline
    assert consumer.get_finished() == (set(), {"r"})


def test_a_request_completes_only_once_every_worker_has_reported(geometry):
    """The rule the connector declares, through the aggregator that keeps it.

    The two workers hold different amounts of the same request, so their
    deadlines differ. The earlier one reports and the request is still not
    finished; it finishes when the later one does.
    """
    model = model_for(geometry, PEAKS[0])
    now = [ISSUE_AT]
    workers = [connector(model, lambda: now[0]) for _ in range(2)]
    workers[0].start_load_kv(announce("r", 8))
    workers[1].start_load_kv(announce("r", 512))
    aggregator = KVOutputAggregator(world_size=2)

    def aggregate_at(when):
        now[0] = when
        outputs = []
        for worker in workers:
            sending, recving = worker.get_finished()
            outputs.append(
                KVConnectorOutput(finished_sending=sending, finished_recving=recving)
            )
        return aggregator.aggregate(outputs)

    assert aggregate_at(model.release_at(ISSUE_AT, 8)).finished_recving == set()
    assert aggregate_at(model.release_at(ISSUE_AT, 512)).finished_recving == {"r"}


def test_blocks_already_held_locally_are_not_transferred(geometry):
    """A prefix the consumer already has shortens the transfer exactly."""
    model = model_for(geometry, PEAKS[0])
    now = [ISSUE_AT]
    worker = connector(model, lambda: now[0])
    worker.start_load_kv(announce("r", 8, computed=24))
    now[0] = model.release_at(ISSUE_AT, 8)
    assert worker.get_finished() == (set(), {"r"})


def test_announcing_a_request_already_in_flight_is_refused(geometry):
    """Two transfers for one request would make its issue time a choice."""
    worker = connector(model_for(geometry, PEAKS[0]), lambda: ISSUE_AT)
    worker.start_load_kv(announce("r", 8))
    with pytest.raises(ValueError, match="still in flight"):
        worker.start_load_kv(announce("r", 8))


def test_a_connector_with_no_clock_bound_refuses_to_be_built(geometry):
    """No clock, no connector -- there is nothing to fall back to."""
    config = atom_config_double(
        kv_transfer_config={
            "kv_connector": "compass",
            TRANSFER_KEY: model_for(geometry, PEAKS[0]),
        }
    )
    with pytest.raises(UnboundSeam, match=CLOCK_KEY):
        KVConnectorFactory.create_connector(config, role="worker")


def test_a_connector_with_no_transfer_model_refuses_to_be_built():
    """Nor is there a default price for a link nobody described."""
    config = atom_config_double(
        kv_transfer_config={"kv_connector": "compass", CLOCK_KEY: lambda: 0.0}
    )
    with pytest.raises(UnboundSeam, match=TRANSFER_KEY):
        KVConnectorFactory.create_connector(config, role="worker")


def test_the_connector_names_no_clock_of_its_own():
    """The strongest form of "it reads the clock it is given": there is no other.

    Asserted against the source, because a wall-clock fallback reached only on
    a path no test drives would pass every timing test here and still silently
    put a real run on the wall clock. The walk is recursive and the list of
    spellings is wider than the module needs today, so a file added to this
    package later is scanned by this test rather than exempt from it.
    """
    package = pathlib.Path(compass_kv.__file__).parent
    scanned = sorted(package.rglob("*.py"))
    assert scanned, "the package was not found, so nothing was checked"
    for module in scanned:
        source = module.read_text()
        for forbidden in (
            "import time",
            "monotonic",
            "perf_counter",
            "time.time",
            "datetime",
            "os.times",
            "get_event_loop",
            "clock_gettime",
        ):
            assert forbidden not in source, f"{module.name} names {forbidden}"


def test_the_derate_is_spent_and_not_the_spec_peak(geometry):
    """A datasheet number is not an achievable one, and is not charged as one."""
    spec = spec_with(PEAKS[0])
    model = TransferModel.from_spec(spec, geometry, Scope.INTRA_NODE)
    derate = spec.value("interconnect.intra_node.derate")
    assert 0 < derate < 1, "this document states no haircut, so nothing is proven"
    assert model.bandwidth_bytes_per_s == PEAKS[0] * derate


def test_the_named_side_of_the_node_boundary_is_the_one_priced(geometry):
    """No transfer is quietly priced on the other side's link."""
    spec = spec_with(PEAKS[0])
    intra = TransferModel.from_spec(spec, geometry, Scope.INTRA_NODE)
    inter = TransferModel.from_spec(spec, geometry, Scope.INTER_NODE)
    assert intra.latency_s == spec.value("interconnect.intra_node.link_latency_s")
    assert inter.latency_s == spec.value("interconnect.inter_node.link_latency_s")
    assert intra.duration_s(512) < inter.duration_s(512)


def test_a_spec_that_cannot_answer_for_the_link_refuses(geometry):
    """The spec's refusal travels out; the other link is not substituted.

    The refusal has to name the field it could not answer for, and must name
    nothing from the other side of the node boundary -- a message mentioning
    the intra-node link here would be a fallback announcing itself. The rule's
    own sentence is deliberately not pinned: a spec built by hand this way is
    declined as "not a field of this schema", which is inaccurate for a field
    the schema does declare, and freezing that wording here would make a
    correction to the spec package fail this test.
    """
    spec = spec_with(PEAKS[0])
    without = MachineSpec(
        values={
            path: value
            for path, value in spec.values.items()
            if not path.startswith("interconnect.inter_node")
        },
        tokenizers=spec.tokenizers,
    )
    TransferModel.from_spec(without, geometry, Scope.INTRA_NODE)
    with pytest.raises(SpecRefusal) as refusal:
        TransferModel.from_spec(without, geometry, Scope.INTER_NODE)
    message = str(refusal.value)
    assert "interconnect.inter_node.link_" in message, message
    assert "intra_node" not in message, message


def test_a_transfer_of_no_blocks_still_costs_the_latency(geometry):
    """Both ends still have to agree that there was nothing to move."""
    assert model_for(geometry, PEAKS[0]).duration_s(0) == pytest.approx(
        model_for(geometry, PEAKS[0]).latency_s
    )


def test_a_negative_block_count_is_refused(geometry):
    """A transfer that finished before it started is not a fast transfer."""
    with pytest.raises(ValueError, match="not a transfer"):
        model_for(geometry, PEAKS[0]).duration_s(-1)


def test_the_scheduler_declines_a_remote_load_it_could_not_wait_for(geometry):
    """The two halves of a remote fill land together, or the request refuses.

    A request asking to be filled from another deployment is client-reachable
    -- `kv_transfer_params` rides the API request onto the sequence -- and the
    engine calls this on every prefill allocation, before it decides whether
    to suspend anything. Queueing the receive here while nothing suspends the
    request would have the workers report a transfer finished against a
    request the scheduler never suspended, and the scheduler has no path to
    take that report back off its list.
    """
    scheduler = connector(
        model_for(geometry, PEAKS[0]), lambda: ISSUE_AT, role="scheduler"
    )
    seq = SimpleNamespace(
        id="r", block_table=[0, 1, 2], kv_transfer_params={"do_remote_prefill": True}
    )
    with pytest.raises(NotImplementedError, match="never suspended"):
        scheduler.update_state_after_alloc(seq)

    assert seq.kv_transfer_params["do_remote_prefill"] is True, "state was changed"
    assert scheduler.build_connector_meta().reqs_to_recv == {}
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)


def test_an_ordinary_request_passes_the_scheduler_untouched(geometry):
    """Only a remote fill is declined; every other allocation is silent."""
    scheduler = connector(
        model_for(geometry, PEAKS[0]), lambda: ISSUE_AT, role="scheduler"
    )
    seq = SimpleNamespace(id="r", block_table=[0, 1, 2], kv_transfer_params=None)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    assert scheduler.build_connector_meta().reqs_to_recv == {}
    assert scheduler.build_connector_meta().reqs_to_save == {}
