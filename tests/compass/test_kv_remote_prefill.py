# SPDX-License-Identifier: MIT
"""The consumer half of a disaggregated prefill, and the blob that starts one.

Two claims live here and they are tested in different ways on purpose.

**The blob's field set is not asserted against a list typed into this file.**
It is parsed out of the backend the router and the consumer were actually
built against, so the comparison is against the source at this head rather
than against somebody's reading of it. A field added there, or renamed, turns
this red; a list copied into a test would not notice either.

**The suspension is not asserted by calling the connector.** The engine asks
whether a request is held elsewhere, allocates its blocks, tells the connector
about the allocation and only then decides to suspend it -- four steps in one
loop, in that order, and the defect this cut exists to close was a connector
that behaved correctly at each step and wrongly across them. So the request
here goes into a real `Scheduler`, and what is asserted is the status the
engine put it in and the step it came back out on. The clock is the list
holding one number that the rest of this package's tests use.
"""

import ast
import json
import pathlib
from types import SimpleNamespace

import pytest
from conftest import MockConfig, atom_config_double
from test_kv_simulated_connector import (
    BLOCK_SIZE,
    CONFIG_JSON,
    ISSUE_AT,
    PEAKS,
    TICK,
    model_for,
)
from transformers import PretrainedConfig

from atom.compass.backends import KvGeometry
from atom.compass.kv import CLOCK_KEY, TRANSFER_KEY
from atom.compass.kv.handoff import (
    SIMULATED_ENGINE_ID,
    SIMULATED_HOST,
    SIMULATED_PORT,
    transfer_params,
)
from atom.kv_transfer import disaggregation
from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus

#: The backend whose blob shape the router and the consumer were built around.
PULL_BACKEND_SOURCE = (
    pathlib.Path(disaggregation.__file__).parent / "moriio" / "moriio_connector.py"
)

#: The first sampled token the producer hands back; the consumer resumes on it.
FIRST_TOKEN_ID = 7

#: A prompt long enough to occupy several blocks of the pool below.
PROMPT = list(range(40))


@pytest.fixture(scope="module")
def geometry():
    """One worker's KV block, for a real model's shape at one rank."""
    raw = json.loads(CONFIG_JSON.read_text())
    hf = PretrainedConfig.from_dict(raw["text_config"])
    return KvGeometry.from_hf_config(hf, block_size=BLOCK_SIZE)


def relayed_fields() -> frozenset:
    """The keys of the pull backend's own transfer blob, read from its source.

    Parsed rather than imported: the module it lives in reaches for an RDMA
    library this tier does not have. Exactly one assignment is expected, so a
    backend that grew a second way to build the blob fails here instead of
    being read half.
    """
    tree = ast.parse(PULL_BACKEND_SOURCE.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(
            isinstance(target, ast.Attribute)
            and target.attr == "kv_transfer_params_output"
            for target in node.targets
        ):
            continue
        assert all(
            isinstance(key, ast.Constant) and isinstance(key.value, str)
            for key in node.value.keys
        ), f"{PULL_BACKEND_SOURCE.name} builds its blob with computed keys"
        found.append(frozenset(key.value for key in node.value.keys))
    assert len(found) == 1, f"expected one blob assignment, found {len(found)}"
    return found[0]


def assert_relays_every_field(blob) -> None:
    """The check the named result makes, and the one the drop test breaks."""
    missing = sorted(relayed_fields() - set(blob))
    assert not missing, f"the relayed blob would not carry {missing}"
    extra = sorted(set(blob) - relayed_fields())
    assert not extra, f"the relayed blob would carry {extra}, which nothing reads"


#: What a finished request holds when the producer hands it back.
FINISHED_ID = 11
FINISHED_BLOCKS = [4, 5, 6]
FINISHED_HIT_TOKENS = 128


def finished_sequence():
    """A request the way `request_finished` finds one: done, with a table."""
    return SimpleNamespace(
        id=FINISHED_ID,
        block_table=list(FINISHED_BLOCKS),
        output_tokens=[FIRST_TOKEN_ID, 8, 9],
        spec_token_ids=[],
        prefix_cache_hit_tokens=FINISHED_HIT_TOKENS,
        kv_transfer_params_output=None,
    )


def simulated_blob(tp_size=4, dp_rank=2) -> dict:
    return transfer_params(finished_sequence(), tp_size=tp_size, dp_rank=dp_rank)


def connector(model, clock, *, role, kv_role="kv_consumer", tp_size=1):
    """A connector built the way the engine builds one."""
    config = atom_config_double(
        tensor_parallel_size=tp_size,
        kv_transfer_config={
            "kv_connector": "compass",
            "kv_role": kv_role,
            CLOCK_KEY: clock,
            TRANSFER_KEY: model,
        },
    )
    return KVConnectorFactory.create_connector(config, role=role)


# -- The blob ------------------------------------------------------------


def test_the_blob_carries_the_field_set_the_router_relays():
    """The named result: the emitted set equals the backend's, re-derived."""
    assert set(simulated_blob()) == relayed_fields()
    assert_relays_every_field(simulated_blob())


@pytest.mark.parametrize("field", sorted(relayed_fields()))
def test_dropping_one_field_from_the_blob_is_refused_by_name(field):
    """Every field is load-bearing, and the check says which one went."""
    short = dict(simulated_blob())
    del short[field]
    with pytest.raises(AssertionError) as refusal:
        assert_relays_every_field(short)
    assert field in str(refusal.value)


def test_no_field_of_the_blob_names_a_machine_the_simulation_runs_on():
    """A simulated peer has no address, and must not borrow a real one."""
    blob = simulated_blob()
    assert blob["remote_host"] == SIMULATED_HOST
    assert blob["remote_host"].endswith(".invalid"), "a name that could resolve"
    assert blob["remote_port"] == SIMULATED_PORT == 0
    assert blob["remote_handshake_port"] == SIMULATED_PORT
    assert blob["remote_engine_id"] == SIMULATED_ENGINE_ID


def test_the_ranks_the_router_reads_are_numbers():
    """The router drops `dp_rank` unless it is a number, and says nothing."""
    blob = simulated_blob(tp_size=8, dp_rank=3)
    assert isinstance(blob["dp_rank"], int) and blob["dp_rank"] == 3
    assert isinstance(blob["tp_size"], int) and blob["tp_size"] == 8


def test_the_blob_carries_the_request_and_not_a_template():
    """What is specific to the request comes off the request."""
    blob = simulated_blob()
    assert blob["do_remote_prefill"] is True
    assert blob["do_remote_decode"] is False
    assert blob["remote_block_ids"] == FINISHED_BLOCKS
    assert blob["transfer_id"] == FINISHED_ID
    assert blob["first_token_id"] == FIRST_TOKEN_ID
    assert blob["draft_token_ids"] == []
    assert blob["prefix_cache_hit_tokens"] == FINISHED_HIT_TOKENS


def test_a_finished_request_carries_the_blob_out(geometry):
    """The connector attaches it where the API layer looks for it."""
    scheduler = connector(
        model_for(geometry, PEAKS[0]), lambda: ISSUE_AT, role="scheduler", tp_size=4
    )
    seq = finished_sequence()
    scheduler.request_finished(seq)
    assert seq.kv_transfer_params_output == transfer_params(seq, tp_size=4, dp_rank=0)


# -- The suspension ------------------------------------------------------


def remote_filled(seq_factory):
    """A request the router sent here to be decoded, not prefilled."""
    return seq_factory(
        PROMPT,
        kv_transfer_params={
            "do_remote_prefill": True,
            "first_token_id": FIRST_TOKEN_ID,
        },
    )


def scheduler_with(connector_half, **config):
    """A real scheduler, sized by the caller, carrying this connector."""
    settings = {
        "num_kvcache_blocks": 100,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 1000,
    }
    settings.update(config)
    engine = Scheduler(MockConfig(**settings))
    engine.kv_connector = connector_half
    return engine


def test_a_remote_filled_request_parks_and_leaves_on_its_deadline(
    geometry, seq_factory
):
    """The named result's other half, driven through the engine's own order.

    Nothing here calls the connector's scheduler methods. The request is added
    to a real scheduler and stepped; the engine asks, allocates, notifies and
    decides in its own sequence, and what is read back is the status it chose.
    The transfer is announced from the metadata that same step produced, so
    the blocks the worker prices are the blocks the engine allocated.

    The request's own `do_remote_prefill` is asserted at both ends of the
    park, because the defect this connector's predecessor was fixed for was a
    flag cleared on the way past the engine -- which no test that calls the
    connector directly can see.
    """
    model = model_for(geometry, PEAKS[0])
    now = [ISSUE_AT]
    engine = scheduler_with(connector(model, lambda: now[0], role="scheduler"))
    worker = connector(model, lambda: now[0], role="worker")
    seq = remote_filled(seq_factory)
    engine.add(seq)

    batch, _ = engine.schedule()
    assert seq.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert engine._num_parked_remote_kv == 1
    assert batch.total_seqs_num_prefill == 0, "it prefilled what it was sent"
    assert seq.kv_transfer_params["do_remote_prefill"] is True, "intent destroyed"

    announced = batch.connector_meta_output.reqs_to_recv
    assert set(announced) == {seq.id}
    assert announced[seq.id].local_block_ids == list(seq.block_table)
    worker.start_load_kv(batch.connector_meta_output)
    deadline = model.release_at(ISSUE_AT, len(seq.block_table))

    now[0] = deadline - TICK
    step(engine, worker)
    assert seq.status is SequenceStatus.WAITING_FOR_REMOTE_KVS, "released early"
    assert engine._num_parked_remote_kv == 1

    now[0] = deadline
    step(engine, worker)
    assert seq.status is SequenceStatus.RUNNING
    assert engine._num_parked_remote_kv == 0
    assert seq.token_ids[seq.num_prompt_tokens] == FIRST_TOKEN_ID
    assert seq.kv_transfer_params["do_remote_prefill"] is True, "intent destroyed"
    assert engine.finished_recving_kv_req_ids == [], "an id with no reachable pop"


def step(engine, worker):
    """One engine step's worth of the completion path, through real parts."""
    sending, recving = worker.get_finished()
    aggregated = KVOutputAggregator(world_size=1).aggregate(
        [KVConnectorOutput(finished_sending=sending, finished_recving=recving)]
    )
    engine._update_from_kv_xfer_finished(aggregated)
    engine.schedule()


def test_nothing_is_announced_for_a_request_the_engine_did_not_suspend(
    geometry, seq_factory
):
    """An offer is dropped when the suspension it was offered for did not happen.

    The engine asks whether a request is held elsewhere *before* it knows
    whether it can admit it, and the claim is spent on the asking. So a
    request bounced by the admission cap arrives at the allocation on a later
    step with its claim gone and its `do_remote_prefill` still set: it is
    prefilled locally, and it is not suspended. Both real backends queue a
    receive for it anyway -- their guard is the flag alone -- and the workers
    then report a transfer finished against a request the scheduler never
    suspended: an id on `finished_recving_kv_req_ids` with no reachable pop,
    after which the engine rebuilds its waiting queue on every step for the
    rest of the run.

    Driven through a real scheduler over two steps, because each step is
    individually correct and only the pair is wrong. The cap that bounced the
    claim is lifted between them, which is the whole content of the second
    step.
    """
    model = model_for(geometry, PEAKS[0])
    now = [ISSUE_AT]
    worker = connector(model, lambda: now[0], role="worker")
    engine = scheduler_with(
        connector(model, lambda: now[0], role="scheduler"), max_num_seqs=1
    )
    parked, bounced = remote_filled(seq_factory), remote_filled(seq_factory)
    engine.add(parked)
    engine.add(bounced)

    first, _ = engine.schedule()
    assert parked.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert bounced.status is SequenceStatus.WAITING, "the cap did not bounce it"
    assert bounced.kv_async_tagged is True, "the claim was not spent"
    assert set(first.connector_meta_output.reqs_to_recv) == {parked.id}

    engine.max_num_seqs = 4
    second, _ = engine.schedule()
    assert bounced.status is not SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert second.total_seqs_num_prefill == 1, "it did not prefill locally"
    assert bounced.id not in second.connector_meta_output.reqs_to_recv

    worker.start_load_kv(second.connector_meta_output)
    now[0] = model.release_at(ISSUE_AT, len(bounced.block_table)) + 1.0
    sending, recving = worker.get_finished()
    engine._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending=sending, finished_recving=recving)
    )
    assert engine.finished_recving_kv_req_ids == [], "an id with no reachable pop"


def test_a_parked_request_is_not_counted_as_admittable_work(geometry, seq_factory):
    """The engine's guards read the suspended state; the request must have it.

    A suspended request must not look like queued prefill to the admission
    signals, or a deployment waiting on a transfer reports work it cannot do.
    """
    model = model_for(geometry, PEAKS[0])
    engine = scheduler_with(connector(model, lambda: ISSUE_AT, role="scheduler"))
    engine.add(remote_filled(seq_factory))
    engine.schedule()

    assert engine._waiting_new_token_count() == 0
    assert engine._can_admit_head_prefill() is False
    assert engine._oldest_waiting_prefill_age_ms() == 0.0


def test_the_prompt_is_claimed_once_and_the_request_keeps_its_intent(
    geometry, seq_factory
):
    """The whole prompt, one claim, and the request's own flag left alone.

    The claim is spent on the first ask, which is what stops a request being
    suspended twice. It is spent on a mark and not by clearing the request's
    `do_remote_prefill`, which is what both real backends do instead -- so the
    request still states its own intent afterwards, and the guard against
    announcing a receive nobody waits for has to live somewhere the request is
    not: it is the suspension check in `build_connector_meta`, asserted here
    by setting the status the engine would have set.
    """
    scheduler = connector(
        model_for(geometry, PEAKS[0]), lambda: ISSUE_AT, role="scheduler"
    )
    seq = remote_filled(seq_factory)
    seq.block_table = [0, 1, 2]

    assert scheduler.get_num_new_matched_tokens(seq) == (len(PROMPT), True)
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)

    scheduler.update_state_after_alloc(seq)
    assert seq.kv_transfer_params["do_remote_prefill"] is True
    seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.build_connector_meta().reqs_to_recv[seq.id].local_block_ids == [
        0,
        1,
        2,
    ]
    assert scheduler.build_connector_meta().reqs_to_recv == {}, "announced twice"


def test_an_ordinary_request_is_neither_claimed_nor_queued(geometry, seq_factory):
    """Only a remote fill is taken on; every other allocation is silent."""
    scheduler = connector(
        model_for(geometry, PEAKS[0]), lambda: ISSUE_AT, role="scheduler"
    )
    seq = seq_factory(PROMPT)
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.build_connector_meta().reqs_to_recv == {}


def test_the_producing_side_refuses_a_remote_fill(geometry, seq_factory):
    """A deployment cannot wait for a transfer it is the source of."""
    scheduler = connector(
        model_for(geometry, PEAKS[0]),
        lambda: ISSUE_AT,
        role="scheduler",
        kv_role="kv_producer",
    )
    seq = remote_filled(seq_factory)
    with pytest.raises(ValueError, match="producer side"):
        scheduler.update_state_after_alloc(seq)


def test_the_emitted_blob_is_one_a_consumer_can_actually_consume():
    """The blob goes back in where a consumer reads it, not just out.

    Key-set equality says the router will relay every field. It does not say
    the consumer half finds what it needs in them, because that half reads the
    blob through the engine's own `ReqMeta` builder rather than by key. So the
    emitted blob is fed back in as a received one.
    """
    blob = simulated_blob(tp_size=4, dp_rank=2)
    meta = ConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id=FINISHED_ID, local_block_ids=[0, 1], kv_transfer_params=blob
    )
    req = meta.reqs_to_recv[FINISHED_ID]

    assert req.remote_block_ids == FINISHED_BLOCKS
    assert req.remote_host == SIMULATED_HOST
    assert req.remote_port == SIMULATED_PORT
    assert req.remote_handshake_port == SIMULATED_PORT
    assert req.remote_engine_id == SIMULATED_ENGINE_ID
    assert req.tp_size == 4
    assert req.transfer_id == FINISHED_ID
    assert req.num_computed_blocks == 0, "the full table, as this field set forces"
