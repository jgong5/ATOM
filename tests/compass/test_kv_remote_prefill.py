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
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
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


def scheduler_with(connector_half):
    """A real scheduler with a pool big enough that nothing else declines."""
    engine = Scheduler(
        MockConfig(num_kvcache_blocks=100, max_num_seqs=4, max_num_batched_tokens=1000)
    )
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


def step(engine, worker):
    """One engine step's worth of the completion path, through real parts."""
    sending, recving = worker.get_finished()
    aggregated = KVOutputAggregator(world_size=1).aggregate(
        [KVConnectorOutput(finished_sending=sending, finished_recving=recving)]
    )
    engine._update_from_kv_xfer_finished(aggregated)
    engine.schedule()


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
    suspended twice. It is spent on a mark rather than by clearing the
    request's `do_remote_prefill`, so anything that looks at the request after
    this -- a sibling connector under the composite backend, or the engine
    itself -- still sees what the request asked for.
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
    assert scheduler.build_connector_meta().reqs_to_recv[seq.id].local_block_ids == [
        0,
        1,
        2,
    ]
    assert scheduler.build_connector_meta().reqs_to_recv == {}, "queued twice"


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
