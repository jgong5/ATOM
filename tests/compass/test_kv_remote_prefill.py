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
loop, in that order, and the defect the suspension tests exist to catch is a connector
that behaves correctly at each step and wrongly across them. So the request
here goes into a real `Scheduler`, and what is asserted is the status the
engine put it in and the step it came back out on. The clock is the list
holding one number that the rest of this package's tests use.
"""

import ast
import inspect
import ipaddress
import json
import pathlib
import socket
import sys
import time
from types import SimpleNamespace

import numpy
import pytest
import torch
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
from atom.compass.kv.connector import SimulatedKVConnector
from atom.compass.kv.handoff import (
    SIMULATED_ENGINE_ID,
    SIMULATED_HOST,
    SIMULATED_PORT,
    transfer_params,
)
from atom.compass.runner.overrides import NonAllocatingRunner
from atom.kv_transfer import disaggregation
from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.kv_transfer.disaggregation.types import ConnectorMetadata, KVConnectorOutput
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.utils import forward_context

#: The backend whose blob shape the router and the consumer were built around.
PULL_BACKEND_SOURCE = (
    pathlib.Path(disaggregation.__file__).parent / "moriio" / "moriio_connector.py"
)

#: The runner whose `process_kvconnector_output` the Compass runner inherits.
ATOM_RUNNER = pathlib.Path(inspect.getfile(Scheduler)).with_name("model_runner.py")

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
    """The check the field-set test makes, and the one the drop test breaks."""
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


def connector(model, clock, *, role, kv_role="kv_consumer", tp_size=1, dp_rank=None):
    """A connector built the way the engine builds one.

    `dp_rank` reaches the connector through `parallel_config`, which is where
    it reads it from; left alone, the double carries the real default.
    """
    settings = {
        "tensor_parallel_size": tp_size,
        "kv_transfer_config": {
            "kv_connector": "compass",
            "kv_role": kv_role,
            CLOCK_KEY: clock,
            TRANSFER_KEY: model,
        },
    }
    if dp_rank is not None:
        settings["parallel_config"] = SimpleNamespace(data_parallel_rank=dp_rank)
    return KVConnectorFactory.create_connector(
        atom_config_double(**settings), role=role
    )


def assert_could_not_be_dialled(value, field) -> None:
    """Refuse a value anything downstream could turn into a connection.

    Asserted as a property of whatever the field holds, not as an identity
    with the constant beside it, because a defect that moves the constant
    moves both sides of such an identity and is invisible to it.
    """
    host = str(value).split(":")[0]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{field} carries the address {host!r}")
    own = socket.gethostname()
    labels = {part.lower() for part in str(value).replace(":", ".").split(".")}
    borrowed = labels & {own.lower(), own.lower().split(".")[0]}
    assert not borrowed, (
        f"{field} carries {sorted(borrowed)}, taken from the name of the "
        "machine this run is on"
    )


# -- The blob ------------------------------------------------------------


def test_the_blob_carries_the_field_set_the_router_relays():
    """The emitted set equals the backend's, re-derived."""
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
    """A simulated peer has no address, and must not borrow a real one.

    Two kinds of assertion, and they catch different things. The identities
    say the blob is built from the constants above rather than from a literal
    typed into the emitter. The properties say what those constants may hold
    -- and they are the half that survives the constants themselves changing,
    which is the way this would actually go wrong: an endpoint reaches a blob
    by somebody filling in the value that is already there.
    """
    blob = simulated_blob()
    assert blob["remote_host"] == SIMULATED_HOST
    assert blob["remote_port"] == SIMULATED_PORT == 0
    assert blob["remote_handshake_port"] == SIMULATED_PORT
    assert blob["remote_engine_id"] == SIMULATED_ENGINE_ID

    host = blob["remote_host"]
    assert host.endswith(".invalid"), "a name that could resolve"
    assert host.count(".") == 1, (
        f"{host!r} is a real name with the reserved suffix pinned on the end; "
        "the whole point is that there is one label and it was invented here"
    )
    assert ":" not in blob["remote_engine_id"], "an identity, not host:port"
    for field in ("remote_host", "remote_engine_id"):
        assert_could_not_be_dialled(blob[field], field)


def test_the_engine_id_is_the_label_the_host_invents_and_nothing_else():
    """The engine id is held to the host's rule, not only to "no colon".

    A dotted name such as `prod-decode-07.internal` carries no colon and no
    address, and is not the name of the machine running the test, so every
    check above passes it on every machine. The rule that makes the host safe
    is that it is one label, invented here, under a suffix that resolves
    nowhere; the engine id is held to that same label, so it names nothing a
    resolver could find either.
    """
    blob = simulated_blob()
    engine_id = blob["remote_engine_id"]
    assert "." not in engine_id, (
        f"remote_engine_id {engine_id!r} is a dotted name, which a resolver "
        "can route; it must be the single invented label"
    )
    assert blob["remote_host"] == f"{engine_id}.invalid", (
        f"remote_engine_id {engine_id!r} is not the label remote_host "
        f"{blob['remote_host']!r} reserves under .invalid"
    )


def test_the_ranks_the_router_reads_are_numbers(geometry):
    """The router drops `dp_rank` unless it is a number, and says nothing.

    Driven through the connector from a config carrying the widths as floats
    with no fractional part, because handing `transfer_params` two literal
    ints asserts nothing the conversion is responsible for. A whole float is
    converted to an `int` on the way in; text is refused, which the test
    below pins. A rank that went out as anything but a number would fail
    silently -- the router substitutes its own registry value for the
    prefilling worker rather than refusing the blob.
    """
    blob = simulated_blob(tp_size=8, dp_rank=3)
    assert isinstance(blob["dp_rank"], int) and blob["dp_rank"] == 3
    assert isinstance(blob["tp_size"], int) and blob["tp_size"] == 8

    scheduler = connector(
        model_for(geometry, PEAKS[0]),
        lambda: ISSUE_AT,
        role="scheduler",
        tp_size=8.0,
        dp_rank=3.0,
    )
    seq = finished_sequence()
    scheduler.request_finished(seq)
    relayed = seq.kv_transfer_params_output
    assert isinstance(relayed["tp_size"], int) and relayed["tp_size"] == 8
    assert isinstance(relayed["dp_rank"], int) and relayed["dp_rank"] == 3


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("tp_size", 8.5, id="tp_size-float"),
        pytest.param("tp_size", "8.5", id="tp_size-text"),
        pytest.param("tp_size", "eight", id="tp_size-word"),
        pytest.param("dp_rank", 2.5, id="dp_rank-float"),
        pytest.param("dp_rank", "2.5", id="dp_rank-text"),
        pytest.param("tp_size", "8.0", id="tp_size-decimal-text"),
        pytest.param("tp_size", "1e1", id="tp_size-exponent-text"),
        pytest.param("tp_size", True, id="tp_size-true"),
        pytest.param("tp_size", False, id="tp_size-false"),
        pytest.param("dp_rank", True, id="dp_rank-true"),
        pytest.param("dp_rank", False, id="dp_rank-false"),
        pytest.param("tp_size", numpy.bool_(True), id="tp_size-numpy-true"),
        pytest.param("tp_size", numpy.bool_(False), id="tp_size-numpy-false"),
        pytest.param("tp_size", torch.tensor(True), id="tp_size-torch-true"),
        pytest.param("tp_size", "8", id="tp_size-integer-text"),
        pytest.param("dp_rank", "3", id="dp_rank-integer-text"),
        pytest.param("tp_size", " 8 ", id="tp_size-padded-text"),
    ],
)
def test_a_width_that_is_not_a_whole_number_is_refused_by_name(geometry, field, value):
    """A width that is not exactly an integer refuses the connector by name.

    Refused when the connector is built, so a malformed config never serves a
    request. Casting 8.5 to 8 would emit a blob for a deployment that was
    never launched, and nothing reading it could tell. A boolean, Python's,
    numpy's or torch's, would go out as a width of 1 or 0. Text is refused
    whatever it spells, "8" included: ATOM's launch path parses its widths
    with `int`, so reading text would be a conversion no launch needs.
    """
    widths = {"tp_size": 8, "dp_rank": 3, field: value}
    with pytest.raises(ValueError, match=f"^{field} is .*not a whole number"):
        connector(
            model_for(geometry, PEAKS[0]), lambda: ISSUE_AT, role="scheduler", **widths
        )


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
    """The module's second claim, driven through the engine's own order.

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


def test_the_compass_runner_builds_the_worker_that_ends_the_park(
    geometry, seq_factory, monkeypatch
):
    """The same park, with the worker built and started by the runner.

    Importing ATOM's runner needs a driver, so its `process_kvconnector_output`
    is compiled from source onto the Compass overrides, and the TP group
    `ModelRunner.__init__` opens is stood in at rank 0.
    """
    source = ast.parse(ATOM_RUNNER.read_text())
    (method,) = [
        n
        for c in source.body
        if isinstance(c, ast.ClassDef) and c.name == "ModelRunner"
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "process_kvconnector_output"
    ]
    namespace = {"torch": torch, "get_kvconnector": forward_context.get_kvconnector}
    exec(ast.unparse(method), namespace)  # noqa: S102
    group = SimpleNamespace(get_tp_group=lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", group)
    monkeypatch.setattr(forward_context, "_global_kvconnector", None)

    model = model_for(geometry, PEAKS[0])
    now = [ISSUE_AT]
    runner = object.__new__(
        type("Runner", (NonAllocatingRunner,), {method.name: namespace[method.name]})
    )
    runner.config = atom_config_double(
        kv_transfer_config={
            "kv_connector": "compass",
            "kv_role": "kv_consumer",
            CLOCK_KEY: lambda: now[0],
            TRANSFER_KEY: model,
        }
    )
    assert runner.allocate_kv_cache(100) is True
    worker = forward_context.get_kvconnector()
    assert isinstance(worker, SimulatedKVConnector)

    engine = scheduler_with(connector(model, lambda: now[0], role="scheduler"))
    seq = remote_filled(seq_factory)
    engine.add(seq)
    batch, _ = engine.schedule()
    assert seq.status is SequenceStatus.WAITING_FOR_REMOTE_KVS
    runner.process_kvconnector_output(batch.connector_meta_output)
    now[0] = model.release_at(ISSUE_AT, len(seq.block_table))
    step(engine, worker)
    assert seq.status is SequenceStatus.RUNNING


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
    seq = remote_filled(seq_factory)
    engine.add(seq)
    engine.schedule()

    assert seq.status is SequenceStatus.WAITING_FOR_REMOTE_KVS, (
        "nothing was parked, so the three signals below are reading an empty "
        "queue rather than a suspended request"
    )
    assert engine._waiting_new_token_count() == 0
    assert engine._can_admit_head_prefill() is False
    assert engine._oldest_waiting_prefill_age_ms() == 0.0


#: The ordinary prompt queued behind a parked one, of a length no other count
#: in the queue can produce.
QUEUED_PROMPT = list(range(24))


def test_the_request_queued_beside_a_parked_one_is_the_one_counted(
    geometry, seq_factory
):
    """The positive control for the test above: the signals do count work.

    Reading zero beside a parked request says the parked one was excluded
    only if the same signals read non-zero for work that is admittable. So an
    ordinary request is queued behind the parked one, and each signal must
    report exactly that request: its token count, not the parked one's added
    to it and not nothing; a head that can be admitted; and its own age, which
    is a second, where the parked request's arrival stamp is decades old.
    """
    model = model_for(geometry, PEAKS[0])
    engine = scheduler_with(connector(model, lambda: ISSUE_AT, role="scheduler"))
    parked = remote_filled(seq_factory)
    engine.add(parked)
    engine.schedule()
    assert parked.status is SequenceStatus.WAITING_FOR_REMOTE_KVS

    queued = seq_factory(QUEUED_PROMPT)
    queued.arrive_time = time.time() - 1.0
    engine.add(queued)

    assert engine._waiting_new_token_count() == len(QUEUED_PROMPT)
    assert engine._can_admit_head_prefill() is True
    assert 1000.0 <= engine._oldest_waiting_prefill_age_ms() < 60_000.0


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


@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_producer"])
def test_an_ordinary_request_is_neither_claimed_nor_announced(
    geometry, seq_factory, kv_role
):
    """Only a remote fill is taken on; every other allocation is silent.

    Both roles, because "silent" is two different things and only one of them
    is visible in the metadata. On the consumer side an ordinary allocation
    is not claimed and no receive is announced for it; whether it was queued
    as an offer is not observed here, because the announcement drops any
    offer the engine did not suspend. On the producer side the refusal below
    stands between every ordinary allocation and a `ValueError`, and no
    announcement check can see that: the raise happens before there is
    anything to announce. One case per role, so a failure names the role.
    """
    scheduler = connector(
        model_for(geometry, PEAKS[0]),
        lambda: ISSUE_AT,
        role="scheduler",
        kv_role=kv_role,
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
