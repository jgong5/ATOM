# SPDX-License-Identifier: MIT
"""The three semantics a predicted step must reproduce, driven through ATOM's own scheduler.

Every measurement here runs the real `Scheduler`, the real `BlockManager` and
the real `Sequence`, in the loop ATOM's engine runs them in -- schedule,
compute the batch's attention aggregates, forward, postprocess -- because all
three semantics are about what that loop does with a reply and none of them can
be seen in the reply alone. A scheduler written for the test would agree with
whatever the test asserted.

The sequence they are measured on is one prompt per step until the prompt is
done, four times over, and then decode: a prefill streak with three middle
chunks and one final chunk per request. `test_the_driven_sequence` pins it, so
a change in ATOM's chunking is a failure with a name rather than a silent
change of subject underneath the other tests.

The wrong implementations are here as controls, each driven through the same
scheduler as the right one. Two of them change when a request's tokens are
offered and neither raises; the third cannot be told apart at all, which is
what it is here to record.
"""

import ast
import inspect
import itertools
import pathlib
import pickle
import queue
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig
from test_runner_rpc_surface import SITES

from atom.compass.runner.overrides import NonAllocatingRunner, RunnerRefusal
from atom.compass.runner.step_output import (
    DeferredTokenStream,
    reported_token_id,
    reports_previous_step,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

REPO = pathlib.Path(__file__).resolve().parents[2]
ATOM_RUNNER = REPO / "atom" / "model_engine" / "model_runner.py"
OVERRIDES = REPO / "atom" / "compass" / "runner" / "overrides.py"
SCHEDULER = (REPO / "atom" / "model_engine" / "scheduler.py").read_text()

# One request's prompt is three full token budgets plus a remainder, so each
# prefill takes three middle chunks and one final chunk, and the four requests
# prefill one after another because a full budget leaves no room for a second.
STREAK = {"requests": 4, "prompt": 200, "budget": 64, "block_size": 16, "max_tokens": 4}


class _Runner(NonAllocatingRunner):
    """The mixin under test, with the one attribute its bodies read."""

    def __init__(self, config, stream=None):
        self.config = config
        if stream is not None:
            self._token_stream = stream


class _PerStep(DeferredTokenStream):
    """Deferral counted in steps: no early return, so every step is meaningful."""

    def step(self, batch):
        prev, self.prev_batch = self.prev_batch, batch
        return self._reply(prev, deferred=True)


class _Eager(DeferredTokenStream):
    """This step's own tokens, reported now, with the deferred flag clear."""

    def step(self, batch):
        if self.is_pure_middle_chunk(batch):
            return super().step(batch)
        return self._reply(batch, deferred=False)


def _drive(stream=None, ignore_eos=True, stop_token_ids=(), **overrides):
    """ATOM's engine step, run to completion, in `engine_core.py:382-412`'s order.

    `ignore_eos` is a parameter because it decides whether the scheduler's own
    end-of-text and stop-token checks run at all: both are gated on
    `not seq.ignore_eos` (`scheduler.py:2630` and `:2633`), so a run that
    leaves it True says nothing about the id this runner reports.
    """
    spec = {**STREAK, **overrides}
    config = MockConfig(
        max_num_seqs=8,
        num_kvcache_blocks=4096,
        kv_cache_block_size=spec["block_size"],
        max_model_len=2048,
        max_num_batched_tokens=spec["budget"],
        enable_chunked_prefill=True,
        stop_token_ids=list(stop_token_ids),
        pipeline_parallel_size=1,
    )
    scheduler = Scheduler(config)
    runner = _Runner(config, stream)
    # Every driven run numbers its requests from zero, so two of them can be
    # compared request by request. The suite-wide reset is per test, and these
    # tests drive more than one run each.
    Sequence.counter = itertools.count()
    sequences = [
        Sequence(
            list(range(5, 5 + spec["prompt"])),
            spec["block_size"],
            sampling_params=SamplingParams(
                max_tokens=spec["max_tokens"], ignore_eos=ignore_eos
            ),
        )
        for _ in range(spec["requests"])
    ]
    for sequence in sequences:
        scheduler.add(sequence)

    offered_to, final_chunk_at, batches, replies = {}, {}, [], []
    stream_queue = queue.Queue()
    for step in itertools.count(1):
        assert step < 200, "the driven sequence does not terminate"
        if not scheduler.has_unfinished_requests():
            break
        scheduled = scheduler.schedule()
        if scheduled is None:
            continue
        batch, seqs = scheduled
        if batch is None or not batch.req_ids:
            continue
        batches.append(batch)
        if batch.is_final_chunk is not None:
            for i, req_id in enumerate(batch.req_ids):
                if batch.is_final_chunk[i]:
                    final_chunk_at.setdefault(req_id, step)
        # Between `schedule()` and `forward` in the engine's own order
        # (`engine_core.py:385`). It attaches the batch's attention aggregates
        # in place and returns early unless profiling is active, so it moves
        # nothing measured here -- it is called so that a successor extending
        # this driver to a cost model inherits the loop and not a summary of
        # it, since those aggregates are the batch-level attention terms such a
        # model would read.
        scheduler.compute_detailed_aggregates(batch, seqs)
        reply = runner.forward(batch)
        replies.append(reply)
        scheduler.postprocess(
            list(seqs.values()), reply, stream_output_queue=stream_queue, batch=batch
        )
        while not stream_queue.empty():
            for req_id, _output in stream_queue.get_nowait():
                offered_to.setdefault(req_id, []).append(step)
    return _Run(batches, final_chunk_at, offered_to, sequences, replies)


class _Run:
    """One driven sequence, and the three things measured off it."""

    def __init__(self, batches, final_chunk_at, offered_to, sequences, replies):
        self.batches = batches
        self.final_chunk_at = final_chunk_at
        self.offered_to = offered_to
        self.sequences = sequences
        self.replies = replies
        self.produces_output = [b.produces_output() for b in batches]

    @property
    def middle_steps(self):
        return [i + 1 for i, p in enumerate(self.produces_output) if not p]

    @property
    def output_steps(self):
        return [i + 1 for i, p in enumerate(self.produces_output) if p]

    @property
    def first_offer(self):
        return {r: steps[0] for r, steps in self.offered_to.items()}

    def next_output_step_after(self, step):
        """The deferral rule: where a batch's tokens can next be reported."""
        return next(s for s in self.output_steps if s > step)


@pytest.fixture(scope="module")
def run():
    return _drive()


# --- the sequence every number below is measured on --------------------------


def test_the_driven_sequence_is_a_prefill_streak_and_then_decode(run):
    assert len(run.batches) == 20
    assert run.middle_steps == [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    assert run.final_chunk_at == {0: 4, 1: 8, 2: 12, 3: 16}
    assert [b.total_seqs_num_decode for b in run.batches[16:]] == [4, 4, 4, 4]
    assert all(s.leave_reason == "max_tokens" for s in run.sequences)


# --- 1 and 2: the tokens are late, and late by output-producing steps ---------


def test_a_requests_tokens_surface_at_the_next_output_producing_step(run):
    """The rule, evaluated against where the scheduler actually offered them."""
    predicted = {
        req: run.next_output_step_after(step)
        for req, step in run.final_chunk_at.items()
    }
    assert run.first_offer == predicted


def test_reporting_this_steps_tokens_strands_the_whole_prefill_streak():
    """No placeholder exists for them yet, so they are dropped and not re-offered."""
    eager = _drive(stream=_Eager(0, deferred=True))
    assert eager.final_chunk_at == {0: 4, 1: 8, 2: 12, 3: 16}
    # Every one of them waits for the first batch that decodes it.
    assert set(eager.first_offer.values()) == {eager.output_steps[-4]}
    assert all(s.leave_reason == "max_tokens" for s in eager.sequences)


def test_counting_the_lag_in_steps_rather_than_meaningful_ones_surfaces_early():
    per_step = _drive(stream=_PerStep(0, deferred=True))
    early = {r: s + 1 for r, s in per_step.final_chunk_at.items()}
    assert per_step.first_offer == early
    correct = _drive()
    understated = [
        correct.first_offer[r] - per_step.first_offer[r] for r in correct.first_offer
    ]
    # One request's final chunk is the last of the streak, so nothing separates
    # it from the next output-producing step and the two rules agree on it.
    assert understated == [3, 3, 3, 0]


# --- 3: the reply a step that samples nothing reports -------------------------


def test_a_middle_chunk_reports_its_requests_with_no_tokens(run):
    middle = [run.batches[s - 1] for s in run.middle_steps]
    stream = DeferredTokenStream(7)
    for batch in middle:
        assert stream.step(batch) == {
            "req_ids": list(batch.req_ids),
            "token_ids": [],
            "num_rejected": None,
            "num_bonus": None,
            "draft_token_ids": None,
            "is_deferred_out": False,
        }
    assert stream.prev_batch is None


def test_that_reply_is_the_one_atoms_own_early_return_builds():
    """Read off ATOM's source, since the scheduler cannot check it for us."""
    forward = _function(ATOM_RUNNER, "ModelRunner", "forward")
    built = next(
        {k.arg: ast.unparse(k.value) for k in n.value.keywords}
        for n in ast.walk(forward)
        if isinstance(n, ast.Return)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "ScheduledBatchOutput"
    )
    assert built == {
        "req_ids": "list(batch.req_ids)",
        "token_ids": "[]",
        "num_rejected": "None",
        "num_bonus": "None",
        "draft_token_ids": "None",
    }
    # ATOM leaves the flag to the constructor's default; this module passes it.
    signature = inspect.signature(ScheduledBatchOutput)
    assert signature.parameters["is_deferred_out"].default is False
    # And the other reader of a reply drops the entry rather than waiting for
    # one, which is why neither path can check this shape for us.
    pp_head = (REPO / "atom" / "model_engine" / "pp_engine_core.py").read_text()
    assert "needs_output = scheduled_batch.produces_output()" in pp_head
    assert "if not needs_output:" in pp_head


@pytest.mark.parametrize("wrong", ["tokens", "no_ids", "deferred_flag"])
def test_the_scheduler_cannot_tell_a_wrong_middle_chunk_reply_apart(run, wrong):
    """Recorded, not asserted away: this is the one of the three that is silent."""

    class _Wrong(DeferredTokenStream):
        def step(self, batch):
            reply = super().step(batch)
            if not self.is_pure_middle_chunk(batch):
                return reply
            if wrong == "tokens":
                reply["token_ids"] = [(self.token_id,) for _ in reply["req_ids"]]
            elif wrong == "no_ids":
                reply["req_ids"] = []
            else:
                reply["is_deferred_out"] = True
                reply["num_rejected"] = np.zeros(len(reply["req_ids"]), np.int32)
                reply["num_bonus"] = np.zeros(len(reply["req_ids"]), np.int32)
            return reply

    silent = _drive(stream=_Wrong(0, deferred=True))
    assert silent.offered_to == run.offered_to
    assert silent.middle_steps == run.middle_steps


# --- what the reply is made of, and what runs around it ----------------------


def test_the_reported_token_is_never_one_that_would_end_a_request():
    assert reported_token_id(2, []) == 0
    assert reported_token_id(0, [1, 2]) == 3
    assert reported_token_id(None, None) == 0


def test_reporting_a_stop_id_lets_the_run_decide_the_length_it_predicts(run):
    """Driven with the scheduler's stop checks live, which the other runs are not.

    Every request in the default run sets `ignore_eos=True`, and both checks
    are gated on `not seq.ignore_eos` (`scheduler.py:2630` and `:2633`). So
    that run would read exactly the same if every step reported the
    end-of-text id, and it is no evidence about the id this module picks. Here
    the checks run, against both ids they test for: reporting either one ends
    every request on its first decoded token, three steps and three tokens
    early, and the run decides for itself the length it was asked to predict.
    """
    eos = MockConfig().eos_token_id
    live = _drive(ignore_eos=False)
    assert [s.leave_reason for s in live.sequences] == ["max_tokens"] * 4
    assert len(live.batches) == len(run.batches)
    assert [s.num_completion_tokens for s in live.sequences] == [4] * 4

    ends_on_eos = _drive(ignore_eos=False, stream=DeferredTokenStream(eos))
    assert [s.leave_reason for s in ends_on_eos.sequences] == ["eos"] * 4
    assert len(ends_on_eos.batches) == 17
    assert [s.num_completion_tokens for s in ends_on_eos.sequences] == [1] * 4

    # The same again for a configured stop id, and then the id the function
    # answers for that config, driven through the same live checks.
    ends_on_stop = _drive(
        ignore_eos=False, stop_token_ids=(0,), stream=DeferredTokenStream(0)
    )
    assert [s.leave_reason for s in ends_on_stop.sequences] == ["stop_0"] * 4
    assert len(ends_on_stop.batches) == 17

    assert reported_token_id(eos, [0]) == 1
    survives = _drive(ignore_eos=False, stop_token_ids=(0,))
    assert [s.leave_reason for s in survives.sequences] == ["max_tokens"] * 4
    assert len(survives.batches) == len(run.batches)


def test_a_pipeline_stage_reports_the_step_it_ran(run):
    assert reports_previous_step(1) is True
    assert reports_previous_step(2) is False
    staged = DeferredTokenStream(0, deferred=False)
    decode = run.batches[-1]
    reply = staged.step(decode)
    assert reply["req_ids"] == list(decode.req_ids)
    assert reply["is_deferred_out"] is False
    assert staged.prev_batch is None


def test_forward_keeps_inference_mode_and_drops_the_expert_load_monitor():
    """The base runs both; the other non-allocating runner here runs one."""
    assert _decorators(ATOM_RUNNER, "ModelRunner", "forward") == [
        "torch.inference_mode()",
        "with_eplb_forward_monitor",
    ]
    assert _decorators(ATOM_RUNNER, "RapidServeModelRunner", "forward") == [
        "torch.inference_mode()"
    ]
    assert _decorators(OVERRIDES, "NonAllocatingRunner", "forward") == [
        "torch.inference_mode()"
    ]


def test_reporting_a_step_from_something_that_is_not_a_batch_refuses():
    with pytest.raises(RunnerRefusal, match="produces output"):
        _Runner(MockConfig(pipeline_parallel_size=1)).forward(object())


@pytest.mark.parametrize("method", ["eagle3", "mtp"])
def test_a_speculative_config_is_refused_before_the_drafter_is_built(run, method):
    """Refused while the model is being built, so the base never builds a drafter.

    Constructing `CompassModelRunner` needs a driver, because importing ATOM's
    runner runs aiter's architecture probe. So `_Base` stands in for
    `ModelRunner.__init__` and records what runs after the model is built. The
    order it copies is read from ATOM's source: `_build_and_load_model` is a
    plain statement of `__init__`'s body, under no `if` or `try`, and it comes
    before the statement that calls `build_drafter`.
    """
    init = _function(ATOM_RUNNER, "ModelRunner", "__init__")
    calls = [
        {ast.unparse(n.func) for n in ast.walk(stmt) if isinstance(n, ast.Call)}
        for stmt in init.body
    ]
    built = next(i for i, c in enumerate(calls) if "self._build_and_load_model" in c)
    drafter = next(i for i, c in enumerate(calls) if "build_drafter" in c)
    assert isinstance(init.body[built], ast.Expr) and built < drafter

    ran = []

    class _Base:
        def __init__(self, config):
            self.config = config
            self._build_and_load_model(object)
            ran.append("build_drafter")

    class _Composed(NonAllocatingRunner, _Base):
        pass

    speculative = SimpleNamespace(method=method)
    with pytest.raises(RunnerRefusal, match="drafts no tokens"):
        _Composed(MockConfig(speculative_config=speculative))
    assert ran == []
    # What the refusal is instead of: a reply nothing rejects, describing a run
    # in which nothing was drafted.
    reply = DeferredTokenStream(0)._reply(run.batches[-1], deferred=True)
    assert reply["draft_token_ids"] is None
    assert not reply["num_rejected"].any() and not reply["num_bonus"].any()
    assert "if self.mtp_k > 0 and draft_token_ids is not None:" in SCHEDULER


def test_forward_never_answers_none_and_so_never_parks_its_caller(run):
    """A present method that answers None parks the caller exactly like an absent one.

    `async_proc.py:243` is `if out is not None:` with both of the loop's
    `put_nowait` calls inside it -- pinned as structure in
    `test_runner_rpc_surface.py`. This is the other side of that: the first
    method on this surface with a body that returns, and the property that its
    body has no way out that answers None.
    """
    assert run.replies and all(reply is not None for reply in run.replies)
    forward = _function(OVERRIDES, "NonAllocatingRunner", "forward")
    returns = [n for n in ast.walk(forward) if isinstance(n, ast.Return)]
    assert [r.value is not None for r in returns] == [True]
    # And no way to fall off the end, which returns None just as quietly.
    assert forward.body[-1] is returns[0]


def _function(path, class_name, name):
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _decorators(path, class_name, name):
    return [ast.unparse(d) for d in _function(path, class_name, name).decorator_list]


def _reply_attribute_reads():
    """Every attribute ATOM reads off a forward reply, from ATOM's own source."""
    names = set()
    for path in (REPO / "atom" / "model_engine").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"fwd_out", "fwd_output"}
            ):
                names.add(node.attr)
    return names


def test_the_reply_answers_every_attribute_atom_reads_off_one(run):
    """Four callers, all of them waiting, and a reply that crosses a process."""
    assert len(SITES["forward"]) == 4
    assert all(site.waits for site in SITES["forward"])
    reads = _reply_attribute_reads()
    assert reads == {
        "req_ids",
        "token_ids",
        "draft_token_ids",
        "num_rejected",
        "num_bonus",
        "is_deferred_out",
        "logprobs",
        "dspark_ell",
        "get_idx",
    }
    # Answered by building ATOM's own object rather than one shaped like it, so
    # a field this runner never sets still carries the default ATOM gives it.
    runner = _Runner(MockConfig(pipeline_parallel_size=1))
    runner.forward(run.batches[-2])
    reply = runner.forward(run.batches[-1])
    assert isinstance(reply, ScheduledBatchOutput)
    assert all(hasattr(reply, name) for name in reads)
    assert reply.get_idx(reply.req_ids[0]) == 0
    restored = pickle.loads(pickle.dumps(reply))
    assert restored.req_ids == reply.req_ids and restored.token_ids == reply.token_ids
