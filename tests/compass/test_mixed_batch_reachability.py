"""Can this engine emit a batch holding both prefill and decode requests?

`NativeAllocation` refuses one, because `BatchSpec` carries a single kind and
`prepare_prefill` walks `range(batch.total_seqs_num_prefill)` -- so a mixed
batch has no encoding, and the refusal names the gap. Before extending
`BatchSpec` and the deriver to close it, the question is whether the gap is
reachable at all: modelling a batch the engine cannot produce would be work
against an imaginary target, and the model would then claim coverage of
behaviour no run can exercise.

The answer here is no, on both the behavioural and the structural reading, and
these tests are the tripwire for the day it changes.
"""

import re
from pathlib import Path

from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler, ScheduledBatchOutput

#: Scheduler source, read once. `atom` is imported above, so its parent is the
#: tree under test rather than whatever `scripts/` a caller happens to be in.
SCHEDULER_PY = (Path(__file__).resolve().parents[2]
                / "atom" / "model_engine" / "scheduler.py")


def decode_ready(sched, batch, seqs):
    """Postprocess a batch so its requests become decodable next tick."""
    for seq in seqs:
        seq.num_cached_tokens = seq.num_prompt_tokens
    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(req_ids=[s.id for s in seqs],
                             token_ids=[(900 + i,) for i in range(len(seqs))],
                             num_rejected=None, num_bonus=None,
                             draft_token_ids=None),
        batch=batch)


def test_a_prefill_arriving_mid_decode_does_not_join_the_decode_batch():
    """The tick that would produce a mixed batch if any tick did.

    Two requests are decoding and two prompts arrive. The scheduler has both
    kinds runnable in the same call, room for all four (`max_num_seqs=4`) and
    token budget to spare -- and it still returns a batch of two prefills,
    leaving the two decodable requests for the following tick.
    """
    from atom.sampling_params import SamplingParams
    from atom.model_engine.sequence import Sequence

    def seq(tokens):
        return Sequence(list(tokens), 4,
                        sampling_params=SamplingParams(max_tokens=64))

    sched = Scheduler(MockConfig(num_kvcache_blocks=64, kv_cache_block_size=4,
                                 max_num_seqs=4, max_num_batched_tokens=256,
                                 max_model_len=64))
    running = [seq([1, 2, 3, 4]), seq([5, 6, 7, 8])]
    for s in running:
        sched.add(s)
    first, _ = sched.schedule()
    assert first.total_seqs_num_prefill == 2
    decode_ready(sched, first, running)

    # Both kinds are now runnable in one call.
    arriving = [seq([11, 12, 13, 14]), seq([15, 16, 17, 18])]
    for s in arriving:
        sched.add(s)
    assert len(sched.running) == 2, "two requests are decodable this tick"

    batch, _ = sched.schedule()
    assert batch.total_seqs_num == 2
    assert batch.total_seqs_num_prefill == 2
    assert batch.total_seqs_num_decode == 0
    # ...and the decodes were passed over, not merged in.
    assert set(batch.req_ids) == {s.id for s in arriving}


def test_every_batch_of_a_staggered_workload_is_of_one_kind():
    """The same question asked over a run rather than a tick.

    Arrivals are staggered against decodes for thirty ticks, which is the
    shape a cc-traces replay has. Every batch comes out pure, and the counts
    are asserted so that a run which happened to schedule nothing cannot pass
    as evidence.
    """
    from atom.sampling_params import SamplingParams
    from atom.model_engine.sequence import Sequence

    sched = Scheduler(MockConfig(num_kvcache_blocks=256, kv_cache_block_size=4,
                                 max_num_seqs=4, max_num_batched_tokens=256,
                                 max_model_len=64))
    kinds = {"prefill": 0, "decode": 0}
    for tick in range(30):
        if tick % 2 == 0:
            # Short generations, so requests retire and later prompts are
            # admitted: without that the four concurrency slots fill once and
            # the run is 26 decode ticks with nothing left to prefill.
            sched.add(Sequence([1, 2, 3, 4, 5, 6, 7, 8][:4 + tick % 4], 4,
                               sampling_params=SamplingParams(max_tokens=2)))
        scheduled = sched.schedule()
        if scheduled is None:
            continue
        batch, seqs = scheduled
        if batch is None or batch.total_seqs_num == 0:
            continue
        prefilling = batch.total_seqs_num_prefill
        assert prefilling in (0, batch.total_seqs_num), (
            f"tick {tick}: {prefilling} prefill rows of "
            f"{batch.total_seqs_num} -- a mixed batch, which "
            "NativeAllocation refuses and BatchSpec cannot encode")
        kinds["prefill" if prefilling else "decode"] += 1
        decode_ready(sched, batch, list(seqs.values()))
    assert kinds["prefill"] >= 8 and kinds["decode"] >= 8, kinds


def test_the_scheduler_still_says_the_mixed_batch_is_a_todo():
    """A tripwire, not a style check.

    The behavioural tests above sample the schedules a workload happens to
    produce; this one reads the two places the engine states the limitation.
    If either marker goes, mixed batches may have become reachable and the
    `BatchSpec` extension is then worth building -- which is exactly the
    decision this file exists to gate.
    """
    source = SCHEDULER_PY.read_text(encoding="utf-8")
    assert "TODO for prefill/decode mixed batch" in source

    runner = (SCHEDULER_PY.parent / "model_runner.py").read_text(
        encoding="utf-8")
    assert ("TODO: remove this when we support mixed prefill and decode in "
            "one batch") in runner
    # The early return that TODO guards: with any prefill request in the
    # batch, input preparation hands back the prefill slice alone, so a mixed
    # batch's decode rows would never reach the model.
    assert re.search(r"if total_reqs_prefill > 0:\s*\n\s*return "
                     r"self\.input_ids\.gpu\[:total_tokens_prefill\]", runner)


def test_two_batch_overlap_does_not_change_the_answer():
    """TBO is a runner-side split, not a scheduling mode.

    Worth stating because "TBO-off" is how the question was posed. `enable_tbo`
    appears nowhere in the scheduler: the batch is already formed when
    `maybe_create_ubatch_slices` divides it by request index, so both halves
    inherit the one kind the scheduler gave it. Turning TBO on cannot produce a
    mixed batch, and `enable_tbo_decode` gates the split to decode anyway.
    """
    assert "tbo" not in SCHEDULER_PY.read_text(encoding="utf-8").lower()

    from atom.config import Config
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(Config)}
    assert fields["enable_tbo"].default is False
    assert fields["enable_tbo_decode"].default is False
