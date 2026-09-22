"""The step trace must say how much of a step's context was a cache hit.

Without it a short prefill step at a deep context is ambiguous between two
things that do not cost the same: the final chunk of a long chunked prefill,
which computes its tokens and writes their KV, and a prefix-cache hit, which
skips the write entirely. Both present as few scheduled tokens against a large
`context_lens`.

`batch.num_cached_tokens` cannot settle it. It is a cursor: it starts at the
prefix hit and then advances by every finished chunk, and `context_lens` is
built from it (`num_cached_tokens + num_scheduled_tokens`), so recording it
would repeat what the trace already says. `seq.prefix_cache_hit_tokens` is set
once at admission and does not move, which is the quantity a cost model needs.

This matters because the two instruments disagree. The calibration sweep gives
every prompt a distinct opening (`scripts/compass/run.py`), so not one sample in
it is a cache hit; a cc-traces replay re-sends whole conversations and is mostly
hits. A table that does not record the difference cannot say whether the sweep
and the replay measured the same shape -- which is exactly the question left
open when the four cc-traces rungs stayed biased low after the short-chunk
rungs went in.
"""

import numpy as np
from conftest import MockConfig

from atom.compass.core.cost.base import StepShape
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler


def _drain(sched, limit=64):
    """Run a prompt to the end of its prefill, returning every prefill batch.

    `schedule()` alone does not advance a chunked prefill: the frontier moves in
    `postprocess`, which the engine calls once the forward has come back. A loop
    that only schedules therefore sees the first chunk forever, and nothing is
    ever hashed into the prefix cache. Stand in for the forward with a
    zero-token output, which is all postprocess needs to commit the chunk.
    """
    batches = []
    for _ in range(limit):
        res = sched.schedule()
        if res is None:
            break
        batch, seqs = res
        if batch is None or not batch.req_ids:
            break
        if not batch.total_seqs_num_prefill:
            break
        batches.append(batch)
        n = len(batch.req_ids)
        sched.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(0,)] * n,
                num_rejected=np.zeros(n, dtype=np.int32),
                num_bonus=np.zeros(n, dtype=np.int32),
                draft_token_ids=None,
                is_deferred_out=False,
            ),
            batch=batch,
        )
    return batches


class TestBatchCarriesTheAdmissionHit:
    def _sched(self, **kw):
        cfg = dict(
            max_num_batched_tokens=32,
            max_model_len=256,
            num_kvcache_blocks=100,
            kv_cache_block_size=4,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
        )
        cfg.update(kw)
        return Scheduler(MockConfig(**cfg))

    def test_cold_chunked_prefill_reports_no_hit_on_any_chunk(self, seq_factory):
        """The frontier climbs; the hit stays at zero. That is the distinction.

        A cache-cold prompt chunked into several forwards produces middle chunks
        whose `context_lens` is large and whose `num_scheduled_tokens` is small
        -- the exact shape a cache hit produces -- and none of them is a hit.
        """
        sched = self._sched()
        sched.add(seq_factory(list(range(128)), block_size=4))
        batches = _drain(sched)

        assert len(batches) > 1, "prompt did not chunk; the test shape is absent"
        for batch in batches:
            assert list(batch.prefix_cache_hit_tokens) == [0]
        # The cursor moved even though the hit did not, which is why the two
        # cannot be the same field.
        frontiers = [int(b.num_cached_tokens[0]) for b in batches]
        assert frontiers == sorted(frontiers)
        assert frontiers[-1] > frontiers[0]

    def test_a_repeat_prompt_reports_its_hit(self, seq_factory):
        """Re-sending the same prompt admits against the cache, and says so."""
        sched = self._sched()
        tokens = list(range(128))
        sched.add(seq_factory(tokens, block_size=4))
        _drain(sched)

        sched.add(seq_factory(list(tokens), block_size=4))
        repeat = _drain(sched)

        assert repeat, "the second prompt never scheduled"
        hit = int(repeat[0].prefix_cache_hit_tokens[0])
        assert hit > 0, "identical prompt admitted with no prefix-cache hit"
        # Reported as the admission hit on every chunk of that prompt, not
        # re-derived per chunk -- a reader summing it would double-count.
        for batch in repeat:
            assert int(batch.prefix_cache_hit_tokens[0]) == hit


class TestShapeCarriesItThrough:
    def test_step_shape_defaults_to_silent_rather_than_zero(self):
        """An engine that does not track hits must not claim there were none.

        Empty says "not recorded"; a tuple of zeros says "measured, and there
        were none". A sweep table written before this field existed is the first
        case, and must not be read as the second.
        """
        shape = StepShape(num_scheduled_tokens=(16,), context_lens=(4096,),
                          num_prefill_tokens=16)
        assert shape.prefix_cache_hit_tokens == ()

    def test_hit_is_per_request_and_in_batch_order(self):
        shape = StepShape(num_scheduled_tokens=(16, 8),
                          context_lens=(4096, 2048),
                          num_prefill_tokens=24,
                          prefix_cache_hit_tokens=(4080, 0))
        assert len(shape.prefix_cache_hit_tokens) == shape.batch_size
        assert shape.prefix_cache_hit_tokens[1] == 0
