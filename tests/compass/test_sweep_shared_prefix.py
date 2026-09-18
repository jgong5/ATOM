"""The sweep's shared-prefix rounds, which are the only cache hits it records.

Every other round gives each prompt a distinct opening so none share a block.
That caps the axis decode is fitted against: decode is fitted per CUDA-graph
rung on *total* context, the sum across the batch, and with no sharing that sum
can never exceed the KV pool, because every token in it is a token stored.
Measured on the 27B at TP1 the pool is 76596 blocks of 16 -- 1225536 tokens --
and the sweep's rung-16 rounds already sat at 86% of it.

A cc-traces replay is not bounded that way: a shared block is stored once and
counted once per request, and one reached 2.9x the whole pool at rung 48,
against a table whose rung-48 samples stopped at 10752. These rounds reach the
same region the same way, so what they have to prove is that the sharing is
real and lands exactly where it was asked for -- text that merely looks similar
shares no blocks at all, and would leave the table exactly where it was while
looking like it had been fixed.

Loaded by path rather than imported, so the check does not need the engine:
`workload.py` has no intra-package imports and these are arithmetic invariants.
"""
import importlib.util
from pathlib import Path

_MODULE = (Path(__file__).resolve().parents[2] / "atom/compass/workload.py")


def _module():
    spec = importlib.util.spec_from_file_location("compass_workload", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words(prompt):
    return prompt.split()


class TestTheSharingIsWhereItWasAskedFor:
    def test_the_prefix_is_identical_across_the_batch(self):
        mod = _module()
        prompts = mod.shared_prefix_prompts([4096] * 8, 2048, session=3)
        assert len({" ".join(_words(p)[:2048]) for p in prompts}) == 1

    def test_and_they_diverge_at_the_first_token_past_it(self):
        mod = _module()
        prompts = mod.shared_prefix_prompts([4096] * 8, 2048, session=3)
        # If they agreed further the physical cost would be lower than the
        # round was sized for, which is the failure that does not announce
        # itself: the run fits, finishes, and measures a smaller batch.
        assert len({_words(p)[2048] for p in prompts}) == 8

    def test_every_prompt_is_exactly_the_length_asked_for(self):
        mod = _module()
        prompts = mod.shared_prefix_prompts([4096, 8192, 4096], 2048, 1)
        assert [len(_words(p)) for p in prompts] == [4096, 8192, 4096]

    def test_a_length_that_is_not_a_whole_block_is_still_exact(self):
        mod = _module()
        # 4100 is not a multiple of 64, so the final block is partly filled and
        # trimmed. Rounding the block count down instead would cost the prompt
        # its last four tokens.
        prompts = mod.shared_prefix_prompts([4100] * 2, 2048, 0)
        assert [len(_words(p)) for p in prompts] == [4100, 4100]

    def test_a_prefix_longer_than_the_prompt_shares_the_whole_prompt(self):
        mod = _module()
        prompts = mod.shared_prefix_prompts([1024] * 3, 4096, 0)
        assert len({p for p in prompts}) == 1
        assert all(len(_words(p)) == 1024 for p in prompts)


class TestTheShapeTheRoundsWereSizedFor:
    def test_physical_blocks_are_shared_plus_one_tail_each(self):
        mod = _module()
        block = mod.SOURCE_BLOCK_TOKENS
        length, count, shared = 4096, 8, 2048
        prompts = mod.shared_prefix_prompts([length] * count, shared, 0)
        # Distinct 64-token block texts across the batch. The rounds were
        # chosen by this arithmetic -- logical total context up to 6.29M
        # against a 1.23M-token pool -- so if it does not hold the run either
        # OOMs or measures a rung it was not aimed at.
        blocks = set()
        for prompt in prompts:
            words = _words(prompt)
            for start in range(0, len(words), block):
                blocks.add(" ".join(words[start:start + block]))
        expected = (shared + count * (length - shared)) // block
        assert len(blocks) == expected


class TestRoundsDoNotShareWithEachOther:
    def test_two_sessions_share_nothing(self):
        mod = _module()
        first = mod.shared_prefix_prompts([4096] * 4, 2048, session=0)
        second = mod.shared_prefix_prompts([4096] * 4, 2048, session=1)
        # The sweep runs its round list twice so the outlier pass can tell a
        # Triton-tuning visit from a steady-state one. Were the second visit
        # served from the first's cache there would be no second measurement,
        # only a second cache hit.
        assert _words(first[0])[0] != _words(second[0])[0]


class TestNoSharingIsStillNoSharing:
    def test_a_zero_prefix_leaves_every_prompt_distinct(self):
        mod = _module()
        prompts = mod.shared_prefix_prompts([1024] * 4, 0, session=0)
        assert len({_words(p)[0] for p in prompts}) == 4
