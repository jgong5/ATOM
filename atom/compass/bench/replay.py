# SPDX-License-Identifier: MIT
"""A `ScheduledBatch` rebuilt from a step-record line, by a fixed, versioned rule.

A step-record line states a batch as its rows, in order, each with its query
tokens, its context tokens and whether it decodes, and whether the step
produces output. ATOM's forward also needs block tables, tokens and, for a
model that keeps per-request state, state slots. `ReplayRule.batch` supplies
them the same way for every line:

- Rows keep the line's order, and are all prefill or all decode: ATOM's
  runner runs a batch holding prefill tokens over its prefill rows alone, and
  its scheduler builds no batch that mixes the two.
- A decode row computes the engine's decode width: one token, or under a
  drafter of draft width K the K + 1 of a verify step, the last of its
  tokens and K drafts, each draft the row's filler, as the scheduler hands
  the batch K drafts a row.
- No row holds more than `max_model_len` tokens: ATOM's scheduler admits no
  such request, and its runner's block tables have no room for one.
- Each row is a new request, numbered by `Sequence`'s own counter, so no two
  replays share a request.
- Row i holds `ceil(context / block_size)` KV blocks, contiguous, from a fresh
  pool: row 0 starts at block 0, or the block `batch` is given, and each row
  starts where the one before ended. Under a drafter each row also holds the
  blocks of the positions past its context its draft round writes
  (`draft_kv_lookahead`), as the scheduler allocates them. A line stating
  `shared`, n, has every
  row's first n blocks be the same n blocks, ahead of the rows' own, as a
  prefix cache shares them: they hold cached tokens alone, so n is at most
  the full blocks each row's cached context fills.
- Every token of row i is the filler id `DeferredTokenStream` reports for a
  request numbered i, out of the ids `filler_token_ids` keeps for the model's
  tokenizer, as the simulated runner keeps them.
- Under a model that keeps per-request state, row i holds state slot i of a
  fresh pool, or that many past the slot `batch` is given.
- A prefill line that produces output has every row on its final chunk, one
  that does not has none: a line does not say which of its chunks ends a
  prompt, and ATOM's runner samples every row of a step that produces output.
  Under a drafter that reads a middle chunk's successor token, each row's
  successor is its filler, as the scheduler hands it the prompt's next token.
- Every row samples with the request parameters the rule was given, ATOM's
  `SamplingParams` defaults unless stated.
- The batch carries the rule's `NAME` and `VERSION` as `replay_rule`, which a
  measurement records.

The line's `rung` is not read: the runner a batch is handed to settles its
own. The tokens are fillers, not the recorded ones, so a replayed MoE step may
route otherwise than the recorded step did.
"""

import numpy as np

from atom.compass.runner.step_output import filler_token_ids
from atom.model_engine.block_manager import draft_kv_lookahead
from atom.model_engine.llm_engine import InputOutputProcessor, _load_tokenizer
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import (
    Sequence,
    SequenceType,
    new_block_table,
)
from atom.sampling_params import SamplingParams

NAME = "fresh-contiguous"
VERSION = 2


class ReplayRule:
    """The rule for one engine configuration: its tokenizer, block size, KV
    pool, drafter, and whether its model keeps per-request state; and for the
    request parameters `sampling` states, as `SamplingParams` fields."""

    def __init__(self, config, sampling: dict | None = None) -> None:
        self.sampling = SamplingParams(**(sampling or {}))
        tokenizer = _load_tokenizer(config.model, config.trust_remote_code)
        self.fillers = filler_token_ids(
            tokenizer, config.eos_token_id, config.stop_token_ids
        )
        if not self.fillers:
            raise ValueError(
                f"the tokenizer of {config.model!r} has no id that is not a stop "
                "id and decodes on its own to ASCII letters and digits, so there "
                "is no filler token to replay a batch with"
            )
        self.block_size = config.kv_cache_block_size
        self.num_blocks = config.num_kvcache_blocks
        self.max_model_len = config.max_model_len
        self.stateful = (
            config.hf_config.model_type
            in InputOutputProcessor._per_req_cache_model_types()
        )
        spec = config.speculative_config
        self.drafts = spec.num_speculative_tokens if spec else 0
        self.lookahead = draft_kv_lookahead(self.drafts)
        self.successor = spec is not None and not spec.use_dspark()

    def batch(self, line: dict, block: int = 0, slot: int = 0) -> ScheduledBatch:
        """The `ScheduledBatch` `line`'s `rows` and `produces_output` state, its
        KV blocks from `block` on and its state slots from `slot` on."""
        rows = line["rows"]
        decode = rows[0][2]
        if any(kind != decode for _, _, kind in rows):
            raise ValueError(
                f"rows {rows} mix prefill and decode rows; ATOM's runner runs a "
                "batch holding prefill tokens over its prefill rows alone"
            )
        width = self.drafts + 1
        if decode and any(query != width for query, _, _ in rows):
            many = "more than one token" if width == 1 else f"other than {width}"
            raise ValueError(
                f"rows {rows} decode {many} in a row, a speculative step of other "
                f"than the engine's {self.drafts} drafts, which this rule does "
                "not build"
            )
        longest = max(context for _, context, _ in rows)
        if longest > self.max_model_len:
            raise ValueError(
                f"rows {rows} hold a row of {longest} tokens, past max_model_len "
                f"{self.max_model_len}"
            )
        final = bool(line["produces_output"])
        shared = line.get("shared", 0)
        full = min((context - query) // self.block_size for query, context, _ in rows)
        if type(shared) is not int or not 0 <= shared <= full:
            raise ValueError(
                f"rows {rows} share {shared!r} KV blocks, and the shortest "
                f"cached context among them fills {full}"
            )
        prefix = list(range(block, block + shared))
        block += shared
        seqs = {}
        fillers = [self.fillers[i % len(self.fillers)] for i in range(len(rows))]
        for i, (query, context, _) in enumerate(rows):
            seq = Sequence(
                [fillers[i]] * context,
                self.block_size,
                self.sampling,
                has_per_req_cache=self.stateful,
            )
            seq.type = SequenceType.DECODE if decode else SequenceType.PREFILL
            seq.num_cached_tokens = context - query
            own = -(-(context + self.lookahead) // self.block_size) - shared
            seq.block_table = new_block_table([*prefix, *range(block, block + own)])
            block += own
            if self.stateful:
                seq.state_slots = [slot + i]
            seqs[seq.id] = seq
        if block > self.num_blocks:
            raise ValueError(
                f"rows {rows} hold {block} KV blocks of {self.block_size} tokens, "
                f"and the pool has {self.num_blocks}"
            )
        queries = [query for query, _, _ in rows]
        tokens, n = sum(queries), len(rows)
        drafts = {}
        if decode and self.drafts:
            drafts = {
                "num_spec_step": self.drafts,
                "scheduled_spec_decode_tokens": {
                    seq: np.full(self.drafts, filler, dtype=np.int32)
                    for seq, filler in zip(seqs, fillers)
                },
            }
        if not decode and self.successor:
            drafts["next_token_ids"] = [-1 if final else f for f in fillers]
        batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=queries,
            total_tokens_num=tokens,
            total_tokens_num_prefill=0 if decode else tokens,
            total_tokens_num_decode=tokens if decode else 0,
            total_seqs_num=n,
            total_seqs_num_prefill=0 if decode else n,
            total_seqs_num_decode=n if decode else 0,
            is_final_chunk=None if decode else [final] * n,
            **drafts,
        )
        if batch.produces_output() != final:
            raise ValueError(
                f"rows {rows} decode, so the step produces output, and the line "
                "says it does not"
            )
        batch.replay_rule = {"name": NAME, "version": VERSION}
        return batch
