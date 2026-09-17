"""Synthetic prompts of an exact token length.

A workload description says how many tokens a request carries, and a driver has
to turn that into text. The obvious construction -- one made-up word per token,
``" ".join(f"w{i}x{j}" ...)`` -- is not one token per word and does not even
hold a constant ratio: ``w3x1`` is one token and ``w3x262143`` is several, so
against the Qwen3.8-27B tokenizer 64 asked gave 314, 1024 gave 6062 and 262144
gave 2248190, a ratio climbing 4.91, 5.92, 8.58.

That is survivable for a campaign read by the shapes its steps actually
recorded, and fatal for one meant to reproduce a measured length distribution,
since a length-dependent distortion reshapes the distribution rather than
shifting it.

So: one single-token word per token. Exact by construction rather than by
measurement, which means it depends on the tokenizer -- see `verify` for the
check, and `replay.py --check-lengths` for the same check against a running
server, which is the one that costs nothing because the count is already in
every response.
"""
from __future__ import annotations

#: Words that are exactly one token each under the Qwen tokenizers, checked
#: with a leading space, which is how a tokenizer sees a word mid-sentence. Any
#: single-token vocabulary would do; these are common English words, so a prompt
#: built from them is ordinary text rather than something a tokenizer or a cache
#: might treat specially.
WORDS = (
    "the of and to in a is that it for on with as was at by an be this from "
    "or has had not but they we you all can her his its our out over new one "
    "two three four five six seven eight nine ten time year day work part"
).split()


def prompt_of_tokens(tokens: int, index: int = 0) -> str:
    """Distinct text of exactly ``tokens`` tokens.

    The first four words encode ``index`` in base 52, which is what keeps two
    requests from sharing prefix-cache blocks: blocks are hashed over the
    sequence from position zero, so differing in the first token makes every
    later block differ too. 52**4 is 7.3 million distinct openings.
    """
    if tokens <= 0:
        return ""
    base = len(WORDS)
    head, n = [], max(0, int(index))
    for _ in range(4):
        head.append(WORDS[n % base])
        n //= base
    words = (head + [WORDS[0]] * tokens)[:tokens]
    # A leading space so the first word tokenises the same way as the rest.
    return " " + " ".join(words)


def verify(tokenizer, lengths=(1, 64, 4096, 262144)) -> dict:
    """Achieved token count per requested one, for a given tokenizer.

    Exactness is a property of the vocabulary above under a particular
    tokenizer, so a caller on a different model family should check rather than
    assume. Returns ``{asked: got}``.
    """
    return {n: len(tokenizer(prompt_of_tokens(n, 7),
                             add_special_tokens=False).input_ids)
            for n in lengths}


#: Words at the start of a source block that carry which block it is. Six is
#: enough for this corpus by a wide margin -- 52**6 is 2e10 against 1e8 distinct
#: (session, hash_id) pairs -- and the number that matters is the upper bound,
#: not the margin: it must stay inside one *native* KV block, which is
#: `block_size` tokens and defaults to 16. Two different source blocks that
#: agreed on their first 16 tokens would earn a native prefix hit the trace does
#: not say exists, and a replay that invents reuse is not this workload.
ENCODE_WORDS = 6

#: Room reserved per session in the ordinal space, so a session-local id can
#: never reach into the next session's. The corpus tops out at 238450, well
#: inside this; `prompt_of_hash_ids` refuses anything larger rather than let two
#: sessions collide, because a collision does not fail, it silently invents
#: reuse and the run still reports "0 failed".
SESSION_STRIDE = 1 << 24

#: Source blocks are this many tokens in cc-traces, which is *not* the engine's
#: `block_size` (16). It does not need to be: 64 is a multiple of 16, so a
#: shared run of whole source blocks is automatically a whole number of native
#: blocks, and the engine can stay at the setting everything else was measured
#: at. Passing --block-size 64 to match would be the same replay with the rest
#: of the calibration invalidated.
SOURCE_BLOCK_TOKENS = 64


def _block_words(ordinal: int, block_size: int) -> list[str]:
    """One source block's worth of words, identified by `ordinal`."""
    base = len(WORDS)
    head, n = [], max(0, int(ordinal))
    for _ in range(ENCODE_WORDS):
        head.append(WORDS[n % base])
        n //= base
    return (head + [WORDS[0]] * block_size)[:block_size]


def prompt_of_hash_ids(hash_ids, tokens: int, session: int = 0,
                       block_size: int = SOURCE_BLOCK_TOKENS) -> str:
    """Text whose blocks are shared exactly where `hash_ids` says they are.

    `prompt_of_tokens` is built to *defeat* prefix caching -- it varies the
    opening so no two requests share a block -- because a sweep measuring
    prefill must actually perform it. This is the opposite construction, and the
    two are both needed: a cc-traces session re-sends its whole conversation
    each turn, so replaying only the lengths measures a workload with the right
    arrival process, the right length multiset, and none of the reuse. That is a
    fair test of step prediction and not a replay of this corpus.

    Each `hash_id` names a 64-token block of the source prompt, and requests in
    a session share their leading ids. Mapping each id to a fixed block of text
    makes two requests that share leading ids share byte-identical leading
    tokens, and `BlockManager.compute_hash` chains from position zero, so the
    engine finds the same reuse the trace recorded. Ids are scoped to a session
    (`hash_id_scope: "local"`), so `session` must namespace them or two sessions
    would collide into reuse neither one had.

    The count is `len(hash_ids) * 64` on every row of this corpus, checked, so
    the truncation below only ever trims a partly-filled final block.
    """
    if tokens <= 0:
        return ""
    stride = max(1, int(block_size))
    words: list[str] = []
    for h in hash_ids:
        if len(words) >= tokens:
            break
        if not 0 <= int(h) < SESSION_STRIDE:
            raise ValueError(
                f"hash_id {h} is outside the {SESSION_STRIDE} reserved per "
                f"session, so it would land in another session's block and "
                f"invent reuse that session never had. Widen SESSION_STRIDE.")
        words.extend(_block_words(int(session) * SESSION_STRIDE + int(h), stride))
    words = (words + [WORDS[0]] * tokens)[:tokens]
    # A leading space so the first word tokenises the same way as the rest.
    return " " + " ".join(words)
