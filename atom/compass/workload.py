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
