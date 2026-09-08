"""A prompt of N tokens must be N tokens.

The old construction was one made-up word per token and those are several
tokens each -- more as the counter grows, so the error was not even a constant
factor. Campaigns read by the shapes their steps recorded survived that; a
benchmark meant to reproduce a measured length distribution would not, because
a length-dependent distortion reshapes a distribution rather than shifting it.
"""
from atom.compass.workload import WORDS, prompt_of_tokens


class TestExactLength:
    def test_a_prompt_has_one_word_per_token(self):
        # Exactness against a real tokenizer needs the model; the structural
        # guarantee is one whitespace-separated word per requested token, and
        # every word being a single token is the vocabulary's job.
        for n in (1, 2, 64, 1024, 40000):
            assert len(prompt_of_tokens(n, 3).split()) == n

    def test_zero_and_below_are_empty(self):
        assert prompt_of_tokens(0, 1) == ""
        assert prompt_of_tokens(-5, 1) == ""

    def test_it_starts_with_a_space(self):
        """So the first word tokenises the way the rest do."""
        assert prompt_of_tokens(4, 0).startswith(" ")

    def test_a_short_prompt_is_not_padded_past_its_length(self):
        assert len(prompt_of_tokens(2, 999).split()) == 2


class TestRequestsDoNotSharePrefixBlocks:
    """Blocks are hashed from position zero, so differing in the first token is
    enough to make every later block differ -- and two requests sharing a
    prefix would let the second skip the prefill being measured."""

    def test_the_opening_differs_between_requests(self):
        openings = {tuple(prompt_of_tokens(64, i).split()[:4])
                    for i in range(2000)}
        assert len(openings) == 2000

    def test_the_opening_encodes_the_index_in_base_52(self):
        assert len(WORDS) == 52
        assert prompt_of_tokens(4, 0).split()[:2] == [WORDS[0], WORDS[0]]
        assert prompt_of_tokens(4, 1).split()[0] == WORDS[1]
        assert prompt_of_tokens(4, 52).split()[:2] == [WORDS[0], WORDS[1]]
