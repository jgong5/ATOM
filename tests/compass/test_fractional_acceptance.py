# SPDX-License-Identifier: MIT
"""A fractional forced acceptance length, served as `compass run` serves it.

MiniMax-M3's InferenceX submission passes ATOM `--method eagle3
--num-speculative-tokens 3 --spec-decode-acceptance-length 2.78`. ATOM's
`SpeculativeConfig` resolves the length into per-position rates, and a
predicted step draws each verify row's accepted drafts from them: the runner's
first step builds its `DeferredTokenStream` from the config's rates, seeded by
its DP rank. Here ATOM's scheduler serves requests four at a time to their end
through that runner's `forward`, and the run summary reads the mean
acceptance length off the scheduler's histogram, as a simulated run's does.
"""

import argparse
import itertools
import queue
from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_run_refusal_summary import STAGE1, summary
from transformers import PretrainedConfig

import atom.config
from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.runner import overrides
from atom.compass.runner.overrides import NonAllocatingRunner, install_cost_backend
from atom.model_engine import llm_engine
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

#: The submission's speculative flags, but the acceptance length.
EAGLE3 = ["--method", "eagle3", "--draft-model", "draft", "--num-speculative-tokens"]
DRAFTS = 3
#: Requests, the concurrency they are served at, and each one's output length.
REQUESTS, CONCURRENCY, MAX_TOKENS = 32, 4, 400

# The price is not the subject here; `test_speculative_step.py` prices drafts.
pytestmark = pytest.mark.usefixtures("unpriced")


def speculative_config(monkeypatch, length):
    """The speculative config ATOM's arguments resolve for `length`."""
    monkeypatch.setattr(
        atom.config, "get_hf_config", lambda *a, **k: PretrainedConfig()
    )
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    argv = [*EAGLE3, str(DRAFTS), "--spec-decode-acceptance-length", str(length)]
    args = parser.parse_args(argv)
    return EngineArgs.from_cli_args(args)._get_engine_kwargs()["speculative_config"]


class _Runner(NonAllocatingRunner):
    """The mixin under test, with the attributes its `forward` reads; its
    token stream is left for the first step to build."""

    def __init__(self, config):
        self.config = config
        install_cost_backend(self, StepStubBackend())
        self.tokenID_processor = SimpleNamespace(is_deferred_out=True)


def serve(monkeypatch, spec, dp_rank=0):
    """ATOM's scheduler after serving `REQUESTS` requests to their end, each
    step the runner's reply, and the number of steps that took."""
    monkeypatch.setattr(llm_engine, "_load_tokenizer", lambda *_: None)
    monkeypatch.setattr(overrides, "filler_token_ids", lambda *_: [7])
    config = MockConfig(
        speculative_config=spec,
        max_num_seqs=CONCURRENCY,
        num_kvcache_blocks=4096,
        kv_cache_block_size=16,
        max_model_len=1024,
        max_num_batched_tokens=512,
        pipeline_parallel_size=1,
        model="m",
        trust_remote_code=False,
        parallel_config=SimpleNamespace(data_parallel_rank=dp_rank),
    )
    scheduler, runner = Scheduler(config), _Runner(config)
    Sequence.counter = itertools.count()
    params = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True)
    seqs = [
        Sequence(list(range(5, 69)), 16, sampling_params=params)
        for _ in range(REQUESTS)
    ]
    for s in seqs:
        scheduler.add(s)
    out = queue.Queue()
    for steps in itertools.count(1):
        assert steps < 2 * REQUESTS * MAX_TOKENS, "the requests do not finish"
        if not scheduler.has_unfinished_requests():
            break
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None:
            continue
        batch, rows = scheduled
        scheduler.compute_detailed_aggregates(batch, rows)
        reply = runner.forward(batch)
        scheduler.postprocess(
            list(rows.values()), reply, stream_output_queue=out, batch=batch
        )
    assert [s.num_completion_tokens for s in seqs] == [MAX_TOKENS] * REQUESTS
    assert runner._token_stream.rates == spec.synthetic_acceptance_rates
    return scheduler, steps


def test_a_fractional_length_is_accepted_on_average(monkeypatch, tmp_path):
    """At 2.78 the summary's mean acceptance length is within 1% of 2.78, and
    each position is accepted at the rate ATOM resolved for it, within 4
    standard errors, on two DP ranks whose seeds draw differently."""
    spec = speculative_config(monkeypatch, 2.78)
    ranks = [serve(monkeypatch, spec, dp_rank=r)[0] for r in (0, 1)]
    said = summary(
        monkeypatch, tmp_path, StepStubBackend(STAGE1), 8, scheduler=ranks[0]
    )["speculative"]["engine"]
    print("speculative:", said)
    assert said["input"] == {
        "flag": "--spec-decode-acceptance-length",
        "value": 2.78,
        "method": "declared",
    }
    assert said["mean_acceptance_length"] == pytest.approx(2.78, rel=0.01)

    histograms = [dict(s.engine_stats.distribution) for s in ranks]
    for distribution in histograms:
        rows = sum(distribution.values())
        assert rows > 4000
        for i, rate in enumerate(spec.synthetic_acceptance_rates):
            seen = sum(n for a, n in distribution.items() if a > i) / rows
            assert abs(seen - rate) <= 4 * (rate * (1 - rate) / rows) ** 0.5
    assert histograms[0] != histograms[1]


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_an_integer_length_is_accepted_on_every_row(monkeypatch, length):
    """At an integer length every verify row accepts `length - 1` drafts,
    whatever the stream's seed, so two DP ranks take the same steps."""
    spec = speculative_config(monkeypatch, length)
    ranks = [serve(monkeypatch, spec, dp_rank=r) for r in (0, 1)]
    for scheduler, _ in ranks:
        distribution = scheduler.engine_stats.distribution
        assert {a for a, n in distribution.items() if n} == {length - 1}
    assert ranks[0][1] == ranks[1][1]
