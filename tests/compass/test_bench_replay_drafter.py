# SPDX-License-Identifier: MIT
"""The bench under a drafter: a step replayed as the batch ATOM's scheduler
builds for it, and each repeat's window split into the target's part and the
drafter's.

Batches come from ATOM's own `Scheduler` under a DSpark drafter of five drafts
and an MTP drafter of one, every draft accepted, replied to by the simulated
runner's token stream; each is rebuilt from its step-record view by the replay
rule over `test_bench_replay`'s tokenizer. The window runs on
`test_bench_window`'s planted device, with a drafter whose passes queue device
work of their own; `measure steps` on `test_bench_measure`'s fake bench.
"""

import itertools
import queue
import statistics
from types import SimpleNamespace

import pytest
from test_bench_measure import Bench as FakeBench
from test_bench_measure import (
    by_view,
    run,  # noqa: F401  the run fixture
)
from test_bench_window import MS, PREPARE, Planted
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_parity_device_window import _Processor
from test_parity_record import BLOCK, _config

from atom.compass.bench import measure, replay, window
from atom.compass.bench.window import BenchWindow
from atom.compass.fit.samples import load
from atom.compass.parity import StepRecording
from atom.compass.runner.step_output import DeferredTokenStream
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

#: Each drafter: its draft width and whether it is DSpark.
DRAFTERS = {"dspark5": (5, True), "mtp1": (1, False)}
K = DRAFTERS["dspark5"][0]
#: Prompt lengths: two open in one prefill, the third chunked behind them.
PROMPTS = (23, 40, 90)


def spec(drafts, dspark):
    return SimpleNamespace(num_speculative_tokens=drafts, use_dspark=lambda: dspark)


def config(model_dir, drafter):  # noqa: F811
    built = _config(blocks=256, speculative_config=spec(*DRAFTERS[drafter]))
    built.model, built.trust_remote_code = str(model_dir), False
    built.hf_config = SimpleNamespace(model_type="llama")
    return built


def scheduled(built):
    """Each batch ATOM's scheduler builds to serve `PROMPTS` under `built`,
    with its rows as a step record states them and its block tables' lengths,
    read as it is built: the scheduler goes on to extend the tables."""
    scheduler = Scheduler(built)
    drafts = built.speculative_config.num_speculative_tokens
    stream = DeferredTokenStream([7], rates=[1.0] * drafts)
    params = SamplingParams(max_tokens=16, ignore_eos=True)
    for i, n in enumerate(PROMPTS):
        tokens = list(range(5 + 1000 * i, 5 + 1000 * i + n))
        scheduler.add(Sequence(tokens, BLOCK, params))
    out = []
    while scheduler.has_unfinished_requests():
        assert len(out) < 100, "the served requests do not finish"
        scheduled = scheduler.schedule()
        if scheduled is None or scheduled[0] is None:
            continue
        batch, seqs = scheduled
        prefill = batch.total_seqs_num_prefill
        rows = zip(batch.num_scheduled_tokens, batch.context_lens)
        line = {
            "rows": [[int(q), int(c), i >= prefill] for i, (q, c) in enumerate(rows)],
            "produces_output": batch.produces_output(),
        }
        out.append((batch, line, [len(t) for t in batch.block_tables]))
        scheduler.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(**stream.step(batch)),
            stream_output_queue=queue.Queue(),
            batch=batch,
        )
    return out


def anchors(batch):
    """Which rows a drafter is handed a successor token for, or None."""
    ids = batch.next_token_ids
    return None if ids is None else [token != -1 for token in ids]


@pytest.mark.parametrize("drafter", DRAFTERS)
def test_a_replayed_step_is_the_batch_atoms_scheduler_built_under_a_drafter(
    model_dir, drafter  # noqa: F811
):
    """Its rows, drafts and successor tokens, and on a step that produces
    output its KV blocks, the draft round's positions past each row included,
    which under five drafts take a block past some row's own: a middle chunk's
    rows hold blocks for the whole prompt in ATOM."""
    built = config(model_dir, drafter)
    rule, (drafts, _) = replay.ReplayRule(built), DRAFTERS[drafter]

    steps = scheduled(built)

    kinds, past = set(), False
    for batch, line, tables in steps:
        rebuilt = rule.batch(line)
        decode = bool(batch.total_seqs_num_decode)
        kinds.add((decode, batch.produces_output()))
        assert rebuilt.num_scheduled_tokens.tolist() == line_tokens(line)
        assert rebuilt.context_lens.tolist() == batch.context_lens.tolist()
        assert rebuilt.num_spec_step == batch.num_spec_step == drafts * decode
        if decode:
            shape = (len(line["rows"]), drafts)
            assert rebuilt.scheduled_spec_decode_tokens.shape == shape
            assert batch.scheduled_spec_decode_tokens.shape == shape
            assert {q for q, _, _ in line["rows"]} == {drafts + 1}
        assert anchors(rebuilt) == anchors(batch)
        if batch.produces_output():
            assert [len(t) for t in rebuilt.block_tables] == tables
            own = [-(-c // BLOCK) for _, c, _ in line["rows"]]
            past |= tables != own
    assert kinds == {(False, False), (False, True), (True, True)}
    assert past == (drafter == "dspark5")
    assert rebuilt.replay_rule == {"name": "fresh-contiguous", "version": 3}


def line_tokens(line):
    return [q for q, _, _ in line["rows"]]


@pytest.mark.parametrize(
    "drafter, rows, refusal",
    [
        ("dspark5", [[1, 9, True]], "decode other than 6 in a row"),
        ("dspark5", [[7, 9, True]], "decode other than 6 in a row"),
        ("mtp1", [[6, 9, True]], "decode other than 2 in a row"),
    ],
)
def test_a_decode_row_of_another_width_is_refused(
    model_dir, drafter, rows, refusal  # noqa: F811
):
    rule = replay.ReplayRule(config(model_dir, drafter))
    with pytest.raises(ValueError, match=f"{refusal}, a speculative step"):
        rule.batch({"rows": rows, "produces_output": True})


TARGET, KV, PROPOSE = 3 * MS, 400_000, 300_000  # ns of device work


class Drafted(Planted):
    """`Planted`, with a DSpark drafter: on a forward that produces output its
    draft-KV pass runs before the target samples and its propose pass after;
    on a middle chunk the draft-KV pass alone runs, after the target."""

    def __init__(self, issue, busy):
        self.tokenID_processor = _Processor()
        self.drafter = SimpleNamespace(compute_draft_kv=lambda: self._queue(KV))
        super().__init__(issue, busy)

    def forward(self, batch):
        reply = super().forward(batch)
        self.drafter.compute_draft_kv()
        if batch.produces_output():
            self.tokenID_processor.prepare_sampled_ids(batch)
            self.propose_draft_token_ids()
        return reply

    def propose_draft_token_ids(self):
        self._queue(PROPOSE)


class Bench(BenchWindow, StepRecording, Drafted):
    pass


@pytest.mark.parametrize("output", [True, False])
def test_each_repeat_splits_where_a_real_steps_labels_split(output):
    """On a forward that produces output the target's part runs to its
    sampling, its draft-KV pass included, and the drafter's is the propose
    pass; on a middle chunk the target's ends where the draft-KV pass starts.
    The target's statistic is the median of its parts, F to B less the
    drafter's, over the same kept repeats. The host takes 2 ms to issue each
    warm-up, which sizes the preload, and 1.5 ms each repeat."""
    issue = itertools.chain([2 * MS] * window.WARMUPS, itertools.repeat(3 * MS // 2))
    busy = [TARGET + 1_000 * n * n for n in range(window.WARMUPS + window.REPEATS)]
    bench = Bench(issue, iter(busy))
    batch = SimpleNamespace(req_ids=[7, 8], rows=[], produces_output=lambda: output)

    result = bench.bench_step(batch)

    draft = PROPOSE if output else KV
    kept = busy[window.WARMUPS :]
    whole = [PREPARE + b + KV + PROPOSE * output for b in kept]
    assert result["repeats"] == [[w, PREPARE, w - PREPARE, draft] for w in whole]
    assert result["median_ns"] == statistics.median(w - draft for w in whole)
    assert result["draft_median_ns"] == draft
    assert result["discarded"] == 0


def test_a_measurement_holds_the_drafters_value_and_fit_reads_it_as_a_draft_label(
    run, monkeypatch  # noqa: F811
):
    """The fake bench replies under a drafter: the target's median at the
    batch's token count in microseconds, the drafter's at half that."""
    reply = FakeBench.measure

    def drafted(self, batch):
        return reply(self, batch) | {"draft_median_ns": 500 * batch.total_tokens_num}

    monkeypatch.setattr(FakeBench, "measure", drafted)
    rec = str(run.record)

    store = run(rec, "--sample", "all", *run.named(rec))

    held = by_view(store).values()
    for _, member in held:
        tokens = sum(q for q, *_ in member["rows"])
        assert (member["seconds"], member["draft_seconds"]) == (
            1e-6 * tokens,
            5e-7 * tokens,
        )
    samples = load({}, [key for key, _ in held], store.root)[0]
    assert sorted((s.device_s, s.draft_s) for s in samples) == sorted(
        (m["seconds"], m["draft_seconds"]) for _, m in held
    )


def test_each_ragged_verify_view_keeps_its_width_and_replays_in_the_pool(
    model_dir,  # noqa: F811
):
    """`measure steps --ragged` under DSpark5 composes from a seed of four
    verify rows, at the pool the seed fills with its draft rounds' positions:
    every view keeps six tokens a row and the replay rule builds it, the
    fullest holding the pool's last block with those positions."""
    cached, ladder = (40, 90, 150, 260), [1, 2, 4, 8]
    seed = {
        "rows": [[K + 1, c + K + 1, True] for c in cached],
        "rung": 4,
        "produces_output": True,
    }
    blocks = sum(-(-(c + K + 1 + K - 1) // BLOCK) for c in cached)
    engine = {
        "kv_cache_block_size": BLOCK,
        "max_model_len": 2048,
        "ladder": ladder,
        "num_kvcache_blocks": blocks,
        "speculative": ["dspark", K],
    }
    contexts = list(range(30, 500, 5))
    lines = [{"rows": [[K + 1, c, True]], "traces": ["trace-0"]} for c in contexts]
    built = config(model_dir, "dspark5")
    built.num_kvcache_blocks = blocks
    rule = replay.ReplayRule(built)

    composed = measure._ragged({"seed": {"view": seed}}, [("in", lines, engine)], 0)

    views = [point["view"] for point in composed.values()]
    assert len(views) > 10
    for view in views:
        assert {q for q, _, _ in view["rows"]} == {K + 1}
        rule.batch(view)
    held = [sum(-(-(c + K - 1) // BLOCK) for _, c, _ in v["rows"]) for v in views]
    assert max(held) == blocks
