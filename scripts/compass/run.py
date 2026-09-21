"""Serve a fixed set of prompts and record each request's latency.

Used by `validate.py` for both halves of the comparison, so that the
real run and the simulated one differ in exactly one thing: whether the forward
pass happened. Anything else that differed would show up as model error.

Prompts are synthetic and fixed-length. Real text would make prompt length vary
with the tokenizer, and the point here is a controlled comparison, not a
realistic workload.
"""

import argparse
import json
import sys
import time

from atom import SamplingParams
from atom.compass.workload import prompt_of_tokens, shared_prefix_prompts
from atom.model_engine.arg_utils import EngineArgs
from atom.utils.arg_parser import FlexibleArgumentParser


def main() -> int:
    parser = FlexibleArgumentParser(description="ATOMCompass fixed-workload run")
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt-tokens", type=int, default=64,
        help="tokens per prompt, exactly. Was words until now, and a word is "
             "five to eight tokens, so an invocation pinned to a value here "
             "produces a different (smaller) shape than it used to -- 64 gave "
             "314 tokens, 2400 gave 15694. Artifacts already on disk are keyed "
             "to the shapes their steps recorded and are unaffected.")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sweep-long-decode", type=int, default=64,
        help="tokens to generate per long round. Decode is fitted per "
             "CUDA-graph rung against total context, so a rung asked about "
             "millions of tokens of history needs samples there; four tokens "
             "a round leaves it fitted on short context and under-predicting "
             "time per output token by 71%%.")
    parser.add_argument(
        "--sweep-long", action="store_true",
        help="add long-context rounds to --sweep, out to a 262144-token "
             "prompt. Needed before predicting a workload with long prompts: "
             "the default ladder stops at 1024 tokens, so a model fitted to it "
             "has no evidence about attention over a long history. Chunked "
             "prefill keeps the steps themselves small, so this costs about "
             "684k tokens of forward.")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Calibration workload: several rounds of varied prompt length and "
             "batch size, so the table holds prefill steps across a range of "
             "sizes rather than the one or two a fixed workload produces. "
             "Fitting three coefficients needs more than two samples, and a "
             "fixed workload prefills everything in a single step.",
    )
    parser.add_argument(
        "--sweep-rounds", default="",
        help="which rounds to run, as `start:stop` over the executed sequence "
             "-- the round list twice. Default: all of them. The full sweep "
             "is six hours on a 27B, and a GPU fault partway through loses "
             "every round behind it, so the remainder has to be runnable on "
             "its own. Indices are stable: a round seeds its prompts from its "
             "index, so round 111 builds the same prompts whether it runs "
             "first or last, and segments concatenate into one table.")
    parser.add_argument(
        "--sweep-round", action="append", default=[], metavar="L,C,S[,D[,SEED]]",
        help="run one explicit round instead of the built-in list: length, "
             "count, shared-prefix tokens, optionally decode tokens and the "
             "prompt seed. Repeatable. For reproducing one round without the "
             "six hours in front of it -- which is what finding the batch "
             "size a round faults at needs.")
    args = parser.parse_args()

    llm = EngineArgs.from_cli_args(args).create_engine()

    # Distinct prefixes: identical prompts would share prefix-cache blocks and
    # the second request onward would skip prefill entirely, which is a real
    # ATOM behaviour but not the one being measured here.
    prompts = [
        prompt_of_tokens(args.prompt_tokens, i)
        for i in range(args.num_prompts)
    ]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    if getattr(args, "compass_trace_prefill", 0) > 1:
        # Triton autotunes a shape on its first launch, benchmarking every
        # candidate configuration, so the first prefill of a workload records a
        # tuning run rather than a serving one. Warm the shapes first.
        #
        # The *same* prompts, not merely same-length ones. Qwen3.8-27B is a
        # hybrid: 48 of its layers are gated DeltaNet, whose Triton kernels
        # autotune per shape, and prompts differing by a token or two are a
        # different shape. Warming with `Warm {i}` text of the same word count
        # left 50451 tuning launches in a prefill graph of 51179 operators, and
        # the two ranks disagreed because they tuned for different times.
        #
        # Same prompts means prefix caching would let the second pass skip the
        # prefill this exists to record, so a trace run wants
        # `--no-enable_prefix_caching`. Said rather than forced: the flag
        # belongs to the caller, and a cold prefill does the same work either
        # way.
        if getattr(args, "enable_prefix_caching", False):
            print("warning: --compass-trace-prefill warms with the same prompts, "
                  "which prefix caching will then serve from cache -- pass "
                  "--no-enable_prefix_caching so the traced prefill is real",
                  file=sys.stderr)
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1))

    if args.sweep:
        # Each round is its own generate, so each contributes at least one
        # prefill step at a different size. Lengths and batch sizes vary
        # together because that is how they vary in a deployment.
        # Each round is one generate, so each contributes at least one prefill
        # step. Sizes are chosen to bracket what an evaluation will ask about
        # rather than to look thorough: ATOM batches several requests' prefill
        # into one step, so what lands in the table is the *batched* token
        # count, and a sweep of large prompts produces only large samples. A
        # model fitted to 1.7k-16k tokens and then asked about 500 extrapolates
        # below everything it has seen, where the intercept dominates and the
        # slope is doing no work.
        # Batch size matters as much as prompt length and is easier to forget:
        # the first version of this sweep varied only length, so it produced
        # decode steps at batch sizes 1-4 and the model was then asked about a
        # workload running 8 concurrent requests -- extrapolating outside its
        # evidence in a dimension nobody had thought to check. Coverage has to
        # bracket the evaluation in *every* dimension the model uses.
        # Concurrency is not a smooth dimension either. With CUDA graphs a
        # decode step replays the smallest capture size no smaller than the
        # batch, so cost steps at that ladder -- and the decode model is now
        # fitted per rung, which means a rung with no samples has no model. The
        # default ladder is [1,2,4,8,16,32,48,64,128,256], and stopping at 16
        # concurrent requests is what left a serving run at batch 63 asking
        # about a rung nothing had ever measured. Each rung appears at two
        # prompt lengths, because a rung needs its own context slope and one
        # length gives one band of history to fit it over.
        rounds = [
            (8, 1), (16, 2), (24, 4), (32, 1), (32, 8), (48, 2), (64, 1),
            (64, 4), (64, 12), (96, 8), (128, 1), (128, 4), (128, 16),
            (192, 2), (192, 8), (256, 1), (256, 6), (256, 12), (384, 2),
            (384, 8), (512, 1), (512, 4), (768, 2), (768, 6), (1024, 1),
            (1024, 3),
            # Rungs 32, 48 and 64.
            (64, 24), (256, 24), (64, 32), (256, 32),
            (64, 48), (192, 48), (64, 64), (128, 64),
        ]
        # (prompt tokens, concurrent requests) so far; the long rounds below
        # need a third field, because sampling decode is the point of them.
        rounds = [(length, count, args.max_tokens) for length, count in rounds]
        if args.sweep_long:
            # Long context. Everything above is a prompt of at most 1024
            # tokens, so the table it produces has no evidence past a context
            # of a few thousand -- and an agentic trace runs to 256k, where
            # attention over the history is most of the step. Asking the fitted
            # model about that is extrapolating two orders of magnitude outside
            # its evidence, in the one dimension where the cost is not linear.
            #
            # Prefill is chunked at `attn_prefill_chunk_size` (16384 by
            # default), so these do not produce enormous *steps*: a 262144
            # token prompt produces sixteen steps of 16384 tokens at contexts
            # 0, 16k, 32k and so on. That is exactly the coverage wanted, and
            # it is why this costs about 684k tokens in total rather than
            # anything alarming -- roughly a minute of forward.
            # Decode is fitted per CUDA-graph rung against *total* context --
            # the sum across the batch, not the average -- so a rung needs
            # samples spanning the total contexts it will be asked about. The
            # first version of these rounds varied only prompt length at one
            # request each, which covers prefill and leaves decode where it
            # was: the cc-traces pilot ran twenty concurrent requests at 164k
            # each, rung 32 at a total context of 3.3M, against rung-32 samples
            # taken at 2k and 8k. Four hundred times outside its evidence, and
            # it under-predicted time per output token by 71%.
            #
            # So each rung is sampled at two or three total contexts reaching
            # into the millions. Long prompts are the cheap way to get there:
            # 32 requests of 49152 tokens is a total context of 1.57M against a
            # pool of about 8M.
            long_decode = args.sweep_long_decode
            # A round longer than the model's window is rejected or truncated,
            # silently: the first version of these rounds asked for 262144
            # against a 262144 window and the tokens it generates, and the
            # coverage stopped 90k short with nothing in the log to say so.
            # Clamped here so the ladder is the same shape on any model, which
            # is what lets a small model stand in for a large one when the
            # question is about scheduling rather than about kernels.
            ceiling = max(1024, int(getattr(args, "max_model_len", 0) or 262144)
                          - long_decode - 64)
            rounds += [
                # Rung 1, out to the model's context limit -- prefill coverage.
                (2048, 1, long_decode), (4096, 1, long_decode),
                (8192, 1, long_decode), (16384, 1, long_decode),
                (65536, 1, long_decode), (131072, 1, long_decode),
                (196608, 1, long_decode), (258048, 1, long_decode),
                # Every length above is a multiple of the 16384-token prefill
                # chunk, or near enough that only 258048 leaves a remainder.
                # So the table has full chunks at every context and almost no
                # SHORT chunk at a deep one -- and a short chunk at a deep
                # context is a request's last chunk, which every request has.
                #
                # Counted: at 512 tokens the table holds three rows, all at a
                # context under 8192, while a one-client cc-traces rung puts
                # thirty past 131072. Per-feature coverage calls that covered,
                # because the table has short chunks and it has deep contexts;
                # it does not have the two together. The fit there is
                # unconstrained, and it shows -- the same rung's 512..4096
                # token steps came out 17% to 25% low, and they are 30% of its
                # prefill seconds against 1% of a sixteen-client rung's, which
                # is the whole of the -8%-to--1% spread across the rungs.
                #
                # A prompt of `chunks * 16384 + r` ends in a chunk of exactly
                # `r` at a context of `chunks * 16384`, so the remainder is
                # chosen by choosing the length. The scheduler then splits that
                # remainder again: `_finalize_prefill_chunk` shortens a chunk
                # to the previous state-checkpoint rung, so 512 goes out as
                # 496 and then 16. Both shapes are wanted -- every ragged
                # request ends that way -- and the 16-token tail is not cheap,
                # 543ms against 979ms for the 496 beside it, because it is
                # almost entirely KV read.
                #
                # Four depths. Measured: these twenty rungs take 72 minutes and
                # move held-out per-step MAPE over the four cc-traces rungs from
                # 10.2% to 8.6%, and the bias spread across them from 7.1 to
                # 5.9 points, by raising the history coefficient 24% -- without
                # short chunks at deep context the fit cannot separate reading
                # the KV from computing the attention pairs. Twenty more rungs
                # at the four depths in between were measured too and reached
                # 8.2% and 5.5 points for another 58 minutes, which is not
                # worth a fifth of the sweep. What is left is not this defect:
                # at the same step shape the one-client rung is 5.9% low and
                # the sixteen-client rung 0.6% low, so it is occupancy.
                (66048, 1, long_decode),    # 512 at 65536
                (66560, 1, long_decode),    # 1024 at 65536
                (67584, 1, long_decode),    # 2048 at 65536
                (69632, 1, long_decode),    # 4096 at 65536
                (73728, 1, long_decode),    # 8192 at 65536
                (131584, 1, long_decode),   # 512 at 131072
                (132096, 1, long_decode),   # 1024 at 131072
                (133120, 1, long_decode),   # 2048 at 131072
                (135168, 1, long_decode),   # 4096 at 131072
                (139264, 1, long_decode),   # 8192 at 131072
                (197120, 1, long_decode),   # 512 at 196608
                (197632, 1, long_decode),   # 1024 at 196608
                (198656, 1, long_decode),   # 2048 at 196608
                (200704, 1, long_decode),   # 4096 at 196608
                (204800, 1, long_decode),   # 8192 at 196608
                (246272, 1, long_decode),   # 512 at 245760
                (246784, 1, long_decode),   # 1024 at 245760
                (247808, 1, long_decode),   # 2048 at 245760
                (249856, 1, long_decode),   # 4096 at 245760
                (253952, 1, long_decode),   # 8192 at 245760
                # Rungs 8, 16 and 32 across the context range, not only at
                # its ends. Sampling a rung at a tiny context and an enormous
                # one bounds it without covering it: the fit is then a line
                # through two distant clusters, and a workload living between
                # them is being interpolated across a region nothing measured.
                # Measured, that is where every real decode step fell -- 117 of
                # 117 at rung 8, 512 of 512 at rung 16, 1642 of 1642 at rung 32
                # -- and the predictions came out 31 to 33% low. Rung 1 was the
                # only rung whose samples surrounded its workload and the only
                # one within a percent.
                (1024, 8, long_decode), (4096, 8, long_decode),
                (16384, 8, long_decode), (65536, 8, long_decode),
                (131072, 8, long_decode),
                (1024, 16, long_decode), (4096, 16, long_decode),
                (16384, 16, long_decode), (65536, 16, long_decode),
                (1024, 32, long_decode), (4096, 32, long_decode),
                (16384, 32, long_decode), (49152, 32, long_decode),
                # Rungs 2 and 4 at long context. They had ragged rounds and no
                # uniform ones, so their only long samples came from `_skewed`,
                # whose single long sequence tops out at 32768. That left rung 2
                # calibrated to a total context of 33856 and rung 4 to 57209 --
                # and a real 27B run asked them for 163676 and 299712, five
                # times outside the evidence, with the extrapolation warning
                # firing on every such step. Rung 8 reached 569320 against a
                # calibrated 524544, so it gets one more sample too.
                #
                # A small batch only reaches a large total context when each of
                # its few members is individually long, which is what a nearly
                # drained queue of long requests looks like -- exactly how a
                # long-prompt workload ends. It is a different gap from the
                # raggedness one closed earlier: that was the spread within a
                # batch, this is the product of a small batch and a long
                # history, and no amount of raggedness reaches it.
                (1024, 2, long_decode), (4096, 2, long_decode),
                (16384, 2, long_decode), (65536, 2, long_decode),
                (131072, 2, long_decode),
                (1024, 4, long_decode), (4096, 4, long_decode),
                (16384, 4, long_decode), (65536, 4, long_decode),
                (131072, 4, long_decode),
            ]
            # Ragged batches, so raggedness varies and can be fitted. Lengths
            # spread geometrically within one batch, which is what a real
            # workload looks like: requests arrive at different times and are
            # at different points in their generation, so a decode batch mixes
            # short histories with long ones.
            def _spread(rung, low, high):
                step = (high / low) ** (1.0 / max(rung - 1, 1))
                return tuple(int(low * step ** i) for i in range(rung))

            def _skewed(rung, short, long_):
                """One long sequence among short ones.

                A geometric spread cannot get very ragged: its mean rises with
                its maximum, so the ratio tops out near 3.5 whatever the range.
                A batch is at its most ragged when one sequence dominates, and
                then the ratio approaches the rung size. That is not a contrived
                case -- it is one long-running request among freshly arrived
                ones, and real steps reach 11.8 at rung 32, which nothing built
                from a spread can reach.
                """
                return tuple([short] * (rung - 1) + [long_])

            # Every rung, not just the large ones. Rung 4 is fully ragged in the
            # workload and was the one rung that got *worse* when the others
            # improved, because it had no ragged samples of its own.
            # Three degrees of raggedness, because how ragged a real batch is
            # depends on the workload and not on the engine. A 0.6B serving
            # short prompts runs 2.8 to 3.9 times ragged, since generation adds
            # a lot relative to a 2k context; a 27B serving 163k prompts runs
            # 1.1 to 1.4, since it adds very little relative to those. Sampling
            # only at 1.0 and at 7 to 21 -- which the first two patterns do --
            # leaves a hole exactly where the second workload lives, and the
            # padding coefficient is then fitted far from where it is used. On
            # the 27B that made rung 2 worse with the feature than without it.
            # Mildly ragged *at long context*. The rounds below are ragged
            # but short, and the uniform long-context rounds above are long but
            # perfectly uniform, so where a long-prompt workload actually lives
            # -- long histories, mildly ragged -- there was nothing. Measured on
            # the 27B: at the contexts its run uses, every one of the sweep's 64
            # rung-16 samples sat at raggedness exactly 1.00 while the run ran
            # at 1.18-1.32, and none of its rung-8 samples were in the run's
            # band either.
            #
            # Padding is identically zero at raggedness 1.00, so the padding
            # coefficient had no variance to be identified from in that region.
            # That is the same rank deficiency the term was introduced to fix,
            # surviving locally after being fixed globally -- a bounding box
            # containing the workload is not evidence near it. Rung 16 was
            # covered on context and on raggedness separately and still came out
            # 22.58% low, five times any other rung's error.
            # Two per rung, so the padding coefficient has a spread to be
            # fitted from there and not a single point. The first sits at about
            # 1.15 and the second above the run's band; the uniform rounds above
            # supply raggedness 1.00, so together they bracket it.
            for rung, low, high in ((2, 98304, 131072), (4, 98304, 131072),
                                    (8, 65536, 98304), (16, 49152, 65536),
                                    (2, 54026, 88064), (4, 53857, 77824),
                                    (8, 40206, 65536), (16, 26757, 53248)):
                rounds += [(_spread(rung, low, high), rung, long_decode)]

            for rung in (2, 4, 8, 16, 32):
                rounds += [
                    # Barely ragged: a batch of similar long histories.
                    (_spread(rung, 12288, 16384), rung, long_decode),
                    (_spread(rung, 512, 8192), rung, long_decode),
                    (_spread(rung, 1024, 32768), rung, long_decode),
                    (_skewed(rung, 512, 32768), rung, long_decode),
                ]
            rounds = [((tuple(min(v, ceiling) for v in length)
                        if isinstance(length, (list, tuple))
                        else min(length, ceiling)), count, decode)
                      for length, count, decode in rounds]
            # Clamping collapses distinct rounds into duplicates on a small
            # model; one of each is enough and the sweep runs every round twice
            # anyway.
            seen, unique = set(), []
            for entry in rounds:
                if entry not in seen:
                    seen.add(entry)
                    unique.append(entry)
            rounds = unique

            # Every round above gives each prompt a distinct opening, so no two
            # ever share a block. That is deliberate -- a sweep measuring
            # prefill must perform it -- but it puts a hard ceiling on the one
            # axis decode is fitted against. Decode is fitted per rung on
            # *total* context, the sum across the batch, and without sharing
            # that sum cannot exceed the KV pool, because every token in it is
            # a token stored. Measured on the 27B at TP1: the pool is 76596
            # blocks of 16, so 1225536 tokens, and the sweep's rung-16 samples
            # already reach 1049600 of it (86%) and rung 32 reaches 1230400
            # (100%). There is no room left to extend them.
            #
            # A real agentic workload is not bounded that way. It re-sends its
            # conversation, 96.3% of its blocks are reusable, and a shared block
            # is stored once but counted once *per request* in total context. So
            # a cc-traces run reached 1.48M at rung 16, 2.85M at rung 32 and
            # 3.58M at rung 48 -- 1.2x, 2.3x and 2.9x the entire pool -- against
            # a table whose rung 48 stopped at 10752. The prediction there was an
            # extrapolation by a factor of 333, and the run's time per output
            # token came out 73.5% high.
            #
            # These rounds reach the same place the same way. Each is `count`
            # prompts of `length` tokens sharing a `shared`-token prefix, so it
            # costs `shared + count * (length - shared)` blocks and presents
            # `count * length` of total context. They are also the only samples
            # in the table that are cache *hits*: until now not one was, and the
            # oracle sees a hit only as reduced num_scheduled_tokens against
            # unchanged context_lens -- indistinguishable from a chunked-prefill
            # middle chunk, which costs differently because it writes KV.
            #
            # Several per rung, spanning the range rather than bracketing it. A
            # rung sampled only at its ends is a line through two distant
            # clusters, which is how rung 16 came out 22.58% low once before.
            #
            # Three heights per rung, not one. The table stopped at 131072 --
            # half the 262144 window -- so every rung's total context topped out
            # at `count * 131136` and a cc-traces rung at 250k sat outside it by
            # construction. #78 measured that: rung 16 EXTRAPOLATED on 415 of
            # 1479 steps and the coverage gate failed. Extending it is not a
            # memory question. These rounds are cache hits, so a round costs
            # `shared + count * (length - shared)` and a full-window one costs
            # what a half-window one does -- 376k-400k tokens against a
            # 1,225,536-token pool, the same size as the 131072 rows below.
            #
            # 196608 is in the list because 131072 and the ceiling alone are two
            # distant clusters, and a rung fitted through two clusters is the
            # 22.58% error this sweep already made once.
            shared_rounds = [
                # (length, count, shared) -- physical cost in the comment.
                (98304, 8, 65536),      # logical 786k, physical 327k
                (131072, 8, 98304),     # logical 1.05M, physical 360k
                (65536, 16, 32768),     # logical 1.05M, physical 557k
                (92672, 16, 65536),     # logical 1.48M, physical 500k
                (131072, 16, 114688),   # logical 2.10M, physical 377k
                (49152, 32, 24576),     # logical 1.57M, physical 811k
                (89088, 32, 81920),     # logical 2.85M, physical 311k
                (131072, 32, 122880),   # logical 4.19M, physical 384k
                (40960, 48, 24576),     # logical 1.97M, physical 810k
                (74752, 48, 70656),     # logical 3.59M, physical 267k
                (131072, 48, 126976),   # logical 6.29M, physical 324k
                (32768, 64, 20480),     # logical 2.10M, physical 806k
                (98304, 64, 94208),     # logical 6.29M, physical 356k
                # Three quarters of the window.
                (196608, 2, 131072),    # logical 393k, physical 262k
                (196608, 4, 163840),    # logical 786k, physical 295k
                (196608, 8, 180224),    # logical 1.57M, physical 311k
                (196608, 16, 188416),   # logical 3.15M, physical 319k
                (196608, 32, 192512),   # logical 6.29M, physical 324k
                (196608, 48, 194560),   # logical 9.44M, physical 293k
                (196608, 64, 195584),   # logical 12.6M, physical 261k
                # The window itself. Clamped to `ceiling` below, which is
                # max_model_len less the decode length and a block -- ask for
                # the full 262144 and the generated tokens run past the window,
                # which is how the first version of these rounds stopped 90k
                # short with nothing in the log to say so.
                #
                # Rungs 2 and 4 appear here and not above because they had no
                # shared rounds at all: their whole column came from the
                # distinct-prefix ladder, which cannot exceed the pool, so they
                # stopped at 131136 per sequence. A c1 or c4 cc-traces rung runs
                # exactly there.
                (262144, 2, 245760),    # logical 524k, physical 278k
                (262144, 4, 253952),    # logical 1.05M, physical 286k
                (262144, 8, 245760),    # logical 2.10M, physical 376k
                (262144, 16, 253952),   # logical 4.19M, physical 383k
                (262144, 32, 258048),   # logical 8.38M, physical 385k
                (262144, 48, 259072),   # logical 12.6M, physical 400k
                (262144, 64, 260096),   # logical 16.8M, physical 383k
            ]
            # Same clamp as above, and the shared prefix with it: a prefix
            # longer than the prompt would silently become no sharing at all.
            rounds = [(length, count, decode, 0)
                      for length, count, decode in rounds]
            for length, count, shared in shared_rounds:
                length = min(length, ceiling)
                rounds.append((length, count, long_decode,
                               min(shared, max(0, length - 64))))
            # Deduped after clamping, not before. On a model whose window is
            # smaller than these lengths the three heights collapse onto the
            # ceiling and become the same round three times -- which is exactly
            # the case a small model standing in for a large one runs.
            seen, unique = set(), []
            for entry in rounds:
                if entry not in seen:
                    seen.add(entry)
                    unique.append(entry)
            rounds = unique
        # Twice through, because Triton autotunes per shape rather than once per
        # process: the first visit to a shape pays a benchmarking cost that
        # steady-state serving never pays again. The second visit is the one
        # worth fitting, and having both lets the outlier rejection see the
        # difference rather than guess at it.
        # `--sweep` without `--sweep-long` never reaches the block above, so
        # normalise here rather than there. Idempotent on purpose.
        rounds = [r if len(r) == 4 else (r[0], r[1], r[2], 0) for r in rounds]
        # Indexed before any slicing, because the index is the prompt seed: a
        # segment run on its own has to build the same prompts the whole sweep
        # would have built at that position, or two halves of one table would
        # disagree about what they measured.
        sequence = list(enumerate(rounds + rounds))
        if args.sweep_round:
            sequence = []
            for spec in args.sweep_round:
                parts = [int(v) for v in spec.split(",")]
                length, count, shared = parts[0], parts[1], parts[2]
                decode = parts[3] if len(parts) > 3 else args.sweep_long_decode
                seed = parts[4] if len(parts) > 4 else 0
                sequence.append((seed, (length, count, decode, shared)))
        elif args.sweep_rounds:
            start, _, stop = args.sweep_rounds.partition(":")
            sequence = sequence[int(start or 0):
                                int(stop) if stop else len(sequence)]
        print(f"sweep: {len(sequence)} of {2 * len(rounds)} rounds, "
              f"indices {[i for i, _ in sequence][:4]}"
              f"{'...' if len(sequence) > 4 else ''}", flush=True)
        for round_index, (length, count, decode, shared) in sequence:
            # A round is either `count` prompts of one length, or an explicit
            # list of lengths. The second exists because every uniform round
            # leaves the batch's *raggedness* at exactly one, and a fit cannot
            # find a coefficient for a quantity that never varies. Measured:
            # real decode batches run 2.8 to 3.9 times ragged (longest sequence
            # over mean) while every sweep batch was 1.0, and the cost model,
            # which sums context across the batch, came out 29 to 32% low at
            # rungs 8 and 16 as a result. Widening the context range did not
            # help, because context was not the missing dimension.
            lengths = (list(length) if isinstance(length, (list, tuple))
                       else [length] * count)
            if shared:
                prompts_here = shared_prefix_prompts(lengths, shared,
                                                     round_index)
            else:
                # Exactly `length` tokens each, and a distinct opening per
                # prompt so no two share prefix-cache blocks. This built its
                # own prompts by hand until the long rounds arrived, and a
                # hand-built word-per-token prompt is five to eight times the
                # length it claims -- which the short ladder survived and the
                # long one did not, being silently truncated at max_model_len.
                prompts_here = [prompt_of_tokens(n, round_index * 10007 + i)
                                for i, n in enumerate(lengths)]
            llm.generate(
                prompts_here,
                SamplingParams(temperature=0.0, max_tokens=decode),
            )
        print("sweep complete")
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"wall": 0.0, "requests": []}, fh)
        return 0

    # A profile of this same workload is what says whether a priced kernel costs
    # what it costs in a step. The two have to be the same workload or the
    # comparison is between different shapes -- so it is a flag here rather than
    # a second script with its own prompts. One warm generate first, because the
    # trace should hold steady-state work and not Triton autotuning its way
    # through every shape.
    profiling = bool(getattr(args, "torch_profiler_dir", None))
    if profiling:
        llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=4))
        llm.start_profile()

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    wall = time.perf_counter() - start

    if profiling:
        llm.stop_profile()
        print(f"profile written to {args.torch_profiler_dir}")

    requests = [
        {
            "ttft": out.get("ttft", 0.0),
            "tpot": out.get("tpot", 0.0),
            "latency": out.get("latency", 0.0),
            "num_tokens_input": out.get("num_tokens_input", 0),
            "num_tokens_output": out.get("num_tokens_output", 0),
        }
        for out in outputs
    ]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"wall": wall, "requests": requests}, fh, indent=2)
    print(f"wrote {len(requests)} requests to {args.out} (wall {wall:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
