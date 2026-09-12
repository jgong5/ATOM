# The source-only serving configuration, and what it still refuses

What a served 27B run passes to `source_cost_oracle` so that its cost oracle is
the frozen source composition, assembled from artifacts that already exist. It
is written down here because the composition was reachable from
`predict_step.py` and from a hand-written script, and a served run names a
factory and a list of `KEY=VALUE` strings instead.

Nothing here is measured at the target deployment. Two different things are
being distinguished, and the difference is the whole point of the source-only
rule:

- **Not allowed, and absent:** any timing or memory reading taken from the
  engine whose behaviour is being predicted. No target step latency, no target
  throughput, no target memory high-water mark enters a price, a template or
  the region model.
- **Allowed, and used:** standalone measurements of a *primitive* at the width
  being predicted, taken outside the target run — a two-rank all-reduce probe,
  a gather probe, a GEMM microbench. These are measurements of hardware, not
  of the deployment, and the TP2 and TP4 collective prices below are exactly
  that.

So "the prices are the source deployment's own" is true of the body and head
GEMM families, which are measured at TP1 and transferred; the collectives at
TP2 and TP4 are standalone primitive measurements at their own width. The
graphs are traced on `meta`; the region model is calibrated at TP1 and applied
unchanged at TP2 and TP4.

## The options

Every width shares these:

    --compass-oracle atom.compass.runtime.source_oracle.source_cost_oracle
    --compass-oracle-option model=Qwen/Qwen3.8-27B
    --compass-oracle-option device=meta
    --compass-oracle-option replay_target=$POC/g5_27b/target.json
    --compass-oracle-option block_size=16
    --compass-oracle-option max_model_len=262144
    --compass-oracle-option position_rows=3
    --compass-oracle-option cudagraph_mode=full
    --compass-oracle-option head=1
    --compass-oracle-option regions=source-27b-tp1-prefill-interp
    --compass-oracle-option require_complete=1
    --compass-oracle-option allocation=native
    --compass-oracle-option derive=1
    --compass-oracle-option interpolate=1

and adds its own prices and templates. At TP1 (`$SRC1` is `g4/src1`):

    --compass-oracle-option tp=1
    --compass-oracle-option price=$SRC1/p27bdec32.tp1.r0.json:$SRC1/b27dec32.tp1.r0.json:unregistered,$SRC1/p27hdec32.tp1.r0.json:$SRC1/h27dec32.tp1.r0.json:unregistered
    --compass-oracle-option template=$SRC1/b27dec32.tp1.r0.json
    --compass-oracle-option head_template=$SRC1/h27dec32.tp1.r0.json


`interpolate=1` turns the family price provider on at the density the provider
itself declares -- it is the word `true`, not a ratio. A number there is read
as a ratio and means something else: the widest ratio between two adjacent
measured row counts the evidence supports interpolating across. `1` used to be
taken literally as that ratio, which no two distinct row counts can meet, and
run 8 refused 2118 of 2443 operators for it.

`regions=source-27b-tp1-prefill-interp` is the successor to
`source-27b-tp1-conc-v2`. It carries the measured one-sequence prefill cells
byte for byte and interpolates only between adjacent measurements of the same
`(sequences, produces_output)` group, refusing outside their span -- run 7
died because an exact-cell lookup cannot answer a final chunk, whose token
count is `prompt mod chunk_budget` and so arbitrary.

The TP1 `price=` line above is the two decode-32 seed pairs, which is where
this set started. The list the registry passes today is longer: it adds the
prefill cells, the long-context and mixed steps, the one-sequence row ladder
and the cached-MHA calibration inputs. `cc_traces_registry.py` is the
authority for its contents; reproducing all of it here would be a third copy
to drift.

At TP2 and TP4 (`$D` is the staged directory below):

    --compass-oracle-option tp=$TP
    --compass-oracle-option price=$D/p27bdec32.json:$D/b27dec32.json:unregistered,$D/p27hdec32.json:$D/h27dec32.json:unregistered,$D/ar_capture.json,$D/ar_plain.json,$D/ag_prices.json
    --compass-oracle-option template=$D/b27dec32.json
    --compass-oracle-option head_template=$D/h27dec32.json

Both all-reduce price lists are loaded on purpose. They hold the same signature
under different registration regimes and the graph selects between them, so
which one answers does not depend on load order.

`allocation=native` and `carry_allocation=1` are mutually exclusive, and only
the first is admissible for acceptance. See refusal 3.

## Staging, and why it is needed

A served run passes one option set to every rank; the rank reaches its own
files through `resolve_rank_path`, which appends the rank's coordinates —
`p27bdec32.json` at `{"tp": 1}` is `p27bdec32.tp1.json`. The pricing artifacts
are named `p27bdec32.tp<width>.r<rank>.json`, which that convention never
produces. So a directory of links in the convention stands in front of them:

    for r in $(seq 0 $((TP-1))); do
      for stem in p27bdec32 p27hdec32 b27dec32 h27dec32; do
        ln -s <source>/$stem.tp$TP.r$r.json $D/$stem.tp$r.json
      done
    done

The shared lists — `ar_capture.json`, `ar_plain.json`, `ag_prices.json` — are
linked unsuffixed and resolve by the documented fallback, recorded in
`SourceComposition.rank_artifacts` as not this rank's own.

## What it answers today

**These are diagnostic step estimates, not cc-traces acceptance cells.** They
say that the composition assembles, resolves per rank, and prices a single
production-shaped decode step with nothing refused. They are not a registered
cc-traces replay, they are not compared against a measured target run, and no
acceptance gate is evaluated on them. The acceptance evidence is the e2e
registered cc-traces TP1/2/4 short and long runs, and none of it is here.

Built per rank with `derive=0`, so that what it answers is exactly what the
artifacts cover, on the production decode step (32 requests of one token at
context 1151, capture bucket 32):

| width | rank | step | body | head |
|-------|------|------|------|------|
| 1 | 0 | 34.767 ms | 33.378 ms, 2615 launches, 2439 operators | 1.156 ms, 1 launch, 1 operator |
| 2 | 0 | 20.970 ms | 19.929 ms, 2573 launches, 2568 operators | 0.780 ms, 2 launches, 2 operators |
| 2 | 1 | 20.814 ms | 19.773 ms, 2573 launches, 2568 operators | 0.779 ms, 2 launches, 2 operators |
| 4 | 0 | 13.510 ms | 12.814 ms, 2573 launches, 2568 operators | 0.435 ms, 2 launches, 2 operators |
| 4 | 1 | 16.158 ms | 15.460 ms, 2573 launches, 2568 operators | 0.436 ms, 2 launches, 2 operators |
| 4 | 2 | 13.459 ms | 12.764 ms, 2573 launches, 2568 operators | 0.433 ms, 2 launches, 2 operators |
| 4 | 3 | 13.452 ms | 12.753 ms, 2573 launches, 2568 operators | 0.436 ms, 2 launches, 2 operators |

Every cell is `complete` and `complete_measured`: no interpolation, no
zero-work, nothing refused. The body's collectives are priced from the probe
(129 all-reduce operators at TP2) and the head's all-gather from the gather
probe, which is the composition `tpN_frozen.py` assembles by hand.

Rank 1 at TP4 is 20% above its peers and the other three agree to half a
percent. It is not averaged away here; a rank aggregation rule that hides it
would hide the thing worth looking at.

## What it refuses

1. **The head is priced at 32 rows and nowhere else.** The head GEMM signature
   is `32,5120;...` at every width; rows 1 and 20 are unpriced. A decode step
   with any other running-request count is refused for the head, and
   `interpolate` cannot answer it either — one measured point fits nothing.
   This is the refusal a real cc-traces workload meets first.
2. **Only the decode-32 structure is seeded.** Any other structure — every
   prefill, every chunked step, every other bucket — needs derivation, which
   needs the model. Derivation is device-free and costs about 10.5 s per
   structure, but it produces rank 0's shard whatever rank asks, so at TP>1 the
   other ranks are served through the representative fallback.
3. **A step nobody offers an allocation for is refused.** The body template
   carries `slot_mapping`, `block_tables` and the two `non_spec_state_indices`
   tensors, and binding will not reuse a capture's copy of them. With
   `allocation=native` the oracle takes the CPU scheduler's own assignment for
   the step being priced — `ScheduledBatch.block_tables`,
   `state_slots_committed` and the `state_rows` that say which batch row each
   of those slots belongs to — and re-encodes it through `BatchSpec`, the same
   code that encodes a capture. A step reached with nothing offered is refused
   by name rather than priced against another step's blocks. Three further
   things it will not guess, each a refusal:
   - **the batch kind**, which comes from `total_seqs_num_prefill` and not from
     the batch's prefill *token* count. A batch holding both prefill and decode
     rows is refused: `BatchSpec` carries one kind for the batch, and the
     attention backend sends the whole batch down `prepare_prefill` while
     preparing metadata for the leading prefill rows only. This is not a gap
     cc-traces needs closed: with TBO off the scheduler cannot build such a
     batch. Its prefill branch returns unconditionally (`scheduler.py:1991`),
     every real `ScheduledBatch(` construction site is of one kind,
     `model_runner.py:481-483` still carries `# TODO: remove this when we
     support mixed prefill and decode in one batch`, and TBO is a runner-side
     split of an already-formed batch (`enable_tbo`/`enable_tbo_decode`
     default False, and `scheduler.py` does not mention TBO at all).
     `tests/compass/test_mixed_batch_reachability.py` drives the real
     scheduler and shows a prompt arriving mid-decode starting its own prefill
     batch, and every batch of a staggered workload being of one kind. No
     per-request kind is being added to `BatchSpec` for a batch the engine
     does not emit; if that TODO is ever done, this refusal is where to look.
   - **which row a state slot belongs to.** A row with no slot is refused, not
     filled with its batch index, which is only what a fresh pool happens to
     hand out.
   - **the padded tail.** When the active request count is below the capture
     bucket, the scheduler's entries are written over the template's head, the
     capture's tail is kept, and the count is recorded per field in
     `provenance.binding.allocation_padding`. Overwriting the tail would invent
     an assignment for rows that are not running.

   `carry_allocation=1` — reuse the template's assignment and declare it
   unmeasured — remains available for diagnostics and is **inadmissible for
   acceptance**.
4. **The capture bucket has to be declared at derivation.** `g4/dec32`'s TP2
   and TP4 graphs record `capture_bucket: null`, so their shape is the
   uncaptured structure and the region model refuses the step as eager. They
   were re-derived with `--capture-bucket 32`. The body graphs are
   operator-identical to the originals. The head graphs differ by one
   `aten::empty.memory_format`, and the reason is the producer's
   fresh-collective recording contract rather than the operator's 0.09 µs:
   simulated TP has no communicator, so it reimplements `all_gather` locally as
   *allocate the full buffer, copy this rank's shard in, movedim, reshape*, and
   `record_collectives` replaces that reimplementation with a synthesized
   `aiter::all_gather_unreg` instead of recording it. Since `6b4bfe8b` the
   synthesized wrapper allocates its declared output through `fresh_shape`,
   under `_disable_current_modes()`, precisely "so the stand-in does not appear
   in the graph as an [allocation] a captured graph has no counterpart for".
   The extra `aten::empty.memory_format` in the older head graphs is that
   stand-in's own buffer, which no rank executes; the real allocation
   production performs is inside the gathered call, and it is already inside
   the two-rank probe measurement that prices `all_gather_unreg`. Recording it
   separately would count it twice. Note that the difference is attributable to
   the code change, not to `--capture-bucket`: the two derivations ran under
   different `graph_diff.py`/producer hashes (see Provenance).
5. **The region model is held out at TP2 and TP4.** `source-27b-tp1-conc-v2`
   is calibrated at TP1 and applied unchanged. Its prepare term was defined as
   the capture's own remainder, so at TP1 it closes by construction and proves
   nothing; at the wider widths it is a falsifiable claim nobody has yet
   falsified.

## Which artifacts a served run should name

The bucket-declared TP2/TP4 graphs are **new source-derived candidate
artifacts**, not replacements of the old ones. `g4/dec32` and every frozen
result computed from it stay exactly as they are and keep their labels; nothing
is retroactively relabelled. What follows is the candidate set, its
configuration, and its hashes.

Produced by `agent_scratch/serving/derive_b32.sh`, which is
`derive_dec32_tp24.sh` plus `--capture-bucket 32`, run device-free under
`--device meta` in `xiaobizh_n18_cpu`, ~10 s per graph:

    dec32b32/  (sha256, first 16)
      ca60fac1fc1c1670  b27dec32.tp2.r0.json
      54230ed28c664311  b27dec32.tp2.r1.json
      f83dce526f979e5b  b27dec32.tp4.r0.json
      fff63aceb6d21a3e  b27dec32.tp4.r1.json
      c80b3c700928a01a  b27dec32.tp4.r2.json
      c1bb1e76a3120f3d  b27dec32.tp4.r3.json
      548078cbe7396bda  h27dec32.tp2.r0.json
      f2418f548800133b  h27dec32.tp2.r1.json
      19458c6f7da0e2a9  h27dec32.tp4.r0.json
      9d2ed08c7e132e95  h27dec32.tp4.r1.json
      a1e4b934f4b91cbd  h27dec32.tp4.r2.json
      f985c61ef4b1ad20  h27dec32.tp4.r3.json

The case for naming them: the originals record no capture bucket, so the region
model refuses them and they cannot serve at all. The body is operator-identical
between the two, so the substitution changes no body price; the head loses the
stand-in's `aten::empty` and with it 0.09 µs, recorded above rather than
absorbed. TP1's `g4/src1` artifacts already declare the bucket and are used
unchanged.

## Provenance

Step numbers above are from `agent_scratch/serving/check_serving_config.py` run
in `xiaobizh_n18_cpu` — device-free — against the committed snapshot of
`e946c6ce`, verified file-by-file against `git show` before the run.

The native-allocation behaviour in refusal 3 is from
`agent_scratch/check_native_allocation.py`, run the same way against `b6280165`
over all seven rank compositions: each refuses with nothing offered, reproduces
the capture's `slot_mapping`, `block_tables` and state indices when the
capture's own assignment is offered, and follows the blocks when they are
rotated by one request.

The graph re-derivations hash `graph_diff.py`, `tracer.py` and `graph.py` into
`agent_scratch/serving/dec32b32/derive_b32.sha256`:

    84d077369e21aee5…  scripts/compass/graph_diff.py
    c5dc0b71879aea9f…  atom/compass/runtime/tracer.py
    c806740b786e5bd7…  atom/compass/core/graph.py

The original `g4/dec32` derivation hashed a different file set
(`agent_scratch/g4/dec32/derive_dec32.sha256`: `graph_diff.py`,
`batch_spec.py`, `graph.py`, `library.py`) and a different `graph_diff.py`
(`62a89c02…`), which is what makes the head-graph difference in refusal 4 a
producer change rather than a bucket effect.
