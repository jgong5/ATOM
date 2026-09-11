# The source-only serving configuration, and what it still refuses

What a served 27B run passes to `source_cost_oracle` so that its cost oracle is
the frozen source composition, assembled from artifacts that already exist. It
is written down here because the composition was reachable from
`predict_step.py` and from a hand-written script, and a served run names a
factory and a list of `KEY=VALUE` strings instead.

Nothing here is measured at the target. The prices are the source deployment's
own; the graphs are traced on `meta`; the region model is calibrated at TP1 and
applied unchanged at TP2 and TP4.

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
    --compass-oracle-option regions=source-27b-tp1-conc-v2
    --compass-oracle-option require_complete=1
    --compass-oracle-option carry_allocation=1
    --compass-oracle-option derive=1

and adds its own prices and templates. At TP1 (`$SRC1` is `g4/src1`):

    --compass-oracle-option tp=1
    --compass-oracle-option price=$SRC1/p27bdec32.tp1.r0.json:$SRC1/b27dec32.tp1.r0.json:unregistered,$SRC1/p27hdec32.tp1.r0.json:$SRC1/h27dec32.tp1.r0.json:unregistered
    --compass-oracle-option template=$SRC1/b27dec32.tp1.r0.json
    --compass-oracle-option head_template=$SRC1/h27dec32.tp1.r0.json

At TP2 and TP4 (`$D` is the staged directory below):

    --compass-oracle-option tp=$TP
    --compass-oracle-option price=$D/p27bdec32.json:$D/b27dec32.json:unregistered,$D/p27hdec32.json:$D/h27dec32.json:unregistered,$D/ar_capture.json,$D/ar_plain.json,$D/ag_prices.json
    --compass-oracle-option template=$D/b27dec32.json
    --compass-oracle-option head_template=$D/h27dec32.json

Both all-reduce price lists are loaded on purpose. They hold the same signature
under different registration regimes and the graph selects between them, so
which one answers does not depend on load order.

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
3. **`carry_allocation=1` is required, and it is an unmeasured assumption.**
   The body template carries `slot_mapping`, `block_tables` and the two
   `non_spec_state_indices` tensors; binding refuses them without an allocation
   source, and the factory cannot name the engine's own allocator. The
   composition therefore reuses the template's assignment and says so.
4. **The capture bucket has to be declared at derivation.** `g4/dec32`'s TP2
   and TP4 graphs record `capture_bucket: null`, so their shape is the
   uncaptured structure and the region model refuses the step as eager. They
   were re-derived with `--capture-bucket 32`; the body graphs are
   operator-identical to the originals, and the head graphs differ by one
   `aten::empty.memory_format` (0.09 µs) that the bucketed derivation does not
   record.
5. **The region model is held out at TP2 and TP4.** `source-27b-tp1-conc-v2`
   is calibrated at TP1 and applied unchanged. Its prepare term was defined as
   the capture's own remainder, so at TP1 it closes by construction and proves
   nothing; at the wider widths it is a falsifiable claim nobody has yet
   falsified.

## Which artifacts a served run should name

For TP2 and TP4 the re-derived bucketed graphs supersede `g4/dec32`: the
originals record no capture bucket and the region model refuses them, so they
cannot serve at all. The body is operator-identical between the two, so the
substitution changes no body price; the head loses one `aten::empty` and with
it 0.09 µs, which is recorded rather than absorbed. TP1's `g4/src1` artifacts
already declare the bucket and are used unchanged.

## Provenance

Numbers above are from `agent_scratch/serving/check_serving_config.py` run in
`xiaobizh_n18_cpu` — device-free — against the committed snapshot of
`e946c6ce`, verified file-by-file against `git show` before the run. The graph
re-derivations hash `graph_diff.py`, `tracer.py` and `graph.py` into
`agent_scratch/serving/derive_b32.sha256`.
