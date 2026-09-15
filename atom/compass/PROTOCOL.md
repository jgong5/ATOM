# Prepare-then-measure: the protocol the acceptance matrix is run under

Registered before the runs it governs, and stamped into every cell those runs
produce. A cell whose stamp does not match the registration is not an
acceptance run — see `scripts/compass/protocol.py`.

The reason it exists. A served engine's first forward costs about 6.7 s more
than its second, and a process that loads a compile cache another process has
just written can spend 45 s on it. Those are real costs and they are reported,
but predicting each of them with one scalar is not what this PoC is for: the
question is whether limited measurements plus ATOM's own serving logic pick the
right configuration for a *running* deployment. So the acceptance matrix
measures a warmed server, the startup costs are measured separately as their
own scenario, and neither is allowed to stand in for the other.

## 1. Scope

Applies to every G1, G2 and G4 acceptance cell.

Does not apply to the cold-start scenario, which has its own probe
(`agent_scratch/poc/first_use_probe.sh`) and its own artifact
(`first_use.json`, reported per regime). Cold-start numbers are never averaged
into a warmed cell and a warmed cell is never presented as a cold-start result.

## 2. Cache state, established before the cell starts

The compile cache for the cell's (model, tp, max-model-len, dtype) must already
exist and must already have been loaded by at least one earlier process.

Each cell reads its own server log and records the regime it was actually in:

| regime | log line | first prefill |
| --- | --- | --- |
| `compiled` | `Compiling a graph for dynamic shape` | ~6.8–7.0 s |
| `cache-first` | `Directly load the compiled graph`, right after a `compiled` process | 45–48 s |
| `cache-steady` | `Directly load the compiled graph`, otherwise | ~6.6–7.1 s |

A cell whose server was in `compiled` or `cache-first` is recorded and is not
an acceptance run. This is a property the cell checks, not one the operator
asserts.

## 3. Preparation, on the real server only

The real server is warmed. After it reports healthy, and before anything is
measured, the cell sends preparation requests of the measured workload's own
shape, unpaced and concurrent, and waits for every one of them to return. The
preparation prompts are drawn from an index range disjoint from the measured
workload's, so no measured prompt has been seen before it is measured. Cell
servers run with `--no-enable_prefix_caching`, so that disjointness is
insurance rather than the load-bearing part.

The predictor is not warmed, and warming it is refused rather than merely
skipped. A predictor has no compilation to warm physically, so an executed
preparation buys nothing — and it is not free:

> A declared arrival is an offset from the engine's epoch. The frontend
> process's virtual clock is frozen at that epoch by design; only the engine
> core advances time (`atom/model_engine/llm_engine.py::_install_compass_clock`).
> An executed preparation therefore moves the core forward while leaving the
> origin that arrivals are stamped against exactly where it was, and its whole
> duration lands inside every measured request's TTFT and latency.

That is not a hypothesis. On the TP=1 short cell, 14.4 s of modelled
preparation produced a 42.0 s modelled window against 28.5 s real and a TTFT
median 71 % high, while TPOT was within 4.3 % and the modelled measured prefill
was slightly *cheaper* than the real one. The error was the origin, not the
model.

So the modelled side starts as a fresh empty run with `warmup_seconds=0`,
standing for the captured warm target state. The cell records this as
`modelled warm state`, in those words. It is not recorded as a warmup that was
executed, because none was.

If a prediction process is ever reused across runs rather than started fresh,
the run boundary must be an explicit synchronized run origin owned by the
engine core. Resetting only the frontend, or rewinding the clock, is not a
permitted substitute.

## 4. Drain, and what counts as proof of it

The measured phase on the real server begins only when all three hold:

1. every preparation request returned a response;
2. `GET /compass/requests` — which drains as it reads — returned exactly the
   preparation requests, so the engine's record store is empty at the
   boundary and no preparation row can reach the measured result;
3. the boundary instant is recorded, so a step in the step table can be
   attributed to preparation or to measurement after the fact.

The drained preparation rows are written to the cell as evidence
(`*.prepare.json`). They are not deleted; they are labelled.

The modelled side has nothing to drain: its run contains only measured
requests.

## 5. Virtual time

The modelled run's epoch is its run origin, because nothing precedes it. A
declared arrival of `t` therefore means `t`, and the measured window starts
where the workload says it does.

Nothing about the measured window is differenced against a preparation
instant. Every reported metric is a within-request difference (`first - arrive`,
`finish - first`) or a difference against the measured window's own first
arrival.

## 6. First use is not charged in a warmed cell

A warmed cell passes no `warmup_seconds`. On the real side the first-use cost
is paid during preparation and lies outside the measured window; on the
modelled side the state being represented is one in which it has already been
paid. Charging a constant into a window that represents warm steady state would
add a moving part with no correspondence on the other side.

The cell asserts the option is absent rather than trusting that it was omitted.

The first-use constant remains a measured cost component, reported in G5c with
the rest of the preparation and calibration cost, and amortised there. It is
never netted out of a serving result.

## 7. Enforcement

Two independent layers, both public and both tested, so a cell cannot leak
preparation into a measurement by operator error:

1. `scripts/compass/replay.py` refuses `--prepare` when
   `GET /compass/provenance` reports a virtual clock in `predict` mode. It
   exits 3 and writes no result.
2. `scripts/compass/compare.py::check_run` fails any saved run — whoever
   produced it, and however long ago — whose earliest measured arrival is
   stamped before that run's own `prepare.boundary_engine_time`.

The regression is `tests/compass/test_run_validity.py::TestPreparationMustPrecedeTheMeasurement`.

## 8. What is measured, unchanged

Workload and configuration are exactly those already registered, and this
protocol does not alter them:

- Qwen3.8-27B, MI308X-class single node, TP ∈ {1, 2, 4};
- short: 64 requests, 1024 input tokens, 128 output tokens, all arriving at 0;
- long: `agent_scratch/cc_pilot.jsonl`, 20 requests, input 640–119,360 tokens,
  arrivals over 244.5 s;
- one calibration table per (tp, workload class), reused across the cells of
  that pair.

## 9. Tolerances, unchanged

Throughput ≤ 10 %; TPOT/ITL ≤ 10 %; TTFT ≤ 15 %; non-KV memory terms ≤ 10 %;
KV block count ≤ 5 %; Spearman ρ ≥ 0.90 within comparable groups; the
deliberately infeasible configuration rejected for the right reason; ≥ 5×
replay speedup with GPU-free replay after capture.

Warming the server does not relax any of these. It fixes which state they are
claimed about.

## 10. Evidence each cell keeps

Server logs for both sides (the regime line is read from them); the drained
real preparation records; the drain assertion output; both step tables, the
real one with the boundary instant recorded so preparation steps stay
identifiable and the modelled one containing measured steps only; the
`modelled warm state` line; both result JSONs; the isolation audit; and the
digest of this file.

## 11. Immutability

`scripts/compass/protocol.py register` writes `atom/compass/protocol.lock.json`
with this file's SHA-256. `verify` recomputes it. Every cell verifies before it
runs and stamps the digest into its own directory.

Changing this file invalidates the registration. The correct response is a new
registration and a re-run of the affected cells; the superseded lock is kept in
`superseded` inside the new one, so the history of what was claimed under which
protocol stays readable. Editing the file and re-registering without re-running
is not a way to make an existing result compliant.

## 12. What this protocol does not claim

It does not make cold start predictable. The `cache-first` regime — 45–48 s,
about six times the constant, seen once at TP=2 and once at TP=4 and never
traced beyond the log line that identifies it — is still unexplained. It is
reported at full size, in its own row, and it is not represented in any warmed
result.

It does not claim the modelled side reproduces warming. It claims the modelled
side represents the state warming leaves the real server in. What that
representation costs — preparation, calibration, startup — is reported in G5c,
separately and with its amortisation stated.

It does not make already-collected cold runs into passes. Runs made before this
registration stay labelled as they were, and the cells measured under the
previous registration (which prepared both sides) are superseded by it, not
relabelled.
