# Defects in ATOM found while building Compass

Bugs in the **engine and the image it ships in**, not in Compass. Collected here because a simulator is an
unusually harsh reader of an engine: it runs the runner in configurations nobody
runs by hand, on a clock that does not tick the way the code assumes, and it
compares the engine against itself.

Each entry says what is wrong, how it was found, what it costs, and whether it
has been fixed. Nothing here is speculative — every one was reproduced.

Companion to [DESIGN_NOTES.md](DESIGN_NOTES.md), which records the design of
Compass itself and the much longer list of defects that were Compass's own.

---

## Open

### 1. `--help` crashes on every entrypoint that uses `EngineArgs` — *fixed*

`atom/model_engine/arg_utils.py:503`. The help text for
`--state-checkpoint-demand` contains bare percent signs:

    "demand is 47% of all checkpoint writes but reads back 2.8% of "
    "the time, against 85.2% for an anchor, ..."

`argparse` expands help through `self._get_help_string(action) % params`, so
`"% o"` in `2.8% of` is read as the `%o` conversion:

    TypeError: %o format: an integer is required, not dict

Reproduce with any of them — the OpenAI server, the benchmarks, the examples:

    python -m atom.entrypoints.openai_server --help

**Cost:** nobody can read the CLI help for any ATOM entrypoint. It is not a
Compass problem; Compass only noticed because its scripts build on `EngineArgs`.

**Fix:** applied. Escaped as `%%`, which argparse renders back to a single `%`.
No behaviour change beyond `--help` working again, on `openai_server`,
`benchmark_serving`, the examples and the Compass scripts alike. Verified by
running `--help` on the server and on `scripts/compass/run.py` and checking the
text still reads `47%`.

### 2. CUDA-graph tail padding zeroes to the largest rung, not the smallest

`atom/model_engine/model_runner.py:590`:

```python
gbs = next((g for g in reversed(self.runner.capture_sizes) if g >= bs), None)
```

Scanning `reversed()` finds the smallest matching rung only if `capture_sizes`
is **descending**. `capture_cudagraph` sorts it descending at line 4032 for the
capture loop and **ascending again** at line 4323 when it finishes, so at
runtime this returns the *largest* rung ≥ `bs` — the top of the ladder for
almost every batch.

Its own comment says otherwise:

> a 65-request batch replays the 128 graph, so 63 requests' worth of slots are
> never written

**Cost:** bandwidth only. This bound decides how much of the input buffer to
zero, not which graph to replay — the replay picks its rung in
`ForwardMode.decide`, which scans the ascending list correctly. So a step zeroes
up to `max(capture_sizes) * tokens_per_seq` slots where it needed
`smallest_rung_ge_bs * tokens_per_seq`. With a default ladder topping out at 256
and a typical decode batch of 8, that is roughly 32x more zeroing than required,
every step.

**How it was found:** Compass copied this expression to work out which rung a
step replays at, and every step in a 1662-row sweep came back as bucket 512 —
impossible for batches of 1 to 16.

**Fix:** not applied. Correcting it changes how much memory the engine writes per
step, which wants a performance check this project is not set up to give. Either
drop the `reversed()` (correct against the ascending list it actually holds), or
write it order-independently as `min((g for g in capture_sizes if g >= bs),
default=None)`, which is what Compass now does.

### 3. A stale comment about that same sort order

`atom/spec_decode/drafter.py:226`:

```python
capture_sizes = sorted(runner.capture_sizes)  # capture leaves it descending
```

Capture leaves it **ascending** (line 4323). The code is right — `sorted()`
gives ascending, which is what the drafter wants — but the comment says the
opposite, and it is exactly the belief that made defect 2 wrong.

**Cost:** none today; it is a trap for the next reader.

### 4. A worker's cause of death is replaced by an unrelated summary

Twice, a specific and nameable failure inside a worker process reached the
terminal as something else:

| what happened | what was printed |
| --- | --- |
| oracle could not open its calibration table (`FileNotFoundError`, path named) | `RuntimeError: Engine Core Mgr: Received unexpected SHUTDOWN signal from DP rank 0 during initialization` |
| `capture_cudagraph` returned `None` where the caller unpacks three values | parent hung forever on a shm broadcast, naming neither CUDA graphs nor the runner |

Neither run used data parallelism, and neither failure was during
initialization. The worker's own traceback *is* written to the log — the problem
is that the manager's summary contradicts it and arrives last, so it is what a
reader believes.

**Cost:** every failure inside a worker costs a debugging session it should not.
The second one presents as a hang rather than an error, which is worse.

**Fix:** not applied — it is a change to ATOM's process supervision, well outside
what Compass should be reaching into. Compass mitigates it only where it owns the
message: its own errors now name the file, the rank and the remedy, on the
assumption that the surrounding report will be misleading.

### 5. The container cannot JIT-build any C++ kernel

Not ATOM's code, but ATOM's shipped environment, and it blocks the model this
project exists to simulate. `rocm/atom-dev:vllm-latest` has gcc-14's runtime
directory (`/usr/lib/gcc/x86_64-linux-gnu/14`, holding `crtbegin.o` and friends)
but no gcc-14 C++ headers — `/usr/include/c++/` contains only `13`. ROCm 7.2.4's
clang prefers the highest version directory it finds, so:

    error: "Could not find standard C++ header 'cmath'..."
    fatal error: 'cstdlib' file not found
    2 errors generated when compiling for gfx942

Everything prebuilt is unaffected, which is why Qwen3-0.6B runs: it takes the ASM
paged-attention path, which ships as a `.co`. Qwen3.8-27B takes the *gluon* path,
which JIT-builds, and dies at engine init.

**Cost:** the target model cannot be loaded. Any AITER kernel needing a JIT C++
build is unavailable in this image.

**How it stayed unknown:** `aiter` runs the build with
`capture_output=AITER_LOG_MORE < 2`, so the compiler's output is discarded unless
that environment variable is set to 2. What surfaces is
`CalledProcessError: Command '['make', 'build', '-j1']' returned non-zero exit
status 2`, which says nothing about headers, and then defect 4 turns *that* into
a report about a DP rank shutting down during initialization. Three layers, each
discarding the one below.

**Fix:** `libstdc++-14-dev` is installable from the configured apt sources, and
adding it puts headers where clang already looks. Verified the mechanism by
compiling the failing translation unit with
`--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13`: exit 0.

Not applied here. It changes the shared container rather than this repository,
and the writable layer does not survive `teardown.sh`, so it belongs in the
image or in `gpu_docker`'s setup script rather than in an ad-hoc install.

### 6. Serving computes no per-request latency at all

`LLMEngine.postprocess` derives per-request TTFT, TPOT and latency from
`arrive_time` / `first_token_time` / `finish_time`, and is reached only from the
offline `generate()` paths (`llm_engine.py:282`, `:306`).
`atom/entrypoints/openai/api_server.py` never calls it and never reads
`first_token_time`. So a served request has no engine-side latency accounting —
what a benchmark reports is measured from outside, over HTTP.

**Cost:** for a normal deployment, a missing observability feature rather than a
bug: a client's own timing is close enough when the engine runs in real time.
Under Compass it is fatal, because the engine's clock is virtual and the client's
is not, so client-side timings describe the simulator.

**Fix:** partially, and only for Compass's purposes. The engine's clock readings
now travel on `RequestOutput` and are served by `GET /compass/requests`. ATOM's
own serving path still computes nothing.

---

## Fixed here

Defect 1 above is also fixed; it is left in place rather than moved because it is
the one an ATOM user is most likely to hit.

The two below are latent under normal operation — the two clocks agree when the engine
runs on the wall clock, so neither is visible without a virtual one. They are
listed because they are wrong in ATOM's code rather than Compass's, and because
a future reader changing that code should know why it looks the way it does.

### 7. `first_token_time` stamped off the wall clock at one of three sites

`atom/model_engine/scheduler.py` stamps `seq.first_token_time` in three places.
Two went through `get_clock().time()`; the third — on the speculative-decode
retention path — used `time.time()` directly. Under a virtual clock that makes
TTFT a wall-clock instant minus a virtual one, which is not a duration.

Fixed in commit `0e2f880f`. A test walks the AST of `scheduler.py` and fails if
any assignment to `first_token_time`, `finish_time` or `arrive_time` is sourced
from the `time` module again.

### 8. `finish_time` stamped after the output that should carry it

The finishing `RequestOutput` was constructed at `scheduler.py:2732` and
`seq.finish_time` assigned at `:2759` — after it. The offline path re-reads the
sequence later so it never noticed; anything reading the output itself saw zero.

Fixed in commit `0e2f880f` by stamping before the output is built, with the later
assignment made conditional so offline behaviour is unchanged.

---

## Not defects, recorded to stop them being rediscovered

* **`ForwardMode.decide` selects the graph correctly.** It scans the ascending
  list without reversing it, so it gets the smallest rung ≥ batch. Defect 2 is
  the padding bound next to it, not this.
* **The scheduler's `time` import is legitimate** where it is used for real
  elapsed time (timeouts, rate limiting). Only the three timestamp fields above
  must go through the injectable clock.
