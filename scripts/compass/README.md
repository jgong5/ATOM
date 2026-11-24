<!-- SPDX-License-Identifier: MIT -->
# Compass gate and pre-flight scripts

The scripts every task runs around its work: two test gates, a GPU pre-flight, the
snapshot builder a gate runs against, and the regenerators of the gates' lists and
baseline. Everything
here runs **inside a container**, which mounts the worktree parent so every task's
tree is reachable, and each script derives its tree from **its own path**, not from
your `$PWD`, so the copy in a worktree exercises that worktree. Run **your**
tree's copy: running another checkout's copy is refused rather than silently acted
on, because until 2026-09-20 it was not — see "Which tree a script acts on" below.

New to Compass? [QUICKSTART.md](QUICKSTART.md) takes a model from an empty store
to a simulation graded against real runs, and `smoke_cell.sh` runs that path in
about an hour. [ADVICE.md](ADVICE.md) takes a user from a space file of
configurations through `compass advice` to the one it recommends.
[USER_QUICKSTART.md](USER_QUICKSTART.md) predicts one configuration from a committed
law file, on a host with no GPU. `calibrate.sh OUT RECORD...` takes its calibration in one command,
from the warm record to the fits, and resumes a stopped run; its variables are in
its header.

`$T` below is the tree a command acts on. Run from `/`, not inside another checkout,
`PYTHONPATH=$T` makes `atom` that tree's; without it, `atom` is the image's `/app/ATOM`.

| Script | Where | What |
|---|---|---|
| `gate_cpu.sh` | CPU container | The CPU gate, the batch gate run before every landing: every test file outside `tests/plugin/` and `cpu_gate_exclude.txt`, no driver **as a batch**: run alone, a file in it can still reach `rocminfo` (`tests/test_postprocess_width.py` did, through `atom.model_engine.model_runner`'s `from aiter import …`, measured on node 18 at `bdd244c57` while it was in the gate). `tests/compass` runs as `COMPASS_GATE_SHARDS` (default 16) pytest processes and the rest of `tests/` as `COMPASS_GATE_REST_SHARDS` (default 4) more, no process running files of both, each set packed by `cpu_gate_durations.txt` and never into more processes than it has files. Each process prints its own summary; compass shard k writes a caller's `--junitxml` or `--basetemp` path with `.shard<k>` appended, rest process k with `.rest<k>`. `COMPASS_GATE_SHARDS=0` runs one process. **Must be green.** Also exits 98 when the diff is in the blind spot below and the GPU gate has not run on *this* tree or on one it carries from (below). Refuses a caller-supplied `-r`. |
| `gate_targeted.sh` | CPU container; GPU container for what it names | The task gate developers run at every round and reviewers at the head they review: the test files under `tests/` the change adds or edits (`test_*.py`, `*_test.py`), the change read from `gate_cpu.sh`'s sources (`COMPASS_CHANGED_FILES`, the diff since `COMPASS_INTEGRATION_REF`'s merge-base, `snapshot.sh`'s `.compass-changed`), in one pytest process that collects all of `tests/` as `gate_cpu.sh`'s do and deselects the rest. Prints `select: none` and passes when the change adds or edits no test file, and passes a selected file that collects no test. Exits 98, naming each file, for a selected file under `tests/plugin/`, in `cpu_gate_exclude.txt`, or with a test that skipped for want of a driver-run input (below); the module-level skip of a test file the change did not select does not count. With `ATOM_COMPASS_SLICE_MODEL` set, in the GPU container, it runs every selected file outside `tests/plugin/` and what `gpu_gate_skip.txt` leaves out on the first card's target, runs the cc-traces smoke alone with `ATOM_COMPASS_CCTRACES_MODEL` as its slice model, as `gate_gpu.sh` does, and exits 92 on such a skip. A selected file that skips because cargo cannot resolve atomesh's crates exits 96. Never asks for `gate_gpu.sh` on a trigger path: the batch gate does. Same tree resolution, `-r` refusal and verdict line (`GATE_TARGETED_RC=`) as `gate_cpu.sh`. |
| `gate_gpu.sh` | GPU container | The GPU gate, run on a landing batch whose diff is in the blind spot below, which holds that change alone, before `gate_cpu.sh` there, and with `COMPASS_GPU_GATE_SCOPE=all` by hand on the integration tip about daily. `tests/` outside `tests/plugin`, of `tests/compass` by default only the driver-run files `gpu_gate_driver_run.txt` lists, and every compass file, the superset, with `COMPASS_GPU_GATE_SCOPE=all` (the daily run); less any file `gpu_gate_skip.txt` names for the first card's GPU target, judged as a **delta** against the baseline of the first card's GPU target in `gpu_gate_baselines/<target>/gpu_gate_baseline.env`, with torch, HIP, ROCm and AITER recorded and compared. The failing node-ids are on file in `gpu_gate_known_failures.txt` beside it and compared **by name**. Calls `preflight.sh` itself, before and after. Runs the driver-run tests once, with the inputs it sets (see "The driver-run tests and their inputs" below). |
| `preflight.sh` | GPU container | `rocminfo` reachability, compute use and VRAM use (checks 1–3), plus a D-state census (check 0), and a list of bookable devices. A gfx1250 card (MI455) reads 100% use with no process on it, so there a queue on the card (`rocm-smi --showpidgpus`) stands for compute use. Run **before and after**. The census states its own scope: in a container it can only see this PID namespace, and says so rather than reporting a clear node. Also worth running for a CPU-only task that imports ATOM's model layer: `import aiter` runs `rocminfo`, and so does any module that imports aiter at load time (`atom.model_engine.model_runner`, `atom.model_ops.linear`), so on a wedged node that import hangs. A bare `import atom` and `atom.compass` run no `rocminfo` and load no torch; `atom.config`, `atom.model_engine.llm_engine` and `atom.entrypoints.openai_server` load torch but run no `rocminfo`. Measured on node 18, 2026-09-23. |
| `regen_cpu_gate_exclude.sh` | CPU container | Regenerate the **GENERATED** section of the exclusion list by iterating `--collect-only` to a fixed point (three passes at `83daf636d`: 37 errors, 1, clean). Preserves the MANUAL section verbatim. |
| `regen_cpu_gate_durations.sh` | CPU container | Rewrite `cpu_gate_durations.txt` from one run of this tree's CPU gate with `--durations=0`, its arguments passed on: per test file, the setup, call and teardown seconds pytest reports. Run it on a snapshot with no other gate in the container, so the seconds are read under the gate's own load. Writes nothing unless pytest was green and reported a duration. |
| `regen_gpu_gate_triggers.sh` | CPU container | Re-derive `gpu_gate_triggers.txt` from the exclusion list and the tree. Run after **any** change to `cpu_gate_exclude.txt` — the two files are derived from the same tree and are wrong separately. Writes the whole file, **header counts included**; nothing in it is maintained by hand. |
| `regen_gpu_gate_baseline.sh` | GPU container | Re-measure the GPU gate's baseline when `gate_gpu.sh` warns that the toolchain differs from it (a new image, a torch, HIP or AITER bump, another GPU): `git archive --prefix=ATOM/ -o BASE.tar <commit>` where the repository is, then `HIP_VISIBLE_DEVICES=<gpu> scripts/compass/regen_gpu_gate_baseline.sh BASE.tar`. Runs the suite `gate_gpu.sh` runs on its first card (outside `tests/plugin` and `tests/compass`) twice, each over a fresh extraction of the tarball, and rewrites `gpu_gate_baseline.env` and `gpu_gate_known_failures.txt` in `gpu_gate_baselines/<target>/` for the card's GPU target, leaving out what `gpu_gate_skip.txt` leaves out on it (`BASE_LEFT_OUT`). Writes nothing when the two runs pass different counts or fail different node-ids (exit 97, naming each node-id one run alone failed), when a run errors or does not complete (1), when torch, HIP or AITER reads `UNKNOWN` or torch reads no GPU target for the card (96), when pre-flight fails before or after (91), or when the tarball names no commit (94). |
| `snapshot.sh` | either | Build the `git archive` tarball a gate runs against (never `rsync`). Stamps `.compass-commit` (so output names its tree, and a `COMPASS_GPU_GATE_DONE` attestation can be checked) and `.compass-changed` (so the blind-spot question is answerable without `.git`); and `.compass-gpu-carry`: with `COMPASS_GPU_GATE_DONE=<G>` set and not HEAD, G, HEAD and the diff from G, else empty, refusing with 93 a G that is not a full commit sha. Refuses a dirty tree. |
| `compass` | either | The `compass` console script, run from this tree, for an image whose ATOM install has none. The stage-2 cell below runs it. |
| `pd_sim.sh` | GPU container | A simulated 1P1D run: the standalone clock authority (`python -m atom.compass.run`), a prefill and a decode API server on the simulated KV connector, and `atomesh` in PD mode between them. `tests/compass/test_pd_slice.py` drives one request through it. Its variables are in its header. |
| `deploy.sh` | GPU container; with `--on`, one per node | A prefill-decode deployment with `atomesh` between its instances: simulated, one `compass run` per instance, joined to the clock authority the script starts, or real, one `compass record` per instance. `--prefill` and `--decode` repeat for more instances of a side (xPyD), every instance of a side on the same ATOM arguments, named by index (`prefill0/`, `prefill1/`); simulated, the router then chooses among them as the run's router LP, and the split tells each transfer's prefill instance apart. `--on` starts an instance on another node through a command prefix (`ssh <host>`, `srun -w <host> ...`, or `spur exec <job> docker exec ...` as [the 005/013 pair](#a-1p1d-deployment-across-crsuse2-m2m-v3-005-and--013) runs it). Simulated with `--dp-aware` among the router's arguments, the router is the run's router LP, and `ATOMESH` must name an `atomesh` built from the tree being run (`cargo build --release` in `atom/mesh`); one built from other router source, as the containers' `/usr/local/bin/atomesh` is, is refused. `PREFILL_ENV` and `DECODE_ENV` set variables in one side's instances alone, written into each instance's command so `--on` carries them. A client drives it through `OUT/server.json` as it drives one server. Its usage is in its header. |
| `encoding_ingress.py` | CPU, beside the router | The C1228 recipe's encoding ingress in front of atomesh (its argv `--processes 4 --workers 16 --queued 1280 --max-connections 2048 ...`): each chat request rendered and tokenized as ATOM's server does and sent on with `prompt_token_ids`, the stream relayed byte for byte, a client hang-up closing the upstream request. `--stamps STAMPS` writes each request's stages to `STAMPS.<pid>`, from which `compass fit ingress STAMPS [--skip N] [--metrics PROM] --out FILE` fits a simulated run's `ingress` term (`atom/compass/ingress.py`): an encode thread held `fixed_s + per_byte_s` a client body byte, `workers` of them first come first served, and atomesh's parse of the forwarded body, `router_per_byte_s`, from its `/metrics` scraped after the last request. `compass fit host --ingress FILE` states the term in a host entry; a run whose host laws carry it has the traffic LP price the hop on the HTTP channel (DSV41_MI355.md, "The encoding ingress"). |
| `parity_one_node.sh` | GPU container, two devices (eight for `CONFIG=dsv4-dpa8`) | The scheduling-parity test: one workload (`python -m atom.compass.parity.traffic`) on a real `-tp 2 --enable-dp-attention` server (`CONFIG=tp2-plain`: `-tp 2` alone; `CONFIG=dsv4-dpa8`: DeepSeek-V4-Pro's nightly `-tp 8 --enable-dp-attention --enable-tbo` server; `CONFIG=pp2`: `-tp 1 -pp 2`, each stage replayed from and compared on its own record) with the step record on, then on a simulated one whose steps each end when the real one's did (`atom/compass/parity/replay.py`; `step_stub` prices their device time into the cost record only), and `python -m atom.compass.parity` over the two records. With `STORE` unset it first runs `compass measure memory` on the real run's GPUs, flags and environment into `OUT_DIR/store`, under `LMCACHE=1` with an lmcache server of its own beside it, and the simulated run reads that store. The last lines are the real run's admission (send to response headers: minimum, median, maximum) and, per DP rank, the steps compared, the steps identical and the first divergence. Its variables are in its header. |
| `PYTHONPATH=$T python -m atom.compass.detect.scenario` | either | The acceptance checks every simulated deployment's runs are held to, over the out_dirs of one run file run under different `PYTHONHASHSEED`: each run reached its finish with no refusal, the step tables are byte-identical, and each channel's messages were released to their receiver once each and none behind its clock. With `--record LP=DIR`, an engine LP's TAR rows must equal the timeline a constant step stub implies, computed from the parity record in `DIR` (`pd_sim.sh` writes one per side when `ATOM_COMPASS_PARITY_RECORD` is set). Prints each failure and exits 1. The vertical, DP, PD and PD-DP slices in `tests/compass/` hold their runs to it, with the traffic's outputs too. |
| `http_intake.py` | GPU container, one real server | `HTTP_INTAKE=STAMPS REAL=1 CONCURRENCY=256 SESSIONS=256 AIPERF_EXTRA_ARGS="--trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0" scripts/compass/cctraces_sim.sh RUN_FILE OUT`, aiperf stopped once the first turns are in, then `python3 scripts/compass/http_intake.py fit STAMPS`: the real server runs under the probe, which stamps each request body its event loop reads and each chat template it renders, and the fit prints the burst's `fixed_s` and `per_byte_s`, a `host` law entry's `http_intake` terms. While the server runs, the stamps are in a file in its temp directory, which `STAMPS.<pid>.local` names; its stop, its exit or a SIGTERM, appends them to STAMPS, the verbs refuse STAMPS until then, and a temp directory on NFS or CIFS refuses. A server that died first leaves the `.local` file, naming a path inside its container: while that container runs, `podman cp` the file out, append it to STAMPS and delete the `.local` file. A tmpfs `/tmp`, as QUICKSTART's container mounts, goes with the container's stop, and the run is taken again. A first-turn burst's bodies, short on messages, understate the per-byte rate, so gpu-40's law files stated the one `python3 scripts/compass/http_intake.py lone STAMPS PASSES` read over a lone run of a served DeepSeek-V4.1-Flash cell's own bodies to a server under the probe ([#2681](https://github.com/jgong5/ATOM/issues/2681)): the bodies sent one at a time, each once the one before has been read to its end, in PASSES passes of as many bodies each, the stamps cut to that phase; it prints the per-byte rate, the median over passes of each pass's Theil-Sen slope, with p10, p90 and the sha256, and `python3 scripts/compass/http_intake.py fit STAMPS 256 PER_BYTE_S` prints a burst's `fixed_s` at that rate, its span kept; those laws' `fixed_s` and `idle_s` were taken at that rate; they now state `compass fit host`'s, fitted on their own records' lone requests ([#3024](https://github.com/jgong5/ATOM/issues/3024), [#2846](https://github.com/jgong5/ATOM/issues/2846)). Check the GPUs are idle before and after. The law files' `host.hops.admission_s` is the transport floor, 0.54 ms on node 18: on one idle GPU, `python3 scripts/compass/http_intake.py serve STAMPS --model /root/models/Qwen3.8-27B --host 127.0.0.1 --server-port PORT -tp 1 --enforce-eager --max-model-len 8192`; once it logs `Uvicorn running`, `python3 -m atom.compass.parity.traffic PORT MODEL RECORD 64 8 0 0 1` (64 requests, one in flight: request 0 read to its end alone, the rest 8 s apart); stop the server, then `python3 scripts/compass/http_intake.py floor STAMPS RECORD/arrivals.json` prints the median it states, with p10, p90 and both inputs' sha256. On the same stamps, `python3 scripts/compass/http_intake.py idle STAMPS PER_BYTE_S`, with the burst's `per_byte_s`, prints the idle intake, the entry's `idle_s`, with p10, p90 and the sha256. With the server started under `ATOM_COMPASS_PARITY_RECORD=RECORD` too, `python3 scripts/compass/http_intake.py hops STAMPS RECORD/dp0.handoff.jsonl RECORD/arrivals.json` prints the ZMQ request hop (from request 1 on) and output hop and the SSE stream hop, each with p10, p90, N and the inputs' sha256; the law files' `host.hops` state the stream median as `stream_s`, and the ZMQ medians as `ipc_s` and `ipc_out_s` from a server started the same way but without `--enforce-eager` and with `--max-model-len 262144`, under `python3 -m atom.compass.parity.traffic PORT MODEL RECORD 64 4 0 0 0 20000`: request 0 asks for 20000 tokens and keeps the engine decoding while the rest are sent 4 s apart. A fresh container's first run spends its first minutes compiling kernels, so its probes find no engine decoding: measure on a later run. `kv_write_req_s` there is the p10 of `by_state.forward` from `python3 scripts/compass/http_intake.py pd STAMPS STEPS`, the KV write request from decode's send call to prefill's receipt for the receipts that found prefill inside a forward, off real 1P1D runs on two idle GPUs: `HTTP_INTAKE=STAMPS MODEL=/root/models/Qwen3.8-27B PREFILL_DP=1 DECODE_DP=1 PREFILL_GPUS=A DECODE_GPUS=B REQUESTS=64 GAP_S=0.3 LONG=20000 SERVER_ARGS="--max-model-len 32768 --no-enable_prefix_caching" scripts/compass/parity_pd.sh OUT`, STEPS being `OUT/real/record/prefill/dp0.jsonl`, where request 0 keeps decode stepping and the rest arrive while prefill steps on the one before; it pooled that run with one at `GAP_S=0.05` (`LONG` unset, `--max-model-len 8192`) and one at `GAP_S=4`. A receipt waits on prefill's worker freeing the GIL, mostly inside its kernel launches, so the median reads that wait, not the hop. |
| `handoff.py` | any container with `atom` | `PYTHONPATH=$T python3 scripts/compass/handoff.py RECORD [LAW]`: over a real run's step record (`compass record`'s `steps/`), the engine IO threads' queue handoffs it stamped, input receipt to queue put, put to the busy loop's get, and output put to send, as p50 and p99 per prompt-token or step-batch bucket against the median step, and the input p10, beside the law file `LAW`'s `ipc_s` hop when given. |
| `ca_grant_cost.py` | any container with `atom`; for the cross-node figure, one on each of two hosts | `python3 scripts/compass/ca_grant_cost.py serve N tcp://0.0.0.0:PORT` on the clock's host, then `python3 scripts/compass/ca_grant_cost.py lps N tcp://HOST:PORT` on the LPs' host: a clock authority over N synthetic LPs, each one channel of `--lookahead` seconds (default 1) from every other, served over the `tcp` carrier, and N LP processes that each ask `TAR` one second past their clock `--rounds` times (default 1000), then `NER(+inf)`. `lps` prints a `TAR`'s round trip, send to grant (p10, p50, p90); `serve` prints the endpoint it bound, then its CPU seconds per request served. At the default lookahead each round is a barrier, every LP granted once the last has asked; at 2 an LP may run one round ahead. The clock authority's sizing in `atom/compass/design/01_execution_and_time_model.md` (D3.1) states its figures. |
| `aiter_c90842add0.sh` | every GPU container on mi355-gpu-40, before its first server | `podman exec CONTAINER bash scripts/compass/aiter_c90842add0.sh`: commit upstream aiter `c90842add0`'s csrc (`aiter_c90842add0.patch`) on the image's aiter `41d4d8d7e` as `09d256420`, the aiter commit gpu-40's laws state, rebuild `module_quant` and `module_moe_sorting_opus` for gfx950, and check that a fresh process loads the files it built, by path and sha256. Without it a MoE pass whose `num_local_tokens` is one short of its rows zeroes the row past its stage-2 activations ([#2538](https://github.com/jgong5/ATOM/issues/2538)), and a run on gpu-40's laws reads a stack other than theirs and refuses. Run again, it rebuilds nothing. |
| `rdma_link.py` | a GPU container on each of two nodes; `fit` in any container with `atom` | Times Mooncake KV writes between two nodes and fits the inter-node link of crsuse2-m2m-v3-005 and -013 that `atom/compass/networks/crs005-013-ionic.json` holds: `target` on the decode node, `bench` on the prefill node, `fit BENCH [RECORD]` anywhere; its docstring lists the experiments, and [the 1P1D section](#the-inter-node-link--rdma_linkpy) the commands. |
| `cpu_gate_exclude.txt` | — | Excluded test files in two marked sections: **GENERATED** (`# BEGIN GENERATED`, driver-dependent at *collection* time; never hand-edit — `regen_cpu_gate_exclude.sh` rewrites it wholesale) and **MANUAL** (`# BEGIN MANUAL`, collects cleanly then fails on a driver call or for want of a `.git`, so the regenerator cannot see it). The MANUAL section **is** hand-edited; every entry must carry its observed failure above it. |
| `cpu_gate_durations.txt` | — | `<seconds> <test file>` lines, generated by `regen_cpu_gate_durations.sh`; never hand-edit. `gate_cpu.sh` deals each set of files, `tests/compass` and the rest of `tests/`, longest first, each to the process with the fewest seconds so far, then the fewest files; a file it does not list counts 0. |
| `gpu_gate_triggers.txt` | — | The source paths that no *running* CPU-gate test names. Generated, not hand-written; matched by `gate_cpu.sh`. A trailing `/` matches a subtree. |
| `gpu_gate_baselines/<target>/gpu_gate_baseline.env` | — | The GPU gate's baseline for one GPU target (`gfx942`, `gfx950`, `gfx1250`), as the `BASE_*` lines `gate_gpu.sh` reads when torch names that target for the first card: the commit measured, its pass and fail counts, how many tests `gpu_gate_skip.txt` left out of the run (`BASE_LEFT_OUT`), and the torch, HIP and AITER versions it ran under. A target with no pair is refused with 93, naming it. Written by `regen_gpu_gate_baseline.sh`; a change is a reviewed edit. |
| `gpu_gate_baselines/<target>/gpu_gate_known_failures.txt` | — | The known-failing GPU node-ids at the `BASE_COMMIT` of the `gpu_gate_baseline.env` beside it, verbatim, written by `regen_gpu_gate_baseline.sh`. Its line count and `BASE_FAILED` are two statements of one fact; `gate_gpu.sh` refuses to run if they disagree. |
| `cpu_gate_aiperf_deps.txt` | — | The packages the CPU gate container installs, without their dependencies, for the `aiperf` wheel `COMPASS_AGENTX_HARNESS` names; "The CPU gate container" below. |
| `gpu_gate_skip.txt` | — | Hand-written `<target> <test file>` lines, read when torch names that GPU target for the first card: files whose result there the target's pair cannot record, the kinds listed in its header. `gate_gpu.sh` leaves each out, a suite file's collected tests coming off the expected passes and a `tests/compass` file only leaving its run; `regen_gpu_gate_baseline.sh` leaves a suite file out the same way and records the count as `BASE_LEFT_OUT`, which goes back on. A `<target> driver-run` line says that target's nodes hold no driver-run inputs (below). Each line goes when the issue or Ruling above it closes. |
| `gpu_gate_driver_run.txt` | — | Hand-written: the `tests/compass` files that skip in the CPU gate for want of a driver-run input (below), which `gate_gpu.sh` runs of `tests/compass` by default. `tests/compass/driver_run_skips.py`, which `tests/conftest.py` loads, fails such a skip in a file it does not list, so the CPU gate holds the list complete. |
| `gate_map.sh` | CPU container | `COMPASS_COVERAGE_WHEEL=<coverage.py wheel> scripts/compass/gate_map.sh OUT`: the CPU gate under coverage.py, and to OUT the map `COMPASS_GATE_SELECT` reads, keyed by the tree's commit (below). |
| `gate_select.py` | either | What a map selects for a tree: `gate_cpu.sh` asks it which test files to deselect, `snapshot.sh` writes the `.compass-select` stamp with it, and `gate_map.sh` starts coverage and builds the map with it. |
| `_lib.sh` | — | Tree resolution, `PYTHONPATH`, the `import atom` assertion, commit stamp, the changed files, exclusion list and compass_harness install `gate_cpu.sh` and `gate_targeted.sh` share, and the pytest summary parse `gate_gpu.sh` and `regen_gpu_gate_baseline.sh` count with. |

## Gate a tree with its own `scripts/compass/` — four `test_snapshot_ref.py` failures mean you overlaid

Stage a tree with the copy of these scripts **that tree itself carries**, and run
that copy. Do not copy `scripts/compass/` from another branch over it. An older
staging recipe did exactly that — overlaying this directory from
`compass/p0.1-env-and-gates` (`105ca4197`, whose `scripts/compass` is tree
`ddb69e7aa`) so that every tree was gated from one known copy. It predates these
scripts being on the integration branch, and on any tree at or after `186d12829`
it now costs four failures on **every** side of a comparison:

| tree | its own `scripts/compass` | with the `ddb69e7aa` overlay |
|---|---|---|
| `b1dca15da`, tree `00386e887` (the integration head's tree at 2026-09-21T21:00Z) | **4594 passed, 0 failed**, rc=0 | **4590 passed, 4 failed**, `GATE_CPU_RC=1` |
| `83ef2a094` — an earlier head, tree `9091c1dc8` | **4570 passed, 0 failed**, rc=0 | **4566 passed, 4 failed**, `GATE_CPU_RC=1` |
| `cf6429387` — a branch, tree `95cb8358d` | **4501 passed, 0 failed**, rc=0 | **4497 passed, 4 failed**, `GATE_CPU_RC=1` |

149 skipped, 3 xfailed on all six runs, and the total is conserved on every tree
— the four are moved out of passed, not added. Measured node 18, container
`xiaobizh_n18_cpu`, 2026-09-21T20:06-20:09Z and 21:00-21:03Z, staged by
`git archive` + `docker cp`, run sequentially and unpiped, `import atom` asserted
under each root from `/` first.

The four are in `tests/compass/test_snapshot_ref.py`, and they are **correct
failures** — each asserts on *that tree's own* `snapshot.sh` messages, which the
tree has and the overlaid older script does not:

| Failing test | What the overlaid `snapshot.sh` does instead |
|---|---|
| `test_unresolvable_ref_refuses_at_ref_resolution` | prints the one merged `REFUSED: no merge-base with <ref>` for both refusals |
| `test_unrelated_history_refuses_at_merge_base` | same message, so neither test can tell which step refused |
| `test_remote_qualified_ref_resolves_and_names_itself` | has no remote-prefix fallback, so it exits 92 where the tree's own script resolves `fork/feature/atomcompass_new` |
| `test_snapshot_carries_both_stamps` | same: it exits 92 before writing a tarball to inspect |

The fifth test in that file passes either way: a local branch of that name
resolves in both scripts and neither announces a fallback it did not take. **Do
not exclude any of them.** A red gate here is the tests working, and it is the
only signal that says the instrument was swapped.

The overlay costs a second thing on the same path: the older `gate_cpu.sh` it
brings with it prints `Baseline is 4030 passed, 0 failed` when the gate fails, a
figure removed at `186d12829` because a script line cannot name its own commit —
**564** behind the 4594 `b1dca15da` reads with its own scripts, **540** behind
`83ef2a094`'s 4570 and **471** behind `cf6429387`'s 4501. It is stale by a
different amount on every tree, because it was never a statement about the tree it
prints on. So the reader of the false red is handed a stale control as well.

**If two trees being compared carry different copies, say so and name both tree
objects** — `git rev-parse <ref>:scripts/compass`. The gate is then a different
instrument on each side, and the delta is not a measurement of the diff. That
question is live, not settled. Any branch that edits this directory carries its own
tree by construction (`cf6429387` carries `95cb8358d`), and so does the integration
head the moment such a branch lands: it carried `9091c1dc8` at 2026-09-21T20:05Z and
`00386e887` at 20:58Z, when #99 landed. A tree census is therefore a reading, not a
property. Read 2026-09-21T20:58:34Z over `refs/remotes/fork/compass/**` after
`git fetch fork --prune`: **45** `compass/*` branches, **35** carrying this
directory, **six** distinct tree objects between them, **26** of those branches
still on the pre-`186d12829` `ddb69e7aa` the overlay recipe copies from. Read the
first three as readings with their times rather than as a current count — three reads
over the preceding hour gave 41 / 31 / four, 43 / 33 / six and 45 / 35 / seven, and
they move in both directions as branches are pushed and rebased. The `26` is the
exposed population and is the figure that justifies this section. "Every tree is
identical now" is what made the overlay look free.

## Running the test files a diff reaches — `gate_map.sh` and `COMPASS_GATE_SELECT`

Opt-in; without `COMPASS_GATE_SELECT` the gate runs as above. A map is built once per
tip, from its snapshot, in the CPU container. The wheel is fetched once where `pip`
reaches PyPI (`pip download --no-deps --only-binary=:all: coverage`):

```
COMPASS_COVERAGE_WHEEL=<wheel> COMPASS_GATE_SHARDS=16 $T/scripts/compass/gate_map.sh <map>
```

The map records, per test file, the functions whose bodies it ran in pytest's process
or in any Python process started under it with the tree on its path, and the tree's `.py`
files those processes opened and directories they listed, other than to import; what a
helper under `tests/` reads as it is imported counts for each test file importing it. A process killed
by a signal other than SIGTERM records nothing. Its run's verdict is not the tree's:
under coverage a test that waits a set time on a process, or counts what one prints as
it exits, can fail, and its file holds what ran up to there. A round stages and runs with it:
`COMPASS_GATE_SELECT=<map> scripts/compass/snapshot.sh <dir>`, then the gate with the same
variable. The gate prints `select: <n> of <m>, map <sha>` and deselects the rest. A test
file runs when it ran a function whose body the diff since the map's commit changes; when
it read a file the diff changes, or listed the directory of one the diff adds; when the
diff changes it, or a file under `tests/` it imports; or when the map does not hold it. It
prints `select: whole gate -- <why>` and runs everything when the map's commit is not an
ancestor of HEAD, the map records no source reads, the snapshot was staged without the
map, or the diff changes module-level code other than an added import or a new name's
definition, changes a definition outside its body, adds a method to a class, or touches
`compass_harness/` (the gate runs an installed copy), a `conftest.py`, `gate_select.py`,
or a non-Python file outside `atom/compass/design/`. Every process still collects all of `tests/`, so a selection of
none passes on that collection. A test that reads state an earlier file left, a session
fixture's or a cache's, is held only by the file that first ran the code.

## Baselines — two gates, two commits, two provenances

Not one baseline. The rows below were measured at different commits by different
tasks, so each row names its own commit and the gate it was measured in.

**Read them as history, not as a current expectation — the CPU total above all.**
It moves whenever `tests/compass/` grows, which is most tasks, so the recorded
figure is stale by construction between one task and the next, and it has twice
been taken as a control by a task that then read the intervening tasks' tests as a
surplus of its own. **A control is measured, not read**: run `gate_cpu.sh` on the
integration head your branch forked from, in the same container, and state both
figures beside their commits. The CPU row below once named no commit, which is
the omission that let it pass for current; the same gate on
`68ef4f329` measured **4380 passed, 149 skipped, 3 xfailed, rc=0** — node 18,
container `xiaobizh_n18_cpu`, 2026-09-22, 36.2 s of pytest inside 43.0 s of wall.

Two runs of the *same* tree can still differ by one: a ±1 in the passed/skipped
split, and — separately — a non-zero `GATE_CPU_RC`, are both outcomes of one
flaky test in ATOM's own suite — see "A red CPU gate that may not be your diff"
below before attributing either to a diff. They are alternatives: the
skip-variant is `rc=0` with the total conserved, the hard failure is `1 failed`
with a non-zero rc and no ±1.

| Gate | Result | Measured |
|---|---|---|
| CPU gate, at `105ca4197` | **4030 passed, 0 failed**, 149 skipped, 3 xfailed, rc=0 — decomposing as **3956 ATOM + 74 `tests/compass`**; the 3956 is PR #6's control, `042aad97d` with PR #6's `scripts/compass/` copied in, since `042aad97d` has none. Six runs at `d737f15e7` and `7ff80cc4b`, before PR #6's restack, read the same counts and **25.6-30.8 s of pytest inside 31.6-36.9 s of wall (`time` real) — a measured spread, not a bound** (`8c0ee374f`) | `105ca4197`, PR #6's head, landed as `4c16792d9` (same tree) — PR #6's gate table; node 18, container `xiaobizh_n18_cpu`, 2026-09-21, against a `git archive` snapshot with `PYTHONPATH` asserted and pytest's own rc captured before any pipe |
| CPU gate, at `186d12829` | **4501 passed, 0 failed**, 149 skipped, 3 xfailed, rc=0 — three runs, identical, 28.8-34.4 s of pytest inside 35-40 s of wall. The **4030** above and the **4380** in the paragraph above are this same gate at earlier trees; all three are history, and this one will be too | `186d12829` — the commit this branch forks from, which is its merge-base with the integration head, **read 2026-09-21T19:21Z**, node 18's own clock — node 18, container `xiaobizh_n18_cpu`, `git archive` snapshot staged by `snapshot.sh`, `PYTHONPATH` asserted, pytest's own rc captured before any pipe |
| GPU superset (`--ignore=tests/plugin`), the baseline #964's upstream merge set; the current ones are in `gpu_gate_baselines/` | **11346 passed, 131 failed**, 0 errors, 497 skipped, 3 xfailed, 1566.2 s; a run on GPU 3 the same day failed the same 131 node-ids with the same counts. Most failures are tests written for gfx950 formats and kernels (OCP FP8, MXFP8, E8M0, NVFP4, Group32 FP8 GEMM); the rest are kernel numerics, a Triton shared-memory limit, test stubs older than upstream's code, state shared between upstream test files, and `test_benchmark_catalog.py`'s `git rev-parse HEAD` in a snapshot with no `.git` | a `git archive` of pure upstream `661ef7661`, the ATOM the nightly 20261013 image ships; node 18's MI308X (gfx942), container `xiaobizh_n18_next13`, `HIP_VISIBLE_DEVICES=1`, 2026-10-14, torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip` **7.2.53211**, ROCm release **7.2.4**, AITER **v0.1.25-44-g9205e1059** |
| GPU superset (`--ignore=tests/plugin`), the baseline until #964's upstream merge | **4779 passed, 5 failed**, 0 errors, 105 skipped, 3 xfailed, **72.6 s**; two runs, byte-identical failing sets | `fe9ea043c`, node 18, container `xiaobizh_n18`, `HIP_VISIBLE_DEVICES=1`, 2026-09-20, torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip` **7.2.53211**, ROCm release **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509** (`git describe`) |
| `ruff check .` | 1003 errors, 640 fixable — the gate is *no new* error, not zero | `83daf636d` |
| `black --check .` | clean, 660 files | `83daf636d` |

**The CPU total moves when `tests/compass/` grows, and one of the things that grows
it is this directory.** `tests/compass/test_cpu_gate_exclude.py` parametrises
`test_every_trigger_path_still_exists` over the entries of `gpu_gate_triggers.txt`,
one case each, so regenerating that file changes the pass count by exactly the
change in the number of entries. The readings in circulation are the same gate
under different exclusion lists and a different `tests/compass`, not discrepancies:
**3925** at `83daf636d`, 32 exclusions and no `tests/compass`; **3956** is the
ATOM-only half, at `042aad97d` above, and 3925 + 31; **4005** at `3afcb4880`, with
`tests/compass` at 49; **4022** at `71d2a1ac2`, with it at 66 — 49 plus the 17 extra
trigger paths the corrected derivation below produces — and **4030** at `105ca4197`,
with it at 74, the 8 cases `test_gate_gpu_surplus.py` adds. The exclusion list went
32 → 29, adding those 31, because `test_dp_metadata.py`, `test_dp_sync_layout.py`
and `test_forward_mode.py` were re-measured **CPU-green**.

**The five GPU failures at `fe9ea043c` were four ULP comparisons and one bitwise check**, and all five are still in `gpu_gate_baselines/gfx950/gpu_gate_known_failures.txt` — not
"five bf16 ULP failures", which is what this file and three design documents said
until the review that produced this paragraph:

| Node-id | Character |
|---|---|
| `tests/test_fused_compress_ragged.py::test_kernel_matches_reference_on_ragged_batches[extend0-context0-cut+whole]` | `allclose`, off by one bf16 ULP: `max\|diff\| = 0.001953125`, exactly 2⁻⁹, against `atol=rtol=1e-3` |
| `…[extend1-context1-whole+cut]` | same |
| `…[extend2-context2-resume+fresh]` | same |
| `…[extend4-context4-tiny-then-long]` | same |
| `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise` | `torch.equal` — **bitwise, no tolerance at all**, so "one ULP" does not describe it and a tolerance bump would not move it |

All five are pre-existing and unrelated to Compass: no importable `atom` module
differs between `fe9ea043c` and the integration base `83daf636d` — the only delta
under `atom/` is markdown documentation, which no test imports. Cite the delta;
do not claim green.

## The CPU gate container

`gate_cpu.sh` and the two list regenerators run in a container on
`rocm/atom-dev:nightly_202611240650`, built from upstream `dc6cd23bb` with aiter
`82b6b4f19`, with no GPU device and the worktree root's parent mounted at its
own path:

```
docker run -d --init --name <user>_gate --network host -v <parent>:<parent> \
  rocm/atom-dev:nightly_202611240650 sleep infinity
docker exec <user>_gate bash -c 'cd /app/ATOM/atom/mesh && cargo fetch --locked'
docker exec <user>_gate bash -c 'tar -C <tree> --exclude=atom/mesh/target -cf - atom/mesh |
  tar -x -C /tmp && cd /tmp/atom/mesh && cp /app/ATOM/atom/mesh/Cargo.lock . &&
  cargo fetch && cp Cargo.lock /app/ATOM/atom/mesh/Cargo.lock'
docker exec <user>_gate git config --global --add safe.directory '*'
docker exec <user>_gate python3 -m pip install --no-deps -r <tree>/scripts/compass/cpu_gate_aiperf_deps.txt
```

"atomesh's cargo tests run offline" below says why `cargo fetch` runs once. The
image built atomesh from `dc6cd23bb`'s manifest. When the tree's
`atom/mesh/Cargo.toml` asks for a crate that lockfile lacks, the gate's cargo
probe fails offline and a diff touching `atom/mesh` exits 96. The third line
resolves the tree's manifest starting from the image's lockfile, which adds
only what it lacks, and puts the result where the gates read their lockfile.
The `safe.directory` line lets git, run as the container's root, read a tree
another user owns; without it `regen_gpu_gate_triggers.sh`
records the tree's commit as `UNKNOWN`. Run the gate with
`COMPASS_AGENTX_HARNESS` naming the wheel the next section builds.

The image installs no `aiperf` in `/opt/venv` (agentx-harness has its own
`/opt/agentx-venv`, at a version `compass_harness` does not pin), so the
wheel's dependencies come from `cpu_gate_aiperf_deps.txt`, installed without
theirs so that no package the image ships changes version. Its mooncake
extension imports; nightly 20261103's did not until moved aside.

On a host with GPUs, also mount an empty directory over `/sys/class/kfd`
(`-v <empty dir>:/sys/class/kfd:ro`). Without `/dev/kfd` the container has no
driver, but torch still counts the host's GPUs from that tree: the tests that
skip below a GPU count start workers and fail
(`tests/test_distributed_store.py`), and vLLM's import raises `No HIP GPUs are
available` (`tests/test_vllm_kda_state_offload.py`). Measured on mi355-gpu-40,
where `torch.cuda.device_count()` reads 8 without the mount and 0 with it.

The GPU container for `gate_gpu.sh` and `regen_gpu_gate_baseline.sh` runs on the
same image, with `/dev/kfd` and `/dev/dri`, and needs two repairs before either
collects `tests/`, or the run stops at collection (pytest rc=2):

```
docker exec <user>_gpu python3 -m pip install llguidance==1.9.1 'google-re2>=1.1'
docker exec <user>_gpu bash -c 'cd /app/aiter-test &&
  git update-index --skip-worktree tests/conftest.py && mv tests/conftest.py /root/ &&
  echo aiter/jit/flydsl_cache/ >> .git/info/exclude'
```

The first installs ATOM's `structured-output` extra, which the image lacks, so
`tests/test_structured_output.py` and its siblings import. The second is for
`tests/test_pp.py`'s `from tests.conftest import MockConfig`: ATOM's `tests/` is
a namespace package, and once collecting
`tests/attentions/deepseek_v41/test_a8w4_attention.py` has put `/app/aiter-test`
on `sys.path` through aiter, aiter's own `tests/conftest.py` answers that
import. Moved that way and with FlyDSL's cache, which a GPU run writes inside
aiter's tree, excluded, aiter's tree stays clean: the gates read aiter by `git
describe --dirty`, and `compass measure memory`, which the driver-run tests run,
refuses a source root with uncommitted files. The CPU gate leaves these files
out instead (`cpu_gate_exclude.txt`).
This image's `/opt/venv` holds no `aiperf`, so a gate there sets
`COMPASS_AIPERF_PYTHON=/opt/agentx-venv/bin/python`, the image's own
agentx-harness. Its `COMPASS_HARNESS_PYTHON`, a venv holding the agentx-harness
wheel and compass-harness, takes every other package from `/opt/agentx-venv`,
then from `/opt/venv`, through a `.pth` file naming both `site-packages` in that
order. `/opt/venv` holds no `cyclopts`, which `aiperf` imports, and versions of
`aiohttp`, `pyzmq`, `pandas` and others outside `aiperf`'s pins, which
`/opt/agentx-venv` meets; compass-harness imports `atom`, whose `torch` only
`/opt/venv` holds. Without either, each `test_cctraces_sim_smoke.py` cell waits
out its deployment for an `aiperf` service that died at import. `test_dp_slice.py`'s DP servers draw no master port of their
own and take ATOM's default, 29500, so a gate needs that port free; another
job's DP group on the host fails them. Setting `ATOM_DP_MASTER_PORT` does not
help: it overrides the `--data-parallel-master-port` that `pd_sim.sh` gives
each server, so the DP instances of one run, as the Store scenario's two
prefills and its decode, all start on that one port.

## The compass_harness tests need an agentx-harness wheel

`tests/compass/test_harness_*.py` skip when `compass-harness` or `aiperf` does not
import. On a tree that carries `compass_harness/`, `gate_cpu.sh` installs the tree's
own `compass_harness` into a run-local directory, with `--no-build-isolation`, so
`setuptools` must be installed. `aiperf` comes from the wheel `COMPASS_AGENTX_HARNESS` names, or
from the environment when that is unset. When the tests' skip condition still holds,
the gate exits 96 before pytest and names the reason.

The gate installs with `--no-index`, so it runs offline (inside `unshare -rn` too).
Build the wheel once, where `pip` can fetch `hatchling`, and stage it beside the
snapshot:

```
git -C <agentx-harness> archive 56a0cf70 | tar -x -C <src>
python3 -m pip wheel --no-deps -w <dir> <src>       # aiperf-0.12.0-py3-none-any.whl
COMPASS_AGENTX_HARNESS=<dir>/aiperf-0.12.0-py3-none-any.whl scripts/compass/gate_cpu.sh
```

`compass_harness` checks the installed `aiperf` source against its pinned
functions on import, so a wheel from another commit fails the same way.

## atomesh's cargo tests run offline, from the container's crate cache

`gate_cpu.sh` runs atomesh's unit tests (`cargo test --lib`) and every Compass
cargo target, selected by file name (`atom/mesh/tests/compass_*.rs`, so a new
Compass target must carry the prefix), in `atom/mesh` before pytest, with `CARGO_NET_OFFLINE=true` unless the caller set
it, and fails after pytest if they fail. `tests/compass/test_mesh_clock_client.py`
skips when `cargo metadata` fails, and the gate runs the same probe first: when it
fails and the diff touches `atom/mesh`, the gate exits 96 before pytest and names
cargo's error; otherwise it prints a `mesh:` line and goes on. It skips, by exact name,
tests that fail on a loaded host:
`core::worker::tests::test_load_counter_performance`, a wall-clock throughput floor
(#515), and `policies::tree::tests::test_tree_structure_integrity_after_stress`, a
thread race that records a placeholder tenant (#514).

The tree has no `Cargo.lock` (`atom/mesh/.gitignore`). `gate_cpu.sh` and `gate_gpu.sh` copy one in from
`COMPASS_CARGO_LOCK`, default `/app/ATOM/atom/mesh/Cargo.lock`, the lockfile the
gate image built atomesh with, unless the tree already has one. The image's cargo
cache holds only what its release build downloaded, so a new gate container needs
the rest fetched once, with the network, outside any netns:

```
docker exec <container> bash -c 'cd /app/ATOM/atom/mesh && cargo fetch --locked'
```

A change to atomesh's `Cargo.toml` that needs a crate the cache lacks makes the
probe fail, and so the gate exit 96, until `cargo fetch` runs in that tree's
`atom/mesh` in the container, with the network.

## The MI455 image, and what stopped a TP4 start in the earlier one

The gate image ships no gfx1250 kernels. MI455 work runs under
`rocm/ufb-private:vllm-dsv4flash-mori-epv2-mi455-20260922-r5`
(`sha256:1a3ff678e922`) with aiter at `b6a9d0aeb`, the stack
MI455 A0's law states and `compass measure memory` holds a start to.
Its aiter is a git checkout (`/sgl-workspace/aiter`) and it carries `rocm-smi`,
the two things that command reads for the stack and the device. Its entrypoint
launches vLLM, so the container sleeps instead; a tree runs as in the gate,
a `snapshot.sh` tarball extracted inside and run from `/` with `PYTHONPATH=$T`:

```
docker run -d --name <user>_ufb --network host --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host -v <parent>:<parent> -v <models>:<models> \
  --entrypoint /bin/sleep rocm/ufb-private:vllm-dsv4flash-mori-epv2-mi455-20260922-r5 infinity
```

In each new container, move aiter from the image's `a6d2b564f` to `b6a9d0aeb`,
and FlyDSL from 0.2.4 to the 0.4.0 it pins. `git stash` sets the image's own aiter
patches aside: a modified checkout reads `b6a9d0aeb-dirty`, and the stack check
compares the commit exactly. Each command has its own line because `set -e` does
not stop on a failure before the last command of an `&&` list.

```
docker exec <user>_ufb bash -c 'set -e
  cd /sgl-workspace/aiter
  git stash
  git fetch origin main
  git checkout --detach b6a9d0aeb
  git submodule update --init --recursive
  git clean -fdX -- aiter/jit
  pip install --no-deps flydsl==0.4.0
  PREBUILD_KERNELS=0 pip install --no-build-isolation --no-deps -e .'
```

Every ATOM process in it runs with these settings:

| Setting | Without it |
|---|---|
| `ENABLE_CK=0` | aiter's JIT fails `module_activation` and `module_sample` in CK's `amd_tdm_store` |
| `GPU_ARCHS=gfx1250` | aiter's JIT fails `module_deepgemm_opus`, built for the card's 192 CUs where OPUS registers gfx1250 at 256 |
| `ATOM_ENGRAM_UVA=1` with `ATOM_ENGRAM_UVA_HBM=1` | `ATOM_ENGRAM_UVA=1` alone page-locks 189 GiB of Engram tables, more than the node's host memory; `ATOM_ENGRAM_UVA=0` gathers each forward's Engram rows on the host, 5 to 50 ms on 28% of one-sequence decode steps ([#2797](https://github.com/jgong5/ATOM/issues/2797)) |
| `NCCL_MIN_NCHANNELS` unset | the image's 112 faults the first RCCL all-reduce |

The GPU gate on MI455 runs in a container of its own from this image, with the
same aiter and FlyDSL and these settings, `ATOM_ENGRAM_UVA` left unset, so
`gpu_gate_baselines/gfx1250/` pins this stack; nothing for the gate is installed
into the container that serves.
It holds what the gate's runs import that this image lacks: LMCache at
`f22dec28`, the commit nightly 20261026 carries, built for this image's torch
(`BUILD_WITH_HIP=1 BUILD_MOONCAKE=0 PYTORCH_ROCM_ARCH=gfx1250 pip wheel
--no-deps --no-build-isolation`, with `grpcio-tools` 1.83.1) and installed
`--no-deps` with numba 0.65.0, llvmlite 0.47.0, sortedcontainers, aiofile,
aiofiles, caio, cryptography 50.0.2, redis, httptools and
`opentelemetry-exporter-prometheus` 0.65b0, the image's own numpy, protobuf and
OpenTelemetry kept; and the crates atomesh's cargo tests build from, fetched
under nightly 20261026's `Cargo.lock`. MI455 nodes run only models verified on them, so
`gpu_gate_skip.txt` marks gfx1250 `driver-run`: the gate there sets no
driver-run input (below), needs one card and accepts the driver-run tests'
skips.

DeepSeek-V4.1-Flash then serves at TP4 with
[its recipe's](../../recipes/DeepSeek-V4.1-Flash.md#gfx1250-mi455x) gfx1250
flags and default hotswap, each rank peaking near 26 GB of host memory
([#2359's finding](https://github.com/jgong5/ATOM/issues/2359#issuecomment-6491285682)).
aiter past `b6a9d0aeb` faults in its TP4 MoE (#2458).

What stopped `ec341a05a` serving V4.1-Flash at TP4 on an A0 node in the earlier
image, `rocm/fw-bringup-mlperf:gfx1250-atom-20261026-dsv41`
(`sha256:fd7450fde2da`), with the measurements
in [#2359](https://github.com/jgong5/ATOM/issues/2359):

- **Every process translates its own GPU code.** On a gfx1250 A0 card ROCr
  rewrites each B0 code object a process loads for A0, in that process, and keeps
  nothing for the next; `HSA_HOTSWAP_VERBOSE=1` logs `installed eager gfx1250
  B0-to-A0 hook` there. Each rank holds its translations in host memory, which a
  TP4 start's four ranks exhaust in `ncclCommInitRank`, before the weights load;
  watch `MemAvailable` while more than one such process starts.
- **Its aiter lacks `ROCm/aiter#6833`:** `fused_dual_rmsnorm_mxfp8_quant` takes
  no `amax_floor`, which V4.1's `DualRMSNormMXFP8` passes.

### A TP4 serve that stops mid-run, and what to capture before stopping it

A V4.1-Flash TP4 serve on the A0 cards can stop mid-run at concurrency 16 and
above ([#2922](https://github.com/jgong5/ATOM/issues/2922)). Every rank stops
logging steps while requests keep arriving, and the stall ends in one of two
forms. In one, after 600 s the NCCL watchdog names the TP group's one-element
`BROADCAST` in `ModelRunner.postprocess` on all four ranks, the first tracked op
that did not finish, and the abort that follows removes the stalled queue. In the
other, something else removes it before the watchdog fires: there is no
watchdog line, and the server log shows HSA's `HW Exception by GPU node-N ...
reason :GPU Hang`. Either way, removing the stalled queue makes one card's MES
time out (`MES(0, 0) failed to respond to msg=REMOVE_QUEUE` in dmesg), and that
card resets (`device wedged, but recovered through reset`) and passes a GEMM
check afterwards. The card differs between runs. Wave dumps taken before a reset
([#2922's record](https://github.com/jgong5/ATOM/issues/2922#issuecomment-6703152077))
show a few waves on one WGP of the stalled card whose stores never complete: in
their kernel while the other cards' waves wait in a collective for them, or in the
context-save trap once a removal has begun. py-spy can show every rank at the next
step's start, waiting in `D2HQueue.receive` for the previous step's output copy.
The record holds the rate, its exposure, and the runs that did not reproduce it:
no fix is shown by three clean runs. For cell A, #2147 gives a run lost to this
signature one retake and makes a second loss at one concurrency a leftover on
#2922 ([#2147's term](https://github.com/jgong5/ATOM/issues/2147#issuecomment-6703124926)).

The state is gone once the card resets, and the node's `mes-monitor` service
dumps waves only on its first MES failure after it starts. So treat no
`Scheduled prefill batch` or `Engine Core: output send` line for 90 s, while the
engine's last `Running: N reqs, Waiting: M reqs` report is not 0 and 0, as a
stall, and before stopping anything take these, with the owner's OK. The
container has its own PID namespace, so take each pid from
`docker exec <user>_ufb ps -ef`, not from the host:

```
sudo /usr/local/bin/ddvs checkmi sq_waves verbose   # resident waves: PC, status, outstanding counts
sudo /usr/local/bin/ddvs checkmi cp_dump            # queue and CP state
docker exec <user>_ufb py-spy dump --pid <pid>      # each rank, the EngineCore, the api_server
dmesg -T
```

## A red CPU gate that may not be your diff — one flaky test in ATOM's suite

`tests/entrypoints/test_stream_marker_properties.py::TestTheRegionIsNotCopiedPerChunk`
is non-deterministic. It is ATOM's own test, it is present at **every** control, and
it is not a Compass defect: do not modify it and do not put it in
`cpu_gate_exclude.txt` — excluding it would change what the gate measures on both
sides of the delta. It is written down here because four agents have been warned
about it by hand and one lost a gate run to it.

The class asserts a **timing** property — that a buffered region is not recopied
per chunk — by streaming a 32 KB and a 128 KB payload and comparing cost per KB,
with a linear-loop control arm and a `pytest.skip` noise guard
(`if not 0.6 < control < 1.6`). Wall clock on a shared box is its input, so its
outcome moves with the box's load.

**It is three-way, not two-way**, and the third outcome is the one worth knowing:

| Outcome | What the run looks like | Count | Rate |
|---|---|---|---|
| nominal | passes | 18 | 85.7% |
| skip-variant | passed **−1**, skipped **+1**, total conserved, `rc=0` | 2 | 9.5% |
| **hard failure** | `1 failed`, **non-zero `GATE_CPU_RC`**, `cost per KB grew 1.89x` | 1 | 4.8% |

A skip-variant is recognisable: the pass count moves by one and the skip count
moves back the other way. A hard failure is not — `GATE_CPU_RC` is non-zero, and
**by the number alone that is indistinguishable from a regression**, while the
instinct on a red gate is to look in your own diff. Read the `FAILED ` line the
gate prints first (that is what its `-rf` is for): if it names this class, re-run
rather than hunt.

**Do not pipe the gate.** A pipeline's status is its *last* command's, so a piped
run hands the caller `tail`'s: measured on a forced failure, `gate_cpu.sh 2>&1 |
tail -6` reported **0** while the gate itself exited **1**. This whole section is
addressed to someone reading a non-zero rc, and a piped run does not give them one.
`tail` truncates from the **top**, so it also decides which half survives: `2>&1 |
tail -6` kept 5 of the 9 stderr lines — the test's name among them — and dropped
pytest's `FAILED ` line; `2>/dev/null | tail -6` dropped the paragraph whole and
kept the `FAILED ` line and the counts. Redirect to a file and read `$?`. The gate
does print `GATE_CPU_RC=` on stdout on every path, so the number survives in the
*text* of an untruncated pipe — but only an unpiped run puts it in `$?`.

**What those rates are, and are not.** n=21, node 18, container `xiaobizh_n18_cpu`,
on a box whose load was not controlled. Those conditions, the table and its one test
id, `test_the_cost_per_byte_does_not_grow`, are issue #93's. 19 of the 21 are pinned:
PR #79's review (issue comment 5765186351) ran `b58a48cc2` 19 times, for 17 nominal,
1 skip-variant and the one hard failure. The other 2, 1 nominal and 1 skip-variant
by subtraction from the table, name no commit, so half the skip-variant rate rests
on #93's count alone. #93 attributes the skip-variants to that test, but no
skip-variant run named it (PR #99). Not-nominal combined is ~1 in 7.
The finding is the third outcome, not the rate. The rates are a **lower bound on
the class**, not a measurement of it: the class holds **3 methods / 4 collected
cases** (one is parametrised `buffered-region` and `kimi-incremental`), of which
**2 methods / 3 cases** share the same timing helpers, the same noise guard and the
same `< 1.5` ratio assertion. Only
`test_the_open_region_is_never_scanned_beyond_the_window` does not: it counts scans
of the open region and is deterministic. The sibling
`test_no_format_pays_more_per_byte_as_the_payload_grows[buffered-region]` has been
observed failing the same way — `qwen: cost per KB grew 1.73x from 32 to 128 KB`,
`1 failed, 4495 passed, rc=1`, on the integration head `669dc3f9d`; node 18,
`xiaobizh_n18_cpu`, 2026-09-21, recorded in the round-2 review of PR #67 (issue
comment 5765660762), which hit it on its own gate run. A sequential re-run of that
same tree passed at **4496**, and 1 + 4495 = 4496 accounts for it exactly. Inherited,
not measured here — so the failure belongs to the mechanism and not to the one
method #93 names. Per #93, the two skip-variants both occurred under `gate_cpu.sh`,
and the one hard failure in the counts under direct pytest; comment 5765186351
gives only its mix, 7 gate + 12 direct.

**Run gates sequentially.** Per #93, the hard failure was reproduced when two gate
loops on node 18 overlapped; #93 names neither that run's harness nor whether it
is the one in the counts. Two gates on one box compete for the CPU the control arm is
measuring, which is the condition this test is least able to survive — check for a
running `gate_cpu.sh` before starting one.

**Seen again since, on another tree.** Three `gate_cpu.sh` runs on `354965883`
(node 18, `xiaobizh_n18_cpu`, 2026-09-21T19:05–19:08Z): the first read **4495
passed, 150 skipped**, the next two **4496 passed, 149 skipped** — `rc=0` on all
three and the total conserved at **4645**, so nothing was added or lost and one
test moved from passed to skipped. That is the skip-variant's signature. The gate
prints no skip *reasons* (and refuses a caller's `-r`, which is how you would ask
for them), so that run did not name the test; the counts and the conserved total
are what was observed.

**Gloo timeouts in `tests/compass/test_dp_step_max.py`,
`tests/compass/test_moe_gather_rows.py` and `tests/compass/test_process_group_lp.py`
were DNS, not load.** Their gloo groups
used to resolve the hostname once per rank to pick an address, and node 18's
hostname is not in `/etc/hosts`. One stalled lookup outlasted the group timeout,
and the others failed with `wait timeout after 5000ms`, `Connect timeout` or
`Connection refused` (#726). The fixtures now set `GLOO_SOCKET_IFNAME=lo`, so no
lookup happens; a red in those files is a real failure.

## Which tree a script acts on

Each script resolves its tree from `$BASH_SOURCE`, so it acts on the checkout it
lives in. That is deliberate, and it is not the same question as "the tree the
caller meant".

**EXECUTED 2026-09-20.** From a checkout at `83daf636d` on `feature/atomcompass_new`,
running another worktree's copy —
`bash …/compass-worktrees/p0/scripts/compass/snapshot.sh <outdir>` — wrote
`compass-d78f3bbd3.tar` stamped `commit: d78f3bbd3…` and exited 0, with no warning.
The stamp was truthful about the tree it archived and silent about the tree the
caller was standing in. All five callers of `compass_tree_root` had the same
exposure; for a gate the consequence is worse, since a pytest result would be
attributed to a commit it did not come from, and a `COMPASS_GPU_GATE_DONE`
attestation would then agree with a stamp that describes the wrong branch.

`compass_tree_root` now **refuses** (exit 99) when `$PWD` is inside a *different*
checkout. Resolving from `$PWD` instead was rejected: it would let a script act on
a tree that is not its own. Being invoked by path from outside any checkout is
still allowed — there nothing contradicts the caller, and it is how
`snapshot.sh`'s own extract instructions say to run a gate.

## The CPU gate's blind spot — a file, and a gate

`gpu_gate_triggers.txt` is the blind spot stated as paths a script can match,
generated by `regen_gpu_gate_triggers.sh`. The rule it applies:

> an `atom` module named by an excluded test is a blind spot **unless a CPU-gate
> test that actually runs names it too**.

Four decisions make that sentence operational. Three were forced by a defect or a
measured counter-example; the fourth, collection, changed no path at `236abfd9a`
and is kept as a forward guard:

1. **Indentation.** The parser anchored on `^`, and at `fada7424e` **190** of the
   `import atom.*` lines in non-plugin test files were indented — function-local and
   guarded imports are the dominant idiom here. It was reading about a third of the
   lines it claimed to read. The excluded side now reads any indentation.
2. **Symbols.** `from atom.model_ops import eplb` parsed to the bare package
   `atom.model_ops`, which resolved to the whole subtree and swallowed
   every sibling entry. `from X import y` is now read as `X.y`, resolved to
   `X/y.py`, else `X/y/`, else `X.py`, else `X/__init__.py`. An `atom.*` reference
   that resolves to none of those is a **refusal**, not a warning: either the parser
   misread a line or a test imports something absent, and both make the derived sets
   incomplete by an unknown amount.
3. **Skips.** An indented import is not proof that anything executed it. The
   excluded side reads any indentation, but coverage is credited **only for a
   module-level import**. At `236abfd9a` (measured in `xiaobizh_n18_cpu`) that half
   carried both counter-examples:
   `tests/model_ops/test_moe_dp_token_capacity.py::test_dpa_capacity_fits_gathered_topk_metadata`
   imported `topK` inside a test marked `skipif(not torch.cuda.is_available())`, and
   `tests/test_mla_index_cache.py::test_model_runner_local_total_layers_adds_mtp_only_on_drafter_stage`
   imported `ModelRunner` **indented four spaces inside that test function**, not at
   module level. Reading coverage at any indentation dropped the set 30 → 29, losing
   `topK.py`; doing that *and* crediting files that collect nothing dropped it 30 → 27,
   losing `atom/model_engine/model_runner.py` (the module Compass's runner seam
   replaces), `aiter_mla.py` and `topK.py`. A later module-level import in a CPU-gate
   test credits any of them; `gpu_gate_triggers.txt` is the current list.
4. **Collection.** A file that collects nothing covers nothing, so coverage is
   credited only from a CPU-gate file `--collect-only` shows collecting at least one
   test. At `236abfd9a` that probe removed no path: **30 triggers with it, 30
   without, difference empty** (measured in `xiaobizh_n18_cpu`; it
   withheld 15 coverage paths, of which the only candidate,
   `atom/model_ops/v4_kernels/state_writes.py`, was absorbed either way by the
   candidate subtree entry `atom/model_ops/v4_kernels/`). It is a forward guard for
   trees that one does not represent, not what kept `model_runner.py` there, and
   it costs a full `pytest --collect-only` over the CPU gate plus a refusal path
   (exit 97).

So the two sides are **asymmetric on purpose**: candidates come from any import in
an excluded test; coverage is subtracted only for a **module-level** import in a
CPU-gate file that collects at least one test, because those are the only imports
that provably execute (collection imports the module). Making both sides
indentation-tolerant would undo point 3 and re-delete `topK.py`.

**It errs in both directions, so it is not a floor.** Toward *firing*: an indented
import in a CPU-gate test that does run is not credited, so its module can be listed
although the CPU gate reaches it; and coverage subtraction is exact-string, so a
candidate **subtree** entry is never cancelled by coverage of the files beneath it,
nor a candidate file by coverage of its package. Toward *silence*: imports are read as text, not
resolved as a graph, so a module reached only transitively is invisible to both
sides. A path **absent** from the file is not a claim that the CPU gate covers it,
and a path **present** is not proof that it does not. Of the two mistakes, firing
spuriously is the one to prefer: running the GPU gate when in doubt is never the
wrong one, and the asymmetry above is that preference written down.

`gate_cpu.sh` **enforces** that preference rather than restating it. It takes the
changed-file list from `COMPASS_CHANGED_FILES`, else from `git diff` against the
merge-base with `feature/atomcompass_new` (`COMPASS_INTEGRATION_REF`, resolved as
below); with neither it exits **98**, naming the missing stamp, rather than
answering "no". When a changed file matches a trigger,
the run ends 98 unless `COMPASS_GPU_GATE_DONE` discharges it:

| `COMPASS_GPU_GATE_DONE` | Outcome |
|---|---|
| unset | **98** — run `gate_gpu.sh`, then re-run with that tree's HEAD |
| set, but this tree has no commit (no `.git`, no `.compass-commit`) | **98** — an attestation names a tree; accepting an uncheckable one makes the rule bypassable in exactly the snapshot where the gate normally runs. Rebuild with `snapshot.sh`. |
| set to another value G, and the diff from G to this tree touches a trigger, or a snapshot has no `.compass-gpu-carry` stamp from G to this HEAD, or G is not a full commit sha | **98** — the GPU gate passed on a different tree and does not carry here |
| set to another full sha G, and the diff from G touches no trigger | discharged, `gpu: satisfied by <G> carried to <HEAD>`; the diff is `git -c core.quotePath=false diff --no-renames --name-only G` in a checkout, or in a snapshot the stamp `snapshot.sh` writes when staged with the same `COMPASS_GPU_GATE_DONE` |
| equal to this tree's HEAD | discharged |

The trigger match is *reported* before pytest and *enforced* after it, so one run
yields both answers instead of trading one for the other. Green at the CPU gate is
not green at the test gate when the diff is in the blind spot.

## Which integration ref resolves, and which step refused

`COMPASS_INTEGRATION_REF` defaults to `feature/atomcompass_new`, which is the
branch's name on the remote and **not** a local branch in a linked worktree or a
fresh clone — there only `fork/feature/atomcompass_new` exists. `_lib.sh`'s
`compass_resolve_ref` therefore tries the bare name first, then the same name
under each configured remote, and reports which one it used: `snapshot.sh` prints
a `ref:` line and names it beside the base, `gate_cpu.sh` names it in the `gpu:`
source. Before that, the default resolved nowhere and `snapshot.sh` exited **92**
in every worktree with nothing set.

A second `ref:` shape appears when the bare name **did** resolve — to a local
branch that has drifted from the remote branch of the same name. Both scripts
print it, in the same position as the one above:

```
ref:    feature/atomcompass_new (1b473e5af) is 2 commit(s) behind fork/feature/atomcompass_new (cae322c86)
        the base is the local branch; COMPASS_INTEGRATION_REF=fork/feature/atomcompass_new uses the remote
```

It reports and does not redirect — the base stays the ref that resolved, in all
three directions (`N commit(s) behind`, `N commit(s) ahead of`, `diverged from
(N ahead, M behind)`), and a local branch level with its remote prints nothing,
which is what makes the line mean something when it appears. Two things to read
it with. `diverged from` is also what **unrelated** histories print, because
`rev-list --left-right --count` returns the whole of each side when there is no
merge-base; it is the merge-base refusal below, not this line, that reports a
pair with no common history at all. And the counterpart is found by **same name** under each remote —
what `compass_resolve_ref` would have picked had the local branch been absent —
so a branch whose configured upstream is a *differently* named remote branch is
either silent or compared against a ref it does not track. That class is the
branches whose `git config --get branch.<name>.merge` names something other
than `refs/heads/<name>`; enumerate it rather than trusting a number, because
`git push -u` moves a branch out of the class the moment it is published — it
creates the same-name ref and rewrites the config in one step — so any count
is stale as soon as anyone pushes. The announcement is not a substitute for
keeping the local integration branch fast-forwarded.

The two ways that can still fail are separate refusals, because one is a
statement about the *ref* and the other about the *history*, and a reader who
confuses them inspects the wrong thing:

| Step | Message | Means |
|---|---|---|
| ref resolution | `REFUSED: ref resolution failed -- <ref> names nothing here…` plus git's own line, verbatim | nothing was compared; set `COMPASS_INTEGRATION_REF` |
| merge-base | `REFUSED: merge-base failed -- <ref> resolved, but shares no commit with HEAD.` | the ref is fine; the histories are unrelated |

Both exit **92**: the exit code is what other scripts key on and did not change,
and the two are told apart by the message. `tests/compass/test_snapshot_ref.py`
covers each path separately so they cannot merge back into one.

A bare `git archive` tree — one extracted without `snapshot.sh`'s stamps — is the
adjacent case, and `gate_cpu.sh` still exits **98** there. The stamps are not
written retroactively: `.compass-changed` is a diff against a base the stamped
tree no longer has any way to compute, so writing one would be a guess in the
shape of a measurement. The refusal names the omission instead.

## What the GPU gate expects, and on a tree that carries no Compass tests

`gate_gpu.sh` judges an **equality**, not a floor: `BASE_PASSED` plus whatever
`tests/compass/` passes on the tree in front of it, minus what it contributed
at the baseline (`BASE_COMPASS_TESTS`). A floor would be loosened by exactly the
tests each task adds, so a task adding 30 tests while silently losing a 20-test file
would still clear it.

It runs pytest twice, over disjoint sets: the compass files its scope names on
every card `HIP_VISIBLE_DEVICES` names (`COMPASS_GPU_GATE_SCOPE`: `batch`, the
default, the files `gpu_gate_driver_run.txt` lists; `all`, every file of
`tests/compass/`), then the rest of `tests/` on the first of them alone,
as the baseline was measured (an upstream test that needs two cards skips there).
The surplus is the first run's pass count, so no test runs twice; the failing and
erroring node-ids of both runs are judged together against the known-failures list.

| this tree's `tests/compass/` | expected passes |
|---|---|
| absent | `BASE_PASSED + 0 - BASE_COMPASS_TESTS` — the figure a tree without this phase's tests measures |
| present, N passing in the scope's files | `BASE_PASSED + N - BASE_COMPASS_TESTS`; a failing Compass test is a failure the baseline does not name |

Both sides of the arithmetic are **pass** counts, so a Compass test that skips or
xfails is accounted for, with one exception below. A suite file `gpu_gate_skip.txt` leaves out on the card's GPU target subtracts its collected tests from both rows, less the baseline's `BASE_LEFT_OUT`.

### The driver-run tests and their inputs

The driver-run tests start ATOM's own servers on the simulated clock (the
vertical slice and every scenario test built on it). Each skips without its
inputs, and a skip is not a failure, so a gate that does not set them passes
without running them. On a tree whose `tests/compass/` reads
`ATOM_COMPASS_SLICE_MODEL`, `gate_gpu.sh` sets each input, keeps one the caller
set, and refuses with **92**, naming every one that is missing, before any test
runs, except on a target `gpu_gate_skip.txt` marks `driver-run`, whose nodes
hold none: there it sets none, needs one card and accepts a driver-run skip.

Of `tests/compass/`, the default run takes only the files
`gpu_gate_driver_run.txt` lists, the cc-traces smoke among them, run apart as
below: every other compass file runs in the CPU gate. A compass test that skips
there on a reason naming one of the inputs below, or `atomesh` on PATH, fails
unless that list names its file (`tests/compass/driver_run_skips.py`), so a driver-run
file is never left to no gate. `COMPASS_GPU_GATE_SCOPE=all` runs every compass
file instead, for the daily run.

| input | default |
|---|---|
| `ATOM_COMPASS_SLICE_MODEL` | the Qwen3-0.6B snapshot in the Hugging Face cache (`$HF_HOME`, else `~/.cache/huggingface`) |
| `ATOM_COMPASS_IMAGE_MODEL` | the Qwen3.5-9B snapshot there |
| `ATOM_COMPASS_MTP_MODEL` | the Qwen3.5-0.8B snapshot there, for `test_spec_scenario.py`'s MTP1 run |
| `ATOM_COMPASS_DSPARK_MODEL` | `/root/cells/deepseek-v41-flash-dspark`, for its DSpark run: `tests/models/deepseek_v41/fixtures/config.json` as `config.json` beside DeepSeek-V4-Flash's `tokenizer.json` and `tokenizer_config.json`. Both speculative runs read a config and a tokenizer, never a weight |
| `ATOM_COMPASS_CCTRACES_MODEL` | the Qwen3.8-27B snapshot there: the cc-traces smoke runs alone with it as its slice model, since its first turn is longer than Qwen3-0.6B's context |
| `ATOM_COMPASS_CCTRACES` | `/root/cells/cc-traces-256k/traces.jsonl` |
| `COMPASS_HARNESS_PYTHON` | `/root/cells/venv/bin/python`, agentx-harness with compass-harness installed; the smoke runs with this tree's `compass_harness` first on `PYTHONPATH` |
| `COMPASS_AIPERF_PYTHON` | `/opt/venv/bin/python`, agentx-harness without compass-harness |
| `ATOMESH` | `atom/mesh/target/release/atomesh` in the tree; its directory goes first on `PATH` |
| cards | four in `HIP_VISIBLE_DEVICES` (the multi-node DP test runs two trees of TP2) |

The router has to pass `router_binary` (`atom/compass/run`): built clean, at a
commit, from this tree's `atom/mesh`. A snapshot has no history, so on a
`snapshot.sh` tree that means a router built (`cargo build --release --bin
atomesh` in `atom/mesh`) in a git checkout of the snapshot's commit; a cargo build
in the snapshot itself names no commit and is refused.

After the run, a `SKIPPED` line under `tests/compass/` whose reason names the slice,
image, MTP or DSpark model, `ATOM_COMPASS_CCTRACES`, `COMPASS_AIPERF_PYTHON` or `atomesh` is a
finding by its file and line: that test did not run. The tests that cargo-build
atomesh run offline (`CARGO_NET_OFFLINE=true` unless set), from the container's
crate cache under the image's `Cargo.lock`, as in `gate_cpu.sh`; where cargo cannot resolve its crates they
skip, and the gate exits 96 naming the first, before the rest of `tests/`.

On node 18 (`xiaobizh_n18_next13`), `tests/attentions/deepseek_v41/test_tbo.py`
hangs; a run that passes `--ignore` for it reads that file's passes short of
`BASE_PASSED`.

## Why the gates refuse a caller-supplied `-r`

Both gates forward `"$@"` to pytest. pytest's `-r` is **store-last-wins**, so a
caller's `-rE` replaces the gate's own `-rf`/`-rfE` and no `FAILED ` lines are
printed at all. `gate_gpu.sh` builds its verdict out of those lines: **EXECUTED on
`d78f3bbd3`, `gate_gpu.sh -rE` compared the five baseline failures against an empty
observed set, reported all five as "no longer failing", and exited
`GATE_GPU_RC=0` while they had in fact failed.** Both gates now refuse the flag
(exit 95) rather than absorb it, and `gate_gpu.sh` treats "baseline failures no
longer failing" as a refusal in its own right — it has three causes (genuinely
fixed, stopped running, could not be parsed) and only the first is a pass.

## Why `import atom` is asserted before every run

A prior run resolved `atom` to a non-git snapshot of a different branch, 72 files
divergent, and failed silently wherever both trees defined the symbol. `_lib.sh`
sets `PYTHONPATH` to the tree and **nothing else** — inherited entries are the
hazard, not a convenience — then asserts the resolved path is under it, and exits
92 if not. The probe runs from `/` on purpose: run in place, Python puts the cwd at
the head of `sys.path` and the assertion passes whatever `PYTHONPATH` says.

## A cc-traces cell on a simulated TP1 run — `cctraces_sim.sh`

`cctraces_sim.sh RUN_FILE OUT_DIR` runs one cell in the GPU container: ATOM's API
server with `--compass-run`, driven by `aiperf profile --scenario
inferencex-agentx-mvp` through compass-harness, on the first `SESSIONS` traces of a
cc-traces `traces.jsonl`. The cell ends when the scenario's `--benchmark-duration`
has passed on the simulated clock; nothing is computed or allocated on the device.
The header of the script lists its environment; `MODEL`, `TRACES` and
`HARNESS_PYTHON` are required.

- `RUN_FILE` is a run file as `atom/compass/run/__init__.py` describes it. The
  script replaces its `out_dir` and `clock_endpoint`, sets `client_runs` to 1, so
  the server leaves at the end of aiperf's run, and adds a `workload` entry:
  the sha256 of `TRACES` and of each replayed trace, the aiperf arguments and the
  pinned benchmark id. The step table's configuration line and the run summary
  carry it.
- The served tokenizer and the HTTP intake are charged from the run file's
  `host` entry (`--host-laws`): a `tokenizer` term fitted on the other backend
  refuses to start the server, and a term the entry lacks is charged 0 s and
  named in the run summary's `uncharged_host_terms`. The channels' lookaheads
  are the `hops` that entry states, each prefill-decode side's from its own;
  a hop it lacks takes 2^-14 s and is named in `unmeasured_hops`.
- `HARNESS_PYTHON` is a Python with agentx-harness 56a0cf70 and compass-harness
  installed; the script puts this tree's `compass_harness` first on `PYTHONPATH`
  and refuses (exit 92) when it resolves elsewhere.
- Each cell writes its aiperf memory-mapped dataset under `OUT_DIR`, so cells run
  side by side.

The last line printed is the cell's result: aiperf's request count and errors,
simulated and wall seconds from the run summary, both prefix-cache hit rates
(`cache_read_pct` is the engine's, `theoretical_hit_pct` is aiperf's oracle), TTFT,
ITL and request throughput in simulated time, and the run's refusals. A
`coverage_report=True` cell is a coverage report, not a result.

A run file with `router_s` makes the cell 1P1D: `pd_sim.sh`
deploys it, taking `DECODE_EXEC`, `ATOMESH` and each side's `PREFILL_ARGS` /
`DECODE_ARGS` flags and `PREFILL_ENV` / `DECODE_ENV` variables from the
environment, and aiperf drives the atomesh router. The traffic LP sends to
`frontend-P` and reads the stream from `frontend-D`, as the run file's channel
table names them; with `--enable-dp-attention` in either side's flags the cell
is agentic, its router is `--dp-aware` and LP `router`, and the traffic LP
talks to `router`. Each hop into LP `router` takes the run file's
`router_in_s`, by default `ipc_s`, borrowed: Atomesh logs no receipt to measure
its own from. The result line adds `kv_transfers` and
`kv_transfer_mean_s`, from the line the prefill engine logs per KV transfer.

`REAL=1` runs the same TP1 cell on a real server, on the GPU and the wall clock,
so a real and a simulated cell at one seed, concurrency and traces hash send the
same requests and warm up on the same ones. The server runs ATOM's own runner
(no `--compass-run`), and aiperf runs as shipped: `HARNESS_PYTHON` must have
agentx-harness without compass-harness (exit 92 otherwise), and the script pins
the benchmark id and draws each session's id from its trace and branch as
compass-harness does (`_lib.sh`'s `COMPASS_REAL_AIPERF`), so both send one
session id, which ATOM's DP router breaks a tie by. The result line reads aiperf's export
alone and adds `contaminated`, true when `rocm-smi` lists any process on one of the
cell's GPUs before or after it, with each one's pid and VRAM bytes. Both modes
write the TP1 step record to `steps/dp0.jsonl` (`atom/compass/parity`);
`RECORD=0` turns it off. `FIRST_SESSION` skips that many traces, so calibration
and graded cells can replay disjoint ones.

Two runs of a cell with different `PYTHONHASHSEED` must give byte-identical step
tables and step records: `cmp A/step_table.txt B/step_table.txt`.
`tests/compass/test_cctraces_sim_smoke.py` runs that pair on one session for 60
simulated seconds when `ATOM_COMPASS_SLICE_MODEL`, `ATOM_COMPASS_CCTRACES` and
`COMPASS_HARNESS_PYTHON` are set, and with `COMPASS_AIPERF_PYTHON` (no
compass-harness) a real cell on zero dummy weights beside it.

The cell's `OUT_DIR` also holds one hand-in per server LP, `lp-<lp>.json`. The
engine's, `lp-engine.json` (`lp-engine-P.json` and `lp-engine-D.json` for 1P1D),
holds the readings the simulated KV pool was sized from.
Against the server log of the paired real run, with the same deployment flags,
`PYTHONPATH=$T python -m atom.compass.memory.check OUT_DIR/lp-engine.json REAL_SERVER_LOG`
prints the per-term comparison of the non-KV memory and the graph pool. It takes
the log of one rank; with `--tp N`, of one DP rank at TP width N, a table per TP
rank's line; with each DP rank's `lp-engine.dp<r>.json` too, of a server
started under `ATOM_LOG_MORE=1`, as `compass record` and `compass measure
memory` start one at DP and TP above 1, whose lines name their process, and
each runner process its DP and TP rank. In place of that log it takes the `--out` of a `compass
measure memory` start of the deployment, and then also compares the `weights`
and `buffers` terms of `peak_torch` with the storage bytes the start loaded.

## The step linear law from real cells — `step_linear_fit`

`PYTHONPATH=$T python -m atom.compass.backends.step_linear_fit RECORD... --out LAW --platform JSON [--form FORM] [--moe-routing E,K] [--engine FILE]`
fits the law a run file's `step_linear` key holds (`atom/compass/backends/step_linear.py`) from
the step records of real cells, `OUT_DIR/steps/dp0.jsonl` of `REAL=1
cctraces_sim.sh` runs, each with the `dp0.engine.json` the run wrote beside it,
whose engine config the law states and whose `data_parallel_size` its ranks
join at. Records that ran before a run wrote that file, or wrote it without
`data_parallel_size`, take an engine file stated from their `server.log` as
`--engine FILE`; `compass fit steps` and `compass fit host` take it too, the
host laws then keyed by it with the records' launch graph sets
(`atom/compass/fit/samples.py`). Engine and host key files name the compilation level only
since [#3696](https://github.com/jgong5/ATOM/pull/3696), so these fits and `measure steps`
read an older record's `--level` from the ATOM arguments after `--` in its `run.json`
(`step_linear_fit.recorded_level`): a refit on cell A's, cell B's or gpu-40's records keys
level 0. An older record whose `run.json` names none, as `calib_capture.sh` writes it, keys
the default, and takes `--engine FILE` naming any other. `atom/compass/backends/step_labels.py` states the label
and what is cut, and the command writes each record's labels beside it for
reading, `dp0.cost.jsonl`; the module's docstring states how outliers are
dropped. A prefill step's label is a trace run's busy time, which `calib_capture.sh`
does not take ([below](#the-stage-2-cell-through-the-compass-cli-from-an-empty-store)).
The written law also carries, per vector, its samples, warm and dropped
counts and condition number, and names any vector it could not fit under
`unfitted`. `--moe-routing E,K`, the model's routed experts and top-k, lets each
decode rung adopt one column over its steps' real MoE tokens, the tokens or the
experts they reach (`atom/compass/backends/step_linear.py`), by the rule the
module's docstring states, and records under the vector's `fit.moe_column` what
the rule read; a law refitted without it adopts none.

`JSON` is the capture host's platform, as `compass run --device DEVICE --ptl PTL`
composes it from the store's `platform_readings`:
the law, as `platform`, and its host key state its device, software stack, PTL
state and torch, each as far as its kind's key holds it, and a run's spec must
match them. The backend package reads no spec, so it is printed by
`PYTHONPATH=$T python -c 'import json, sys; from atom.compass.artifacts import ArtifactStore, Kind, keyed_platform as k, machine_platform as m; from atom.compass.fit.device import compose; p = m(compose(sys.argv[1], ArtifactStore(sys.argv[3]), sys.argv[2])); print(json.dumps({x.value: k(x, p) for x in (Kind.STEP_LINEAR_LAW, Kind.HOST_LAWS)}))' DEVICE PTL STORE`.

Calibration cells replay traces the graded cells do not: `FIRST_SESSION=256
SESSIONS=137` against graded traces 0 to 255, by the
[owner's ruling on #627](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096543740).
`calib_capture.sh RUN_FILE OUT_DIR` captures them: the real cells one at a time,
each with the host load and the GPUs' processes, VRAM and use at its start and end; its header
lists its environment.
Given two or more records, the command also prints each form's error on every
record under a law fitted on the others, per decode rung and prefill token band.

`scripts/compass/laws/qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json.xz` is the
`history` form, labelled by device time, fitted by `compass fit steps` on
records captured on one card of hjbog-srdc-39 at 731debf47 (ROCm/ATOM 661ef7661)
in the nightly-20261013 image, in this order, by the cells `calib_capture.sh`'s
header names for this model: concurrency 1, 4 and 16, then first-turn cells
at 4, 8, 10 and 12, then a late-start cell at 1. The cells were chosen by decode rung: each rung the graded
runs decode at, 1 to 8, has steps in two or more cells
([#1024](https://github.com/jgong5/ATOM/issues/1024)), and rung 1 reaches the graded runs' longest
contexts ([#2180](https://github.com/jgong5/ATOM/issues/2180)); it has no decode vector above rung 16, since
this model is not captured or graded at concurrency 64 or 256.
Its host laws (`host`, with `compass fit host`'s report as `host_fit`) are fitted
apart, by `compass fit host`, on calibration cells at concurrency 1, 4 and 16
over traces 256 to 392, each a `compass record` server and one `client.sh` run
with aiperf's scenario warmup, captured one at a time on one card of
hjbog-srdc-39 at ae5bab999 in the same image after a warm cell nothing reads.
Those records stamp the engine's input pulls, so every kind carries `E_out`,
`E` runs from the pull and `W_launch` ends at the token read. The host key is the one those records carry. The law
states hjbog-srdc-39's platform, so it
refuses under any other device, software stack, PTL state or torch; its
host key states that platform less the PTL state. Its name states that PTL state
rather than the host it was captured on.

`scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-gpu45-{prefill,decode}.json.xz`
are DeepSeek-V4-Pro's DPA-TP8 laws on mi355-gpu-45, one per side of a 1P1D
deployment: the `base` form with host laws, each fitted alone on its side's
capture at ATOM 2d4044b15 by `compass fit steps` and `compass fit host`.
Each side's cells are concurrency 1, 4, 16, 64 and 256 at `cctraces_sim.sh`'s
default `SEED` and 64 and 256 at `SEED=20260708`; the decode side adds 64 at
`SEED=20260709`, the prefill side two more at 64 and `SEED=20260708`. At DP
width 8 a step is fitted when every rank ran it, a dummy forward included;
each decode rung a law fits carries a held-out hull, except the decode
side's rung 2, so its steps price unvalidated. Their host laws (`host`, with
`compass fit host`'s report as `host_fit`) were refitted on the same cells by
`compass fit host` for #1893, with `E` labelled by `step_labels.label_run`.
Those records stamp the engine's input
pulls, so every kind carries `E_out`, `E` runs from the latest pull over the
eight ranks before the same forward and `W_launch` ends at the token read.
Their host keys carry the `atom`
digest of the tree they were last re-keyed at; `provenance` names the
capture tree's. Their engine config, in each law and its host key, states
`index_cache_dtype` fp4: each cell ran one server with no `--kv-transfer-config`,
for which ATOM resolves the fp4 indexer at gfx950, so a P/D side, which ATOM
resolves to fp8, refuses them ([#2998](https://github.com/jgong5/ATOM/issues/2998)).

`scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-crs005-{prefill,decode}.json.xz`
are the same pair for the cell on crsuse2-m2m-v3-005 (prefill) and -013 (decode),
keyed by the platform their `platform` states, in the image of that stack, at
`--gpu-memory-utilization 0.75`: the `base` form with host laws, each written by
`step_linear_fit` from its side's cells. The cells are the gpu-45 pair's, run by
`calib_capture.sh`, the prefill side's on 005 and the decode side's on 013, and a
cell that built an aiter JIT module was run again. The decode side adds a c=256 cell
at seed 20260709 and, for rung 12, c=96 at seeds 20260707 and 20260708 and c=112 at
seed 20260709: a lone decode server at c=256 decodes at rungs 16 to 28, and at c=64
at rungs 4 and 8. The decode law's rung 1, 2, 4 and 8 vectors are refitted with
`--moe-routing 384,6`, DeepSeek-V4-Pro's routed experts and top-k, on those cells
and on the decode sides of two 1P1D `deploy.sh record` cells (below) at c=256 over
traces 256 to 392, seeds 20260707 and 20260708, whose decode ranks run unevenly loaded
([#2493](https://github.com/jgong5/ATOM/issues/2493)), with a step's DP dummy forwards
counted as one token in `moe_experts` ([#2536](https://github.com/jgong5/ATOM/issues/2536));
each adopts `moe_experts`.
Its other vectors and its host laws are the ones fitted on the lone cells alone, and
its `sources` names the records each came from. The prefill law's rung-2 vector is the
decode law's, stating `declared`, since the prefill law's own fit left rung 2 unfitted
([#2660](https://github.com/jgong5/ATOM/issues/2660)). Its prefill vector is refitted
in the `cached_seqs` form on records that ran the fp8 indexer alone: the prefill sides of
#975's 1P1D cell, at c=1 over traces 256 to 392 and 0 to 255, seed 20260707
([#2853](https://github.com/jgong5/ATOM/issues/2853)), and of #975's own rounds at c=1 to
256, and two lone prefill cells at c=64 that ran fp8
([#3073](https://github.com/jgong5/ATOM/issues/3073), [#3402](https://github.com/jgong5/ATOM/issues/3402));
its other vectors but rung 2, and its host laws, are the lone cells' fit. The pair's lone
cells ran the fp4 indexer and its 1P1D cells the fp8 one. Each law is keyed to the fp8 a P/D side resolves, without a refit: it
states no `index_cache_dtype`, which then reads as `kv_cache_dtype`'s fp8, so a server that
resolves fp4 refuses it. Its provenance states the gap that interleaved fp8 and fp4 lone cells
measured ([#3073](https://github.com/jgong5/ATOM/issues/3073#issuecomment-6757202887)).
Each carries, as `memory_readings`, the readings `compass measure memory` took of its side
of [#3498](https://github.com/jgong5/ATOM/issues/3498)'s DPA8 1P1D deployment on
crsuse2-m2m-v3-005, each member stating `assumed`, so a run of that side reads them from
a store that holds none.

`scripts/compass/laws/deepseek-v4-pro-dpa8-ep8-mi355x-ptl-disabled-661ef7661-{prefill,decode}.json.xz`
are the same pair with expert parallelism on (MORI): the `history` form, fitted by
`compass fit steps` and `compass fit host` on each side's cells, captured on mi355-gpu-40
at ATOM 78f8c0adf, which shares `mi355x-gpu45.json`'s device, stack and PTL state:
concurrency 1, 4, 16, 64 and 256 at `cctraces_sim.sh`'s default `SEED`
([#1526](https://github.com/jgong5/ATOM/issues/1526)) and 64 and 256 at `SEED=20260708`;
the prefill side adds 64 three more times at `SEED=20260708` and once at the default,
the decode side 64 once more at the default ([#1810](https://github.com/jgong5/ATOM/issues/1810)).
Each decode rung a law fits has steps in two or more cells, so each carries a held-out
hull. Host keys and the fp4 indexer as the gpu-45 pair's.

`scripts/compass/laws/deepseek-v4-pro-dpa8-ep8-mi355x-crs005-{prefill,decode}.json.xz`
are the EP8 pair for the cell on crsuse2-m2m-v3-005 (prefill) and -013 (decode), keyed by
the platform their `platform` states, in the image of that stack: the `base` form with host
laws, each written by `step_linear_fit` from its side's cells. The cells are the gpu-40 EP8
pair's, run by `calib_capture.sh` at `--gpu-memory-utilization 0.75` with each side's EP8
arguments, the prefill side's on 005. -013's page cache served other bytes than its disk while
its first decode cells ran ([#2794](https://github.com/jgong5/ATOM/issues/2794)), so the decode
cells were run again, on 005 and, after -013's reboot, on both nodes at once, each started once
its node's page cache read the same bytes as its disk. A cell that built an aiter JIT module,
or whose text came back garbled, was run again. The decode side adds c=32 cells at seeds
20260707 and 20260708 and a c=24 cell at seed 20260707: a lone decode server at c=16 decodes at
rung 1 and at c=64 at rungs 4 and 8, and the 1P1D decode side also at rung 2. The prefill law's
prefill vector, and the decode law's rung 1, 2, 4 and 8 vectors with `--moe-routing 384,6`,
DeepSeek-V4-Pro's routed experts and top-k, are refitted on their side's lone cells and on their
side of three 1P1D `deploy.sh record` cells at c=256 over traces 256 to 392, seeds 20260707,
20260708 and 20260709, the prefill side at `--gpu-memory-utilization 0.72`. Each law's other
vectors, the prefill law's rung-1 vector among them, and its host laws are the ones fitted on its
lone cells alone, and its `sources` names the records each came from. The prefill law's rung-2
vector is not: it is the decode law's, stating `declared`, since no prefill-side record decodes at
rung 2, which the 1P1D prefill side reaches at c=64 ([#2878](https://github.com/jgong5/ATOM/issues/2878)).
Host keys and the fp8 indexer key as the crs005 DPA-TP8 pair's, its lone cells fp4 and its 1P1D
cells fp8; the gap between the two is unmeasured on this pair.

`scripts/compass/laws/deepseek-v4-pro-tp8-mi355x-crs005-{prefill,decode}-host.json.xz` hold host
laws alone (`host`, `host_fit`, `platform` and `provenance`, no step law), one per side of
[#3179](https://github.com/jgong5/ATOM/issues/3179)'s TP cell on crsuse2-m2m-v3-005 (prefill) and
-013 (decode): the crs005 DPA-TP8 cell less `--enable-dp-attention`, the prefill side at
`--gpu-memory-utilization 0.65` and the decode side at 0.70. They are for a 1P1D run whose steps
another backend prices, both entries in its run file, where each server's own start picks the
entry keyed to its engine config. `deploy.sh run ... --host-laws P --host-laws D` builds that run
file with the decode started through `--on "env -u GPU_MAX_HW_QUEUES -u ATOM_NUMA_BIND"`: the two
entries' hops differ, and each side's are picked under the environment it starts in. Each
law file is `compass fit host` on its side of three 1P1D `deploy.sh record` cells at ATOM
259bc70f0, at c=1, 4 and 16 over traces 256 to 392 for 1800 s each, every server loading the
blocking-call tool through `ROCP_TOOL_LIBRARIES`; the c=1 cell's prefill side ran at
0.70 ([#3194](https://github.com/jgong5/ATOM/issues/3194#issuecomment-6869663548)). `host_fit.held_out`
holds each band's errors with each record held out in turn. A term a side's records cannot give is
the crs005 DPA8 law's of that side, as `provenance` names: `admission_s` and `stream_s`, since a
side's requests and streams pass the router; `kv_write_req_s` and `router_s`, which `compass fit
host` does not fit; the prefill side's `ipc_out_s`, since no output message ran alone on it; and the
decode side's `http_intake`, since its requests render no chat template
([#3229](https://github.com/jgong5/ATOM/issues/3229)). A decode side encodes no prompt and admits no
request, so its entry states no encode term and `admit` null.

`scripts/compass/laws/deepseek-v4-pro-tp8-mtp3-mi355x-crs005-{prefill,decode}.json.xz` and
`deepseek-v4-pro-dpa8-mtp1-mi355x-crs005-{prefill,decode}.json.xz` hold host laws alone, one per
side of [#1499](https://github.com/jgong5/ATOM/issues/1499)'s two MTP cells on crsuse2-m2m-v3-005
(prefill) and -013 (decode): [#3179](https://github.com/jgong5/ATOM/issues/3179)'s TP8 cell with
`--method mtp --num-speculative-tokens 3` on both sides, and [#975](https://github.com/jgong5/ATOM/issues/975)'s
DPA-TP8 cell with `--method mtp --num-speculative-tokens 1` on both sides, for a run whose step
costs come from another backend, each side taking its own entry. Each is `compass fit host` on its
side of three 1P1D `deploy.sh record` cells at c=1, 4 and 16 over traces 256 to 392, 1800 s each,
every server started with `ROCP_TOOL_LIBRARIES` naming the blocking-call tool
(`atom/compass/parity/README.md`), and states the `platform` its records ran on. The host key
states the drafter (`speculative`), so a side started without it, or at another draft width,
refuses the law. A term a side's records give none of, or that `compass fit host` does not fit,
is the crs005 DPA8 law's of that side, copied without a refit; `provenance` names each, with why.
Each prefill law's `prefill.L_mtp` is its own `decode.L_mtp`, copied the same way: the status
copy it prices is the previous output forward's, whatever this forward's kind, and the prefill
forwards waited on it too rarely to fit one. Each host key's `atom` is re-keyed without a refit
from the digest its records ran at; `provenance` says why.

`scripts/compass/laws/deepseek-v4.1-flash-tp4-dspark5-mi455-a0.json.xz` is DeepSeek-V4.1-Flash's
law for cell B of [#2138](https://github.com/jgong5/ATOM/issues/2138), TP4 with the routed
experts sharded across TP and DSpark with 5 speculative tokens, on one MI455 A0 node: the
`history` form with host laws, keyed by the platform its `platform` states, fitted by `compass fit
steps` at ATOM ed5fa7697 and `compass fit host` at a tree that places `L_mtp` on the drifting
host and device clocks ([#3096](https://github.com/jgong5/ATOM/issues/3096)), on `calibrate.sh`
records at ed5fa7697, taken in the image and aiter of that stack
([#2765](https://github.com/jgong5/ATOM/issues/2765)). The records are concurrency 1, 4, 16
and 64, then 8, 12 and 56 from each trajectory's first turn for rungs 7, 8, 12 and 56. No
c=256 record is in it: that take stalled the node
([#2922](https://github.com/jgong5/ATOM/issues/2922)), so decode rungs past 24 rest on the
first-turn c=56 record. The draft's `propose` vector is fitted apart, by `compass fit steps`
at that later tree on two 120 s trace runs at c=16 and c=4 under the same settings, whose busy
time labels an output prefill's propose pass; that tree's fit refuses the ed5fa7697 records
beside them ([#3607](https://github.com/jgong5/ATOM/issues/3607)). Each server ran with
`ENABLE_CK=0
GPU_ARCHS=gfx1250 ATOM_ENGRAM_UVA=1 ATOM_ENGRAM_UVA_HBM=1 AITER_LOG_LEVEL=WARNING OMP_NUM_THREADS=4
ATOM_DISABLE_MMAP=true AITER_REUSE_IDENTICAL_COMM_GROUPS=1`, `NCCL_MIN_NCHANNELS` unset, and
`SERVER_ARGS` of `--trust-remote-code -tp 4 --kv_cache_dtype bf16 --index-cache-dtype fp8
--method dspark --num-speculative-tokens 5 --gpu-memory-utilization 0.9
--max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --block-size 16 --level 0
--max-num-seqs 64 --cudagraph-capture-sizes [1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]
--max-model-len 262144 --spec-decode-acceptance-length 3`; the law's engine config holds
`AITER_LOG_LEVEL`, `OMP_NUM_THREADS` and `ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048,3072`, so a
run sets them and passes the acceptance length too. A record that built a Triton variant was
taken again; no record of the law built one.

`scripts/compass/laws/deepseek-v4.1-flash-tp4-ep4-dspark5-mi455-a0.json.xz` is the same law for
cell A, TP4 with whole-expert EP4: `compass fit steps` at ATOM ed5fa7697 and `compass fit host`
at a tree that places `L_mtp` on the drifting host and device clocks
([#3273](https://github.com/jgong5/ATOM/issues/3273)), on one `calibrate.sh` run's records at
ed5fa7697 in the same image ([#2744](https://github.com/jgong5/ATOM/issues/2744)),
under the same environment and the same `SERVER_ARGS` with `--enable-expert-parallel` added. Its engine config
holds `AITER_LOG_LEVEL`, `OMP_NUM_THREADS` and `ATOM_TOKEN_GRAPH_SIZES` at ed5fa7697's
default buckets; a run priced by it sets them and passes the acceptance length too. The
records are concurrency 1, 4, 16, 64 and 256, then 8, 12 and 56 from each trajectory's
first turn: at c=256 all but a few hundred decode steps run at rung 64. No record built a
JIT kernel. From 350 s to 824 s of the c=56 record its MTP status waits returned late, p50 168
to 391 us after the fastest return within 16 waits, while no other host segment moved; the host
laws keep those labels, which lift both `L_mtp` laws by about 0.25% of a c=256 decode forward,
and that record's near-zero labels come from its clock lead, set too high by launch bounds in
that span, the class of [#3133](https://github.com/jgong5/ATOM/issues/3133)
([#3721](https://github.com/jgong5/ATOM/issues/3721)).

`scripts/compass/laws/deepseek-v4.1-flash-tp4-ep4-dspark5-mi355x-gpu40.json.xz` is cell A's law
on mi355-gpu-40, keyed by the platform its `platform` states (aiter `09d256420`,
`aiter_c90842add0.sh` run in the container): the `history` form with host laws, fitted by
`compass fit steps` and `compass fit host` on records at ATOM 1227c0f25 in nightly 20261026,
on cards 4-7, with the [stage-4 smoke](#a-stage-4-cell-smoked-on-mi355x)'s `$ATOM` and
`--enable-expert-parallel`
([#2846](https://github.com/jgong5/ATOM/issues/2846)). The records are four 1800 s
calibration records of traces 256 to 392, at concurrency 1, 4 and 16 and from each
trajectory's first turn at 16, and two 120 s trace runs, the c=16 and c=4 recipes, from
which the prefill vector is fitted. `KINETO_CONFIG` named a file holding
`ROCTRACER_MAX_EVENTS=2000000000`, so no rank's trace dropped rows, and `client.sh` ran under
`WAIT_S=3600` for their stops. Its host laws fit the tokenizer, the segment tokenizer's own
work with `chars_per_s`, the HTTP intake and the hops on those calibration records,
refitted at ATOM ac272852e ([#3291](https://github.com/jgong5/ATOM/issues/3291)): the
records carry no `blocked_ns`, so each `W_launch` label is its launch's wall time. A warm record came
first, and a record that built a Triton variant, or held a decode stall near a server change
on cards 0-3, was taken again.

The law files in `scripts/compass/laws/` are stored xz-compressed, as `NAME.json.xz`, so
that none nears GitHub's file size limits. Every command that takes a law file reads a
name ending in `.xz` decompressed (`atom/compass/backends/json_file.py`), and
`step_linear_fit` writes one so when `--out` or `--law` names one. A law changed outside
a fit, re-keyed or merged, or a refit written as `NAME.json`, is stored again by
`PYTHONPATH=$T python -m atom.compass.backends.step_linear_fit pack scripts/compass/laws`,
which compresses each `*.json` there into `*.json.xz` and removes it. The same document
always packs into the same bytes, so a law that did not change shows no diff. A packed
law holds its top-level values longest first, so a re-key, which edits short ones,
changes only the compressed stream's end, and git stores it as a small delta. `xz -dc
NAME.json.xz`, or `git show REV:PATH | xz -dc`, gives the law's JSON on one line, for a
diff or an edit. A merge that conflicts on a law file is resolved on
that JSON: the base and both sides, `git show :1:PATH`, `:2:PATH` and `:3:PATH` each
through `xz -dc`, are merged as JSON into `NAME.json`, which `pack` then stores. A branch
that changed a law as `NAME.json` merges the same way, the tip's `NAME.json.xz` its other
side.

A simulated run prices by a checked-in law when the law file stands in for both tokens:
`compass run --step-linear LAW --host-laws LAW $M $S --out D -- $ATOM`, `LAW` a file in
`scripts/compass/laws/` (or any law file `step_linear_fit` writes), and `$S` the store that
holds the platform's readings at the stack the file states, and the configuration's
`memory_readings` unless a law file named carries them. A law file carries a store
entry's readings once
`PYTHONPATH=$T python -m atom.compass.run.memory LAW --store S --token T --assumed NOTE`
writes them in, each member stating NOTE, where and at which tree they were taken; a run
whose store holds none at their key reads them, at the `atom` digest of the law's host
key (`atom/compass/run/memory.py`). The run takes the step law at the platform
and engine config the file states, and its `host` at the host key the file states, the
`atom` digest it was last re-keyed at, so a law re-keyed to this tree runs here though its
records ran at another; each refuses where a store entry's would, but host laws serve a run
on any host CPU, which predicts the host they timed. A law file names no fit
inputs, so grading cannot check its fitted traces against the cell's; it takes the run,
since calibration replays traces no cell is graded on (the
[#2698 ruling](https://github.com/jgong5/ATOM/issues/2698#issuecomment-6591251259)).

## A calibration designed from the deployment — `gen_v1.py`, `plan_v1.py`, `dry_v1.sh`, `cal_v1.sh`, `check_v1.py`

The unified calibration procedure v1.1
([LLM-Booster#739](https://github.com/wuhuikx/LLM-Booster/issues/739) sections 2-4): one
step-law calibration record whose workload is derived from the deployment's configuration
and the law's columns, never from a target workload or a concurrency, the same for every
model. A config (`scripts/compass/calib/*.json`, `atom/compass/calib/deploy.py`) states the
deployment (DP ranks, capture rungs, scheduler bounds, a rank's KV pool, drafts and forced
acceptance, the prefix-cache unit), the column set per vector (an input: a column this
tree's backend prices is counted by it, an opt-in one plugs in by name), the hardware noise,
the acceptance criteria frozen before a record, and the segments. Configs: `cellA_v1.json`
(DeepSeek-V4.1-Flash, DPA4-EP4), `cellA_m31.json` (MiniMax-M3.1-Flash, TP4, one DP rank) and
`cellA_k31.json` (Kimi-K3.1, TP8 + DCP8, one DP rank); `cellA_m31.json` states its step-A smoke
record's deployment, and `cellA_k31.json` holds a placeholder KV pool until its smoke record
states one.

```
PYTHONPATH=$T python3 $T/scripts/compass/gen_v1.py CONFIG DESIGN       # traces, design.json
PYTHONPATH=$T python3 $T/scripts/compass/plan_v1.py DESIGN --law LAW --host HOST   # plan.json, plan.md
$T/scripts/compass/dry_v1.sh CONFIG LAW_DIR OUT                         # both, and the verdict
MODEL=... HARNESS_PYTHON=$REAL_PY $T/scripts/compass/cal_v1.sh DESIGN OUT record --gpu 0,1,2,3 --store $ST -- $(args agg)
COMPASS_PREFILL_LABEL=record PYTHONPATH=$T python3 $T/scripts/compass/check_v1.py DESIGN OUT --law LAW --host HOST
```

- `gen_v1.py` writes each segment's lanes as weka traces (`atom/compass/calib/design.py`):
  decode drains whose rows leave on one departure schedule shared by every drain (each rung
  a drain passes holds `rung_steps` steps, 60 by default), a lane pinned to a DP rank by
  `data_parallel_rank` or placed by the router, and prefill quads, backlogs, trickles and
  many-row bursts.
- `plan_v1.py` writes plan.json: the segments in order, each started by an event (every
  lane of the one before has exited and the server holds no request), never by a predicted
  time. With a law it rehearses the design (`atom/compass/calib/engine.py`: a simplified DP
  scheduler, deterministic, seconds of CPU; no prefill cost balancing, no KV eviction),
  prices each step on the step and host laws, keeps the GPU-bound ones (a decode whose
  device outlasts the host's step, a prefill whose device outlasts its launch by
  `PACED_S`: what a fit keeps), and judges each vector (`atom/compass/calib/ident.py`):
  full rank, VIF at most 10, G, the worst relative prediction error over the reachable
  range (`g_z` standard errors of a fitted law's price at the range's vertices under the
  stated noise), at most 1% decode and 2% prefill, and three levels a column. With one DP
  rank `graph_padding` and `row_padding` stay, their check is reported failed and the rung
  is judged by G (#739 2.4 rule 6). It predicts the GPU minutes, the server's start and a
  segment's client start and drain included.
- `dry_v1.sh` runs both on a law directory and exits 1 when a vector fails; given
  `run ... -- SERVE_ARGS`, it also replays the design on a simulated server (`cal_v1.sh` in
  `run` mode) and checks that record, many CPU minutes.
- `cal_v1.sh` records the design on one server: its lanes start together, each a
  `client.sh` run, pinned ones with `--extra-inputs data_parallel_rank:R`, under
  `ATOM_DP_SESSION_AFFINITY=1 ATOM_DP_LB_REQ_EQUIV=1000000000`.
- `check_v1.py` reads the record per segment and lane (`parity.served`), compares each
  segment's step combinations (rung, who holds it, ranks busy; ranks prefilling, tokens,
  rows, history, duty) with the plan's, and repeats the identifiability on the record's own
  GPU-bound steps, with each base form's residuals per segment.
- `plan_v1.py --target RECORD` (repeatable; a real record, or a simulated one, which writes
  the same step lines) also reports each vector's error at the target workload's own
  GPU-bound steps: the mean (target-weighted I criterion), 95th percentile and worst, not
  gated. G and I are variance under the stated form; form bias shows only in the record's
  residuals per segment and in held-out grading.
- A design follows its deployment: `gen_v1.py --record RECORD` (`RECORD=` for `dry_v1.sh`)
  takes the deployment a record ran (`atom/compass/calib/deploy.py` `from_record`: the
  ladder, the scheduler's bounds, the KV pool, the widths, the drafts and acceptance, the
  MoE routing), a deployment block may hold `from_record` with fields that override it, and
  a segment's sizes are relative to the deployment (`deploy.count`, `deploy.tokens`); a
  segment needing more DP ranks than the deployment has is skipped (`min_dp`), and a rung
  the law holds no vector for is rehearsed on its nearest rung's (`engine.proxied`).
  `scripts/compass/calib/README.md` is the procedure for a deployment change, and the
  `compass-calibration` skill points at it.
- A config may `extend` another and `insert` segments into it, and a segment of kind
  `copy` generates another segment again under its own name (new traces, so it hits no
  cache the first left): `cellA_v1x.json` is `cellA_v1.json` with D-v1 repeated mid-record
  and at the end as anchors (a per-record scale factor), and lockstep drains of k = 2, 3, 4
  ranks at one per-rank size plus a 64/32/16/8 mix, for a lockstep term the law does not
  price yet: about 30 GPU minutes rehearsed, against cellA_v1's 25.

On pit2-p03-g08, `cellA_v1.json` rehearsed on the e3v2 baseline law p752b0 (LAW
70714789a2a1, HOST cb0f72ca70f2) in 2 CPU s: 11 segments, about 25 GPU minutes predicted
(the same rehearsal of recipe C's record, `design_trace.py`'s traces, predicts its real
client spans of 115, 865 and 219 s within -5% to +13%), every vector passing: decode
G 0.10-0.58% per rung with VIF at most 2.1, prefill G 1.4% with VIF at most 8.5.

## Cost accuracy on a real run's own steps — `validate.cost`

`PYTHONPATH=$T python -m atom.compass.validate.cost RECORD LAW` prices each step of a real
cell's record (`OUT_DIR/steps` of a `REAL=1 cctraces_sim.sh` run) with a law
file as `step_linear_fit` writes it, and prints the error per kind, per decode rung
and per prefill token band; a record of several DP ranks is joined on step index,
each step priced and banded as its slowest rank, and the error is also given per rank
with its cards' PCI buses where `run.json` names them. The module's docstring states what each figure is.
Grade a law only on cells whose traces its calibration cells do not replay.

## Schedule and end-to-end grades — `validate.runs`

`PYTHONPATH=$T python -m atom.compass.validate.runs SIMULATED REAL...` grades a
simulated `cctraces_sim.sh` cell against two or more spaced real runs of it
(`REAL=1`), each metric against the real runs' own spread. The module's
docstring lists the metrics, their distances and the verdicts.

## A cell graded from every source — `python -m atom.compass.validate`

`PYTHONPATH=$T python -m atom.compass.validate CELL...` joins the commands above and the run
summaries into one table per cell directory: `step_linear/` or `step_grid/`, and two or more
`real*/` runs of one workload. Its docstring lists the rows.

A prefill-decode cell's runs are `deploy.sh` directories: real ones from `deploy.sh
record`, the simulated one from `deploy.sh run`. Each holds `prefill/` and `decode/`
and the client's `artifacts/` beside them, and the table grades it per side: schedule
over every DP rank of that side, cost at the law fitted under that side's engine config,
memory against that side's hand-in (`lp-engine-P.json`, `lp-engine-D.json`); end to
end once, from the deployment's export. The export names decode's response ids, so
prefill's side is split through decode's `transfers.jsonl` (`parity.transfers`).

## Curves and a verdict per point — `python -m atom.compass.validate --curves`

`PYTHONPATH=$T python -m atom.compass.validate --curves OUT ARM...` reads each ARM's
points, a directory per concurrency holding two or more `real*/` runs and a simulated run
per backend, computes InferenceX's agentic metrics from each run's client export, draws
the real and simulated curves on InferenceX's axes as PNG files in OUT, and prints a
verdict per point and metric. The `validate.curves` docstring states the metrics, the
limits and the charts.

## A 1P1D deployment across crsuse2-m2m-v3-005 and -013

`deploy.sh` runs from a GPU container on 005, which holds the prefill instance
and the router; the decode instance runs in a GPU container on 013, started through
`--on`. Both containers run one image, named by digest, with `--device /dev/kfd
--device /dev/dri --device /dev/infiniband --group-add video --ipc host --network
host --cap-add IPC_LOCK --ulimit memlock=-1` and `/shared_nfs/xingjche` and the
model directory mounted at their own paths. Each needs `git config --global --add
safe.directory '*'`: git runs as root there on a tree another user owns, and
without it `compass record` and `compass measure memory` refuse the tree as no git
checkout, and an `atomesh` built there records no commit, which `deploy.sh run`
refuses.

**The prefix.** `spur exec <job> docker exec -e ... <container>`, where `<job>`
is 013's hold job (`squeue -u xingjche -o %i,%j`). It runs from inside the 005
container when that container has `/usr/local/bin/spur`, `/etc/spur/spur.conf`
and `/run/spur` mounted, and `SPUR_CONTROLLER_ADDR` set as on the host. `spur
exec` carries no stdin and returns its command's output only when the command
exits, so `OUT/decode.log` fills once the decode instance has stopped; it
returns the command's exit status. Neither gap matters to `deploy.sh`: the
decode's command reads nothing from stdin, and its stop is a second command
through the same prefix, `kill -TERM <pid>` in the container, whose process ids
`OUT/decode.pid` holds. A file the decode writes, such as its `server.json`,
reaches 005 through `/shared_nfs` seconds late, so set `PREFIX_S=300`. The
one use of the prefix the gaps defeat is the pairing of 013's clock with 005's
(`OUT/clock_pairs.jsonl`, `atom/compass/drive/clock_pair.py`), hundreds of
lines each way over its stdin and stdout, as `ssh` and `srun` carry them: each
pair fails, `clock_pair: 10.245.149.94: ...` on `deploy.sh`'s stderr, after
`PREFIX_S` at most, and the split joins the decode's events to 005's clock by
the bounds its requests set (`atom/compass/report/split.py`). A prefix that
carries both, set as `CLOCK_PAIR_PREFIX_<node>` in `deploy.sh`'s environment,
pairs that node in place of `--on`'s: `<node>` is the address the decode's
`server.json` URL names, each character other than a letter, a digit or `_`
an `_`, so `CLOCK_PAIR_PREFIX_10_245_149_94='ssh -o BatchMode=yes <013>'`
here. It runs `python3 -I -u <this tree>/atom/compass/drive/clock_pair.py
serve` there, so it must reach a host or container that sees this tree at its
path; a container reads its host's CLOCK_MONOTONIC, so either serves.

| | 005 (prefill) | 013 (decode) |
|---|---|---|
| `ATOM_HOST_IP` | `10.245.151.159` | `10.245.149.94`, also the decode's `--host` |
| `ATOM_MOONCAKE_IB_DEVICE` | unset | unset |
| `ATOM_MOONCAKE_MATCHED_RAILS` | `auto` | `auto` |
| `MC_IB_TC` | `96` | `96` |
| `MOONCAKE_DISABLE_HIP_DMABUF` | `0` | `0` |

`ATOM_HOST_IP` is each node's `ens3` address, which the router, Mooncake's
handshake and its notifications reach. With `ATOM_MOONCAKE_IB_DEVICE` unset,
each GPU N registers with `ionic_N`, which carries RoCE v2 on IPv6 only (GID
index 1). These keep the KV on it rather than on Mooncake's TCP fallback
(`Found 0 HCAs` in `server.log`), whose transfers time out at c=256:

- **The host's ionic provider.** The image's `libionic.so.1.0.54.0` does not
  support the host kernel's ionic ABI. Create each container with the host's
  mounted over it: `-v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.2410:/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71:ro`.
- **`MC_IB_TC=96`.** On the default traffic class RoCE between the nodes stops
  with retry-exceeded completions.
- **Matched rails.** `ionic_N` on one node reaches only `ionic_N` on the other,
  and the router pairs any prefill rank with any decode rank: without
  `ATOM_MOONCAKE_MATCHED_RAILS=auto` ([Mooncake matched rails](../../docs/mooncake_matched_rails.md))
  every write from GPU N to a decode rank on another GPU fails with retry
  exceeded, and its transfer times out.
- **dmabuf registration, at `--gpu-memory-utilization 0.75` or below.** With no
  ionic peer-memory module loaded on the hosts, `ibv_reg_mr` on GPU memory fails
  (`ret=-202`), so `MOONCAKE_DISABLE_HIP_DMABUF=0` overrides the image's `1`.
  amdgpu then counts each registered buffer twice against the free memory, as
  allocated and as pinned, after ATOM has sized its KV cache: at `0.85`
  DeepSeek-V4-Pro at TP8 runs out of memory right after registration.

The prefill side's own
variables go in `deploy.sh`'s environment, the decode side's as `-e` words of the
prefix, or each side's in `PREFILL_ENV` or `DECODE_ENV`, which that side's
instance commands set wherever they run:

```
ATOM_HOST_IP=10.245.151.159 PREFIX_S=300 scripts/compass/deploy.sh OUT record --gpu 0,1,2,3,4,5,6,7 \
  --prefill -- <prefill ATOM args> \
  --decode --on "spur exec <job> docker exec -e ATOM_HOST_IP=10.245.149.94 -e <decode env> <container>" \
    --host 10.245.149.94 -- <decode ATOM args> \
  --router -- <router args>
```

A simulated deployment, `deploy.sh run`, runs both instances on 005: each
reaches the clock authority and the write-request block at this node's
127.0.0.1, which 013 cannot. A local prefix, such as `env -u GPU_MAX_HW_QUEUES
-u ATOM_NUMA_BIND`, still starts the decode apart and makes the KV link
`inter_node`, as the real pair spans. The run file then takes the decode side's
hops from the host entry fitted under the environment that prefix gives it, as
the prefill side's under the script's own. Its router arguments switch the router's
wall-clock detectors off, `--disable-retries` among them, as `deploy.sh`'s
header requires. The script
sets `--request-timeout-secs` itself: a stream lasts the wall time the run
takes to finish it, and the router's default 1800 s cut c=256 streams.

### The EP8 cell

The EP8 cell is the DPA-TP8 cell
[as registered](https://github.com/jgong5/ATOM/issues/975#issuecomment-6570682669)
with expert parallelism on. Both sides add `--enable-expert-parallel --moe-backend
standard` to their ATOM arguments and `MORI_SHMEM_HEAP_SIZE=16G` to their
environment, the decode side as an `-e` word of its prefix. MORI's high-throughput
all-to-all is the default; EPLB, MTP and the decode's low-latency mode stay off.
A real run is `deploy.sh record ... --store S`, which merges in the
stack's JIT cache library from `S` and writes it back. A simulated run is
`deploy.sh run --step-linear P --step-linear D --host-laws P --host-laws D
--device mi355x --ptl disabled --network crs005-013-ionic --store S`, which builds the
cell's run file from these flags and both sides' arguments. `P` and `D` are
`scripts/compass/laws/deepseek-v4-pro-dpa8-ep8-mi355x-crs005-prefill.json.xz` and
`-decode.json.xz`, and `S` holds each side's `compass measure memory` readings, taken
at the run's own commit on the node where that side's real instance runs (013 for
the decode): decode readings taken on 005 sized the simulated decode 79 to 80 KV
blocks per rank above every real decode on 013
([ruling](https://github.com/jgong5/ATOM/issues/1111#issuecomment-6659985920)).

### MORI's dispatch and combine at EP8

At EP8 each MoE layer exchanges its rows through two leaves the EP-off graphs lack,
MORI's dispatch and combine. Like every collective, only a `measure ops` run on all
eight GPUs measures them, and a one-GPU run refuses them. Each rank calls them at
MORI's entry point on the handle ATOM builds for the model's first MoE layer, and a
point records each graph it was timed by under `parts_ns` (`_Mori` in
`atom/compass/bench/ops.py`). Trace and measure under the EP8 cell's decode
arguments and environment above, `MORI_SHMEM_HEAP_SIZE=16G` among them, with its
Mooncake role, here `D` and `KV`:

```
compass trace ops --model <DeepSeek-V4-Pro> --tp 8 --arch gfx950 --width 1 --width 8 \
  --width 32 --store S -- D --kv-transfer-config KV
compass measure ops --width 1 --width 8 --width 32 --context 8192 --llc-mib 256 \
  --gpu 0,1,2,3,4,5,6,7 --store S --out O -- D --kv-transfer-config KV --enforce-eager --level 0
```

`measure ops` skips a point its store holds, so each repeat measures into a copy of
the traced store of its own. `fit ops` prices both leaves over their message bytes,
as any collective. The EP8 cells' grade is
[#3409](https://github.com/jgong5/ATOM/issues/3409)'s.

### The inter-node link — `rdma_link.py`

`rdma_link.py` times Mooncake writes from 005's GPUs into 013's over the
rails, as the cell's KV moves under matched rails, and `fit` publishes the
link of the network file `atom/compass/networks/crs005-013-ionic.json`, which
holds the pair's: the port rate and latency from the bench, and the derate a recorded deployment's own KV writes
attained (`atom/compass/parity/writes.py`). Both ends run in a node's GPU
container, created as above, with both nodes' locks and both page caches
checked before and after, as the Rulings on #2794 have every 1P1D run's:

```
# 013: an engine per rail, stopped once the bench has ended
PYTHONPATH=$T MC_IB_TC=96 python3 $T/scripts/compass/rdma_link.py target --ip 10.245.149.94 --out TARGETS
# 005, once TARGETS is there
PYTHONPATH=$T MC_IB_TC=96 python3 $T/scripts/compass/rdma_link.py bench --ip 10.245.151.159 --targets TARGETS --out BENCH
# anywhere atom imports; RECORD, a `deploy.sh record` deployment's directory
PYTHONPATH=$T python3 $T/scripts/compass/rdma_link.py fit BENCH RECORD
```

`fit` also replays the record's writes on that link per rank pair, per
prefill rank and per rail, each one write at a time or sharing the
bandwidth, and prints each one's error on every `_rdma_write` call and on
the contended ones.

### The DPA-TP8 cell under the op-level backend

The DPA-TP8 cell [as registered](https://github.com/jgong5/ATOM/issues/975#issuecomment-6570682669)
runs simulated under the op-level backend by steps 10 to 13 of the stage-2 cell, below,
each side apart: its own ATOM arguments, its environment and its Mooncake
`--kv-transfer-config` (`kv_producer` on the prefill side, `kv_consumer` on the decode).
Without the transfer config ATOM builds DeepSeek-V4's FP4 indexer where a transfer side
runs the FP8 one, so its graphs and points would price kernels the cell never runs. One
store `S` holds the node's `platform_readings`, which a run's `--device` composes its spec
from, each side's `compass measure memory` readings, taken as the EP8 cell's
are, the graphs, the points and one price list: a price list's key holds the model, the
parallel layout, the platform the spec names and the ATOM source, and neither
`--enable-tbo` nor a variable, so both sides' points key one list. Every command runs from one tree, the one
the readings were taken at, since readings, graphs, points and price lists each refuse
another ATOM source, under the side's environment and no other performance variable
(`atom/benchmarks/results/metadata.py::PERFORMANCE_KEYS`): `OMP_NUM_THREADS` is one, so
pin processes that run at once to cores with `taskset`.

- **Graphs.** `compass trace ops --model <DeepSeek-V4-Pro> --tp 8 --arch gfx950
  --population REC --store S -- <side's ATOM args>`, for each plan record and each real run
  graded. Under DP-attention a graph is keyed by the rows its step's MoE gathers over the
  group, `G=`, and under `--enable-tbo` a step the group split is its two micro-batches,
  each traced on one stream and keyed apart from a whole step of its shape, so a real
  run's prefill side holds far more shapes than its decode side, whose steps pad to the
  ladder's rungs. A store refuses a second publish of one key, so trace each record into a
  store of its own, then copy into `S` each entry it lacks, once no other process
  publishes into `S`. For the same reason a trace cut short cannot be resumed into its
  store: split a record whose trace would run for hours, a c=64 or 256 prefill side, by
  step into part records, every rank's line of a step in one part, and trace each part
  into a store of its own; together the parts hold the record's shapes. A trace builds
  the model without its weights, and a graph keys `config.json`'s digest, not its path: a
  container that does not mount the model, such as a CPU gate container, traces from a
  copy of the model's directory less its weight files.
- **Points.** `compass measure ops --plan REC --llc-mib 256 --gpu C --store S --out O --
  <side's ATOM args> --enforce-eager --level 0 --cudagraph-capture-sizes "[R]"`, over each
  side's plan record, `C` the cards the run takes and `R` the decode rungs it measures,
  with that side's Mooncake variables, the node's own `ATOM_HOST_IP`,
  and a `handshake_port` of its own for each bench that runs at once. One GPU runs one DP
  rank and every leaf but the collectives, so one-GPU runs split the decode rungs between
  them, the ladder in no graph's key; one run on all eight GPUs at every rung then
  measures the collectives, the other points already held. The plan reads a split step as
  its two micro-batches, the shapes `trace ops` publishes, and densifies a micro-batch's
  points at its own graph. A bench a signal stops can leave its measuring child holding
  the card ([#3224](https://github.com/jgong5/ATOM/issues/3224)): run each in a session of
  its own, and once it ends, kill by PID what the session left before the card serves again.
- **Price list.** `compass fit ops --store S --out F O...`, `F` the fit's own output
  directory and `O...` every points run's `--out` of both sides.
- **Runs.** `deploy.sh OUT run --op-graph --host-laws P --host-laws D --device mi355x
  --ptl disabled --network crs005-013-ionic --store S`, the decode through a local
  prefix as above, `P` and `D` the tree's
  `laws/deepseek-v4-pro-dpa8-mi355x-crs005-prefill.json.xz` and `-decode.json.xz`, whose
  host laws the run reads. Each engine traces a shape `S` lacks itself, a model build in a
  process of its own, and the run's wall time pays for it. At c=64 and 256 a prefill side
  meets thousands of shapes, so trace the real runs graded into `S` before the run: the
  summary's `cost` counts the graphs it held and those it traced, `held_lookups` and
  `lazy_traces`, and times each, `lookup_wall_seconds` and `lazy_trace_wall_seconds`.
  Such a run is judged faster than real over a store so warmed, its lazy traces reported
  apart ([Ruling](https://github.com/jgong5/ATOM/pull/3124#issuecomment-6768936508)).
- **Grading.** Link `OUT` as `op_graph` beside the real runs and run `python -m
  atom.compass.validate` on that directory: it reads each side's graphs under the
  environment that side's engine file states. It labels a prefill step only from a
  trace run's busy time, so on real runs recorded without a profiler, as the cell's are,
  it leaves every prefill step of the prefill side out as `untraced`, counted per run in
  that side's `left out` row, and that side's prefill has no cost row until a gated
  record grades it ([#3028](https://github.com/jgong5/ATOM/issues/3028)).

### The TP8 cell under the op-level backend

The TP8 cell is the DPA-TP8 cell less `--enable-dp-attention`: each side runs `-tp 8
--trust-remote-code --kv_cache_dtype fp8 --block-size 16 --no-enable_prefix_caching
--max-num-seqs 512` under `AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1`, the prefill side
also `--enable-tbo` under `GPU_MAX_HW_QUEUES=5 ATOM_NUMA_BIND=1`, and the decode side also
`--cudagraph-capture-sizes` of 1, 2, 4 and every multiple of 4 from 8 to 256, the ladder its
host law keys. The prefill side runs at `--gpu-memory-utilization 0.65` and the decode side
at 0.70: prefill ranks ran out of device memory at 0.72, and at 0.70 on a 219,399-token
prompt, and every decode rank did at #975's 0.75
([evidence](https://github.com/jgong5/ATOM/issues/3179#issuecomment-6788714068),
[Ruling](https://github.com/jgong5/ATOM/issues/3179#issuecomment-6789775071),
[Ruling](https://github.com/jgong5/ATOM/issues/3179#issuecomment-6841461259)). A real run is
`deploy.sh record` as above. A simulated run follows the DPA-TP8 cell's Graphs, Points, Price
list, Runs and Grading steps, with these differences:

- **One group of eight ranks.** No graph holds `G=`. A prefill step that `--enable-tbo`
  splits runs as two micro-batches on one DP rank, and `trace ops --population`,
  `measure ops --coverage` and the analytical bound read the split as the DPA cell's.
  The plan's later rounds do not reach most of the prefill points past its measured range,
  so check each real record with `compass run --op-graph --coverage` against the fitted price
  list and measure only the points that alone leave steps out ([Ruling on #3567](https://github.com/jgong5/ATOM/issues/3567#issuecomment-6951471078)):
  `trace ops --prefill T B --whole`, then `measure ops --prefill T B --context C`. Such a run
  names a whole step's graph, whose leaves key the same points its micro-batches' do. A step
  whose `aiter.tbo_all_reduce` the price list declines is refused
  ([#3672](https://github.com/jgong5/ATOM/issues/3672)).
- **The decode side's points at 0.6.** At 0.70 the decode side's `measure ops` runs out of
  device memory on its MoE points ([#3632](https://github.com/jgong5/ATOM/issues/3632)).
  No graph or point key holds the utilization, so the decode side's simulated run stays at 0.70.
- **Host laws only.** `P` and `D` are the tree's
  `laws/deepseek-v4-pro-tp8-mi355x-crs005-prefill-host.json.xz` and `-decode-host.json.xz`,
  host terms with no step linear law: the op-level backend prices every device step. The
  decode side's `http_intake` is the crs005 DPA8 law's, whose fitted body range stops below
  the largest bodies traces 0 to 255 send, so a run of them refuses each larger intake and
  `validate` grades no such run ([#3919](https://github.com/jgong5/ATOM/issues/3919)).

### The EP8 cell under the op-level backend

The EP8 cell runs simulated under the op-level backend as the DPA-TP8 cell above does,
each side under the EP8 cell's arguments and environment, with these differences.

- **Points.** A point's key holds the parallel layout only for a leaf on a process group
  (`point_key` in `atom/compass/bench/ops.py`), so the DPA-TP8 cell's points serve EP8 at
  the leaves both cells run. A store measured before that key changed is republished at
  today's keys by `python3 scripts/compass/migrate_keys.py --machines tests/compass/machines
  --store OLD NEW`. Over `NEW`, `measure ops --plan` measures only what EP8 adds: on one
  GPU the routed MoE over one rank's experts, and on all eight the collectives, MORI's
  dispatch and combine among them.
- **MORI on the prefill side.** A prefill step's shape under DP-attention states its
  group's rows, `G=`, and a MORI exchange there takes about what its largest rank sends,
  not `G`. `measure ops` sends each rank its own share of `G` through MORI's dispatch and
  combine, and `fit ops` prices MORI at a shape stating `G` as a family of its own, over
  EP times the rows the step's largest rank scheduled, which the plan does not densify
  ([#3729](https://github.com/jgong5/ATOM/issues/3729)). Bound the eight-GPU prefill run
  with `timeout`: `fit ops` reads every round a run published and declines the gaps
  beside a point off its neighbours' line. `measure ops --coverage`, as in step 11 of the
  stage-2 cell below, names the steps the price list leaves out; its `--out` is a
  directory holding every run's `plan.jsonl` lines of the side and one run's `run.json`.
- **Runs.** As the DPA-TP8 cell's, with the EP8 cell's host laws,
  `laws/deepseek-v4-pro-dpa8-ep8-mi355x-crs005-prefill.json.xz` and `-decode.json.xz`.
- **Trees.** Every run the cell adds, EP8's at every concurrency and EP off's at c=64 and
  256, runs at the run tree: `c92c36110` with the commits that closed #3688, #3729 and
  #3657, the last pricing a step again only at the leaves whose reads moved. The replay
  tree is the run tree without #3657's commit
  ([Ruling](https://github.com/jgong5/ATOM/issues/3409#issuecomment-6999401243)). A side's
  `steps/dp<r>.cost.jsonl` holds the terms each line of its `steps/dp<r>.jsonl` was priced
  at; a replay prices those lines again with the run's price list and graphs through the
  replay tree's `OpGraphBackend` and compares each term's name, seconds, method, validated
  flag and source. A line that differs voids the run, and its cell runs at the replay
  tree.
- **Ranking.** EP8 is ordered against the DPA-TP8 cell only as both are simulated at one
  tree (the ranking gate of `atom/compass/design/08_validation_protocol.md`), the run tree.
  The DPA-TP8 cell's runs at c=1, 4 and 16, simulated at `c92c36110`, count as run there
  when every line of both sides replays through the run tree alike; a line that differs
  is escalated, and the ranking waits for its ruling.

## The analytical bound — `python -m atom.compass.validate.bound`

`PYTHONPATH=$T python -m atom.compass.validate.bound ... RECORD...` exits 1 when a real
record's step is priced above its measured device time; `compass project` prices real
records on another device. Each module's docstring states its arguments and report.

## The stage-2 cell through the `compass` CLI, from an empty store

Qwen3.8-27B TP1 on MI308X, priced by a step linear law `compass fit steps` fits on
this node, in two checks: check 1, that `cctraces_sim.sh` handed the run file a
`compass run` kept schedules the same steps; check 2, the cell graded against real
runs. The cell runs at concurrency 1, 4 and 16 only, in calibration and grading:
at 64 and 256 aiperf's own warmup of this workload cannot finish within a
capture's window, so the owner dropped them for this model and machine as not a
sustainable configuration
([ruling on #1590](https://github.com/jgong5/ATOM/issues/1590#issuecomment-6281930268)).
Run everything from one tree, `$T` below, in the GPU container: host laws are
keyed by ATOM's source, so records, fits and runs from two trees refuse each other.
In the image, `compass` is `$T/scripts/compass/compass`, linked anywhere, and `python -m`
runs with `PYTHONPATH=$T`, both from `/` so every path is absolute. Below, `$ATOM` is
`--model <Qwen3.8-27B> --max-model-len 262144`, `$M` is `--device` and `--ptl`, this node's platform
([QUICKSTART.md](QUICKSTART.md#environment) says which), `$R` is a directory under `/data`
or `/tmp` and `$S` is `--store $R/store`, holding only the platform's readings at the start. The raw export keeps
no prompt: aiperf leaves each request's payload out of it, and the client
gzips it once the client run's export is written.

**A client run.** No command starts one: once `OUT/server.json` exists,
`$T/scripts/compass/client.sh OUT --traces A-B --concurrency C --stop`, `client.sh`
below, drives one against the server whose `--out` is `OUT`, writes its export,
`OUT/artifacts/client_run.json`, and stops the command, which stops its server.
Its header lists its options and the variables it reads: a `run` server's client
is the Python `cctraces_sim.sh` takes as `HARNESS_PYTHON`, a `record` server's the
image's aiperf. Both sides run the scenario's own warmup, as `calib_capture.sh`
does: aiperf starts each trajectory at a sampled point and first sends, for
each session active there, its last request before that point, so the cold
first forward and the prefix cache's fill fall outside the graded window.
A cell that runs the nightly's client arguments passes them in `AIPERF_EXTRA_ARGS`.
The grace period after the duration stays aiperf's finite default:
`--benchmark-grace-period inf` can hang aiperf's agentic replay after the cutoff,
waiting on a credit the server no longer holds (upstream, in agentx-harness's
aiperf).
Until #1301, a real server's environment sets `TRITON_CACHE_AUTOTUNING=1` and one
`TRITON_CACHE_DIR` for every run, as `cctraces_sim.sh` does for a real cell. aiter
builds a JIT module the first time a server meets a kernel variant its image did
not prebuild, such as `mha_varlen_fwd`'s for a prefill step of one new token a
row, about 60 s inside the run, and keeps it in the container's aiter tree, which
no cache directory above holds. A record names each module its server built, or
began to, in run.json's `aiter_jit_builds`, and grading, `compass fit steps`,
`compass fit host`, `step_linear_fit` and `validate.cost` refuse it, naming them.
So sweep a fresh container before its first record, on a server whose record
nothing reads: `compass record --sweep --gpu G $S --out $R/sweep1 -- $ATOM` with
`client.sh $R/sweep1 --sweep --traces A-B --stop`, A to B every trace the cell's
records serve. `--sweep` adds the pacing variables the sweep's steps need
(`sweep.pacing` in `atom/compass/drive/sweep.py`): ATOM's PrefillDelayer
coalesces each step, and DP prefill balancing cuts no row; `client.sh --sweep`
refuses a server started without them. The sweep sends every prefill token count
up to the budget once, one count to each DP rank a round, then lockstep steps of
designed rows queued behind parked decodes (`atom/compass/drive/sweep_plan.py`),
and after the stop keeps the points the step record covers in the store's JIT
cache library, per stack and cell. Each round sends only the steps still
missing; take rounds with a fresh `--out` until its last line reads `swept`. A
third round's missing points, `missing` in its `sweep.json`, are the residual to
report with the cell: the sweep never constructs preemption, recompute or an
error path. Under pipeline parallelism or RapidServe it sends the token counts
alone and never marks the cell swept. A 1P1D cell's sweep runs on its prefill
side, with that side's own ATOM arguments; its decode side and KV connector path
are warmed by one short run of the whole cell after it, whose records nothing
grades and whose stops write back to the library as any record's does. Take
again any record refused for a build.
What a container compiled need not be compiled again in the next one on the same
stack: `compass record` with `$S`, and a `REAL=1 cctraces_sim.sh` run with `STORE`
set, keep a JIT cache library per stack under the store's `jit` directory
(`atom/compass/drive/jit_cache.py`). The server's start merges it into the
container's caches, replacing no file there, and its stop writes back each artifact
the server finished compiling; `compass measure memory` does the same with its
store. A record without a store touches no library.

1. `compass measure memory --gpu G $S --out $R/mem -- $ATOM`.
2. Calibration, alone on the node: per concurrency 1, 4 and 16,
   `compass record --gpu G --out $R/cal/c<C> -- $ATOM` and `client.sh $R/cal/c<C>
   --traces 256-392 --concurrency <C> --stop`, each server started with
   `ROCP_TOOL_LIBRARIES` naming the tool `python -m atom.compass.parity.blocked
   $R/tool` prints, so its launches' `W_launch` labels leave out their time in
   HIP's blocking calls (`atom/compass/parity/README.md`); `compass fit host`
   names a record taken without it. Then the trace runs (below), which take no
   tool: per concurrency 16 and 4, `compass record --trace --gpu G --out
   $R/trace/c<C> -- $ATOM` and `client.sh $R/trace/c<C> --traces 256-392
   --concurrency <C> --duration 120 --stop`.
3. `compass fit steps --backend step-linear --form history --model $MODEL $R/cal/c*
   $R/trace/c* --client-run $R/cal/c<C> $R/cal/c<C>/artifacts/client_run.json ... $S
   --out $R/fit`, every record and trace run named before the first `--client-run`,
   `--model` naming the model directory, whose config gives a MoE model's routing, and
   `compass fit host` on the calibration records alone; each prints its store token.
   Steps 1 to 3, after a warm record, are `calibrate.sh $R/cal 1 4 16 "trace 16" "trace 4"` with
   `MODEL`, `GPU`, `TRACES` and `AIPERF_PYTHON` set, its store the one `STORE`
   names, else `$R/cal/store`; it reads no platform, so it prints the tokens, the law's held-out
   errors per band and the `compass run` line of step 5 with `$M` left as `--device
   DATASHEET --ptl PTL`. `$R/cal/store` holds no `platform_readings` entry, and step 5's
   `compass run` refuses a store with none: publish one into it with
   `compass measure device` and `compass fit device`, as the platform's bullet in
   [QUICKSTART.md](QUICKSTART.md#environment) does, before step 5, or name a
   store that holds one as `STORE`. A RECORD after its
   concurrency names the client's scenario, such as `"8 --trajectory-start-min-ratio 0
   --trajectory-start-max-ratio 0"`, the first-turn record that gives the law
   its decode rung 8; `trace` before one makes it a trace run of that recipe.
4. Real runs: per concurrency, two `compass record --gpu G --out
   $R/cell/c<C>/real<k> -- $ATOM`, each with `client.sh $R/cell/c<C>/real<k>
   --traces 0-255 --concurrency <C> --stop`, at most two GPUs at once.
5. `compass run --step-linear LAW --host-laws HOST $M $S --out
   $R/cell/c<C>/step_linear -- $ATOM`, with `client.sh $R/cell/c<C>/step_linear
   --traces 0-255 --concurrency <C> --stop`.

**The trace runs.** Only a trace run labels a prefill step: without one, `compass fit
steps` names `prefill` under the law's `unfitted`, as `no prefill step carries a trace
run's busy time`, and step 5 refuses every prefill step. The prefill vector needs more
steps than coefficients, spread across them, or stays unfitted with the reason named
(`step_linear_fit.fit`); its `fit.samples` is how many it took. Its held-out band needs
a second trace run, and one of the two to fit the vector alone, since the fit prices
each run by the law fitted on the others; with one trace run, every prefill term prices
unvalidated. Leave the trace runs out of `compass fit host`: their host segments ran
under the profiler. A higher concurrency holds more prefill steps in the same time.
Under a speculative drafter, `compass fit steps --ops DIR...` takes the profiler's
floor, the `device.profiler_floor_s` of the store's readings at its points' platform, out of the trace
runs' drafter labels, each kernel's own length from `compass measure ops --drafter` at
the trace runs' step shapes, traced first by `compass trace ops`; the fit's report
names each kernel no point gave its own length.
`compass record --trace` raises kineto's row cap, 5,000,000 rows a process by default
([#3106](https://github.com/jgong5/ATOM/issues/3106)), and a run that lost device rows
all the same is refused at its stop, naming each rank's first forward without them:
take it again. Its stop writes and reduces every row, within `client.sh`'s `WAIT_S` plus
serve's `TRACE_STOP_S`, 4200 s by default, so keep a trace run short enough to stop
inside that.

Developer checks, outside the `compass` path:
- After step 2: `python -m atom.compass.parity.recurrence $R/cal/c<C>/steps`.
- Check 2: `python -m atom.compass.validate $R/cell/c<C>` after step 5, `$R/cell-grid/c<C>`
  after step 9, whose bundle it reads from the store the run file names, by key and digest.
- Check 1: `cctraces_sim.sh` with `RUN_FILE` the `run_file` of
   `$R/cell/c<C>/step_linear/run.json`, `FIRST_SESSION=0 SESSIONS=256
   CONCURRENCY=<C>`, `SERVER_ARGS="--max-model-len 262144"` and no
   `AIPERF_EXTRA_ARGS`; then `python -m atom.compass.parity
   $R/cell/c<C>/step_linear/steps OUT/steps`.

### The same cell under the step grid

The step grid reuses steps 1 to 4: the calibration records are its residual's
targets and, through the ragged views step 7 composes from them, alpha's; the
real runs are its coverage check's inputs and its graded side.
Its points are planned from the shape populations and step 4's real runs, so
every cell a real step falls in is measured: a population alone leaves the cells
of shapes only the real server scheduled unmeasured (#1884). The plan reads
shapes only, so its inputs may serve any traces, the graded ones included; a
point's time is the bench's alone. A real start's KV block count moves with
its free memory, so the grid is planned for the smallest count any input
states. The grid definition is a calibration input; `$G` below holds nodes at
powers of two and no margin:

```json
{"prefill": {"B": [1, 2, 4, ..., 256], "T": [1, 2, 4, ..., 16384],
             "c": [0, 1, 2, 4, ..., 131072]},
 "decode": {"c": [0, 1, 2, 4, ..., 131072]}, "margin": 0}
```

6. Per concurrency, `compass trace steps --step-linear LAW --host-laws HOST $M $S
   --out $R/pop/<traces>/c<C> -- $ATOM`, with `client.sh $R/pop/<traces>/c<C>
   --traces <traces> --concurrency <C> --stop`, `<traces>` 256-392 and 0-255; its
   stop publishes a `shape_population`, `POP` below.
7. `compass measure steps --grid POP... $R/cell/c*/real* --grid-definition $G
   --client-run POP $R/pop/<traces>/c<C>/artifacts/client_run.json --client-run
   <real run> <real run>/artifacts/client_run.json ... --sampling '{"ignore_eos":
   true}' --gpu G $S --out $R/grid -- $ATOM`, naming every population and every
   real run. The deep-context prefill corners take most of its hours. Several
   GPUs each take a subset of the inputs into a store of its own, starts a minute
   apart; copying each store's `step_measurement` entries into one, and this step
   again over every input, measures what remains.
   Then `compass measure steps $R/cal/c* --sample 6 --seed N --ragged --grid POP...
   $R/cell/c*/real* --grid-definition $G --client-run $R/cal/c<C>
   $R/cal/c<C>/artifacts/client_run.json ... --sampling '{"ignore_eos": true}'
   --gpu G $S --out $R/ragged -- $ATOM`, naming every record, population and real
   run: it draws six decode steps a rung and measures the ragged views composed
   from each, and the cells the planner places for them, each view's longest
   row's among them, which alpha reads; the grid's other points are held. Alpha
   is fitted on these views alone. Each view measured prints `measured ragged
   decode rung <R> <token>: ...`; keep every such token, since a rerun prints
   only the views the store lacked.
8. `compass fit steps --backend step-grid --grid POP... $R/cell/c*/real*
    --grid-definition $G $R/cal/c* RAGGED... --client-run ...`, naming every
    population, real run and record, and as `RAGGED` every view token step 7's
    `--ragged` run printed, `$S --out $R/fit-grid`; it plans the views' cells
    again and refuses naming any not measured. It prints the bundle's
    token, and its report's `alpha` states per rung whether alpha was kept. A
    second fit of one definition into one store is refused as an overwrite:
    refit into a fresh store.
9. `compass run --step-grid BUNDLE --host-laws HOST --coverage $R/cell/c<C>/real*
    $M $S --out $R/cell-grid/c<C>/step_grid -- $ATOM`, serving the same client run as
    step 5. The check counts the real steps the bundle covers, extrapolates past
    the residual's hull or the KV pool, and leaves to the step roofline; grading
    takes only a run that ran it.
    Link `$R/cell/c<C>/real*` into `$R/cell-grid/c<C>`: grading refuses a cell holding
    both `step_linear/` and `step_grid/`.

### The same cell under the op-level backend

The op-level backend reuses steps 1 to 3 and step 6's populations, and is graded on
the same real runs as step 5: step 4's, or `REAL=1 cctraces_sim.sh` runs of traces 0 to
255 at the same concurrency and seed, which `client.sh` sends the same requests as.
Its graphs, points and price list are keyed by the environment as well as ATOM's
arguments, `AITER_LOG_LEVEL` among the variables, so trace, measure, run and grade in
one: grading in another refuses every real step's graph.

10. `compass trace ops --model <Qwen3.8-27B> --population $R/pop/<traces>/c<C>...
    --population <real run>... $S -- --max-model-len 262144`, naming every population
    of step 6 and every real run graded: an `op_graph` per shape their steps hold,
    traced with no device. `run` traces a shape no graph holds itself, a model build
    each, which its wall time pays; `run --coverage` and grading trace none, and
    leave a real step whose shape no graph holds out of their counts. A graph
    traced from another ATOM source or torch refuses where `run` reads it, and
    its shape is not traced again, since its key names neither.
11. `compass measure ops --plan $R/pop/256-392/c<C>... --llc-mib 256 --gpu G $S --out
    $R/ops -- $ATOM --enforce-eager --level 0`: the op bench's planner over the
    calibration traces' populations, in rounds, on one card. The bench runs eager and
    refuses a compiled level; neither flag enters a graph's or a point's key.
    The plan stops when a round finds nothing left to measure, hours in on this
    cell; run it under `timeout`, and run a stopped one again into the same `--out`,
    which keeps the rounds already there.
    `compass measure ops --coverage <real run>... $S --out $R/ops -- <the same
    arguments>` prints how the plan's points cover the real steps, by phase, and
    why each leaf kind left steps out.
12. `compass fit ops $S --out $R/fit-ops $R/ops` publishes the `price_list`, naming
    every `--out` that measured a point.
13. `compass run --op-graph --host-laws HOST --coverage <real run>... $M $S --out
    $R/cell-ops/c<C>/op_graph -- $ATOM`, serving the same client run as step 5. The
    check counts the real steps whose every leaf the price list prices; a leaf it
    declines is priced by its analytical law, not refused. Link the real runs into
    `$R/cell-ops/c<C>` and grade with `python -m atom.compass.validate
    $R/cell-ops/c<C>`, whose cost rows add, per phase, the priced seconds by method,
    each op kind's composed seconds over the measured ones, and warm over cold per
    kind. Beside `cctraces_sim.sh` real runs, which grading takes whole, the `run`
    cell is graded whole too, with the request aiperf cancelled at its cutoff: the
    server steps it alone until `client.sh --stop` stops the server, and grading
    takes those steps only from an export that `--stop` marked `server_stopped`.

### A stage-4 cell smoked on MI355X

The DeepSeek-V4.1-Flash cells of [#2138](https://github.com/jgong5/ATOM/issues/2138),
TP4 with DSpark5, are smoked on mi355-gpu-40 by steps 1 to 5 and check 2, steps 4
and 5 at concurrency 16 alone, with these changes:

- The image is `rocm/atom-dev:vllm-v0.31.0-nightly_20261026`, whose aiter the tree
  serves V4.1-Flash on, with `aiter_c90842add0.sh` run in each container before its
  first server, and `$M` is `--device mi355x --ptl disabled`, the store holding
  gpu-40's readings at that stack.
- `$ATOM` is `--model <DeepSeek-V4.1-Flash> --max-model-len 262144 -tp 4
  --kv_cache_dtype bf16 --index-cache-dtype fp8 --method dspark
  --num-speculative-tokens 5 --spec-decode-acceptance-length 3 --level 0`, plus
  `--enable-expert-parallel` for the EP4 cell, and `G` names four GPUs. Every record and
  every run states the acceptance length, 3, near the 3.13 the smoke measured; another
  length is another value of the flag, the same in every record and run
  ([ruling](https://github.com/jgong5/ATOM/pull/2683#issuecomment-6578751421)).
- The warm record serves traces 0 to 255 at 16. V4.1's Triton kernels still compile
  new variants after it, so take again every record that names `aiter_jit_builds`.
- Calibration adds a first-turn record at 16, its `client.sh` run with
  `AIPERF_EXTRA_ARGS="--trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0"`:
  sampled starts leave rung 6 and above too few steps for a vector, and a step at a
  rung with none is refused.
- The trace runs stay at 120 s for their stop's cost: the EP4 cell's passed the default
  row cap 41 to 49 s in and logged 14.3M to 14.7M rows a rank, and each stop, SIGTERM
  to exit, took about 11 min: c=4's 671 s for 121 s traced, about 46 us a row a rank
  ([#3113](https://github.com/jgong5/ATOM/pull/3113#issuecomment-6759346059),
  [#3111](https://github.com/jgong5/ATOM/pull/3111)). At c=4's rate the default 4200 s
  covers about 750 s traced. In their first 120 s, #2789's cell-B calibration records
  at `aeffa7d63` hold 30 prefill steps at c=4 and 125 at c=16.
- Steps 4 and 5 take `--duration 3600`, with a calibration record between step 4's
  two real runs. A step 5 at another duration than step 4's is graded as another
  workload, and refused.
- The draft vectors hold no held-out error: check 2's `cost` rows of kind `draft` give
  the draft law's error on the real runs' own steps.

### A stage-4 cell graded on MI455

[#2147](https://github.com/jgong5/ATOM/issues/2147) grades cell A on one MI455 A0 node by steps
4 and 5 at concurrency 1, 4, 16 and 64, every run from one tree, with these changes from the
smoke above:

- `$ATOM`, its environment and the container are those
  `scripts/compass/laws/deepseek-v4.1-flash-tp4-ep4-dspark5-mi455-a0.json.xz` states (its
  paragraph above), and `$M` is `--device mi455x-a0 --ptl disabled`, the store holding the
  node's readings at that stack.
- Step 5 prices by the law file, `--step-linear LAW --host-laws LAW`, with the
  `memory_readings` that `compass measure memory` takes at the same tree and settings.
- Steps 4 and 5 take `--duration 3600`, one server on the node at a time. Step 4 runs three
  spaced rounds at c=1, 4, 16 and 64: the first ran c=1, 4 and 64, then 16, as the interim
  check ordered them; the second and third ran 1, 4, 16, 64.
- `python -m atom.compass.validate` grades each concurrency's simulated run against its three
  real runs, with throughput and TPOT targets of 10% and TTFT's of 15%, beside the real spread.
- A real run has hung when, for 90 s while requests run or wait, its log shows neither a step
  line nor a load report with nonzero prompt or generation throughput
  ([#2922](https://github.com/jgong5/ATOM/issues/2922)). Check the node after the GPU reset
  that follows: one such reset left RCCL's all-reduces above aiter's 64 MiB custom
  all-reduce cutoff slow until the node was rebooted, every prefill chunk over 6553 tokens
  37 to 60% slower on the device.
- `python -m atom.compass.memory.check --tp 4` checks each rank's non-KV memory against the
  real run's log; a concurrency's memory is graded on the mean of its 12 readings, three runs
  of four ranks. The `non_torch` term, the driver and collective reserve, moves from start to
  start: one rank alone read +10.43% against the 10% gate twice, beside its start's 10.3%
  spread across ranks, while the mean stayed within 2.3%.

[#2148](https://github.com/jgong5/ATOM/issues/2148) grades cell B alike, by
`scripts/compass/laws/deepseek-v4.1-flash-tp4-dspark5-mi455-a0.json.xz` (its paragraph above),
with two changes:

- Each round runs c=1, 4, 16, 64, after an unscored 600 s warm run at c=64 and one at c=1.
- Since [#2086](https://github.com/jgong5/ATOM/pull/2086) a prefill step's label is a trace
  run's busy time, so `validate.cost` leaves a plain real run's prefill steps out as
  `untraced`; cell B's prefill cost is read by `validate.cost` at `ed5fa7697`, the step law's
  fit tree, which labels a prefill step by its device window.
