<!-- SPDX-License-Identifier: MIT -->
# Compass advice: from a space file to a recommended configuration

This page takes a user from the configurations they would deploy to the one Compass
recommends. You write them into a space file, run `compass advice SPACE OUT`, start
the calibrations it prints for what the store lacks, and run it again until every
point is simulated. Each step below is one command and the output it prints. A
person and an agent follow the same steps; an agent also keeps the rules in
[With an agent](#with-an-agent).

[QUICKSTART.md](QUICKSTART.md) is the developer's path for one configuration:
records, fits, real runs and grading. This page uses its container, variables and
platform readings, and never grades. [USER_QUICKSTART.md](USER_QUICKSTART.md)
predicts one configuration on a host with no GPU at all.

## Limits

- **A GPU container.** The search uses no GPU compute, but a simulated server
  imports aiter, which needs a ROCm driver. Run it in the GPU container of
  QUICKSTART's [Environment](QUICKSTART.md#environment).
- **The target machine type.** Host laws are keyed on the host's CPU, so the search
  runs on a node of the type it predicts, and another CPU is refused by name. A
  calibration runs there too, on the GPUs the configuration uses.
- **The integration head.** The tree runs at the tip of
  `origin/feature/atomcompass_new` as last fetched, or at the commit the space file
  freezes. A tree whose ATOM digest (`atom/compass/run/__init__.py::atom_digest`)
  differs from that commit's is refused, both digests named. A change outside
  ATOM's Python, such as a Compass task's own, leaves the digest alone and passes.

## Before the first search

Set up QUICKSTART's [Environment](QUICKSTART.md#environment) and
[Variables](QUICKSTART.md#variables), with these differences.

- **The tree is a clone, not a snapshot.** The check reads the head from git, so
  `$T` is a clone of `https://github.com/jgong5/ATOM` checked out at the head, and
  the container mounts it at its own path. A `snapshot.sh` tree is refused: `the
  tree atom resolves under, ..., is not a git repository's root`. Before each
  search:

  ```bash
  # host, in the clone
  git fetch origin
  git checkout --detach origin/feature/atomcompass_new   # or the commit SPACE freezes
  ```

  Once, in the container, since git there runs as root on your files:

  ```bash
  git config --global --add safe.directory "$T"
  ```

- **The variables a search and a calibration read**, in every shell that runs one:

  ```bash
  export HARNESS_PYTHON=$SIM_PY    # each simulated point's client; unset, no point runs
  export AIPERF_PYTHON=$REAL_PY    # a calibration's real clients
  export AITER_LOG_LEVEL=WARNING   # as QUICKSTART: a law's key holds the environment
  compass() { $T/scripts/compass/compass "$@"; }
  cd /
  ```

A search reads the platform readings of the target's cards (QUICKSTART's
Environment, "The platform") from the store the space file names. Where that
store holds none at SPACE's machine, it reads the copy committed for that
machine under [laws/platform_readings/](laws/platform_readings/), and looks
every law and readings entry up at the copy's stack, the `software_stack` and
`torch` of its `key`; with neither, every candidate is refused with the store's
reason. A calibration publishes at the stack it ran at, so a store holding no
platform readings serves what it published only when that stack is the copy's:
unless the target runs that stack, publish the target's readings into the shared
store, as QUICKSTART's Environment does, before the first calibration.

## 1. Fill the space file

A space file is one JSON object. Its fields, their values and the expansion are
[19_adoption.md, D110](../../atom/compass/design/19_adoption.md#d110-the-space-file);
read it first. The templates under [spaces/](spaces/) show the shape, and no
command reads one unless you name it as SPACE.

Take each field from what the user asks for:

| Field | From |
|---|---|
| `model` | the model directory the user serves; the recipe's `--model` names one too |
| `machine` | `device`, the target's datasheet under `atom/compass/devices/` (`mi355x`); `ptl`, what `cat /sys/bus/pci/devices/*/ptl/ptl_enable` prints on the target, `disabled` where it finds no file; `network`, the name of a file under `atom/compass/networks/` without `.json`, only when a 1P1D candidate puts its sides on two nodes; `stack`, left out until a search refuses, listing the store's several readings, then the one the user runs |
| `store` | the shared store, an absolute path |
| `commit` | only when the user freezes the search at one commit |
| `workload` | the traces file's absolute path; `select`, the traces (`0-255`) or `agentx`; each concurrency the user asks for, whatever band a recipe runs its variant at; `duration_s` only when the user names one |
| `slo` | the user's bounds; a bound they do not state is left out, and bounds nothing |

Then the configurations. One dimension whose values are whole configurations, as
the DeepSeek-V4-Pro template's `config`, fits any list of configurations. Several
dimensions, such as `tp` and `mtp`, fit a product the user asks for, with `exclude`
removing the combinations they do not want.

### From a recipe's server command to a block

Each configuration has a basis: a recipe and the variant in it, or the user's own
real-run command. Copy its server command into blocks by these rules.

- **`basis`** names the recipe's path and its variant, in any words that tell it
  from the recipe's others, such as
  `recipes/DeepSeek-V4-Agentic-InferenceX.md, DP attention`; for a user's command,
  where it came from.
- **What every candidate shares goes into `common`**, the rest into the value's
  block. Each argument or variable is set in one block only: two blocks that set
  one thing to different values refuse the candidate.
- **`--model`** is the `model` field, never an argument, and the recipe's own
  path takes no departure.
- **An argument is a long flag** mapped to its value: `--level 3` is
  `"--level": 3`, a number written as a number; a bare flag is `true`, such as
  `"--enable-dp-attention": true`, and its negation keeps the `--no-` spelling,
  `"--no-enable_prefix_caching": true`. A short flag takes its long name from
  `atom/model_engine/arg_utils.py`: `-tp` is `--tensor-parallel-size`, `-pp`
  `--pipeline-parallel-size`, `-dp` `--data-parallel-size`. The parser reads
  `--a_b` as `--a-b`, so either spelling of the recipe's serves, once.
- **A value the recipe computes from the concurrency** goes into
  `per_concurrency` as the factor: `--max-num-seqs $(( CONC * 2 ))` is
  `"per_concurrency": {"--max-num-seqs": 2}`.
- **Each `export` before the server** is `env`, its value a string.
- **A 1P1D recipe** gives the candidate `"deployment": "1p1d"`. What both sides'
  commands share goes into the block's own `args` and `env`, what one side alone
  takes into its `prefill` or `decode` block.
- **A variant that adds to another**, as the single-node recipe's LMCache variant
  adds variables to the DP attention one, is written whole in its own value.
- **An LMCache server's command** (`lmcache server ...`) goes into the block's
  `lmcache_server`, each flag written as an argument is, but `--host` and
  `--port`: `compass advice` gives each point its own server address.
- **A Mooncake Store**, on a configuration whose prefill offloads to one
  (`ATOM_KV_OFFLOAD=mooncake_store`), is the block's `mooncake_store`: the
  absolute path of the file holding the `mooncake_store` entry that `compass run
  --mooncake-store` takes, with the master's flags, the owners' bytes and the
  laws pricing the Store's lookups and copies. Every point of the candidate runs
  with that file, each deployment starting its own Store at a port it draws, so
  the block gives the Store no address.
- **Every other argument and variable** of the server command is kept as the
  recipe gives it; only the next two sections take any out or change one.

### What a space file leaves out

A recipe's command also holds what `compass run`, `deploy.sh` and `client.sh`
set themselves. Leave each of these out. Where it is an argument or a variable of
the server, write it into the block's `departures` with this reason, in `common`
when every candidate's recipe sets it:

| Left out | Reason to write |
|---|---|
| `--host`, `--server-port` | unset: compass run and compass record assign it and refuse it among ATOM's arguments |
| `--served-model-name`, when the recipe sets it to the model path | unset: ATOM then serves the model under its --model path, the value the recipe gives |
| `--kv-transfer-config` | unset: deploy.sh adds each side's and refuses one among ATOM's arguments |
| `HIP_VISIBLE_DEVICES` | unset: a real side's cards are deploy.sh's --gpu, and a simulated side uses none |
| `ATOM_HOST_IP` | unset: a node's own address, which a real side takes from scripts/compass/README.md's 1P1D deployment |
| `MC_ENABLE_DEST_DEVICE_AFFINITY` | unset: a real side on crsuse2-m2m-v3-005 and -013 takes scripts/compass/README.md's Mooncake variables instead |
| `ATOM_PD_RANK_MAPPING_POLICY` | unset: CI's router script reads it, and ATOM does not |
| `AITER_LOG_LEVEL`, on a 1P1D side | unset: deploy.sh sets WARNING itself |
| `PYTHONUNBUFFERED` | unset: it orders log lines, not the engine's work |
| `--block-size`, on a DeepSeek-V4 checkpoint | unset: ATOM sets 256 on any DeepSeek-V4 checkpoint, so the recipe's value has no effect |
| `ATOM_KV_OFFLOAD_EXTRA_CONFIG`, when it names only the LMCache server's host and port | unset: compass advice sets it to each point's own lmcache server, whose host and port it assigns |
| the router's `--policy` | unset, not the recipe's: compass advice starts the router with its own flags, the same for every candidate, and with one worker a side no policy chooses |

Some things a recipe sets are no instance's and take no departure: the rest of
the router's command (`deploy.sh` starts it, with `compass advice`'s flags), the
client's with its `AIPERF_*` variables (`client.sh` runs it from `workload`), a
variable ATOM does not read that the recipe sets only to build another value, as
`NODE_IP` builds `ATOM_HOST_IP` and `--kv-transfer-config`, the Mooncake
variables the README's 1P1D deployment gives a real side, and the commands that
start a Mooncake Store's master and owners (the entry holds their flags). Keep
each other variable the recipe gives the server for LMCache, such as
`ATOM_KV_OFFLOAD`, as it gives it, and any key of `ATOM_KV_OFFLOAD_EXTRA_CONFIG`
but `lmcache.mp.host` and `lmcache.mp.port`. Keep the variables the recipe gives
a side for a Mooncake Store, `ATOM_KV_OFFLOAD` and its
`ATOM_KV_OFFLOAD_EXTRA_CONFIG`, as it gives them: a start refuses a protocol,
chunk size or pool size there other than the entry's.

### What the user decides

Some values no recipe gives: one value where the recipe varies it with the
concurrency, a switch turned the other way, a memory fraction the target needs.
Take each from the user, write it in its block, and give it a departure with the
user's reason. A departure is keyed by the argument in either spelling, `--a_b`
or `--a-b`, `--no-x` and `--x` being one. It sits in the block of each candidate
whose recipe it departs from, even when the value sits in `common`, as S1's
`--max-num-seqs` does below, and never in a side's: a value that differs by side
states each side's in one reason. Without the user's value, keep the recipe's
and the decision open: an agent asks, and never picks one. A configuration that
does not start on the target, such as one out of memory, comes back from step 2
refused, with the start's error.

### Example: S1 from its recipe

The DeepSeek-V4-Pro template's S1 is the TP server of
`recipes/DeepSeek-V4-Agentic-InferenceX.md`, with one decision of the user's:
`--max-num-seqs` 512, not twice the concurrency, so each configuration takes one
memory-readings entry. Its two variables, `AITER_BF16_FP8_MOE_BOUND` and
`ATOM_MOE_GU_ITLV`, and its `-tp 8`, `--kv_cache_dtype fp8` and
`--enable_prefix_caching` are every configuration's, so they sit in `common` with
`--max-num-seqs`, and S1's block holds the rest:

```json
"S1": {
  "basis": "recipes/DeepSeek-V4-Agentic-InferenceX.md, TP",
  "args": {
    "--index_cache_dtype": "fp4", "--method": "mtp", "--num-speculative-tokens": 3,
    "--spec-decode-acceptance-length": 2.49, "--gpu-memory-utilization": 0.9,
    "--max-num-batched-tokens": 16384, "--attn-prefill-chunk-size": 16384,
    "--state-checkpoint-interval-tokens": 8192, "--level": 3, "--cudagraph-mode": "FULL"
  },
  "departures": {
    "--served-model-name": "unset: ATOM then serves the model under its --model path, the value the recipe gives",
    "--max-num-seqs": "512, not twice the concurrency: one value keeps one memory-readings entry per configuration"
  }
}
```

`--host` and `--server-port` take their departures in `common`, since every
recipe sets them.

## 2. Run the search

```bash
compass advice SPACE OUT
```

OUT is the search's directory, made when absent; keep it for every rerun. The
command checks, simulates and ranks in one run. Each step prints its lines once
it has ended, so the simulation prints nothing until its last point ends:

1. **The expansion.** A space file it cannot expand is refused in one line naming
   the file and the field, such as `compass advice: SPACE: candidate S1:
   --max-num-seqs 512 in common, but --max-num-seqs 256 in
   dimensions.config.S1`, and nothing is
   written. Otherwise `OUT/candidates.json` holds each candidate and point with
   its full arguments.
2. **The check.** First the head it took, `the reuse check at <commit>
   (origin/feature/atomcompass_new), ATOM digest <digest>: verdicts in
   OUT/reuse.json`, or `the reuse check refuses: <reason>`. Then each candidate:
   its deployment and basis, each departure, where it was looked up, each law and
   readings verdict, and its state (below). It ends `<N> points searchable, each
   state in OUT/states.json`. A searchable point has started once with no traffic,
   its log under `OUT/starts/`. Nothing is calibrated or measured. On the
   DeepSeek-V4-Pro template with no law in the store, this took about a minute on
   16 cores ([#3497](https://github.com/jgong5/ATOM/pull/3497)).
3. **The simulation.** Each searchable point runs on a fresh server with one
   client run, into `OUT/runs/<candidate>-c<concurrency>/`, its server's output in
   `<point>.log` and its client's in `<point>.client.log` beside it. Points run in
   parallel, as many as the host's CPUs hold at one CPU per rank. Meanwhile
   `OUT/runs/<point>/point.json` appears as each point finishes. At the end,
   each prints `<point>: finished in <dir>, <s> CPU s`, or `failed after 2
   tries` with each try's reason, then a total line. A rerun skips a finished
   point.
4. **The results**, in `OUT/results.json`: the SLO, the result table, the command
   that makes what each coverage report lacks, and the recommendation's text,
   which starts `Recommended:` or `No recommendation:`.

Ctrl-C stops every point; a finished point keeps its run, and the next search
takes up the rest.

### Each candidate's state

| State | What it prints | Next |
|---|---|---|
| `searchable` | `N of M points searchable`, each other point with its own state | step 3 simulates it |
| `missing a law` | `made by:` the command that makes the law; `calibration:` the last line of one that stopped or ended | [3. Calibrate what is missing](#3-calibrate-what-is-missing) |
| `calibrating` | `calibration:` its status file, how many steps have ended and its last line; `resumed, once no calibrate.sh runs, by:` its command | run the search again later |
| `missing readings`, a point of a searchable candidate | `made by:` its `compass measure memory` command | [3. Calibrate what is missing](#3-calibrate-what-is-missing) |
| `refused` | `refused:` ATOM's or Compass's refusal, verbatim | below |

A refusal is a result, not a step to work around:

- **An argument ATOM rejects** (`ATOM's parser: ...`, `ATOM's config: ...`) or a
  start's error, such as an out-of-memory one: correct the space file with the
  user, and run the search again.
- **Another tree**: check the clone out at the head again (Before the first
  search).
- **A law whose own held-out figure misses the 5% gate** at the head: no
  calibration of the same configuration fixes it. Tell the user; the candidate
  stays out.
- **A configuration Compass refuses**, such as one with
  `--enable-dynamic-chunking`: it stays out.

## 3. Calibrate what is missing

Each `made by:` command runs on the target's GPUs. An agent shows it to the user
and starts it only on the user's yes ([With an agent](#with-an-agent)).

**A single-node law** is one `calibrate.sh` command: the records the law's backend
plans from the configuration, then the fits. It prints as one line of variables,
the script, its OUT and one quoted argument per record, followed by indented
notes:

```
AITER_BF16_FP8_MOE_BOUND=0 ... MODEL=... GPU=0,1,2,3,4,5,6,7 SERVER_ARGS='...' STORE=<the shared store> <T>/scripts/compass/calibrate.sh OUT/cal/S1 '1 --trajectory-start-min-ratio 0 ...' ... 'trace 16' 'trace 4'
  before it runs, set AIPERF_PYTHON, which calibrate.sh requires and this environment does not set
  expected time: unknown, as no earlier calibration's status is read for it
```

The first line is the command. `GPU` names every card the configuration uses.
Its GPU time is those cards times its wall time: say `unknown` as printed, with
the number of records and cards, and add no estimate of your own. A note `no
trace run, which compass record refuses` says why the law's prefill will stay
unfitted; show it with the command.

Start it in the background, so it outlives the shell, its output beside its
directory, with the variables set as Before the first search sets them:

```bash
mkdir -p OUT/cal
setsid nohup env <the first line> > OUT/cal/S1.log 2>&1 < /dev/null &
```

Run it alone on the node, on idle GPUs, as QUICKSTART's Environment says: a
calibration beside other work fits that load into its laws.

**Where it publishes.** With `STORE=<the shared store>`, as printed when the
shared store holds neither readings at the configuration's key nor a law of it
at the head, it publishes the law and the memory readings there, and the next
`compass advice SPACE OUT` reads them. Without
`STORE`, as printed when the shared store holds one of them, which a second
publish at its key would meet, it publishes into its own
`OUT/cal/<candidate>/store`. Once a calibration has started, its printed command
keeps the store it started with, which its `OUT/cal/<candidate>/inputs` names,
so a resume publishes where the first run did. A search looks each candidate up
in those stores, the latest published into first, then in the shared store. A
store holding no platform readings, as a calibration's own store starts, is read
with the copy committed for SPACE's machine (Before the first search), and
passed by where there is none. So once a calibration has ended whose stack is
not the copy's, or whose machine has no copy, take the platform readings into
its store on two idle cards of the target, as QUICKSTART's Environment ("The
platform") takes them, `<token>` the one the first command prints:

```bash
compass measure device --gpu <a card>,<another card> --store OUT/cal/S1/store --out OUT/cal/S1.device
compass fit device <token> --store OUT/cal/S1/store --out OUT/cal/S1.readings
```

- **Check on it**: `compass advice SPACE OUT` prints the candidate `calibrating`
  with a `calibration:` line: its status file, `OUT/cal/S1/status`, how many
  steps have ended and the file's last line. The file holds a line per step's
  start, its end with its wall seconds, or its failure with its reason, each
  stamped in UTC: `cat OUT/cal/S1/status`. Each step's own log is in
  `OUT/cal/S1/`, and the script's in `OUT/cal/S1.log`. A status file whose last
  line is a start long past, with no `calibrate.sh` running, is a calibration
  killed with no chance to record it, as a node's job end kills one: resume it
  with the command its `resumed, once no calibrate.sh runs, by:` line prints.
- **Resume a stopped one**: run the same command again, in the background as
  above. A failed calibration's state is `missing a law`, its `calibration:`
  line naming the step that failed and why, and its `made by:` command is the
  same. `calibrate.sh` skips each step whose output exists and takes a stopped
  one again; its other refusals and exit codes are in its header.
- **Its end** prints the published tokens, the law's held-out errors, a table of
  each step's wall seconds, the node-hours every take of every step held, and
  this run's wall time; the status file's last line is then `end calibration
  <h> node-hours`.

**A missing point's readings** take `compass measure memory --gpu G --out D
--store <store> -- <ATOM's arguments>`, as printed, under the variables it
names: `G` the cards the configuration uses, `D` a fresh directory. It is one
engine start on those cards.

**A 1P1D candidate's laws** take one `calibrate.sh` command, as one server's
do, on one node: each record is a `deploy.sh record` of both sides, the prefill
side on the first cards and the decode side on the next. In place of `GPU` and
`SERVER_ARGS` it sets `PREFILL_GPU` and `DECODE_GPU`, `PREFILL_ARGS` and
`DECODE_ARGS`, each side's own variables as `PREFILL_ENV` and `DECODE_ENV`, the
router's arguments as `ROUTER_ARGS`, and `FIT`, the sides whose laws it fits:

```
AITER_LOG_LEVEL=WARNING ... MODEL=... PREFILL_GPU=0,1,2,3 DECODE_GPU=4,5,6,7 PREFILL_ARGS='...' DECODE_ARGS='...' PREFILL_ENV='...' FIT='prefill decode' ROUTER_ARGS='--backend atom ...' STORE=<the shared store> <T>/scripts/compass/calibrate.sh OUT/cal/A '1 --trajectory-start-min-ratio 0 ...' ...
```

Its records are the decode side's rungs, and the trace runs when it fits the
prefill side. It fits only the sides its store will lack. A side whose law an
earlier candidate's calibration publishes into that store, as the development
template's B shares A's prefill side, is borrowed: the candidate prints that
calibration for it, as `<A>'s calibration, whose law is at this configuration's
key, publishing into <store>`, and its own command fits the other side alone,
`FIT=decode`, into the same store. Where that store is A's own, B's command
reads `its calibration, into <store>, once A's calibration has started` until
A has started; run `compass advice SPACE OUT` again then. Its status file,
resume and end are one server's, above, and its end prints each fitted side's
tokens and held-out errors. Show the user every command the candidate prints.

Then run `compass advice SPACE OUT` again, now or in a later session, rather than
waiting on the calibration: a candidate whose law has published is searchable,
and one whose calibration is still running reads `calibrating` until it has.

## 4. Read the results

Once every point you need is simulated, the last lines of `compass advice SPACE
OUT` are the result. The table holds one row per point: configuration,
concurrency, both throughputs per GPU, both interactivities, P90 TTFT, P90
end-to-end latency, whether it meets the SLO and is on the frontier, its grading
state, its pricing-method shares, its extrapolated share of steps and its
refusal. The text then names the recommendation and its metrics, the runner-up
at its concurrency or why there is none, its confidence, its laws, and each
point left out with its reason. The metrics, the frontier and the rules that
choose them are
[19_adoption.md, D111](../../atom/compass/design/19_adoption.md#d111-results-the-frontier-the-recommendation-the-runner-up).
No search point is graded, so every point reads `ungraded` or `coverage report`.

**The dashboard.** The same run writes `OUT/dashboard.html` and prints
`the dashboard in OUT/dashboard.html`: one static page of the search, opened
from the file in a browser with no server. It draws each point's throughput per
GPU against its interactivity, each axis switching between its two definitions,
then the recommendation's text and the table, and a point whose run wrote
`OUT/runs/<point>/report.html` opens it. What it shows is
[19_adoption.md, D115](../../atom/compass/design/19_adoption.md#d115-the-dashboard-and-the-per-run-report).
To draw another source's curves on it, such as InferenceX's for the same model,
write them into a file and name it:

```bash
compass advice SPACE OUT --reference REFERENCE
```

REFERENCE is one JSON object, its fields as `dashboard.reference` in
`atom/compass/advice/dashboard.py` reads them: `source`, which labels the
curves on the page, and `curves`, each name to its points. A file of any other
shape is refused by name, and nothing is written. The page drawn without
`--reference` has no such curves, an SLO-only rerun's included.

**Another SLO.** Change only `slo` in the space file and run `compass advice
SPACE OUT` again. It prints `SPACE differs from the search in OUT only in its
SLO: its results again, nothing checked or simulated`, and the results follow in
seconds. A change to any other field checks again and simulates every point that
has not finished at the head.

## With an agent

An agent that runs this guide for a user keeps these rules:

1. Fill the space file from what the user asks for and the recipes or real-run
   arguments they name (step 1). Each value it cannot take from them is a
   question to the user.
2. Run `compass advice SPACE OUT` and report each candidate's state as printed.
3. For each command step 3 runs, show the user the command and its GPU time:
   the cards it names, times its expected time as printed, `unknown` while it
   prints so. Start none without the user's yes.
4. On a yes, start it in the background (step 3) and do not wait for it. Run
   `compass advice SPACE OUT` again later, or in a new session, with the same
   SPACE and OUT.
5. Report the recommendation and the result table as printed, with no judgement
   of its own.

## Traps

| Symptom | Cause | Fix |
|---|---|---|
| `the reuse check refuses: the tree atom resolves under, ..., is not a git repository's root` | a snapshot tree, or git refusing a clone it does not own | a clone, with `safe.directory` set in the container (Before the first search) |
| `... has 0 commits at a remote-tracking ref of feature/atomcompass_new` | the clone has not fetched the branch | `git fetch origin` on the host |
| every candidate `refused`, requirement 1, both digests named | the clone is checked out off the head | check it out at the head again |
| `the frozen commit <c> is not one of origin/feature/atomcompass_new's` | the commit is not on the branch, or the fetch is older than it | `git fetch origin`; freeze a commit of the branch |
| every point `not run`: `HARNESS_PYTHON is unset` | the simulated client's Python is not set | `export HARNESS_PYTHON=$SIM_PY` (Before the first search) |
| `before it runs, set AIPERF_PYTHON` under a printed command | the shell that ran the search did not set it | set it in the shell that starts the calibration |
| a candidate `refused` with `ATOM's parser: unrecognized arguments: ...` | a flag ATOM does not have, such as a misspelt one | correct the space file |
| a candidate `refused` with `--server-port ... is among ATOM's arguments; the driver owns it` | an argument `compass run` assigns, `--host` or a port | leave it out, as What a space file leaves out says |
| a candidate `refused` with `this deployment offloads KV to a Mooncake Store and the run file has no mooncake_store entry` | its prefill offloads to a Mooncake Store and no block of it names the entry | the entry's path in its block's `mooncake_store` (step 1) |
| a calibrated candidate still `missing a law` | its calibration failed, or it published into a store holding no platform readings, at another stack than the committed copy's or on a machine with none | its `calibration:` line; resume it, or take its store's `measure device` and `fit device` (step 3) |
