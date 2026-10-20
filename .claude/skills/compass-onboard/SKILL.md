---
name: compass-onboard
description: Run scripts/compass/QUICKSTART.md with a developer new to Compass, on their own GPU, and explain each step from their own outputs. Checks the node's profile, the disk and an idle GPU before anything starts, announces each step with what it holds and asks before running it, smoke cell first and the full path after. Use when someone wants to get started with Compass, run the quickstart or the smoke cell, or learn Compass by running it.
version: 1.0.0
scope: jgong5/ATOM, scripts/compass/QUICKSTART.md, scripts/compass/smoke_cell.sh
last_updated: 2026-10-20
---

# compass-onboard

You run `scripts/compass/QUICKSTART.md` with a developer who is new to Compass,
on their own GPU, one step at a time, and explain each step from what it
printed and wrote. They learn Compass from their own run.

## Rules

1. **This file holds no step's command.** At the start, read `QUICKSTART.md`
   and `smoke_cell.sh` in full from the developer's checkout (`git rev-parse
   --show-toplevel` where they invoked you) and take every step's command from
   them as they read then, placeholders filled from this session. This file
   holds its read-only checks, what to explain, how to read a failure and where
   to stop. Where those files and this one disagree on a command or a figure,
   the files win: tell the developer what differs. They never stand in for a
   check here.
2. **Announce, ask, run, explain.** Before each step say what it does, what it
   holds (which GPU, how long it takes, how much disk and where) and what
   success looks like, from QUICKSTART's own figures and success lines. Then
   ask, and run it only on a yes. After it, explain it from its outputs
   (Explaining a step). A check or a file read holds nothing and needs no
   yes. For a step QUICKSTART gives no size, time or success line for (the
   container, the tree, the venv), say so, say what you will take as success,
   and report its `du -sh` after.
3. **Stop on any failure.** A non-zero exit, a refusal, or an output you cannot
   explain stops the session at that step: read it (Reading a failure), tell the
   developer the cause and the fix, and ask before anything runs again. Never
   edit a script, a machine spec or a profile to get past a refusal. One
   exception: when you have verified in the container that a container option
   the profile gives is the cause, you may, on the developer's yes, take the
   container step again with that option fixed (Taking the container step
   again). Tell them it is a defect in this skill's profile, to file as an
   issue.
4. **Safety.**
   - A GPU only while the GPU check (Checks) reads it idle; check again before
     each GPU step.
   - `HIP_VISIBLE_DEVICES` set to the GPU's HIP ordinal in every container
     shell that runs a step (`docker exec -e`). The compass commands set their
     servers' own from `--gpu`; this keeps anything else in that shell on the
     same GPU.
   - Containers named `<user>_<purpose>`, `<user>` from `id -un`: QUICKSTART's
     `<user>_<issue>_gpu` with a purpose for the issue, such as
     `<user>_onboard_gpu`. Act only on containers whose name starts with
     `<user>_`. Never stop, remove, enter or signal another user's container or
     process, even one holding the GPU you wanted.
   - Disk checked before anything writes.

## Checks, before anything starts

Each is read-only; report each that passes in one line. The first that fails
refuses and ends the session: say what failed, what is missing, and that
nothing was started. A check that offers the developer a choice (Disk, GPU)
runs again on their answer, and the session ends only when they take none.

1. **Target.** Read `hostname` and `rocm-smi --showhw` (each GPU's GFX version
   and device id, DID), name them, and ask which machine they want Compass to
   simulate: these GPUs, or another target. A target with no hardware they can
   run, such as MI455, cannot be onboarded: the quickstart calibrates on real
   runs of the device and grades against them. Explain that such a target is a
   projection only: a machine spec whose unmeasured fields are marked assumed
   (`atom/compass/design/05_machine_spec_and_probes.md`, D25's schema rules and
   D26), priced analytically from peaks carried over from a measured device,
   and held to soundness checks instead of a grade
   (`atom/compass/design/08_validation_protocol.md`, D50.2). Offer to onboard
   on a device they can run, and stop.
2. **Profile and machine spec.** Find the profile in this skill's `profiles/`
   whose `hosts` holds the hostname (Profiles), then the machine spec it
   names. Refuse, naming each missing piece, when:
   - no profile names this host: name the host, its GPUs' GFX version and DID,
     the directory searched and what a profile states, and whether
     `scripts/compass/machines/` has a spec whose `device.arch` is that GFX
     version (specs are named for device and host; a spec says what the device
     is, not how to run on this node). Say that someone who has run the
     quickstart on this node adds the profile, and offer another node that has
     one;
   - the profile lacks a front-matter field or a heading Profiles lists;
   - the machine spec it names is not in the tree;
   - the GPUs' GFX version or DID differs from the profile's, or the GFX
     version from the spec's `device.arch`;
   - the cards' PTL state, read as QUICKSTART's Environment says, differs from
     the spec's `device.ptl`.
3. **Runtime.** The profile's container tool runs for this user (`docker ps`),
   and its image is present (`docker image inspect`). A missing image does not
   refuse, but its pull is a step of its own: announce its size and ask.
4. **Disk.** `df -h`, as the developer, on the profile's data directory (its
   nearest existing parent if it does not exist yet) and on Docker's storage
   (`docker info -f '{{.DockerRootDir}}'`). The data directory needs what
   QUICKSTART's Disk gives for the path about to run (the smoke cell's figures
   now, the full cell's before the full path) and a running client's map, with
   room to spare; without it, offer the profile's fallback or ask for a
   directory with room. Docker's storage takes the container's writable layer:
   its cache and whatever else the profile does not mount elsewhere. When it
   reads no room, give the developer the profile's figure for that layer and
   where the daemon, which writes as root, would put it, and ask. Keep runs,
   trees, venvs and caches off the root disk; QUICKSTART's `git fetch` in the
   checkout is the only write there.
5. **Inputs.** The model directory (QUICKSTART's `MODEL`, a container path:
   read it on the host through the profile's mounts) resolves `config.json`;
   the traces file's size is QUICKSTART's (its sha256 reads the whole file, up
   to a minute when it is not cached, against the prefix QUICKSTART gives:
   offer it).
6. **GPU.** Run the checkout's `scripts/compass/preflight.sh` on the host, then
   QUICKSTART's check of a shared GPU (its use, VRAM and process listing). A GPU
   is idle when `PREFLIGHT_RC=0`, preflight lists it under `idle and empty`,
   and QUICKSTART's check reads it at 0% use and 0% VRAM with no process on it:
   a process is on it when `--showpidgpus` lists the GPU's index under the PID
   (the `GPU(s)` column of `--showpids` is a count, not an index).
   Preflight's own compute rows can read 1% on every GPU while its `rocminfo`
   runs; judge by its idle list and the check after it. Ask which GPU they want,
   within the profile's Allocation (ask which GPUs it leaves to them). Refuse
   a busy one, naming its VRAM or use reading or the PID on it, and offer the
   idle ones within the Allocation, or none. Its HIP ordinal is the entry of
   preflight's `export HIP_VISIBLE_DEVICES=` line at the GPU's position in the
   idle list.

Then summarize in one block: host, machine spec, GPU and HIP ordinal, data
directory, image, container name. Offer the smoke path.

## The smoke path

Each step announced, confirmed, run and explained. QUICKSTART's blocks marked
`# host` run on the node, the rest through `docker exec` in the container.
Create the data directory as the developer before the container mounts it, or
the daemon creates it as root. Once the tree step names `$T`, write
QUICKSTART's Variables, filled for this session, into one file in the data
directory and source it in every `docker exec`: a tool call keeps no shell.
Announce that file and show it; it holds nothing, so it needs no yes.

1. **Container.** QUICKSTART's `docker run`, with the profile's image, mounts
   and options, named `<user>_<purpose>`. If a container of that name is the
   developer's already, ask whether to reuse it. It holds no GPU until a
   command runs. Explain: one container serves every step, and its image is the
   stack the machine spec was measured on; a fit refuses records from another
   (a record's `run.json` `stack` against the spec's
   `device.software_pinned_to`).
2. **Tree.** QUICKSTART's One tree, on the host. Its lines on commits behind,
   changed files and `gate_cpu.sh` are for the gates: say so. Explain from the
   extracted tree's `.compass-commit`: every record, fit and run is keyed by its
   tree, so two trees' artifacts refuse each other.
3. **Client Python.** QUICKSTART's venv recipe, in the container. Explain the
   two client Pythons: a real server's client is the image's aiperf; a
   simulated server's paces aiperf on the server's virtual clock through
   compass-harness (`atom/compass/design/06_workload_harness_contract.md`, D34).
4. **The smoke cell.** QUICKSTART's The smoke cell block. It holds the GPU for
   the time QUICKSTART measured and takes the disk QUICKSTART's Disk gives. Its
   calibration times the host, so show what the GPU check reads on the node's
   other GPUs and ask the developer to confirm the node is quiet enough. Say
   that this one yes covers all its steps, which run back to back. Run it
   detached in the container, its output in a log outside its OUT (it refuses
   an OUT that is not empty), since it outlives any one tool call. Follow the
   log without holding a tool call open: each `smoke_cell: <step>` line means
   the step before it finished, so explain that step from its directory while
   the next runs. When it exits, explain the grading table and its last
   line.

Then offer the full path, or Cleanup.

## The full path

QUICKSTART's The path, steps 0 to 6, each a step of its own: announce it from
QUICKSTART's times and Disk, check the GPU (steps 0 to 3) and the disk, ask,
run it detached as above, explain it. Use a new cell directory. The smoke
cell's container and tree may be reused; that container is warm, so step 0 may
be skipped: say so and let the developer choose. Step 2 times the host as well:
before it, show what preflight reads on the other GPUs and ask the developer to
confirm the node is quiet enough, since a neighbour's run skews a calibration
record. Run step 3's two loops at once only on two GPUs that both pass the GPU
check. Then offer Cleanup.

## Stopping a step

Signal only the step's own processes, in the container. SIGTERM the detached
step's script first, then whatever it started that still runs: `smoke_cell.sh`
on its exit stops a server it started in the background, but not the command it
was running in the foreground (`measure memory`, a fit, `client.sh`,
grading), and a stopped `serve` loop leaves its compass command running. `ps
-ef` in the container shows them; SIGTERM each, and a compass command stops its
own server and engine. The step has stopped once the GPU check reads no
process on the GPU. Take it again into a fresh OUT (QUICKSTART, stopping a
step).

## Taking the container step again

Under Rule 3's exception only. Stop any step still running in the container
(Stopping a step), remove that container, and run QUICKSTART's `docker run`
again with the same name, image and mounts and the one option fixed. Check the
fix in the new container (for a mount option, `/proc/mounts`). The tree, the
venv (`$V`) and the variables file live in the data directory on the host and
are kept, so steps 2 and 3 are not taken again; the failed step is, into a
fresh OUT.

## Cleanup

A step of its own, offered when the developer is done. Say what it removes and
what to copy out first, such as the cell's `grade.log` and logs. The container
runs as root, so what it wrote in the data directory is root's on the host:
remove the data directory's contents from inside the container, then the
container, then the empty data directory on the host. After it, `docker ps -a`
lists no container of this session and the GPU check reads the GPU free.
Remove nothing outside this session's data directory and container.

## Explaining a step

Open the step's outputs, quote the developer's own values and say what they
mean. Design topics are under `atom/compass/design/`; give the one named.

| After | Open | Explain |
|---|---|---|
| memory | `memory/memory_readings.json`; the log's `published ... as <token>` line; `memory/engine.log` | What an engine start reads off the device (`members`: capacity, reserves, activations, the graph pool), taken once so that a simulated run sizes ATOM's own KV pool from them with no device. Its `key` is what it was measured under; the token names it in the store. `03` D14. The engine sees only the GPU `HIP_VISIBLE_DEVICES` names and numbers it 0, so its log's `GPUs 0 to 0` and `cuda:0` are rocm-smi's GPU G. |
| warm | `warm/run.json`, `aiter_jit_builds` | The kernels aiter built the first time this container met them. The cost model does not model warmth, so nothing reads this record and grading refuses a record that built one (`06` D27, *Warmup is not a protocol feature*). |
| a record: calibration, real1, real2 | `run.json`; `steps/dp0.jsonl` and `dp0.engine.json`; `artifacts/client_run.json`; `server.log`; `preflight_*.txt`, `gpu_*.txt` | `run.json`: the tree (`source`), the stack, the device and PTL, the environment, the engine's bounds (`engine`). The step record: one line per forward step, its batch and its times (`atom/compass/parity/__init__.py` says what each field holds); show one prefill and one decode line with their durations. The client run names every request sent. The GPU checks before and after the server. `07` D38 |
| fit_steps | `fit_steps/step_linear_law.json` | The step linear law: per vector, a step cost fitted in relative error on the calibration steps (`09` D53 to D55). `params.unfitted` must be `{}`; `report.vectors` holds the samples behind each. `report.held_out` is empty with one calibration record, which is why the smoke cell's cost rows read unvalidated. |
| fit_host | `fit_host/host_laws.json` | The CPU time around each step, per step kind, which a simulated server charges on its clock (`09` D54, composed by `01` D4). |
| real2 | real1's and real2's step records | Two real runs of one workload: how far apart they land is the spread every metric is judged against (`08` D45). |
| step_linear | `run.json`; `client-0/summary.json`; `step_table.txt`; `steps/dp0.cost.jsonl` | The same server and scheduler on a virtual clock, each forward priced by the law, no GPU (the design README, *The one idea*). `coverage_report` must be false; `cost` gives simulated over wall seconds. `step_table.txt` is the Clock Authority's grants (`01` D3); `dp0.cost.jsonl` what each step's charge was made of. |
| grade | `grade.log` | Below. |

The grading table's columns are cell, result, metric, value and verdict. Read
its row families side by side, never as one number (`08` D44): counts and
distributions (does the simulated schedule agree with the real ones), end to
end (TTFT, TPOT, throughput), and cost (each real run's own steps priced by the
law, `08` D47). Memory rows are per term (`08` D48); warmup rows state each
run's warmup (`06` D27); summary rows give the refusals, the extrapolated and
fallback steps, each method's share of the seconds (the design README,
principles 6 and 7) and the speed. QUICKSTART's step 6 says what the value and
each verdict mean. Show one row of each family from their table. An `outside`
row is a result, not a failure.

## Reading a failure

1. Find the log. `smoke_cell.sh` names the failed step and its log,
   `OUT/<step>.log` or a client's `OUT/<step>.client.log`; a server's own log
   is `server.log` in its directory. Read the tail first.
2. Match the symptom against QUICKSTART's Traps and give its fix.
3. A GPU refusal before a run (`... before the run, so it was not started`)
   means a neighbour arrived: never stop it; run the GPU check and ask. After a
   run (`... after the run, so it fails`), run the GPU check at once. If the
   PIDs it named are gone and the run's engine had just shut down (its log's
   last lines), the check most likely caught the run's own processes still
   exiting, a known defect of the check
   ([#1787](https://github.com/jgong5/ATOM/issues/1787)): say so, and offer the
   step again into a fresh OUT. If one stays on the GPU, the run shared it: say
   its record cannot be trusted, and ask before taking it again.
4. Exits 90 to 99 are the tree checks in `scripts/compass/_lib.sh`: `atom`
   resolved outside `$T`, usually a missing `PYTHONPATH=$T` or a command not
   run from `/`.
5. `failed to map segment from shared object`, or `Triton cache error:
   compiled module ... could not be loaded`: a cache on a mount without `exec`.
   Read `/proc/mounts` in the container: `noexec` on `/root/.cache`,
   `/root/.aiter`, `/root/.triton` or `/tmp` is the cause, and the fix is that
   mount with `exec` (Rule 3's exception).
6. A step taken again needs a fresh OUT (QUICKSTART, stopping a step).
7. A symptom Traps does not hold: show the developer the log lines that matter
   and ask. Do not guess a fix or patch a script; a quickstart defect is an
   issue to file. A cause you have verified in a container option the profile
   gives is Rule 3's exception.

## Profiles

`profiles/<name>.md`, one per node or cluster, written by someone who has run
the quickstart there. Its front matter: `hosts` (hostnames, short or full),
`machine_spec` (the spec's path in the tree), `gfx` and `did` (its GPUs'). Its
body, one heading each:

- **Runtime**: the container tool, the image's tag, and the mounts and options
  QUICKSTART's `docker run` needs here.
- **Allocation**: how a GPU is booked (shared, or by a scheduler) and which.
- **Inputs**: the model directory and traces, and where they mount.
- **Disk**: the data directory, where Docker's storage is, the fallback when
  either is full, and what the container's writable layer takes then.

MI355X has a machine spec, `mi355x-gpu45.json`; its profile follows #1032.
