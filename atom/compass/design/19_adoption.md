# ATOM Compass — Design Topic 19: Adoption, a User's Configuration Space Searched

**Status:** drafted 2026-11-13 by an AI assistant from the decisions of the design issue
[#3355](https://github.com/jgong5/ATOM/issues/3355), which the owner took from 2026-11-10
to 2026-11-13, and the owner's rulings on its design PR #3363
([first](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6845852318),
[second](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6851839086),
[third](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6854911957),
[fourth](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6857982216),
[fifth](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6869501859),
[sixth](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6871001927)). Its code tasks are cut from *Summary* by
[#3355's plan](https://github.com/jgong5/ATOM/issues/3355#issuecomment-6878189760).

**Depends on:** `01` (D6, D6.2), `07` (the store, D37), `08` (D44, D45, D45.1, D50, D50.1),
`13` (D79, D80), `16` (D95.1), and the M3-Agentic stage's `03` D13.1, `06` D35.1 and `14`
D87.1.

| Touches | How | Decision |
|---|---|---|
| `13` D79: precedence has no level but CLI and artifact, and no Compass config file | not amended: the space file is the search's input, as traces are a client's, not the config file D79 rejects, as the [sixth ruling](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6871001927) settles | D110 |
| `13` D80: `run`'s flags | `--lmcache FILE`, the run file's `lmcache` entry | D110 |
| `13` D80: no command starts the client or takes a workload | `compass advice` does both | D109 |
| `13` D80: a prefill or decode `run` refuses without `--join` | not under `--check`, which resolves the side as `deploy.sh` does | D109 |
| `13` D80: what `run --check` lists | a step linear law's scheduled decode rungs with no vector, too | D112 |
| `13` D80: what `run`, `record` and `deploy.sh` leave | each run's per-request split and `report.html`, and a simulated run's `prometheus.html`, written as it ends by `client.sh --stop` for an instance and by `deploy.sh` for a deployment | D114, D115 |
| `08` D50: the cell matrix | a MiniMax-M3 showcase set: the recommended configuration and the runner-up | D117 |
| `01` D6: both sides of a write plan one block, each summed over its stages | a TP1 prefill stage writes into a wider decode as Mooncake cuts its heads; an option the connector does not carry is refused by name | D110, D116 |

---

## Scenarios

Each walks one use #3355 or #4145 names through the decisions to its outcome; *Summary*
lists the outcomes as acceptance items. A and B are D116's configurations, on pit2's MI355X
nodes; D117's development nodes are crsuse2-m2m-v3-005 and -013.

### 1. A search, from the space file to the recommendation

```
 a space file (D110), by hand or by an agent following the guide (D108);
 the two templates are space files in scripts/compass/spaces/, never a default
   |
   v
 compass advice SPACE OUT (D109)
   1 check     expand into candidates and points (D110); give each a state from
               the reuse check (D112), run --check and a start with no traffic
               (scenario 2)
   2 simulate  each searchable point on a fresh server: compass run, or deploy.sh
               run on 1P1D (scenario 5), one client.sh run; no step roofline;
               a finished point skipped
   3 results   each point's metrics from its client export (D111): the frontier,
               the SLO filter, the recommendation, the runner-up, the result table
               with each point's extrapolated share (D112) and the recommendation's
               text; dashboard.html (D115)
   |
   v
 OUT: dashboard.html, the recommendation, the result table, runs/<config>-c<n>/
   a point with a refusal: a grey coverage report, out of the frontier (D111)
   a changed SLO in SPACE: step 3 again, nothing simulated (D109)
```

### 2. A candidate the store cannot serve yet

```
 step 1's state (D109): per candidate, and per point for its readings
   searchable ......... simulated in step 2
   missing a law ...... the calibrate.sh command of its record list, every
                        argument and the expected time (D113)
                        -> started by the user on the target machine type, in
                           the background (D108): it publishes into STORE and
                           appends each step to its status file
                        -> compass advice again: calibrating, then searchable
   calibrating ........ its status file's progress; compass advice again later
   missing readings ... one point: the compass measure memory command that
                        run --check prints for it (D113)
                        -> compass advice again: searchable
   refused ............ ATOM's or Compass's refusal verbatim, never simulated:
                        another tree or a law at the head past the 5% gate
                        (D112), a flag Compass refuses
                        (--enable-dynamic-chunking)
```

### 3. The reuse check's verdicts

```
 the ATOM digest of the tree atom resolves under, against the head's: the
   digest of the integration branch's tip as last fetched, or of the commit
   SPACE freezes (D110), computed where the check runs
     another digest -> refused, both digests named (requirement 1)
     the head's     -> each input by its key (07 D41), its ATOM digest
                       against the tree's (D112)

 the candidate's law, before the search      verdict
   none at its key                           calibrate
   a store entry at another digest           calibrate, or in development
                                             checked in re-keyed
   re-keyed past a change moving costs (2)   calibrate
   a scheduled decode rung with no vector (3)
                                             calibrate, the rungs listed
   a 08 D47 figure past 5% (4), the law's
     records at the head's digest            refused, the figure named
     re-keyed from an older tree             calibrate
   otherwise                                 reuse
 each point's readings
   none at its key, or another digest        measure
   otherwise                                 reuse
 each point, after the search
   steps outside a vector's hull (3)         extrapolated, never refused;
                                             its share reported
```

### 4. A recalibration and the configurations that borrow its law

```
 the reuse check: MiniMax-M3's TP2 law lacks a decode rung (D112, requirement 3)
   |
   v
 compass advice prints together (D113):
   calibrate.sh on TP2's old records and the missing ones  -> a new OUT, store N
   compass measure memory for TP2_LMCache, which borrows
     TP2's law (D117)                                      -> store N
   |
   v
 both done: N is the latest store holding all of TP2's and TP2_LMCache's inputs,
 so their points run on N (compass run --store N); the shared store is unchanged
```

### 5. A 1P1D candidate

```
 A, then B: one PP4 x TP1 prefill beside a Mooncake Store, writing chunked into its
 decode; A's decode TP4, B's DEP4, each admitting by pd_admission (D116)
   check      each side by compass run --check, resolved as deploy.sh resolves it,
              with --kv-transfer-config and the side's own P/D options, each under
              its own variables (D109, D110); a law and readings per side, B's
              prefill side at A's keys (D112)
   calibrate  A: one calibrate.sh, each record a deploy.sh record of both sides,
              fitting both sides into one store S (D113)
              B: its prefill side named as A's calibration, which publishes its
              law and readings into S; one calibrate.sh fitting B's decode alone
              into S, its records the decode's rungs, no second law at a key
   both done  S holds A's two sides and B's decode: both searchable on S
   simulate   deploy.sh run with the Store's entry (D110): both sides, each side's
              P/D options in its connector, LP mooncake_store, the router, one
              client.sh run; on A each TP1 stage writes every decode rank its
              heads (01 D6)
   results    both sides' GPUs counted, TP x PP x DP each, from its own engine
              record (D111)
   the run    its per-request split joins both sides on each request (D114)
```

### 6. With an agent

```
 the user asks an agent for a search (D108)
   -> the agent fills SPACE from what the user asks for, a recipe or the
      user's real-run arguments, following the guide (D110)
   -> compass advice SPACE OUT: each state and printed command (D109)
   -> the agent shows the user each calibration's command and its GPU time
   -> on the user's yes, the agent starts it in the background and does not
      wait (D108, D113)
   -> later, or in a new session: compass advice SPACE OUT again, reading each
      status file and the store, finished points skipped
   -> the recommendation and dashboard.html, with no judgement of the agent's
```

### 7. What each run writes, and the views

```
 a run stops: client.sh --stop on an instance, a deploy.sh deployment (D114, D115)
   -> its per-request split, a file of its own: queue, prefill, KV transfer,
      decode, LMCache store and load, from dp<r>.jsonl, transfers.jsonl and
      aiperf's per-request export
   -> on an op-level run, its op-family split: op.<kind> summed per step phase
   -> a simulated run's prometheus.html: CI's Prometheus report over its LPs'
      series on the simulated clock (`11` D74), or prometheus.txt saying why not
   -> report.html: the run, headline metrics, trust, both splits, its
      cell's grades.json once the cell is graded, and a link to prometheus.html
 compass advice ends -> dashboard.html: the frontier, the recommendation, two
   marks per point; a point opens its run's report.html
```

### 8. The showcase on MiniMax-M3

```
 MiniMax-M3's Compass features landed and #3265 passes -> the move (D117)
   the environment check on the development nodes: each arm served once
   the template: the arms kept, at every concurrency; SPACE freezes commit C
   calibration at C, step linear laws alone: TP4, TP4 with indexer CP and TP2,
     pooling the LMCache arms' records if they came up; readings per point;
     a step past the records extrapolated and its share reported; the step
     grid revisited only on the M3-Agentic stage's grades
   compass advice at C -> the recommendation and the runner-up (D111)
   an 08 D50 set registered -> each run real three times, spaced, 3600 s
     -> each cell graded by 08 D44; the real order against the simulated one
     -> beside each grade, not gated: its 08 D45.1 verdict, and the
        M3-Agentic stage's verdict on gpu-40, as another platform
```

### 9. The development template, checked against held-out real runs

```
 SPACE: D116's template, frozen at commit C, the first holding every piece
 D116's table lists; every step below runs on pit2 at C, in C's image, but the
 platform readings and the network file, which may come before
   |
   v
 the template's inputs (#3468), whole-node gpool leases:
   platform readings, a network file of pit2's NICs, the Store's entry
   environment check: each decode served by the recipe's scripts;
     gsm8k.sh <label> 32 200 through the router exits 0 with an accuracy  A3
   calibrate.sh for A, then B, as compass advice prints them; each
     status file holds every step's node-hours                          A6
   compass advice SPACE OUT: every point searchable and finished,
     results.json opening "Recommended:", its CPU seconds              A1, A6
   |
   v
 the grading (D116, Validation), whole-node gpool leases:
   for each decode, at c = 16, 32, 64, 140, and at advice's pick when 8 or 256:
     store.sh stop, start; deploy.sh record at candidates.json's arguments;
     client.sh --agentx --concurrency c --duration 900, aiperf scraping the
       router and the prefill's /metrics
     one_real.py REAL SIM: each metric, SIM's error, REAL's own 95%
       interval over its 60 s blocks, in or out                          A2
   A at C140: the decode's /metrics before the stop, real, and simulated by
     simulated.json's ran again; the atom:pd_admission families side by side A4
   at advice's pick: the recommendation and the runner-up in REAL's order by
     throughput per GPU, or the gap within REAL's intervals              A5
   |
   v
 the report on #4145: the tables, A4's values, A5's statement, each step's
 node-hours and the search's CPU seconds; the dashboard reads every point
 ungraded (D115), no 08 D44 cell holding two real runs
```

---

## D108. Fitting apart from searching

**Rule.** A candidate, one configuration of the user's space, is simulated once its law
exists; fitting and searching meet only through the artifact store and a calibration's
`status` file (D113). The search consumes laws and calls the reuse check (D112); it owns
no calibration, fitting, reuse judgement or version management, and reaches profiling
(D114) and the views (D115) only through files in each run's directory. A user guide
comes first, and no skill: `scripts/compass/ADVICE.md`, linked both ways with
`QUICKSTART.md` and #2893's page, gives each step one command and its expected output,
has a section on filling the space file (D110) and states D109's limits, exact enough
that an agent following it fills the development template's space file from the
recipe D116 names. An agent shows the user a printed calibration's GPU time, starts it
in the background only on the user's yes, reruns `compass advice` later rather than
waiting, and adds no judgement of its own.

**Reason.** A calibration spends GPU hours and a search CPU minutes: apart, each half is
built and replaced alone, and a rerun takes up whatever calibration has finished; a guide
an agent follows saves the user the split's round trips. **Basis.** The owner on #3355 and
#3363; `07` D37; #2317 (`calibrate.sh` plans nothing). A skill needs its own ruling: the
[#2722 Ruling](https://github.com/jgong5/ATOM/issues/2722#issuecomment-6678010091)
removed `compass-onboard`.

**Rejected.** `compass advice` calibrating what a candidate lacks, one command from space
to recommendation: it would spend GPU hours without the user's yes and own the
calibration #3355 keeps outside the search.

---

## D109. `compass advice`: one command, three steps

**Rule.** `compass advice SPACE OUT` is a `compass` subcommand whose steps are its own:

1. **Check.** Expand SPACE (D110) and give each candidate one state: *searchable*;
   *missing a law*, the reuse check's *calibrate*, a law absent or short of a decode
   rung, with the `calibrate.sh` command of D113's record list; *calibrating*, with its
   `status` file's progress; or *refused*, with ATOM's or Compass's refusal verbatim. A
   searchable candidate's point whose readings D112 answers *measure* is *missing
   readings*, with the `compass measure memory` command `compass run --check` prints for
   it (`atom/compass/drive/lacking.py::MADE_BY`): a readings entry keys the scheduler's
   bounds and the KV offload (`atom/compass/run/memory.py::key_fields`), so each
   MiniMax-M3 point takes its own, and an LMCache arm its own beside its base arm's law,
   whose key holds no connector. The check runs the reuse check (D112), `compass run
   --check`, which on 1P1D resolves each side as `scripts/compass/deploy.sh` does, and a
   start with no traffic, for what only a start shows: arguments ATOM rejects, a model
   that does not fit memory, a configuration Compass refuses, such as
   `--enable-dynamic-chunking`. It never calibrates.
2. **Simulate.** Each searchable point, a fresh server per point (`13` D80): `compass
   run`, or `deploy.sh run` on 1P1D, driven by one `client.sh` run, points in parallel
   over the host's cores and ports, as many at once as its cores give each rank one
   (tensor times pipeline times data parallel over a point's instances), a failed one
   retried and then listed with its reason, a finished one skipped. A 1P1D point's
   router takes Compass's own flags, the same for every candidate: `--backend atom`, and
   the health check, circuit breaker and retries off, as `scripts/compass/pd_sim.sh`
   starts it; the space file states none
   (the [#3477 Ruling](https://github.com/jgong5/ATOM/issues/3477#issuecomment-6887140315)).
   An offloading point's lmcache server runs at its own port (D110). No
   `--step-roofline`: a refusal means a calibration is missing, and the law is completed,
   as stages 3 (#975, #2660) and 4 (#2544) completed theirs.
3. **Results.** D111's, with D115's dashboard; a changed SLO reruns this step only.

OUT holds `dashboard.html`, the recommendation, the result table and
`runs/<config>-c<concurrency>/`, each point's run directory. The search runs at the
integration head (D112), on any host's CPU and with no GPU device (the
[#4006 Ruling](https://github.com/jgong5/ATOM/pull/4006#issuecomment-7130990480)): a
simulated server starts with no ROCm driver (#3750), and host laws fitted on another CPU
serve, predicting the host they timed (#3751). Only the commands it prints, a
calibration or a readings measurement (D113), take the target's GPUs, on a node of its
type. It wraps the commands it runs, so no user step needs `python -m` (the owner's
ruling on #1108, recorded in #1527).

**Reason.** A command rerun until every point is searchable leaves the user nothing to
compute; with no roofline every point is priced by its own law; a later client run meets
the prefix cache an earlier one left (`03` D13). **Basis.** The owner on #3355; the
[ruling on #2312](https://github.com/jgong5/ATOM/issues/2312#issuecomment-6462166589);
`08` D50.1. `run --check` refuses a P/D side's arguments, and without
`--kv-transfer-config` DeepSeek-V4's index cache resolves to fp4, not fp8
(`atom/config.py`), a value both the law's and the readings' keys hold.

**Rejected.** Three commands, a check, a simulation and a report, each run by hand: #3355
makes the search one command whose rerun is the whole loop, and step 3 alone already
answers a changed SLO.

---

## D110. The space file

**Rule.** The user provides the space, by hand or through an agent that fills it,
following the guide, not a skill (D108), from what the user asks for, a recipe or the
user's real-run arguments. It states the model, the machine, the shared store (D113), the
workload (traces and concurrencies), the SLO, the ATOM arguments every candidate shares,
the dimensions searched with their values, exclusions, arguments that vary with the
concurrency, as MiniMax-M3's `--max-num-seqs` is twice it, and, for a search frozen at
one commit as the showcase is, that commit, whose ATOM digest stands for the head's
(D112). It expands into the product of the dimensions less the exclusions, always into
the same candidates in the same order, and refuses by name an unknown field,
contradictory values or an empty expansion. Each candidate names its ATOM basis, the
recipe its arguments follow, and each departure with its reason. Its format follows.
The two templates, D116's and D117's, are space files in `scripts/compass/spaces/`,
beside `laws/`; `compass advice` reads the SPACE it is given and falls
back on neither. The file is the search's input, as traces are a client's: each
candidate's run takes `compass run`'s flags and ATOM's arguments from `compass advice`,
so the file adds no level to `13` D79's precedence and is not the Compass config file D79
rejects.

**The format.** One JSON object, as the law and machine files beside the templates are. A
field not listed here, or a key given twice in one object, is refused by name.

| Field | Values | Default |
|---|---|---|
| `model` | ATOM's `--model`, every instance's first argument | required |
| `machine` | `device`; `ptl`, `enabled` or `disabled`; `stack`; `network`: `compass run`'s platform flags | `stack` and `network` unset |
| `store` | the shared store (D113), an absolute path | required |
| `commit` | the frozen commit, 7 to 40 hex digits (D112) | unset: the integration head |
| `workload` | `traces`, a traces file's absolute path; `select`, `A-B` or `agentx`, as `client.sh`'s `--traces A-B` or `--agentx`; `concurrency`, distinct positive integers; `duration_s`, as `client.sh`'s `--duration` | `traces`, `select` and `concurrency` required; `duration_s` `client.sh`'s |
| `slo` | `min_interactivity_p90`, tokens a second per user, and `max_ttft_p90_s`, seconds, each positive (D111) | required; a bound left out bounds nothing |
| `common` | a block every candidate takes | none |
| `dimensions` | each dimension's name to its values, each value's name to a block; names of letters, digits, `.` and `_` | required, with at least one dimension and one value each |
| `exclude` | a list of objects, each naming a value of some dimensions; a combination holding every one of them is left out | none |

A block's fields are all optional: `basis`, the recipe its arguments follow; `deployment`,
`single` or `1p1d`, `single` when no block names one; `args`, each ATOM long flag to its
value, a string, a number, or `true` for a bare flag; `env`, each environment variable to
its value; `per_concurrency`, each ATOM long flag to a positive integer, its value at a
point that integer times the point's concurrency, as MiniMax-M3's `--max-num-seqs` is 2;
`departures`, each argument, by its long flag, or variable that departs from the basis to
its reason, the flag read as an argument is and written with dashes; `lmcache_server`, on a
candidate offloading to an lmcache server (`lmcache_mp`), each `lmcache server` long flag
to its value as `args` takes one, `--host` and `--port` refused by name;
`mooncake_store`, on a candidate offloading to a Mooncake Store (`mooncake_store`), the
absolute path of the run file's `mooncake_store` entry (`01` D6.2); and, on a 1P1D
candidate, `prefill` and `decode`, each holding one side's `args`, `env`,
`per_concurrency` and `kv_transfer`. A side's `kv_transfer` is an object of its P/D
connector's options, which `deploy.sh` merges into that side's `--kv-transfer-config`: a
real side's into Mooncake's config, a simulated side's into the `compass` connector's,
which takes the options its simulation carries and the transport choices a network file
prices in their place, `protocol` and `ib_device`, and refuses any other by name (`01`
D6). The keys `deploy.sh` sets itself, `kv_connector`, `kv_role`, `handshake_port`,
`compass_kv_write_req` and `compass_instance`, are refused in it by name. Router arguments
are no block's: a 1P1D point's router takes Compass's own (D109).

**The expansion.** Each combination of the dimensions' values that no exclusion names is a
candidate, named by its values joined with `-`, the first dimension varying slowest and each
dimension's values in the file's order. A candidate takes `common`'s block, then each of its
values' blocks in the dimensions' order; an instance's arguments are `--model`, then each
block's, a side's after its own block's. It is refused, naming both blocks, when two of
them set one thing apart: an argument, which ATOM's parser reads with `--a_b` as `--a-b`
and `--no-x` as `--x`'s negation, a variable, an lmcache server flag, which lmcache's
parser reads with `--no-x` as `--x`'s negation, a P/D option, a departure's reason, the
basis, the deployment or the Store's entry; or an argument both fixed and varying with the
concurrency. It is refused when
no block names its basis, and when a block gives a single-node candidate a side. An
excluded combination is never judged. A point is a candidate at one concurrency, in the
workload's order, named `<candidate>-c<concurrency>`. A setting that holds only for some
combination of values is written in one value of one dimension, as D116's template writes
each of its decodes as one value of one dimension. Each point of an offloading candidate
takes its own lmcache server port, from 5555 up in the points' order, so points run at
once on one host each reach their own server: `compass advice` writes it, with host
127.0.0.1, into the server's `--host` and `--port` and into the
`ATOM_KV_OFFLOAD_EXTRA_CONFIG` of each instance offloading through `lmcache_mp`, refusing
a block's value of it that names either, and hands the
point's `compass run` or `deploy.sh run` the run file's `lmcache` entry, those flags with
the request loop's declared laws (`atom/compass/kv/lmcache_server.py::DECLARED_LAWS`,
`01` D6.1), through `--lmcache FILE` (`13` D80). Each point of a candidate naming a
Store's entry hands its `compass run` or `deploy.sh run` that file through
`--mooncake-store FILE` (`13` D80), the same file for every point: each point's
deployment starts its own LP `mooncake_store`, which binds a random port and names it in
the run's endpoint file (`atom/compass/run/__init__.py::mooncake_store`), so points run at
once need no port assigned.
`compass advice` writes the expansion
into OUT's `candidates.json`: the space file's path, its model, machine, store, commit,
workload and SLO; each candidate with its values, deployment, basis, departures, each
instance's arguments, variables, P/D options and per-concurrency factors, and an
offloading one's `lmcache_server` flags or Store's entry; and each point with each
instance's arguments, variables and P/D options, and an offloading one's `lmcache_port`.

**Reason.** The user, not Compass, knows what they would deploy; a template shows the
shape, and an agent saves the user copying arguments by hand; the frozen commit has to be
stated before the calibration it governs, which precedes the search. A setting given twice
is refused rather than one overriding the other, so what a candidate runs is what its
blocks say. An lmcache server's flags set its hit rate, so a candidate states them, as it
states ATOM's arguments; its address is the search's, since points run at once on one host.
A Store's entry holds the master's flags, the owners' bytes and the laws its copies and
lookups are priced by, so a candidate names it as it names an lmcache server's flags. A
side's P/D options are its own: ATOM refuses `pd_admission` on a producer, so one object
for both sides cannot carry a recipe that admits on its decode.
**Basis.** The owner
on #3355 and #3363, the [sixth ruling](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6871001927) settling that the file is no config file;
the [#3477 Ruling](https://github.com/jgong5/ATOM/issues/3477#issuecomment-6887140315) on
the lmcache server's flags and the router's;
`13` D79; `08` D50, whose registration freezes a set before evaluation; `client.sh`'s
`--traces`, `--agentx` and `--duration`; `atom/utils/arg_parser.py::FlexibleArgumentParser`,
which reads both spellings of a long flag;
`atom/compass/run/__init__.py::_launch_mooncake_store`, which refuses a deployment
offloading to a Store whose run file has no entry; #3417, which decides what command
writes an offload entry; `scripts/compass/deploy.sh`, which refuses `--kv-transfer-config`
among a side's ATOM arguments and merges `KV_EXTRA` into every real instance alike; and,
at upstream `a62134f99`,
`kv_transfer/disaggregation/pd_admission.py::PdAdmissionController.from_config`, which
raises "pd_admission requires a Mooncake consumer" on a producer.

**Rejected.** A shipped default space, the development template, that the command falls
back on: a default is the silent answer `13` D79 refuses, and it would search a space the
user never chose. YAML, as #3355 sketched the file: the law and machine files beside the
templates are JSON, which the standard library parses. A block keyed by a combination of values: a
value of one dimension already holds a whole configuration, and a setting then has one
source to trace it to. A Store port per point, as an lmcache server takes one: LP
`mooncake_store` binds a random port of its own and writes it to its run's endpoint file,
so no point needs one assigned. A side's P/D options as two `deploy.sh` variables,
`PREFILL_KV_EXTRA` and `DECODE_KV_EXTRA`, with no field: `compass advice` and
`calibrate.sh` start every run from the space file, as each side's `args` and `env`
reach them.

---

## D111. Results: the frontier, the recommendation, the runner-up

**Rule.** A point's metrics come from its client export. InferenceX's are computed by
#3264's code (`atom/compass/validate/curves.py`), which follows InferenceX's
`request_metrics.py` and grades the M3-Agentic stage (`08` D45.1): throughput per GPU of
the input and output tokens, cache hits included; P90 interactivity, one over the P90 ITL,
and the normalised one, one over the P90 of each request's end-to-end latency per output
token; P90 TTFT. The search adds two of its own from the same export: throughput per GPU
of the tokens the engine computed, the input and output tokens less
`usage_prompt_cache_read_tokens`; P90 end-to-end latency, from `request_latency`. A
point's GPUs are those its engine records state, each tensor parallel times pipeline
parallel times data parallel, a record that states no pipeline width counting one: a 1P1D
point's are its prefill side's plus its decode side's, each read from that side's own
record, in the `prefill/` and `decode/` that `deploy.sh` writes.

- **Frontier**: throughput per GPU, InferenceX's, against P90 interactivity; the points
  no other dominates on both.
- **Recommendation**: the most throughput per GPU, InferenceX's, among points that meet
  the SLO, its interactivity bound on P90 interactivity, and have no refusal.
  **Runner-up**: at its concurrency, another configuration's point chosen by the same
  rule.
- **A point with a refusal** is a coverage report (`08` D50.1), out of the frontier,
  recommendation and runner-up, listed with what it lacks and the command that
  calibrates it.
- **The result table**, a row per point: configuration, concurrency, both throughputs,
  both interactivities, P90 TTFT, P90 end-to-end latency, SLO met, on the frontier,
  grading state, pricing-method shares under `08` D50.1's names, extrapolated share of
  steps (D112), refusal reason.
- **The recommendation's text**: its metrics, the runner-up and the gap, its confidence
  (grading state, method shares, extrapolated share of steps), its laws' sources, every
  point left out with its reason; each number in it is in the table.

**Reason.** InferenceX's definitions, so a user reads the curves as InferenceX's, from the
code that grades the M3-Agentic stage, so a search and a grade share them; the search's
own add the engine's work net of prefix-cache hits and a request's whole wait; a refused
step charges less than it ran by an amount nothing measured (`08` D50.1). Both sides of a
1P1D point hold their cards for the whole run, and their widths may differ, so each side
counts its own; each pipeline stage holds its own cards, as D116's PP4 x TP1 prefill holds
four. **Basis.** The owner on #3355; #3264, landed by #3325; the GPU count,
#3454; `08` D45.1, D50.1; `atom/compass/run/step_grid.py::keyed`, which states
`pipeline_parallel_size` in the engine record above one stage, and
`atom/compass/drive/lacking.py::calibrate_inputs`, whose GPU count already multiplies it.

**Rejected.** Ranking by the computed-token throughput: the tokens a user is served,
cache hits included, are what InferenceX ranks and the M3-Agentic stage grades, so the
computed throughput stands beside them in the table.

---

## D112. Calibrated data is reused only when every requirement holds

**Rule.** Laws, memory readings and platform readings, in a store or under
`scripts/compass/laws/`, serve a candidate only when every requirement below holds. The
reuse check gives each candidate's law one verdict, *reuse*, *calibrate* or *refused*,
and each point's readings one, *reuse*, *measure* or *refused*, naming the requirement
that failed; `compass advice` calls it and judges nothing. A reused input is stated in
the run's provenance.

1. **Same key**, as `07` D41 keys each kind, at the head's ATOM, a law's and its host
   laws' capture ladder one the candidate's is a non-empty prefix of (`09` D54). The check compares the
   ATOM digest of the tree `atom` resolves under
   (`atom/compass/run/__init__.py::atom_digest`, the sha256 of ATOM's Python less
   Compass's own code, which every key reads) with the head's: the digest of the `atom/`
   the integration branch's tip holds, or the commit the space file freezes (D110), which
   must be one of that branch's. It computes that digest where it runs, in the search's
   container, over the `atom/` that `git archive` ships from the clone the tree belongs
   to, at the integration branch's remote-tracking ref, which the guide fetches on the
   host before a search, or at the frozen commit: git reads the local repository, which
   needs git in the container, as the ATOM images carry it, and no network. It names the
   commit it took as the head, so a stale fetch shows, and refuses by name a tree with no
   repository to read. A tree whose digest differs, as an older checkout, a release or a
   local patch may, is *refused*, both digests named:
   only the head is supported, other trees waiting for after v1 (#3122). A tree that
   differs from the head only outside ATOM's Python, a Compass code task's own included,
   reads every key the head reads, and passes. Each input's ATOM digest (a step law's
   `source_root`, host laws' `atom`, readings' `atom_src`) is then held against the
   tree's. A law with no entry at its key is *calibrate*; readings with none, or measured
   at another digest, *measure*, since they re-key with a law only under #3122's design.
2. **Not invalidated**: no ATOM change since the calibration moves step costs, and only
   then is a law re-keyed onto the new tree (README principle 10, #2594); a change that
   moves what a start under the law's engine config holds takes its readings again
   (`07` D41, *A law file's readings*). A store entry is never re-keyed: it is immutable
   and keyed by the digest its records ran at. A store law reaches a later head as a checked-in law
   file: checked in under `scripts/compass/laws/` re-keyed to the head, its provenance
   naming the commit its records ran at and saying re-keyed, not re-validated. The
   commits that re-key the checked-in laws carry it from then on,
   `tests/compass/test_platform_keys.py::test_every_checked_in_law_stamps_this_trees_atom_digest`
   holding each to its tree's digest, and a run takes it as `compass run --step-linear
   LAW --host-laws LAW`. A search may mix laws fitted at different commits, each point's
   provenance naming its law's; the showcase fits all its laws at one frozen commit (D117)
   and measures again after a later source change, until #2311 settles who judges "moves
   step costs" and how. Failing: *calibrate*.
3. **Enough coverage**, judged in two parts, as the
   [fourth ruling](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6857982216) has
   it. Before the search, from the configuration alone: the law has a decode vector for
   every rung the capture ladder schedules
   (`atom/compass/runner/overrides.py::schedulable_ladder`), and for the eager rung where
   the scheduler's bound passes the widest captured size, since the law refuses a decode
   step at a rung it has no vector for (`atom/compass/backends/step_linear.py`); `compass
   run --check` names each such rung. Failing: *calibrate*, the rungs listed. After the
   search, each point reports its extrapolated share of steps from its own simulated
   steps: a step outside a vector's hull, such as a context past the law's longest record,
   is priced by that vector and never refused, counted apart (`08` D49), and its point is
   no coverage report (`08` D50.1). No priced run is added before the search, and a shape
   outside the records that is no rung, such as that context, reaches no record list this
   round.
4. **Accurate enough**, as the
   [fifth ruling](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6869501859) has it: the
   fit reports `08` D47's three figures, on totals, holding out the step that contributes
   most and the median per step, per kind, a drafter's draft included, as
   `atom/compass/validate/cost.py` computes them, over each of its leave-one-run-out folds
   (`atom/compass/backends/step_linear_fit.py::mark_held_out`), the held-out run's steps
   priced by the law fitted on the others. Each run is taken alone, as `08` D47 grades a
   run's own step sequence. A law with any figure past #627's 5% gate fails. One whose
   records ran at the head's digest is *refused*, the run, kind and figure named, since a
   calibration of the same configuration takes the same record list (D113) and refits the
   same miss; one re-keyed from an older tree is *calibrate*. The check tells them apart by
   the records' ATOM digest, `records_atom`, which the fit writes into the law beside its
   host key and re-keying leaves alone, since a re-key moves only the host key's `atom`; a
   law that states none counts as re-keyed. A law fitted before the fit reported the
   figures, as the shipped DeepSeek-V4-Pro laws are, gains them in a commit to its law
   file, on the head's fit, CPU only, its provenance saying so: each source its `sources`
   splits the vectors over is folded with that source's own records, form and options, as
   `sources` and the provenance state them, and a law with no `sources` as one fit. A law
   with no figures is *calibrate*.

**Reason.** Each requirement guards one way a reused law stops matching the subject it was
fitted on, and one check outside the search gives every caller one answer. A law fitted at
an older tree and a user on one miss a key alike; only the head's digest, read from the
integration branch apart from both, tells which is behind. A point the search has not run
has no steps, and a law's own records lie inside its hulls, so only its rungs can be judged
before a run. A law the head's fit misses with cannot be calibrated into passing, so it is
refused, not sent round again. Readings are measured, not calibrated: a law's records take
hours, and a second law at a borrowed law's key is refused by the store. **Basis.** The
owner on #3355 and #3363: the
[third ruling](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6854911957) tells the
head by ATOM's digest, the
[fourth](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6857982216) judges coverage in
two parts, the [fifth](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6869501859) sets the accuracy bar; `07` D41, D43; `08` D47;
`atom/compass/backends/step_linear.py`, which refuses a rung with no vector and
extrapolates outside a hull; README principle 10; #2594, under which `compass
run` takes a law file's step law and host laws at the key the file states
(`atom/compass/drive/runfile.py::resolve`); #2899's head-only rule; `13` D80, under which a
backend fitted under another engine configuration refuses to run.

**Rejected.** Comparing commits to tell the head: it also refuses every tree that differs
from the head only outside ATOM's Python, most trees and a Compass code task's own among
them, though each reads the head's keys; in the week to `671e2ac9d`, 250 integration
commits moved the tree and 8 moved the digest (`07` D41). A re-keyed copy of a store law
published at the head's key: it would answer as fitted at the head, the re-validation
#2311 designs. A priced run per point before the search, `trace steps` and the coverage
check as `07` D37 answers a new workload: it adds a run per point beside the search's own,
which the fourth ruling rules out. The folds' steps pooled into one figure per kind: a
run priced high and one priced low cancel in a pooled total, where each run's own shows
both (README finding 2).

---

## D113. Calibrating what the search lacks

**Rule.**

- **The record list.** For a configuration missing a law or a decode rung, the step
  linear backend derives the records from the configuration alone: from its capture
  ladder, the concurrencies and the decode rungs, each recorded from every trajectory's
  first turn (#1840), and the trace runs. `compass advice` prints them as one
  `calibrate.sh` command with every argument and its expected time, unknown for a
  configuration never calibrated. Planning sits beside the backend (`07` D37).
- **`calibrate.sh`** publishes into the shared store `STORE` names, or, with none named,
  its own `store/` as today; its `GPU` names every card the configuration uses, as
  `--gpu` takes them; each step appends its start, end or failure to `status` in its
  output directory (`cal/<config>/` under the search's OUT), which `compass advice`
  reads; a failed calibration resumes where it stopped. The guide runs it under `setsid`
  or `nohup`, its exit trap unchanged.
- **A 1P1D configuration** takes one `calibrate.sh` command too, each record one
  `deploy.sh record` of both sides, and publishes into one store, since its candidate's
  runs read the one store holding both sides' inputs
  (`atom/compass/drive/reuse.py::stores`). It fits only the sides the reuse check sends
  to calibrate in that store, less a side an earlier candidate's calibration publishes
  there at the same key, since the store refuses a second publish at a key: that side is
  borrowed, as on one server (`atom/compass/advice/check.py::_borrowing`), with no
  readings command of its own where its readings key is the lender side's, as B's prefill
  side's is A's. Its store is the one a side's lender
  publishes into, an ended lender's too, when that holds nothing at the keys of the sides
  it fits, once the lender has started where it is the lender's own; else the one a lone
  server's calibration would take, judged over the sides it fits. Into a store of its own
  it fits both sides, as that store holds neither. Its records are the decode side's
  rungs at its DP width, on which a prefill side's host laws fit too, and the trace runs
  when it fits the prefill. Both sides take cards of one node: a configuration whose
  sides take more cards than one node holds gets why no command serves, as a deployment
  across two nodes is not one command, and lends no side. D116's configurations are the
  first 1P1D ones to need a new law (#4053); scenario 5 walks A, then B.
- **A readings gap is measured**: `compass measure memory` for the point, the command
  `compass run --check` prints, into the store its candidate's law is read from, the
  shared one for a checked-in law.
- **No top-up in place.** A store entry is immutable, a second publish at its key
  refused, and `calibrate.sh` takes no record once a fit has run: a configuration lacking
  a decode rung is calibrated again, old records with new, in a new output directory and
  into that calibration's own store. Every configuration that borrows its law, as
  MiniMax-M3's LMCache arms borrow their base arms' (D117), has its
  readings measured into that store too, `compass advice` printing them with the
  calibration command. A readings key holds no ATOM digest, so after an ATOM change an
  entry measured before it still holds its key: a calibration or a readings measure that
  meets one publishes into a new store too. The reuse check looks a candidate up in the
  latest store holding all its store inputs, a new one before the shared one, and the
  candidate's runs read that store; `compass run` keeps its one `--store`.
- **Timing.** Each calibration records each step's wall time (the warm record, `measure
  memory`, each record, the trace runs, `fit steps`, `fit host`) and its node-hours.

**Reason.** The backend whose law prices the steps knows which it lacks; an immutable
entry cannot change under a run that read it, so a gap is filled by a new law beside the
old one, and a borrower's readings go with it, or its one store would keep it on the old
law; the timing sets fitting-efficiency targets (#2678). **Basis.** The owner on #3355 and
#3363; `07` D37, D41; #2317; #1840; `atom/compass/artifacts/store.py`, whose entries
refuse a second publish; `atom/compass/artifacts/keys.py::OWN_FIELDS`, whose readings key
holds no ATOM digest; `atom/compass/drive/gpu.py::add_gpu_flag`, whose `--gpu` takes a
list.

**Rejected.** Topping a law up in place, the new records fitted into the configuration's
existing entry: store entries are immutable and `calibrate.sh` refuses a record once a fit
has run, and #3355 defers changing either.

---

## D114. Profiling: the per-request split and the op-family split

**Rule.**

- **The per-request split**: each request's queue, prefill, KV transfer and decode time,
  both 1P1D sides mapped to one request, and its LMCache store and load, read from the
  step record (`dp<r>.jsonl`), a P/D run's `transfers.jsonl` and aiperf's per-request
  export, which simulated and real runs both have, so one code path serves both. A real
  forward's entry and return are its record line's; a simulated one's are on its cost
  line, the engine's clock at its entry and that plus its charge. Every run writes it
  when it ends, into its own file in its run directory, never `summary.json`, changing
  none of the run's results: `client.sh --stop` for an instance, as it stops `compass
  run` or `compass record`, `deploy.sh` for a deployment, both sides joined. The
  segments run from the request's send to its first prefill forward's entry, to its
  final prefill chunk's return, to its decode side's first forward's entry, to its end
  at the client, so they sum to its end-to-end latency within the clock's resolution. A
  real deployment's decode node's clock is joined to its client's, the bounds its
  forwards and first tokens set stated in the file. On a real run its TTFT, from the
  send to the return of the forward that hands its first token over, matches aiperf's
  within a tolerance measured first; off a pipeline, the ATOM runner a client's tokens
  come from hands an output-producing forward's tokens over at the return of its rank's
  next one. A run the split cannot be made for gets the reason in its file, a
  pipeline's among them. Its format, `request_split.json`: each request aiperf finished
  as an `atom-benchmark/v1` request record with its segments, that TTFT, its LMCache
  tokens loaded and stored and its response id; every other request with why; the
  clock.
- **The op-family split**, op level only: the prices per op kind a `compass run
  --op-graph` run holds per step (`op.<kind>` in `dp<r>.cost.jsonl`), summed per step
  phase into MoE, attention, dense GEMM, collectives, norms and elementwise, and other,
  with no recomputation or estimate. Each phase's families reconcile with its predicted
  time, a remainder listed with its cause: host time, its steps' charge less their priced
  device time, which under a host law holds the DP group's slowest device end, and the DP
  group's slowest rank, their `group_s` less their charge; a refused op is reported as
  coverage. A term no one op kind holds shows whole with the reason, since a split by
  kind would estimate it: a TBO step's one `op.tbo` term, the finish time of its two
  micro-batches on a compute and a comm stream, and a `resource_bound` `Par`'s
  `par.resource_bound`, one roofline over its branches' summed work, which a run with a
  price list charges only where it is at least as long as the longest branch. That term
  hides the kinds it holds, so their families read short where it is charged until the
  split files a `Par` under one
  ([#3706](https://github.com/jgong5/ATOM/issues/3706)). A step-level run says it has no
  op split.

**Reason.** The split says where a request's time went, the same way for a simulated and
a real run; the op split adds no estimate. **Basis.** The owner on #3355; `01` D3.5;
`08` D50.1. #2763's one-off script split TTFT only and is not in the tree.

**Rejected.** The split inside `summary.json`: a run summary copies no latency of the
simulated system (`01` D3.5, rule 4).

---

## D115. The dashboard and the per-run report

**Rule.** Two static HTML views, served by no server, reading files and never calling the
search's or profiling's code.

- **The dashboard**, the search's `dashboard.html`: the frontier curves, then the
  recommendation and its trust, then each point opening its run's report. The main chart
  draws throughput per GPU against interactivity, both definitions of each switchable,
  with TTFT, total throughput, GPU count, versions and law sources. Each point carries two
  marks, its grading state (graded, ungraded, coverage report) and its pricing-method
  shares under `08` D50.1's names; a point with a refusal is grey with its reason. An
  optional reference layer, labelled with its source: none on the development template
  (D116), whose recipe publishes no results; on MiniMax-M3, the M3-Agentic stage's
  per-point verdicts on mi355-gpu-40, as another platform; never InferenceX's MiniMax-M3
  points. The layer is
  the file `compass advice --reference` names, its source and its curves, each point's
  metrics under the result table's names, so no code branches on the model. The page is
  Compass's own (`atom/compass/advice/dashboard.py`), its four views drawn when it is
  written and switched with no script; ATOM's Benchmark Dashboard was tried first and
  serves neither mark (T111, below).
- **The per-run report**, `report.html`, which every run, simulated or real, writes into
  its directory when it ends: the run (configuration, machine, ATOM tree, law sources
  from `run.json`, workload); headline metrics by #3264's code, as the dashboard computes
  them, and the speed ratio; trust, from `summary.json` (method shares of seconds,
  refusals, extrapolated and fallback steps as shares of steps, whether it is a coverage
  report); a graded cell's grades, from the `grades.json` that `python -m
  atom.compass.validate CELL...` writes into each cell it grades, the table it prints,
  rendering again the `report.html` of each run in that cell, since a run writes its own
  before grading; the per-request split, beside the real run's when the cell has one; the
  op-family split on op-level runs. Each number comes from a file in the run directory or
  its cell's `grades.json`, a part with no data says so, and the page changes none of the
  run's results. It links `prometheus.html`, or quotes `prometheus.txt`.
- **The Prometheus report**, `prometheus.html`, which every simulated run writes beside
  `report.html` when it ends (`python -m atom.compass.report.prometheus OUT`, after the
  split and before `report.html`): ATOM's CI report, `export_report.py` imported unchanged
  from the run's source tree, its panels for the run's deployment kind (`pd` when OUT holds
  `prefill/` and `decode/`, else `standalone`) over the run's series (`11` D74) from their
  first instant to the first grid point at or past their last, as CI's
  `wait_for_final_scrape` ends its query, at CI's 5 s step and `max(60, 4 x interval)`
  window, the interval the shortest step between two instants of one file, each file's
  last step left out, since its final sample lands under one interval after the timer's
  last (`11` D74). Each sampler's file is decompressed, since promtool reads no gzip, and
  backfilled by promtool into one TSDB that a Prometheus with no scrape
  serves to `collect_report`; both come from `collect_metrics.py::ensure_prometheus`, which takes a
  `tool` argument, reads `ATOMESH_PROMTOOL_BIN` beside `ATOMESH_PROMETHEUS_BIN`, and reuses a
  tool it extracted into the directory it is given, here `~/.cache/compass/prometheus-3.5.0`,
  so a node downloads the checksummed 3.5.0 archive once and one that cannot is given both
  binaries beforehand. Each file lands there under its name only once whole, written under
  a per-process name and renamed, the archive once its checksum holds, so `compass
  advice`'s parallel points, each stopped by `client.sh --stop`
  (`atom/compass/advice/simulate.py`), never run another's part-written tool; two racing on
  a node's first use each download the archive. The page says it is simulated through the
  data it renders: its title ("Simulated agentic PD inference report", "Simulated agentic
  aggregated inference report"), its notes (the simulated clock from the run's start shown
  from 1970-01-01 00:00 UTC, the GPU panels' predicted target device time on TP rank 0,
  any frontend or engine member with no file, any family it queries that the sampler's class table lacks, `11`
  D77), and the two GPU panels' label and description, which say predicted (`11` D75). A
  deployment's Overall
  TTFT, the router's `mesh_router_ttft_seconds`, stays empty and the notes say why: the
  router times it on the wall clock (`atom/mesh/src/observability/ttft.rs::track_http_ttft`),
  and the client's TTFT is on `report.html`. Its data goes to `prometheus.json`, from which
  `export_report.py --input-json` rebuilds the page. When none can be made (no series, a
  real run's included; no tool and no download; no `export_report.py` in the tree; a backfill
  or query that failed), `prometheus.txt` says why. It exits 0 either way; `client.sh` and
  `deploy.sh` report a failure and keep their exit status.

**Reason.** A static page travels with its run; reading files keeps the views apart from
the code they show (D108); two marks keep an ungraded or guessed number from reading as
measured. CI's own report shows a simulated run in the view a real run's is read in,
from series sampled on the simulated clock. **Basis.** The owner on #3355; `08` D50.1;
`11`, which owns `/metrics`; `atom/compass/validate/__main__.py`, which prints its grades
and writes none; for the Prometheus report, the owner's answers at #3584's plan review (no
Compass report for a real run; predicted GPU panels; the tools fetched or placed, a missing
one not failing the stop), and `export_report.py`'s public `collect_report` and
`write_report`.

**Rejected.** ATOM's own report in place of `report.html`: it plots Prometheus series and
holds no run, law source, trust or split, so it sits beside it. ATOM's Benchmark Dashboard
(`.github/dashboard/index.html`) for the search's dashboard (T111, settled by
[#3465](https://github.com/jgong5/ATOM/issues/3465)): its points are CI benchmark entries
from `data.js`, with no field for a grading state, method shares, a refusal or a
recommendation; its Disaggregated view (`renderAtomeshPerformanceTab`) switches between
InferenceX's two interactivities, but draws the throughput of the tokens served alone, with
no computed-token throughput; a point links only an `https://` URL (`safeHref`), never a
run's `report.html` beside the page; and it loads Chart.js from a CDN, so it draws nothing
offline. Each would change ATOM's CI page for one Compass view. For the Prometheus report:
moving `export_report.py` into `atom/`, which moves CI's module and its test for a caller
that imports it in place; reimplementing `rate` and `histogram_quantile` in Python, a second
copy of PromQL that drifts from CI's numbers; a "SIMULATED" badge in CI's page, an edit to
CI's template for what the title says; a router TTFT built from aiperf's, a client
measurement under the router's name; CI's `metrics/` layout, a second `report.html` in one
run directory; writing it after `report.html`, which could then only link a file it
cannot know exists; a grid shifted to end on the last instant, which then starts after the
first, unlike CI's from the benchmark's start; the interval read from
`ATOM_METRICS_UPDATE_INTERVAL_S`, which in the report's environment under `client.sh` need
not be the server's (#3630); and a lock around the tools' download,
which a killed report leaves blocking every later one, against a download each node makes
once (#3611).

---

## D116. The development template: MiniMax-M3.1-Flash's 1P1D recipe on pit2's MI355X, checked against held-out real runs

**Rule.** The search, profiling and views are developed, and `compass advice`'s answer is
checked, on MiniMax-M3.1-Flash's one-node 1P1D recipe,
[ROCm/ATOM `0368ec741` `recipes/mesh/MiniMax-M3.1-PD-PP4-MooncakeStore.md`](https://github.com/ROCm/ATOM/blob/0368ec7410ccbda844d743e474d36d4403092a83/recipes/mesh/MiniMax-M3.1-PD-PP4-MooncakeStore.md),
in its two decodes, under the `agentx` workload (`06` D35.1) for 900 s at c = 8, 16, 32,
64, 140 and 256, 140 the recipe's benchmark point and 256 the most either decode admits,
with no SLO, on pit2's MI355X nodes (pit2-p03-gXX). Its points are checked against
held-out real runs (*Validation*, below), reported and not gated: none of its
configurations enters `08` D50, and the dashboard marks every point ungraded, since a cell
`08` D44 grades holds two real runs (`atom/compass/validate/curves.py::point`). Its space
file writes the decodes as the values of one dimension and the prefill in `common` (D110).

| # | Prefill, GPUs 0-3 | Decode, GPUs 4-7 | Recipe | Law |
|---|---|---|---|---|
| A | PP4 x TP1 (`VLLM_PP_LAYER_PARTITION=16,16,16,12`), DSpark k=7, eager; a Mooncake Store its prefix tier (`ATOM_KV_OFFLOAD=mooncake_store`); chunked P->D transfer, its admission deadline 1500 s | TP4 with indexer CP (`ATOM_M3_INDEXER_CP=1`), DSpark k=7; `pd_admission` at 0.15 of the KV pool and 512 reserved tokens | `minimax_m31_pd_store/`: `store.sh`, `prefill.sh`, `decode.sh` at `MODE=tp4`, `router.sh` | calibrated anew, both sides |
| B | A's | DEP4: DP attention, MoE EP4 over MoRI, `--dp-load-balance least_requests`; A's admission, per DP rank | `decode.sh` at `MODE=dep4`, its tree with ROCm/ATOM PRs 3563 and 3564 merged | the decode calibrated anew; the prefill's law and readings A's, at the same keys (D112), which B's calibration borrows and publishes beside (D113) |

Each takes the recipe's arguments, variables and P/D options, NVFP4 KV and index cache on
both sides (`KV_DTYPE=nvfp4`) among them, and a side's options in its `kv_transfer` (D110):
the prefill's `enable_chunked_transfer` and `chunked_admission_timeout_s`, the decode's
`ib_device` and `pd_admission`, and each side's `protocol`. Both sides take
`FORCE_AL=4.04` as its benchmarks set it, since a simulated step draws its drafts from a
declared acceptance; a fractional stated length has the gaps #3355's template lists.

**Chunked transfer and `pd_admission` simulated**, as #4054 designs them before their
code, amending `01` D6 and D7. Under chunked transfer the router sends a request to both
sides at once ("Chunked mode establishes D before P finishes",
`atom/mesh/src/core/placement/backend/atom.rs::inject_decode_fields` at upstream
`a62134f99`): the decode admits and posts its write requests before the prefill
allocates, the prefill admits once every stage has reported the decode ready
(`MooncakeConnectorScheduler.admission_ready`, `pd_admission_ready`), each stage writes
its pages after each of its forwards, and the admission deadline runs on
`time.monotonic()`. That breaks `01` D6's premise that a write request exists only once
the prefill has returned (`r <= a`) and `01` D7's relay timestamp on the
prefill-to-decode forward. Under `pd_admission` ATOM's `PdAdmissionController` runs as
it is, once the connector states `supports_pd_admission` and answers
`pd_receive_block_offset`, so ATOM's exporter renders its three `atom:pd_admission_*`
families on the simulated decode too. A connector protocol is `01`'s, as the Store under
a pipelined prefill is (#4051), so this section states only that the template carries
both. **Rejected:** designing them here: `01` owns the connector, and this section's
table already sends the Store under a pipelined prefill to a design of its own.

**One tree for both decodes.** ATOM is the integration branch once it holds upstream main
`a62134f99` (#4047) and the two PR heads the recipe merges for B, each one commit on
`9f98bffc8`, which `a62134f99` holds: `1551211d0` (PR 3563, on upstream main since as
`7d2f16afe`) and `59e0a20ad` (PR 3564, open), merged by sha and labelled
`agent-team/land-as-merge` (#4048). A runs them too, since one frozen commit holds both
decodes (#4145's A1): PR 3564 changes only `atom/models/minimax_m3/dspark.py`, whose
refusal of more draft KV heads than a rank's TP width (`M3DSparkAttention.__init__`)
A's TP4 ranks, one draft head each, never meet. **Rejected:** B on the first upstream tip
holding both PRs, as #4048's brief had it: B idles on an upstream review, and that tip
brings commits neither recipe runs, five between `a62134f99` and `7d2f16afe` alone, the
DP state sync `49e3aee9a` among them.

**The departures**, those #4145's Goal 3 allows:

- **Capture sizes** thinned to powers of two, A's to 256 and B's to 64 a rank, against the
  recipe's 32 and 20 sizes, in calibration, search and validation alike: each decode rung
  is one record (D113), so the calibration takes about a third of the node-hours, as the
  [#3468 Ruling](https://github.com/jgong5/ATOM/issues/3468#issuecomment-6944202685)
  thinned the earlier template's.
- **The router** takes Compass's flags (D109), with no `--policy`, so atomesh's
  `cache_aware` (`atom/mesh/src/cliargs.rs`), and no `--dp-aware`: with one worker a side
  every policy picks that worker (`01` D7), as the recipe's `POLICY=random` does, and
  ATOM's `least_requests` places B's requests on its ranks; `dp_sticky` waits on #3496,
  which the template does not need.
- **What Compass assigns itself**: the ports, the servers', the handshake's and
  `ATOM_DP_MASTER_PORT`, and the `--kv-transfer-config` keys `deploy.sh` sets (D110).

Every record and run takes the prefill's `OFFLOAD_PROFILE=1`, as `prefill.sh` sets it;
the Store's two profiled records add `ATOM_LOG_LEVEL=DEBUG` alone (`01` D6.2's
calibration). **Rejected:** keeping `OFFLOAD_PROFILE` out of the step law's records:
Goal 3 does not list it, and the validation runs, which take it, would run a flag their
law's records never did.

The space file's machine names a network file of pit2's NICs, so `compass advice` starts
the decode apart under its own variables and prices the KV on that link (D109), as the
recipe moves it through the NICs (`MC_DISABLE_HIP=1`); every point takes the Store's
`mooncake_store` entry (D110).

**The frozen commit**, D110's `commit` in the space file: the first integration commit
holding every task the table below lists. Every calibration, readings measure,
environment check, simulated run and real run of the template is taken there, in the
image #4047 gates on, `rocm/atom-dev:nightly_202611240650`, named by digest; pit2's
platform readings and its network file, which key no ATOM digest, may come before
(#4145's A1, A3; D112 keys a law by its records' digest). **Rejected:** A at an earlier
commit, its law re-keyed onto B's (D112, requirement 2): A's real runs and its law's
records would run different trees, and a re-keyed law is not re-validated.

**Calibration**, on pit2 under a gpool lease of a whole node, at the frozen commit:

1. pit2's platform readings (`compass fit device`) and a network file of its NICs, probed
   as `atom/compass/networks/crs005-013-ionic.json` was; the Store's master and owners as
   `store.sh` starts them.
2. An environment check: each decode served once by the recipe's scripts, and the
   recipe's `gsm8k.sh <label> 32 200` through the router, which exits 0 and prints an
   accuracy; a decode that does not serve leaves the template until it does.
3. A: the warm record; both sides' readings, the prefill's with the Store on; D113's
   records at A's rungs and the prefill's trace runs; `01` D6.2's two profiled prefill
   records for the Store's entry. The fits run on CPU, after the lease.
4. B: the warm record, the decode's readings and its records, published into the store
   A's calibration publishes into; the prefill's law and readings are A's there (D113).

At the thinned rungs A takes about 7 node-hours and B about 4.3, against about 20 and 12
at the recipe's; the environment check about 1, estimated; step 1 is untimed. Derived
from #3468's steps on DeepSeek-V4-Pro at crs005, a record 2034 to 2059 s and the warm
record 907 s measured
([#3468](https://github.com/jgong5/ATOM/issues/3468#issuecomment-7001160408)), readings
0.1 h and a trace run 0.15 h estimated; `calibrate.sh`'s `status` records the real hours
(D113).

**Validation**, after the search, at the frozen commit and image, each run under a
whole-node gpool lease:

- **Points.** Each decode at c = 16, 32, 64 and 140, none a calibration input, and, when
  `compass advice` recommends c = 8 or 256, each decode there too
  ([#4146 Ruling](https://github.com/jgong5/ATOM/issues/4146#issuecomment-7156425733)):
  one real run a point.
- **A real run** is `deploy.sh record` at the point's arguments, variables and P/D
  options as `candidates.json` states them, driven by `client.sh --agentx --concurrency c --duration
  900`, the Store's master and owners started afresh before it (`store.sh stop`, then
  `start`), since a simulated point's LP `mooncake_store` starts empty (`01` D6.2).
  `one_real.py` reads a `deploy.sh` OUT's engine records and client export, and refuses a
  REAL and SIM given other client arguments. **Rejected:** the recipe's own scripts: they
  write no engine record or client export.
- **Each point's table** is `one_real.py REAL SIM`, SIM the search's run of that point:
  every metric #4145's A2 names, TTFT p50 and p99, TPOT p50 and p90, throughput per GPU
  and both hit rates, with SIM's error and REAL's own 95% interval over resamples of its
  60 s blocks of sends, each error marked in or out, as `one_real.py::tails` resamples
  a tail's, since a burst's requests move together. A hit rate's block is the change of
  the prefill's counters over 60 s of its scrape series, each draw the ratio of its sums.
  **Rejected:** resampling requests for the medians, as `one_real.py::paired` does for
  E2E p50: two methods in one table, and a burst moves a median's neighbours as it moves
  a tail.
- **A 1P1D point's hit rates are its prefill side's**, real and simulated: the HBM hits
  and the Store's loads, each over the prefill's prompt tokens. A real run's aiperf also
  scrapes the prefill's `/metrics`, named by `prefill/server.json`, and keeps the scrape
  as a series (aiperf's `jsonl`), since the run's own endpoint, the router, renders
  `mesh_self_metrics` (`atom/mesh/src/observability/metrics/routes.rs`), with no engine
  counter. A simulated run sums its prefill engines' `hit_counters` alone. **Rejected:**
  both sides' counters summed: a P/D decode counts its own local reuse over every prompt
  again (`Scheduler._record_cache_reuse` at upstream `a62134f99`), adding its received
  prompts to the denominator and the blocks it skips receiving to the hits.
- **Admission** (#4145's A4): at C140 on A, the decode's `/metrics`, read once the client
  run ends and before the stop, in the real run, driven by `client.sh` without `--stop`,
  and in a simulated run repeating the search's (`simulated.json`'s `ran`), which the
  same inputs repeat (*Summary*, item 6): the three `atom:pd_admission_*` families, the
  simulated values beside the real. A simulated frontend passes an unstamped `GET`
  straight through (`atom/utils/compass_loop.py::HttpChannel`). **Rejected:** the
  families added to the series a simulated run samples (`atom/compass/run/metrics.py::FAMILIES`)
  and to the real run's scrape, to report them over the run: more than A4's check reads.
- **The order** (#4145's A5): at the recommended point's concurrency, whether it and the
  runner-up (D111) keep their order by throughput per GPU in the real runs, or the gap
  lies within REAL's intervals.
- **The report**, on #4145: the tables, A4's values, A5's statement, each calibration
  step's node-hours from its `status` and the search's CPU seconds from
  `simulated.json`.

A real run takes about 0.6 h, a 900 s client beside both sides' start and a fresh Store,
estimated: about 4.8 node-hours for the eight, 1.2 more for the Ruling's pair.

**Who takes what.** #3468, re-briefed to `0368ec741`, takes the inputs and the answer:
calibration steps 1 to 4 and the Store's entry, then `compass advice` on the template
(#4145's A1, A3, A6). #4170, after it, takes the validation (A2, A4, A5) and the report.
#3469 keeps the guide's walk-through. **Rejected:** the validation folded into #3468: one
issue would hold two GPU campaigns and a report, and the validation reads advice's answer
first.

**What Compass lacks**, read at the integration head, each with its task:

| Lacks | For | Task |
|---|---|---|
| ATOM main `a62134f99` and image `nightly_202611240650`: the branch's upstream base, `b7b5ec3f2`, has neither MiniMax's PP-to-TP PAGE staging and head cut, nor NVFP4 over Mooncake P/D and the Store, nor `pd_admission`, its metrics and chunked admission, nor the shared kv_store layer | A, B | #4047, a sync |
| ROCm/ATOM PR heads `1551211d0` and `59e0a20ad`, which the recipe merges for B: without PR 3564, `M3DSparkAttention.__init__` refuses B's TP1 attention ranks | A, B | #4048, a sync of both heads, checking DEP4 |
| MiniMax-M3.1-Flash with its DSpark draft built and served on meta | A, B | #4015 |
| the simulated P/D connector at an NVFP4 KV beside a DSpark draft, which ATOM refuses | A, B | #4152 |
| a pipeline with a drafter, refused on a stale reason (`runner/overrides.py::NonAllocatingRunner._build_and_load_model`) | the prefill | #4049 |
| a simulated pipeline stage priced by its own part of the step linear law and host laws, and the stage labels checked on pit2 (`09` D54.1) | the prefill | #4069, #4071 |
| the Store under a pipelined prefill, which `01` D6.2 refuses, its entry one chunk size for every writer | the prefill | #4051, a design |
| each side's P/D options: `deploy.sh` fills each side's `--kv-transfer-config` and merges `KV_EXTRA` into both alike (D110) | A, B | #4165 |
| chunked transfer and `pd_admission` in the simulated connector (above) | A, B | #4054, a design of `01` D6 and D7, then its code |
| a TP1 prefill stage's write into A's TP4 decode, which `run/__init__.py::_planned_transfer` refuses (`01` D6) | A | #4166 |
| a point's GPUs and ranks counting pipeline stages: `validate/curves.py::gpus` and `advice/simulate.py` count TP x DP, so A's eight cards read as five (D109, D111) | A, B | #4167 |
| a real 1P1D run's hit rates, which aiperf's scrape of the router lacks, and a simulated one's from the prefill side alone | the validation | #4168 |
| REAL's interval for every metric A2 names: `one_real.py::tails` gives TTFT p99, p90 TTFT and TPOT p90 alone | the validation | #4169 |

`compass advice`'s check starts every point (D109), and whatever MiniMax-M3.1 forces to
change there is filed as a defect, as for MiniMax-M3 (D117). #3624 does not block: a
producer schedules no decode rung (`atom/compass/drive/lacking.py::uncovered_rungs`), and
the prefill runs eager. The template needs no `03` change: NVFP4 KV sizing is ATOM's
arithmetic over measured readings (`03` D13, D14).

The configurations cut from it can come back: DeepSeek-V4-Pro's S1 (TP8, MTP3), S2 (DPA8,
MTP3, TBO), S5 (S2 with LMCache), P1 (1P1D TP8) and P3 (1P1D DPA8), with S3, S4, P2 and P4
to P7 cut before them; and the recipe's `MODE=mega`. DeepSeek-V4.1-Flash, which #3468's
Ruling of 2026-11-19 named, never entered it.

**Reason.** The owner's Rulings on #3670 move the template to pit2's MI355X under the
team's GPU pool, crsuse2-m2m-v3-005 staying reserved for stage 3d/3e
([first](https://github.com/jgong5/ATOM/issues/3670#issuecomment-7127634115)), and to
this recipe in its two decodes
([second](https://github.com/jgong5/ATOM/issues/3670#issuecomment-7130535680)),
superseding #3468's move to DeepSeek-V4.1-Flash; #4145's requirement takes it to
`0368ec741` as written and checks advice's answer against held-out real runs. Both decodes
share one prefill, so B's calibration borrows A's prefill law and B's points reuse it: the
template runs a new law and a borrowed one, and A against B puts the recipe's own
comparison, TP4 against DEP4 on one AgentX workload, on its frontier, where the real runs
check it. **Basis.** The Rulings; #4145's requirement and the
[#4146 Ruling](https://github.com/jgong5/ATOM/issues/4146#issuecomment-7156425733); the
recipe and its scripts at `0368ec741`; D112's keys (`atom/compass/drive/reuse.py`,
`atom/compass/drive/lacking.py::_producer`); D113 and
`atom/compass/backends/step_linear_plan.py::records`; `01` D6, D6.2;
`scripts/compass/one_real.py`.

**Rejected.** Departing from the recipe where Compass lacks a feature, a TP prefill, one
without the Store or the P/D options left unset, to calibrate sooner: the second Ruling
names the PP4 x TP1 prefill and its Store tier among the configurations' parts, and
#4145's Goal 3 the recipe as written.

---

## D117. MiniMax-M3: the move, its template and the showcase

**Rule.**

- **The move**, once the Compass features MiniMax-M3 needs have landed and the M3-Agentic
  stage's small case (#3265) passes, whatever the development template's work has
  reached; #3256 (`16` D95.1) tracks that support.
- **Our platform.** Laws, readings, offload terms and real runs are taken on the
  development nodes at one frozen commit, which the template's space file states (D110):
  every candidate's law is fitted there, a recalibration included, and the showcase's
  simulations and validation runs run there, since a law's version is the commit it was
  fitted at, not only the code a simulation runs. The M3-Agentic stage runs on
  mi355-gpu-40 at a frozen commit of its own, its spec pinning another aiter (`09d256420`
  against crs005's `41d4d8d7e`) and so another platform key, on another host CPU: its
  code and method are reused, not its laws, readings or real runs.
- **The environment check**, first: each of #3256's arms served once with the
  submission's arguments, as #3258 did on gpu-40: whether it comes up, fits memory at
  `--gpu-memory-utilization 0.95`, has the host memory its LMCache tier pins, places
  across NUMA nodes and reads its checkpoints; an arm that does not leaves the template.
- **The template**: #3256's arms with the submission's arguments as #3256 records them,
  `--max-num-seqs` twice the concurrency, EAGLE3 at `14` D87.1's forced length and the
  `agentx` workload (`06` D35.1), each arm at every concurrency InferenceX ran; a point
  that cannot be priced is a grey coverage report. A law serves each point whose capture
  ladder is a prefix of its own, so an arm's law keyed at its top point serves its lower
  ones (#3275, T113).
- **Calibration** as `16` D95.1's, after the environment check: a base law each for TP4,
  TP4 with indexer CP and TP2, the last two pooling their LMCache arms' records, with `03`
  D13.1's offload terms, only if those arms come up; a readings entry per point; each step
  timed (D113). Every point is priced by the step linear law alone, departing from
  [#3256's ruling](https://github.com/jgong5/ATOM/issues/3256#issuecomment-6822133541),
  which prices the M3-Agentic stage's cells with the step grid beside it because the law's
  records end near 255k tokens a request while the submission serves contexts to 1048576:
  the search runs with no step roofline (D109), while every step grid run composes one
  below its grid (`17` D102), and the coverage check (D112) and the record list (D113) are
  the step linear law's. The `agentx` client sends every cc-trace at full length (`06`
  D35.1), so a step past the law's records is priced by extrapolation, never refused,
  counted apart and shown as its point's extrapolated share (`08` D49), unless the
  calibration records it. If the M3-Agentic stage's grading shows the step linear law
  failing `08` D45.1 where the step grid passes, on a cell the showcase runs, the
  showcase's pricing is a design change before the showcase runs (the
  [fourth ruling](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6857982216),
  correcting the second's last sentence).
- **The showcase run**: `compass advice` on that template, the model differing only in
  the space file, no search, profiling or view code branching on it; whatever MiniMax-M3
  forces to change is filed as a defect; an op-family split only where the op-level
  backend prices the cell (#3126 records MiniMax-M3's sparse layers as missing at op
  level). Its CPU time, the machine stated, is set against an InferenceX-style sweep's GPU
  time, 3600 s a point plus the measured start cost.
- **The showcase validation**, an `08` D50 set registered and frozen before it runs: at
  the recommended point's concurrency, the recommended configuration and the runner-up
  (D111) each run real three times on the development nodes, spaced (`08` D45), 3600 s
  each as the `agentx` workload runs. Each cell is graded by `08` D44, and their real
  order by throughput per GPU, InferenceX's, is compared with the simulated one: it
  matches, or the gap lies within the real runs' spread and is reported undetermined.
  Each point also reports its `08` D45.1 verdict beside its grade, reported and not
  gated, to compare with the M3-Agentic stage's step linear verdicts, a simulated run's
  hit rates being its engines' own counters across the client run (`08` D45.1, #3326).
  The M3-Agentic stage's verdict on the same arm at the same concurrency stands beside
  too, labelled as another platform. The simulation is not
  compared with InferenceX's published results. How calibration and grading divide the
  traces is T112.

**Reason.** MiniMax-M3 is ATOM's AgentX submission to InferenceX, for which the M3-Agentic
stage builds what Compass needs, so the search is shown on a space users deploy; a law is
keyed to its platform and host; the recommendation against its runner-up is the choice a
user acts on, and their order is what real runs confirm. **Basis.** The owner on #3355 and
#3363, the second ruling settling the pricing and the overlap verdict and the fourth what a
step past the law's records costs and when the pricing is reopened; `16` D95.1; `08` D44,
D45, D45.1, D49; `14` D87.1; `03` D13.1; `06` D35.1; `17` D102.

**Rejected.** Pricing with the step grid beside the step linear law, as #3256's ruling has
the M3-Agentic stage do: it reaches the step roofline the search runs without, and the
coverage check and record list that complete a law are the step linear law's; the stage's
grades reopen it.

---

## Simpler alternative

A guide over the commands that exist, with no `compass advice`: `compass run --check` per
candidate (`atom/compass/drive/lacking.py`), which lists each missing or stale input with
the command that makes it; `compass run` and one `client.sh` run per point; and `python
-m atom.compass.validate --curves` for the curves. It fails #3355: the search is not one
command; `--check` holds no law's decode rungs against the ladder and has no accuracy
bar, so reuse requirements 3 and 4 go unjudged; `--check` refuses a prefill or decode
side's arguments (`13` D80), so no 1P1D candidate is checked; and a user step needs
`python -m`, which the owner's ruling on #1108, recorded in #1527, rules out.

For D116's template, the simplest is the one #3670 left: the P/D options unset, the real
runs taken by the recipe's own scripts and `one_real.py` as it is. It fails #4145: Goal 3,
the recipe as written, its chunked transfer and `pd_admission` among it; A4, with no
simulated admission to export; and A2, a real 1P1D run having no hit rate, no engine
record or client export for `one_real.py` to read, and three intervals of the seven
metrics A2 names.

---

## Low confidence

Each entry is a choice for D116's template that no measurement settles, the owner's to
overturn. The earlier entry for D110's space file against the config file `13` D79 rejects
is settled by the
[sixth ruling](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6871001927), and the
one for the P/D options left unset by #4145's Goal 3, which carries them.

- **Thinned capture sizes.** The decodes calibrate at powers of two, about 11 node-hours
  for both, against about 32 at the recipe's sizes; the template then departs from the
  recipe in the batches its graphs pad to. It rests on the #3468 Ruling that thinned the
  earlier template's, a Ruling on another template.
- **A 1P1D point's hit rates are its prefill side's.** Which counters a 1P1D hit rate
  reads is a definition, and none published settles it: InferenceX's ATOM adapter sums
  the counters of every endpoint aiperf scrapes, with no role filter
  ([`backends/atom.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/4b94f096b74191b8af6b7c89e50e0b6bc69f7aba/inferencex-e2e/infx/results/agentic/backends/atom.py)),
  and publishes no 1P1D hit rate. The template counts the side that computes the prompt;
  another definition changes no simulation, only `client.sh`'s scrape and the sum in
  `validate/curves.py`.

---

## Dependencies

No spike: no decision waits on a measurement. D116's template waits on two syncs with
ROCm/ATOM, each labelled `agent-team/land-as-merge`: #4047 takes upstream main `a62134f99`
and the image `nightly_202611240650`, and #4048, after it, the PR heads `1551211d0` and
`59e0a20ad`, which decode B needs and A runs too. Two designs settle what D116 leaves to
them, each amending `01`: #4051 the Store under a pipelined prefill, #4054 chunked
transfer and `pd_admission` in the simulated connector. Its other tasks are in
[D116's table](#d116-the-development-template-minimax-m31-flashs-1p1d-recipe-on-pit2s-mi355x-checked-against-held-out-real-runs)
of what Compass lacks; #3468 takes the template's inputs and answer, #4170 its validation. No configuration
of the template offloads through LMCache, so #3712's two points on their own LMCache ports
have no subject in it. Other issues settle what decisions leave to them: #3256's
grading, which reopens the showcase's pricing if it fails the step linear law where the
step grid passes on a cell the showcase runs (D117); #3054 the indexer-CP key MiniMax-M3's move waits on (D117); #2311 who judges a change that
moves step costs (D112); `12` T107 trees other than the head, after v1 (D112); #1498 the
op-family split's inputs on DeepSeek-V4-Pro (D114).

---

## Summary

**Deliverable.** `compass advice`, which searches a space the user provides and recommends
a configuration; the reuse check it calls and the calibration it prints; each run's
per-request split, op-family split and report; the dashboard; the guide. Built on
MiniMax-M3.1-Flash's 1P1D recipe on pit2's MI355X, its answer there checked against
held-out real runs, and shown on MiniMax-M3.

**Scope and non-goals.** How a user finds the configuration to deploy, what each run
writes for a person to read, and where the tool is built, on pit2's MI355X, and shown, on
the development nodes. Out: #3355's *Deferred* list, among it topping a law up in
place, less search from any host (D109); for D116's template, `MODE=mega`, other models
and nodes, an accuracy threshold on its real runs and any `08` D50 entry; and work the
search meets, such as `inferencex_export` labelling every row
single-node, tracked apart.

**Changes.**

| Where | Change | Decisions |
|---|---|---|
| `atom/compass/advice/` (new), `atom/compass/cli.py` | `compass advice`: the space file, its Store entry included, the check's states, the simulation, the results, `dashboard.html` | D109, D110, D111, D115 |
| `atom/compass/drive/reuse.py` (new), `drive/lacking.py`, `drive/run.py`, `run/__init__.py::atom_digest` | the reuse check, the head's digest taken over the `atom/` a commit holds; `run --check` on a P/D side, resolved as `deploy.sh` does, and naming a step linear law's scheduled decode rungs with no vector | D109, D112 |
| `atom/compass/backends/step_linear_plan.py` (new), beside `step_grid_plan.py` | the record list | D113 |
| `atom/compass/backends/step_linear_fit.py` | `08` D47's three figures per kind over each leave-one-run-out fold, as `validate/cost.py` computes them, in the law and its report; the records' ATOM digest, `records_atom` | D112 |
| `scripts/compass/calibrate.sh`, `drive/lacking.py`, `advice/check.py` | `STORE`, a GPU list, `status`, per-step timing; a 1P1D configuration, each record a `deploy.sh record`, fitting only the sides its store will lack, a side an earlier candidate's calibration publishes there borrowed | D113 |
| `atom/compass/report/` (new), called where `client.sh --stop` and `deploy.sh` stop a run | the per-request split, the op-family split, `report.html`, a simulated run's `prometheus.html` | D114, D115 |
| `atom/compass/run/metrics.py` (new), `run/__init__.py`, `runner/overrides.py`, `drive/run.py`, `.github/scripts/atomesh/observability/collect_metrics.py` | the series the Prometheus report reads, its predicted GPU panels, and its tools (`11`, *The first build*) | D115 |
| `atom/compass/validate/__main__.py`, `validate/curves.py` | `grades.json` in each graded cell, its runs' `report.html` rendered again; a 1P1D point's GPUs over both sides | D115, D111 |
| `scripts/compass/spaces/` (new) | the development template, MiniMax-M3.1-Flash's 1P1D, and MiniMax-M3's | D110, D116, D117 |
| `scripts/compass/ADVICE.md` (new), `QUICKSTART.md` | the guide, linked both ways | D108 |
| `atom/compass/advice/space.py`, `advice/check.py`, `advice/simulate.py`, `drive/run.py`, `scripts/compass/deploy.sh`, `atom/compass/kv/connector.py` | a side's `kv_transfer`, merged into its `--kv-transfer-config`, real and simulated, `run --check` included; the `compass` connector refusing an option it does not carry | D110 |
| `atom/compass/run/__init__.py::_planned_transfer`, `runner/overrides.py::publish_planned_transfer` | a TP1 prefill stage's write into a wider decode, cut to each decode rank's heads (`01` D6) | D116 |
| `atom/compass/validate/curves.py::gpus`, `advice/simulate.py` | a point's GPUs and ranks counting pipeline stages | D109, D111 |
| `scripts/compass/client.sh`, `validate/curves.py` | a 1P1D point's hit rates from its prefill side, real and simulated, the real scrape kept as a series | D116 |
| `scripts/compass/one_real.py` | REAL's interval for every metric #4145's A2 names | D116 |

**Acceptance items**, each walked by the scenario named:

1. A space file, by hand or filled by an agent from the recipe D116 names, expands twice
   into the same candidates; an unknown field, contradictory values or an empty expansion
   is refused by name (1).
2. Every candidate is searchable, missing a law with its `calibrate.sh` command,
   calibrating with its progress, or refused verbatim; every point missing readings
   carries its `measure memory` command; the search calibrates nothing (2).
3. Each reuse verdict names its requirement: a tree whose ATOM digest differs from the
   head's refused with both digests, a key mismatch, an invalidation, a scheduled decode
   rung with no vector listed before the search, and a `08` D47 figure past 5% on a
   held-out run, refused at the head and sent to calibration when re-keyed; after the
   search each point reports its extrapolated share of steps (3).
4. A recalibration is printed with its borrowers' readings commands, and both run on the
   new store (4).
5. A 1P1D candidate is checked per side as `deploy.sh` resolves it; calibrated by one
   command fitting only the sides its store will lack, B's naming A's calibration for its
   prefill and publishing its decode beside it, no publish refused; simulated by
   `deploy.sh run` with the Store's entry its space file names; and counts both sides'
   GPUs (5).
6. Each searchable point leaves a run directory with `summary.json` and `report.html`; a
   rerun skips finished points; the same inputs give the same results; a changed SLO
   simulates nothing (1).
7. InferenceX's metrics equal #3264's on the same export; the frontier holds exactly the
   points no other dominates; the recommendation and the runner-up follow D111 (1).
8. An agent following the guide fills the development template's space file and
   reaches the recommendation, starting a calibration only on the user's yes (6).
9. A request's split segments sum to its end-to-end latency within the clock's
   resolution; an op-level run's families reconcile with its predicted time (7).
10. The dashboard and each report read files only; each number on a report comes from its
    run directory or its cell's `grades.json`, a graded cell's reports rendered again by
    grading; a part with no data says so (7).
11. The showcase calibrates and runs at one frozen commit, prices by the step linear law
    alone, a step past its records extrapolated and its share reported, grades both cells
    by `08` D44 with their order compared, and reports each point's `08` D45.1 verdict
    beside its grade, not gated (8).
12. Every simulated run's end writes `prometheus.html` beside `report.html`, CI's panels
    over its series on the simulated clock, or `prometheus.txt` with why, its stop not
    failing; `report.html` links or quotes it (7; `11`, *The first build*).
13. D116's template meets #4145's A1 to A6 at its frozen commit and image: advice answers
    both decodes; a `one_real.py` table per held-out point gives every A2 metric with
    SIM's error and REAL's interval; the recipe serves; C140's `/metrics` export the
    admission families, simulated beside real; the real order at advice's pick; each
    step's cost (9).

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D108 | Fitting apart from searching: a candidate is simulated once its law exists; the halves meet only through the store and a calibration's `status` file, the search owning no calibration, fitting, reuse judgement or version management. A user guide, `scripts/compass/ADVICE.md`, exact enough that an agent following it fills the development template's space file, and no skill; an agent starts a calibration only on the user's yes. | 2026-11-13 ([#3355](https://github.com/jgong5/ATOM/issues/3355), [#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6845852318)) |
| D109 | `compass advice SPACE OUT` checks each candidate (searchable, missing a law, calibrating, refused) and each point's readings (missing readings, with the `measure memory` command `run --check` prints), simulates each searchable point on a fresh server with no step roofline, and writes the results; it never calibrates, and runs at the integration head, on any host's CPU with no GPU device. Amends `13` D80: it starts the client and takes a workload, and `run --check` resolves a prefill or decode side. | 2026-11-13 (#3355, #3363); any host's CPU with no GPU device 2026-11-24 ([#4006 Ruling](https://github.com/jgong5/ATOM/pull/4006#issuecomment-7130990480)) |
| D110 | The user provides the space file, by hand or through an agent following the guide from what the user asks for, a recipe or real-run arguments; it names the shared store and, for a frozen search, its commit; it expands deterministically into candidates, each naming its ATOM basis and departures; the templates live in `scripts/compass/spaces/`, no default; it is the search's input, not a level of `13` D79's precedence nor the config file it rejects. Its format: one JSON object, the dimensions' product less the exclusions, each candidate `common`'s block and its values', two blocks setting one thing apart refused; a candidate offloading to a Mooncake Store names its run-file entry, which every point takes; a 1P1D side states its P/D options, `kv_transfer`, merged into that side's `--kv-transfer-config`. | 2026-11-13 (#3355, #3363); D79 settled 2026-11-14 ([#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6871001927)); the format 2026-11-14 (#3448); the `lmcache_server` field, each offloading point's port and the router's flags 2026-11-15 ([#3477](https://github.com/jgong5/ATOM/issues/3477#issuecomment-6887140315)); the `mooncake_store` field 2026-11-24 ([#3670](https://github.com/jgong5/ATOM/issues/3670)); a side's `kv_transfer` 2026-11-25 ([#4146](https://github.com/jgong5/ATOM/issues/4146)) |
| D111 | InferenceX's metrics by #3264's code, and the search's computed-token throughput and P90 end-to-end latency from the same export; the frontier of throughput per GPU against P90 interactivity; the recommendation, the most throughput per GPU meeting the SLO, which bounds P90 interactivity, with no refusal; the runner-up, another configuration's best at its concurrency; the result table and the recommendation's text. A 1P1D point's GPUs are its prefill side's plus its decode side's, each from its own engine record, tensor times pipeline times data parallel. | 2026-11-13 (#3355); the GPU count 2026-11-14 ([#3454](https://github.com/jgong5/ATOM/issues/3454)); pipeline stages counted 2026-11-25 ([#4146](https://github.com/jgong5/ATOM/issues/4146)) |
| D112 | Calibrated data serves a candidate only with the same key, not invalidated, enough coverage (a decode vector for every scheduled rung before the search, each point's extrapolated share reported after it) and `08` D47's three figures on each leave-one-run-out fold within #627's 5% gate, a law at the head past it refused; a law's verdict is reuse or calibrate, a point's readings' reuse or measure, and a tree whose ATOM digest is not the head's, the remote tip's or a frozen commit's, is refused naming both. A store law reaches a later head only checked in re-keyed. A search may mix laws fitted at different commits, each point's provenance naming its law's commit; the showcase's are all fitted at its one frozen commit. | 2026-11-13 (#3355, #3363; the head by ATOM's digest, [#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6854911957); coverage in two parts, [#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6857982216); the accuracy bar, [#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6869501859)); requirement 2's readings taken again by `07` D41's rule, 2026-11-17 ([#2899](https://github.com/jgong5/ATOM/issues/2899)) |
| D113 | The step linear backend lists a configuration's missing records, for a missing law or decode rung, as one `calibrate.sh` command, which takes a GPU list, publishes into `STORE` and reports each step in `status`; a readings gap is measured alone; with no top-up in place, a configuration lacking coverage is calibrated again into a new store, the readings of every configuration borrowing its law with it, as is a publish meeting a readings key an older tree's entry holds, and each candidate runs on the latest store holding its store inputs; each calibration is timed per step; a 1P1D configuration takes one command, its records `deploy.sh record`s, publishing into one store and fitting only the sides that store will lack, a side an earlier candidate's calibration publishes there borrowed. | 2026-11-13 (#3355, #3363); the 1P1D calibration due 2026-11-24 ([#3670](https://github.com/jgong5/ATOM/issues/3670)) |
| D114 | Every run writes its per-request split when it ends, apart from `summary.json`; an op-level run's op-family split sums pricing's prices per step phase. Amends `13` D80's outputs of `run`, `record` and `deploy.sh`. | 2026-11-13 (#3355); the format 2026-11-14 ([#3458](https://github.com/jgong5/ATOM/issues/3458)); a `resource_bound` `Par` shown whole 2026-11-18 ([#3707](https://github.com/jgong5/ATOM/pull/3707#issuecomment-6968684379)), host time named beside the DP group's slowest rank the same day ([#3707](https://github.com/jgong5/ATOM/pull/3707#discussion_r4609386973)) |
| D115 | Two static views reading files only: the search's dashboard, a page of Compass's own, each point marked by grading state and method shares, with a reference layer from the file `--reference` names; and every run's `report.html`, a graded cell's grades read from the `grades.json` grading writes into the cell, which renders the cell's reports again. Beside `report.html`, every simulated run's `prometheus.html`: CI's report, `export_report.py` unchanged, over the run's series (`11` D74), marked simulated, the GPU panels predicted, the router's TTFT empty; or `prometheus.txt` with why, the stop not failing. | 2026-11-13 (#3355, #3363); the page Compass's own 2026-11-15 ([#3465](https://github.com/jgong5/ATOM/issues/3465)); the Prometheus report 2026-11-16 ([#3584](https://github.com/jgong5/ATOM/issues/3584)); its query's end and the tools written whole 2026-11-17 ([#3611](https://github.com/jgong5/ATOM/issues/3611)); no reference layer on the development template 2026-11-24 ([#3670](https://github.com/jgong5/ATOM/issues/3670)) |
| D116 | Developed on MiniMax-M3.1-Flash's one-node 1P1D recipe (ROCm/ATOM `0368ec741`) on pit2's MI355X: one PP4 x TP1 prefill with DSpark, NVFP4 and a Mooncake Store tier, writing chunked into its decode; decode A, TP4 with indexer CP, and decode B, DEP4, each admitting by `pd_admission`; both at one frozen commit holding the PR heads B needs; each calibrated anew under a gpool lease, B's calibration borrowing A's prefill law; searched at c = 8 to 256 for 900 s with no SLO; checked by `one_real.py` against one held-out real run a point at c = 16 to 140, and at advice's pick when it is 8 or 256, a 1P1D hit rate the prefill side's, reported and outside `08` D50; each departure from the recipe stated; DeepSeek-V4-Pro's S1, S2, S5, P1 and P3 cut. | 2026-11-13 (#3355); P1 calibrated anew 2026-11-16 ([#3493](https://github.com/jgong5/ATOM/pull/3493#issuecomment-6906479025)); moved to pit2's MI355X and to the MiniMax-M3.1-Flash recipe 2026-11-24 ([#3670](https://github.com/jgong5/ATOM/issues/3670#issuecomment-7127634115), [Ruling](https://github.com/jgong5/ATOM/issues/3670#issuecomment-7130535680)); moved to `0368ec741` and checked against held-out real runs 2026-11-25 ([#4145](https://github.com/jgong5/ATOM/issues/4145), [#4146 Ruling](https://github.com/jgong5/ATOM/issues/4146#issuecomment-7156425733)) |
| D117 | Moved to MiniMax-M3 once its Compass features land and #3265 passes, calibrated on the development nodes from #3256's arms at one frozen commit, where its simulations and validation runs run too, every point priced by the step linear law alone, a step past its records extrapolated and reported, and the pricing reopened if #3256's grading fails the law where the step grid passes ([#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6857982216)); the showcase validation runs the recommended configuration and the runner-up real, graded by `08` D44, their real order compared with the simulated one, each point's `08` D45.1 verdict beside its grade, not gated. Amends `08` D50. | 2026-11-13 (#3355, #3363) |

---

## Open items

T108 to T114, registered in [`12_open_items.md`](12_open_items.md) under *Topic 19*.
