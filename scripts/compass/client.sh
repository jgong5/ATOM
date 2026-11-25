#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# One client run against a `compass record` or `compass run` server: aiperf
# with cctraces_sim.sh's arguments over traces A to B, or with InferenceX's
# agentx client over every trace, then the run's export.
#
# Usage: client.sh OUT --traces A-B --concurrency C [--duration S] [--stop]
#        client.sh OUT --agentx --concurrency C [--duration S] [--stop]
#        client.sh OUT --sweep --traces A-B --stop
#   OUT              the server's --out, whose server.json names its URL and,
#                    for a simulated server, its run file; a deploy.sh OUT's
#                    prefill/server.json names its prefill instance's URL
#   --traces A-B     TRACES' traces A to B, each counted from 0
#   --concurrency C  aiperf --concurrency
#   --duration S     aiperf --benchmark-duration, seconds on the server's clock
#                    (default 1800, 3600 with --agentx)
#   --agentx         in place of --traces, the defaults of agentx-harness's
#                    agentx scenario, which InferenceX's agentic submissions
#                    run, as flags on the base scenario: every trace in TRACES
#                    (the scenario's is the full-length corpus; a warmup
#                    request longer than the server's context aborts aiperf's
#                    run, so a shorter context takes the 256k-capped copy), seed
#                    42, a trajectory start between 0.25 and 0.75 of each
#                    trace, 10 warmup requests per lane (--warmup-per-lane N)
#                    with 1800 s grace, a
#                    per-trace idle cap of 300 s, a system idle cap of 10 s,
#                    the chat template applied, and against a real server
#                    aiperf's scrape of its /metrics, which InferenceX takes
#                    hit rates from
#   --stop           then SIGTERM the command that started the server, wait
#                    WAIT_S (default 600) for it to write OUT/run.json and leave
#                    (as a client run that fails once aiperf has run does too),
#                    a trace run's (OUT/trace) serve.TRACE_STOP_S more, set
#                    the export's server_stopped and, for an instance, write
#                    each request's split, OUT/request_split.json
#                    (atom/compass/report/split.py), then CI's Prometheus
#                    report of a simulated run's series, OUT/prometheus.html,
#                    or why there is none, OUT/prometheus.txt
#                    (atom/compass/report/prometheus.py), a failure of which
#                    fails no stop, then the run's report, OUT/report.html
#                    (atom/compass/report/html.py), all of which deploy.sh
#                    writes for a deployment as it stops
#   --sweep          in place of aiperf, send the JIT sweep of the cell the
#                    server's engine and traces A to B make, then after the
#                    stop keep the points its record covers in the server's
#                    JIT cache library (atom/compass/drive/sweep.py); only
#                    points not yet covered are sent, and the last line says
#                    whether the cell is swept. Refuses a server that
#                    `compass record --sweep` did not start. Needs TRACES alone
#   MODEL, TRACES    as cctraces_sim.sh takes them
#   HARNESS_PYTHON   the Python with agentx-harness, and with compass-harness
#                    when the server is simulated and only then
#   AIPERF_EXTRA_ARGS  more aiperf arguments, after client.sh's own, as
#                    cctraces_sim.sh takes them (default none); a --random-seed
#                    there replaces client.sh's; aiperf appends a --url, --model
#                    or --extra-inputs there to client.sh's, and refuses any
#                    other option given twice. aiperf reads its AIPERF_* settings,
#                    its timeouts among them, from client.sh's environment, and
#                    runs with HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE unset
#   COMPASS_HARNESS_SEND_ORDER_SEED  against a simulated server, the order its
#                    traffic LP numbers the requests aiperf sends at one
#                    instant in, which its router places them in: unset, by
#                    each one's phase, conversation and turn, so a run repeats;
#                    set, by a digest of the seed and those, one draw of that
#                    order per seed, the same in every run. The client run's
#                    export names it, send_order_seed; the workload does not,
#                    as a real run has no such order (compass_harness/traffic_lp.py)
#
# A simulated server's client paces aiperf on the server's clock through this
# tree's compass-harness; a real one's pins aiperf's benchmark id as
# compass-harness does, so both send the same prompts, after the scenario's own
# warmup. OUT/client/ holds the traces aiperf is given, each a file
# named by its line in TRACES, aiperf's dataset and its log; OUT/artifacts/
# aiperf's artifacts at --export-level raw, its raw records with no payload
# and gzipped (profile_export_raw.jsonl.gz), that scrape's export
# (server_metrics_export.json) where it ran, and the client run's export,
# client_run.json (atom/compass/parity/client_run.py), and its workload,
# workload.json, as cctraces_sim.sh writes it into run.json, so
# `python -m atom.compass.validate` grades the two in one cell. One client run per OUT:
# an OUT/client/ there refuses. The server keeps serving unless --stop. Exits 2
# on bad arguments, 90 to 99 as _lib.sh's tree checks refuse, else 1 naming the
# step that failed. aiperf exiting 1 with the scenario's coverage check as its
# only exit error is no failure: this prints that check's line and goes on.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

USAGE="usage: client.sh OUT --traces A-B --concurrency C [--duration S] [--stop] | OUT --agentx --concurrency C [--duration S] [--warmup-per-lane N] [--stop] | OUT --sweep --traces A-B --stop"
bad() { echo "client: $1; $USAGE" >&2; exit 2; }
fail() { echo "client: $1" >&2; exit 1; }
ROOT=$(compass_tree_root) || exit $?
(($#)) && [[ $1 != -* ]] || bad "no OUT"
OUT=$1 stop=0 sweep=0 agentx=0
shift
while (($#)); do
    case $1 in
    --traces)
        [[ ${2-} =~ ^(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$ ]] && ((BASH_REMATCH[1] <= BASH_REMATCH[2])) ||
            bad "--traces takes A-B, A at most B"
        first=${BASH_REMATCH[1]} last=${BASH_REMATCH[2]}
        shift ;;
    --concurrency) [[ ${2-} =~ ^[1-9][0-9]*$ ]] || bad "--concurrency takes a positive integer"; c=$2; shift ;;
    --duration) [[ ${2-} =~ ^[1-9][0-9]*$ ]] || bad "--duration takes whole seconds"; duration=$2; shift ;;
    --stop) stop=1 ;;
    --sweep) sweep=1 ;;
    --agentx) agentx=1 ;;
    --warmup-per-lane) [[ ${2-} =~ ^(0|[1-9][0-9]*)$ ]] || bad "--warmup-per-lane takes a whole number"; warmup=$2; shift ;;
    *) bad "unknown argument $1" ;;
    esac
    shift
done
[[ -z ${warmup-} ]] || ((agentx)) || bad "--warmup-per-lane is the agentx scenario's warmup, and takes --agentx"
if ((agentx)); then
    ((!sweep)) && [[ -z ${first-} ]] || bad "--agentx takes every trace, and neither --traces nor --sweep"
    [[ -n ${c-} ]] || bad "--concurrency is required"
    need=(MODEL TRACES HARNESS_PYTHON)
elif ((sweep)); then
    [[ -n ${first-} ]] && ((stop)) || bad "--sweep takes --traces and --stop"
    need=(TRACES)
else
    [[ -n ${first-} && -n ${c-} ]] || bad "--traces and --concurrency are required"
    need=(MODEL TRACES HARNESS_PYTHON)
fi
for v in "${need[@]}"; do [[ -n ${!v-} ]] || bad "$v is not set"; done
# The agentx scenario's defaults, as flags on the base scenario: the scenario
# itself arrived in a later agentx-harness than the one compass-harness pins.
agentx_args=()
if ((agentx)); then
    last=$(awk 'END {print NR - 1}' "$TRACES") || fail "traces: cannot read $TRACES"
    ((last >= 0)) || fail "traces: $TRACES holds no trace"
    first=0 duration=${duration:-3600} random_seed=42
    agentx_args=(--trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75
        --warmup-requests-per-lane "${warmup:-10}" --agentic-warmup-grace-period 1800
        --trace-idle-gap-cap-seconds 300 --system-idle-gap-cap-seconds 10 --apply-chat-template)
fi
duration=${duration:-1800}

[[ -s $OUT/server.json ]] || fail "server: no $OUT/server.json; a compass record or compass run --out $OUT writes it once its server answers"
OUT=$(realpath "$OUT")
read -r url pid run_file < <(python3 -c 'import json, sys
s = json.load(open(sys.argv[1]))
print(s["url"], s["pid"], s.get("run_file", ""))' "$OUT/server.json") ||
    fail "server: $OUT/server.json names no url and pid"
curl -sf --max-time 10 "$url/health" >/dev/null || fail "server: $url/health does not answer"

compass_env "$ROOT"
export PYTHONPATH=$ROOT:$ROOT/compass_harness
compass_require_tree "$ROOT" || exit $?
unset ATOM_COMPASS_RUN ATOM_COMPASS_CLOCK_ENDPOINT COMPASS_HARNESS_PREFILL_URL

stop_server() {
    # The command that started the server is its parent.
    local stat command deadline allow=${WAIT_S:-600} more
    stat=$(cat "/proc/$pid/stat" 2>/dev/null) || fail "stop: the server, process $pid, has left"
    read -r _ command _ <<<"${stat##*\) }"
    [[ $(tr '\0' ' ' <"/proc/$command/cmdline") == *compass* ]] ||
        fail "stop: process $command, the server's parent, is no compass command"
    # A trace run's command first writes every rank's trace, which serve gives
    # TRACE_STOP_S, then reduces them (parity.busy). With every profiler row
    # kept, a 120 s V4.1-Flash TP4 trace run's command left about 11 min after
    # SIGTERM: 178 s to export, 445 s to reduce (#3106).
    if [[ -d $OUT/trace ]]; then
        more=$(cd / && python3 -c 'from atom.compass.drive.serve import TRACE_STOP_S as s; print(s)') ||
            fail "stop: cannot read serve's TRACE_STOP_S"
        allow=$((allow + more))
    fi
    kill -TERM "$command" || fail "stop: SIGTERM to process $command failed"
    deadline=$((SECONDS + allow))
    # Until it has exited; a zombie has.
    while stat=$(cat "/proc/$command/stat" 2>/dev/null) && [[ ${stat##*\) } != Z* ]]; do
        ((SECONDS < deadline)) || fail "stop: the command, process $command, is up $allow s after SIGTERM"
        sleep 1
    done
    [[ -s $OUT/run.json ]] || fail "stop: the command left without writing $OUT/run.json"
    echo "client: stopped the server; $OUT/run.json written"
}

if ((sweep)); then
    traces=$(realpath "$TRACES")
    (cd / && python3 -m atom.compass.drive.sweep send "$OUT" "$traces" "$first" "$last") ||
        fail "sweep: atom.compass.drive.sweep send exited $?"
    stop_server
    (cd / && python3 -m atom.compass.drive.sweep cover "$OUT") ||
        fail "sweep: atom.compass.drive.sweep cover exited $?"
    exit 0
fi
# Importing compass_harness also checks the aiperf it would run against.
harness=$(cd / && "$HARNESS_PYTHON" -c 'import compass_harness; print(compass_harness.__file__)') ||
    fail "harness: $HARNESS_PYTHON cannot import compass_harness"
[[ $harness == "$ROOT"/* ]] || fail "harness: compass_harness resolved to $harness"
# Its plugin paces aiperf on a simulated clock: a real server has none, and a
# simulated one waits at time 0 for an unpaced aiperf. PYTHONPATH imports
# compass_harness without installing the plugin.
plugins=$(cd / && "$HARNESS_PYTHON" -c 'from importlib.metadata import entry_points as e
print(*(p.value for p in e(group="aiperf.plugins")))') ||
    fail "harness: $HARNESS_PYTHON cannot list aiperf's plugins"
if [[ -n $run_file ]]; then
    [[ $plugins == *compass_harness* ]] || fail "harness: the server is simulated, and $HARNESS_PYTHON has no compass-harness installed"
    export ATOM_COMPASS_RUN=$run_file
    # deploy.sh's prefill instance: behind a router that is no LP, the traffic
    # LP posts each cancel there (compass_harness/traffic_lp.py).
    if [[ -s $OUT/prefill/server.json ]]; then
        COMPASS_HARNESS_PREFILL_URL=$(python3 -c 'import json, sys
print(json.load(open(sys.argv[1]))["url"])' "$OUT/prefill/server.json") ||
            fail "server: $OUT/prefill/server.json names no url"
        export COMPASS_HARNESS_PREFILL_URL
    fi
    aiperf=(-m aiperf)
else
    [[ $plugins != *compass_harness* ]] || fail "harness: the server is real, and $HARNESS_PYTHON has compass-harness installed"
    aiperf=(-c "$COMPASS_REAL_AIPERF" "$ROOT/compass_harness/compass_harness/session_ids.py")
fi

mkdir "$OUT/client" 2>/dev/null || fail "out: $OUT/client exists; one client run per server --out"
mkdir "$OUT/client/traces"
awk -v d="$OUT/client/traces" -v first="$first" -v last="$last" '
    NR > first {f = sprintf("%s/%05d.json", d, NR - 1); print > f; close(f)}
    NR > last {exit}' "$TRACES" || fail "traces: cannot read $TRACES"
n=$(find "$OUT/client/traces" -type f | wc -l)
((n == last - first + 1)) || fail "traces: $TRACES holds $n of traces $first to $last"
read -rd '' -a extra <<<"${AIPERF_EXTRA_ARGS-}" || :
[[ " ${extra[*]} " =~ \ --random-seed[\ =] ]] && seed=() || seed=(--random-seed "${random_seed:-20260707}")
# aiperf's scrape of the server's /metrics changes what aiperf writes, not what
# it sends. Base mode turns it off in the hashed arguments, as its graded runs
# recorded them; --agentx leaves it out of them, as the export level, so a real
# and a simulated run hash alike: on against a real server, as InferenceX's
# agentx client runs it, off against a simulated one, whose compass-harness
# refuses it.
hashed_scrape=(--no-server-metrics) scrape=()
if ((agentx)); then
    hashed_scrape=()
    [[ -z $run_file ]] || scrape=(--no-server-metrics)
fi
# What aiperf sends: all but paths, the URL, the export level and the scrape
# --agentx sets, as cctraces_sim.sh hashes them into its run.json's workload.
args=(--endpoint-type chat --custom-dataset-type weka_trace
    --concurrency "$c" --benchmark-duration "$duration"
    "${seed[@]}" --scenario inferencex-agentx-mvp --unsafe-override
    --streaming --use-server-token-count --extra-inputs ignore_eos:true
    "${hashed_scrape[@]}" --no-gpu-telemetry --ui none "${agentx_args[@]}" "${extra[@]}")

# Each request carries X-Session-ID, as the nightly's agentic cell sends it, or
# in its place the Dynamo session header where the arm sets it, as ATOM's DP
# submissions do, which workload.json then names. The raw records
# leave out each request's payload, its prompt, and are compressed once the
# export is written. aiperf runs as a job, so that any exit of this script,
# a SIGTERM's included, stops it. Its tokenizer is MODEL's directory, read with
# no network; offline, aiperf looks the name up in the Hugging Face cache alone
# and refuses a directory, so it runs with the offline switches unset.
dynamo=$(compass_dynamo_header) session=true
[[ -z $dynamo ]] || session=false
trap 'set +e; kill -TERM $(jobs -p) 2>/dev/null; wait' EXIT
(cd "$OUT/client" && unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE &&
    AIPERF_DATASET_MMAP_BASE_PATH=$OUT/client AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=$session \
    AIPERF_RECORD_STRIP_PAYLOAD_BYTES=true exec "$HARNESS_PYTHON" "${aiperf[@]}" \
    profile --url "${url#http://}" --model "$MODEL" --tokenizer "$MODEL" \
    --input-file "$OUT/client/traces" --artifact-dir "$OUT/artifacts" "${scrape[@]}" \
    --export-level raw "${args[@]}") \
    >"$OUT/client/aiperf.log" 2>&1 &
# A failure from here on stops a --stop server first: a caller waiting on the
# command that started it, as a driver does, would otherwise wait for ever.
stop_and_fail() {
    ((stop)) && (stop_server)
    fail "$1"
}
# The coverage check wants a first token or a request's end in the duration's
# last 5%. A request in flight across that window, which aiperf cancels at the
# cutoff and records nothing of, fails it with the export whole; and every run
# here is already no valid submission of the scenario (--unsafe-override).
wait $! || {
    rc=$? log=$OUT/client/aiperf.log
    errors=$(sed -n '/AIPerf System Exit Errors/,/╰/s/^│    Error: \([^ ]*\).*/\1/p' "$log" | sort -u)
    ((rc == 1)) && [[ $errors == ProfileMetricCoverageError ]] || stop_and_fail "aiperf: exited $rc; see $log"
    echo "client: aiperf failed its coverage check alone: $(grep -om1 'Profiling metric coverage below .* duration' "$log")"
}
(cd / && python3 -m atom.compass.parity.client_run "$OUT/artifacts") ||
    stop_and_fail "export: atom.compass.parity.client_run exited $? on $OUT/artifacts"
if [[ -n $run_file && -n ${COMPASS_HARNESS_SEND_ORDER_SEED-} ]]; then
    python3 - "$OUT/artifacts/client_run.json" <<'EOF' || stop_and_fail "export: cannot name the send order seed in $OUT/artifacts/client_run.json"
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
seed = {"send_order_seed": os.environ["COMPASS_HARNESS_SEND_ORDER_SEED"]}
path.write_text(json.dumps(json.loads(path.read_text()) | seed, sort_keys=True))
EOF
fi
gzip "$OUT/artifacts/profile_export_raw.jsonl" || stop_and_fail "export: cannot compress $OUT/artifacts/profile_export_raw.jsonl"
python3 - "$OUT" "$TRACES" "$dynamo" "${args[@]}" <<'EOF' || stop_and_fail "workload: cannot write $OUT/artifacts/workload.json"
import hashlib, json, sys
from pathlib import Path

out, traces, header, *args = sys.argv[1:]
sha256 = lambda p: hashlib.file_digest(open(p, "rb"), "sha256").hexdigest()
export = json.loads(Path(out, "artifacts/profile_export_aiperf.json").read_text())
given = sorted(Path(out, "client/traces").iterdir())
workload = {"dataset_sha256": sha256(traces), "traces_sha256": [*map(sha256, given)]}
workload |= {"aiperf_args": args, "benchmark_id": export["benchmark_id"]}
workload |= {"session_header": header} if header else {}
Path(out, "artifacts/workload.json").write_text(json.dumps(workload))
EOF

((stop)) || exit 0
stop_server
# No later client run reached the server: whole grading takes the steps after
# this one's own, of requests aiperf cancelled at its cutoff and recorded nothing of.
python3 - "$OUT/artifacts/client_run.json" <<'EOF' || fail "stop: cannot mark $OUT/artifacts/client_run.json"
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
export = json.loads(path.read_text()) | {"server_stopped": True}
path.write_text(json.dumps(export, sort_keys=True))
EOF
[[ -d $OUT/prefill ]] && exit 0
(cd / && python3 -m atom.compass.report.split "$OUT") ||
    fail "split: atom.compass.report.split exited $?"
(cd / && python3 -m atom.compass.report.prometheus "$OUT") ||
    echo "client: prometheus: atom.compass.report.prometheus exited $?" >&2
(cd / && python3 -m atom.compass.report.html "$OUT") ||
    fail "report: atom.compass.report.html exited $?"
