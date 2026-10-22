#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# One client run against a `compass record` or `compass run` server: aiperf
# with cctraces_sim.sh's arguments over traces A to B, then the run's export.
#
# Usage: client.sh OUT --traces A-B --concurrency C [--duration S] [--stop]
#   OUT              the server's --out, whose server.json names its URL and,
#                    for a simulated server, its run file
#   --traces A-B     TRACES' traces A to B, each counted from 0
#   --concurrency C  aiperf --concurrency
#   --duration S     aiperf --benchmark-duration, seconds on the server's clock
#                    (default 1800)
#   --stop           then SIGTERM the command that started the server, and wait
#                    WAIT_S (default 600) for it to write OUT/run.json and leave
#   MODEL, TRACES    as cctraces_sim.sh takes them
#   HARNESS_PYTHON   the Python with agentx-harness, and with compass-harness
#                    when the server is simulated and only then
#
# A simulated server's client paces aiperf on the server's clock through this
# tree's compass-harness; a real one's pins aiperf's benchmark id as
# compass-harness does, so both send the same prompts, after the scenario's own
# warmup. OUT/client/ holds the traces aiperf is given, each a file
# named by its line in TRACES, aiperf's dataset and its log; OUT/artifacts/
# aiperf's artifacts at --export-level raw and the client run's export,
# client_run.json (atom/compass/parity/client_run.py). One client run per OUT:
# an OUT/client/ there refuses. The server keeps serving unless --stop. Exits 2
# on bad arguments, 90 to 99 as _lib.sh's tree checks refuse, else 1 naming the
# step that failed.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

USAGE="usage: client.sh OUT --traces A-B --concurrency C [--duration S] [--stop]"
bad() { echo "client: $1; $USAGE" >&2; exit 2; }
fail() { echo "client: $1" >&2; exit 1; }
ROOT=$(compass_tree_root) || exit $?
(($#)) && [[ $1 != -* ]] || bad "no OUT"
OUT=$1 duration=1800 stop=0
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
    *) bad "unknown argument $1" ;;
    esac
    shift
done
[[ -n ${first-} && -n ${c-} ]] || bad "--traces and --concurrency are required"
for v in MODEL TRACES HARNESS_PYTHON; do [[ -n ${!v-} ]] || bad "$v is not set"; done

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
unset ATOM_COMPASS_RUN ATOM_COMPASS_CLOCK_ENDPOINT
# Importing compass_harness also checks the aiperf it would run against.
harness=$(cd / && "$HARNESS_PYTHON" -c 'import compass_harness; print(compass_harness.__file__)') ||
    fail "harness: $HARNESS_PYTHON cannot import compass_harness"
[[ $harness == "$ROOT"/* ]] || fail "harness: compass_harness resolved to $harness"
if [[ -n $run_file ]]; then
    export ATOM_COMPASS_RUN=$run_file
    aiperf=(-m aiperf)
else
    # Its plugin paces aiperf on a simulated clock, which a real server has not got.
    plugins=$(cd / && "$HARNESS_PYTHON" -c 'from importlib.metadata import entry_points as e
print(*(p.value for p in e(group="aiperf.plugins")))') ||
        fail "harness: $HARNESS_PYTHON cannot list aiperf's plugins"
    [[ $plugins != *compass_harness* ]] || fail "harness: the server is real, and $HARNESS_PYTHON has compass-harness installed"
    aiperf=(-c 'import uuid, aiperf.cli_runner as r
r.uuid4 = lambda: uuid.UUID(int=0)
from aiperf.cli import app
app()')
fi

mkdir "$OUT/client" 2>/dev/null || fail "out: $OUT/client exists; one client run per server --out"
mkdir "$OUT/client/traces"
awk -v d="$OUT/client/traces" -v first="$first" -v last="$last" '
    NR > first {f = sprintf("%s/%05d.json", d, NR - 1); print > f; close(f)}
    NR > last {exit}' "$TRACES" || fail "traces: cannot read $TRACES"
n=$(find "$OUT/client/traces" -type f | wc -l)
((n == last - first + 1)) || fail "traces: $TRACES holds $n of traces $first to $last"

# Each request carries X-Session-ID, as the nightly's agentic cell sends it: a
# dp_sticky router keys off it, and a server alone ignores it.
(cd "$OUT/client" && AIPERF_DATASET_MMAP_BASE_PATH=$OUT/client AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true \
    exec "$HARNESS_PYTHON" "${aiperf[@]}" \
    profile --url "${url#http://}" --model "$MODEL" --tokenizer "$MODEL" \
    --input-file "$OUT/client/traces" --artifact-dir "$OUT/artifacts" --export-level raw \
    --endpoint-type chat --custom-dataset-type weka_trace \
    --concurrency "$c" --benchmark-duration "$duration" \
    --random-seed 20260707 --scenario inferencex-agentx-mvp --unsafe-override \
    --streaming --use-server-token-count --extra-inputs ignore_eos:true \
    --no-server-metrics --no-gpu-telemetry --ui none) \
    >"$OUT/client/aiperf.log" 2>&1 || fail "aiperf: exited $?; see $OUT/client/aiperf.log"
(cd / && python3 -m atom.compass.parity.client_run "$OUT/artifacts") ||
    fail "export: atom.compass.parity.client_run exited $? on $OUT/artifacts"

((stop)) || exit 0
# The command that started the server is its parent.
stat=$(cat "/proc/$pid/stat" 2>/dev/null) || fail "stop: the server, process $pid, has left"
read -r _ command _ <<<"${stat##*\) }"
[[ $(tr '\0' ' ' <"/proc/$command/cmdline") == *compass* ]] ||
    fail "stop: process $command, the server's parent, is no compass command"
kill -TERM "$command" || fail "stop: SIGTERM to process $command failed"
deadline=$((SECONDS + ${WAIT_S:-600}))
# Until it has exited; a zombie has.
while stat=$(cat "/proc/$command/stat" 2>/dev/null) && [[ ${stat##*\) } != Z* ]]; do
    ((SECONDS < deadline)) || fail "stop: the command, process $command, is up ${WAIT_S:-600} s after SIGTERM"
    sleep 1
done
[[ -s $OUT/run.json ]] || fail "stop: the command left without writing $OUT/run.json"
echo "client: stopped the server; $OUT/run.json written"
