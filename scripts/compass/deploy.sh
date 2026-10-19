#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# A simulated prefill-decode deployment on one node: one prefill and one decode
# instance, each one `compass run` with its own ATOM arguments that joins the
# clock authority this script starts, and the Atomesh router between them.
#
# Usage: deploy.sh OUT run <run flags> --prefill -- <ATOM args> --decode -- <ATOM args> --router -- <router args>
#   OUT            the deployment's directory, absent or empty
#   <run flags>    `compass run`'s run-file flags: the backend, --machine,
#                  --store, ... (atom/compass/drive/runfile.py)
#   <ATOM args>    each instance's ATOM arguments; the script adds its
#                  --kv-transfer-config, the simulated connector and the
#                  write-request address, and refuses one among them
#   <router args>  `atomesh launch` arguments, as given; the script adds the
#                  router's host and ports, --pd-disaggregation and each
#                  instance's URL, and refuses --dp-aware, which makes the
#                  router choose among DP ranks
#   ATOMESH        the router binary (default atomesh)
#   WAIT_S         seconds each instance and the router have to answer, and
#                  the deployment to leave once stopped (default 600)
#
# OUT gets run_file.json, the deployment's one run file, built from the run
# flags and both sides' ATOM arguments (`runfile.prefill_decode`), whose
# out_dir is OUT and whose clock_endpoint the authority's; authority.log;
# prefill/ and decode/, each instance's `compass run --out`, its step record
# among it, with its command's log beside it; router.log; and, once the router
# answers, server.json as `compass run` writes one: the router's URL and
# process id, the run file and the clock endpoint. The authority writes the
# step table, the run summaries and the hand-ins in OUT. A client drives the
# deployment through server.json as it drives one server (client.sh); no
# client starts here. It serves until SIGTERM or SIGINT, then stops the
# instances, the router and the authority, writes OUT/run.json, the command
# line, the run file and the router's command, and exits 0, or 1 when
# something left on its own or failed to stop. Exits 2 on bad arguments.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

USAGE="usage: deploy.sh OUT run <run flags> --prefill -- <ATOM args> --decode -- <ATOM args> --router -- <router args>"
bad() { echo "deploy: $1; $USAGE" >&2; exit 2; }
fail() { echo "deploy: $1" >&2; exit 1; }
ROOT=$(compass_tree_root) || exit $?
ARGV=("$0" "$@")
(($# >= 2)) && [[ $1 != -* ]] || bad "no OUT"
[[ $2 == run ]] || bad "the command after OUT is run"
OUT=$1 side=flags seen=
shift 2
flags=() prefill=() decode=() router=()
while (($#)); do
    if [[ $1 =~ ^--(prefill|decode|router)$ && ${2-} == -- ]]; then
        side=${BASH_REMATCH[1]} seen+=" $side"
        shift 2
        continue
    fi
    case $side in
    flags) flags+=("$1") ;; prefill) prefill+=("$1") ;;
    decode) decode+=("$1") ;; router) router+=("$1") ;;
    esac
    shift
done
[[ $seen == " prefill decode router" ]] || bad "--prefill --, --decode -- and --router -- come once each, in that order"
[[ " ${prefill[*]} ${decode[*]} " != *" --kv-transfer-config"* ]] || bad "an instance's ATOM arguments carry --kv-transfer-config, which the script fills"
[[ " ${router[*]} " != *" --dp-aware "* ]] || bad "the router's arguments carry --dp-aware, which has the router choose among DP ranks, and this deployment's router has one worker per side"
mkdir -p "$OUT"
[[ -z $(ls -A "$OUT") ]] || bad "$OUT holds an earlier deployment's files"
OUT=$(realpath "$OUT")
ATOMESH=${ATOMESH:-atomesh}
WAIT_S=${WAIT_S:-600}

compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
RUN_FILE=$OUT/run_file.json
# The run file, and the ports: the authority's, a block of the write
# requests' (one per prefill and decode rank pair), the router's and its
# metrics', all held together so no two are the same.
ports=$(python3 - "$RUN_FILE" "${#flags[@]}" "${flags[@]}" "${#prefill[@]}" "${prefill[@]}" "${decode[@]}" <<'EOF'
import argparse, json, socket, sys
from pathlib import Path
from atom.compass.artifacts import ArtifactRefusal
from atom.compass.drive import runfile
from atom.compass.spec import SpecRefusal

run_file, n, *words = sys.argv[1:]
flags, words = words[: int(n)], words[int(n) :]
k = int(words[0])
parser = argparse.ArgumentParser(prog="deploy.sh run")
runfile.add_arguments(parser)
args = parser.parse_args(flags)
try:
    run = runfile.prefill_decode(args, words[1 : 1 + k], words[1 + k :])
except (ValueError, OSError, ArtifactRefusal, SpecRefusal) as refused:
    sys.exit(f"deploy: {refused}")
pairs = run["prefill_data_parallel_size"] * run["decode_data_parallel_size"]
held = []
def bind(port=0):
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    held.append(s)
    return s.getsockname()[1]
clock, router, prometheus = bind(), bind(), bind()
while True:
    try:
        base = bind()
        for p in range(base + 1, base + pairs):
            bind(p)
        break
    except OSError:
        continue
run |= {"out_dir": str(Path(run_file).parent), "clock_endpoint": f"tcp://127.0.0.1:{clock}"}
Path(run_file).write_text(json.dumps(run))
print(clock, base, router, prometheus)
EOF
)
# Its last line; an import may print before it.
read -r CLOCK_PORT KV_PORT ROUTER_PORT PROMETHEUS_PORT <<<"${ports##*$'\n'}"
CLOCK=tcp://127.0.0.1:$CLOCK_PORT

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
python3 -m atom.compass.run --compass-run "$RUN_FILE" --compass-clock-endpoint "$CLOCK" \
    >"$OUT/authority.log" 2>&1 &
authority=$!
until (exec 3<>"/dev/tcp/127.0.0.1/$CLOCK_PORT") 2>/dev/null; do
    kill -0 $authority 2>/dev/null || fail "the authority exited; see $OUT/authority.log"
    sleep 0.2
done

instance() { # SIDE KV_ROLE ATOM_ARGS...: one `compass run` joined to the authority
    local kv="{\"kv_role\":\"$2\",\"kv_connector\":\"compass\",\"compass_kv_write_req\":\"tcp://127.0.0.1:$KV_PORT\"}"
    python3 -c 'from atom.compass.cli import main; main()' run --join "$RUN_FILE" \
        --out "$OUT/$1" -- "${@:3}" --kv-transfer-config "$kv" >"$OUT/$1.log" 2>&1 &
}
url() { # SIDE PID: the instance's URL, once its server.json names it
    local deadline=$((SECONDS + WAIT_S))
    until [[ -s $OUT/$1/server.json ]]; do
        kill -0 "$2" 2>/dev/null || fail "the $1 instance exited before it answered; see $OUT/$1.log"
        ((SECONDS < deadline)) || fail "no $1 instance after $WAIT_S s"
        sleep 1
    done
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["url"])' "$OUT/$1/server.json"
}
instance prefill kv_producer "${prefill[@]}"
P=$!
instance decode kv_consumer "${decode[@]}"
D=$!
prefill_url=$(url prefill $P)
decode_url=$(url decode $D)

ROUTER=("$ATOMESH" launch --host 127.0.0.1 --port "$ROUTER_PORT" --prometheus-port "$PROMETHEUS_PORT"
    --pd-disaggregation --prefill "$prefill_url" --decode "$decode_url" "${router[@]}")
"${ROUTER[@]}" >"$OUT/router.log" 2>&1 &
R=$!
compass_wait_up router $R "http://127.0.0.1:$ROUTER_PORT/v1/models"
# The router's parent is this script, which a client's stop signals.
python3 -c 'import json, sys
url, pid, run_file, clock, out = sys.argv[1:]
ready = {"url": url, "pid": int(pid), "clock_endpoint": clock, "run_file": run_file}
open(out + ".part", "w").write(json.dumps(ready))' \
    "http://127.0.0.1:$ROUTER_PORT" $R "$RUN_FILE" "$CLOCK" "$OUT/server.json"
mv "$OUT/server.json.part" "$OUT/server.json"
echo "deploy: ready at http://127.0.0.1:$ROUTER_PORT"

stopped=0
trap 'stopped=1' TERM INT
wait -n $authority $P $D $R || true
rc=$((!stopped))
((stopped)) || echo "deploy: a process of the deployment left before a stop" >&2
# Each instance stops its server, whose frontend ends the run through the
# authority; the authority leaves once it has written the summary.
kill -TERM $P $D $R 2>/dev/null || true
deadline=$((SECONDS + WAIT_S))
for pid in $P $D $R $authority; do
    while kill -0 $pid 2>/dev/null; do
        ((SECONDS < deadline)) || { kill -KILL $pid; rc=1; echo "deploy: process $pid killed after $WAIT_S s" >&2; }
        sleep 1
    done
    wait $pid || ((pid == R)) || rc=1
done
trap - EXIT
python3 -c 'import json, sys
out, n, *rest = sys.argv[1:]
run = {"command": rest[: int(n)], "run_file": json.load(open(out + "/run_file.json")),
       "router": rest[int(n) :]}
open(out + "/run.json.part", "w").write(json.dumps(run))' \
    "$OUT" "${#ARGV[@]}" "${ARGV[@]}" "${ROUTER[@]}"
mv "$OUT/run.json.part" "$OUT/run.json"
exit $rc
