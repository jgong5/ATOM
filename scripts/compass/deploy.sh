#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# A prefill-decode deployment: one prefill and one decode instance, each one
# `compass run` with its own ATOM arguments that joins the clock authority this
# script starts, or, real, one `compass record`, and the Atomesh router between
# them, beside the prefill instance.
#
# Usage: deploy.sh OUT run|record <flags> --prefill [<own>] -- <ATOM args> --decode [<own>] -- <ATOM args> --router -- <router args>
#   OUT            the deployment's directory, absent or empty
#   <flags>        run: `compass run`'s run-file flags: the backend, --machine,
#                  --store, ... (atom/compass/drive/runfile.py); record: the
#                  `compass record` flags both instances take, e.g. --gpu
#   <own>          that instance's own `compass record` flags, e.g. --gpu, or
#                  --host, an address the router reaches; and the decode's
#                  --on PREFIX (below)
#   <ATOM args>    each instance's ATOM arguments; the script adds its
#                  --kv-transfer-config, the simulated connector and the
#                  write-request address, or, real, Mooncake and a handshake
#                  port free here, and refuses one among them
#   <router args>  `atomesh launch` arguments, as given; the script adds the
#                  router's host and ports, --pd-disaggregation and each
#                  instance's URL; run refuses --dp-aware, which makes the
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
# line, the KV link, the run file and the router's command, and exits 0, or 1
# when something left on its own or failed to stop. Exits 2 on bad arguments.
#
# Real, there is no run file or authority: each instance is one `compass
# record --out`, which checks its node's GPUs before and after.
# --on PREFIX, a command prefix, one word split on whitespace, such as `ssh
# <host>` or `srun -w <host> ...`, starts the decode instance on another node,
# and the KV link is then inter_node, else intra_node. Each instance runs its
# command, OUT/<side>.sh, which first writes its process id to OUT/<side>.pid;
# the decode's as `PREFIX bash OUT/decode.sh`, stopped as `PREFIX kill -TERM
# <pid>`, words a shell may parse again. That node sees OUT, this tree and the
# caller's directory at their paths, and, simulated, this node's 127.0.0.1.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

USAGE="usage: deploy.sh OUT run|record <flags> --prefill [<own>] -- <ATOM args> --decode [<own>] -- <ATOM args> --router -- <router args>"
bad() { echo "deploy: $1; $USAGE" >&2; exit 2; }
fail() { echo "deploy: $1" >&2; exit 1; }
ROOT=$(compass_tree_root) || exit $?
ARGV=("$0" "$@")
(($# >= 2)) && [[ $1 != -* ]] || bad "no OUT"
[[ $2 == run || $2 == record ]] || bad "the command after OUT is run or record"
OUT=$1 cmd=$2 side=flags seen= head=0
shift 2
flags=() prefill=() decode=() router=() prefill_own=() decode_own=() on=()
while (($#)); do
    # An instance's own flags run from its marker to its --; the router has none.
    if [[ $side != router && ($1 == --prefill || $1 == --decode || $1 == --router && ${2-} == --) ]]; then
        side=${1#--} seen+=" $side" head=1
    elif ((head)) && [[ $1 == -- ]]; then
        head=0
    elif [[ $side$head$1 == decode1--on ]]; then
        read -ra on <<<"${2-}"
        shift
    else
        case $side$head in
        flags0) flags+=("$1") ;; prefill0) prefill+=("$1") ;; decode0) decode+=("$1") ;;
        router0) router+=("$1") ;; prefill1) prefill_own+=("$1") ;; decode1) decode_own+=("$1") ;;
        esac
    fi
    shift
done
[[ $seen == " prefill decode router" ]] || bad "--prefill --, --decode -- and --router -- come once each, in that order"
[[ " ${prefill[*]} ${decode[*]} " != *" --kv-transfer-config"* ]] || bad "an instance's ATOM arguments carry --kv-transfer-config, which the script fills"
[[ $cmd == record || " ${router[*]} " != *" --dp-aware "* ]] || bad "the router's arguments carry --dp-aware, which has the router choose among DP ranks, and this deployment's router has one worker per side"
mkdir -p "$OUT"
[[ -z $(ls -A "$OUT") ]] || bad "$OUT holds an earlier deployment's files"
OUT=$(realpath "$OUT")
ATOMESH=${ATOMESH:-atomesh}
WAIT_S=${WAIT_S:-600}
KV_LINK=intra_node
((${#on[@]} == 0)) || KV_LINK=inter_node

compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
RUN_FILE= CLOCK=
[[ $cmd == record ]] || RUN_FILE=$OUT/run_file.json
# The ports, all held together so no two are the same: the router's and its
# metrics', and each side's KV block. Simulated, both sides share one block of
# write requests, one per prefill and decode rank pair, and the authority
# takes one more, as the run file states; real, each side's Mooncake side
# channels take one per rank from its handshake port.
ports=$(python3 - "$RUN_FILE" "$KV_LINK" "${#flags[@]}" "${flags[@]}" "${#prefill[@]}" "${prefill[@]}" "${decode[@]}" <<'EOF'
import argparse, json, socket, sys
from pathlib import Path
from atom.compass.artifacts import ArtifactRefusal
from atom.compass.drive import runfile
from atom.compass.spec import SpecRefusal

run_file, kv_link, n, *words = sys.argv[1:]
flags, words = words[: int(n)], words[int(n) :]
k = int(words[0])
prefill, decode = words[1 : 1 + k], words[1 + k :]
held = []
def bind(port=0):
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    held.append(s)
    return s.getsockname()[1]
def block(n):
    while True:
        try:
            base = bind()
            for p in range(base + 1, base + n):
                bind(p)
            return base
        except OSError:
            continue
router, prometheus = bind(), bind()
if not run_file:
    def ranks(atom_args):
        engine = runfile._engine_args(atom_args)
        return engine.data_parallel_size * engine.tensor_parallel_size * engine.pipeline_parallel_size
    print(router, prometheus, block(ranks(prefill)), block(ranks(decode)), "-")
    sys.exit()
parser = argparse.ArgumentParser(prog="deploy.sh run")
runfile.add_arguments(parser)
args = parser.parse_args(flags)
try:
    run = runfile.prefill_decode(args, prefill, decode, kv_link)
except (ValueError, OSError, ArtifactRefusal, SpecRefusal) as refused:
    sys.exit(f"deploy: {refused}")
clock = bind()
base = block(run["prefill_data_parallel_size"] * run["decode_data_parallel_size"])
run |= {"out_dir": str(Path(run_file).parent), "clock_endpoint": f"tcp://127.0.0.1:{clock}"}
Path(run_file).write_text(json.dumps(run))
print(router, prometheus, base, base, clock)
EOF
)
# Its last line; an import may print before it.
read -r ROUTER_PORT PROMETHEUS_PORT PREFILL_KV_PORT DECODE_KV_PORT CLOCK_PORT <<<"${ports##*$'\n'}"

stop_decode() { # SIGTERM the decode instance's command, through its prefix
    [[ ! -s $OUT/decode.pid ]] || "${on[@]}" kill -TERM "$(<"$OUT/decode.pid")" 2>/dev/null || true
}
trap 'stop_decode; kill $(jobs -p) 2>/dev/null || true' EXIT
authority=
if [[ $cmd == run ]]; then
    CLOCK=tcp://127.0.0.1:$CLOCK_PORT
    python3 -m atom.compass.run --compass-run "$RUN_FILE" --compass-clock-endpoint "$CLOCK" \
        >"$OUT/authority.log" 2>&1 &
    authority=$!
    until (exec 3<>"/dev/tcp/127.0.0.1/$CLOCK_PORT") 2>/dev/null; do
        kill -0 $authority 2>/dev/null || fail "the authority exited; see $OUT/authority.log"
        sleep 0.2
    done
fi

instance() { # SIDE KV_ROLE KV_PORT: write the instance's command, OUT/SIDE.sh, and run it
    local -n atom=$1 own=$1_own
    local prefix=() command=("$cmd" --join "$RUN_FILE")
    local kv="{\"kv_role\":\"$2\",\"kv_connector\":\"compass\",\"compass_kv_write_req\":\"tcp://127.0.0.1:$3\"}"
    if [[ $cmd == record ]]; then
        command=("$cmd" "${flags[@]}")
        kv="{\"kv_role\":\"$2\",\"kv_connector\":\"mooncake\",\"handshake_port\":$3}"
    fi
    [[ $1 == prefill ]] || prefix=("${on[@]}")
    command+=("${own[@]}" --out "$OUT/$1" -- "${atom[@]}" --kv-transfer-config "$kv")
    # Its process id, which exec keeps; then this tree, checked, and the caller's directory.
    printf 'echo $$ >%q\n. %q && compass_env %q && compass_require_tree %q && cd %q &&\n    exec' \
        "$OUT/$1.pid" "$ROOT/scripts/compass/_lib.sh" "$ROOT" "$ROOT" "$PWD" >"$OUT/$1.sh"
    printf ' %q' python3 -c 'from atom.compass.cli import main; main()' "${command[@]}" >>"$OUT/$1.sh"
    "${prefix[@]}" bash "$OUT/$1.sh" >"$OUT/$1.log" 2>&1 &
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
instance prefill kv_producer "$PREFILL_KV_PORT"
P=$!
instance decode kv_consumer "$DECODE_KV_PORT"
D=$!
prefill_url=$(url prefill $P)
decode_url=$(url decode $D)

ROUTER=("$ATOMESH" launch --host 127.0.0.1 --port "$ROUTER_PORT" --prometheus-port "$PROMETHEUS_PORT"
    --pd-disaggregation --prefill "$prefill_url" --decode "$decode_url" "${router[@]}")
"${ROUTER[@]}" >"$OUT/router.log" 2>&1 &
R=$!
compass_wait_up router $R "http://127.0.0.1:$ROUTER_PORT/v1/models"
# A stop from here on, one server.json's reader sends included, also stops the
# router, so the wait below ends even when the stop comes before it.
stopped=0
trap 'stopped=1; kill -TERM $R 2>/dev/null || true' TERM INT
# The router's parent is this script, which a client's stop signals.
python3 -c 'import json, sys
url, pid, run_file, clock, out = sys.argv[1:]
ready = {"url": url, "pid": int(pid)}
if run_file:
    ready |= {"clock_endpoint": clock, "run_file": run_file}
open(out + ".part", "w").write(json.dumps(ready))' \
    "http://127.0.0.1:$ROUTER_PORT" $R "$RUN_FILE" "$CLOCK" "$OUT/server.json"
mv "$OUT/server.json.part" "$OUT/server.json"
echo "deploy: ready at http://127.0.0.1:$ROUTER_PORT"

wait -n $authority $P $D $R || true
rc=$((!stopped))
((stopped)) || echo "deploy: a process of the deployment left before a stop" >&2
# Each instance stops its server, whose frontend ends a simulated run through
# the authority; the authority leaves once it has written the summary.
kill -TERM $P $R 2>/dev/null || true
stop_decode
deadline=$((SECONDS + WAIT_S))
for pid in $P $D $R $authority; do
    while kill -0 $pid 2>/dev/null; do
        ((SECONDS < deadline)) || { kill -KILL $pid; rc=1; echo "deploy: process $pid killed after $WAIT_S s" >&2; }
        sleep 1
    done
    wait $pid || ((pid == R)) || rc=1
done
trap - EXIT
python3 -c 'import json, os, sys
out, link, n, *rest = sys.argv[1:]
run = {"command": rest[: int(n)], "kv_link": link, "router": rest[int(n) :]}
if os.path.exists(out + "/run_file.json"):
    run["run_file"] = json.load(open(out + "/run_file.json"))
open(out + "/run.json.part", "w").write(json.dumps(run))' \
    "$OUT" "$KV_LINK" "${#ARGV[@]}" "${ARGV[@]}" "${ROUTER[@]}"
mv "$OUT/run.json.part" "$OUT/run.json"
exit $rc
