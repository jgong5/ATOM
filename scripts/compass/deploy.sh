#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# A prefill-decode deployment: one or more prefill and one or more decode
# instances, each one `compass run` with its own ATOM arguments that joins the
# clock authority this script starts, or, real, one `compass record`, and the
# Atomesh router between them, beside the first prefill instance.
#
# Usage: deploy.sh OUT run|record <flags> --prefill [<own>] -- <ATOM args> [--prefill ...] --decode [<own>] -- <ATOM args> [--decode ...] --router -- <router args>
#   OUT            the deployment's directory, absent or empty
#   <flags>        run: `compass run`'s run-file flags: the backend, --device,
#                  --ptl, --network across nodes,
#                  --store, ... (atom/compass/drive/runfile.py); record: the
#                  `compass record` flags both instances take, e.g. --gpu
#   <own>          that instance's own `compass record` flags, e.g. --gpu, or
#                  --host, an address the router reaches; --on PREFIX
#                  (below); and --env NAME=VALUE, repeated, a variable that
#                  instance's command alone runs under, such as a prefill's
#                  ATOM_KV_OFFLOAD=mooncake_store and its
#                  ATOM_KV_OFFLOAD_EXTRA_CONFIG, which ATOM folds into the
#                  instance's --kv-transfer-config as the recipe's `multi`
#                  connector (atom/model_engine/arg_utils.py)
#   <ATOM args>    each instance's ATOM arguments; the script adds its
#                  --kv-transfer-config, the simulated connector and the
#                  write-request address, or, real, Mooncake and a handshake
#                  port, each prefill's free here, and refuses one among them.
#                  Every instance of a side takes the same ATOM arguments, the
#                  run file's one engine config of that side
#   --prefill, --decode
#                  repeated, each one more instance of that side, every
#                  --prefill before the first --decode; above one instance of
#                  a side each is named by its index, prefill0/ and prefill1/
#                  in place of prefill/, and, simulated, names it in its
#                  kv_transfer_config as `compass_instance`, so its LPs carry
#                  it (engine-P1), and the router, choosing among them, runs as
#                  LP `router` (the run file's `prefill_instances` and
#                  `decode_instances`)
#   <router args>  `atomesh launch` arguments, as given; the script adds the
#                  router's host and ports, --pd-disaggregation and each
#                  instance's URL; run with --dp-aware runs the router as LP
#                  `router`, refusing an ATOMESH not built from this tree,
#                  and the run file gains `router_in_s`, each hop into it,
#                  at `ipc_s`, borrowed: Atomesh logs no receipt to measure
#                  its own from. run refuses --prefill and --decode, a worker more to
#                  choose, and --max-concurrent-requests and
#                  --rate-limit-tokens-per-second, a limiter on the wall
#                  clock; it sets --request-timeout-secs,
#                  --prepare-queue-timeout-ms and --prepare-timeout-ms
#                  (`_lib.sh`: COMPASS_SIM_ROUTER_BOUNDS) and refuses the
#                  caller's (compass_sim_router_refusal, which pd_sim.sh's
#                  ROUTER_ARGS meets too); the caller switches the router's
#                  wall-clock detectors off, as pd_sim.sh does:
#                  --disable-health-check
#                  --disable-circuit-breaker --disable-retries. run refuses
#                  a router LP without --disable-retries: one that retries
#                  resends a request under the seq it was first sent with,
#                  and the frontend refuses it as arrived twice
#   ATOMESH        the router binary (default atomesh)
#   SEED           the router LP's seed (default 20260707, client.sh's)
#   WAIT_S         seconds each instance and the router have to answer, and
#                  the deployment to leave once stopped (default 600)
#   PREFIX_S       seconds --on's prefix has to start the decode's command,
#                  to show its environment, to send it a stop, and to pair
#                  its node's clock (default 60)
#   CLOCK_PAIR_S   seconds between two pairs of a node's clock (below; default
#                  60)
#   CLOCK_PAIR_PREFIX_<node>
#                  a command prefix, split on whitespace, that pairs <node>'s
#                  clock in place of the --on prefix of its instances, such as
#                  `ssh -o BatchMode=yes <host>` where that one carries no
#                  stdin, as a relay may not; it reaches the node as --on's
#                  does, this tree at its path. <node> is the address its
#                  instances' server.json URLs name, each character other
#                  than a letter, a digit or _ an _: 10.19.0.127's is
#                  CLOCK_PAIR_PREFIX_10_19_0_127, fd00::7's
#                  CLOCK_PAIR_PREFIX_fd00__7
#   PREFILL_ENV DECODE_ENV
#                  NAME=VALUE words, split on whitespace, set in that side's
#                  instances only, as pd_sim.sh takes them: each instance's
#                  command (OUT/<name>.sh, below) sets them, so --on carries
#                  them, e.g. `ATOM_KV_OFFLOAD=mooncake_store
#                  ATOM_KV_OFFLOAD_EXTRA_CONFIG=<JSON>` for a prefill side on
#                  the Mooncake Store; a word that is no NAME=VALUE exits 2
#
# OUT gets run_file.json, the deployment's one run file, built from the run
# flags and both sides' ATOM arguments (`runfile.prefill_decode`), whose
# out_dir is OUT and whose clock_endpoint the authority's; authority.log;
# prefill/ and decode/, each instance's `compass run --out`, its step record
# among it, with its command's log beside it; router.log; and, once the router
# answers, server.json as `compass run` writes one: the router's URL, the
# process id of a child of this script, the run file and the clock endpoint.
# The authority writes the step table, the run summaries and the hand-ins in
# OUT. A client drives the deployment through server.json as it drives one
# server (client.sh); no client starts here. It serves until SIGTERM or SIGINT,
# then stops the instances, the router and the authority, writes OUT/run.json,
# the command line, the KV link, whether the decode ran on another node
# (`across_nodes`), the run file and the router's command, and,
# once a client's export is complete (OUT/artifacts/client_run.json), each
# request's split, OUT/request_split.json (atom/compass/report/split.py), CI's
# Prometheus report of a simulated run's series, OUT/prometheus.html, or why
# there is none, OUT/prometheus.txt (atom/compass/report/prometheus.py), which
# fails no stop, then the run's report, OUT/report.html
# (atom/compass/report/html.py), and exits 0, or 1 when something left on its
# own, a router LP as the traffic closes aside, failed to stop, or the split or
# the run's report failed. Exits 2 on bad arguments.
#
# Real, there is no run file or authority: each instance is one `compass
# record --out`, which checks its node's GPUs before and after.
# --on PREFIX, a command prefix, one word split on whitespace, such as `ssh
# <host>` or `srun -w <host> ...`, starts the decode instance on another node,
# and the KV link is then inter_node, else intra_node. --kv-over-network, among
# <flags>, keeps both instances on this node but makes the KV link inter_node
# too: Mooncake moves the KV through the NICs (RDMA) or over TCP, never over the
# GPUs' intra-node links, so an 8-GPU node holding both instances (an MI355X's,
# 1P1D at DPA4 a side) prices its KV on --network's link, as a measured or
# assumed network file states it. run.json names where the decode ran,
# `across_nodes`, apart from `kv_link`. KV_EXTRA, a JSON object, is merged into
# each real instance's Mooncake kv config, such as '{"protocol": "tcp"}' on a
# node with no RDMA HCA; a simulated instance's connector takes none.
# PREFILL_KV_EXTRA and DECODE_KV_EXTRA, JSON objects, are merged into that
# side's kv config after KV_EXTRA, such as a decode's `pd_admission` (the M3.1
# 1P1D recipe's) or a prefill's chunked-transfer settings, which a real
# instance takes whole; a simulated one takes `SIM_KV_KEYS` among them, the
# settings its connector honours as Mooncake's does (`pd_admission`, and the
# chunked transfer's), and the script names on stderr each key it leaves out.
# A prefill's `enable_chunked_transfer` true makes a simulated run's KV
# transfer chunked: the run file's `kv_chunked`, and each instance's
# write-done notice address (`compass_kv_write_done`), from a port block of
# its own.
# Each instance runs its
# command, OUT/<name>.sh, which first writes its process id to OUT/<name>.pid;
# one given --on as `PREFIX bash OUT/<name>.sh`, stopped as `PREFIX kill -TERM
# <pid>`, a second command under the first, so srun inside an allocation takes
# --overlap. --on on any instance makes the KV link inter_node. Each node
# other than this one that an --on prefix starts an instance on, named by the
# address that instance's server.json URL gives (none for a loopback or
# unspecified one, which the split takes as this node), has its CLOCK_MONOTONIC
# paired with this one's over that prefix, or CLOCK_PAIR_PREFIX_<node>'s
# (atom/compass/drive/clock_pair.py):
# as the router answers and every CLOCK_PAIR_S while the deployment serves and
# stops, apart from it, and once it has stopped, each pair a line of
# OUT/clock_pairs.jsonl, by which the split joins that node's instances to the
# client's clock. A pair that fails, the prefix closing or not answering in
# PREFIX_S, writes no line, why on stderr, and fails nothing. run.json's
# `instances` names each instance, its side and index and, real, its Mooncake
# handshake port, by which the split tells a decode's transfers apart by the
# prefill instance they came from (atom/compass/report/split.py). A shell may
# parse those words again, so OUT, this tree and the caller's directory must be
# paths `printf %q` leaves as they are. That node sees them at their paths, and,
# simulated, this node's 127.0.0.1. Simulated, where the host entries' hops
# differ, the decode's are picked under the environment the first decode
# instance's prefix starts its command in, which `PREFIX env -0` shows, one more
# command under it, as the prefill's are under this script's.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

USAGE="usage: deploy.sh OUT run|record <flags> --prefill [<own>] -- <ATOM args> [--prefill ...] --decode [<own>] -- <ATOM args> [--decode ...] --router -- <router args>"
bad() { echo "deploy: $1; $USAGE" >&2; exit 2; }
fail() { echo "deploy: $1" >&2; exit 1; }
push() { local -n into=$1; into+=("$2"); } # ARRAY WORD
quoted() { local -n words=$1; printf '%q ' "${words[@]}"; } # ARRAY: its words, one string
ROOT=$(compass_tree_root) || exit $?
ARGV=("$0" "$@")
(($# >= 2)) && [[ $1 != -* ]] || bad "no OUT"
[[ $2 == run || $2 == record ]] || bad "the command after OUT is run or record"
OUT=$1 cmd=$2 state=flags seen= head=0 KV_OVER_NETWORK=0
shift 2
# Instance i: its side, sides[i], prefill or decode; its ATOM arguments atom_i,
# its own flags own_i and its --on prefix on_i.
flags=() router=() sides=()
while (($#)); do
    # An instance's own flags run from its marker to its --; the router has none.
    if [[ $state != router && ($1 == --prefill || $1 == --decode || $1 == --router && ${2-} == --) ]]; then
        state=${1#--} head=1
        [[ $seen == *" $state" ]] || seen+=" $state"
        if [[ $state != router ]]; then
            i=${#sides[@]}
            sides+=("$state")
            declare -ga "atom_$i=()" "own_$i=()" "on_$i=()" "env_$i=()"
        fi
    elif ((head)) && [[ $1 == -- ]]; then
        head=0
    elif [[ $state$head$1 == prefill1--on || $state$head$1 == decode1--on ]]; then
        [[ ${2-} == [!-]* ]] || bad "--on takes a command prefix, a command and its words"
        read -ra "on_$i" <<<"$2"
        shift
    elif [[ $state$head$1 == prefill1--env || $state$head$1 == decode1--env ]]; then
        [[ ${2-} =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || bad "--env takes NAME=VALUE"
        push "env_$i" "$2"
        shift
    else
        case $state$head in
        flags0) [[ $1 == --kv-over-network ]] && KV_OVER_NETWORK=1 || flags+=("$1") ;;
        prefill0 | decode0) push "atom_$i" "$1" ;;
        prefill1 | decode1) push "own_$i" "$1" ;;
        router0) router+=("$1") ;;
        esac
    fi
    shift
done
[[ $seen == " prefill decode router" ]] || bad "--prefill -- (once or more), --decode -- (once or more) and --router -- come once each, in that order"
NP=0 ND=0 names=() index=() any_on=0
for i in "${!sides[@]}"; do
    if [[ ${sides[i]} == prefill ]]; then index+=($((NP++))); else index+=($((ND++))); fi
    declare -n on=on_$i
    ((${#on[@]} == 0)) || any_on=1
    unset -n on
done
for i in "${!sides[@]}"; do
    n=$([[ ${sides[i]} == prefill ]] && echo $NP || echo $ND)
    names+=("${sides[i]}$( ((n == 1)) || echo "${index[i]}")")
    [[ " $(quoted "atom_$i") " != *" --kv-transfer-config"* ]] || bad "an instance's ATOM arguments carry --kv-transfer-config, which the script fills"
    first=$([[ ${sides[i]} == prefill ]] && echo 0 || echo $NP)
    [[ $(quoted "atom_$i") == "$(quoted "atom_$first")" ]] || bad "${names[i]}'s ATOM arguments differ from ${names[first]}'s: every instance of a side takes the same"
done
declare -n prefill=atom_0 decode=atom_$NP decode_on=on_$NP
ATOMESH=${ATOMESH:-atomesh} ROUTER_LP=
# A simulated router chooses among a side's instances as LP `router`, as a
# --dp-aware one chooses among DP ranks.
[[ $cmd == record ]] || { [[ " ${router[*]} " != *" --dp-aware "* ]] && ((NP == 1 && ND == 1)); } || ROUTER_LP=$ATOMESH
if [[ $cmd == run ]]; then
    why=$(compass_sim_router_refusal "$([[ -n $ROUTER_LP ]] && echo 1 || echo 0)" "${router[@]}")
    [[ -z $why ]] || bad "$why"
fi
for path in "$(realpath -m "$OUT")" "$ROOT" "$PWD"; do
    ((!any_on)) || [[ $(printf %q "$path") == "$path" ]] || bad "--on's prefix may parse $path again, which a shell would split or expand"
done
mkdir -p "$OUT"
[[ -z $(ls -A "$OUT") ]] || bad "$OUT holds an earlier deployment's files"
OUT=$(realpath "$OUT")
WAIT_S=${WAIT_S:-600}
PREFIX_S=${PREFIX_S:-60}
KV_LINK=intra_node ACROSS_NODES=0
((!any_on)) || KV_LINK=inter_node ACROSS_NODES=1
((!KV_OVER_NETWORK)) || KV_LINK=inter_node
((!ACROSS_NODES || !KV_OVER_NETWORK)) || bad "--kv-over-network keeps the decode on this node, and --on starts it on another"
for extra in KV_EXTRA PREFILL_KV_EXTRA DECODE_KV_EXTRA; do
    [[ -z ${!extra:-} ]] ||
        python3 -c 'import json, sys; assert isinstance(json.loads(sys.argv[1]), dict)' "${!extra}" 2>/dev/null ||
        bad "$extra is no JSON object: ${!extra}"
done
# The side kv settings a simulated instance's connector honours.
SIM_KV_KEYS="pd_admission enable_chunked_transfer chunked_admission_timeout_s"
KV_CHUNKED=0
if [[ -n ${PREFILL_KV_EXTRA:-} ]] && python3 -c 'import json, sys
sys.exit(json.loads(sys.argv[1]).get("enable_chunked_transfer") is not True)' "$PREFILL_KV_EXTRA"; then
    KV_CHUNKED=1
fi
read -ra prefill_env <<<"${PREFILL_ENV:-}"
read -ra decode_env <<<"${DECODE_ENV:-}"
for word in "${prefill_env[@]}" "${decode_env[@]}"; do
    [[ $word =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || bad "PREFILL_ENV and DECODE_ENV take NAME=VALUE words, not $word"
done

compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
RUN_FILE= CLOCK=
[[ $cmd == record ]] || RUN_FILE=$OUT/run_file.json
# The ports, all held together so no two are the same: the router's and its
# metrics', and each side's KV block. Simulated, both sides share one block of
# write requests, one per prefill and decode rank pair, and the authority
# takes one more, as the run file states; real, the prefill's Mooncake write
# listeners take one per rank from its handshake port. The decode's handshake
# port, the prefill's plus its ranks as the nightly's 1P1D launch gives it, is
# not held: its listeners bind ports the kernel hands out.
ports=$(DEPLOY_KV_CHUNKED=$KV_CHUNKED python3 - $$ "$RUN_FILE" "$KV_LINK" "$ROUTER_LP" "${decode_on[*]}" "$PREFIX_S" "$NP" "$ND" "${#flags[@]}" "${flags[@]}" "${#prefill[@]}" "${prefill[@]}" "${decode[@]}" <<'EOF'
import argparse, json, os, random, socket, subprocess, sys
from pathlib import Path
from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactRefusal
from atom.compass.drive import runfile, serve
from atom.compass.drive.run import side_env
from atom.compass.spec import SpecRefusal

# Python names a program read from stdin <stdin>, a file that a process this
# one spawns to resolve a side's config would run as its main, and not find.
globals().pop("__file__", None)
script, run_file, kv_link, router_lp, on, prefix_s, np_, nd, n, *words = sys.argv[1:]
np_, nd = int(np_), int(nd)
flags, words = words[: int(n)], words[int(n) :]
k = int(words[0])
prefill, decode = words[1 : 1 + k], words[1 + k :]
held, room = [], serve.below_ephemeral(serve.DEPLOY_PORTS)
def block(n):
    bases = range(room.start, room.stop - n + 1)
    for base in random.sample(bases, len(bases)):
        try:
            for p in range(base, base + n):
                held.append(socket.socket())
                held[-1].bind(("", p))
            return base
        except OSError:
            [held.pop().close() for _ in range(p - base + 1)]
    sys.exit(f"deploy: no {n} free ports in a row from {room.start} below {room.stop}, where {serve.PORT_RANGE} starts")
def hand(*ports):
    """Print `ports` and exit; a holder keeps every drawn port until the script
    exits, as compass_held_port's does, spawned since this process runs threads."""
    for s in held:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    hold = "import os, select, sys; select.select([os.pidfd_open(int(sys.argv[1]))], [], [])"
    subprocess.Popen([sys.executable, "-c", hold, script], pass_fds=[s.fileno() for s in held], stdout=subprocess.DEVNULL)
    print(*ports)
    sys.exit()
router, prometheus = block(1), block(1)
if not run_file:
    # Real: each prefill instance's Mooncake handshake block, one port a rank;
    # each decode's, the first prefill's plus its ranks, then one more each.
    engine = runfile._engine_args(prefill)
    ranks = engine.data_parallel_size * engine.tensor_parallel_size * engine.pipeline_parallel_size
    bases = [block(ranks) for _ in range(np_)]
    hand(router, prometheus, "-", "-", *bases, *(bases[0] + ranks + k for k in range(nd)))
def decode_env():
    """The environment --on's prefix starts the decode's command in, with what
    compass_env sets over it there."""
    shown = subprocess.run([*on.split(), "env", "-0"], stdin=subprocess.DEVNULL, capture_output=True, timeout=float(prefix_s))
    if shown.returncode:
        said = shown.stderr.decode(errors="replace").strip()
        raise ValueError(f"--on's prefix exited {shown.returncode} showing the decode's environment: {said}")
    env = dict(v.split("=", 1) for v in shown.stdout.decode(errors="surrogateescape").split("\0") if v)
    return env | side_env(env)
parser = argparse.ArgumentParser(prog="deploy.sh run")
runfile.add_arguments(parser)
args = parser.parse_args(flags)
try:
    run = runfile.prefill_decode(args, prefill, decode, kv_link, decode_env if on else None)
    chunked = os.environ.get("DEPLOY_KV_CHUNKED") == "1"
    if chunked:
        run[compass_run.KV_CHUNKED] = True
    for side, count in (("prefill_instances", np_), ("decode_instances", nd)):
        if count > 1:
            run[side] = count
    lookaheads = "-"
    if router_lp:
        run["router"] = compass_run.router_binary(router_lp)
        run.setdefault("router_in_s", compass_run.hop(run, "ipc_s"))
        table = compass_run.channel_table(run)
        lookaheads = json.dumps({c.name: table.lookahead(c.name) for c in table.channels_from(compass_run.ROUTER)}, separators=(",", ":"))
except (ValueError, OSError, subprocess.TimeoutExpired, ArtifactRefusal, SpecRefusal) as refused:
    sys.exit(f"deploy: {refused}")
clock = block(1)
# One write-request port a (prefill instance, decode instance, prefill rank,
# decode rank), from one base every instance is given (`kv.connector`).
pairs = np_ * nd * run["prefill_data_parallel_size"] * run["decode_data_parallel_size"]
base = block(pairs)
# Chunked: one write-done port a pair, which decode binds and prefill connects to.
done = [block(pairs)] * (np_ + nd) if chunked else []
run |= {"out_dir": str(Path(run_file).parent), "clock_endpoint": f"tcp://127.0.0.1:{clock}"}
Path(run_file).write_text(json.dumps(run))
hand(router, prometheus, clock, lookaheads, *[base] * (np_ + nd), *done)
EOF
)
# Its last line; an import may print before it.
# Then each instance's KV port, in instance order.
read -ra drawn <<<"${ports##*$'\n'}"
ROUTER_PORT=${drawn[0]} PROMETHEUS_PORT=${drawn[1]} CLOCK_PORT=${drawn[2]} LOOKAHEADS=${drawn[3]}
KV_PORTS=("${drawn[@]:4:$((NP + ND))}") DONE_PORTS=("${drawn[@]:$((4 + NP + ND))}")

pids=()
instance_pid() { # I: instance I's command's process id, waiting up to PREFIX_S while its prefix runs
    local deadline=$((SECONDS + PREFIX_S)) file=$OUT/${names[$1]}.pid
    until [[ -s $file ]]; do
        [[ -n ${pids[$1]-} ]] && kill -0 "${pids[$1]}" 2>/dev/null && ((SECONDS < deadline)) || return 1
        sleep 0.2
    done
    cat "$file"
}
stop_instance() { # I: SIGTERM instance I's command, through its prefix
    local -n prefix=on_$1
    local pid
    if ((${#prefix[@]} == 0)); then
        [[ -z ${pids[$1]-} ]] || kill -TERM "${pids[$1]}" 2>/dev/null || true
        return 0
    fi
    pid=$(instance_pid "$1") || return 0
    timeout "$PREFIX_S" "${prefix[@]}" kill -TERM "$pid" </dev/null 2>/dev/null || true
}
stop_remote() { # SIGTERM each instance an --on prefix runs elsewhere
    local i
    for i in "${!sides[@]}"; do
        local -n prefix=on_$i
        ((${#prefix[@]} == 0)) || stop_instance "$i"
        unset -n prefix
    done
}
trap 'stop_remote; kill $(jobs -p) 2>/dev/null || true' EXIT
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

instance() { # I: write instance I's command, OUT/<name>.sh, and run it
    local i=$1 name=${names[$1]}
    local -n atom=atom_$1 own=own_$1 prefix=on_$1 side_vars=${sides[$1]}_env own_vars=env_$1
    # Its side's variables (PREFILL_ENV, DECODE_ENV), then its own (--env), which win.
    local vars=("${side_vars[@]}" "${own_vars[@]}")
    local role=kv_consumer command=("$cmd" --join "$RUN_FILE") several=$ND
    [[ ${sides[i]} == decode ]] || role=kv_producer several=$NP
    local kv="{\"kv_role\":\"$role\",\"kv_connector\":\"compass\",\"compass_kv_write_req\":\"tcp://127.0.0.1:${KV_PORTS[i]}\""
    ((several == 1)) || kv+=",\"compass_instance\":${index[i]}"
    [[ $cmd != run ]] || ((!KV_CHUNKED)) || kv+=",\"compass_kv_write_done\":\"tcp://127.0.0.1:${DONE_PORTS[i]}\""
    kv+="}"
    if [[ $cmd == record ]]; then
        command=("$cmd" "${flags[@]}")
        kv="{\"kv_role\":\"$role\",\"kv_connector\":\"mooncake\",\"handshake_port\":${KV_PORTS[i]}}"
        [[ -z ${KV_EXTRA:-} ]] || kv=$(python3 -c 'import json, sys
print(json.dumps({**json.loads(sys.argv[2]), **json.loads(sys.argv[1])}))' "$kv" "$KV_EXTRA")
    fi
    local side_extra=PREFILL_KV_EXTRA
    [[ ${sides[i]} == prefill ]] || side_extra=DECODE_KV_EXTRA
    [[ -z ${!side_extra:-} ]] || kv=$(python3 -c 'import json, sys
kv, extra, keep = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3].split()
if sys.argv[4] == "run":
    left = sorted(k for k in extra if k not in keep)
    if left:
        print(f"deploy.sh: {sys.argv[5]} leaves {left} out of a simulated instance", file=sys.stderr)
    extra = {k: v for k, v in extra.items() if k in keep}
print(json.dumps({**extra, **kv}))' "$kv" "${!side_extra}" "$SIM_KV_KEYS" "$cmd" "$side_extra")
    command+=("${own[@]}" --out "$OUT/$name" -- "${atom[@]}" --kv-transfer-config "$kv")
    # Its process id, which exec keeps; then this tree, checked, and the caller's directory.
    printf 'echo $$ >%q\n. %q && compass_env %q && compass_require_tree %q && cd %q &&\n    exec' \
        "$OUT/$name.pid" "$ROOT/scripts/compass/_lib.sh" "$ROOT" "$ROOT" "$PWD" >"$OUT/$name.sh"
    # Its variables, after compass_env so they win, quoted so a JSON value arrives whole.
    ((${#vars[@]} == 0)) || printf ' %q' env "${vars[@]}" >>"$OUT/$name.sh"
    printf ' %q' python3 -c 'from atom.compass.cli import main; main()' "${command[@]}" >>"$OUT/$name.sh"
    "${prefix[@]}" bash "$OUT/$name.sh" >"$OUT/$name.log" 2>&1 &
    pids[i]=$!
}
url() { # I: instance I's URL, once its server.json names it, while every instance runs
    local deadline=$((SECONDS + WAIT_S)) j
    until [[ -s $OUT/${names[$1]}/server.json ]]; do
        for j in "${!sides[@]}"; do
            kill -0 "${pids[j]}" 2>/dev/null || fail "the ${names[j]} instance exited before it answered; see $OUT/${names[j]}.log"
        done
        ((SECONDS < deadline)) || fail "no ${names[$1]} instance after $WAIT_S s"
        sleep 1
    done
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["url"])' "$OUT/${names[$1]}/server.json"
}
for i in "${!sides[@]}"; do
    instance "$i"
    # Its process id first, so that any stop from here on reaches it.
    instance_pid "$i" >/dev/null || {
        kill -0 "${pids[i]}" 2>/dev/null || fail "the ${names[i]} instance exited before it answered; see $OUT/${names[i]}.log"
        fail "the ${names[i]} instance wrote no process id in $PREFIX_S s; see $OUT/${names[i]}.log"
    }
done
urls=()
for i in "${!sides[@]}"; do urls+=("$(url "$i")"); done
# Real, each --on instance's URL and prefix, whose node's clock is paired with
# this one's; the pairing leaves out an instance on this node.
clocked=() watch=
if [[ $cmd == record ]]; then
    for i in "${!sides[@]}"; do
        declare -n on=on_$i
        ((${#on[@]} == 0)) || clocked+=("${urls[i]}" "${#on[@]}" "${on[@]}")
        unset -n on
    done
fi
pair=(python3 -m atom.compass.drive.clock_pair "$OUT/clock_pairs.jsonl" --timeout "$PREFIX_S")
# The router's requests to each instance, and the wait on the router, go
# direct, never through a proxy the environment names.
hosts=127.0.0.1
for u in "${urls[@]}"; do u=${u#*://} && hosts+=,${u%:*}; done
export NO_PROXY=$hosts${NO_PROXY:+,$NO_PROXY} no_proxy=$hosts${no_proxy:+,$no_proxy}

ROUTER=("$ATOMESH" launch --host 127.0.0.1 --port "$ROUTER_PORT" --prometheus-port "$PROMETHEUS_PORT")
[[ -z $RUN_FILE ]] || ROUTER+=("${COMPASS_SIM_ROUTER_BOUNDS[@]}")
ROUTER+=(--pd-disaggregation)
for i in "${!sides[@]}"; do ROUTER+=("--${sides[i]}" "${urls[i]}"); done
ROUTER+=("${router[@]}")
router_env=()
[[ -z $ROUTER_LP ]] || router_env=(ATOM_COMPASS_CLOCK_ENDPOINT="$CLOCK" ATOM_COMPASS_SEED="${SEED:-20260707}" ATOM_COMPASS_ROUTER_LOOKAHEADS="$LOOKAHEADS")
if [[ -n $ROUTER_LP ]] && ((NP > 1 || ND > 1)); then
    # As pd_sim.sh gives them: the router LP's workers in the run file's
    # worker_order as (URL, DP rank) pairs, and above one instance of a side
    # each URL's frontend LP.
    order=$(python3 - "$RUN_FILE" "$NP" "${urls[@]}" <<'EOF'
import json, sys
from atom.compass.run import worker_order

run_file, np_, *urls = sys.argv[1:]
url = {"P": urls[: int(np_)], "D": urls[int(np_) :]}
try:
    order = worker_order(json.load(open(run_file)))
except Exception as e:
    sys.exit(f"deploy: {e}")
workers = {u: f"frontend-{s}{k}" for s in url if len(url[s]) > 1 for k, u in enumerate(url[s])}
print(json.dumps([[url[s][k], r] for s in order for k, r in order[s]]))
print(json.dumps(workers))
EOF
) || fail "the router LP's worker order: $order"
    router_env+=(ATOM_COMPASS_ROUTER_ORDER="${order%%$'\n'*}" ATOM_COMPASS_ROUTER_WORKERS="${order#*$'\n'}")
fi
# Atomesh holds a descriptor for each client connection and each upstream one,
# and raises no limit of its own: at a soft limit of 1024 a c=1228 router ran
# out, answered 502 for the prefill instances it could not reach, and a request
# the router LP was released never reached it, so the run hung (#3720). It starts
# at the hard limit, as a `compass run` server raises its own (#3819).
(ulimit -n "$(ulimit -Hn)" 2>/dev/null || true; exec env "${router_env[@]}" "${ROUTER[@]}") >"$OUT/router.log" 2>&1 &
R=$!
compass_wait_up router $R "http://127.0.0.1:$ROUTER_PORT/v1/models"
# A stop from here on, one server.json's reader sends included, also stops the
# prefill instances, so the wait below ends even when the stop comes before it:
# each here by a SIGTERM, each elsewhere through its prefix, as the stop below
# reaches every instance. A signal to the prefix itself would end it apart
# from its command, with a status of its own, ssh's 255 or a shell's 143, on
# which a normal stop failed; ended by its command, it ends with its status.
stop_prefills() {
    local i
    for ((i = 0; i < NP; i++)); do stop_instance "$i"; done
}
stopped=0
trap 'stopped=1; stop_prefills' TERM INT
# Each other node's clock paired as the deployment starts and every
# CLOCK_PAIR_S after, apart, so a prefix slow to pair delays no start.
if ((${#clocked[@]})); then
    (cd / && exec "${pair[@]}" --every "${CLOCK_PAIR_S:-60}" -- "${clocked[@]}") &
    watch=$!
fi
# server.json names a child of this script that lives as long as it does, so a
# client's stop, sent to its parent, reaches the script whatever else has left.
# It holds none of the script's output, which its caller may read to the end.
tail --pid=$$ -f /dev/null >/dev/null 2>&1 &
python3 -c 'import json, sys
url, pid, run_file, clock, out = sys.argv[1:]
ready = {"url": url, "pid": int(pid)}
if run_file:
    ready |= {"clock_endpoint": clock, "run_file": run_file}
open(out + ".part", "w").write(json.dumps(ready))' \
    "http://127.0.0.1:$ROUTER_PORT" $! "$RUN_FILE" "$CLOCK" "$OUT/server.json"
mv "$OUT/server.json.part" "$OUT/server.json"
echo "deploy: ready at http://127.0.0.1:$ROUTER_PORT"

wait -n $authority "${pids[@]}" $R || true
# A router LP leaves when the traffic closes its clock connection as aiperf
# ends, which ends the run: no failure, so serve on until the stop.
if ((!stopped)) && [[ -n $ROUTER_LP ]] && grep -qs 'ClockAbort: traffic closed its connection' "$OUT/router.log"; then
    until ((stopped)); do sleep 1; done
fi
rc=$((!stopped))
((stopped)) || echo "deploy: a process of the deployment left before a stop" >&2
# Each instance stops its server, whose frontend ends a simulated run through
# the authority; the authority leaves once it has written the summary.
# The router stops last: a router LP's run needs it until then.
# A prefill instance the stop's trap reached is not stopped again: a second
# SIGTERM after `compass record` has stopped its server would end it there.
for i in "${!sides[@]}"; do ((stopped && i < NP)) || stop_instance "$i"; done
deadline=$((SECONDS + WAIT_S))
for pid in "${pids[@]}" $authority $R; do
    ((pid != R)) || kill -TERM $R 2>/dev/null || true
    while kill -0 $pid 2>/dev/null; do
        ((SECONDS < deadline)) || { kill -KILL $pid; rc=1; echo "deploy: process $pid killed after $WAIT_S s" >&2; }
        sleep 1
    done
    wait $pid || ((pid == R)) || rc=1
done
if [[ -n $watch ]]; then
    kill -TERM "$watch" 2>/dev/null || true
    wait "$watch" || true
    (cd / && "${pair[@]}" -- "${clocked[@]}") || true
fi
trap - EXIT
# Each instance: its directory's name, side and index, and, real, its
# Mooncake handshake port, which a prefill's transfers carry to the decode.
instances=()
for i in "${!sides[@]}"; do
    instances+=("${names[i]} ${sides[i]:0:1} ${index[i]} $([[ $cmd == record ]] && echo "${KV_PORTS[i]}" || echo -)")
done
python3 -c 'import json, os, sys
out, link, across, k, *rest = sys.argv[1:]
held, rest = rest[: int(k)], rest[int(k) :]
n, rest = int(rest[0]), rest[1:]
run = {"command": rest[:n], "kv_link": link, "across_nodes": across == "1", "router": rest[n:]}
run["instances"] = []
for line in held:
    name, side, index, port = line.split()
    run["instances"].append({"name": name, "side": side.upper(), "index": int(index)})
    if port != "-":
        run["instances"][-1]["handshake_port"] = int(port)
if os.path.exists(out + "/run_file.json"):
    run["run_file"] = json.load(open(out + "/run_file.json"))
open(out + "/run.json.part", "w").write(json.dumps(run))' \
    "$OUT" "$KV_LINK" "$ACROSS_NODES" "${#instances[@]}" "${instances[@]}" "${#ARGV[@]}" "${ARGV[@]}" "${ROUTER[@]}"
mv "$OUT/run.json.part" "$OUT/run.json"
if [[ -s $OUT/artifacts/client_run.json ]]; then
    (cd / && python3 -m atom.compass.report.split "$OUT") ||
        { echo "deploy: split: atom.compass.report.split exited $?" >&2; rc=1; }
    (cd / && python3 -m atom.compass.report.prometheus "$OUT") ||
        echo "deploy: prometheus: atom.compass.report.prometheus exited $?" >&2
    (cd / && python3 -m atom.compass.report.html "$OUT") ||
        { echo "deploy: report: atom.compass.report.html exited $?" >&2; rc=1; }
fi
exit $rc
