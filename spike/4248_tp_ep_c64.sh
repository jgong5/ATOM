#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Spike #4248: is Compass within 10% of a real AgentX c=64 run of DeepSeek-V4.1-Flash at TP4 with expert
# parallelism, one DP rank, fp8 KV (#4247's configuration), when its step and host laws are fitted on one
# calibration record of that configuration alone? LLM-Booster #752's protocol: TTFT p90, TPOT p90 and
# throughput per GPU first, each the mean over three send orders.
#
#   spike/4248_tp_ep_c64.sh design DATA  the calibration design: 4248_cal_design.json through gen_v1.py and
#                                        plan_v1.py of CAL_TREE, rehearsed on LAW
#   spike/4248_tp_ep_c64.sh gpu DATA     on four GPUs of an isolated node: memory readings, a warm record
#                                        nothing reads, the target record (its client under
#                                        collect_metrics.py --standalone), then the calibration record: the
#                                        design's segments on one server, each started once the one before
#                                        has drained, as CAL_TREE's cal_v1.sh runs them
#   spike/4248_tp_ep_c64.sh fit DATA     the step law and host laws from the calibration record alone
#   spike/4248_tp_ep_c64.sh sim DATA K   simulated run K of the target's client, send order seed K
#   spike/4248_tp_ep_c64.sh score DATA   one_real.py on the target and every sim<K>; each metric's error per
#                                        send order, their mean and mean absolute value; DASH_TREE's
#                                        compare_dashboard.py page
#
# DATA is a directory every node reaches. A gpu step whose output is complete is skipped. MODEL
# (/models/dsv41) and TRACES (/cc/traces.jsonl, sha256 29b6a19e...) as client.sh reads them; GPU the
# rocm-smi indices (0,1,2,3); WHEEL the agentx-harness 56a0cf70 aiperf wheel; PROM a Prometheus 3.5.0
# binary; CAL_TREE a 752/dev snapshot (07deefe54) holding the #739 v1.1 tools; LAW the law a design is
# rehearsed on (this tree's gpu40 law at TP4 EP4, bf16 KV); DASH_TREE a snapshot of #3939's head. A record
# reads the cards' PTL state, disabled on pit2's MI355X; every simulated run states it, as DSV41_MI355.md's.
set -uo pipefail
T=$(cd "$(dirname "$0")/.." && pwd)
CMD=${1:?design, gpu, fit, sim or score}
DATA=$(mkdir -p "${2:?DATA}" && cd "$2" && pwd)
: "${MODEL:=/models/dsv41}" "${TRACES:=/cc/traces.jsonl}" "${GPU:=0,1,2,3}" "${PROM:=}" "${CAL_TREE:=}" "${DASH_TREE:=}"
: "${WHEEL:=/share_nfs/xingjche/agent_team/wheels/aiperf-0.12.0-py3-none-any.whl}"
: "${LAW:=$T/scripts/compass/laws/deepseek-v4.1-flash-tp4-ep4-dspark5-mi355x-gpu40.json.xz}"
export MODEL TRACES AITER_LOG_LEVEL=WARNING PYTHONPATH=$T
# Cell A's environment, real and simulated alike, and its client's session header (#4247)
export ATOM_MONO_ENABLE=1 ATOM_FRONTEND_PREFIX_HASH=1 ATOM_DP_SESSION_AFFINITY=0 ATOM_DISABLE_MORI_EP=1 \
  ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_DSPARK_DISABLE_COMPILE=1 ATOM_DISABLE_MMAP=true PYTHONUNBUFFERED=1 \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1
ARGS=(--model "$MODEL" --trust-remote-code -tp 4 --enable-expert-parallel --moe-backend standard
  --all2all-backend none --kv_cache_dtype fp8 --index-cache-dtype fp8 --gpu-memory-utilization 0.85
  --max-num-seqs 64 --max-num-batched-tokens 32768 --attn-prefill-chunk-size 65536 --enable_prefix_caching
  --block-size 128 --state-checkpoint-interval-tokens 8192 --level 3 --cudagraph-mode FULL
  --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,64]" --method dspark --num-speculative-tokens 5
  --tool-call-parser dsml_v41 --dspark-config '{"confidence_schedule":false}'
  --spec-decode-acceptance-length 3.51)
CLIENT=(--agentx --concurrency 64 --duration 900)
S=$DATA/store C=$DATA/cal
mkdir -p "$DATA/logs" "$S"
compass() { HF_HUB_OFFLINE=1 "$T/scripts/compass/compass" "$@"; }
die() { echo "$CMD: $*" >&2; exit 1; }
say() { echo "$(date -u +%FT%TZ) $*"; }
token() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["token"])' "$1"; }

venv() {  # DIR [HARNESS]: the wheel's aiperf, with compass-harness when HARNESS, its other packages the image's
  [ -x "$1/bin/python" ] && return 0
  /opt/venv/bin/python3 -m venv "$1" && echo 'import site; site.addsitedir("/opt/agentx-venv/lib/python3.12/site-packages"); site.addsitedir("/opt/venv/lib/python3.12/site-packages")' \
    >"$1/lib/python3.12/site-packages/zz_image.pth" || return 1
  [ -z "${2-}" ] || { cp -r "$T/compass_harness" "$1/src" && set -- "$1" "$1/src"; }
  "$1/bin/pip" install -q --no-deps --no-build-isolation "$WHEEL" ${2:+"$2"}
}

serve() {  # OUT [VAR=VALUE...]: `compass record`, or with SIM_LAWS `compass run`, into OUT, until it answers
  local out=$1; shift
  if [ -n "${SIM_LAWS-}" ]; then
    env "$@" HF_HUB_OFFLINE=1 "$T/scripts/compass/compass" run $SIM_LAWS --device mi355x --ptl disabled --store "$S" \
      --out "$out" -- "${ARGS[@]}" >"$out.log" 2>&1 &
  else
    env "$@" HF_HUB_OFFLINE=1 "$T/scripts/compass/compass" record --gpu "$GPU" --store "$S" --out "$out" -- "${ARGS[@]}" >"$out.log" 2>&1 &
  fi
  SERVER=$!
  until [ -s "$out/server.json" ]; do
    kill -0 $SERVER 2>/dev/null || { tail -20 "$out.log"; die "no server for $out"; }
    sleep 10
  done
  say "$out: serving"
}

stop() {  # OUT: SIGTERM the command serving OUT, wait for its run.json, and mark its export as client.sh --stop does
  local pid; pid=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["pid"])' "$1/server.json")
  kill -TERM "$(awk '{print $4}' "/proc/$pid/stat")" && wait "$SERVER"
  [ -s "$1/run.json" ] || die "$1: no run.json after the stop"
  python3 - "$1/artifacts/client_run.json" <<'EOF' && (cd / && python3 -m atom.compass.report.split "$1")
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps(json.loads(path.read_text()) | {"server_stopped": True}, sort_keys=True))
EOF
}

drained() {  # URL: until the server holds no request, as cal_v1.sh waits between segments
  for _ in $(seq 180); do
    [ "$(curl -s --noproxy '*' -m 10 "$1/metrics" | awk '/^atom:requests_(running|waiting)[ {]/ {s += $NF} END {print s + 0}')" = 0 ] && return 0
    sleep 5
  done
}

calibrate() {  # the design's segments on one server under its server_env, each lane a client.sh run of its traces.
  # No lane is pinned: at one DP rank each lands on rank 0, and a pin's data_parallel_rank would give the record a
  # second set of request parameters, which compass fit refuses
  local plan=$DATA/design/plan.json url k lane file c n seed extra pids
  plan() { python3 -c "import json, sys; p = json.load(open('$plan')); $1" "${@:2}"; }
  rm -rf "$C" && mkdir -p "$C/logs" && cp "$plan" "$C/"
  serve "$C/real" $(plan 'print(" ".join(f"{k}={v}" for k, v in p["server_env"].items()))')
  url=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["url"])' "$C/real/server.json")
  for k in $(seq 0 $(($(plan 'print(len(p["sequence"]))') - 1))); do
    seg=$(plan 'print(p["sequence"][int(sys.argv[1])]["segment"])' "$k") pids=()
    say "$seg: start"
    while read -r lane file c n seed; do
      mkdir -p "$C/$seg/$lane" && cp "$C/real/server.json" "$C/$seg/$lane/"
      extra="--apply-chat-template --trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0 --random-seed $seed --num-conversations $n"
      TRACES=$DATA/design/$file AIPERF_EXTRA_ARGS=$extra HARNESS_PYTHON=$DATA/venv-real/bin/python \
        "$T/scripts/compass/client.sh" "$C/$seg/$lane" --traces "0-$((n - 1))" --concurrency "$c" --duration 3600 \
        >"$C/logs/client_${seg}_$lane.log" 2>&1 &
      pids+=($!)
    done < <(plan 'print("\n".join(" ".join(str(x[f]) for f in ("name", "file", "concurrency", "traces", "seed")) for x in p["sequence"][int(sys.argv[1])]["lanes"]))' "$k")
    for p in "${pids[@]}"; do wait "$p" || { stop "$C/real"; die "a client of $seg failed; see $C/logs"; }; done
    drained "$url" && say "$seg: drained"
    sleep 15
  done
  local cmd; cmd=$(awk '{print $4}' "/proc/$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["pid"])' "$C/real/server.json")/stat")
  kill -TERM "$cmd" && wait "$SERVER"; [ -s "$C/real/run.json" ] || die "calibration: no run.json"
}

lanes() { find "$C" -path "$C/real" -prune -o -name client_run.json -print | sort | while read -r e; do echo --client-run "$C/real" "$e"; done; }

case $CMD in
design)
  [ -n "$CAL_TREE" ] || die "CAL_TREE is not set"
  rm -rf "$DATA/design" && xz -dc "$LAW" >"$DATA/logs/law.json" || exit 1
  PYTHONPATH=$CAL_TREE python3 -P "$CAL_TREE/scripts/compass/gen_v1.py" "$T/spike/4248_cal_design.json" "$DATA/design" &&
    PYTHONPATH=$CAL_TREE python3 -P "$CAL_TREE/scripts/compass/plan_v1.py" "$DATA/design" --law "$DATA/logs/law.json" >/dev/null &&
    sed -n '/^Predicted/p; /^| segment | starts/,/^$/p; /^Verdict/p' "$DATA/design/plan.md"
  ;;
gpu)
  [ -s "$DATA/design/plan.json" ] && [ -x "$PROM" ] || die "no design, or PROM is not a Prometheus binary"
  venv "$DATA/venv-real" || die "venv"
  say "node $(hostname): $(uptime)"
  if [ ! -s "$DATA/memory/memory_readings.json" ]; then
    compass measure memory --gpu "$GPU" --store "$S" --out "$DATA/memory" -- "${ARGS[@]}" >"$DATA/memory.log" 2>&1 || die "measure memory; see $DATA/memory.log"
  fi
  say "memory: $(token "$DATA/memory/memory_readings.json")"
  if ! grep -qs '"server_stopped": true' "$DATA/warm/artifacts/client_run.json"; then
    rm -rf "$DATA/warm" && serve "$DATA/warm"
    HARNESS_PYTHON=$DATA/venv-real/bin/python "$T/scripts/compass/client.sh" "$DATA/warm" --agentx --concurrency 64 \
      --duration 120 --warmup-per-lane 1 --stop >"$DATA/warm.client.log" 2>&1 || die "warm; see $DATA/warm.client.log"
    wait "$SERVER"
  fi
  if ! grep -qs '"server_stopped": true' "$DATA/target/artifacts/client_run.json"; then
    rm -rf "$DATA/target" "$DATA/target-prom" && serve "$DATA/target"
    say "target: $(uptime)"
    url=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["url"])' "$DATA/target/server.json")
    ATOMESH_PROMETHEUS_BIN=$PROM python3 "$T/.github/scripts/atomesh/observability/collect_metrics.py" \
      --output "$DATA/target-prom" --model "$MODEL" --standalone "${url#http://}" -- \
      env HARNESS_PYTHON="$DATA/venv-real/bin/python" "$T/scripts/compass/client.sh" "$DATA/target" "${CLIENT[@]}" \
      >"$DATA/target.client.log" 2>&1 || { stop "$DATA/target"; die "target; see $DATA/target.client.log"; }
    stop "$DATA/target"
    say "target: done, $(uptime)"
  fi
  [ -s "$C/real/run.json" ] || calibrate
  say "calibration: done, $(uptime)"
  for r in warm target cal/real; do python3 - "$DATA/$r/run.json" <<'EOF'; done
import json, sys
from atom.compass.backends import step_labels
print(sys.argv[1], step_labels.refusal(json.load(open(sys.argv[1]))) or "no JIT build while serving")
EOF
  ;;
fit)
  mapfile -t runs < <(lanes | tr ' ' '\n')
  COMPASS_PREFILL_LABEL=record compass fit steps --backend step-linear --form history --model "$MODEL" "$C/real" \
    "${runs[@]}" --store "$S" --out "$DATA/fit/steps" >"$DATA/logs/fit_steps.log" 2>&1 || die "fit steps; see $DATA/logs/fit_steps.log"
  compass fit host "$C/real" "${runs[@]}" --store "$S" --out "$DATA/fit/host" >"$DATA/logs/fit_host.log" 2>&1 ||
    die "fit host; see $DATA/logs/fit_host.log"
  python3 - "$DATA/fit/steps/step_linear_law.json" <<'EOF'
import json, sys
law = json.load(open(sys.argv[1]))
print(f"step law {law['token']}: unfitted {law['params']['unfitted']}, decode rungs {sorted(law['params']['decode'], key=int)}")
EOF
  echo "host laws $(token "$DATA/fit/host/host_laws.json")"
  ;;
sim)
  K=${3:?K, the send order seed} OUT=$DATA/sim$K
  [ ! -e "$OUT" ] || die "$OUT exists"
  venv "$DATA/venv-sim" harness || die "venv"
  SIM_LAWS="--step-linear $(token "$DATA/fit/steps/step_linear_law.json") --host-laws $(token "$DATA/fit/host/host_laws.json")"
  serve "$OUT"
  COMPASS_HARNESS_SEND_ORDER_SEED=$K HARNESS_PYTHON=$DATA/venv-sim/bin/python "$T/scripts/compass/client.sh" "$OUT" \
    "${CLIENT[@]}" --stop >"$OUT.client.log" 2>&1 || die "client; see $OUT.client.log"
  wait "$SERVER"
  python3 - "$OUT/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["schedule"]
r = s["refusals"]
m = {k: round(v["fraction_of_predicted_seconds"], 4) for k, v in s["seconds_by_method"].items() if v["fraction_of_predicted_seconds"]}
print(f"steps {r['steps']}, refused {r['count']}, extrapolated {s['extrapolated_steps']}, seconds by method {m}")
EOF
  ;;
score)
  sims=("$DATA"/sim[0-9]*/)
  python3 "$T/scripts/compass/one_real.py" "$DATA/target" "${sims[@]%/}" | tee "$DATA/one_real.txt"
  tail -1 "$DATA/one_real.txt" | python3 -c '
import json, sys
rows = json.load(sys.stdin)
real = rows.pop("real")
for m in ("p90 TTFT", "TPOT p90 ms", "throughput per GPU"):
    e = [rows[s][m] / real[m] - 1 for s in rows]
    print(f"{m}: " + ", ".join(f"{s} {x:+.1%}" for s, x in zip(rows, e))
          + f"; mean {sum(e) / len(e):+.1%}, mean absolute {sum(map(abs, e)) / len(e):.1%}")'
  [ -n "$DASH_TREE" ] || exit 0
  cell="$DATA/dash/DeepSeek-V4.1-Flash/aggregated/TP4 EP4/c64"
  rm -rf "$DATA/dash" && mkdir -p "$cell/prom" "$cell/sims" && ln -s "$DATA/target" "$cell/real" &&
    ln -s "$DATA/target-prom/report-data.json" "$cell/prom/" && cp "$DATA/one_real.txt" "$cell/" &&
    for s in "${sims[@]%/}"; do ln -s "$s" "$cell/sims/${s##*/}"; done
  python3 "$DASH_TREE/scripts/compass/compare_dashboard.py" --scan "$DATA/dash" --out "$DATA/dashboard.html" --window 900
  ;;
*) sed -n '2,25p' "$0"; exit 2 ;;
esac
