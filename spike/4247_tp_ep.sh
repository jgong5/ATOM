#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Spike #4247: DeepSeek-V4.1-Flash at -tp 4 --enable-expert-parallel, dp 1, fp8 KV, otherwise cell A's
# flags (scripts/compass/DSV41_MI355.md, `args agg`, less --enable-dp-attention). Does ATOM serve it, and
# does Compass simulate it?
#
#   spike/4247_tp_ep.sh gpu OUT   a real server and R short AgentX client runs, in a GPU container whose
#                                 ROCR_VISIBLE_DEVICES names the four leased GPUs and nothing else
#   spike/4247_tp_ep.sh sim OUT   the same server simulated on placeholder memory readings, priced by the
#                                 step stub, and the same client; then the step roofline over the modules
#                                 a simulated run walks, and its MoE's exchange; in a container with no
#                                 GPU driver
#
# Each takes an OUT that does not exist and prints its result. Image rocm/atom-dev:nightly_202611240650,
# the one this tree's gate names. MODEL (/models/dsv41) and TRACES (/cc/traces.jsonl, sha256 29b6a19e...)
# as client.sh reads them; WHEEL the agentx-harness 56a0cf70 aiperf wheel; C, D and W the client's
# concurrency, duration and warmup requests per lane (8, 300 s, 1); R the client runs against one real
# server (2); PORT the real server's port (8000).
set -uo pipefail
T=$(cd "$(dirname "$0")/.." && pwd)
CMD=${1:?gpu or sim}
OUT=${2:?OUT}
[ ! -e "$OUT" ] || { echo "refused: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
: "${MODEL:=/models/dsv41}" "${TRACES:=/cc/traces.jsonl}" "${C:=8}" "${D:=300}" "${W:=1}" "${R:=2}" "${PORT:=8000}"
: "${WHEEL:=/share_nfs/xingjche/agent_team/wheels/aiperf-0.12.0-py3-none-any.whl}"
export MODEL TRACES AITER_LOG_LEVEL=WARNING PYTHONPATH=$T
# Cell A's environment, real and simulated alike, and its client's session header. Only ATOM runs offline
# (HF_HUB_OFFLINE): offline, aiperf 0.12.0 takes a tokenizer path for a hub repo id and refuses it.
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
CLIENT=(--agentx --concurrency "$C" --duration "$D" --warmup-per-lane "$W")
# The platform flags, and the same as the Python below reads them: MI355X at the committed readings
PLATFORM=(--device mi355x --ptl disabled)
SPEC='from types import SimpleNamespace as N
from atom.compass import run as R
from atom.compass.artifacts import ArtifactStore
from atom.compass.drive import runfile
def machine(store):
    flags = N(device="mi355x", ptl="disabled", stack=None, network=None, step_linear=None, step_grid=None, host_laws=None)
    return R._machine({"machine": runfile.spec(flags, ArtifactStore(store)).echo()})
'

venv() {  # DIR [HARNESS]: the wheel's aiperf, with compass-harness when HARNESS, its other packages the image's
  /opt/venv/bin/python3 -m venv "$1" && echo 'import site; site.addsitedir("/opt/agentx-venv/lib/python3.12/site-packages"); site.addsitedir("/opt/venv/lib/python3.12/site-packages")' \
    >"$1/lib/python3.12/site-packages/zz_image.pth" || return 1
  [ -z "${2-}" ] || { cp -r "$T/compass_harness" "$1/src" && set -- "$1" "$1/src"; }
  "$1/bin/pip" install -q --no-deps --no-build-isolation "$WHEEL" ${2:+"$2"}
}

report() {  # OUT: the client's summary and, for a simulated run, its steps priced, refused and their terms
  python3 - "$1" <<'EOF'
import collections, glob, json, os, re, sys
out = sys.argv[1]
export = os.path.join(out, "artifacts/profile_export_aiperf.json")
if os.path.exists(export):
    e = json.load(open(export))
    stat = lambda name, k="p50": (e.get(name) or {}).get(k)
    errors = sum(x.get("count", 0) for x in e.get("error_summary") or [])
    print(f"client: requests {stat('request_count', 'avg')}, errors {errors}, TTFT p50 {stat('time_to_first_token')} ms, "
          f"TPOT p50 {stat('inter_token_latency')} ms, output tokens/s {stat('output_token_throughput', 'avg')}")
else:
    print(f"client: no {export}")
if not os.path.exists(os.path.join(out, "summary.json")):
    sys.exit(0)
s = json.load(open(os.path.join(out, "summary.json")))["schedule"]
r = s["refusals"]
print(f"steps {r['steps']}, refused {round(r['fraction_of_steps'] * r['steps'])}, priced seconds {s['predicted_seconds']:.1f}")
for reason, n in r["reasons"]:
    print(f"refusal x{n}: {reason}")
terms, lines = collections.Counter(), 0
for path in glob.glob(os.path.join(out, "steps/*.cost.jsonl")):
    for row in map(json.loads, open(path)):
        lines += "terms" in row
        terms.update(t["name"] for t in row.get("terms", ()))
print(f"cost lines {lines}, terms charged: {dict(terms)}")
print(f"all-to-all terms: {[t for t in terms if re.search(r'all.?to.?all|dispatch|combine|mori', t, re.I)] or 'none'}")
EOF
}

readings() {  # STORE: placeholder memory readings at this configuration's key, every member assumed
  HF_HUB_OFFLINE=1 python3 - "$1" "${ARGS[@]}" <<EOF
$SPEC
import datetime, json, sys
from atom.compass.artifacts import GateState, Provenance, machine_platform, member_name, roots_for
from atom.compass.drive import run as drive_run
from atom.compass.run import memory
store, *atom_args = sys.argv[1:]
spec = machine(store)
config = drive_run._config(atom_args, spec.value("device.arch"))
key, topology = memory.key_of(config, spec), memory.topology(config)
# A DeepSeek-V4.1-Flash fp8 KV start's readings on MI355X: a DPA2-EP2 prefill side's of cell B
member = dict(capacity_bytes=309220868096, driver_and_collective_reserve_bytes=13706985472,
    allocator_retained_after_load_bytes=2267146496, persistent_forward_buffer_bytes=35517165568,
    activation_bytes=12924484096, warmup_tokens=32768, cudagraph_pool_bytes=2468347904, record="sha256:" + "0" * 64,
    assumed="spike 4247 placeholder: a DPA2-EP2 fp8 start's readings; no TP4-EP4 dp 1 start was measured")
ArtifactStore(store).publish(key, topology=topology, gates=GateState.of(),
    provenance=Provenance("spike/4247_tp_ep.sh", datetime.datetime.now(datetime.timezone.utc).isoformat(),
        roots_for({"atom": "atom", "aiter": "aiter"})),
    conditions=memory.conditions(key.values, **machine_platform(spec)),
    members={member_name(memory.STEM, at, memory.EXTENSION): json.dumps(member).encode() for at in topology.ranks()})
print(f"readings: published {key.token()}, {len(list(topology.ranks()))} ranks")
EOF
}

roofline() {  # STORE: the step roofline on eight DSpark5 verify rows, and each MoE shape's exchange
  HF_HUB_OFFLINE=1 python3 - "$1" "${ARGS[@]}" <<EOF
$SPEC
import sys
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.step_roofline import StepRooflineBackend, _moe
from atom.compass.backends.work import allreduce
from atom.compass.validate.bound import walked
store, *atom_args = sys.argv[1:]
spec = machine(store)
modules = walked(atom_args, spec.value("device.arch"))  # as a simulated run walks its meta build
rows = (RequestShape(6, 4096, True),) * 8
try:
    cost = StepRooflineBackend(spec, modules).price(BatchView(rows, produces_output=True))
    print(f"roofline: priced {cost.target_seconds:.6f} s")
except CostRefused as refused:
    for r in refused.declined:
        print(f"roofline refuses: {r.source}: {r.reason}")
for m in {(m.draft > 0, m.dims): m for m in modules if m.kind == "moe"}.values():
    experts, local, k, inter, hidden, ranks, dp = m.dims[:7]
    before, _, after = _moe(m, BatchView(rows, produces_output=True), 48)
    print(f"moe {'draft' if m.draft else 'target'}: {local} of {experts} experts a rank, top {k}, {ranks} ranks, "
          f"dp {dp}; collectives before the experts {before.comm}, after {after.comm}; "
          f"a TP all-reduce of the 48 rows {allreduce(ranks, 48 * hidden * 2).comm}")
EOF
}

case $CMD in
gpu)
  venv "$OUT/venv" || exit 1
  (cd / && HF_HUB_OFFLINE=1 exec python3 -m atom.entrypoints.openai_server "${ARGS[@]}" --server-port "$PORT") >"$OUT/server.log" 2>&1 &
  pid=$! end=$((SECONDS + 3600))
  until curl -sf --noproxy '*' --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $pid 2>/dev/null && ((SECONDS < end)) || { echo "server: no health; tail of $OUT/server.log:"; tail -20 "$OUT/server.log"; kill $pid 2>/dev/null; exit 1; }
    sleep 10
  done
  echo "server up after $SECONDS s"
  for k in $(seq "$R"); do  # R client runs, one after another, against the one server
    mkdir "$OUT/run$k" && printf '{"url": "http://127.0.0.1:%s", "pid": %s}\n' "$PORT" $pid >"$OUT/run$k/server.json"
    HARNESS_PYTHON=$OUT/venv/bin/python "$T/scripts/compass/client.sh" "$OUT/run$k" "${CLIENT[@]}" >"$OUT/run$k.log" 2>&1
    echo "run$k: client rc=$?, after $SECONDS s"; tail -2 "$OUT/run$k.log"
    report "$OUT/run$k"
  done
  kill -TERM $pid; wait $pid
  echo "server log, first lines naming an error or refusal:"; grep -m5 -iE "error|traceback|refus" "$OUT/server.log"
  ;;
sim)
  venv "$OUT/venv" harness && readings "$OUT/store" || exit 1
  # A speculative step's stub price needs a draft term, or the runner refuses the step
  echo '{"draft_step": 0.001}' >"$OUT/stub.json"
  HF_HUB_OFFLINE=1 "$T/scripts/compass/compass" run --step-stub "$OUT/stub.json" "${PLATFORM[@]}" \
    --store "$OUT/store" --out "$OUT/stub" -- "${ARGS[@]}" >"$OUT/stub.log" 2>&1 &
  pid=$!
  until [ -s "$OUT/stub/server.json" ]; do
    kill -0 $pid 2>/dev/null || { echo "stub: no server; tail of $OUT/stub.log:"; tail -20 "$OUT/stub.log"; exit 1; }
    sleep 5
  done
  HARNESS_PYTHON=$OUT/venv/bin/python "$T/scripts/compass/client.sh" "$OUT/stub" "${CLIENT[@]}" --stop >"$OUT/stub.client.log" 2>&1
  echo "stub: client rc=$?, after $SECONDS s"; tail -2 "$OUT/stub.client.log"
  wait $pid
  report "$OUT/stub"
  roofline "$OUT/store"
  ;;
*) sed -n '2,18p' "$0"; exit 2 ;;
esac
