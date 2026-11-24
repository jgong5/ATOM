"""sample.py OUT DELAY SECONDS [RATE]: stack samples of every Python LP process of run.sh's run OUT (#4029).

Runs in a container with CAP_SYS_PTRACE that shares the sim container's PID namespace. When OUT's client starts
its profiling phase, waits DELAY seconds, then py-spy records each process non-blocking at RATE Hz (default 50)
for SECONDS, raw, into OUT.spy/<lp>.<pid>.raw: frontend-P0 .. frontend-D3, engine-P0 .. engine-D3 (every engine
core under that frontend), the Clock Authority (ca), the traffic LP (traffic) and the Store (mooncake_store).
Over the same SECONDS, each of their threads' CPU and run-queue wait seconds (/proc schedstat) go into
OUT.spy/schedstat.json, and how many of the run's CPUs were busy, second by second, into OUT.spy/cpus.json.
"""

import json
import os
import re
import subprocess
import sys
import time

out, delay, seconds = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
rate = sys.argv[4] if len(sys.argv) > 4 else "50"
log = f"{out}/client/aiperf.log"
while not (os.path.exists(log) and "Phase profiling (profiling) started" in open(log, errors="replace").read()):
    time.sleep(10)
time.sleep(delay)


def proc(pid):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        comm = open(f"/proc/{pid}/comm").read().strip()
        stat = open(f"/proc/{pid}/stat").read()
    except OSError:
        return None
    return cmd, comm, int(stat[stat.rfind(")") + 2 :].split()[1])


procs = {p: v for p in os.listdir("/proc") if p.isdigit() and (v := proc(p)) and v[0]}
names = {}
for pid, (cmd, comm, _) in procs.items():
    if (m := re.search(re.escape(out) + r"/(prefill|decode)(\d)/", cmd)) and "api_server" in cmd:
        names[pid] = f"frontend-{m.group(1)[0].upper()}{m.group(2)}"
    elif f"{out}/run_file.json" in cmd and "--compass-clock" in cmd:
        names[pid] = "ca"
    elif "run.mooncake_store(" in cmd:
        names[pid] = "mooncake_store"
    elif "timing_manager" in cmd:
        names[pid] = "traffic"
for pid, (cmd, comm, ppid) in procs.items():
    if comm.startswith("ATOM::EngineCor") and str(ppid) in names:
        names[pid] = names[str(ppid)].replace("frontend", "engine")
os.makedirs(f"{out}.spy", exist_ok=True)
print(f"sampling {len(names)} processes for {seconds}s at {rate} Hz: {sorted(names.values())}", flush=True)


def schedstat():
    out = {}
    for pid, name in names.items():
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                comm = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
                run_ns, wait_ns, _ = map(int, open(f"/proc/{pid}/task/{tid}/schedstat").read().split())
            except (OSError, ValueError):
                continue
            out[f"{name}.{pid}/{tid}"] = (comm, run_ns, wait_ns)
    return out


cpus = sorted(os.sched_getaffinity(int(next(p for p, n in names.items() if n == "ca"))))


def cpu_ticks():  # over the run's CPUs: all ticks, idle and iowait ticks
    total = idle = 0
    for line in open("/proc/stat"):
        f = line.split()
        if f[0].startswith("cpu") and f[0][3:].isdigit() and int(f[0][3:]) in cpus:
            total, idle = total + sum(map(int, f[1:])), idle + int(f[4]) + int(f[5])
    return total, idle


before, busy = schedstat(), []
runs = [
    subprocess.Popen(
        ["py-spy", "record", "--nonblocking", "--format", "raw", "-r", rate, "-d", str(seconds), "-p", pid,
         "-o", f"{out}.spy/{name}.{pid}.raw"],
        stdout=subprocess.DEVNULL, stderr=open(f"{out}.spy/{name}.{pid}.err", "w"),
    )
    for pid, name in names.items()
]
last, ncpu = cpu_ticks(), len(cpus)
while any(r.poll() is None for r in runs):
    time.sleep(1)
    now = cpu_ticks()
    busy.append(ncpu * (1 - (now[1] - last[1]) / max(now[0] - last[0], 1)))
    last = now
after = schedstat()
deltas = {k: (c, (r - before[k][1]) / 1e9, (w - before[k][2]) / 1e9) for k, (c, r, w) in after.items() if k in before}
json.dump(deltas, open(f"{out}.spy/schedstat.json", "w"), indent=0)
json.dump({"cpus": f"{cpus[0]}-{cpus[-1]}" if cpus == list(range(cpus[0], cpus[-1] + 1)) else cpus, "busy_per_s": busy}, open(f"{out}.spy/cpus.json", "w"))
print("py-spy exits:", [r.returncode for r in runs])
