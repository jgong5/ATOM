"""Compare a real serving run against a simulated one, on the engine's clock.

`benchmark_serving` times requests with `perf_counter` around an HTTP stream and
divides every metric by its own wall-clock duration, so against a Compass server
it reports how fast the *simulator* ran -- the forward is predicted rather than
performed, and the answers come back several times faster than the system they
stand for. Those numbers are not predictions and comparing them to a real run
measures nothing.

The engine stamps its own arrival, first-token and finish times on whichever
clock it is running: wall time normally, virtual time under Compass. Those are
the simulated latencies, and `GET /compass/requests` is what carries them out.
This script reads both sides and reports each, so the difference between the two
readings is visible rather than implied:

    scripts/compass/serve_probe.sh real    out/real
    scripts/compass/serve_probe.sh predict out/predict --compass ...
    python scripts/compass/serve_compare.py out/real out/predict
"""

import json
import statistics
import sys
from pathlib import Path


def _engine(run: Path) -> tuple[list[float], list[float], str]:
    """TTFTs and latencies the engine reported, in seconds."""
    path = run / "engine.json"
    if not path.exists():
        return [], [], "missing"
    blob = json.loads(path.read_text())
    rows = blob.get("requests", [])
    ttft = [r["ttft"] for r in rows if r.get("ttft") is not None]
    lat = [r["latency"] for r in rows if r.get("latency") is not None]
    return ttft, lat, blob.get("clock", "?")


def _client(run: Path) -> dict:
    path = run / "bench.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _pct(real: float, sim: float) -> str:
    if not real:
        return "     n/a"
    return f"{100.0 * (sim - real) / real:+7.1f}%"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    real_dir, sim_dir = Path(sys.argv[1]), Path(sys.argv[2])

    r_ttft, r_lat, r_clock = _engine(real_dir)
    s_ttft, s_lat, s_clock = _engine(sim_dir)
    rc, sc = _client(real_dir), _client(sim_dir)

    print("=" * 74)
    print(f"  real: {real_dir}  (engine clock: {r_clock}, {len(r_lat)} requests)")
    print(f"  sim : {sim_dir}  (engine clock: {s_clock}, {len(s_lat)} requests)")
    print("=" * 74)

    if not r_lat or not s_lat:
        print("  No engine-side timings. Is /compass/requests present on both?")
        return 1

    print("\n  ENGINE-SIDE -- what the engine measured, on its own clock.")
    print("  This is the comparison that means something.")
    print(f"    {'metric':<10} {'real':>10} {'simulated':>11} {'error':>9}")
    for name, real, sim in (("TTFT", r_ttft, s_ttft), ("latency", r_lat, s_lat)):
        if not real or not sim:
            continue
        a, b = statistics.fmean(real), statistics.fmean(sim)
        print(f"    {name:<10} {a*1000:>9.2f}ms {b*1000:>10.2f}ms {_pct(a, b)}")

    if rc and sc:
        print("\n  CLIENT-SIDE -- what benchmark_serving timed off the socket.")
        print("  Under a virtual clock this measures the simulator, not the")
        print("  simulation. Shown to make that difference visible, not to be")
        print("  read as accuracy.")
        print(f"    {'metric':<10} {'real':>10} {'simulated':>11} {'error':>9}")
        for name, key, scale in (("TTFT", "mean_ttft_ms", 1.0),
                                 ("TPOT", "mean_tpot_ms", 1.0),
                                 ("duration", "duration", 1000.0)):
            if key in rc and key in sc:
                a, b = rc[key] * scale, sc[key] * scale
                print(f"    {name:<10} {a:>9.2f}ms {b:>10.2f}ms {_pct(a, b)}")

    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
