# SPDX-License-Identifier: MIT
"""Sum the --junitxml files of one gate run per process and per test file.

    python3 spike/junit_per_file.py [--top N] <junit.xml>...

Each file is one pytest process; its name after ".xml" (".shard3", ".rest1")
names the process, and a bare ".xml" is the main one. Test cases are read with
`-o junit_family=xunit1`, whose `file` attribute names the test file. A
process's suite time is its pytest session, collection included; what its test
cases do not account for is printed as collection (plus per-test overhead).
"""

import argparse
import collections
import datetime
import re
import xml.etree.ElementTree as ET


def outcome(case):
    for kind in ("failure", "error", "skipped"):
        if case.find(kind) is not None:
            return kind
    return "passed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="file rows to print; 0 prints all")
    ap.add_argument("xml", nargs="+")
    a = ap.parse_args()

    files = collections.defaultdict(lambda: [0.0, 0, ""])  # file -> [seconds, cases, process]
    fails = []
    rows = []
    starts, ends = [], []
    for path in a.xml:
        proc = re.sub(r".*\.xml\.?", "", path) or "main"
        suite = ET.parse(path).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        count = collections.Counter()
        run = 0.0
        for case in suite.iter("testcase"):
            t = float(case.get("time", 0))
            f = case.get("file") or case.get("classname", "?")
            run += t
            files[f][0] += t
            files[f][1] += 1
            files[f][2] = proc
            o = outcome(case)
            count[o] += 1
            if o in ("failure", "error"):
                msg = (case.find(o).get("message") or "").splitlines()
                fails.append((proc, o, f"{f}::{case.get('name')}", msg[0][:120] if msg else ""))
        wall = float(suite.get("time", 0))
        start = datetime.datetime.fromisoformat(suite.get("timestamp"))
        starts.append(start)
        ends.append(start + datetime.timedelta(seconds=wall))
        rows.append((proc, count, wall, run))

    print(f"{'process':<9}{'passed':>7}{'failed':>7}{'error':>6}{'skip':>6}"
          f"{'suite_s':>9}{'tests_s':>9}{'collect_s':>10}{'collect%':>9}")
    for proc, c, wall, run in sorted(rows, key=lambda r: -r[2]):
        col = wall - run
        print(f"{proc:<9}{c['passed']:>7}{c['failure']:>7}{c['error']:>6}{c['skipped']:>6}"
              f"{wall:>9.1f}{run:>9.1f}{col:>10.1f}{100 * col / wall if wall else 0:>8.1f}%")
    walls = [r[2] for r in rows]
    print(f"span {(max(ends) - min(starts)).total_seconds():.1f} s from the first process's start; "
          f"slowest {max(walls):.1f} s = {max(walls) / (sum(walls) / len(walls)):.2f}x the mean of {len(walls)}")

    ranked = sorted(files.items(), key=lambda kv: -kv[1][0])
    print(f"\n{len(files)} files, {sum(1 for _, v in ranked if v[0] > 180)} over 180 s")
    print(f"{'seconds':>8}{'cases':>7}  {'process':<9}file")
    for f, (t, n, proc) in ranked[: a.top or None]:
        print(f"{t:>8.1f}{n:>7}  {proc:<9}{f}")

    if fails:
        print(f"\n{len(fails)} failed or errored")
        for proc, o, nodeid, msg in sorted(fails):
            print(f"{proc:<9}{o:<8}{nodeid}  {msg}")


if __name__ == "__main__":
    main()
