#!/usr/bin/env python3
"""Form checks for GitHub prose. Usage: lint_gh_prose.py --kind KIND FILE. Exit 1 on hits."""

import argparse
import re
import sys

BUDGET = {
    "issue": 150,
    "brief": 250,
    "pr": 300,
    "ruling": 150,
    "review": 200,
    "comment": 80,
}
STATE = re.compile(
    r"\b(Closes|Fixes|Part of|Blocked|Needs owner ruling|Ready|No blocking|APPROVE|REQUEST CHANGES|Round \d+ pushed)\b"
)
PATTERNS = {
    "private ref (link it or say it)": r"§|\b[DPTW]\d+(\.\d+)?\b|[Pp]rinciple \d|\bthe brief\b|\b[Dd]oc \d+",
    "line number (cite path + symbol)": r"\.(py|sh|json|md|ya?ml|toml|cpp|h)`?:\d+|`:\d+`",
    "process narration": r"(?i)md5|docker cp|git archive|__file__|tarball",
    "round history in body": r"(?i)\bround \d",
    "non-English text": r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]",
    "em dash": r"—",
    "emoji": r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
}


def visible(text):
    """Text outside <details> and fenced code: where budgets apply."""
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.S)
    return re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)


def lint(text, kind):
    out = visible(text)
    lines = out.splitlines()
    hits = []
    words = len(out.split())
    if words > BUDGET[kind]:
        hits.append(f"{words} words outside <details>/code, budget {BUDGET[kind]}")
    bold = re.sub(r"`[^`]*`", "", out).count("**") // 2
    if bold > 3:
        hits.append(f"{bold} bold spans, max 3")
    first = next((l for l in lines if l.strip()), "")
    if kind not in ("issue", "brief") and not STATE.search(first):
        hits.append(f"first line states no state or ask: {first[:60]!r}")
    for n, line in enumerate(lines, 1):
        line = re.sub(r"\[[^\]]*\]\([^)]*\)", "", line)  # a linked ref is fine
        for name, pat in PATTERNS.items():
            if name == "round history in body" and kind in ("comment", "review"):
                continue
            m = re.search(pat, line)
            if m:
                hits.append(f"line {n}: {name}: {m.group(0)!r}")
    for block in re.findall(r"(?m)(?:^[ \t]*(?:[-*] |\d+\. |\|).*\n?)+", out):
        run = sum(not re.match(r"\s*\|[-| :]+\|\s*$", l) for l in block.splitlines())
        if run > 6:  # ponytail: 5 items + a table header, lists get the same slack
            hits.append(f"list/table of {run} lines, max 5 items")
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=BUDGET, required=True)
    ap.add_argument("file")
    a = ap.parse_args()
    hits = lint(open(a.file, encoding="utf-8").read(), a.kind)
    print("\n".join(hits) or "clean")
    sys.exit(1 if hits else 0)
