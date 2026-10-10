#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The `module: compass-*` and `dev-process` label rules, and a CLI to apply them.

`labels_for(paths, title)` is the deterministic half of `SKILL.md`'s rules: what
a changed-file set and a PR title decide on their own. The judgement calls --
the purpose label for a compass PR that also touches ATOM code outside these
paths, and which labels an issue inherits -- stay prose in `SKILL.md`; they need
a reader, not a rule.

The module set is never hard-coded: `discover_modules` reads the current
`atom/compass/` tree, so a module added or removed there is picked up with no
edit here.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPASS_ROOT = REPO_ROOT / "atom" / "compass"

DEV_PROCESS = "dev-process"
MODULE_COLOR = "0E8A16"
DEV_PROCESS_COLOR = "BFD4F2"
DEV_PROCESS_DESCRIPTION = (
    "Development workflow: rules, review and landing process, gates, agent-team overlay"
)
FIXED_MODULE_LABELS = {
    "module: compass-script": "scripts/compass/ and its tests",
    "module: compass-tools": "tools/compass/ and its tests",
}
_TITLE_SCOPES = {"rules", "process"}


def module_label(name):
    return "module: compass-" + name.replace("_", "-")


def discover_modules(compass_root=COMPASS_ROOT):
    """Module directory names under `atom/compass/`: a subdirectory carrying a
    file other than `__init__.py`. Excludes dotfiles and `__pycache__`."""
    if not compass_root.is_dir():
        return []
    modules = []
    for entry in sorted(compass_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if any(f.is_file() and f.name != "__init__.py" for f in entry.iterdir()):
            modules.append(entry.name)
    return modules


def _flat_test_module(filename, modules):
    """The module a `tests/compass/<filename>` (no subdirectory) names, by the
    longest matching `test_<module>` / `test_<module>_...` prefix -- so
    `clock_transport` wins over `clock` for `test_clock_transport_wire.py`."""
    candidates = [
        m for m in modules
        if filename == "test_%s.py" % m or filename.startswith("test_%s_" % m)
    ]
    return max(candidates, key=len) if candidates else None


def _path_labels(path, modules):
    parts = path.split("/")
    if path.startswith("atom/compass/") and len(parts) > 3 and parts[2] in modules:
        return {module_label(parts[2])}
    if path.startswith("tests/compass/"):
        if len(parts) > 3 and parts[2] in modules:
            return {module_label(parts[2])}
        if len(parts) == 3:
            m = _flat_test_module(parts[2], modules)
            if m:
                return {module_label(m)}
        return set()
    if path.startswith("scripts/compass/"):
        return {"module: compass-script"}
    if path.startswith("tools/compass/"):
        return {"module: compass-tools"}
    return set()


def _is_dev_process_path(path):
    return path.split("/")[-1] in ("AI_DEV_RULES.md", "CLAUDE.md") or path.startswith(".claude/")


def _title_is_dev_process(title):
    title = title or ""
    m = re.match(r"compass\(([^)]*)\)", title)
    if m and {s.strip() for s in m.group(1).split(",")} & _TITLE_SCOPES:
        return True
    return title.startswith("Process:") or title.startswith("agent-team")


def labels_for(paths, title, compass_root=COMPASS_ROOT):
    """The labels the path and title rules give for a changed-file set and a
    PR/issue title. Gate-script changes (`scripts/compass/**`) only ever reach
    `module: compass-script` here, never `dev-process`."""
    modules = discover_modules(compass_root)
    labels = set()
    for path in paths:
        labels |= _path_labels(path, modules)
        if _is_dev_process_path(path):
            labels.add(DEV_PROCESS)
    if _title_is_dev_process(title):
        labels.add(DEV_PROCESS)
    return labels


# --- CLI -------------------------------------------------------------------

def _gh_json(args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def _pr_files_and_title(repo, number):
    title = _gh_json(
        ["pr", "view", str(number), "-R", repo, "--json", "title", "--jq", ".title"]
    ).strip()
    files = _gh_json(
        ["api", "--paginate", "repos/%s/pulls/%s/files" % (repo, number), "--jq", ".[].filename"]
    ).split("\n")
    return [f for f in files if f], title


def _existing_labels(repo, number):
    out = _gh_json(["api", "repos/%s/issues/%s" % (repo, number), "--jq", ".labels[].name"])
    return {line for line in out.splitlines() if line}


def _all_labels(modules):
    labels = {module_label(m): "atom/compass/%s/ and its tests" % m for m in modules}
    labels.update(FIXED_MODULE_LABELS)
    labels[DEV_PROCESS] = DEV_PROCESS_DESCRIPTION
    return labels


def cmd_pr(repo, number, apply_):
    files, title = _pr_files_and_title(repo, number)
    labels = sorted(labels_for(files, title))
    for label in labels:
        print(label)
    if apply_ and labels:
        args = ["api", "repos/%s/issues/%s/labels" % (repo, number)]
        for label in labels:
            args += ["-f", "labels[]=%s" % label]
        subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return 0


def cmd_sync(repo):
    modules = discover_modules()
    wanted = _all_labels(modules)
    have = {
        line for line in _gh_json(
            ["api", "--paginate", "repos/%s/labels" % repo, "--jq", ".[].name"]
        ).splitlines() if line
    }
    created = []
    for name, description in wanted.items():
        if name in have:
            continue
        color = DEV_PROCESS_COLOR if name == DEV_PROCESS else MODULE_COLOR
        subprocess.run(
            ["gh", "api", "repos/%s/labels" % repo,
             "-f", "name=%s" % name, "-f", "color=%s" % color,
             "-f", "description=%s" % description],
            capture_output=True, text=True, check=True,
        )
        created.append(name)
    for name in created:
        print(name)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="compass_labels.py")
    parser.add_argument("--repo", default="jgong5/ATOM")
    sub = parser.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("pr", help="print the labels a PR's files and title give")
    pr.add_argument("number", type=int)
    pr.add_argument("--apply", action="store_true", help="add the missing labels to the PR")

    sub.add_parser("sync", help="create any module label the tree implies that the repo lacks")

    args = parser.parse_args(argv)
    if args.command == "pr":
        return cmd_pr(args.repo, args.number, args.apply)
    if args.command == "sync":
        return cmd_sync(args.repo)
    return 2


if __name__ == "__main__":
    sys.exit(main())
