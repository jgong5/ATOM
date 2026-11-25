#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The test files a diff reaches, from a map of the functions each test file ran.

  gate_select.py select MAP ROOT [IGNORED...]
      `select: <n> of <m>, map <sha>`, then each test file under ROOT/tests to
      deselect; or `select: whole gate -- <why>` alone
  gate_select.py stamp MAP ROOT   the .compass-select stamp of ROOT's HEAD
  gate_select.py build WORK ROOT SHA OUT   the map of gate_map.sh's run

A test file is kept when it ran a function whose body the diff since the map's
commit changes; when it read a changed file's source, or listed a directory an
added file is in; when the diff changes it or a file under tests/ it imports; or
when the map does not hold it. Any other change runs the whole gate.

Under gate_map.sh this module also starts coverage in every Python process
(startup) and is the gate's pytest plugin (-p gate_select): a test file's
collection and its tests run under that file's coverage context, and a process
another one starts takes the context of its nearest ancestor that has one. An
audit hook records, under that context, each .py file of the tree a process
opens and each directory of it a process lists, other than the import system's.
"""

import ast
import bisect
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

DESIGN = "atom/compass/design/"
GATE = "scripts/compass/gate_select.py"
HARNESS = "compass_harness/"
IMPORT = ("<frozen importlib", "<frozen zipimport")
EMPTY = {"code": [], "names": {}, "bodies": {}}
SKIP_DIRS = {"__pycache__", "target", "build", "dist", "node_modules", "venv"}
DIFF = ["-c", "core.quotePath=false", "diff", "--no-renames", "--name-only", "-z"]
_work = _root = _ctx = _fd = None
_seen = set()


def _h(parts):
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]


def _dump(nodes):
    return [ast.dump(n) for n in nodes if n is not None]


def _code(body):
    """`body` less its docstring, which is not code."""
    doc = (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    )
    return body[1:] if doc else body


def _bound(node):
    """The names a module-level statement binds, or None when it does more than bind."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    targets = getattr(node, "targets", [getattr(node, "target", None)])
    names = [n for t in targets for n in getattr(t, "elts", [t])]
    return (
        [n.id for n in names] if all(isinstance(n, ast.Name) for n in names) else None
    )


def _scan(body, prefix, s, spans, code):
    for node in _code(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            q = prefix + node.name
            head = ["class" if isinstance(node, ast.ClassDef) else "def"]
            head += _dump(node.decorator_list) + _dump(getattr(node, "type_params", []))
            if isinstance(node, ast.ClassDef):
                inner = []
                _scan(node.body, q + ".", s, spans, inner)
                head += _dump(node.bases) + _dump(node.keywords) + [h for h, _ in inner]
            else:
                body_h = _h(_dump([node.args, node.returns, *_code(node.body)]))
                s["bodies"].setdefault(q, []).append(body_h)
                spans.append((node.body[0].lineno, node.end_lineno, q))
            s["names"].setdefault(q, []).append(_h(head))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:  # one entry per name, so adding a name adds an entry
                key = [getattr(node, "module", None), getattr(node, "level", 0)]
                bound = (
                    None if a.name == "*" else [a.asname or a.name.partition(".")[0]]
                )
                code.append([_h([*key, a.name, a.asname]), bound])
        elif not (
            isinstance(node, ast.If)
            and "TYPE_CHECKING" in ast.dump(node.test)
            and not node.orelse
        ):
            code.append([_h(ast.dump(node)), _bound(node)])


def structure(src):
    """A module's shape and its functions' body spans; (None, []) when it does not parse.

    `code` holds each module-level statement other than a definition, with the
    names it binds when binding is all it does; `names` each definition's hash
    outside function bodies; `bodies` each function body's hash."""
    try:
        mod = ast.parse(src)
    except (SyntaxError, ValueError):
        return None, []
    s, spans = {"code": [], "names": {}, "bodies": {}}, []
    _scan(mod.body, "", s, spans, s["code"])
    return s, sorted(spans)


def _compare(p, old, new, funcs):
    """Why `old` to `new` needs the whole gate, else None, adding to `funcs`."""
    was = collections.Counter(h for h, _ in old["code"])
    now = collections.Counter(h for h, _ in new["code"])
    if was - now:
        return f"{p} changes or removes module-level code"
    bound = {n for _, names in old["code"] for n in names or ()} | set(old["names"])
    binds = dict(map(tuple, new["code"]))
    for h in now - was:
        if binds[h] is None or bound.intersection(binds[h]):
            return f"{p} adds module-level code other than a new name's definition"
    for q, h in old["names"].items():
        if q not in new["names"]:
            return f"{p} removes {q}"
        if new["names"][q] != h:
            return f"{p} changes {q} outside a function body"
    for q in new["names"].keys() - old["names"].keys():
        if q in bound:
            return f"{p} rebinds {q}"
        if q.rpartition(".")[0] in old["names"]:
            # Nothing in the map says which files build the class's instances.
            return f"{p} adds {q} to a class"
    funcs.update(f"{p}::{q}" for q, h in old["bodies"].items() if new["bodies"][q] != h)
    return None


def _is_test(p):
    name = os.path.basename(p)
    test = (
        name.startswith("test_") and name.endswith(".py") or name.endswith("_test.py")
    )
    return p.startswith("tests/") and test


def _files(root, top, keep):
    for d, dirs, names in os.walk(os.path.join(root, top)):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and x[0] != ".")
        for name in sorted(names):
            p = os.path.relpath(os.path.join(d, name), root)
            if keep(p):
                yield p


def _importers(root, p):
    """The files under tests/ that import the module `p` by its name."""
    stem = re.escape(os.path.basename(p)[:-3])
    pat = re.compile(rf"^\s*(from|import)\s+([\w.]+\.)?{stem}\b", re.MULTILINE)
    found = set()
    for f in _files(root, "tests", lambda f: f.endswith(".py") and f != p):
        with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
            if pat.search(fh.read()):
                found.add(f)
    return found


def _git(root, *args):
    cmd = ["git", "-C", root, *args]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _since(root, sha, *rev):
    """(is sha an ancestor of HEAD, the paths changed from sha to `rev` or the work tree)."""
    if _git(root, "merge-base", "--is-ancestor", sha, "HEAD").returncode:
        return False, []
    out = _git(root, *DIFF, sha, *rev)
    return out.returncode == 0, [p for p in out.stdout.split("\0") if p]


def _changed(root, sha):
    """(why the whole gate runs, the paths changed since the map's sha)."""
    if _git(root, "rev-parse", "--git-dir").returncode == 0:
        ok, paths = _since(root, sha)
    else:
        head = rest = commit = ""
        try:
            with open(os.path.join(root, ".compass-select"), encoding="utf-8") as f:
                head, _, rest = f.read().partition("\n")
            with open(os.path.join(root, ".compass-commit"), encoding="utf-8") as f:
                commit = f.read().strip()
        except OSError:
            pass
        if not head:
            return (
                "the snapshot has no .compass-select stamp; stage it with the map",
                [],
            )
        stamped, at, ok = head.split()
        if [stamped, at] != [sha, commit]:
            return f"the .compass-select stamp is of map {stamped[:9]} at {at[:9]}", []
        ok, paths = ok == "1", [p for p in rest.split("\0") if p]
    return (None if ok else f"the map's {sha[:9]} is not an ancestor of HEAD"), paths


def _keep(m, root):
    """(why the whole gate runs, else None; the test files the diff reaches)."""
    why, changed = _changed(root, m["sha"])
    if why:
        return why, None
    if "reads" not in m:
        return f"map {m['sha'][:9]} records no source reads; build it again", None
    funcs, own, read = set(), set(), set()
    for p in changed:
        if p == GATE or os.path.basename(p) == "conftest.py":
            return f"{p} is a conftest.py or the gate's own", None
        if p.startswith(HARNESS):
            return f"{p}: the gate runs an installed copy of {HARNESS}", None
        if not p.endswith(".py"):
            if p.startswith(DESIGN):
                continue
            return f"{p} is not Python", None
        try:
            with open(os.path.join(root, p), "rb") as f:
                new = structure(f.read())[0]
        except FileNotFoundError:
            return f"{p} is removed", None
        old = m["tree"].get(p, EMPTY)
        if new is None or old is None:
            return f"{p} does not parse", None
        why = _compare(p, old, new, funcs)
        added = p not in m["tree"]
        for t, rs in m["reads"].items():
            if p in rs or added and any(p.startswith(r) for r in rs if r[-3:] != ".py"):
                read |= {t} if _is_test(t) else _importers(root, t)
        mine = {p} if _is_test(p) else set()
        own |= mine
        if why and p.startswith("tests/"):
            # A file under tests/ runs at the import of the test files that import it.
            users = _importers(root, p) | mine
            why = None if users and all(map(_is_test, users)) else why
            own |= users
        if why:
            return why, None
    hit = {i for i, f in enumerate(m["funcs"]) if f in funcs}
    return None, own | read | {
        t for t, ids in m["ran"].items() if hit.intersection(ids)
    }


def select(path, root, *ignored):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    out = tuple(i.rstrip("/") + "/" for i in ["tests/plugin", *ignored])
    tests = [t for t in _files(root, "tests", _is_test) if not t.startswith(out)]
    tests = [t for t in tests if t not in ignored]
    why, keep = _keep(m, root)
    if why:
        print(f"select: whole gate -- {why}")
        return
    keep = {t for t in tests if t in keep or t not in m["ran"]}
    print(f"select: {len(keep)} of {len(tests)}, map {m['sha'][:9]}")
    print("".join(f"{t}\n" for t in tests if t not in keep), end="")


def stamp(path, root):
    with open(path, encoding="utf-8") as f:
        sha = json.load(f)["sha"]
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    ok, paths = _since(root, sha, "HEAD")
    sys.stdout.write(f"{sha} {head} {int(ok)}\n" + "\0".join(paths))


def build(work, root, sha, out):
    import sqlite3

    from coverage.numbits import numbits_to_nums

    tree, spans, starts = {}, {}, {}
    for p in _files(root, "", lambda p: p.endswith(".py")):
        with open(os.path.join(root, p), "rb") as f:
            tree[p], spans[p] = structure(f.read())
        starts[p] = [s[0] for s in spans[p]]
    ids, ran, bad = {}, collections.defaultdict(set), 0
    query = (
        "select file.path, context.context, numbits from line_bits"
        " join file on file.id = file_id join context on context.id = context_id"
    )
    for name in sorted(os.listdir(os.path.join(work, "cov"))):
        try:
            con = sqlite3.connect(os.path.join(work, "cov", name))
            rows = con.execute(query).fetchall()
            con.close()
        except sqlite3.Error:
            bad += 1
            continue
        for path, ctx, bits in rows:
            p, got = os.path.relpath(path, root), ran[ctx] if ctx else set()
            for n in numbits_to_nums(bits) if p in spans else ():
                k = bisect.bisect_right(starts[p], n) - 1
                if k >= 0 and n <= spans[p][k][1]:
                    got.add(ids.setdefault(f"{p}::{spans[p][k][2]}", len(ids)))
    ran = {t: sorted(v) for t, v in sorted(ran.items())}
    reads = collections.defaultdict(set)
    for name in os.listdir(os.path.join(work, "reads")):
        with open(os.path.join(work, "reads", name), encoding="utf-8") as f:
            for line in f:
                ctx, _, r = line.rstrip("\n").partition("\t")
                reads[ctx].add(r)
    reads = {t: sorted(v) for t, v in sorted(reads.items())}
    with open(out, "w", encoding="utf-8") as f:
        m = {"sha": sha, "funcs": list(ids), "ran": ran, "tree": tree, "reads": reads}
        json.dump(m, f)
    print(f"map:    {out}: {len(ran)} test files, {len(ids)} functions")
    print(f"        {bad} coverage data files unreadable")


def _audit(event, args):
    """Record a source read: a .py file of the tree opened, or a directory of it
    listed, under the test file and any helper under tests/ whose import made it.
    The import system's reads, and coverage's, are their own."""
    global _fd
    if event not in ("open", "os.listdir", "os.scandir") or not _ctx or not args:
        return
    up = sys._getframe().f_back  # `os` is frozen too: skip only the import system
    if isinstance(args[0], int) or not up or up.f_code.co_filename.startswith(IMPORT):
        return
    r = os.path.relpath(os.path.abspath(os.fsdecode(args[0] or ".")), _root)
    if r[:2] == ".." or event == "open" and r[-3:] != ".py":
        return
    who = {_ctx}
    while up:
        code, up = up.f_code, up.f_back
        if f"{os.sep}coverage{os.sep}" in code.co_filename:
            return
        if code.co_name == "<module>" and code.co_filename.endswith(".py"):
            h = os.path.relpath(code.co_filename, _root)
            who |= {h} if h[:6] == "tests/" else set()
    r = r if event == "open" else "" if r == "." else r + "/"
    for ctx in who:
        if (ctx, r) not in _seen:
            _seen.add((ctx, r))
            if _fd is None:
                flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
                _fd = os.open(f"{_work}/reads/{os.getpid()}", flags, 0o644)
            os.write(_fd, f"{ctx}\t{r}\n".encode())


def startup(work, root):
    """Start coverage here, under the context of the nearest ancestor that has one."""
    global _work, _root, _ctx
    _work, _root, ctx, pid = work, root, "", os.getppid()
    while pid > 1:
        try:
            with open(f"{work}/ctx/{pid}", encoding="utf-8") as f:
                ctx = f.read()
            break
        except OSError:
            pass
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                pid = int(f.read().rpartition(")")[2].split()[1])
        except (OSError, IndexError, ValueError):
            break
    os.environ["COVERAGE_PROCESS_START"] = f"{work}/rc"
    os.environ["COMPASS_MAP_CONTEXT"] = _ctx = ctx
    sys.addaudithook(_audit)
    import coverage

    coverage.process_startup()
    # A fork restarts coverage under the parent's static context: restore the file's.
    os.register_at_fork(
        after_in_child=lambda: _ctx and coverage.Coverage.current().switch_context(_ctx)
    )


def _switch(path):
    global _ctx
    ctx = os.path.relpath(path, _root) if str(path).endswith(".py") else ""
    if ctx != _ctx:
        import coverage

        _ctx = ctx
        coverage.Coverage.current().switch_context(ctx)
        tmp = f"{_work}/ctx/.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(ctx)
        os.replace(tmp, f"{_work}/ctx/{os.getpid()}")


def pytest_collectstart(collector):
    _switch(collector.path)


def pytest_runtest_protocol(item):
    _switch(item.path)


if __name__ == "__main__":
    {"select": select, "stamp": stamp, "build": build}[sys.argv[1]](*sys.argv[2:])
