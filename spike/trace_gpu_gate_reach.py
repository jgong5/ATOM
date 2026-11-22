# SPDX-License-Identifier: MIT
"""Which tests in gate_gpu.sh's batch scope reach the device-side functions of
EPLB, CUDA-graph capture bounds and block tables, by import chain and call.

Static: every file is parsed, nothing is imported. Run from the tree's root:

    python3 spike/trace_gpu_gate_reach.py [--precise]

The batch scope is tests/ outside tests/plugin and tests/compass, plus the
files scripts/compass/gpu_gate_driver_run.txt names. A test file's roots are
its test functions, its module body and the module bodies of the conftests
above it; a test's fixtures are reached by name, as any other def.

Reach is a fixpoint over references, not only calls, so a function passed as
a callback is reached where it is named:
- a bare name resolves to the referring module's own top-level or nested
  function and class defs, and through its `from X import name` lines,
  following re-exports;
- `alias.name` resolves when `alias` names an imported module;
- calling a class reaches its __init__, __post_init__ and __call__, and
  an nn.Module's forward (the module is called, not its forward);
- reaching a decorated def reaches its decorators, so a wrapper they return
  is reached with the def it replaces;
- a method resolves by its receiver's type where the code shows it:
  `self.name` in a method of C to `name` on C or a reached subclass,
  `super().name` to the nearest base def, `x.name` and `self.f.name` to
  `name` on C when `x` or `self.f` is bound to `C(...)`;
- without `--precise`, any other `obj.name` but a dunder resolves to `name`,
  own or inherited, on every class a reached def names (rapid type analysis;
  an import alone names nothing).

So `--precise` prints chains that are call paths, branch conditions aside,
and misses calls on receivers of unknown type; the default prints every
candidate, a method name shared across reached classes over-reaching. A
"none" in the default misses only where dispatch hides the name (getattr, a
registry, a string such as `call_func("capture_cudagraph")`) or a process
runs the code (an engine a CLI starts in a subprocess).

A test runs on a device when its function, its decorators, its classes'
decorators or its module body name CUDA (a "cuda" string, `.cuda(`,
`torch.cuda.<device API>`, or an availability check, true on a GPU box), or
when scripts/compass/cpu_gate_exclude.txt lists its file. A line that patches
`torch.cuda` does not count.
"""

import ast
import os
import re
import sys
from collections import defaultdict, deque

ROOT = os.getcwd()

TARGETS = {
    "EPLB": [
        ("atom.model_ops.eplb", n)
        for n in (
            "count_physical_load",
            "physical_load_to_logical_load",
            "ExpertLocationMetadata.from_trivial",
            "ExpertLocationMetadata.update",
            "ExpertLoadMonitor.initialize",
            "ExpertLoadMonitor.on_forward_start",
            "ExpertLoadMonitor.record",
            "ExpertLoadMonitor.on_forward_end",
            "ExpertLoadMonitor.dump_global_physical_load",
            "_execute_batched_p2p_ops",
            "_migrate_single_layer",
            "move_from_buffer",
            "migrate_and_commit_chunk",
            "EPLBManager._maybe_initialize_runtime",
            "EPLBManager._bind_layer_expert_maps",
            "EPLBManager.fill_redundant",
            "EPLBManager.on_forward_pass_end",
            "EPLBManager._execute_runtime_rebalance",
            "initialize_eplb_runtime",
            "with_eplb_forward_monitor.wrapper",
            "eplb_map_logical_to_physical",
            "record_eplb_expert_load",
            "eplb_map_and_record_fused",
            "eplb_append_shared_map_and_record_fused",
            "_eplb_route",
            "_eplb_route_kernel",
        )
    ],
    "CUDA-graph capture bounds": [
        ("atom.model_engine.model_runner", n)
        for n in (
            "ModelRunner.capture_cudagraph",
            "ModelRunner._maybe_calibrate_dspark_sps",
            "ModelRunner.run_model",
            "tokenIDProcessor._stage_input_ids",
        )
    ],
    "block tables": [("atom.utils.block_tables", "BlockTableState.publish")],
}

DEVICE_TEXT = re.compile(
    r"""["']cuda(:\d+)?["']|\.cuda\(|torch\.cuda\.(current_device|set_device|"""
    r"""Stream|stream|graph|CUDAGraph|synchronize|Event|is_available|device_count)\b"""
)
PATCHED_CUDA = re.compile(r"setattr\(\s*(torch\.cuda|[\"']torch\.cuda)")


def module_name(path):
    rel = os.path.relpath(path, ROOT)[:-3].split(os.sep)
    if rel[-1] == "__init__":
        rel = rel[:-1]
    return ".".join(rel)


class Def:
    def __init__(self, module, qualname, node):
        self.module, self.qualname, self.node = module, qualname, node
        self.key = f"{module.name}:{qualname}"
        self.names, self.attrs = set(), set()
        self.attr_on = set()  # (alias, attr); alias may name a module
        self.called = set()  # bare names and `alias.attr` this def calls
        self.self_attrs = set()  # attributes it reads on self or cls
        self.super_attrs = set()  # attributes it reads on super()
        self.owner = None  # the class def a method belongs to
        # A name bound to a call's result -> the callee refs (`C(...)` types it).
        self.var_types = defaultdict(set)
        self.field_attrs = set()  # (field, attr) read as self.field.attr
        self.field_types = defaultdict(set)  # on a class: self.field = C(...)


def call_ref(node):
    """The callee of a call as a bare name or an (alias, attr) pair."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return (f.value.id, f.attr)
    return None


def self_field(node):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("self", "cls")
    )


class Module:
    def __init__(self, name, path, source):
        self.name, self.path, self.source = name, path, source
        self.is_pkg = path.endswith("__init__.py")
        self.imports = set()  # modules imported anywhere in the file
        self.from_names = {}  # local name -> (module, original name)
        self.mod_alias = {}  # local name -> module
        self.defs = {}  # qualname -> Def
        self.classes = {}  # class qualname -> [method names]
        self.by_name = defaultdict(list)  # bare-name-visible defs
        self.tree = ast.parse(source, path)
        self.body = self.defs["<module>"] = Def(self, "<module>", None)
        self.walk_body(self.tree.body, self.body, [])

    def resolve_from(self, node):
        if node.level == 0:
            return node.module or ""
        base = self.name.split(".")
        if not self.is_pkg:
            base = base[:-1]
        base = base[: len(base) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def walk_body(self, stmts, scope, stack):
        for st in stmts:
            self.walk(st, scope, stack)

    def walk(self, node, scope, stack):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qual = ".".join(stack + [node.name])
            d = self.defs[qual] = Def(self, qual, node)
            parent = ".".join(stack)
            if parent in self.classes:
                d.owner = self.defs[parent]
            else:
                self.by_name[node.name].append(d)
            for dec in node.decorator_list:
                self.walk(dec, d, stack)
            if isinstance(node, ast.ClassDef):
                self.classes[qual] = [
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                for b in node.bases + [k.value for k in node.keywords]:
                    self.walk(b, scope, stack)
            else:
                for a in node.args.defaults + node.args.kw_defaults:
                    if a is not None:
                        self.walk(a, scope, stack)
            self.walk_body(node.body, d, stack + [node.name])
            return
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                for i in range(1, len(parts) + 1):
                    self.imports.add(".".join(parts[:i]))
                self.mod_alias[a.asname or parts[0]] = a.name if a.asname else parts[0]
            return
        if isinstance(node, ast.ImportFrom):
            src = self.resolve_from(node)
            parts = src.split(".")
            for i in range(1, len(parts) + 1):
                self.imports.add(".".join(parts[:i]))
            for a in node.names:
                if a.name == "*":
                    continue
                self.from_names[a.asname or a.name] = (src, a.name)
                self.imports.add(f"{src}.{a.name}")  # when it is a submodule
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.withitem)):
            value = node.context_expr if isinstance(node, ast.withitem) else node.value
            targets = (
                [node.optional_vars]
                if isinstance(node, ast.withitem)
                else getattr(node, "targets", [getattr(node, "target", None)])
            )
            ref = call_ref(value)
            for t in targets if ref else ():
                if isinstance(t, ast.Name):
                    scope.var_types[t.id].add(ref)
                elif self_field(t) and scope.owner is not None:
                    scope.owner.field_types[t.attr].add(ref)
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and self_field(node.value)
        ):
            scope.field_attrs.add((node.value.attr, node.attr))
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                scope.called.add(f.id)
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                scope.called.add(f"{f.value.id}.{f.attr}")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            scope.names.add(node.id)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", "cls")
            and scope.owner is not None
        ):
            scope.self_attrs.add(node.attr)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "super"
            and scope.owner is not None
        ):
            scope.super_attrs.add(node.attr)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            scope.attrs.add(node.attr)
            if isinstance(node.value, ast.Name):
                scope.attr_on.add((node.value.id, node.attr))
        for child in ast.iter_child_nodes(node):
            self.walk(child, scope, stack)


def load_tree():
    mods = {}
    for top in ("atom", "tests"):
        for dirpath, dirnames, files in os.walk(os.path.join(ROOT, top)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf-8") as fh:
                    source = fh.read()
                try:
                    m = Module(module_name(path), path, source)
                except SyntaxError:
                    continue
                mods[m.name] = m
    return mods


def read_list(path):
    with open(os.path.join(ROOT, path)) as fh:
        lines = (line.strip() for line in fh)
        return [line for line in lines if line.startswith("tests/")]


def batch_scope():
    files = []
    for dirpath, dirnames, names in os.walk(os.path.join(ROOT, "tests")):
        rel = os.path.relpath(dirpath, ROOT)
        if rel in ("tests/plugin", "tests/compass"):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(dirnames)
        for f in sorted(names):
            if f.endswith(".py") and (f.startswith("test_") or f.endswith("_test.py")):
                files.append(os.path.join(rel, f))
    driver_run = read_list("scripts/compass/gpu_gate_driver_run.txt")
    return files + [f for f in driver_run if os.path.exists(f)]


class Graph:
    def __init__(self, mods):
        self.mods = mods
        self._static, self._ancestors, self._methods = {}, {}, {}

    def resolve_name(self, mod, name, depth=0):
        """Defs a bare `name` in `mod` stands for."""
        out = list(mod.by_name.get(name, ()))
        if name in mod.from_names and depth < 8:
            src, orig = mod.from_names[name]
            if src in self.mods:
                out += self.resolve_name(self.mods[src], orig, depth + 1)
            if f"{src}.{orig}" in self.mods:
                out.append(self.mods[f"{src}.{orig}"].body)
        return out

    def resolve_ref(self, mod, alias, attr):
        """Defs `alias.attr` stands for when `alias` names a module."""
        target = mod.mod_alias.get(alias)
        if target is None and alias in mod.from_names:
            target = ".".join(mod.from_names[alias])
        if target in self.mods:
            return self.resolve_name(self.mods[target], attr)
        return []

    def ancestors(self, cls):
        """The class and every in-tree base class, nearest first."""
        if cls.key not in self._ancestors:
            self._ancestors[cls.key] = [cls]
            out = [cls]
            for b in cls.node.bases:
                if isinstance(b, ast.Name):
                    found = self.resolve_name(cls.module, b.id)
                elif isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name):
                    found = self.resolve_ref(cls.module, b.value.id, b.attr)
                else:
                    found = []
                for base in found:
                    if isinstance(base.node, ast.ClassDef):
                        out += [a for a in self.ancestors(base) if a not in out]
            self._ancestors[cls.key] = out
        return self._ancestors[cls.key]

    def classes_of(self, mod, ref):
        """Class defs a callee ref in `mod` names; a call to one types its result."""
        found = (
            self.resolve_name(mod, ref)
            if isinstance(ref, str)
            else self.resolve_ref(mod, *ref)
        )
        return [c for c in found if isinstance(c.node, ast.ClassDef)]

    def is_nn_module(self, cls):
        """Whether some base outside the tree is named `Module` (torch.nn's)."""
        return any(
            ast.unparse(b).split(".")[-1] == "Module"
            for a in self.ancestors(cls)
            for b in a.node.bases
        )

    def methods_of(self, cls):
        """Method name -> the def an instance of `cls` runs, in tree."""
        if cls.key not in self._methods:
            table = {}
            for a in self.ancestors(cls):
                for name in a.module.classes[a.qualname]:
                    table.setdefault(name, a.module.defs[f"{a.qualname}.{name}"])
            self._methods[cls.key] = table
        return self._methods[cls.key]

    def static_edges(self, d):
        """(def, called) pairs `d` names directly: bare names, module attributes."""
        if d.key not in self._static:
            mod, out = d.module, []
            for name in sorted(d.names):
                called = name in d.called
                out += [(t, called) for t in self.resolve_name(mod, name)]
            for alias, attr in sorted(d.attr_on):
                called = f"{alias}.{attr}" in d.called
                out += [(t, called) for t in self.resolve_ref(mod, alias, attr)]
            self._static[d.key] = out
        return self._static[d.key]


def conftests(mods, test_path):
    parts = test_path.split("/")[:-1]
    names = (".".join(parts[:i] + ["conftest"]) for i in range(1, len(parts) + 1))
    return [mods[n] for n in names if n in mods]


def device_reason(mod, d, driver):
    """Why the test `d` runs on a device, or None."""
    if os.path.relpath(mod.path, ROOT) in driver:
        return "its file is in cpu_gate_exclude.txt"
    lines = mod.source.splitlines()
    # The test with its decorators, and the decorators of its classes.
    owners = [d.node] + [
        mod.defs[q].node
        for q in (
            d.qualname.rsplit(".", i)[0] for i in range(1, d.qualname.count(".") + 1)
        )
        if q in mod.classes
    ]
    spans = [(d.node.lineno, d.node.end_lineno, "test body")]
    spans += [
        (x.lineno, x.end_lineno, "test decorator")
        for o in owners
        for x in o.decorator_list
    ]
    spans += [
        (n.lineno, n.end_lineno, "module body")
        for n in mod.tree.body
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for start, end, where in spans:
        for line in lines[start - 1 : end]:
            if DEVICE_TEXT.search(line) and not PATCHED_CUDA.search(line):
                return f"{where}: {line.strip()[:72]}"
    return None


def reach(graph, roots, wanted, generic):
    """Breadth-first from the roots: the first chain to each wanted key.

    Methods are reached by rapid type analysis over the classes reached so
    far: `self.name` in a method of C reaches `name` on C and on reached
    subclasses of C, `super().name` the nearest base def; with `generic`, any
    other `obj.name` but a dunder reaches `name` on every reached class.
    Calling a class reaches its __init__, __post_init__, __call__ and an
    nn.Module's forward; naming it alone reaches only the class.
    """
    parent, todo, found = {}, deque(), {}
    attrs = {}  # attribute named on a non-self receiver -> its first referrer
    self_attrs = {}  # (class key, attribute) named on self -> first referrer
    reached = defaultdict(list)  # method name -> (class, def) of reached classes

    def push(t, frm):
        if t is not None and t.key not in parent:
            parent[t.key] = frm
            todo.append(t)

    def reach_class(cls, frm, called):
        table = graph.methods_of(cls)
        hooks = ("__init__", "__post_init__", "__call__") if called else ()
        if called and graph.is_nn_module(cls):
            hooks += ("forward",)
        for h in hooks:
            push(table.get(h), frm)
        if cls.key in parent:
            return
        push(cls, frm)
        keys = {a.key for a in graph.ancestors(cls)}
        for name, t in table.items():
            reached[name].append((keys, t))
            if name in attrs:
                push(t, attrs[name])
            for k in keys:
                if (k, name) in self_attrs:
                    push(t, self_attrs[(k, name)])

    for r in roots:
        push(r, None)
    while todo and len(found) < len(wanted):
        d = todo.popleft()
        if d.key in wanted and d.key not in found:
            chain, k = [], d.key
            while k is not None:
                chain.append(k)
                k = parent[k]
            found[d.key] = chain[::-1]
        for t, called in graph.static_edges(d):
            if isinstance(t.node, ast.ClassDef):
                reach_class(t, d.key, called)
            else:
                push(t, d.key)
        for attr in sorted(d.attrs) if generic else ():
            if attr not in attrs and not attr.startswith("__"):
                attrs[attr] = d.key
                for _, t in reached[attr]:
                    push(t, d.key)
        if d.owner is not None:
            for attr in sorted(d.super_attrs):
                for base in graph.ancestors(d.owner)[1:]:
                    t = base.module.defs.get(f"{base.qualname}.{attr}")
                    if t is not None:
                        push(t, d.key)
                        break
            for attr in sorted(d.self_attrs):
                key = (d.owner.key, attr)
                if key not in self_attrs:
                    self_attrs[key] = d.key
                    push(graph.methods_of(d.owner).get(attr), d.key)
                    for keys, t in reached[attr]:
                        if d.owner.key in keys:
                            push(t, d.key)
            for field, attr in sorted(d.field_attrs):
                for a in graph.ancestors(d.owner):
                    for ref in sorted(a.field_types.get(field, ()), key=str):
                        for cls in graph.classes_of(a.module, ref):
                            push(graph.methods_of(cls).get(attr), d.key)
        for alias, attr in sorted(d.attr_on):
            types = d.var_types.get(alias, set()) | d.module.body.var_types.get(
                alias, set()
            )
            for ref in sorted(types, key=str):
                for cls in graph.classes_of(d.module, ref):
                    push(graph.methods_of(cls).get(attr), d.key)
    return found


def main():
    generic = "--precise" not in sys.argv[1:]
    mods = load_tree()
    graph = Graph(mods)
    driver = set(read_list("scripts/compass/cpu_gate_exclude.txt"))
    wanted = {}
    for area, items in TARGETS.items():
        for m, q in items:
            if m not in mods or q not in mods[m].defs:
                sys.exit(f"target missing at this tree: {m}:{q}")
            wanted[f"{m}:{q}"] = area
    files = batch_scope()
    n_tests = 0
    # target key -> {"device": [...], "host": [...]} of (chain, test, reason)
    hits = defaultdict(lambda: {"device": [], "host": []})
    for path in files:
        mod = mods.get(module_name(os.path.join(ROOT, path)))
        if mod is None:
            continue
        cfs = conftests(mods, path)
        shared = [mod.body] + [c.body for c in cfs]
        groups = {"device": [], "host": []}
        reasons = {}
        for q, d in mod.defs.items():
            if q.rsplit(".", 1)[-1].startswith("test") and q not in mod.classes:
                n_tests += 1
                reasons[d.key] = device_reason(mod, d, driver)
                groups["device" if reasons[d.key] else "host"].append(d)
        # Module and conftest bodies run with every test, so they root the
        # host pass only: a device hit is a device test's own chain.
        passes = (("device", groups["device"]), ("host", groups["host"] + shared))
        for kind, roots in passes:
            if not roots:
                continue
            for key, chain in reach(graph, roots, wanted, generic).items():
                hits[key][kind].append((len(chain), path, reasons.get(chain[0]), chain))
    print(f"batch-scope test files: {len(files)}, test functions: {n_tests}")
    for area, items in TARGETS.items():
        print(f"\n## {area}")
        print(
            "| function | first device test file (shortest chain) | device files | host-only files |"
        )
        print("|---|---|---|---|")
        for m, q in items:
            h = hits[f"{m}:{q}"]
            dev = sorted(h["device"])
            host = {x[1] for x in h["host"]} - {x[1] for x in dev}
            first = f"`{dev[0][1]}`" if dev else "none"
            print(f"| `{q}` | {first} | {len(dev)} | {len(host)} |")
    # Per device file: what it reaches, and its chain into each area by the
    # shortest one, which is where a reader checks the reach.
    by_file = defaultdict(dict)
    for key in wanted:
        for n, path, why, chain in hits[key]["device"]:
            by_file[path][key] = (n, why, chain)
    print("\n## device files: each one's targets, and its shortest chain per area")
    for path in sorted(by_file):
        found = by_file[path]
        print(f"\n{path} [{min(found.values())[1]}]")
        for area, items in TARGETS.items():
            keys = [f"{m}:{q}" for m, q in items if f"{m}:{q}" in found]
            if keys:
                print(f"  {area}: " + ", ".join(k.split(":")[1] for k in keys))
                chain = min(found[k] for k in keys)[2]
                print("    " + " -> ".join(chain))


if __name__ == "__main__":
    main()
