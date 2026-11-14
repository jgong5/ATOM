# SPDX-License-Identifier: MIT
"""The space file `compass advice` searches, expanded into candidates and points.

A space file is one JSON object. `expand` reads it and returns its model,
machine, store, frozen commit, workload and SLO, beside ``candidates``, one
per combination of the dimensions' values that no exclusion names, and
``points``, each candidate at each of the workload's concurrencies, in that
order. A candidate is named by its values joined with ``-``, in the
dimensions' order, and takes ``common``'s block and each of its values'
blocks; each instance's arguments are ATOM's, ``--model`` first, and a
point's add each argument that varies with the concurrency, its factor times
the point's concurrency.

Every refusal is a ValueError naming the file and the field: an unknown field,
a key given twice in one object, a value of the wrong kind; contradictory
values, two blocks of one candidate that set an argument, a variable, a
departure's reason, its basis or its deployment apart, or an argument both
fixed and varying with the concurrency; a candidate that names no basis; and
an empty expansion. An excluded combination is never judged.
"""

import itertools
import json
import re
from pathlib import Path

#: A dimension's or a value's name; a candidate's joins its values with "-".
NAME = re.compile(r"[A-Za-z0-9_.]+")
FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")
VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COMMIT = re.compile(r"[0-9a-f]{7,40}")
#: `client.sh`'s ``--traces A-B`` or ``--agentx``.
SELECT = re.compile(r"(0|[1-9][0-9]*)-(0|[1-9][0-9]*)|agentx")
#: Each deployment's instances.
INSTANCES = {"single": ("server",), "1p1d": ("prefill", "decode")}
#: What a block sets for one instance, and what it sets for the candidate.
SIDE = ("args", "env", "per_concurrency")
BLOCK = (*SIDE, "basis", "deployment", "departures", "prefill", "decode")


def expand(path) -> dict:
    """The space file at `path`, expanded."""
    where = str(path)
    space = _read(Path(path))
    _check(space, where)
    dims = space["dimensions"]
    kept = []
    for values in itertools.product(*dims.values()):
        chosen = dict(zip(dims, values))
        if not any(ex.items() <= chosen.items() for ex in space.get("exclude", [])):
            kept.append(chosen)
    if not kept:
        raise ValueError(
            f"{where}: the exclusions remove every combination of the dimensions, "
            "so it expands to no candidate"
        )
    common = [("common", space.get("common", {}))]
    candidates = [
        _candidate(
            chosen,
            common + [(f"dimensions.{d}.{v}", dims[d][v]) for d, v in chosen.items()],
            space["model"],
            where,
        )
        for chosen in kept
    ]
    points = [
        _point(candidate, n)
        for candidate in candidates
        for n in space["workload"]["concurrency"]
    ]
    given = ("model", "machine", "store", "commit", "workload", "slo")
    return {
        "space": str(Path(path).resolve()),
        **{field: space.get(field) for field in given},
        "candidates": candidates,
        "points": points,
    }


def _point(candidate: dict, n: int) -> dict:
    """`candidate` at concurrency `n`: each argument varying with the
    concurrency is its factor times `n`."""
    instances = {}
    for side, one in candidate["instances"].items():
        scaled = [w for f, k in one["per_concurrency"].items() for w in (f, str(k * n))]
        instances[side] = {"args": one["args"] + scaled, "env": one["env"]}
    name = candidate["name"]
    return {
        "name": f"{name}-c{n}",
        "candidate": name,
        "concurrency": n,
        "instances": instances,
    }


def _read(path: Path) -> dict:
    def unique(pairs):
        keys = [key for key, _ in pairs]
        if twice := next((k for k in keys if keys.count(k) > 1), None):
            raise ValueError(f"{path}: {twice!r} is given twice in one object")
        return dict(pairs)

    try:
        return json.loads(path.read_text(), object_pairs_hook=unique)
    except json.JSONDecodeError as bad:
        raise ValueError(f"{path}: not JSON: {bad}") from bad


def _candidate(chosen: dict, blocks: list, model: str, where: str) -> dict:
    name = "-".join(chosen.values())
    where = f"{where}: candidate {name}"
    once, departures = {}, {}
    for source, block in blocks:
        for field in ("basis", "deployment"):
            if field in block:
                value = block[field]
                _put(once, field, value, f"{field} {value}", source, where)
        for what, reason in block.get("departures", {}).items():
            key = what  # a variable's name
            if what.startswith("--"):  # an argument, as ATOM's parser reads it
                flag, (what,) = _argument(what, True)
                key = f"--{flag}"
            shown = f"{what} departs: {reason}"
            _put(departures, key, (what, reason), shown, source, where)
    if "basis" not in once:
        raise ValueError(f"{where} names no ATOM basis: give it to common or a value")
    deployment = once.get("deployment", ("single",))[0]
    for source, block in blocks if deployment == "single" else ():
        if side := next((s for s in INSTANCES["1p1d"] if s in block), None):
            raise ValueError(
                f"{where} is single-node, and {source} gives it a {side} side"
            )
    return {
        "name": name,
        "values": chosen,
        "deployment": deployment,
        "basis": once["basis"][0],
        "departures": dict(value for value, *_ in departures.values()),
        "instances": {
            side: _instance(side, blocks, model, where)
            for side in INSTANCES[deployment]
        },
    }


def _instance(side: str, blocks: list, model: str, where: str) -> dict:
    args, env, scaled = {}, {}, {}
    _put(args, "model", ("--model", model), f"--model {model}", "model", where)
    for source, block in blocks:
        for part, at in ((block, source), (block.get(side, {}), f"{source}.{side}")):
            for flag, value in part.get("args", {}).items():
                key, words = _argument(flag, value)
                _put(args, key, words, " ".join(words), at, where)
            for var, value in part.get("env", {}).items():
                _put(env, var, str(value), f"{var}={value}", at, where)
            for flag, factor in part.get("per_concurrency", {}).items():
                key, (spelled, _) = _argument(flag, factor)
                shown = f"{spelled} {factor}x the concurrency"
                _put(scaled, key, (spelled, factor), shown, at, where)
    if both := next((key for key in scaled if key in args), None):
        (_, fixed, was), (_, varied, shown) = args[both], scaled[both]
        raise ValueError(f"{where}: {was} in {fixed}, but {shown} in {varied}")
    return {
        "args": [word for words, *_ in args.values() for word in words],
        "env": {var: value for var, (value, *_) in env.items()},
        "per_concurrency": dict(value for value, *_ in scaled.values()),
    }


def _argument(flag: str, value) -> tuple[str, tuple]:
    """`flag`'s identity as ATOM's parser reads it, ``--a_b`` and ``--a-b``
    one flag and ``--no-x`` the flag ``--x``, and its words."""
    name = flag[2:].replace("_", "-")
    words = ("--" + name,) if value is True else ("--" + name, str(value))
    return name.removeprefix("no-"), words


def _put(table: dict, key: str, value, shown: str, source: str, where: str) -> None:
    """`table[key]` is `value`, shown as `shown`, which `source` sets, unless
    another source set it apart."""
    if key in table and table[key][0] != value:
        _, by, was = table[key]
        raise ValueError(f"{where}: {was} in {by}, but {shown} in {source}")
    table.setdefault(key, (value, source, shown))


def _check(space: dict, where: str) -> None:
    need = ("model", "machine", "store", "workload", "slo", "dimensions")
    _object(where, space, need, ("commit", "common", "exclude"))
    _is(f"{where}: model", _text(space["model"]), "a model directory or name")
    machine = _object(
        f"{where}: machine", space["machine"], ("device", "ptl"), ("stack", "network")
    )
    for field, value in machine.items():
        _is(f"{where}: machine.{field}", _text(value), "a string")
    _is(
        f"{where}: machine.ptl",
        machine["ptl"] in ("enabled", "disabled"),
        "enabled or disabled",
    )
    _is(f"{where}: store", _absolute(space["store"]), "an absolute path")
    if "commit" in space:
        commit = space["commit"]
        ok = isinstance(commit, str) and COMMIT.fullmatch(commit)
        _is(f"{where}: commit", ok, "a commit's hash, 7 to 40 hex digits")
    at = f"{where}: workload"
    workload = _object(
        at, space["workload"], ("traces", "select", "concurrency"), ("duration_s",)
    )
    _is(f"{at}.traces", _absolute(workload["traces"]), "an absolute path")
    select = workload["select"]
    picked = isinstance(select, str) and SELECT.fullmatch(select)
    ok = picked and (select == "agentx" or int(picked[1]) <= int(picked[2]))
    _is(f"{at}.select", ok, "A-B, A at most B, or agentx")
    concurrency = workload["concurrency"]
    ok = isinstance(concurrency, list) and all(_count(n) for n in concurrency)
    _is(f"{at}.concurrency", ok, "a list of positive integers")
    if not concurrency:
        raise ValueError(f"{at}.concurrency lists none, so it expands to no point")
    if twice := next((n for n in concurrency if concurrency.count(n) > 1), None):
        raise ValueError(f"{at}.concurrency lists {twice} twice")
    if "duration_s" in workload:
        _is(f"{at}.duration_s", _count(workload["duration_s"]), "a positive integer")
    bounds = ("min_interactivity_p90", "max_ttft_p90_s")
    for field, value in _object(f"{where}: slo", space["slo"], (), bounds).items():
        _is(f"{where}: slo.{field}", _number(value) and value > 0, "a positive number")
    _block(f"{where}: common", space.get("common", {}))
    dims = space["dimensions"]
    _is(f"{where}: dimensions", isinstance(dims, dict), "an object")
    if not dims:
        raise ValueError(
            f"{where}: dimensions names none, so it expands to no candidate"
        )
    for dim, values in dims.items():
        _name(f"{where}: dimensions", dim)
        _is(f"{where}: dimensions.{dim}", isinstance(values, dict), "an object")
        if not values:
            raise ValueError(
                f"{where}: dimensions.{dim} has no value, so it expands to no candidate"
            )
        for value, block in values.items():
            _name(f"{where}: dimensions.{dim}", value)
            _block(f"{where}: dimensions.{dim}.{value}", block)
    exclude = space.get("exclude", [])
    _is(f"{where}: exclude", isinstance(exclude, list), "a list")
    for i, excluded in enumerate(exclude):
        _is(f"{where}: exclude[{i}]", isinstance(excluded, dict), "an object")
        for dim, value in excluded.items():
            if dim not in dims:
                raise ValueError(f"{where}: exclude[{i}]: unknown dimension {dim!r}")
            if not isinstance(value, str) or value not in dims[dim]:
                raise ValueError(f"{where}: exclude[{i}]: {dim} has no value {value!r}")


def _block(where: str, block, sides: bool = True) -> None:
    block = _object(where, block, (), BLOCK if sides else SIDE)
    for flag, value in _map(f"{where}.args", block.get("args", {})).items():
        _is(f"{where}.args: {flag!r}", FLAG.fullmatch(flag), "a long flag, --name")
        ok = value is True or _number(value) or isinstance(value, str)
        _is(f"{where}.args.{flag}", ok, "a string, a number, or true for a bare flag")
    for var, value in _map(f"{where}.env", block.get("env", {})).items():
        _is(f"{where}.env: {var!r}", VARIABLE.fullmatch(var), "a variable's name")
        ok = _number(value) or isinstance(value, str)
        _is(f"{where}.env.{var}", ok, "a string or a number")
    at = f"{where}.per_concurrency"
    for flag, factor in _map(at, block.get("per_concurrency", {})).items():
        _is(f"{at}: {flag!r}", FLAG.fullmatch(flag), "a long flag, --name")
        _is(f"{at}.{flag}", _count(factor), "a positive integer factor")
    if "basis" in block:
        _is(f"{where}.basis", _text(block["basis"]), "a recipe, as a string")
    if "deployment" in block:
        ok = isinstance(block["deployment"], str) and block["deployment"] in INSTANCES
        _is(f"{where}.deployment", ok, " or ".join(INSTANCES))
    at = f"{where}.departures"
    for what, reason in _map(at, block.get("departures", {})).items():
        ok = FLAG.fullmatch(what) or VARIABLE.fullmatch(what)
        _is(f"{at}: {what!r}", ok, "a long flag, --name, or a variable's name")
        _is(f"{at}.{what}", _text(reason), "its reason, as a string")
    for side in ("prefill", "decode"):
        if side in block:
            _block(f"{where}.{side}", block[side], sides=False)


def _object(where: str, given, required: tuple, optional: tuple) -> dict:
    _map(where, given)
    if unknown := next((f for f in given if f not in (*required, *optional)), None):
        fields = ", ".join((*required, *optional))
        raise ValueError(f"{where}: unknown field {unknown!r}; the fields are {fields}")
    if missing := next((f for f in required if f not in given), None):
        raise ValueError(f"{where}: no {missing!r}")
    return given


def _name(where: str, name: str) -> None:
    if not NAME.fullmatch(name):
        raise ValueError(f"{where}: {name!r} is not a name of letters, digits, . and _")


def _map(where: str, given) -> dict:
    _is(where, isinstance(given, dict), "an object")
    return given


def _is(where: str, ok, what: str) -> None:
    if not ok:
        raise ValueError(f"{where} takes {what}")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _text(value) -> bool:
    return isinstance(value, str) and value != ""


def _absolute(value) -> bool:
    return _text(value) and Path(value).is_absolute()
