"""A resolved deployment record, read as the scope a ragged law is filed under.

The measured process writes down what it stood up: the environment it
resolved, the pools it allocated, the view every bound layer holds over them,
and the backend class each attention module was given. That record is not a
scope. It is a few hundred facts, and which five of them the kernel turns on
is a reading somebody has to make and stand behind. This module is that
reading, written once, with every value carrying the path it came from so a
reader checks it against the record instead of trusting this file.

Two rules it keeps.

*Nothing is defaulted.* A fact the record does not prove is not filled in from
a config default, and not taken from whichever layer happened to state it: the
translation refuses and names what was missing. A scope with an invented
member matches laws it has no evidence it belongs to, which is worse than
having no scope at all -- an absent scope refuses, a wrong one answers.

*Requested is not resolved.* ``config.kv_cache_dtype`` is what the deployment
ASKED for, and ``auto`` is a legal value there; the dtype every bound layer's
view actually has is what ran. Both are kept, only the second becomes the
scope, and a config that names a real dtype the layers contradict is refused
rather than resolved in favour of either.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attention import GDN, GDN_SCOPE, UNIFIED, UNIFIED_SCOPE

__all__ = ["Declaration", "Fact", "FAMILY_LABELS", "declaration_of",
           "read_resolved"]


#: The family label each ragged operator is filed under. A declaration is
#: per-family because the two families are naturally measured under different
#: treatments and identified by different facts -- the GDN kernel never reads
#: the paged KV cache, so a KV layout is not a fact about it.
FAMILY_LABELS = {UNIFIED: "unified", GDN: "gdn"}

#: How the record names each family's module class. Matched on the class the
#: process actually instantiated, not on the module path: a layer called
#: `self_attn` is not evidence of anything, and two models name them
#: differently.
_FAMILY_CLASSES = {
    "unified": ("atom.model_ops.paged_attention.Attention",),
    "gdn": ("atom.model_ops.base_attention.LinearAttention",),
}

#: Environment the unified dispatcher branches on. Part of the backend fact
#: rather than a separate key: "AiterBackend" does not say which branch ran,
#: and two runs that differ in any of these took different kernels through the
#: same backend class.
_DISPATCH_ENVS = ("ATOM_USE_UNIFIED_ATTN", "ATOM_FORCE_ATTN_TRITON",
                  "ATOM_V4_BACKEND", "ATOM_V4_BACKEND_LAYERS")

#: Per-layer facts that size the GDN fixed state. Deliberately without
#: `layer_num`, `layer_name` and `prefix`, which say which layer this is and
#: not what its state costs.
_STATE_ATTRS = ("activation", "head_k_dim", "head_v_dim", "hidden_size",
                "key_dim", "num_k_heads", "num_v_heads", "value_dim")

#: What a tensor's LAYOUT is. `data_ptr`, `storage_offset`, `bytes` and
#: `numel` locate a tensor rather than arrange it, and they differ between
#: layers that hold identical views -- keying on them would make every layer
#: its own deployment.
_VIEW_FIELDS = ("shape", "stride", "dtype", "element_size", "is_contiguous")

_ABSENT = object()


@dataclass(frozen=True)
class Fact:
    """One member of a scope, and where in the record it was read from."""

    #: The scope key this fact became, or the key it was checked against.
    key: str
    #: Which family's scope it belongs to.
    family: str
    value: object
    #: A path into the record, as a reader would follow it.
    source: str
    #: ``resolved`` -- what the process stood up; ``requested`` -- what it was
    #: asked for, kept beside the resolved value and never substituted for it;
    #: ``declared`` -- written down directly by whoever resolved it.
    status: str

    def as_dict(self) -> dict:
        return {"key": self.key, "family": self.family,
                "value": _plain(self.value), "source": self.source,
                "status": self.status}


@dataclass(frozen=True)
class Declaration:
    """The scope each family is asked for prices in, and its evidence."""

    #: family label -> scope mapping
    scopes: dict
    #: every Fact behind it, resolved and requested alike
    facts: tuple = ()

    def for_family(self, label) -> dict:
        return dict(self.scopes.get(label) or {})

    def for_op(self, op: dict) -> dict:
        """The scope for the family this operator belongs to."""
        label = FAMILY_LABELS.get((op or {}).get("name"))
        return self.for_family(label) if label else {}

    def as_dict(self) -> dict:
        return {
            "scopes": {name: {k: _plain(v) for k, v in scope.items()}
                       for name, scope in self.scopes.items()},
            "facts": [fact.as_dict() for fact in self.facts],
        }


def _plain(value):
    """JSON-safe, for a record that leaves the process."""
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _freeze(value):
    """A record's nested lists as something a scope can be compared by."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    return value


def declaration_of(payload, *, where: str) -> Declaration:
    """Whatever ``payload`` is -- a declaration or a resolution record.

    A file may say the scope outright, in which case it is used as written:
    somebody resolved it and put their name on it. Failing that, a record of
    what the process stood up is READ as one here, which is the judgement this
    module exists to make. A payload that is neither is refused, naming the
    facts a ragged law is identified by, because passing a hundred unrelated
    keys down as a deployment makes every law fail to match for reasons
    nobody can see.
    """
    if not isinstance(payload, dict):
        raise ValueError("%s holds %s, not a mapping of deployment facts"
                         % (where, type(payload).__name__))
    block = payload.get("attention_scope")
    if isinstance(block, dict):
        return _declared(block, where + ".attention_scope")
    labels = set(FAMILY_LABELS.values())
    if payload and set(payload) <= labels and all(
            isinstance(value, dict) for value in payload.values()):
        # A per-family declaration: the two families run different kernels and
        # nothing says one deployment fact has to mean anything to both.
        return _declared(payload, where)
    declared = tuple(UNIFIED_SCOPE) + tuple(GDN_SCOPE)
    if any(key in payload for key in declared):
        return _declared(payload, where)
    if any(key in payload for key in ("atom_envs", "caches_summary", "layers",
                                      "kv_views")):
        return read_resolved(payload, where=where)
    raise ValueError(
        "%s states none of the facts a ragged attention law is identified by "
        "(%s), and is not a record of a stood-up deployment this could read "
        "them from. A scope has to be written down by whoever resolved it."
        % (where, ", ".join(declared)))


def _declared(block: dict, where: str) -> Declaration:
    """A scope somebody wrote down, per family or for both.

    Values are frozen the way a collected scope is frozen. A scope is compared
    for equality against the one a law was fitted under, and a JSON file gives
    lists where a record gives tuples; leaving that difference in place would
    refuse a matching deployment on the notation it was written in.
    """
    def frozen(scope):
        return {key: _freeze(value) for key, value in scope.items()}

    per_family = {label: frozen(block[label])
                  for label in FAMILY_LABELS.values()
                  if isinstance(block.get(label), dict)}
    scopes = per_family or {label: frozen(block)
                            for label in FAMILY_LABELS.values()}
    facts = tuple(
        Fact(key, label, value, "%s.%s" % (where, key), "declared")
        for label, scope in sorted(scopes.items())
        for key, value in sorted(scope.items(), key=lambda kv: str(kv[0])))
    return Declaration(scopes=scopes, facts=facts)


def read_resolved(payload: dict, *, where: str = "resolved_scope"
                  ) -> Declaration:
    """Read a stood-up deployment's record as a per-family scope.

    Raises with every unproved fact named at once, rather than one per call:
    whoever has to close the gap needs the whole list, and a record missing
    two facts that reports one reads as nearly complete.
    """
    if not isinstance(payload, dict):
        raise ValueError("%s holds %s, not a mapping of deployment facts"
                         % (where, type(payload).__name__))
    envs = payload.get("atom_envs") or {}
    caches = payload.get("caches_summary") or {}
    config = payload.get("config") or {}
    views = payload.get("kv_views") or {}
    layers = payload.get("layers") or {}
    facts: list = []
    missing: list = []
    scopes: dict = {}
    for label in ("unified", "gdn"):
        members = _layers_of(layers, label)
        if not members:
            continue
        reader = _unified_scope if label == "unified" else _gdn_scope
        scopes[label] = reader(members, views, envs, config, caches, where,
                               facts, missing)
    if not scopes:
        missing.append(
            "%s.layers names no module of either family (%s), so there is "
            "nothing here to read a ragged attention scope from"
            % (where, ", ".join(sorted(
                cls for classes in _FAMILY_CLASSES.values()
                for cls in classes))))
    if missing:
        raise ValueError(
            "%s cannot be read as a ragged attention scope: %s. Nothing is "
            "defaulted here -- a scope with a member this record does not "
            "prove would match laws it has no evidence of belonging to."
            % (where, "; ".join(missing)))
    return Declaration(scopes=scopes, facts=tuple(facts))


def _layers_of(layers: dict, label: str) -> list:
    wanted = _FAMILY_CLASSES[label]
    return sorted((name, entry) for name, entry in (layers or {}).items()
                  if isinstance(entry, dict) and entry.get("class") in wanted)


def _agreed(members, read, what, where, missing):
    """The one value every member of this family states, or None.

    Silence is a disagreement: a layer that does not say has not said it
    matches, and taking the value from the layers that did would put a fact in
    the scope that part of the deployment contradicts.
    """
    seen: dict = {}
    for name, entry in members:
        value = read(entry)
        if value is _ABSENT or value is None:
            missing.append("%s.layers[%s] does not state %s"
                           % (where, name, what))
            return None
        seen.setdefault(repr(value), [value, []])[1].append(name)
    if len(seen) > 1:
        missing.append(
            "the %d modules disagree on %s (%s), so this deployment has no "
            "one value for it" % (len(members), what, "; ".join(
                "%s on %d" % (key, len(names))
                for key, (_v, names) in sorted(seen.items()))))
        return None
    return next(iter(seen.values()))[0]


def _witnessed(members, read, what, where, missing):
    """The one value every member that HAS a record of it states.

    Unlike `_agreed`, a member the record is silent about is not counted as a
    disagreement. This is for facts recorded per holder rather than per
    module: a collector writes the first few holders of a shared pool, not
    one entry per bound layer, and every layer of a family indexes the same
    allocation. Silence there is a short table, not a second layout.

    Refused when NOTHING witnesses it -- a layout no recorded tensor shows is
    not resolved -- and when two witnesses disagree, which would mean the
    family does not have one layout at all. The witnesses are returned so the
    fact can say how many there were: a value one holder proves and a value
    all sixty-four prove are not equally evidenced, and a reader has to be
    able to see which one this is.
    """
    seen: dict = {}
    for name, entry in members:
        value = read(entry)
        if value is _ABSENT or value is None:
            continue
        seen.setdefault(repr(value), [value, []])[1].append(name)
    if not seen:
        missing.append(
            "%s records %s for none of the %d modules of this family"
            % (where, what, len(members)))
        return None, ()
    if len(seen) > 1:
        missing.append(
            "the modules that record %s disagree on it (%s), so this "
            "deployment has no one value for it" % (what, "; ".join(
                "%s on %d" % (key, len(names))
                for key, (_v, names) in sorted(seen.items()))))
        return None, ()
    value, names = next(iter(seen.values()))
    return value, tuple(names)


def _dig(entry, *path):
    for step in path:
        if not isinstance(entry, dict) or step not in entry:
            return _ABSENT
        entry = entry[step]
    return entry


def _layer_view(views: dict, entry: dict):
    """The view the layer with this number holds, by the key the record uses.

    Joined on the layer number the module itself reports rather than on
    position in the view table: the table is written in bind order, and
    reading it positionally would silently pair a GDN layer's state with an
    MHA layer's KV on any model that binds them in a different order.
    """
    number = _dig(entry, "impl_attrs", "layer_num")
    if number is _ABSENT:
        number = _dig(entry, "attrs", "layer_num")
    if number is _ABSENT:
        return _ABSENT
    suffix = "layer_%d" % int(number)
    for key, view in (views or {}).items():
        if str(key).rsplit(":", 1)[-1] == suffix:
            return view
    return _ABSENT


def _view_layout(view):
    """The arrangement of a k/v pair, without where in memory it sits."""
    if not isinstance(view, dict):
        return _ABSENT
    out = []
    for part in ("k", "v"):
        tensor = view.get(part)
        if not isinstance(tensor, dict):
            return _ABSENT
        fields = []
        for field in _VIEW_FIELDS:
            if field not in tensor:
                return _ABSENT
            fields.append((field, _freeze(tensor[field])))
        out.append((part, tuple(fields)))
    return tuple(out)


def _unified_scope(members, views, envs, config, caches, where, facts,
                   missing) -> dict:
    scope: dict = {}
    dtype = _agreed(members, lambda e: _dig(e, "impl_attrs", "kv_cache_dtype"),
                    "impl_attrs.kv_cache_dtype", where, missing)
    if dtype is not None:
        scope["kv_cache_dtype"] = dtype
        facts.append(Fact("kv_cache_dtype", "unified", dtype,
                          "%s.layers.*.impl_attrs.kv_cache_dtype (%d modules "
                          "agree)" % (where, len(members)), "resolved"))
    asked = config.get("kv_cache_dtype")
    if asked is not None:
        facts.append(Fact("kv_cache_dtype", "unified", asked,
                          "%s.config.kv_cache_dtype" % where, "requested"))
        if dtype is not None and str(asked) not in ("auto", "None") \
                and str(asked) != str(dtype):
            missing.append(
                "the deployment asked for kv_cache_dtype %r and its bound "
                "modules hold %r; one of the two is wrong and this cannot "
                "choose" % (asked, dtype))

    layout, witnesses = _witnessed(
        members, lambda e: _view_layout(_layer_view(views, e)),
        "a k/v view in kv_views", where, missing)
    if layout is not None:
        scope["kv_cache_layout"] = layout
        facts.append(Fact("kv_cache_layout", "unified", layout,
                          "%s.kv_views[%s].{k,v}.{%s}"
                          % (where, "|".join(sorted(witnesses)),
                             ",".join(_VIEW_FIELDS)), "resolved"))

    block = config.get("kv_cache_block_size")
    if block is None:
        missing.append("%s.config does not state kv_cache_block_size" % where)
    else:
        facts.append(Fact("kv_cache_block_size", "unified", block,
                          "%s.config.kv_cache_block_size" % where,
                          "requested"))
        # A requested block size becomes a resolved one only where the pool
        # that was actually allocated carries it. Without that check this is
        # the deployment's ask, and an ask that the allocator overrode would
        # file every law under a block size no kernel ever saw.
        # The pool is recorded as ``[shape, dtype]``, so the shape is its
        # first member and not the record itself.
        recorded = _dig(caches, "pool_shapes", "kv_cache")
        shape = (recorded[0] if isinstance(recorded, list) and recorded
                 else _ABSENT)
        pool = shape if isinstance(shape, list) else []
        if int(block) in [int(d) for d in pool if isinstance(d, int)]:
            scope["kv_cache_block_size"] = int(block)
            facts.append(Fact("kv_cache_block_size", "unified", int(block),
                              "%s.caches_summary.pool_shapes.kv_cache "
                              "corroborates the requested value" % where,
                              "resolved"))
        else:
            missing.append(
                "kv_cache_block_size %r is what the deployment asked for and "
                "the KV pool it allocated (%s) does not carry it, so nothing "
                "here proves the block size the kernel saw" % (block, pool))

    window = _agreed(members, lambda e: _dig(e, "impl_attrs", "sliding_window"),
                     "impl_attrs.sliding_window", where, missing)
    if window is not None:
        scope["sliding_window"] = window
        facts.append(Fact("sliding_window", "unified", window,
                          "%s.layers.*.impl_attrs.sliding_window (%d modules "
                          "agree)" % (where, len(members)), "resolved"))

    backend = _agreed(members,
                      lambda e: (e.get("attn_backend", _ABSENT), e.get("impl")),
                      "attn_backend and impl", where, missing)
    dispatch = []
    for name in _DISPATCH_ENVS:
        if name not in envs:
            missing.append(
                "%s.atom_envs does not state %s, which decides which branch "
                "the backend dispatches to; the backend class alone does not "
                "name the kernel that ran" % (where, name))
        else:
            dispatch.append((name, str(envs[name])))
    if backend is not None and len(dispatch) == len(_DISPATCH_ENVS):
        value = (("backend", backend[0]), ("impl", backend[1])) \
            + tuple(dispatch)
        scope["attention_backend"] = value
        facts.append(Fact("attention_backend", "unified", value,
                          "%s.layers.*.{attn_backend,impl} with %s"
                          % (where, ", ".join(
                              "atom_envs.%s" % n for n in _DISPATCH_ENVS)),
                          "resolved"))
    return scope


def _gdn_scope(members, views, envs, config, caches, where, facts,
               missing) -> dict:
    scope: dict = {}
    name = "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST"
    raw = envs.get(name)
    flag = {"true": True, "1": True, "false": False, "0": False}.get(
        str(raw).strip().lower()) if raw is not None else None
    if isinstance(raw, bool):
        flag = raw
    if flag is None:
        missing.append(
            "%s.atom_envs does not state %s as a flag this can read (%r); the "
            "guarded branch is a different kernel, not a faster setting of "
            "this one" % (where, name, raw))
    else:
        scope["gdn_decode_lossy_fast"] = flag
        facts.append(Fact("gdn_decode_lossy_fast", "gdn", flag,
                          "%s.atom_envs.%s" % (where, name), "resolved"))

    attrs = _agreed(
        members,
        lambda e: tuple((a, _freeze(_dig(e, "impl_attrs", a)))
                        for a in _STATE_ATTRS)
        if all(_dig(e, "impl_attrs", a) is not _ABSENT for a in _STATE_ATTRS)
        else _ABSENT,
        "impl_attrs (%s)" % ", ".join(_STATE_ATTRS), where, missing)
    layout, state_witnesses = _witnessed(
        members, lambda e: _view_layout(_layer_view(views, e)),
        "a state view in kv_views", where, missing)
    backend = _agreed(members,
                      lambda e: (e.get("attn_backend", _ABSENT), e.get("impl")),
                      "attn_backend and impl", where, missing)
    slots = caches.get("state_slots")
    if slots is None:
        missing.append("%s.caches_summary does not state state_slots" % where)
    if attrs is not None and layout is not None and backend is not None \
            and slots is not None:
        value = (("attrs", attrs), ("state_view", layout),
                 ("state_slots", int(slots)),
                 ("backend", backend[0]), ("impl", backend[1]))
        scope["gdn_state_geometry"] = value
        facts.append(Fact(
            "gdn_state_geometry", "gdn", value,
            "%s.layers.*.{impl_attrs,attn_backend,impl} (%d modules agree) "
            "with %s.kv_views[%s] and %s.caches_summary.state_slots"
            % (where, len(members), where, "|".join(sorted(state_witnesses)),
               where), "resolved"))
    return scope
