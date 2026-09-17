"""Compose independently qualified reached-primitive groups over legacy prices.

Only validated heldout work identities receive frozen reference predictions.
References by themselves never become active lookup entries. Domain groups
without an independently qualified handoff refuse before legacy fallback.
"""
import json

from atom.compass.core.cost.cached_q16 import GDN, MHA
from atom.compass.core.cost.library import INTERPOLATED_FLAG, PriceLibrary
from atom.compass.core.cost.low_query import GDN_LAYERS, MHA_LAYERS, _key, _layer
from atom.compass.core.cost.prepared import PreparedOperator

INVALID_ALIAS = "invalid-state-index-evidence"


def work_identity(operator):
    """Keep existing cost/layout identity and the joint GDN state alias relation."""
    op = operator.as_dict() if isinstance(operator, PreparedOperator) else operator
    layer = None
    if op.get("name") in (GDN, MHA):
        canonical = 0 if op["name"] == GDN else 3
        selected = _layer(op, GDN_LAYERS if op["name"] == GDN else MHA_LAYERS, canonical)
        if selected is None:
            return None, None
        layer, op = selected
    alias = None
    if op.get("name") == GDN:
        context = dict(op.get("context") or ())
        arrays = [context.get(name) for name in (
            "non_spec_state_indices_in_tensor", "non_spec_state_indices_tensor")]
        if any(not isinstance(value, (list, tuple)) or len(value) != 2 or value[1] != "int32"
               or not isinstance(value[0], (list, tuple)) for value in arrays):
            return (_key(op), INVALID_ALIAS), layer
        if len(arrays[0][0]) != len(arrays[1][0]):
            return (_key(op), INVALID_ALIAS), layer
        names, rows = {}, []
        for values, _ in arrays:
            row = []
            for value in values:
                if type(value) is not int or value < -1 or value >= 32:
                    return (_key(op), INVALID_ALIAS), layer
                if value < 0:
                    row.append(value)
                else:
                    if value not in names:
                        names[value] = len(names)
                    row.append(names[value])
            rows.append(tuple(row))
        alias = tuple(rows)
    return (_key(op), alias), layer


class ReachedPrimitivePrices(PriceLibrary):
    """Disjoint, independently validated whole-group source exports."""

    def __init__(self, base, handoffs, *, deployment_scope_sha256):
        super().__init__()
        from atom.compass.core.cost.reached_primitive_evidence import load_campaign

        if isinstance(handoffs, str):
            handoffs = json.loads(handoffs)
        if not isinstance(handoffs, list) or not handoffs:
            raise ValueError("reached primitive sources require a nonempty list of explicit handoff pins")
        if getattr(base, "launch_charge_seconds", 0) != 0:
            raise ValueError("reached primitive sources do not qualify an extra launch charge")
        self.base = base
        self._domain, self._selected = {}, {}
        self.campaigns, self.handoff_sha256s = [], []
        loaded, sources, claimed_groups = [], [], set()
        domain_sha = None
        for index, reference in enumerate(handoffs):
            campaign = load_campaign(reference, deployment_scope_sha256, index=index, base=base)
            if domain_sha is not None and campaign["domain_sha256"] != domain_sha:
                raise ValueError("reached source handoffs declare different operator domains")
            domain_sha = campaign["domain_sha256"]
            if claimed_groups.intersection(campaign["selected_groups"]):
                raise ValueError("reached source handoffs overlap selected groups; no load-order choice is allowed")
            claimed_groups.update(campaign["selected_groups"])
            for key, group in campaign["domain"].items():
                if key in self._domain and self._domain[key] != group:
                    raise ValueError("reached work identity collapses different score groups")
                self._domain[key] = group
            for key, record in campaign["records"].items():
                if key in self._selected:
                    raise ValueError("multiple reached sources claim the same work identity")
                self._selected[key] = record
            loaded.extend(campaign["loaded_inputs"])
            sources.extend(campaign["sources"])
            self.campaigns.append(campaign["provenance"])
            self.handoff_sha256s.append(reference["sha256"])
        self.domain_sha256 = domain_sha
        self._domain_work = {key[0] for key in self._domain}
        self.selected_groups = tuple(sorted(claimed_groups))
        self.source_qualified = True
        self.loaded_inputs = base.loaded_inputs + tuple(loaded)
        self.sources = base.sources + sources
        self._prices, self.address_shifted = base._prices, base.address_shifted

    def _qualified_legacy_alias(self, op, topology, registration):
        from atom.compass.core.cost.cached_q16 import CachedQ16Prices
        from atom.compass.core.cost.low_query import LowQueryPrices
        from atom.compass.core.cost.root_prefill import RootPrefillPrices

        answer = self.base.lookup(op, topology, registration)
        if answer[0] is None:
            return None
        provider = self.base
        while provider is not None:
            qualified = isinstance(provider, CachedQ16Prices) or (
                isinstance(provider, (LowQueryPrices, RootPrefillPrices))
                and provider.source_qualified is True)
            if qualified:
                selected = provider._source_lookup(op, topology)
                if selected is not None and selected == answer:
                    return answer
            provider = getattr(provider, "base", None)
        return None

    def _source_lookup(self, op, topology, registration=None):
        key, layer = work_identity(op)
        if key not in self._domain:
            if op.get("name") == GDN and key is not None and key[0] in self._domain_work:
                legacy = (None if key[1] == INVALID_ALIAS else
                          self._qualified_legacy_alias(op, topology, registration))
                return legacy if legacy is not None else (
                    None, "recognized reached GDN work has an unsupported joint state-alias relation")
            return None
        if op.get("group") is not None or any(int(value) != 1 for value in (topology or {}).values()):
            return None, "reached primitive source requires its TP1 noncollective work"
        group = self._domain[key]
        record = self._selected.get(key)
        if record is None:
            return None, "reached primitive group has no independently qualified heldout export: " + group
        if layer is not None and layer != record.get("source_layer"):
            record = dict(record, **{INTERPOLATED_FLAG: True}, layer_transfer={
                "source_layer": record["source_layer"], "target_layer": layer,
                "source_handoff_sha256": record["source_handoff_sha256"]})
        source = ("interpolated://reached-primitive/" + group
                  if record.get(INTERPOLATED_FLAG) else record["source"])
        return record, source

    def lookup(self, op, topology=None, registration=None):
        selected = self._source_lookup(op, topology, registration)
        return selected if selected is not None else self.base.lookup(op, topology, registration)

    def _body_lookup(self, op, topology, registration, modelled_memo):
        selected = self._source_lookup(op, topology, registration)
        return selected if selected is not None else self.base._body_lookup(op, topology, registration, modelled_memo)

    def add(self, *args, **kwargs):
        raise ValueError("build baseline prices before attaching reached primitive sources")

    def host_sync_reason(self, op):
        return self.base.host_sync_reason(op)

    def _can_reuse_prepared_lookups(self):
        return self.base._can_reuse_prepared_lookups()

    def _prepared_config_key(self, topology, registration):
        key = self.base._prepared_config_key(topology, registration)
        return None if key is None else (key, tuple(self.handoff_sha256s))

    def describe(self):
        return (f"ReachedPrimitivePrices({len(self.selected_groups)} qualified whole groups; "
                f"sources={self.handoff_sha256s}; base={self.base.describe()})")
