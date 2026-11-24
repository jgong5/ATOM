# SPDX-License-Identifier: MIT
"""The memory readings in a run: the rank's member of the configuration's
`memory_readings` entry, loaded from the artifact store by key.

The run file's ``store`` is the artifact store's root. The key is the
platform the machine spec states and `key_fields`, read off the resolved ATOM
config: the served config.json's digest, the parallel layout, and as sorted
JSON the engine config the readings move with, `step_grid.keyed`'s fields and the
scheduler bounds, the resolved `index_cache_dtype`, any KV offload connector
and RapidServe. `compass measure memory` publishes one member per rank,
``readings.<rank>.json``, holding `ENTRY_FIELDS`; each rank reads its own. A configuration with no entry
refuses, naming its key and the fields the nearest entry differs in, and so
does an entry `ArtifactStore.load` refuses under `conditions`.

A store with no entry at the key reads the readings a law file carries
(`carried`, the run file's `CARRIED`): each law file's `LAW_MEMBER` holds
the key its readings were measured at and their members, every one stating
`assumed`, and their ATOM source is the `atom` digest of the law's host key,
so a re-keyed law re-keys them. Readings at another key refuse, naming the
fields they differ in, and readings at another ATOM digest name both.

    python -m atom.compass.run.memory LAW --store S --token T --assumed NOTE

writes a store's entry into the law file LAW (`main`).
"""

import argparse
import json

from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Conditions,
    GateState,
    Key,
    Kind,
    RankCoords,
    Rule,
    Topology,
    keyed_platform,
    machine_platform,
    member_name,
)
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.memory.readings import ASSUMED, MemoryReadings

STEM, EXTENSION = "readings", "json"
#: A law file's member holding its readings: ``key``, the `memory_readings`
#: key they were measured at, and ``members``, each rank's by its member name.
LAW_MEMBER = "memory_readings"
#: The run file's list of the readings its law files carry, each a law file's
#: `LAW_MEMBER` with ``atom``, its host key's ATOM digest, and ``law``, its path.
CARRIED = "memory_readings"


def topology(config) -> Topology:
    """The widths of the deployment `config` resolved to."""
    return Topology(
        dp=config.parallel_config.data_parallel_size,
        pp=config.pipeline_parallel_size,
        pcp=config.prefill_context_parallel_size,
        tp=config.tensor_parallel_size,
    )


def coords(config, rank: int) -> RankCoords:
    """Where the runner of rank `rank` of its engine sits: its DP rank, its
    stage, and in the stage PCP-major, as ATOM's rank arithmetic orders them."""
    pc, width = config.parallel_config, config.tp_world_size
    return RankCoords(
        topology(config),
        dp=pc.data_parallel_rank,
        pp=pc.pipeline_parallel_rank,
        pcp=rank // width,
        tp=rank % width,
    )


def key_fields(config) -> dict:
    """The key of `config`'s readings, less its platform."""
    from atom.compass.run import _file_digest
    from atom.compass.run.step_grid import BOUNDS, keyed
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    engine = keyed(config) | {name: getattr(config, name) for name in BOUNDS}
    # Any kv_transfer_config resolves DeepSeek-V4's indexer to fp8, which sizes
    # its KV blocks and buffers apart from the fp4 one a plain start resolves.
    engine["index_cache_dtype"] = getattr(config, "index_cache_dtype", None)
    leaves = KVConnectorFactory.leaf_connectors(config) or ()
    offload = [n for n, _ in leaves if KVConnectorFactory.is_offload_backend(n)]
    if offload:
        engine["kv_offload"] = sorted(offload)
    # A RapidServe prefill engine reads the card with its decode process on it.
    if getattr(config, "enable_rapidserve", False):
        engine["enable_rapidserve"] = True
    return {
        "model": _file_digest(config.model, "config.json"),
        **topology(config).widths,
        "enable_expert_parallel": config.enable_expert_parallel,
        "enable_dp_attention": config.enable_dp_attention,
        "decode_context_parallel_size": config.decode_context_parallel_size,
        "engine_config": json.dumps(engine, sort_keys=True),
    }


def conditions(fields: dict, **platform) -> Conditions:
    """The readings an entry keyed by `fields` is published or loaded under:
    `platform`'s device, software stack and torch, this process's ATOM source
    and host CPU, and the key's model and engine config."""
    from atom.compass.run import atom_source, host_cpu

    return Conditions.of(
        **platform,
        atom_src=atom_source(),
        model=fields["model"],
        host_cpu=host_cpu(),
        engine_config=fields["engine_config"],
    )


def key_of(config, machine) -> Key:
    """The key of `config`'s readings on the platform `machine` states."""
    stated = keyed_platform(Kind.MEMORY_READINGS, machine_platform(machine))
    return Key.of(Kind.MEMORY_READINGS, **stated, **key_fields(config))


def load(store: str, config, machine, rank: int, carried=()) -> MemoryReadings:
    """Rank `rank`'s readings of `config` on the platform `machine` states, from
    `store`, else from the law files' readings `carried` when `store` has no
    entry at their key, or a refusal naming what is missing or differs: an
    entry there that does not read refuses, whatever a law file carries."""
    key, platform = key_of(config, machine), machine_platform(machine)
    held = ArtifactStore(store)
    try:
        entry = held.load(
            key, conditions=conditions(key.values, **platform), gates=GateState.of()
        )
    except ArtifactRefusal as refused:
        if not carried or (held.directory_for(key) / ENTRY_FILE).is_file():
            raise
        return _carried(carried, key, coords(config, rank), refused)
    member = entry.read_member(STEM, coords(config, rank), EXTENSION)
    return MemoryReadings.from_member(json.loads(member), f"{key.kind} {key.token()}")


def carried_by(law: dict, where: str) -> dict | None:
    """The run file's `CARRIED` item for the law file `law` read at `where`,
    or None when it carries no readings."""
    if LAW_MEMBER not in law:
        return None
    atom = law.get("host", {}).get("key", {}).get("atom")
    return {**law[LAW_MEMBER], "atom": atom, "law": where}


def _differ(ours: dict, theirs: dict) -> list[str]:
    """The key fields `theirs` differs from `ours` in, the engine config's by
    name, as ``engine_config.<name>``."""
    fields = sorted(ours.keys() | theirs.keys())
    differ = [f for f in fields if theirs.get(f) != ours.get(f)]
    if "engine_config" not in differ:
        return differ
    mine, held = (json.loads(k.get("engine_config") or "{}") for k in (ours, theirs))
    names = [n for n in sorted(mine.keys() | held.keys()) if mine.get(n) != held.get(n)]
    return [f for f in differ if f != "engine_config"] + [
        f"engine_config.{n}" for n in names
    ]


def _carried(carried, key: Key, at, missed: ArtifactRefusal) -> MemoryReadings:
    """The readings at `key` among the law files' `carried`, rank `at`'s
    member, or a refusal naming the fields the nearest differ in; `missed` is
    the store's refusal, which a refusal here names too."""
    from atom.compass.run import atom_digest

    ours, nearest = key.values, None
    for law in carried:
        differ = _differ(ours, law["key"])
        if not differ:
            if (theirs := law["atom"]) != (here := atom_digest()):
                raise ArtifactRefusal(
                    Rule.RESOLUTION,
                    f"{missed.what}, and {law['law']}'s readings at it are at ATOM "
                    f"digest {theirs}, its host key's, and this tree's is {here}",
                    "a law file loads only at the tree its host key names",
                )
            source = f"{law['law']}'s {LAW_MEMBER}"
            member = law["members"].get(member_name(STEM, at, EXTENSION))
            read = MemoryReadings.from_member(member, source)
            if not read.assumed:
                raise ValueError(
                    f"{source} states no assumed: it was measured elsewhere"
                )
            return read
        if nearest is None or len(differ) < len(nearest[1]):
            nearest = law["law"], differ
    raise ArtifactRefusal(
        Rule.RESOLUTION,
        f"{missed.what}, and the readings {nearest[0]} carries differ in "
        f"{', '.join(nearest[1])}",
        "compass measure memory takes readings at this key",
    )


def main(argv=None) -> None:
    """Write into the law file LAW the members of the `memory_readings` entry
    the store S holds at token T, each stating `assumed` NOTE, with the
    entry's key, as `LAW_MEMBER`: a run reads them where its store holds none,
    at the ATOM digest of the law's host key. An entry measured under another
    engine config than the law's refuses, naming the fields that differ."""
    from atom.compass.backends.json_file import read_json, write_json

    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.run.memory",
        description=main.__doc__.split("\n\n")[0],
    )
    parser.add_argument("law", help="the law file written into")
    parser.add_argument("--store", required=True, help="the artifact store's root")
    parser.add_argument("--token", required=True, help="the readings entry's token")
    parser.add_argument("--assumed", required=True, help="where they were taken")
    args = parser.parse_args(argv)
    store = ArtifactStore(args.store)
    entry = store.read(store.key_from_token(Kind.MEMORY_READINGS, args.token))
    law = read_json(args.law)
    measured = json.loads(entry.key.values["engine_config"])
    engine = law["engine_config"]
    if differ := [n for n, v in engine.items() if measured.get(n) != v]:
        parser.error(f"{args.token} was measured under another {differ} than the law")
    members = {
        member_name(STEM, at, EXTENSION): json.loads(
            entry.read_member(STEM, at, EXTENSION)
        )
        | {ASSUMED: args.assumed}
        for at in entry.topology.ranks()
    }
    law[LAW_MEMBER] = {"key": entry.key.values, "members": members}
    write_json(law, args.law)
    print(f"{args.law}: {len(members)} members of {entry.key.kind} {args.token}")


if __name__ == "__main__":
    main()
