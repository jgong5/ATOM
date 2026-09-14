"""Loader source coordinates and prerequisites survive complete-root export."""

import importlib.util
import json
from pathlib import Path

import pytest

from .test_fixed_absolute import bundle, plan_file


spec = importlib.util.spec_from_file_location(
    "fixed_absolute_export", Path(__file__).resolve().parents[2] / "scripts/compass/export_fixed_absolute.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class Record:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self, **kwargs):
        def convert(value):
            if isinstance(value, Record):
                return {k: convert(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value
        return convert(self)

    def metadata(self):
        return self


class Tokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, text):
        return [100, 101]


def loader_fixture(tmp_path, *, join=True):
    data = bundle(root_times=(0., 20.), child_times=((10.,),), join=join)
    _, _, plan = plan_file(tmp_path, data)
    rows, conversations = plan.rows, []
    location = {row["index"]: (row["conversation_id"], row["turn_index"]) for row in rows}
    for c in data["conversations"]:
        turns = []
        for index in c["request_indices"]:
            row = rows[index]
            own_branches = [b for b in data["branches"] if b["after_request"] == index]
            gates = [b for b in data["branches"] if b["join_before_request"] == index]
            turns.append(Record(source_trace_id=row["root_id"],
                source_outer_idx=int(row["source_path"].split("/")[-1]), source_inner_idx=None,
                timestamp=row["arrival_s"] * 1000, max_tokens=row["output_tokens"],
                reset_context=False, raw_messages=[{"role": "user", "content": f"source {index}"}],
                branch_ids=[b["branch_id"] for b in own_branches],
                prerequisites=[Record(kind="spawn_join", branch_id=b["branch_id"],
                    child_conversation_ids=None, timer_seconds=None, barrier_id=None, event_name=None) for b in gates],
                replay_predecessors=[Record(conversation_id=location[p][0], turn_index=location[p][1])
                                     for p in row["loader_replay_predecessors"]]))
        branches = [Record(branch_id=b["branch_id"], child_conversation_ids=b["child_conversation_ids"],
            mode=b["mode"], dispatch_timing=b["dispatch_timing"], is_background=b["is_background"])
            for b in data["branches"] if b["parent_conversation_id"] == c["conversation_id"]]
        conversations.append(Record(session_id=c["conversation_id"], agent_depth=c["agent_depth"],
            is_root=c["agent_depth"] == 0, parent_conversation_id=c["parent_conversation_id"],
            branches=branches, turns=turns))
    sources = [root["source"] for root in plan.roots]
    return sources, [json.loads(Path(r["path"]).read_text()) for r in sources], conversations


def test_export_preserves_all_source_paths_and_derives_exact_join(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "tokenizer_identity", lambda *_: {"fixture": True})
    sources, data, conversations = loader_fixture(tmp_path)
    out = export.compile_bundle(sources, data, conversations, Tokenizer(), {"model": "fixture"})
    assert [r["source_path"] for r in out["requests"]] == ["/requests/0", "/requests/1", "/requests/2"]
    assert [r["depends_on"] for r in out["requests"]] == [[], [0, 2], [0]]
    assert out["requests"][1]["arrival_s"] == 20.
    assert out["requests"][2]["arrival_s"] == 10.
    assert out["clients"] == 1 and len(out["conversations"]) == 2


@pytest.mark.parametrize("damage", ["duplicate_source", "omitted_leaf", "changed_timestamp", "extra_barrier", "timer"])
def test_export_refuses_unsupported_loader_output_without_pruning_metadata(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(export, "tokenizer_identity", lambda *_: {"fixture": True})
    sources, data, conversations = loader_fixture(tmp_path, join=False)
    if damage == "duplicate_source":
        conversations[1].turns[0].source_outer_idx = 0
    elif damage == "omitted_leaf":
        conversations.pop()
    elif damage == "changed_timestamp":
        conversations[0].turns[1].timestamp -= 1000.
    elif damage == "extra_barrier":
        conversations[0].turns[1].replay_predecessors = [Record(conversation_id="root0/child0", turn_index=0)]
    elif damage == "timer":
        conversations[0].turns[1].prerequisites = [Record(kind="timer", branch_id=None, timer_seconds=1.)]
    with pytest.raises(export.UnsupportedExport) as error:
        export.compile_bundle(sources, data, conversations, Tokenizer(), {"model": "fixture"})
    assert error.value.metadata["conversations"]
    if damage == "extra_barrier":
        assert error.value.missing_predecessors == [{"request_index": 1, "predecessor_index": 2}]
