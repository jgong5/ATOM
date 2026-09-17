"""Proper profiles vary root clients without weakening history or source checks."""
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from atom.compass.core.proper_replay import profile_identity
from atom.compass.replay.aiperf_profile import _check_source_options, load_profile


def prepared(clients):
    return dict(config=dict(scenario='inferencex-agentx-mvp',
        loadgen=dict(concurrency=clients, benchmark_duration=900, derived_clients=clients * 7),
        input=dict(random_seed=42, file='source'),
        endpoint=dict(type='chat', streaming=True, use_server_token_count=True), benchmark_id='same-paired-marker'),
        source={'path': 'source', 'sha256': 'source'}, metadata={'fixture': True},
        conversations=[dict(context_mode='deltas_with_responses', session_id='root', turns=[{'max_tokens': 7}])])


@pytest.mark.parametrize('clients', [1, 2, 4, 8])
def test_profile_identity_records_declared_root_clients(clients):
    identity = profile_identity(prepared(clients))
    assert identity['clients'] == clients
    assert identity['profile_seconds'] == 900 and identity['seed'] == 42
    assert identity['request_calendar'] is None


@pytest.mark.parametrize('clients', [0, 3, 16, True, 2.0, '2'])
def test_other_client_counts_and_nonintegers_are_refused(clients):
    value = prepared(1)
    value['config']['loadgen']['concurrency'] = clients
    with pytest.raises(ValueError, match='C1/C2/C4/C8'):
        profile_identity(value)


@pytest.mark.parametrize('clients,aware,passes', [(1, False, True), (2, False, False), (1, True, True),
                                               (2, True, True), (4, True, True), (8, True, True)])
def test_pinned_constructor_receives_clients_before_deriving_config(tmp_path, monkeypatch, clients, aware, passes):
    models = ModuleType('aiperf.common.models')
    models.Conversation = SimpleNamespace(model_validate=lambda row: SimpleNamespace(
        session_id=row['session_id'], turns=[SimpleNamespace(**turn) for turn in row['turns']]))
    models.DatasetMetadata = SimpleNamespace(model_validate=lambda row: row)
    monkeypatch.setitem(sys.modules, 'aiperf.common.models', models)
    source = tmp_path / 'source.jsonl'
    source.write_text('{"source":"unchanged"}\n')
    value = prepared(clients)
    value['source'] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    value['config']['input']['file'] = str(source)
    data = tmp_path / 'prepared.json'
    data.write_text(json.dumps(value))
    builder = tmp_path / 'builder.py'
    template = repr(prepared(1)['config'])
    signature = 'def config(*, clients=1):\n    return Config(clients)\n' if aware else 'def config():\n    return Config(1)\n'
    builder.write_text('from copy import deepcopy\nSOURCE=None\nclass Config:\n'
        '    def __init__(self, clients):\n        self.clients=clients\n        self.benchmark_id="unset"\n'
        '    def model_dump(self, mode):\n'
        f'        value=deepcopy({template})\n'
        '        value["benchmark_id"]=self.benchmark_id\n'
        '        value["input"]["file"]=str(SOURCE)\n'
        '        value["loadgen"]["concurrency"]=self.clients\n'
        '        value["loadgen"]["derived_clients"]=self.clients*7\n'
        '        return value\n' + signature)
    pin = lambda p: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    plan = {'prepared': pin(data), 'config_builder': pin(builder)}
    if not passes:
        with pytest.raises(ValueError, match='semantic configuration'):
            load_profile(plan)
    else:
        _, _, _, identity, caps = load_profile(plan)
        assert identity['clients'] == clients and caps == {('root', 0): 7}


@pytest.mark.parametrize('option', ['diagnostic_only', 'include_failed_outputless', 'include_failed_final',
                                   'low_q_allow_failed_spread', 'root_prefill_allow_failed_spread'])
def test_failed_source_optins_are_diagnostic_only_and_acceptance_stays_strict(option):
    options = {'require_complete': True, option: True}
    _check_source_options({'purpose': 'diagnostic'}, options)
    with pytest.raises(ValueError, match='unqualified source opt-in'):
        _check_source_options({'purpose': 'acceptance'}, options)
    with pytest.raises(ValueError, match='unqualified source opt-in'):
        _check_source_options({}, options)


@pytest.mark.parametrize('purpose', ['diagnostic', 'acceptance'])
def test_diagnostics_never_omit_unpriced_work_or_use_fixed_workload_sources(purpose):
    with pytest.raises(ValueError, match='unpriced work cannot be omitted'):
        _check_source_options({'purpose': purpose}, {'require_complete': False})
    with pytest.raises(ValueError, match='fixed-workload'):
        _check_source_options({'purpose': purpose}, {'root_prefill_diagnostic_handoff': 'old-fixed-profile.json'})
