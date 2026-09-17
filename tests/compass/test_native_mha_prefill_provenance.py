"""API manifests and source checks retain every new source or validation input."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.core.cost import native_mha_prefill
from atom.compass.core.loaded_input import file_digests, load_json
from .test_native_prefill_provenance import grouped_inputs
from .test_opening_harness import wrapper_evidence, opening, validate


@pytest.mark.parametrize("damage",[None,"missing_read","unregistered","aggregate","unconfigured"])
def test_prefill_source_manifest_and_contract(tmp_path,monkeypatch,damage):
    compass,registry=wrapper_evidence.__wrapped__(tmp_path)
    path=tmp_path/'prefill-handoff.json';path.write_text('{"source":"frozen"}')
    _,loaded=load_json(str(path),role=native_mha_prefill.ROLE_PREFIX+'handoff')
    options=compass['oracle_options']
    options.update(native_mha_prefill_handoff=str(path),native_mha_prefill_handoff_sha256=loaded.sha256)
    rank=compass['loaded_inputs']['ranks'][0];rank['inputs'].append(loaded.as_dict())
    monkeypatch.setattr(native_mha_prefill,'NativeMhaPrefillFallback',
                        lambda *args,**kwargs:SimpleNamespace(loaded_inputs=(loaded,)))
    files=grouped_inputs(rank)['native_mha_prefill_handoff']
    digest=validate._rolled_digest(files)
    compass['oracle_option_files']['native_mha_prefill_handoff']=files
    compass['oracle_option_sha256']['native_mha_prefill_handoff']=digest
    provenance={key:value for key,value in registry['artifacts'][0].items() if key not in ('sha256','contents')}
    registry['artifacts'].extend([dict(provenance,sha256=loaded.sha256,contents=files),
                                  dict(provenance,sha256=digest,contents=files)])
    if damage=='missing_read':rank['inputs'].remove(loaded.as_dict())
    elif damage=='unregistered':registry['artifacts']=[a for a in registry['artifacts'] if a['sha256']!=loaded.sha256]
    elif damage=='aggregate':compass['oracle_option_files']['native_mha_prefill_handoff']={}
    elif damage=='unconfigured':
        options.pop('native_mha_prefill_handoff');options.pop('native_mha_prefill_handoff_sha256')
    bad,notes=opening.check_source_contract(SimpleNamespace(manifest={'server':{'compass':compass,'tensor_parallel_size':1}}),
        registry,'7'*64,{},'prefill fixture')
    if damage is None:
        assert not bad,bad
        assert any('Refusal-only cached-prefill' in note for note in notes)
    else:assert bad,damage


@pytest.mark.parametrize('damage',[None,'missing_read','aggregate','wrong_pin'])
def test_extension_validation_is_reused_and_inventoried_separately(tmp_path,monkeypatch,damage):
    from atom.compass.core.cost import composition_qualification

    compass,registry=wrapper_evidence.__wrapped__(tmp_path)
    rank=compass['loaded_inputs']['ranks'][0];options=compass['oracle_options']
    loaded=[]
    for filename,role in [('old-qualification.json','validation.forward_composition.qualification'),
                          ('extension.json','validation.forward_extension.receipt')]:
        path=tmp_path/filename;path.write_text(json.dumps({'name':filename}))
        _,item=load_json(str(path),role=role);loaded.append(item);rank['inputs'].append(item.as_dict())
    old,added=loaded
    options.update(composition_qualification=old.requested,composition_qualification_sha256=old.sha256,
                   composition_extension=added.requested,composition_extension_sha256=added.sha256)
    extension=SimpleNamespace(loaded_inputs=[added])
    active=SimpleNamespace(compass_composition_extension=extension,compass_loaded_inputs=(),
                           compass_composition_qualification={'ok':True},regions=None)
    monkeypatch.setattr(opening,'_source_contract_oracle',lambda options,inputs,live:active)
    seen=[]
    def qualify(*args,**kwargs):
        seen.append(kwargs['extension'])
        return {'ok':True},loaded
    monkeypatch.setattr(composition_qualification,'validate',qualify)
    files=grouped_inputs(rank)['composition_extension']
    compass['oracle_option_files']['composition_extension']=files
    compass['oracle_option_sha256']['composition_extension']=validate._rolled_digest(files)
    # Validation evidence is deliberately absent from the fitting registry.
    if damage=='missing_read':rank['inputs'].remove(added.as_dict())
    elif damage=='aggregate':compass['oracle_option_files']['composition_extension']={}
    elif damage=='wrong_pin':options['composition_extension_sha256']='f'*64
    bad,_=opening.check_source_contract(SimpleNamespace(manifest={'server':{'compass':compass,'tensor_parallel_size':1}}),
        registry,'7'*64,{},'extension fixture',live_oracle=active)
    if damage is None:
        assert not bad,bad
        assert seen==[extension]
    else:assert bad,damage
