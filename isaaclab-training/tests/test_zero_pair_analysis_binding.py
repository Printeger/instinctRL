"""CPU evidence bindings; no model forward, simulator, GPU query or optimizer."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.analyze_r5_zero_correction_pair import BASE, digest, evaluation_attempt, pair, read


def test_actual_accepted_c0_spec_binds_real_attempt_and_checkpoint_hashes():
    packet=read(BASE/'launch_inputs.fd-fix-proposed.json')
    path=BASE/'inputs/C0-evaluation-spec.authorized.json'
    attempt,binding=evaluation_attempt(path,'C0',packet)
    spec=read(path)
    assert str(attempt)==spec['h1']['attempt']
    assert binding['sha256']==digest(path)
    assert binding['authorization_sha256']==spec['pins']['authorization']['sha256']


def test_new_proposal_cannot_be_read_as_issued_evaluation(tmp_path):
    packet=read(BASE/'launch_inputs.fd-fix-proposed.json')
    spec=read(BASE/'inputs/C1-extra-evaluation-spec.draft.json')
    spec['pins']['authorization']['path']=str(tmp_path/'unissued_AUTH.json')
    path=tmp_path/'unissued_spec.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='issued evaluation AUTH bytes missing'):
        evaluation_attempt(path,'C1',packet)


def test_valid_auth_cannot_substitute_for_an_actual_terminal_job_request(tmp_path):
    packet=read(BASE/'launch_inputs.fd-fix-proposed.json')
    spec=read(BASE/'inputs/C0-evaluation-spec.authorized.json')
    spec['id']='UNSUBMITTED-CPU-TEST-ONLY'
    path=tmp_path/'never_submitted.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='unique actual job request'):
        evaluation_attempt(path,'C0',packet)


@pytest.mark.parametrize('change',['group','weight','budget','argv'])
def test_mismatched_auth_or_argv_rejected_without_touching_actual_evidence(tmp_path,change):
    packet=read(BASE/'launch_inputs.fd-fix-proposed.json')
    spec=copy.deepcopy(read(BASE/'inputs/C0-evaluation-spec.authorized.json'))
    auth=copy.deepcopy(read(Path(spec['pins']['authorization']['path'])))
    if change=='group': auth['parameters']['group']='C1'
    if change=='weight': auth['parameters']['residual_neutrality_weight']=.05
    if change=='budget': auth['budgets']['training_transitions']=1
    path=tmp_path/'AUTH.json';path.write_text(json.dumps(auth))
    spec['pins']['authorization']={'path':str(path),'sha256':digest(path)}
    spec['argv'][spec['argv'].index('--authorization')+1]=str(path)
    if change=='argv': spec['argv'][-1]='pilot-training'
    draft=tmp_path/'spec.json';draft.write_text(json.dumps(spec))
    with pytest.raises(ValueError): evaluation_attempt(draft,'C0',packet)


def test_missing_c1_z_cannot_produce_a_complete_pair_or_load_training_tensors():
    with patch('tools.analyze_r5_zero_correction_pair.torch.load',side_effect=AssertionError('must not load models before coverage')) as loader:
        with pytest.raises(AssertionError): pair()
        loader.assert_not_called()


def test_symlinked_spec_rejected(tmp_path):
    linked=tmp_path/'spec.json';linked.symlink_to(BASE/'inputs/C0-evaluation-spec.authorized.json')
    with pytest.raises(ValueError,match='regular file'):
        evaluation_attempt(linked,'C0',read(BASE/'launch_inputs.fd-fix-proposed.json'))
