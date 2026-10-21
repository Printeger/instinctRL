"""Real u640 tensors in an explicitly CPU-only derivative, never a live model.

The original CUDA RNG cannot be restored on CPU. Verify that rejection, then
clear only CUDA RNG in a temporary test copy. Original bytes stay untouched.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from instinctlab.training.networks import HistoryActor,NetworkConfig
from instinctlab.training.networks_h1 import PrivilegedCriticH1
from instinctlab.training.ppo import PPOConfig,PPOTrainer
from instinctlab.training.checkpoint import CheckpointError
from instinctlab.training.checkpoint_h1 import load_checkpoint_h1,save_checkpoint_h1,_rng_payload
from training_workloads.unified_h1.backend import _state_fingerprint
from training_workloads.unified_h1.robustness_contract import BASELINE_SHA,BINDINGS_SHA
from training_workloads.unified_h1.robustness_training_rng import branch_from_u640
from tools.test_r6_training_rng import preserve_global_rng,weights

ROOT=Path(__file__).resolve().parents[1]
RECIPE='1'*64  # Explicit test identity, not a registered scientific recipe.


def trainer(payload):
    config=NetworkConfig(**payload['network_config'])
    return PPOTrainer(HistoryActor(config),PrivilegedCriticH1(config),PPOConfig(**payload['ppo_config']))


@pytest.fixture
def origin(tmp_path):
    bindings=ROOT/'docs/experiments/r6-robustness-auto/BASELINE_BINDINGS.json'
    assert hashlib.sha256(bindings.read_bytes()).hexdigest()==BINDINGS_SHA
    b=json.loads(bindings.read_text());source=Path(b['selected_model']['path'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==BASELINE_SHA
    payload=torch.load(source,map_location='cpu',weights_only=True)
    assert payload['rng']['torch_cuda']
    assert payload['method_hashes']==b['method_hashes'] and payload['source_identity']==b['source_identity']
    t=trainer(payload);g=torch.Generator(device='cpu')
    with pytest.raises(CheckpointError,match='CPU trainer cannot restore CUDA'):
        load_checkpoint_h1(source,t,expected_method_hashes=b['method_hashes'],
            expected_source_identity=b['source_identity'],curriculum_generator=g)
    cpu=copy.deepcopy(payload);cpu['rng']['torch_cuda']=[]
    path=tmp_path/'CPU_ONLY_DERIVATIVE.pt';torch.save(cpu,path)
    meta=load_checkpoint_h1(path,t,expected_method_hashes=b['method_hashes'],
        expected_source_identity=b['source_identity'],curriculum_generator=g)
    assert _state_fingerprint([t.actor.state_dict(),t.critic.state_dict(),
        t.actor_optimizer.state_dict(),t.critic_optimizer.state_dict()])==_state_fingerprint([
        payload['actor'],payload['critic'],payload['actor_optimizer'],payload['critic_optimizer']])
    yield t,g,meta,payload
    assert hashlib.sha256(source.read_bytes()).hexdigest()==BASELINE_SHA


def save(path,t,g,meta,seed,**overrides):
    fields=dict(method_hashes={'r6_recipe':RECIPE,'scope':'CPU_TEST_ONLY'},
        source_identity={'scope':'CPU_TEST_ONLY'},transition_counter=640*512,global_policy_step=640*16,
        curriculum=dict(version='curriculum-h1-r6-v1',completed_policy_steps=640*16,
                        recipe_sha256=RECIPE,origin_checkpoint_sha256=BASELINE_SHA,branch_seed=seed),
        training_seed=seed,reset_number=meta['reset_number'],curriculum_generator=g)
    fields.update(overrides)
    return save_checkpoint_h1(path,t,**fields)


@pytest.mark.parametrize('seed',[2026102101,2026102201,2026102301])
def test_real_tensor_branch_roundtrip_keeps_adam_and_full_cpu_rng(origin,tmp_path,seed):
    t,g,meta,payload=origin;before=weights(t)
    branch_from_u640(t,g,seed=seed,origin_sha256=BASELINE_SHA,metadata=meta)
    assert weights(t)==before
    rng=_state_fingerprint(_rng_payload(g,t));path=tmp_path/'BRANCHED_CPU_TEST.pt'
    save(path,t,g,meta,seed)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    other=trainer(saved);other_g=torch.Generator(device='cpu')
    restored=load_checkpoint_h1(path,other,expected_method_hashes=saved['method_hashes'],
        expected_source_identity=saved['source_identity'],curriculum_generator=other_g)
    assert restored['training_seed']==seed and other.config.permutation_seed==seed
    assert weights(other)==before and _state_fingerprint(_rng_payload(other_g,other))==rng
    assert restored['curriculum']['recipe_sha256']==RECIPE


@pytest.mark.parametrize('kind',['legacy_seed','recipe','permutation','origin','branch_seed'])
def test_inconsistent_branch_metadata_cannot_be_saved(origin,tmp_path,kind):
    t,g,meta,_=origin;seed=2026102101
    branch_from_u640(t,g,seed=seed,origin_sha256=BASELINE_SHA,metadata=meta)
    overrides={}
    if kind=='legacy_seed':overrides['curriculum']=meta['curriculum']
    if kind=='recipe':overrides['method_hashes']={'r6_recipe':'2'*64}
    if kind=='permutation':t.config=PPOConfig()
    if kind in ('origin','branch_seed'):
        overrides['curriculum']=dict(version='curriculum-h1-r6-v1',completed_policy_steps=10240,
            recipe_sha256=RECIPE,origin_checkpoint_sha256=BASELINE_SHA,branch_seed=seed)
        overrides['curriculum']['origin_checkpoint_sha256' if kind=='origin' else 'branch_seed']='bad'
    path=tmp_path/'REJECTED.pt'
    with pytest.raises(CheckpointError):save(path,t,g,meta,seed,**overrides)
    assert not path.exists()


def test_tampered_load_rejected_before_any_mutation(origin,tmp_path):
    t,g,meta,_=origin;seed=2026102101
    branch_from_u640(t,g,seed=seed,origin_sha256=BASELINE_SHA,metadata=meta)
    path=tmp_path/'VALID.pt';save(path,t,g,meta,seed)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    payload['method_hashes']['r6_recipe']='2'*64
    bad=tmp_path/'TAMPERED.pt';torch.save(payload,bad)
    other=trainer(payload);other_g=torch.Generator(device='cpu');before=weights(other)
    rng=_state_fingerprint(_rng_payload(other_g,other))
    with pytest.raises(CheckpointError,match='recipe/permutation'):
        load_checkpoint_h1(bad,other,expected_method_hashes=payload['method_hashes'],
            expected_source_identity=payload['source_identity'],curriculum_generator=other_g)
    assert weights(other)==before and _state_fingerprint(_rng_payload(other_g,other))==rng
