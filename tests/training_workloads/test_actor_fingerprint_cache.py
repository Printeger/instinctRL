"""Exact fingerprint cache tests on AST-extracted product method, CPU only."""
import ast
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any,Mapping
import pytest
import torch
from training_workloads.unified_h1.b0_observer import PROTECTED_STATE,ReadOnlyObserver

def cache_fixture():
    source=Path('isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')
    method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_diagnostic_cached_actor_digest')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Cache',bases=[],keywords=[],decorator_list=[],body=[method])],type_ignores=[]))
    namespace={'torch':torch,'Any':Any,'Mapping':Mapping,'hashlib':hashlib,'json':json}
    exec(compile(module,str(source),'exec'),namespace)
    return namespace['Cache']()

def test_unchanged_bytes_reuse_hash_with_private_clone_and_changed_values_rehash(monkeypatch):
    cache=cache_fixture();values=torch.arange(1024.).reshape(32,32);calls=[]
    original=torch.Tensor.cpu
    def track(self,*args,**kw):calls.append(self.numel());return original(self,*args,**kw)
    monkeypatch.setattr(torch.Tensor,'cpu',track)
    first=cache._diagnostic_cached_actor_digest({'lidar':values})
    assert calls==[1024]
    assert cache._diagnostic_cached_actor_digest({'lidar':values})==first and calls==[1024]
    clone=next(iter(cache._enhanced_b0_actor_digest_cache.values()))[1]
    assert clone.data_ptr()!=values.data_ptr()
    values.data[0,0]=3.  # deliberately bypass Tensor._version
    assert cache._diagnostic_cached_actor_digest({'lidar':values})!=first and calls==[1024,1024]

def test_byte_comparison_distinguishes_signed_zero_and_preserves_nan_payload():
    cache=cache_fixture();values=torch.tensor([0.,float('nan')])
    first=cache._diagnostic_cached_actor_digest(values)
    assert cache._diagnostic_cached_actor_digest(values)==first
    values.data[0]=-0.
    assert cache._diagnostic_cached_actor_digest(values)!=first

def test_observer_still_rejects_actor_mutation_and_rng_is_unchanged():
    cache=cache_fixture();values=torch.ones(16);rng=torch.get_rng_state().clone()
    def probe():return OrderedDict((key,{'digest':cache._diagnostic_cached_actor_digest(values)} if key=='actor_history' else {'fixture':0}) for key in PROTECTED_STATE)
    observer=ReadOnlyObserver(lambda row:values.data.fill_(2.),probe)
    with pytest.raises(RuntimeError,match='changed protected'):observer([{} for _ in range(32)])
    assert observer.calls==observer.rows==0 and torch.equal(rng,torch.get_rng_state())

def test_shape_and_structure_changes_cannot_reuse_a_stale_fingerprint():
    cache=cache_fixture();values=torch.ones(16)
    first=cache._diagnostic_cached_actor_digest({'lidar':values})
    assert cache._diagnostic_cached_actor_digest({'lidar':values.reshape(4,4)})!=first
    assert cache._diagnostic_cached_actor_digest({'renamed':values})!=first
