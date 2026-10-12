"""CPU integration of exact H1 resource pins and the reused gate.

The six query outputs and launcher are inert test substitutes, never GPU calls.
Legacy SCALE authorization is deliberately kept independent and strict.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from training_workloads.unified_h1.config import resource_binding_pins
from training_workloads.scale_003 import resources

def test_h1_bound_resources_run_real_gate_and_inert_launcher_without_scale_identity(tmp_path):
    pins = resource_binding_pins()
    queries = []
    def query(argv, **kwargs):
        queries.append(argv)
        output = ('0, NVIDIA GeForce RTX 5070, 580.173.02, 12227, 394, 0\n'
                  if argv[1].startswith('--query-gpu=') else
                  '3950, /usr/libexec/gnome-remote-desktop-daemon, 164\n')
        return NS(returncode=0,stdout=output,stderr='')
    original_window = resources.collect_coexistence_window
    def window(*,sample_provider):
        return original_window(sample_provider=sample_provider,sleeper=lambda _:None,monotonic=lambda:0.)
    calls = []
    def gate(*args, **kwargs):
        calls.append('gate')
        return resources.run_prelaunch_gate(*args,**kwargs,resource_pins=pins)
    def launcher(path):
        calls.append('launcher')
        return NS(app=NS(config={'fast_shutdown':False}))
    with patch.object(resources,'load_config',side_effect=AssertionError('old SCALE product must not bind H1')), \
         patch.object(resources.subprocess,'run',side_effect=query), \
         patch.object(resources.time,'monotonic_ns',side_effect=[0,5_000_000_000,10_000_000_000]), \
         patch.object(resources,'collect_coexistence_window',side_effect=window):
        result,_,app = resources.start_after_coexistence_gate(tmp_path,mode='enhanced-b0-diagnostic',
            attempt_id='INERT',authorization_sha256='a'*64,gate_runner=gate,
            launcher_factory=launcher,resource_pins=pins)
        assert resources.validate_graceful_shutdown_configuration(app,resource_pins=pins)['fast_shutdown'] is False
    assert result['status']=='PASS' and len(queries)==6 and calls==['gate','launcher']
    assert all(q[0]==pins['nvidia_smi_path'] for q in queries)

def test_corrupt_bound_executable_rejects_before_any_query_or_launcher(tmp_path):
    pins = resource_binding_pins();pins['nvidia_smi_sha256']='0'*64
    with patch.object(resources.subprocess,'run',side_effect=AssertionError('no query permitted')):
        with pytest.raises(ValueError,match='identity changed'):
            resources.run_prelaunch_gate(tmp_path,mode='enhanced-b0-diagnostic',attempt_id='INERT',
                authorization_sha256='a'*64,resource_pins=pins)

def test_h1_runtime_source_pin_failure_is_detectable_by_static_validation():
    with patch('training_workloads.unified_h1.config.sha256_file',return_value='0'*64):
        with pytest.raises(ValueError,match='resource source identity mismatch'):
            resource_binding_pins()

def test_legacy_default_still_requires_its_own_product_authority(tmp_path):
    with patch.object(resources,'load_config',side_effect=ValueError('legacy product identity mismatch')), \
         patch.object(resources.subprocess,'run',side_effect=AssertionError('no query permitted')):
        with pytest.raises(ValueError,match='legacy product'):
            resources.run_prelaunch_gate(tmp_path,mode='training',attempt_id='INERT',authorization_sha256='a'*64)
