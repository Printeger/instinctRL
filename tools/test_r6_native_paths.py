"""CPU AppLauncher/settings substitutes; never import Carb or start simulation."""
import json
import ast
import os
from functools import partial
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
from training_workloads.scale_003 import resources as r


class NativePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.calls=[];self.module=ModuleType('isaaclab.app')
        def create(**kwargs):self.calls.append(kwargs);return SimpleNamespace(app='CPU_APP')
        self.module.AppLauncher=create

    def launch(self,mode='research-r6-evaluation',gate=None):
        with patch.dict(sys.modules,{'isaaclab.app':self.module}),patch.object(r,'validate_shutdown_runtime_sources',return_value={}):
            return r.start_after_coexistence_gate(self.root,mode=mode,attempt_id='CPU',authorization_sha256='a'*64,
                gate_runner=gate or (lambda *a,**kw:dict(status='PASS')))

    def test_requested_paths_pass_to_default_launcher_without_disabling_evidence(self):
        self.launch();args=self.calls[0]['kit_args'].split()
        self.assertIn('--/crashreporter/dumpDir='+str(self.root/'native_crash'),args)
        self.assertIn('--/crashreporter/preserveDump=true',args)
        self.assertIn('--/crashreporter/skipOldDumpUpload=true',args)
        self.assertNotIn('enable_crashreporter',self.calls[0])
        self.assertTrue((self.root/'native_crash').is_dir())
        record=json.loads((self.root/'native_path_request.json').read_text())
        self.assertEqual(record['status'],'REQUESTED_NOT_VERIFIED')
        self.assertFalse(record['total_native_capacity_verified'])

    def test_readback_requires_exact_types_and_values(self):
        self.launch();expected=r._r6_native_settings(self.root)
        with patch.object(r,'_read_r6_native_settings',return_value=expected):r.verify_r6_native_paths(self.root)
        record=json.loads((self.root/'native_path_runtime.json').read_text());self.assertEqual(record['status'],'SETTINGS_VERIFIED')
        for bad in [{'wrong':'path'},{**expected,'/crashreporter/preserveDump':1},{**expected,'/crashreporter/dumpDir':'/tmp'}]:
            with self.subTest(bad=bad),patch.object(r,'_read_r6_native_settings',return_value=bad):
                with self.assertRaisesRegex(RuntimeError,'settings differ'):r.verify_r6_native_paths(self.root)
        record=json.loads((self.root/'native_path_runtime.json').read_text());self.assertEqual(record['status'],'SETTINGS_MISMATCH')

    def test_existing_crash_directory_and_symlink_are_never_reused(self):
        (self.root/'native_crash').mkdir()
        with self.assertRaises(FileExistsError):self.launch()
        self.assertEqual(self.calls,[])
        (self.root/'native_crash').rmdir();(self.root/'native_crash').symlink_to(self.tmp.name)
        with self.assertRaises(FileExistsError):self.launch()
        self.assertEqual(self.calls,[])

    def test_declined_gate_creates_no_native_paths(self):
        def fail(*a,**kw):raise RuntimeError('CPU busy')
        with self.assertRaisesRegex(RuntimeError,'busy'):self.launch(gate=fail)
        self.assertEqual(list(self.root.iterdir()),[]);self.assertEqual(self.calls,[])

    def test_legacy_launcher_args_remain_without_r6_settings(self):
        self.launch('research-formal-evaluation')
        self.assertNotIn('crashreporter',self.calls[0]['kit_args'])
        self.assertFalse((self.root/'native_crash').exists())

    def test_actual_backend_registers_app_for_cleanup_before_readback_failure(self):
        source=Path(__file__).resolve().parents[1]/'training_workloads/unified_h1/backend.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='create_live_backend')
        assets=self.root/'assets';assets.mkdir();attempt=self.root/'attempt';(attempt/'results').mkdir(parents=True)
        app=object();launch=SimpleNamespace(app=app);self.module.AppLauncher=lambda **kw:launch
        namespace=dict(__package__='training_workloads.unified_h1',Path=Path,partial=partial,
            load_config=lambda:dict(execution=dict(isaacsim_asset_root=str(assets))),
            run_prelaunch_gate=lambda *a,**kw:dict(status='PASS'),start_after_coexistence_gate=r.start_after_coexistence_gate)
        exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(source),'exec'),namespace)
        owners={}
        with patch.dict(sys.modules,{'isaaclab.app':self.module}),patch.dict(os.environ,{'ISAACSIM_ASSET_ROOT':str(assets)}), \
             patch('training_workloads.unified_h1.config.validate_prelaunch_environment'), \
             patch('training_workloads.unified_h1.config.resource_binding_pins',return_value={}), \
             patch('training_workloads.unified_h1.resource_admission.select_gate',return_value=namespace['run_prelaunch_gate']), \
             patch.object(r,'validate_shutdown_runtime_sources'),patch.object(r,'_read_r6_native_settings',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'settings differ'):
                namespace['create_live_backend'](authorization=dict(mode='research-r6-evaluation',authorization_sha256='a'*64),attempt=attempt,resources=owners)
        self.assertIs(owners['app'],app);self.assertIs(owners['launcher'],launch)
        self.assertNotIn('environment',owners)

    def test_whitespace_path_rejected_before_launcher_argument_split(self):
        self.root=self.root/'has space';self.root.mkdir()
        with self.assertRaisesRegex(ValueError,'whitespace'):self.launch()
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
