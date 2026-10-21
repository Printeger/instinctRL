"""CPU source-manifest rejection tests; no runtime or GPU tool execution."""
import copy
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from training_workloads.unified_h1 import config as c
from training_workloads.unified_h1.authorization import load_r3_authorization

ROOT=Path(__file__).resolve().parents[1]


def module(path):
    s=path.removeprefix('isaaclab-training/').removesuffix('.py').replace('/','.')
    return s.removesuffix('.__init__')
def resolve(name):
    if not name.startswith(('training_workloads','instinctlab','evaluation_workloads')):return None
    base=ROOT/('isaaclab-training' if name.startswith('instinctlab') else '')/name.replace('.','/')
    for p in [base.with_suffix('.py'),base/'__init__.py']:
        if p.is_file():return str(p.relative_to(ROOT))
def closure(seeds):
    result=set(seeds);pending=list(seeds)
    while pending:
        path=pending.pop();name=module(path);package=name if path.endswith('/__init__.py') else name.rpartition('.')[0]
        imports=set()
        for i in range(1,len(name.split('.'))):imports.add('.'.join(name.split('.')[:i]))
        for node in ast.walk(ast.parse((ROOT/path).read_text())):
            if isinstance(node,ast.Import):imports.update(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):
                target=importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else node.module
                if target:
                    imports.add(target);imports.update(target+'.'+x.name for x in node.names if x.name!='*')
        for dep in imports:
            f=resolve(dep)
            if f and f not in result:result.add(f);pending.append(f)
    return result

class SourceConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_transitive_local_imports_and_package_initializers_are_bound(self):
        sources=c.R6_PRODUCT_SOURCES|c.R6_WORKLOAD_SOURCES
        self.assertEqual(closure(sources)-sources,set())

    def test_legacy_sets_are_retained_and_r6_additions_exist(self):
        product,workload=c.required_source_sets('unified-h1-r2-config.v3')
        self.assertEqual((product,workload),(c.PRODUCT_SOURCES,c.WORKLOAD_SOURCES))
        product,workload=c.required_source_sets(c.R6_CONFIG_SCHEMA)
        self.assertEqual(len(product-c.PRODUCT_SOURCES),26)
        self.assertEqual(len(workload-c.WORKLOAD_SOURCES),22)
        self.assertTrue({'training_workloads/unified_h1/robustness_training_plan.py',
                         'training_workloads/unified_h1/robustness_training_rng.py'} <= workload)
        for name in product|workload:self.assertTrue((ROOT/name).is_file(),name)
        with self.assertRaises(ValueError):c.required_source_sets('unknown')

    def test_every_added_source_is_required_and_content_bound(self):
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        subprocess.run(['git','-c','user.name=CPU fixture','-c','user.email=cpu@example.invalid','commit','--allow-empty','-qm','fixture'],cwd=self.root,check=True)
        git=lambda *args:subprocess.check_output(['git',*args],cwd=self.root,text=True).strip()
        for sources,old in ((c.R6_PRODUCT_SOURCES,c.PRODUCT_SOURCES),(c.R6_WORKLOAD_SOURCES,c.WORKLOAD_SOURCES)):
            for name in sources:
                p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
            spec=dict(path=str(self.root),base_commit=git('rev-parse','HEAD'),candidate_commit='SELF',candidate_tree='SELF',branch=git('branch','--show-current'),source_sha256={n:c.sha256_file(self.root/n) for n in sources})
            c._validate_repository(spec,label='CPU',required_sources=sources,self_identity=True)
            for name in sources-old:
                with self.subTest(source=name):
                    missing=copy.deepcopy(spec);del missing['source_sha256'][name]
                    with self.assertRaisesRegex(ValueError,'manifest set mismatch'):
                        c._validate_repository(missing,label='CPU',required_sources=sources,self_identity=True)
                    p=self.root/name;original=p.read_bytes();p.write_bytes(original+b'\n# changed\n')
                    with self.assertRaisesRegex(ValueError,'source changed'):
                        c._validate_repository(spec,label='CPU',required_sources=sources,self_identity=True)
                    p.write_bytes(original)

    def test_old_config_cannot_authorize_r6(self):
        old=dict(schema_version='unified-h1-r2-config.v3',future_entry=dict(modes=['research-r6-evaluation']))
        with patch('training_workloads.unified_h1.authorization.load_config',return_value=old):
            with self.assertRaisesRegex(PermissionError,'complete R6 execution config'):
                load_r3_authorization(self.root/'absent.json',self.root,'research-r6-evaluation')

    def test_r6_config_requires_declared_mode(self):
        value=dict(schema_version=c.R6_CONFIG_SCHEMA,future_entry=dict(modes=[]))
        with patch('training_workloads.unified_h1.authorization.load_config',return_value=value):
            with self.assertRaisesRegex(PermissionError,'complete R6 execution config'):
                load_r3_authorization(self.root/'absent.json',self.root,'research-r6-evaluation')

    def test_load_config_selects_r6_manifests_and_mode_without_live(self):
        source=ROOT/'docs/experiments/r5-w-closure-auto/experiments/R5-W-L1-INDEPENDENT-RESOURCE-V2/candidate-config.json'
        value=json.loads(source.read_text());value['schema_version']=c.R6_CONFIG_SCHEMA
        value['method']['proposal_path']=str(c.RESEARCH_METHOD_PATH)
        value['future_entry']['modes'].append('research-r6-evaluation')
        configpath=self.root/'config.json'
        checked=[]
        def repository(spec,**kwargs):
            checked.append((kwargs['label'],kwargs['required_sources']))
            return dict(commit='a'*40,tree='b'*40)
        # External frozen runtime/assets are replaced only for this CPU test;
        # source hash rejection is exercised by the real validator above.
        with patch.object(c,'_validate_repository',side_effect=repository),patch.object(c,'sha256_file',side_effect=lambda p: value['execution']['python_sha256'] if Path(p)==Path(c.sys.executable).resolve() else c.RESEARCH_METHOD_SHA256),patch.object(c,'sys',**{'executable':value['execution']['python_path']}),patch('training_workloads.unified_h1.asset_closure.validate_asset_closure'),patch.object(c,'validate_loaded_module_origins'),patch.object(c,'resource_binding_pins'):
            configpath.write_text(json.dumps(value));result=c.load_config(configpath)
            self.assertEqual(checked,[('product',c.R6_PRODUCT_SOURCES),('workload',c.R6_WORKLOAD_SOURCES)])
            self.assertIn('research-r6-evaluation',result['future_entry']['modes'])
            value['future_entry']['modes'].remove('research-r6-evaluation');configpath.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'mode set mismatch'):c.load_config(configpath)
            checked.clear();value['schema_version']='unified-h1-r2-config.v3';configpath.write_text(json.dumps(value))
            c.load_config(configpath)
            self.assertEqual(checked,[('product',c.PRODUCT_SOURCES),('workload',c.WORKLOAD_SOURCES)])


if __name__=='__main__':unittest.main()
