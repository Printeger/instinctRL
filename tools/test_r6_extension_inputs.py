import hashlib
import json
from pathlib import Path
import pytest
from training_workloads.unified_h1 import extension_inputs as inputs


def bind(tmp_path):
    root=tmp_path/'installed';root.mkdir()
    (root/'module.py').write_text('value=1\n')
    (root/'alias.py').symlink_to('module.py')
    p=tmp_path/'manifest.json'
    p.write_text(json.dumps(dict(schema_version='r6-extension-inputs/1',tree=inputs.tree_identity(root))))
    return root,dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


@pytest.mark.parametrize('mutation',['content','add','remove','link'])
def test_full_input_tree_drift_rejected(tmp_path,mutation):
    root,pin=bind(tmp_path)
    assert inputs.validate(pin)==str(root)
    if mutation=='content':(root/'module.py').write_text('value=2\n')
    elif mutation=='add':(root/'extra.py').write_text('new')
    elif mutation=='remove':(root/'alias.py').unlink()
    else:
        (root/'alias.py').unlink();(root/'alias.py').symlink_to(tmp_path/'manifest.json')
    with pytest.raises(ValueError):inputs.validate(pin)


def test_manifest_hash_is_required(tmp_path):
    root,pin=bind(tmp_path)
    Path(pin['path']).write_text('{}')
    with pytest.raises(Exception):inputs.validate(pin)
