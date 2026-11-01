"""Fail before importing the retired scheduler in a direct process."""
import importlib.abc
import sys


class NoRlflow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rlflow' or fullname.startswith('rlflow.'):
            raise ImportError('DIRECT_FORBIDDEN_DEPENDENCY: ' + fullname)


def install():
    if any(n == 'rlflow' or n.startswith('rlflow.') for n in sys.modules):
        raise RuntimeError('DIRECT_FORBIDDEN_DEPENDENCY_ALREADY_LOADED')
    if any('/tools/rlflow' in str(p) for p in sys.path):
        raise RuntimeError('DIRECT_FORBIDDEN_PYTHONPATH')
    if not any(isinstance(finder, NoRlflow) for finder in sys.meta_path):
        sys.meta_path.insert(0, NoRlflow())
