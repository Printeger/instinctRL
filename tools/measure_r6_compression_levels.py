"""CPU-only compression experiment; no production codec or quota changes.

Use the same deterministic measurement fixture for every zlib level. The fixture
checks bitwise reconstruction through the unchanged archive decoder. Override
only compression level, leaving field/reference selection and inputs unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import zlib

from tools.measure_r6_capacity_v2 import run


def measure(level):
    if level not in (3, 6, 9):
        raise ValueError('SUPPORTED_LEVEL_REQUIRED')
    original = zlib.compress
    with patch('training_workloads.unified_h1.range_archive.zlib.compress',
               side_effect=lambda data, level=3: original(data, compression_level)):
        # Bind separately: the production function explicitly supplies level=3.
        compression_level = level
        result = run(temporal=True)
    result['compression_level_probe'] = level
    result['production_compression_level_unchanged'] = 3
    result['probe_tool'] = {
        'path': str(Path(__file__).resolve()),
        'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result['capacity_pass'] = False
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', required=True, type=int, choices=(3, 6, 9))
    args = parser.parse_args()
    print(json.dumps(measure(args.level), indent=2))
