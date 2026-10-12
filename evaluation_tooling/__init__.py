"""Certified-candidate shared evaluation evidence tooling.

This package is not certified merely because it is importable.  A Commander
certification record must name the exact Git commit before evaluation use.
"""

from .canonical import CanonicalJSONError, canonical_dumps, canonical_load, canonical_loads
from .findings import Finding, FindingCollector, RuntimeFailure, RuntimeFailureClass
from .float_compare import FLOAT_POLICIES, FloatComparison, compare_floats
from .schemas import INPUT_HASHES_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION

__all__ = [
    "CanonicalJSONError",
    "Finding",
    "FindingCollector",
    "RuntimeFailure",
    "RuntimeFailureClass",
    "FloatComparison",
    "FLOAT_POLICIES",
    "INPUT_HASHES_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "canonical_dumps",
    "canonical_load",
    "canonical_loads",
    "compare_floats",
]

__version__ = "1.1.0"
