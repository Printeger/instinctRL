"""Commit/reveal primitives for the R6 final common random realizations.

Pure functions: no nonce generation, checkpoint selection, release authorization,
filesystem writes or live launch. The controller must enforce freeze ordering.
Model identity is deliberately absent from stream derivation so paired policies
receive identical conditions. This is not an actor input.
"""
import hashlib
import json
import re

VERSION = 'r6-final-realization/1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def hexadecimal(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('R6_HOLDOUT_SHA256_OR_256BIT_NONCE_REQUIRED')
    return value


def commitment(nonce, *, protocol_sha256, freeze_sha256):
    """Bind secret entropy to the complete protocol and model-selection freeze."""
    return dict(schema_version=VERSION,
                protocol_sha256=hexadecimal(protocol_sha256),
                freeze_sha256=hexadecimal(freeze_sha256),
                nonce_commitment=digest([VERSION, hexadecimal(nonce),
                                         protocol_sha256, freeze_sha256]))


def reveal(record, nonce):
    """Verify the committed entropy before deriving three common stream seeds.

All three are used for BASELINE_SUFFICIENT independent conditions. A trained
model comparison uses the same preselected realization for baseline and each
candidate; it must never index streams by model identity.
"""
    expected = commitment(nonce, protocol_sha256=record['protocol_sha256'],
                          freeze_sha256=record['freeze_sha256'])
    if record != expected:
        raise ValueError('R6_HOLDOUT_COMMITMENT_MISMATCH')
    seeds = []
    # Deterministic rejection handles the rare 31-bit collision, not outcomes.
    for ordinal in range(3):
        counter = 0
        while True:
            seed = int(digest([VERSION, nonce, record, ordinal, counter])[:16], 16) % 2**31
            if seed not in seeds:
                seeds.append(seed)
                break
            counter += 1
    return dict(schema_version=VERSION, commitment=record, nonce=nonce,
                split='final', realization_seeds=seeds,
                pairing='same realization seed for baseline and every compared model')
