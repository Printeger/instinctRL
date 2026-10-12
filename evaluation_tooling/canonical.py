"""Strict canonical JSON parsing, writing, and schema-labelled comparison."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Literal


class CanonicalJSONError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError(f"non-standard/nonfinite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate object key: {key}")
        result[key] = value
    return result


def canonical_dumps(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise CanonicalJSONError(str(error)) from error


def canonical_loads(data: bytes, *, require_canonical: bool = True) -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CanonicalJSONError("invalid UTF-8") from error
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    try:
        value, end = decoder.raw_decode(text)
    except (json.JSONDecodeError, CanonicalJSONError) as error:
        if isinstance(error, CanonicalJSONError):
            raise
        raise CanonicalJSONError(f"malformed JSON: {error.msg}") from error
    remainder = text[end:]
    if remainder not in ("", "\n"):
        if remainder.strip():
            raise CanonicalJSONError("trailing data")
        raise CanonicalJSONError("noncanonical trailing whitespace")
    if require_canonical and data != canonical_dumps(value):
        raise CanonicalJSONError("noncanonical JSON bytes")
    return value


def canonical_load(path: Path, *, require_canonical: bool = True) -> Any:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise CanonicalJSONError(f"cannot read {path}: {error}") from error
    return canonical_loads(data, require_canonical=require_canonical)


def atomic_canonical_create(path: Path, value: Any) -> bytes:
    """Create one canonical file without replacing an existing destination."""

    import os

    data = canonical_dumps(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return data


ListSemantics = Literal["ordered", "set", "multiset"]


def normalize_list(values: list[Any], semantics: ListSemantics) -> Any:
    if semantics == "ordered":
        return values
    encoded = [canonical_dumps(value) for value in values]
    if semantics == "set":
        if len(set(encoded)) != len(encoded):
            raise CanonicalJSONError("set-labelled list contains a duplicate")
        return tuple(sorted(encoded))
    if semantics == "multiset":
        return tuple(sorted(Counter(encoded).items()))
    raise CanonicalJSONError(f"unknown list semantics: {semantics}")


def schema_equal(left: Any, right: Any, list_semantics: dict[str, ListSemantics], path: str = "$") -> bool:
    """Compare objects by keys and lists by schema-declared semantics."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            schema_equal(left[key], right[key], list_semantics, f"{path}.{key}") for key in left
        )
    if isinstance(left, list):
        semantics = list_semantics.get(path)
        if semantics is None:
            raise CanonicalJSONError(f"list semantics not declared for {path}")
        if semantics == "ordered":
            return len(left) == len(right) and all(
                schema_equal(a, b, list_semantics, f"{path}[]") for a, b in zip(left, right)
            )
        return normalize_list(left, semantics) == normalize_list(right, semantics)
    return left == right
