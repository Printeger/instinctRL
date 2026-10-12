from __future__ import annotations

import unittest

from evaluation_tooling.canonical import CanonicalJSONError, canonical_dumps, canonical_loads, normalize_list, schema_equal


class CanonicalJSONTests(unittest.TestCase):
    def test_canonical_bytes(self) -> None:
        self.assertEqual(canonical_dumps({"b": 1, "a": "é"}), b'{"a":"\\u00e9","b":1}\n')

    def test_round_trip(self) -> None:
        value = {"x": [1, True, None]}
        self.assertEqual(canonical_loads(canonical_dumps(value)), value)

    def test_duplicate_key_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "duplicate"):
            canonical_loads(b'{"x":1,"x":2}\n')

    def test_nan_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "nonfinite"):
            canonical_loads(b'{"x":NaN}\n')

    def test_infinity_rejected(self) -> None:
        with self.assertRaises(CanonicalJSONError):
            canonical_loads(b'{"x":Infinity}\n')

    def test_invalid_utf8_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "UTF-8"):
            canonical_loads(b'{"x":"\xff"}\n')

    def test_trailing_data_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "trailing data"):
            canonical_loads(b'{}\n{}')

    def test_noncanonical_whitespace_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "noncanonical"):
            canonical_loads(b'{ "x": 1 }\n')

    def test_missing_final_newline_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "noncanonical"):
            canonical_loads(b'{"x":1}')

    def test_object_permutation_equal(self) -> None:
        self.assertTrue(schema_equal({"a": 1, "b": 2}, {"b": 2, "a": 1}, {}))

    def test_ordered_list_permutation_fails(self) -> None:
        self.assertFalse(schema_equal([1, 2], [2, 1], {"$": "ordered"}))

    def test_set_list_permutation_passes(self) -> None:
        self.assertTrue(schema_equal([1, 2], [2, 1], {"$": "set"}))

    def test_set_duplicate_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "duplicate"):
            normalize_list([1, 1], "set")

    def test_multiset_permutation_passes(self) -> None:
        self.assertTrue(schema_equal([1, 2, 1], [2, 1, 1], {"$": "multiset"}))

    def test_undeclared_list_semantics_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalJSONError, "not declared"):
            schema_equal([], [], {})
