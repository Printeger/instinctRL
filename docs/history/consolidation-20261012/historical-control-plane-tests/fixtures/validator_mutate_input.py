#!/usr/bin/env python3
"""Persisted negative fixture that mutates its declared test-only input."""
import sys
from pathlib import Path

Path(sys.argv[1]).write_text("mutated\n", encoding="utf-8")
print("FIXTURE_RESULT=MUTATED_INPUT")
