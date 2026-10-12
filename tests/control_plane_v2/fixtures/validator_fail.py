#!/usr/bin/env python3
"""Persisted qualification-path fixture that reports a product gate failure."""
import sys

print("FIXTURE_RESULT=FAIL")
raise SystemExit(1)
