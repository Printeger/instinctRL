#!/usr/bin/env python3
"""Persisted qualification-path fixture for signal routing."""
import os
import signal

os.kill(os.getpid(), signal.SIGTERM)
