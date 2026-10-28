"""Legacy test suite (archived 2026-10-28, refactor). Kept runnable for comparison:

    pytest archive/tests

Shares the helpers of the current suite (test/_helpers).
"""
import os
import sys

_TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test")
if os.path.abspath(_TEST_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(_TEST_DIR))
