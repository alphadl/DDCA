#!/usr/bin/env python3
"""Run tests without pytest (plain unittest)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def run():
    from tests import test_advantage, test_metrics, test_verl_integration
    import unittest
    load = unittest.defaultTestLoader.loadTestsFromModule
    suite = unittest.TestSuite([
        load(test_advantage), load(test_metrics), load(test_verl_integration)
    ])
    runner = unittest.runner.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run())
