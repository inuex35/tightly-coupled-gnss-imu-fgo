"""Check the standalone example with the existing numeric Jacobian helper."""
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from factor_check import check_factor_jacobians

DEMO = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_factor_demo.py'


def test_synthetic_factor():
    demo = runpy.run_path(str(DEMO))
    _, factor, truth, initial = demo['build_problem']()
    np.testing.assert_allclose(factor.unwhitenedError(truth), 0.0, atol=1e-7)
    np.testing.assert_allclose(factor.unwhitenedError(initial), -1.903, atol=1e-7)
    check_factor_jacobians(factor, initial, atol=2e-4)


def test_standalone_demo(tmp_path):
    # An unrelated cwd with no data directory: exercises the documented CLI.
    result = subprocess.run([sys.executable, str(DEMO)], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert 'PASS: synthetic residual and float ambiguity checks' in result.stdout
