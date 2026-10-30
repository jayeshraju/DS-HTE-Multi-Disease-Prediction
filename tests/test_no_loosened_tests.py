"""
Section 0.7: Verification that no invariant tests have loosened tolerances.
Asserts that equality checks in test_cross_table_invariants.py have tolerances <= 1e-9.
"""

import os
import re
import pytest

TESTS_DIR = "tests"
INVARIANTS_TEST_FILE = os.path.join(TESTS_DIR, "test_cross_table_invariants.py")


def test_no_loosened_tolerances_in_invariants():
    assert os.path.exists(INVARIANTS_TEST_FILE), f"Invariants test file missing at {INVARIANTS_TEST_FILE}"

    with open(INVARIANTS_TEST_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Search for pytest.approx usages
    approx_matches = re.findall(r"pytest\.approx\([^)]+\)", content)
    for match in approx_matches:
        # Check if abs or rel tolerance is explicitly specified
        tol_match = re.search(r"(?:abs|rel)\s*=\s*([0-9eE\.\-]+)", match)
        if tol_match:
            tol_val = float(tol_match.group(1))
            assert tol_val <= 1e-9, f"Loosened tolerance detected in invariant test: {match} has tol={tol_val} > 1e-9"

    # Search for manual tolerance checks like abs(a - b) < tol
    manual_tol_matches = re.findall(r"abs\([^)]+\)\s*<\s*([0-9eE\.\-]+)", content)
    for tol_str in manual_tol_matches:
        tol_val = float(tol_str)
        assert tol_val <= 1e-9, f"Loosened manual tolerance detected: abs(...) < {tol_val} > 1e-9"

    print("Verified: all invariant test equality checks have tolerance <= 1e-9.")
