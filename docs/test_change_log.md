# Test Change Log & Integrity Audit (Section 0.7)

This document tracks all modifications made to the test suite, providing rationale and verifying that tolerances have not been loosened.

---

## 1. `tests/test_cross_table_invariants.py`
- **Change**: Specified `dtype=str` when loading `table_1_cv_mean.csv` and `table_1_extended.csv`.
- **Rationale**: Pandas' default `read_csv` coerces numeric text like `"0.670"` to float `0.67`, inadvertently stripping the formatting required by the strict regex `^\d\.\d{3}$`. Reading as string preserves the exact CSV representation without altering the underlying precision.
- **Tolerance verification**: Equality assertions comparing Table I and Table II remain exact bit-for-bit string matches (`assert row["F1"] == expected_f1`), with numerical comparisons strictly having `abs(a - b) <= 1e-9`.

---

## 2. `tests/test_explain_and_stability.py`
- **Change**: Updated boundary validation in `kuncheva_index` to assert `total_features > 0` before processing identical sets, and correctly return `float("nan")` when $k = n$ (all features selected, where denominator $n - k = 0$).
- **Rationale**: Aligns mathematical behavior with Kuncheva (2007) and Somol & Novovičová (2010), where full feature selection ($k = n$) has an undefined consistency index.
- **Tolerance verification**: Random subset expected value tests remain bounded by strict mathematical limits $[-1.0, 1.0]$.

---

## 3. Added `tests/test_ensemble_weights_audit.py`
- **Change**: New test suite validating Section 0.1 requirements.
- **Rationale**: Verifies that ensemble weights are logged per-fold to `results/audit/ensemble_weights.parquet`, confirms that weights sum to $1.0 \pm 10^{-4}$, checks that the minimum weight floor ($w_i \ge 0.01$) is respected, and flags components exceeding 0.90 mean weight.

---

## 4. Added `tests/test_no_loosened_tests.py`
- **Change**: Programmatic linter test asserting all invariant test assertions have numerical tolerances $\le 1 \times 10^{-9}$.
- **Rationale**: Guarantees that future pipeline adjustments cannot quietly widen test margins to pass artificial benchmarks.
