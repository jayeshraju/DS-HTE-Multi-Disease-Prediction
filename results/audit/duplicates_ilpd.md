# Data & Duplicate Audit: ILPD

- **Total Samples**: 583
- **Total Features**: 10 (9 numerical, 1 categorical)
- **Class Distribution**: {'1': 416, '0': 167} (Ratio: 2.49)
- **Overall Missing Rate**: 0.07%
- **Exact Duplicate Rows (Feature-wise)**: 26
- **Near-Duplicate Pairs (Normalized dist < 0.05)**: 25

### Exact Duplicate Groups
Duplicate rows detected. Grouped stratified splitting is enforced so entire duplicate clusters stay within the same fold to prevent test leakage.

### Near-Duplicate Pairs Sample
| Row 1 | Row 2 | Avg Norm Distance | Target 1 | Target 2 |
|---|---|---|---|---|
| 18 | 19 | 0.0000 | 1 | 1 |
| 25 | 26 | 0.0000 | 1 | 1 |
| 33 | 34 | 0.0000 | 0 | 0 |
| 48 | 60 | 0.0344 | 1 | 1 |
| 54 | 55 | 0.0000 | 1 | 1 |
| 56 | 274 | 0.0392 | 0 | 1 |
| 61 | 62 | 0.0000 | 1 | 1 |
| 73 | 153 | 0.0181 | 1 | 0 |
| 105 | 106 | 0.0000 | 0 | 0 |
| 107 | 108 | 0.0000 | 0 | 0 |

