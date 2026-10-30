# Table IV: Statistical Significance Comparisons and Directional Labeling

| Dataset | Comparison | Direction | Wilcoxon W | Wilcoxon p | Holm Adj p | BH FDR p | DeLong p | McNemar p (Holdout) | Nadeau-Bengio p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PIMA | DS-HTE vs LR | No significant difference | 461.5 | 0.5271 | 1.0000 | 0.6588 | 0.8962 | 0.0129 | 0.9831 |
| PIMA | DS-HTE vs RF | No significant difference | 514.5 | 0.9730 | 1.0000 | 0.9776 | 0.0000 | 0.0129 | 0.9648 |
| PIMA | DS-HTE vs XGBoost | No significant difference | 589.5 | 0.8189 | 1.0000 | 0.8870 | 0.0000 | 0.0923 | 0.9543 |
| PIMA | DS-HTE vs LightGBM | No significant difference | 455.0 | 0.8278 | 1.0000 | 0.8870 | 0.0000 | 0.4545 | 0.8979 |
| PIMA | DS-HTE vs CatBoost | Baseline better | 245.5 | 0.0036 | 0.0606 | 0.0076 | 0.0000 | 1.0000 | 0.4250 |
| PIMA | DS-HTE vs Transformer-Only | No significant difference | 433.5 | 0.1669 | 1.0000 | 0.2504 | 0.0000 | 0.0013 | 0.6861 |
| Cleveland | DS-HTE vs LR | No significant difference | 162.5 | 0.3533 | 1.0000 | 0.4608 | 0.1943 | 1.0000 | 0.8612 |
| Cleveland | DS-HTE vs RF | DS-HTE better | 165.0 | 0.0006 | 0.0130 | 0.0021 | 0.0000 | 0.0312 | 0.3075 |
| Cleveland | DS-HTE vs XGBoost | DS-HTE better | 157.0 | 0.0007 | 0.0146 | 0.0021 | 0.0000 | 0.1250 | 0.2962 |
| Cleveland | DS-HTE vs LightGBM | DS-HTE better | 234.5 | 0.0109 | 0.1530 | 0.0193 | 0.0000 | 0.1250 | 0.4318 |
| Cleveland | DS-HTE vs CatBoost | DS-HTE better | 207.5 | 0.0008 | 0.0161 | 0.0021 | 0.0000 | 1.0000 | 0.3209 |
| Cleveland | DS-HTE vs Transformer-Only | DS-HTE better | 90.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0826 |
| CKD | DS-HTE vs LR | No significant difference | 67.5 | 0.9776 | 1.0000 | 0.9776 | 0.7021 | 1.0000 | 0.8973 |
| CKD | DS-HTE vs RF | No significant difference | 67.5 | 0.6584 | 1.0000 | 0.7901 | 0.9422 | 1.0000 | 0.8916 |
| CKD | DS-HTE vs XGBoost | DS-HTE better | 6.5 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 1.0000 | 0.0422 |
| CKD | DS-HTE vs LightGBM | DS-HTE better | 5.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0094 |
| CKD | DS-HTE vs CatBoost | DS-HTE better | 46.0 | 0.0000 | 0.0004 | 0.0001 | 0.2273 | 1.0000 | 0.2470 |
| CKD | DS-HTE vs Transformer-Only | No significant difference | 123.5 | 0.1046 | 1.0000 | 0.1743 | 0.0001 | 1.0000 | 0.6250 |
| ILPD | DS-HTE vs LR | No significant difference | 394.0 | 0.1626 | 1.0000 | 0.2504 | 0.1351 | 0.3750 | 0.8486 |
| ILPD | DS-HTE vs RF | No significant difference | 550.5 | 0.7004 | 1.0000 | 0.8082 | 0.0001 | 0.3877 | 0.9801 |
| ILPD | DS-HTE vs XGBoost | No significant difference | 441.5 | 0.1943 | 1.0000 | 0.2776 | 0.0000 | 0.0034 | 0.7222 |
| ILPD | DS-HTE vs LightGBM | DS-HTE better | 323.5 | 0.0066 | 0.1062 | 0.0132 | 0.0000 | 0.3018 | 0.4194 |
| ILPD | DS-HTE vs CatBoost | Baseline better | 102.5 | 0.0000 | 0.0000 | 0.0000 | 0.0150 | 0.4545 | 0.0710 |
| ILPD | DS-HTE vs Transformer-Only | Baseline better | 153.5 | 0.0000 | 0.0004 | 0.0001 | 0.0001 | 0.0000 | 0.1446 |
| WBCD | DS-HTE vs LR | No significant difference | 29.0 | 0.2673 | 1.0000 | 0.3645 | 0.1146 | 1.0000 | 0.8766 |
| WBCD | DS-HTE vs RF | DS-HTE better | 237.0 | 0.0070 | 0.1062 | 0.0132 | 0.0025 | 1.0000 | 0.4593 |
| WBCD | DS-HTE vs XGBoost | DS-HTE better | 215.5 | 0.0031 | 0.0552 | 0.0071 | 0.0000 | 1.0000 | 0.3290 |
| WBCD | DS-HTE vs LightGBM | DS-HTE better | 168.5 | 0.0000 | 0.0006 | 0.0001 | 0.0000 | 1.0000 | 0.1475 |
| WBCD | DS-HTE vs CatBoost | DS-HTE better | 150.0 | 0.0008 | 0.0161 | 0.0021 | 0.0000 | 1.0000 | 0.2818 |
| WBCD | DS-HTE vs Transformer-Only | DS-HTE better | 177.5 | 0.0010 | 0.0191 | 0.0025 | 0.0000 | 0.0000 | 0.3307 |

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*
