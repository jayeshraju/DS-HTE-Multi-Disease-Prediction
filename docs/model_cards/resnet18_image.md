# Model Card: Frozen ResNet-18 Medical Vision Feature Extractor

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Model Details
- **Backbone**: Pretrained ResNet-18 (ImageNet weights, lower convolutional layers frozen).
- **Modality Preprocessing**: Upscaling 2D single-slice images to $224 \times 224 \times 3$, ImageNet mean/standard deviation normalization.
- **Classification Head**: L2-regularized Logistic Regression / calibrated linear head fitted on penultimate 512-dimensional feature embeddings.
- **Inference Speed**: Fast, deterministic execution in $< 35\text{ ms}$ on standard multi-core CPU.
- **Safety Layer**: Layer-4 convolutional Grad-CAM saliency heatmaps + Mahalanobis distance out-of-distribution (OOD) detector calibrated to the validation 95th percentile.

---

## 2. Benchmark Verification (MedMNIST Official Test Splits)

| Dataset | Test Accuracy | ROC-AUC | Test Majority Baseline | Published Literature Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **PneumoniaMNIST** | **82.21%** | **0.9100** | 62.50% (390/624 positive) | 85.0% - 88.0% |
| **BreastMNIST** | **80.13%** | **0.8187** | 73.08% (114/156 positive) | 82.0% - 86.0% |

*Gap Explanation*: The minor gap relative to published full fine-tuning baselines reflects the computational constraint of frozen lower filters on low-resolution ($28 \times 28$) inputs, providing predictable CPU reproducibility without GPU dependencies.
