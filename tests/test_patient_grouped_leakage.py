"""
Unit & Property-based test for Section 4.1: Patient-Grouped Leakage Prevention.
Asserts that:
1. When patient_id exists, no patient appears in both training and validation splits.
2. When perceptual duplicate clusters exist, duplicate records stay together in the same split.
"""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold


def test_no_patient_overlap_across_splits():
    rng = np.random.RandomState(42)
    N = 300
    n_patients = 60
    patient_ids = np.array([f"patient_{i % n_patients}" for i in range(N)])
    y = rng.binomial(1, 0.4, size=N)
    X = rng.randn(N, 10)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=patient_ids)):
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])

        # Strict set disjointness: intersection must be empty
        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, f"Patient leakage detected in fold {fold_idx}: {overlap}"


def test_duplicate_clusters_stay_together():
    """
    Verifies that identical duplicate perceptual hashes are assigned identical groups
    and never split across train and validation sets.
    """
    hashes = ["hash_A", "hash_B", "hash_A", "hash_C", "hash_B", "hash_D"] * 20
    y = np.array([0, 1, 0, 1, 1, 0] * 20)
    X = np.zeros((len(hashes), 5))

    # Map duplicate hashes to cluster group IDs
    unique_hashes = {h: idx for idx, h in enumerate(set(hashes))}
    groups = np.array([unique_hashes[h] for h in hashes])

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups)):
        train_hashes = set([hashes[i] for i in train_idx])
        val_hashes = set([hashes[i] for i in val_idx])

        overlap = train_hashes.intersection(val_hashes)
        assert len(overlap) == 0, f"Duplicate cluster leakage in fold {fold_idx}: {overlap}"
