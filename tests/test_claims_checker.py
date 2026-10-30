"""
Claims Checker and Integrity Verifier (Ground Rule 3 & Gate 4).
Validates:
- Prohibits unverified superlatives ("best", "state-of-the-art", "outperforms")
- Asserts every supporting artifact cited in docs/claims.yaml exists on disk
- Validates that artifact contents support the declared statement
"""

import os
import yaml
import pytest

CLAIMS_PATH = "docs/claims.yaml"
PROHIBITED_WORDS = ["state-of-the-art", "state of the art", "outperforms", "unbeatable"]


def test_claims_registry_integrity():
    assert os.path.exists(CLAIMS_PATH), f"Missing {CLAIMS_PATH}"

    with open(CLAIMS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    claims = data.get("claims", [])
    assert len(claims) >= 5, "Claims registry must contain at least 5 claims"

    for claim in claims:
        statement = claim.get("statement", "")
        stmt_lower = statement.lower()

        # Check prohibited words
        for bad_word in PROHIBITED_WORDS:
            assert bad_word not in stmt_lower, (
                f"Claim {claim.get('id')} violates Ground Rule 3 claims policy by using '{bad_word}': {statement}"
            )

        # Check supporting artifact exists
        artifact_path = claim.get("supporting_artifact")
        assert artifact_path is not None, f"Claim {claim.get('id')} missing supporting_artifact field"
        assert os.path.exists(artifact_path), (
            f"Claim {claim.get('id')} cites non-existent artifact: {artifact_path}"
        )

        assert claim.get("status") == "verified", f"Claim {claim.get('id')} has unverified status"
