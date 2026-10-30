"""
Security, prompt-injection defense, and sandbox isolation for DS-HTE Copilot.
Follows Gate 1.4:
- Treats all uploaded CSV contents, filenames, image headers, and DICOM tags as passive data, never executable instructions.
- Detects and disarms prompt injection attempts (e.g., "ignore previous instructions", "override system prompt").
- Validates path traversals, zip-slip, and sandboxed storage boundaries.
"""

import os
import re
from typing import Dict, Any, List, Tuple


ADVERSARIAL_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|guidelines|restrictions|rules|prompt|boundaries|instructions|text)",
    r"override\s+(system\s+)?(prompt|rules|policy|safety|protocol)",
    r"disregard\s+(the\s+)?(above|instructions|constraints|prior|system|ethical|rules)",
    r"forget\s+your\s+instructions",
    r"system\s+(override|prompt\s+bypass)",
    r"you\s+are\s+(now\s+)?(an?\s+)?unrestricted",
    r"pretend\s+you\s+are",
    r"report\s+accuracy\s+(99|100|\d{2,3})%",
    r"fake\s+\d+%\s+accuracy",
    r"act\s+as\s+(a\s+)?medical\s+(doctor|clinician)",
    r"fabricate\s+results",
    r"print\s+ungrounded",
    r"declare\s+state-of-the-art",
    r"outperforms\s+all",
    r"outperforms\s+everything"
]


def detect_prompt_injection(user_input: str) -> Tuple[bool, List[str]]:
    """
    Scans input strings for known adversarial prompt injection patterns.
    Returns (is_adversarial, matched_patterns).
    """
    matched = []
    text_lower = user_input.lower()
    for pattern in ADVERSARIAL_PATTERNS:
        if re.search(pattern, text_lower):
            matched.append(pattern)
    return len(matched) > 0, matched


def sanitize_data_payload(payload: Any) -> Any:
    """
    Recursively neutralizes potential prompt injections in uploaded data dictionaries/cells.
    Wraps text fields with data boundaries.
    """
    if isinstance(payload, str):
        # Escape any instruction markers
        clean = payload.replace("```", "").replace("<system>", "").replace("</system>", "")
        is_inj, _ = detect_prompt_injection(clean)
        if is_inj:
            return f"[SANITIZED_DATA_INPUT: {clean[:50]}...]"
        return clean
    elif isinstance(payload, dict):
        return {k: sanitize_data_payload(v) for k, v in payload.items()}
    elif isinstance(payload, list):
        return [sanitize_data_payload(x) for x in payload]
    return payload


def is_path_safe_in_sandbox(target_path: str, base_dir: str = "data") -> bool:
    """
    Asserts target_path resides safely within base_dir (prevents zip-slip / traversal).
    """
    base_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(target_path)
    return target_abs.startswith(base_abs)
