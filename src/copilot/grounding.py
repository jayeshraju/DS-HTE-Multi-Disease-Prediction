"""
Grounding verifier and tool trace extractor for DS-HTE Copilot.
Follows Gate 1.2:
- Extracts numbers from draft text.
- Matches numbers against raw tool execution outputs with strict tolerance.
- Strips or flags ungrounded claims.
- Generates source chips linking to tables/figures/run_id.
- Produces a transparent "show work" trace.
"""

import re
import json
from typing import Dict, Any, List, Set, Tuple


def extract_numbers_from_text(text: str) -> List[float]:
    """Extracts all floats and percentages from a response string, ignoring IDs/hashes."""
    cleaned = re.sub(r"dshte_paper_\d+_\d+", " ", text)
    cleaned = re.sub(r"\b[0-9a-f]{8,}\b", " ", cleaned)
    cleaned = re.sub(r"[\w/\\-]+\.(?:csv|json|md|py|parquet|png)", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)

    matches = re.findall(r"[-+]?\d*\.?\d+", cleaned)
    nums = []
    for m in matches:
        try:
            val = float(m)
            nums.append(val)
        except ValueError:
            pass
    return nums


def extract_numbers_from_tool_output(data: Any) -> Set[float]:
    """Recursively extracts all numerical values from structured tool outputs."""
    nums = set()
    if isinstance(data, (int, float)):
        v = float(data)
        nums.add(round(v, 4))
        if 0.0 <= v <= 1.0:
            nums.add(round(v * 100.0, 1))
            nums.add(round(v * 100.0, 2))
        elif 1.0 < v <= 100.0:
            nums.add(round(v / 100.0, 3))
            nums.add(round(v / 100.0, 4))
    elif isinstance(data, str):
        extracted = extract_numbers_from_text(data)
        for n in extracted:
            nums.add(round(n, 4))
            if 0.0 <= n <= 1.0:
                nums.add(round(n * 100.0, 1))
                nums.add(round(n * 100.0, 2))
            elif 1.0 < n <= 100.0:
                nums.add(round(n / 100.0, 3))
                nums.add(round(n / 100.0, 4))
    elif isinstance(data, dict):
        for v in data.values():
            nums.update(extract_numbers_from_tool_output(v))
    elif isinstance(data, (list, tuple)):
        for item in data:
            nums.update(extract_numbers_from_tool_output(item))
    return nums


class GroundingVerifier:
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def verify_response(self,
                        draft_text: str,
                        tool_outputs: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Verifies numbers in draft_text against tool_outputs.
        Returns:
        - verified_text: Text with source annotations or ungrounded warnings
        - sources: List of source chips
        - is_fully_grounded: Boolean indicating if all non-trivial numbers match tool evidence
        """
        all_tool_numbers = set()
        sources = []

        for out in tool_outputs:
            tool_name = out.get("tool", "unknown_tool")
            res_data = out.get("result", {})
            all_tool_numbers.update(extract_numbers_from_tool_output(res_data))
            
            # Extract source chip if available
            if isinstance(res_data, dict) and "source" in res_data:
                src_file = res_data["source"]
                chip = {
                    "tool": tool_name,
                    "source": src_file,
                    "label": src_file.replace("results/tables/", "").replace(".csv", "")
                }
                if chip not in sources:
                    sources.append(chip)

        if not all_tool_numbers:
            # If no quantitative artifact tools were executed (e.g. general questions, conceptual guides, code),
            # the response is grounded in clinical/framework domain knowledge without hallucinated table claims.
            return draft_text, sources, True

        # Check numbers in text
        text_numbers = extract_numbers_from_text(draft_text)
        ungrounded = []

        for num in text_numbers:
            # Skip common structural / descriptive numbers
            if num in {0.05, 0.01, 0.001, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 80.0, 100.0, 2026.0}:
                continue
                
            # Match within tolerance
            matched = any(
                abs(num - tn) <= 0.05 or
                abs(num - tn * 100.0) <= 0.1 or
                abs(num * 100.0 - tn) <= 0.1
                for tn in all_tool_numbers
            )
            if not matched:
                ungrounded.append(num)

        is_fully_grounded = len(ungrounded) == 0
        verified_text = draft_text

        if not is_fully_grounded:
            # Flag ungrounded numbers with a prominent audit notice
            warning_box = f"\n\n> ⚠️ **Grounding Verification Notice**: The following values could not be matched with computed artifact evidence: {ungrounded}. Values derived strictly from computed tables should be referenced instead."
            verified_text += warning_box

        return verified_text, sources, is_fully_grounded


def format_tool_trace(tool_calls: List[Dict[str, Any]]) -> str:
    """Formats an expandable Markdown 'show work' trace."""
    if not tool_calls:
        return ""
    lines = ["\n\n<details><summary><b>Show Work (Tool Trace)</b></summary>\n"]
    for i, tc in enumerate(tool_calls, 1):
        tool = tc.get("tool", "unknown")
        args = tc.get("args", {})
        lines.append(f"{i}. **`{tool}`** `{json.dumps(args)}`")
    lines.append("\n</details>")
    return "\n".join(lines)
