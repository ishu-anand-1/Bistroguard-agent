import re
from typing import Tuple

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|api\s+key|secret|credentials)",
    r"show\s+(me\s+)?(every|all)\s+(customer|order|ticket)",
    r"dump\s+(the\s+)?(database|orders|table)",
    r"developer\s+mode",
    r"select\s+\*\s+from",
    r"drop\s+table"
]

def check_prompt_injection(user_message: str) -> Tuple[bool, str]:
    """Deterministic pre-LLM filter blocking prompt injection and unauthorized data dump attempts."""
    cleaned = user_message.lower().strip()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned):
            return (
                False,
                "Security Guardrail: Request refused. I cannot override system instructions, expose internal configuration, or access bulk customer records."
            )
    return True, ""