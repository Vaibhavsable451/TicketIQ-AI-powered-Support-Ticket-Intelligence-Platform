"""Feature 3: PII redaction before any text is sent to an external LLM."""
import re

PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(r"\+?\d[\d\s().-]{8,}\d"),
}


def redact(text: str):
    """Returns (redacted_text, list_of_pii_types_found)."""
    found = []
    for name, pat in PATTERNS.items():
        if pat.search(text):
            found.append(name)
            text = pat.sub(f"[{name}]", text)
    return text, found