import re

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
}

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "you are now",
    "act as if",
    "system prompt",
    "reveal your instructions",
]


def contains_pii(text):
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            return True, pii_type
    return False, None


def contains_injection_attempt(text):
    lowered = text.lower()
    for phrase in INJECTION_PATTERNS:
        if phrase in lowered:
            return True, phrase
    return False, None