import re

CITATION_PATTERN = re.compile(r"\[\d+\]")


def has_citation(answer_text):
    return bool(CITATION_PATTERN.search(answer_text))