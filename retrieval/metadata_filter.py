import json
from datetime import datetime

METADATA_PATH = "data/processed/paper_metadata.json"


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_by_date(candidates, metadata, after=None, before=None):
    """
    candidates: list of dicts with 'paper_id'
    after / before: 'YYYY-MM-DD' strings, either optional
    """
    filtered = []
    for c in candidates:
        pub_date_str = metadata.get(c["paper_id"], {}).get("published")
        if not pub_date_str:
            continue

        pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")

        if after and pub_date < datetime.strptime(after, "%Y-%m-%d"):
            continue
        if before and pub_date > datetime.strptime(before, "%Y-%m-%d"):
            continue

        filtered.append(c)

    return filtered


def filter_by_paper_id(candidates, paper_ids):
    """paper_ids: list of allowed paper_id strings"""
    return [c for c in candidates if c["paper_id"] in paper_ids]