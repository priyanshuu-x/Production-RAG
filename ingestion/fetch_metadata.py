import arxiv
import json

OUT_PATH = "data/processed/paper_metadata.json"

client = arxiv.Client()

search = arxiv.Search(
    query="retrieval augmented generation",
    max_results=20,
    sort_by=arxiv.SortCriterion.Relevance  
)

metadata = {}

for result in client.results(search):
    paper_id = result.get_short_id()
    metadata[paper_id] = {
        "title": result.title,
        "published": result.published.strftime("%Y-%m-%d"),
        "categories": result.categories,
    }

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata for {len(metadata)} papers to {OUT_PATH}")