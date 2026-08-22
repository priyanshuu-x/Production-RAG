import arxiv
import requests

client = arxiv.Client()

search = arxiv.Search(
    query="retrieval augmented generation",
    max_results=20,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in client.results(search):
    response = requests.get(result.pdf_url)
    with open(f"data/raw_pdfs/{result.get_short_id()}.pdf", "wb") as f:
        f.write(response.content)