import pymupdf
import os
import json

RAW_DIR = "data/raw_pdfs"
OUT_PATH = "data/processed/chunks.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def extract_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.replace("\n", " ")  # avoid words fusing across line breaks


def _split_recursive(text, separators, chunk_size):
    """Pure splitting only — no overlap here."""
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    parts = text.split(separator)

    chunks = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size and len(separators) > 1:
                chunks.extend(_split_recursive(part, separators[1:], chunk_size))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


def add_overlap(chunks, overlap):
    """Add overlap exactly once, after all splitting is done."""
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_tail = chunks[i - 1][-overlap:]
            chunk = prev_tail + " " + chunk
        overlapped.append(chunk)
    return overlapped


def recursive_chunk(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    separators = ["\n\n", ". ", " "]
    raw_chunks = _split_recursive(text, separators, chunk_size)
    return add_overlap(raw_chunks, overlap)


def main():
    os.makedirs("data/processed", exist_ok=True)
    all_chunks = []

    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".pdf"):
            continue

        paper_id = filename.replace(".pdf", "")
        pdf_path = os.path.join(RAW_DIR, filename)

        text = extract_text(pdf_path)
        chunks = recursive_chunk(text)

        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{paper_id}_{idx}",
                "paper_id": paper_id,
                "text": chunk_text.strip(),
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Processed {len(os.listdir(RAW_DIR))} PDFs into {len(all_chunks)} chunks")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()