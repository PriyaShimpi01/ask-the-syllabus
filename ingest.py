"""Step 1 of the pipeline: PDFs -> clean text -> chunks -> embeddings -> FAISS index.

Usage:   python ingest.py
Input:   data/pdfs/*.pdf
Output:  faiss_index/                    (the searchable vector store)
         data/processed/chunks.jsonl     (every chunk, human-readable)
         data/processed/ingestion_report.json  (preprocessing statistics)
"""
import json
import re
import statistics
import sys
from collections import Counter

import config


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank lines
    return text.strip()


def main() -> int:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import rag_core

    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.PDF_DIR}. Copy Unit_I.pdf and 'Unit II.pdf' there first.")
        return 1

    # ---- 1. load + clean -------------------------------------------------
    pages, pages_total, pages_empty = [], Counter(), Counter()
    for pdf in pdfs:
        for doc in PyPDFLoader(str(pdf)).load():
            pages_total[pdf.name] += 1
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = pdf.name
            if not doc.page_content:
                pages_empty[pdf.name] += 1        # image-only slide: nothing to extract
                continue
            pages.append(doc)
        print(f"Loaded {pdf.name}: {pages_total[pdf.name]} pages "
              f"({pages_empty[pdf.name]} without extractable text)")

    # ---- 2. split --------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    n_before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= config.MIN_CHUNK_CHARS]
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    print(f"Chunks: {n_before} produced, {n_before - len(chunks)} dropped as too short "
          f"(< {config.MIN_CHUNK_CHARS} chars), {len(chunks)} kept")

    # ---- 3. embed + index -------------------------------------------------
    print(f"Embedding with {config.EMBEDDING_MODEL} (first run downloads the model)...")
    store = FAISS.from_documents(chunks, rag_core.get_embeddings())
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.INDEX_DIR))

    # ---- 4. save processed data + report ----------------------------------
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.metadata["chunk_id"],
                "source": c.metadata["source"],
                "page": c.metadata.get("page_label", c.metadata.get("page")),
                "chars": len(c.page_content),
                "text": c.page_content,
            }, ensure_ascii=False) + "\n")

    lengths = [len(c.page_content) for c in chunks]
    report = {
        "files": {p.name: {"pages": pages_total[p.name], "pages_without_text": pages_empty[p.name]}
                  for p in pdfs},
        "total_pages": sum(pages_total.values()),
        "pages_with_text": len(pages),
        "chunks_before_filter": n_before,
        "chunks_kept": len(chunks),
        "chunk_chars": {"min": min(lengths), "mean": round(statistics.mean(lengths), 1),
                        "max": max(lengths)},
        "settings": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                     "min_chunk_chars": config.MIN_CHUNK_CHARS,
                     "embedding_model": config.EMBEDDING_MODEL, "normalized_embeddings": True},
        "vectors_in_index": int(store.index.ntotal),
    }
    with open(config.PROCESSED_DIR / "ingestion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone. Index saved to", config.INDEX_DIR)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
