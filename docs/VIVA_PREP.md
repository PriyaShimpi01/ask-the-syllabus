# Viva preparation (every member should read all of this)

## Two questions the project statement says to expect

**How does retrieval reduce hallucination?**
Without retrieval the model must recall facts from training, and when recall is weak it fills the gap with
plausible text. With retrieval we hand it the relevant passages and instruct it to use only those, so the task
becomes reading and summarising instead of remembering. Retrieval does not make hallucination impossible; it
shrinks it and makes the answer checkable through citations. Our closed-book comparison in `eval_report.md` measures this.

**What happens when the answer isn't in the documents?**
FAISS still returns the k nearest chunks (it has no concept of "no match"). The prompt lets the model return
`found: false`; the app then shows "I could not find this information" and no sources. Out-of-scope refusal rate is in our results.

## Algorithms and tools

* **What is an embedding?** A vector that places text in a space where similar meaning means small distance.
* **Why the same embedding model for chunks and questions?** Distances are only meaningful inside one space.
* **L2 vs cosine?** Our vectors are normalised, so L2 distance and cosine similarity rank results identically (d² = 2 - 2cos).
* **FAISS index type?** LangChain's default is exact flat L2 search; no approximation is needed at this size.
* **Why chunk_size 1000 / overlap 150?** About one slide of text; 15% overlap protects boundaries. Too large dilutes retrieval, too small loses context.
* **Why k = 4?** Enough to cover an answer spanning slides without flooding the prompt. Test other values with `--k`.
* **Why temperature 0?** We want repeatable, literal, factual answers.
* **Is anything trained?** No. MiniLM and gpt-oss-20b are pretrained; our artifact is the FAISS index.
* **Why MiniLM?** Small, free, CPU-friendly, 384 dimensions, good semantic quality for English academic text.

## Implementation and testing

* **What was the bug you found?** Sources were built from all retrieved chunks, so "not found" appeared above citations. We now show only chunks the model reports using, with a regression test (`test_not_found_has_no_sources`).
* **Why not refuse using a distance threshold?** FAISS distances for relevant and irrelevant questions overlap (notebook: quantum computing 1.31 vs a real syllabus question 1.34). The report's distance section shows it on our data.
* **How do you measure hallucination?** Out-of-scope questions that get answered, plus in-scope answers the LLM judge flags as unsupported. Definitions are at the top of `metrics.py`.
* **Limits of that measurement?** Keyword matching is crude; the judge is an LLM; the test set is small (25 questions), so percentages move a lot per question.
* **What are retrieval failures vs generation failures?** Retrieval failure: the right chunk is not in the top-k (retrieval hit rate drops). Generation failure: chunk retrieved but answer wrong or unsupported (faithfulness drops).

## Twist questions

* *What if the PDF contains instructions like "ignore previous rules"?* Prompt injection: chunk text is untrusted input; mitigation is delimiting context and output validation. We did not build a full defence.
* *What if two units contradict each other?* Both chunks are retrieved and cited; the bot does not arbitrate.
* *How would you improve it?* OCR for image slides, hybrid keyword+vector search, a reranker, metadata filters by unit, a larger test set.
* *Why not fine-tune the LLM on the syllabus?* Fine-tuning bakes knowledge into weights (no citations, costly to update, still hallucinates). RAG updates by re-indexing a PDF.

## Suggested ownership (adjust to your group; rubric needs a matching contribution statement)

| Member | Section | Must be able to explain |
|---|---|---|
| 1 | Ingestion and preprocessing (`ingest.py`) | loading, cleaning, chunk size/overlap, report |
| 2 | Embeddings and FAISS retrieval | embeddings, distance, k, index files |
| 3 | Prompting, LLM, citations (`rag_core.py`) | prompt rules, JSON contract, not-found handling |
| 4 | Evaluation and UI (`evaluate.py`, `metrics.py`, `app.py`) | metrics, test set, results, demo |
