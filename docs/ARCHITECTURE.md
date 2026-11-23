# Architecture

## Offline (run once): `ingest.py`

```
PDFs (Unit_I, Unit II)
   -> PyPDFLoader            text + {file, page} per page
   -> clean_text             whitespace fixes, skip empty pages
   -> RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
   -> drop chunks < 40 chars
   -> all-MiniLM-L6-v2       each chunk -> 384-number vector (normalised)
   -> FAISS index            saved to faiss_index/
```

## Online (every question): `rag_core.answer_question()`

```
Question
   -> embed with the SAME MiniLM model
   -> FAISS top-k nearest chunks (L2 distance)            [RETRIEVAL]
   -> prompt = rules + numbered chunks [1]..[k] + question
   -> gpt-oss-20b, temperature 0                           [GENERATION]
   -> JSON: {found, answer, used_chunks}
   -> found=false  -> "I could not find this information", NO sources
      found=true   -> answer + only the cited chunks (file + page)
```

## Why each part exists

| Part | Reason |
|---|---|
| Chunking with overlap | precise retrieval; an idea cut at a boundary still appears whole in the neighbour |
| Embeddings | match by meaning, not exact words |
| FAISS (exact L2 search) | fast nearest-neighbour search; 156 vectors is small enough for exact search |
| "Use ONLY the chunks" prompt | turns recall from memory into reading comprehension |
| `found` flag | lets the model abstain instead of guessing |
| `used_chunks` | citations reflect what the answer actually used |
| Temperature 0 | deterministic, literal output (no sampling randomness) |

## Mapping to Generative AI syllabus concepts (for the mandatory slide)

* **Latent space / embeddings:** chunks and questions live in the same 384-dim vector space; similarity = distance.
* **Prompt engineering:** the grounding rules, numbered context and JSON output contract.
* **Transformers & LLMs:** MiniLM is a transformer encoder (embeddings); gpt-oss-20b is a transformer decoder (generation).
* **Hallucination and its mitigation:** retrieval grounding, abstention, citation checking.
* **Probability / sampling:** temperature 0 means greedy decoding (always the most likely next token).
