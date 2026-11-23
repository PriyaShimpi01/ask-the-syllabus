# Ask-the-Syllabus Bot (RAG)

A Retrieval-Augmented Generation Q&A app over the NLP course PDFs. It answers only from the
syllabus, shows page-level citations, and says "I could not find this information" (with no
sources) when the answer is not in the documents.

**Stack:** Python, LangChain, `sentence-transformers/all-MiniLM-L6-v2` (embeddings), FAISS (vector DB),
Groq `openai/gpt-oss-20b` (LLM), Streamlit (UI).

## Project structure

```
ask-the-syllabus/
├── data/pdfs/              <- PUT Unit_I.pdf and "Unit II.pdf" HERE (the dataset)
├── data/processed/         <- chunks.jsonl + ingestion_report.json (created by ingest.py)
├── faiss_index/            <- saved vector store: the "model file" (created by ingest.py)
├── config.py               <- every setting in one place
├── ingest.py               <- PDFs -> clean -> chunk -> embed -> FAISS
├── rag_core.py             <- retrieve -> prompt -> generate -> verify citations
├── app.py                  <- Streamlit demo (localhost)
├── metrics.py              <- evaluation maths (unit-tested)
├── evaluate.py             <- runs the labelled test set, writes results/
├── tests/test_set.json     <- 25 labelled questions (16 in-scope, 9 out-of-scope)
├── tests/test_logic.py     <- 19 offline unit tests
├── results/                <- eval_results.csv, eval_summary.json, eval_report.md
└── docs/                   <- ARCHITECTURE.md, VIVA_PREP.md
```

## Setup (about 10 minutes, once)

```bash
python -m venv venv
venv\Scripts\activate            # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env
# open .env and paste your GROQ_API_KEY  (free key: https://console.groq.com)
```

Copy your two PDFs into `data/pdfs/`.

## Run

```bash
python ingest.py                 # 1. build the index (first run downloads the ~90 MB embedding model)
streamlit run app.py             # 2. demo at http://localhost:8501
python evaluate.py --judge --baseline     # 3. testing + performance evaluation -> results/
python -m unittest discover -s tests -v   # 4. offline unit tests (no internet needed)
```

## Preprocessing (what `ingest.py` does)

1. Extract text page by page with `PyPDFLoader`; keep file name + page number as metadata.
2. Clean: remove null bytes, collapse repeated spaces and blank lines.
3. Skip pages with no extractable text (image-only slides) and report how many.
4. Split with `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap).
5. Drop chunks shorter than 40 characters (title-only slides such as "NLP Phases" that match many queries but answer nothing).
6. Embed every chunk (384-dim, normalised) and store in FAISS.

`data/processed/ingestion_report.json` records the exact counts from your run.

## Model files

No model is trained in this project. Embeddings (MiniLM) and the LLM (gpt-oss-20b) are pretrained. The
trained-artifact equivalent is **`faiss_index/`** (`index.faiss` + `index.pkl`), produced by `ingest.py`.

## Testing and evaluation

`evaluate.py` runs `tests/test_set.json` and reports, per question and overall:

| Metric | Meaning |
|---|---|
| Correct refusal rate / hallucination rate | out-of-scope questions refused vs answered anyway |
| Key-fact pass rate | in-scope answers containing the expected facts |
| Retrieval hit rate | expected facts present in the retrieved chunks (retrieval quality alone) |
| Citation support rate | cited chunks actually contain the expected facts |
| Faithfulness (`--judge`) | LLM judge: is every claim supported by the retrieved chunks |
| Refusal precision / recall / F1 | quality of the "should I refuse?" decision |
| Closed-book comparison (`--baseline`) | same LLM with no retrieval, for the "why RAG" argument |

**Before you evaluate:** the expected keywords in `tests/test_set.json` were written from the syllabus
text visible in the Colab notebook. Open your PDFs and confirm each in-scope question really is answered
there (the `note` field marks the ones to double-check). Questions tagged `near_domain_VERIFY...` must truly
be absent from your PDFs, otherwise move them to `in_scope`.

**All numbers on your slides must come from your own `results/eval_report.md`.**

## Key design decisions

* **Citations come from the model, not from retrieval.** The prompt numbers the chunks and the model
  returns `used_chunks`. Only those are shown. (Original notebook bug: it listed all 4 retrieved chunks even
  when the answer was "not found".)
* **Abstention is done by the LLM, not by a distance cut-off.** FAISS always returns k results. In the notebook,
  "What is quantum computing?" scored a top distance of 1.31, closer than "What topics are covered in Unit II?"
  (1.34), so a distance threshold cannot separate them. `evaluate.py` measures this on your data.
  Set `MAX_DISTANCE` in `.env` only if the report shows a clean gap.

## Known limitations (say these before the jury does)

* Image-only slides and diagrams are not searchable (no OCR).
* Slide text is short and fragmentary, so some definitions are thin; answers cannot exceed the slide content.
* Lexical overlap can fool retrieval (e.g. "computer" in a quantum question matches NLP definitions).
* Keyword-based scoring is an approximation; the LLM judge is also an LLM and can be wrong.
* Depends on the Groq API (internet + rate limits). Keep a recorded demo as backup.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Vector index not found` | run `python ingest.py` first |
| `GROQ_API_KEY is not set` | create `.env` from `.env.example` |
| Groq rate limit / 429 | wait a minute; `evaluate.py --sleep 3` |
| `faiss` install error on Windows | use Python 3.10-3.12; `pip install faiss-cpu` |
| Port in use | `streamlit run app.py --server.port 8502` |

## Demo script (5 minutes)

1. Show `data/pdfs/` and `ingestion_report.json` (dataset + preprocessing).
2. App: "What topics are covered in Unit II?" then expand a source.
3. App: "What is the difference between Earley parsing and CKY parsing?"
4. App: "What is quantum computing?" -> "could not find", no sources. Tick *debug* to show what was retrieved.
5. Show `results/eval_report.md`: grounded vs hallucinated, closed-book comparison, failure cases.
6. **Record this whole run as a video backup** (mandatory for Model Making).
