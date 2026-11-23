"""Core RAG logic: retrieve -> build prompt -> generate -> verify citations.

Design points
-------------
* The LLM sees NUMBERED chunks and must reply with JSON:
      {"found": bool, "answer": str, "used_chunks": [ints]}
* Sources shown to the user are ONLY the chunks the model says it used.
  If found == false, no sources are shown (fixes the "not found + citations" bug).
* All heavy imports (langchain, faiss, torch) are lazy so the pure logic in this
  file can be unit-tested without a network or GPU.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import config

NOT_FOUND_MSG = "I could not find this information in the provided syllabus."

PROMPT_TEMPLATE = """You are an academic assistant for the MIT ADT University Natural Language Processing (NLP) course material.

Answer the question using ONLY the numbered context chunks below. Do not use outside knowledge.

Rules:
1. Use only facts explicitly stated in the chunks. Do not add examples or definitions from your own knowledge.
2. If the chunks do not contain the answer, set "found" to false. Do not guess. A chunk that merely shares a keyword with the question but does not answer it does not count.
3. Keep the answer concise; use bullet points for lists of topics.
4. In "used_chunks" list the numbers of the chunks you actually used.

Respond with ONLY a JSON object (no markdown fences) in exactly this shape:
{{"found": true or false, "answer": "<answer text, or empty string if not found>", "used_chunks": [<chunk numbers>]}}

Context:
{context}

Question: {question}
"""

BASELINE_PROMPT = "Answer the following question concisely.\n\nQuestion: {question}\n"

JUDGE_PROMPT = """You are a strict fact-checker. Below are context chunks and an answer.
Decide whether EVERY factual claim in the answer is supported by the context chunks.

Context:
{context}

Answer:
{answer}

Respond with ONLY a JSON object (no markdown fences):
{{"supported": true or false, "unsupported_claims": "<short note, or empty string>"}}
"""


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Source:
    file: str
    page: Any
    text: str
    distance: float
    chunk_no: int = 0          # 1-based position in the numbered context

    @property
    def snippet(self) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= 300 else t[:297] + "..."


@dataclass
class RAGResult:
    question: str
    answer: str
    found: bool
    sources: list = field(default_factory=list)     # chunks to show as citations
    retrieved: list = field(default_factory=list)   # all chunks retrieved (for debugging/eval)
    citations_verified: bool = True                  # False = model gave no usable citation numbers
    latency_s: float = 0.0
    raw_output: str = ""
    gated_by_distance: bool = False


# --------------------------------------------------------------------------
# Pure helper functions (unit-tested)
# --------------------------------------------------------------------------
def format_context(sources: list) -> str:
    parts = []
    for s in sources:
        parts.append(f"[{s.chunk_no}] ({s.file}, page {s.page})\n{s.text.strip()}")
    return "\n\n".join(parts)


def parse_model_output(text: Optional[str]) -> Optional[dict]:
    """Parse the model's JSON reply. Returns a normalised dict or None if unparseable."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1], strict=False)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "found" not in obj:
        return None

    found = obj.get("found")
    if isinstance(found, str):
        found = found.strip().lower() == "true"
    answer = obj.get("answer", "")
    answer = answer if isinstance(answer, str) else str(answer)

    used = []
    for u in obj.get("used_chunks", []) or []:
        try:
            used.append(int(u))
        except (TypeError, ValueError):
            continue
    return {"found": bool(found), "answer": answer.strip(), "used_chunks": used}


def dedupe_sources(sources: list) -> list:
    """Keep the first chunk for each (file, page)."""
    seen, out = set(), []
    for s in sources:
        key = (s.file, str(s.page))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _to_source(doc, score, chunk_no) -> Source:
    meta = getattr(doc, "metadata", {}) or {}
    page = meta.get("page_label", meta.get("page", "?"))
    return Source(
        file=meta.get("source", "unknown"),
        page=page,
        text=doc.page_content,
        distance=float(score),
        chunk_no=chunk_no,
    )


def _not_found(question, retrieved, t0, raw="", gated=False) -> RAGResult:
    return RAGResult(
        question=question, answer=NOT_FOUND_MSG, found=False, sources=[],
        retrieved=retrieved, citations_verified=True,
        latency_s=time.perf_counter() - t0, raw_output=raw, gated_by_distance=gated,
    )


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def answer_question(question: str, vectorstore, llm, k: int = None,
                    max_distance: Optional[float] = "config") -> RAGResult:
    """Run the full RAG pipeline. `vectorstore` and `llm` are injected so tests can fake them."""
    k = k or config.TOP_K
    if max_distance == "config":
        max_distance = config.MAX_DISTANCE
    t0 = time.perf_counter()

    # 1. Retrieve
    hits = vectorstore.similarity_search_with_score(question, k=k)
    retrieved = [_to_source(doc, score, i) for i, (doc, score) in enumerate(hits, 1)]
    if not retrieved:
        return _not_found(question, retrieved, t0)

    # 2. Optional distance gate (off by default; see README "Findings")
    if max_distance is not None and min(s.distance for s in retrieved) > max_distance:
        return _not_found(question, retrieved, t0, gated=True)

    # 3. Generate
    prompt = PROMPT_TEMPLATE.format(context=format_context(retrieved), question=question)
    raw = getattr(llm.invoke(prompt), "content", "") or ""

    # 4. Interpret the output
    parsed = parse_model_output(raw)
    if parsed is None:
        # Model ignored the JSON format. Fall back to plain-text handling.
        if NOT_FOUND_MSG.lower() in raw.lower():
            return _not_found(question, retrieved, t0, raw)
        return RAGResult(
            question=question, answer=raw.strip(), found=bool(raw.strip()),
            sources=dedupe_sources(retrieved), retrieved=retrieved,
            citations_verified=False, latency_s=time.perf_counter() - t0, raw_output=raw,
        )

    if not parsed["found"] or not parsed["answer"]:
        return _not_found(question, retrieved, t0, raw)

    cited = [s for s in retrieved if s.chunk_no in set(parsed["used_chunks"])]
    verified = bool(cited)
    sources = dedupe_sources(cited if verified else retrieved)
    return RAGResult(
        question=question, answer=parsed["answer"], found=True, sources=sources,
        retrieved=retrieved, citations_verified=verified,
        latency_s=time.perf_counter() - t0, raw_output=raw,
    )


def baseline_answer(question: str, llm) -> str:
    """No-retrieval baseline: the LLM answers from its own memory (closed book)."""
    return getattr(llm.invoke(BASELINE_PROMPT.format(question=question)), "content", "") or ""


def judge_faithfulness(answer: str, sources: list, llm) -> Optional[bool]:
    """LLM-as-judge: is every claim in `answer` supported by `sources`? None if judge failed."""
    prompt = JUDGE_PROMPT.format(context=format_context(sources), answer=answer)
    raw = getattr(llm.invoke(prompt), "content", "") or ""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(t[s:e + 1], strict=False)
    except json.JSONDecodeError:
        return None
    sup = obj.get("supported")
    if isinstance(sup, str):
        sup = sup.strip().lower() == "true"
    return bool(sup) if sup is not None else None


# --------------------------------------------------------------------------
# Component loaders (lazy imports)
# --------------------------------------------------------------------------
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore():
    from langchain_community.vectorstores import FAISS
    if not (config.INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "Vector index not found. Put your PDFs in data/pdfs/ and run:  python ingest.py"
        )
    return FAISS.load_local(
        str(config.INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def get_llm():
    from langchain_groq import ChatGroq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGroq(model=config.GROQ_MODEL, temperature=config.TEMPERATURE, api_key=key)
