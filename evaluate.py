"""Step 4 of the project: measure grounded vs hallucinated answers on a labelled test set.

Usage:
    python evaluate.py                      # RAG bot only
    python evaluate.py --judge --baseline   # + LLM faithfulness judge + closed-book comparison
Outputs (results/):  eval_results.csv, eval_summary.json, eval_report.md
"""
import argparse
import csv
import json
import time

import config
import metrics
import rag_core


def with_retry(fn, tries=3, wait=5):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:                       # rate limits, network blips
            if attempt == tries:
                raise
            print(f"   retry {attempt}/{tries - 1} after error: {e}")
            time.sleep(wait * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=str(config.TESTSET_PATH))
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness check")
    ap.add_argument("--baseline", action="store_true", help="also run the closed-book LLM")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between questions (rate limits)")
    args = ap.parse_args()

    cases = json.load(open(args.testset, encoding="utf-8"))
    vs, llm = rag_core.load_vectorstore(), rag_core.get_llm()
    rows = []

    for n, case in enumerate(cases, 1):
        q, kws, mn = case["question"], case["expected_keywords"], case["min_keywords"]
        print(f"[{n}/{len(cases)}] {case['id']}: {q}")
        res = with_retry(lambda: rag_core.answer_question(q, vs, llm, k=args.k))

        context_all = " ".join(s.text for s in res.retrieved)
        cited_text = " ".join(s.text for s in res.sources)
        row = {
            "id": case["id"], "category": case["category"], "question": q, "note": case.get("note", ""),
            "answered": res.found, "answer": res.answer,
            "sources": "; ".join(f"{s.file} p.{s.page}" for s in res.sources),
            "n_sources": len(res.sources), "citations_verified": res.citations_verified,
            "keyword_hits": metrics.keyword_hits(res.answer, kws) if res.found else 0,
            "min_keywords": mn,
            "retrieval_hit": metrics.keyword_hits(context_all, kws) >= mn if mn else False,
            "citation_supported": metrics.keyword_hits(cited_text, kws) >= 1 if kws else False,
            "top_distance": round(min((s.distance for s in res.retrieved), default=0.0), 4) if res.retrieved else None,
            "latency_s": round(res.latency_s, 3), "faithful": None,
        }
        row["keyword_pass"] = bool(res.found and mn and row["keyword_hits"] >= mn)

        if args.judge and res.found:
            row["faithful"] = with_retry(lambda: rag_core.judge_faithfulness(res.answer, res.retrieved, llm))

        if args.baseline:
            base = with_retry(lambda: rag_core.baseline_answer(q, llm))
            row["base_answer"] = base
            row["base_refused"] = metrics.looks_like_refusal(base)
            row["base_keyword_pass"] = bool(mn and metrics.keyword_hits(base, kws) >= mn)

        rows.append(row)
        time.sleep(args.sleep)

    summary = metrics.summarize(rows)
    baseline = metrics.summarize_baseline(rows) if args.baseline else None
    dist = metrics.distance_stats(rows)
    settings = {"embedding_model": config.EMBEDDING_MODEL, "llm": config.GROQ_MODEL,
                "temperature": config.TEMPERATURE, "top_k": args.k,
                "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                "llm_judge": args.judge, "max_distance_gate": config.MAX_DISTANCE,
                "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(config.RESULTS_DIR / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "summary": summary, "baseline": baseline, "distance": dist}, f, indent=2)
    report = metrics.render_report(summary, rows, baseline, dist, settings)
    (config.RESULTS_DIR / "eval_report.md").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"Saved to {config.RESULTS_DIR}/ (eval_results.csv, eval_summary.json, eval_report.md)")


if __name__ == "__main__":
    main()
