"""Pure evaluation logic (no network, no LLM) so it can be unit-tested.

Outcome categories for every test question
------------------------------------------
in-scope question (answer IS in the PDFs)
    grounded_correct   answered, key facts present, citations shown, not flagged unfaithful
    incomplete_answer  answered but key facts missing or no citations
    false_refusal      bot said "could not find" although the answer exists
out-of-scope question (answer is NOT in the PDFs)
    correct_refusal    bot said "could not find"
hallucinated           out-of-scope question answered anyway, OR an answer the
                       judge flagged as containing unsupported claims
"""
from __future__ import annotations

import re
from typing import Optional

NOT_FOUND_PHRASES = [
    "could not find this information", "cannot find", "can't find", "unable to find",
    "not covered", "i don't know", "i do not know", "no information", "not mentioned",
    "not provided in", "does not contain",
]


def normalize(text: str) -> str:
    t = (text or "").lower()
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "-", "_"):
        t = t.replace(ch, " ")
    return re.sub(r"\s+", " ", t).strip()


def keyword_hits(text: str, keywords: list) -> int:
    t = normalize(text)
    return sum(1 for k in keywords if normalize(k) in t)


def looks_like_refusal(text: str) -> bool:
    t = normalize(text)
    return any(normalize(p) in t for p in NOT_FOUND_PHRASES)


def classify(row: dict) -> str:
    """Return the outcome category for one evaluated row."""
    in_scope = row["category"] == "in_scope"
    answered = row["answered"]
    faithful = row.get("faithful")          # True / False / None (judge off or failed)

    if in_scope:
        if not answered:
            return "false_refusal"
        if faithful is False:
            return "hallucinated"
        if row["keyword_pass"] and row["n_sources"] > 0:
            return "grounded_correct"
        return "incomplete_answer"
    return "hallucinated" if answered else "correct_refusal"


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def summarize(rows: list) -> dict:
    """Aggregate per-question rows into the headline numbers."""
    for r in rows:
        r["outcome"] = classify(r)

    in_rows = [r for r in rows if r["category"] == "in_scope"]
    out_rows = [r for r in rows if r["category"] == "out_of_scope"]
    counts = {k: 0 for k in ("grounded_correct", "incomplete_answer", "false_refusal",
                             "correct_refusal", "hallucinated")}
    for r in rows:
        counts[r["outcome"]] += 1

    # Refusal treated as the positive class: "should the bot refuse this question?"
    tp = sum(1 for r in out_rows if not r["answered"])
    fn = sum(1 for r in out_rows if r["answered"])
    fp = sum(1 for r in in_rows if not r["answered"])
    tn = sum(1 for r in in_rows if r["answered"])
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    answered_in = [r for r in in_rows if r["answered"]]
    judged = [r for r in answered_in if r.get("faithful") is not None]

    return {
        "n_total": len(rows),
        "n_in_scope": len(in_rows),
        "n_out_of_scope": len(out_rows),
        "outcome_counts": counts,
        "overall_accuracy": _ratio(counts["grounded_correct"] + counts["correct_refusal"], len(rows)),
        "in_scope": {
            "answer_rate": _ratio(len(answered_in), len(in_rows)),
            "keyword_pass_rate": _ratio(sum(1 for r in answered_in if r["keyword_pass"]), len(in_rows)),
            "retrieval_hit_rate": _ratio(sum(1 for r in in_rows if r["retrieval_hit"]), len(in_rows)),
            "citation_support_rate": _ratio(
                sum(1 for r in answered_in if r["citation_supported"]), len(answered_in)),
            "faithfulness_rate": _ratio(sum(1 for r in judged if r["faithful"]), len(judged)),
            "n_judged": len(judged),
        },
        "out_of_scope": {
            "correct_refusal_rate": _ratio(tp, len(out_rows)),
            "hallucination_rate": _ratio(fn, len(out_rows)),
        },
        "refusal_detection": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                              "precision": precision, "recall": recall, "f1": f1},
        "hallucination_rate_overall": _ratio(counts["hallucinated"], len(rows)),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3) if rows else None,
    }


def summarize_baseline(rows: list) -> dict:
    """Closed-book LLM (no retrieval). It can never cite the syllabus, so every
    answered question is 'ungrounded' by construction."""
    in_rows = [r for r in rows if r["category"] == "in_scope"]
    out_rows = [r for r in rows if r["category"] == "out_of_scope"]
    return {
        "in_scope_keyword_pass_rate": _ratio(sum(1 for r in in_rows if r["base_keyword_pass"]), len(in_rows)),
        "out_of_scope_answered_anyway_rate": _ratio(sum(1 for r in out_rows if not r["base_refused"]), len(out_rows)),
        "out_of_scope_refusal_rate": _ratio(sum(1 for r in out_rows if r["base_refused"]), len(out_rows)),
        "answers_with_syllabus_citation": 0.0,
    }


def distance_stats(rows: list) -> dict:
    """How well does 'top-1 FAISS distance' alone separate in-scope from out-of-scope?"""
    ins = [r["top_distance"] for r in rows if r["category"] == "in_scope" and r["top_distance"] is not None]
    outs = [r["top_distance"] for r in rows if r["category"] == "out_of_scope" and r["top_distance"] is not None]
    stats = {"in_scope": _minmeanmax(ins), "out_of_scope": _minmeanmax(outs)}
    if ins and outs:
        best_t, best_acc = None, -1.0
        for t in sorted(set(ins + outs)):
            correct = sum(1 for d in ins if d <= t) + sum(1 for d in outs if d > t)
            acc = correct / (len(ins) + len(outs))
            if acc > best_acc:
                best_t, best_acc = t, acc
        stats["best_threshold"] = round(best_t, 4)
        stats["best_threshold_accuracy"] = round(best_acc, 4)
    return stats


def _minmeanmax(xs: list):
    if not xs:
        return None
    return {"min": round(min(xs), 4), "mean": round(sum(xs) / len(xs), 4), "max": round(max(xs), 4)}


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_report(summary: dict, rows: list, baseline: Optional[dict], dist: dict, settings: dict) -> str:
    c = summary["outcome_counts"]
    L = []
    L.append("# Ask-the-Syllabus Bot - Evaluation Report\n")
    L.append("## Settings\n")
    for k, v in settings.items():
        L.append(f"- **{k}**: {v}")
    L.append("\n## Headline results\n")
    L.append(f"- Questions: **{summary['n_total']}** "
             f"({summary['n_in_scope']} in-scope, {summary['n_out_of_scope']} out-of-scope)")
    L.append(f"- Overall accuracy (grounded-correct + correct refusals): **{_pct(summary['overall_accuracy'])}**")
    L.append(f"- Hallucination rate (overall): **{_pct(summary['hallucination_rate_overall'])}**")
    L.append(f"- Average latency per question: **{summary['avg_latency_s']} s**\n")

    L.append("### Outcome breakdown\n")
    L.append("| Outcome | Count |\n|---|---|")
    for k in ("grounded_correct", "incomplete_answer", "false_refusal", "correct_refusal", "hallucinated"):
        L.append(f"| {k} | {c[k]} |")

    i, o, rd = summary["in_scope"], summary["out_of_scope"], summary["refusal_detection"]
    L.append("\n### In-scope questions (answer exists in the PDFs)\n")
    L.append("| Metric | Value |\n|---|---|")
    L.append(f"| Answer rate | {_pct(i['answer_rate'])} |")
    L.append(f"| Key-fact pass rate | {_pct(i['keyword_pass_rate'])} |")
    L.append(f"| Retrieval hit rate (key facts present in retrieved chunks) | {_pct(i['retrieval_hit_rate'])} |")
    L.append(f"| Citation support rate (cited chunks contain key facts) | {_pct(i['citation_support_rate'])} |")
    L.append(f"| Faithfulness (LLM judge, n={i['n_judged']}) | {_pct(i['faithfulness_rate'])} |")

    L.append("\n### Out-of-scope questions (answer NOT in the PDFs)\n")
    L.append("| Metric | Value |\n|---|---|")
    L.append(f"| Correct refusal rate | {_pct(o['correct_refusal_rate'])} |")
    L.append(f"| Hallucination rate | {_pct(o['hallucination_rate'])} |")

    L.append("\n### Refusal detection (positive class = 'should refuse')\n")
    L.append("| TP | FP | FN | TN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|")
    L.append(f"| {rd['tp']} | {rd['fp']} | {rd['fn']} | {rd['tn']} | "
             f"{_pct(rd['precision'])} | {_pct(rd['recall'])} | {_pct(rd['f1'])} |")

    if baseline:
        L.append("\n## RAG vs closed-book LLM (no retrieval)\n")
        L.append("| Metric | Closed-book LLM | RAG bot |\n|---|---|---|")
        L.append(f"| In-scope key-fact pass rate | {_pct(baseline['in_scope_keyword_pass_rate'])} | {_pct(i['keyword_pass_rate'])} |")
        L.append(f"| Out-of-scope questions answered anyway | {_pct(baseline['out_of_scope_answered_anyway_rate'])} | {_pct(o['hallucination_rate'])} |")
        L.append("| Answers carrying a syllabus citation | 0% (impossible) | see citation support rate |")

    L.append("\n## Does retrieval distance alone separate in-scope from out-of-scope?\n")
    L.append(f"- Top-1 FAISS L2 distance, in-scope: `{dist.get('in_scope')}`")
    L.append(f"- Top-1 FAISS L2 distance, out-of-scope: `{dist.get('out_of_scope')}`")
    if "best_threshold" in dist:
        L.append(f"- Best single distance threshold: `{dist['best_threshold']}` "
                 f"-> accuracy `{_pct(dist['best_threshold_accuracy'])}`")
    L.append("- Interpretation: if the two ranges overlap, a distance cut-off alone cannot decide "
             "'answerable or not'; that is why the bot lets the LLM abstain (`found: false`).")

    bad = [r for r in rows if r["outcome"] not in ("grounded_correct", "correct_refusal")]
    L.append("\n## Failure cases (inspect these for your limitations slide)\n")
    if not bad:
        L.append("_None._")
    for r in bad:
        L.append(f"- **[{r['id']}] {r['outcome']}** - {r['question']}  \n"
                 f"  answered={r['answered']}, key-facts={r['keyword_hits']}/{r['min_keywords']} needed, "
                 f"top distance={r['top_distance']}")
    return "\n".join(L) + "\n"
