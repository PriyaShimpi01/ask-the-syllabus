"""Offline unit tests (no network, no LLM, no FAISS). Run:  python -m unittest discover -s tests -v"""
import json
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics
import rag_core
from rag_core import NOT_FOUND_MSG


class FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def similarity_search_with_score(self, q, k=4):
        return self.hits[:k]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.reply)


def doc(text, page, source="Unit II.pdf"):
    return SimpleNamespace(page_content=text, metadata={"source": source, "page_label": str(page)})


HITS = [
    (doc("POS tagging assigns a part of speech tag to each word.", 21), 0.80),
    (doc("HMM taggers use transition and emission probabilities.", 24), 0.95),
    (doc("Generic NLP system overview.", 10, "Unit_I.pdf"), 1.30),
]


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        p = rag_core.parse_model_output('{"found": true, "answer": "x", "used_chunks": [1, 2]}')
        self.assertEqual(p, {"found": True, "answer": "x", "used_chunks": [1, 2]})

    def test_fenced_json_and_string_bool(self):
        p = rag_core.parse_model_output('```json\n{"found": "false", "answer": "", "used_chunks": []}\n```')
        self.assertFalse(p["found"])

    def test_raw_newline_inside_string(self):
        p = rag_core.parse_model_output('{"found": true, "answer": "a\nb", "used_chunks": ["1"]}')
        self.assertEqual(p["used_chunks"], [1])

    def test_garbage_returns_none(self):
        self.assertIsNone(rag_core.parse_model_output("Sure! POS tagging is..."))
        self.assertIsNone(rag_core.parse_model_output(""))
        self.assertIsNone(rag_core.parse_model_output('{"answer": "no found key"}'))


class PipelineTests(unittest.TestCase):
    def test_answer_shows_only_cited_sources(self):
        llm = FakeLLM('{"found": true, "answer": "POS tagging labels words.", "used_chunks": [1]}')
        r = rag_core.answer_question("What is POS tagging?", FakeStore(HITS), llm, k=3, max_distance=None)
        self.assertTrue(r.found)
        self.assertEqual([(s.file, s.page) for s in r.sources], [("Unit II.pdf", "21")])
        self.assertEqual(len(r.retrieved), 3)
        self.assertTrue(r.citations_verified)

    def test_not_found_has_no_sources(self):
        """Regression test for the bug: 'I could not find...' must not list citations."""
        llm = FakeLLM('{"found": false, "answer": "", "used_chunks": []}')
        r = rag_core.answer_question("What is quantum computing?", FakeStore(HITS), llm, k=3, max_distance=None)
        self.assertFalse(r.found)
        self.assertEqual(r.answer, NOT_FOUND_MSG)
        self.assertEqual(r.sources, [])
        self.assertEqual(len(r.retrieved), 3)        # still available for debugging

    def test_plain_text_refusal_fallback(self):
        r = rag_core.answer_question("q", FakeStore(HITS), FakeLLM(NOT_FOUND_MSG), k=3, max_distance=None)
        self.assertFalse(r.found)
        self.assertEqual(r.sources, [])

    def test_unparseable_answer_falls_back_to_unverified_sources(self):
        r = rag_core.answer_question("q", FakeStore(HITS), FakeLLM("POS tagging is tagging."), k=3, max_distance=None)
        self.assertTrue(r.found)
        self.assertFalse(r.citations_verified)
        self.assertEqual(len(r.sources), 3)

    def test_found_but_no_valid_chunk_numbers_is_unverified(self):
        llm = FakeLLM('{"found": true, "answer": "x", "used_chunks": [9]}')
        r = rag_core.answer_question("q", FakeStore(HITS), llm, k=3, max_distance=None)
        self.assertTrue(r.found)
        self.assertFalse(r.citations_verified)

    def test_distance_gate_skips_llm(self):
        llm = FakeLLM("should not be called")
        r = rag_core.answer_question("q", FakeStore(HITS), llm, k=3, max_distance=0.5)
        self.assertFalse(r.found)
        self.assertTrue(r.gated_by_distance)
        self.assertEqual(llm.calls, 0)

    def test_empty_retrieval(self):
        r = rag_core.answer_question("q", FakeStore([]), FakeLLM("x"), k=3, max_distance=None)
        self.assertFalse(r.found)

    def test_dedupe_sources_by_page(self):
        same_page = [(doc("a", 5), 0.1), (doc("b", 5), 0.2), (doc("c", 6), 0.3)]
        srcs = [rag_core._to_source(d, s, i) for i, (d, s) in enumerate(same_page, 1)]
        self.assertEqual(len(rag_core.dedupe_sources(srcs)), 2)

    def test_judge_parsing(self):
        srcs = [rag_core._to_source(*HITS[0], 1)]
        self.assertTrue(rag_core.judge_faithfulness("a", srcs, FakeLLM('{"supported": true}')))
        self.assertFalse(rag_core.judge_faithfulness("a", srcs, FakeLLM('{"supported": "false"}')))
        self.assertIsNone(rag_core.judge_faithfulness("a", srcs, FakeLLM("no json")))


class MetricsTests(unittest.TestCase):
    def test_keyword_hits_normalises_dashes_and_case(self):
        self.assertEqual(metrics.keyword_hits("Part\u2011of\u2011speech tagging", ["part of speech", "TAG"]), 2)

    def test_refusal_detection(self):
        self.assertTrue(metrics.looks_like_refusal(NOT_FOUND_MSG))
        self.assertFalse(metrics.looks_like_refusal("Quantum computing uses qubits."))

    def _row(self, cat, answered, kp=False, ns=0, faithful=None, dist=1.0):
        return {"category": cat, "answered": answered, "keyword_pass": kp, "n_sources": ns,
                "faithful": faithful, "retrieval_hit": answered, "citation_supported": kp,
                "top_distance": dist, "latency_s": 1.0, "id": "x", "question": "q",
                "keyword_hits": 0, "min_keywords": 1}

    def test_classify_all_outcomes(self):
        c = metrics.classify
        self.assertEqual(c(self._row("in_scope", True, True, 2)), "grounded_correct")
        self.assertEqual(c(self._row("in_scope", True, False, 2)), "incomplete_answer")
        self.assertEqual(c(self._row("in_scope", True, True, 2, faithful=False)), "hallucinated")
        self.assertEqual(c(self._row("in_scope", False)), "false_refusal")
        self.assertEqual(c(self._row("out_of_scope", False)), "correct_refusal")
        self.assertEqual(c(self._row("out_of_scope", True, ns=1)), "hallucinated")

    def test_summary_numbers(self):
        rows = [
            self._row("in_scope", True, True, 1, dist=0.8),
            self._row("in_scope", True, True, 1, dist=0.9),
            self._row("in_scope", False, dist=1.2),                  # false refusal
            self._row("out_of_scope", False, dist=1.3),              # correct refusal
            self._row("out_of_scope", True, ns=1, dist=1.1),         # hallucination
        ]
        s = metrics.summarize(rows)
        self.assertEqual(s["outcome_counts"]["grounded_correct"], 2)
        self.assertEqual(s["overall_accuracy"], round(3 / 5, 4))
        rd = s["refusal_detection"]
        self.assertEqual((rd["tp"], rd["fp"], rd["fn"], rd["tn"]), (1, 1, 1, 2))
        self.assertEqual(rd["precision"], 0.5)
        self.assertEqual(rd["recall"], 0.5)
        self.assertEqual(s["out_of_scope"]["hallucination_rate"], 0.5)

    def test_distance_stats_and_report_render(self):
        rows = [self._row("in_scope", True, True, 1, dist=0.8), self._row("out_of_scope", False, dist=1.4)]
        s = metrics.summarize(rows)
        d = metrics.distance_stats(rows)
        self.assertEqual(d["best_threshold_accuracy"], 1.0)
        md = metrics.render_report(s, rows, None, d, {"llm": "test"})
        self.assertIn("Headline results", md)


class TestSetTests(unittest.TestCase):
    def test_test_set_is_well_formed(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "test_set.json")
        cases = json.load(open(path, encoding="utf-8"))
        ids = [c["id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)))
        for c in cases:
            self.assertIn(c["category"], ("in_scope", "out_of_scope"))
            if c["category"] == "in_scope":
                self.assertTrue(c["expected_keywords"])
                self.assertLessEqual(c["min_keywords"], len(c["expected_keywords"]))


if __name__ == "__main__":
    unittest.main()
