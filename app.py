"""Streamlit demo:  streamlit run app.py   ->  http://localhost:8501"""
import streamlit as st

import config
import rag_core

st.set_page_config(page_title="Ask the Syllabus", layout="centered")

EXAMPLES = [
    "What topics are covered in Unit II?",
    "What is POS tagging?",
    "What is the difference between Earley parsing and CKY parsing?",
    "What are transition and emission probabilities?",
    "What is quantum computing?",
]


@st.cache_resource(show_spinner="Loading vector index and models...")
def load_components():
    return rag_core.load_vectorstore(), rag_core.get_llm()


try:
    vectorstore, llm = load_components()
except (FileNotFoundError, RuntimeError) as e:
    st.error(str(e))
    st.stop()

# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. PDFs are split into chunks\n"
        "2. Each chunk becomes an embedding (meaning vector)\n"
        "3. FAISS finds the chunks closest to your question\n"
        "4. The LLM answers **only** from those chunks\n"
        "5. Only chunks the answer used are shown as sources"
    )
    st.divider()
    k = st.slider("Chunks retrieved (k)", 2, 8, config.TOP_K)
    debug = st.checkbox("Show all retrieved chunks (debug)")
    st.divider()
    st.caption(f"Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]}")
    st.caption(f"LLM: {config.GROQ_MODEL} (temperature {config.TEMPERATURE})")
    st.caption(f"Vectors in index: {vectorstore.index.ntotal}")
    st.divider()
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["question"] = ex
            st.session_state["run_now"] = True

# ---------------- main ----------------
st.title("Ask the Syllabus")
st.caption("Natural Language Processing course assistant | MIT ADT University")

with st.form("ask_form"):
    question = st.text_input("Ask a question about the syllabus", key="question",
                             placeholder="e.g. What is CKY parsing?")
    submitted = st.form_submit_button("Ask", type="primary")

if "history" not in st.session_state:
    st.session_state["history"] = []

if submitted or st.session_state.pop("run_now", False):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the syllabus..."):
            try:
                result = rag_core.answer_question(question.strip(), vectorstore, llm, k=k)
            except Exception as e:  # network / rate-limit problems during the demo
                st.error(f"The language model call failed: {e}")
                st.stop()
        st.session_state["history"].insert(0, result)

        st.subheader("Answer")
        if result.found:
            with st.container(border=True):
                st.markdown(result.answer)
            st.subheader("Sources")
            if not result.citations_verified:
                st.caption("The model did not say which chunks it used, "
                           "so these are all retrieved chunks (unverified).")
            for s in result.sources:
                with st.expander(f"{s.file}  |  page {s.page}"):
                    st.write(s.snippet)
        else:
            st.warning(result.answer)
            st.caption("No sources are shown because none of the retrieved chunks answered the question.")

        st.caption(f"Answered in {result.latency_s:.1f}s")

        if debug:
            with st.expander("Retrieved chunks (debug)", expanded=True):
                for s in result.retrieved:
                    st.markdown(f"**[{s.chunk_no}] {s.file} p.{s.page}** - L2 distance `{s.distance:.3f}`")
                    st.caption(s.snippet)

if len(st.session_state["history"]) > 1:
    with st.expander("Earlier questions this session"):
        for r in st.session_state["history"][1:]:
            st.markdown(f"**Q:** {r.question}")
            st.markdown(r.answer)
            st.divider()
