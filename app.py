"""
Streamlit chat UI for the RAG assistant - a visual front end over the same
retrieve() + generate_answer() functions src/ask.py uses on the CLI.

    streamlit run app.py

(Must be run with the project's venv - e.g. venv\\Scripts\\streamlit.exe -
not a system-wide Python, or none of the project's dependencies resolve.)
"""

from pathlib import Path

import streamlit as st

from src import config
from src.chat_log import append_turn, clear_history, load_recent_turns
from src.generate import generate_answer
from src.ingest import rebuild_index
from src.memory import condense_question, extract_history
from src.retrieve import list_sources, reset_vector_store, retrieve

MY_DOCS_DIR = config.DATA_DIR / "my_docs"

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Same design language as pages/1_📊_Evaluation_Dashboard.py: one accent
# color, card-style chat bubbles, accent-underlined section headers instead
# of plain st.divider() lines.
ACCENT = "#7C9EFF"

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2.5rem; max-width: 800px; }}
    [data-testid="stChatMessage"] {{
        border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 0.25rem;
        border: 1px solid rgba(255,255,255,0.06);
    }}
    .status-pill {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: rgba(124, 158, 255, 0.15); color: {ACCENT};
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }}
    .section-title {{
        display: flex; align-items: center; gap: 10px; margin-bottom: 2px;
    }}
    .section-title .icon {{ font-size: 1.1rem; }}
    .section-title .text {{ font-size: 1rem; font-weight: 700; color: #F0F1F5; }}
    .section-rule {{
        height: 3px; width: 44px; background: {ACCENT};
        border-radius: 2px; margin: 6px 0 18px 0;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        border-color: {ACCENT}; color: {ACCENT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-title"><span class="icon">{icon}</span>'
        f'<span class="text">{title}</span></div>'
        f'<div class="section-rule"></div>',
        unsafe_allow_html=True,
    )


st.title("📚 Document Assistant")
st.markdown(
    f'<span class="status-pill">🦙 {config.CHAT_MODEL}</span>'
    f'<span class="status-pill">🧭 {config.EMBEDDING_MODEL.split("/")[-1]}</span>'
    f'<span class="status-pill">top_k={config.TOP_K}</span>'
    f'<span class="status-pill">🔒 fully local</span>',
    unsafe_allow_html=True,
)
st.caption(
    "Ask a question about the documents in `data/`. See the "
    "**📊 Evaluation Dashboard** page in the sidebar for how well this "
    "assistant is actually measured to perform."
)
st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

# --- Document scope picker ---
# Fixes a real failure mode: a vague question like "tell me about this
# project" is ambiguous across a multi-document corpus (there's no other
# way to say which document "this" refers to), and can retrieve chunks
# from the wrong document entirely (e.g. boilerplate acknowledgement text
# that happens to repeat the word "project", instead of the actual
# document you meant). Narrowing to one document sidesteps both problems.
all_sources = list_sources()
_name_counts: dict[str, int] = {}
for _s in all_sources:
    _name_counts[Path(_s).name] = _name_counts.get(Path(_s).name, 0) + 1


def _display_label(path_str: str) -> str:
    name = Path(path_str).name
    if _name_counts[name] > 1:  # disambiguate same-named files in different folders
        return f"{name} ({Path(path_str).parent.name})"
    return name


path_by_label = {_display_label(s): s for s in all_sources}
scope_options = ["All documents"] + sorted(path_by_label.keys())
selected_label = st.selectbox(
    "🔎 Scope to a document",
    scope_options,
    help=(
        "Narrow retrieval to one document instead of searching the whole "
        "corpus. Fixes ambiguous questions like \"tell me about this "
        "project\" when multiple documents are indexed."
    ),
)
selected_source = path_by_label.get(selected_label)  # None means "All documents"

if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role", "content", "sources": [...]}

# Replay chat history on every rerun (Streamlit re-executes top to bottom).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source chunk(s)"):
                for i, chunk in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}] `{chunk['source']}`**")
                    st.text(chunk["snippet"])

question = st.chat_input("Ask a question about your documents...")

if question:
    # Snapshot *before* appending the current question, so a follow-up is
    # never resolved against itself.
    history = extract_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks..."):
            # When scoped to one document, retrieving more chunks is cheap
            # (no risk of diluting across dozens of other documents) and
            # measurably necessary: verified directly that for a vague
            # query like "tell me about this project", the substantive
            # content ranks ~11th-12th within a single document, well
            # outside the default top_k=4 tuned for whole-corpus search.
            k = config.SCOPED_TOP_K if selected_source else config.TOP_K
            # A bare follow-up ("and what about NoSQL?") has no topic
            # keywords of its own to search on - rewrite it into a
            # standalone question using recent history *before* retrieval,
            # not just when phrasing the final answer.
            search_question = condense_question(question, history)
            chunks = retrieve(search_question, k=k, source=selected_source)

        if not chunks:
            answer = "No chunks retrieved — has `python -m src.ingest` been run?"
            sources = []
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, chunks, history=history)
            sources = [
                {
                    "source": chunk.metadata.get("source", "unknown"),
                    "snippet": chunk.page_content[:300],
                }
                for chunk in chunks
            ]

        if search_question != question:
            st.caption(f"🔎 Searched for: \"{search_question}\"")
        st.markdown(answer)
        if sources:
            with st.expander(f"📎 {len(sources)} source chunk(s)"):
                for i, chunk in enumerate(sources, start=1):
                    st.markdown(f"**[{i}] `{chunk['source']}`**")
                    st.text(chunk["snippet"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    # Persisted separately from the live chat above - survives a browser
    # refresh or app restart, but is never read back into generate_answer().
    # See src/chat_log.py for why that separation is deliberate.
    append_turn(
        question,
        answer,
        [s["source"] for s in sources],
        scope=selected_label if selected_source else None,
    )

with st.sidebar:
    if "rebuild_message" in st.session_state:
        st.success(st.session_state.pop("rebuild_message"))

    section_header("ℹ️", "About")
    st.markdown(
        "This assistant answers questions using only documents in `data/` — "
        "it's instructed to say \"I don't know\" rather than guess when the "
        "answer isn't in the retrieved chunks."
    )
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_header("📤", "Add a document")
    uploaded_files = st.file_uploader(
        "PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption(
        "Saves into `data/my_docs/` and rebuilds the whole index from "
        "scratch (fast — no LLM calls, just local embedding of every "
        "chunk). Existing chat history isn't affected."
    )
    if st.button(
        "🔄 Save & rebuild index", use_container_width=True, disabled=not uploaded_files
    ):
        MY_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded in uploaded_files:
            (MY_DOCS_DIR / uploaded.name).write_bytes(uploaded.getvalue())

        with st.spinner(f"Rebuilding index ({len(uploaded_files)} new file(s))..."):
            num_documents, num_chunks = rebuild_index()
            reset_vector_store()  # otherwise retrieve() keeps using the old, now-replaced collection

        # Rerun so the document-scope picker (built at the top of the
        # script, before this button's code runs) refreshes to include the
        # new file immediately instead of needing one more interaction.
        # st.success() here would otherwise be wiped out by that rerun
        # before the user ever saw it, so stash it in session_state instead.
        st.session_state.rebuild_message = (
            f"Indexed {num_documents} document(s), {num_chunks} chunk(s)."
        )
        st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_header("📜", "History")
    st.caption(
        "A persistent log of past questions — survives a browser refresh "
        "or app restart. Purely for your own reference; never fed back "
        "into the model as conversation context."
    )
    past_turns = load_recent_turns(limit=20)
    if not past_turns:
        st.caption("No questions asked yet.")
    else:
        for turn in reversed(past_turns):
            label = turn["question"]
            if turn.get("scope"):
                label += f"  ·  {turn['scope']}"
            with st.expander(label):
                st.caption(turn["timestamp"])
                st.markdown(turn["answer"])
        if st.button("🗑️ Clear history", use_container_width=True):
            clear_history()
            st.rerun()
