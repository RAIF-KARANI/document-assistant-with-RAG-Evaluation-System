"""
Multi-turn conversational memory: lets a follow-up question like "and what
about NoSQL?" get resolved using the *current session's* recent turns,
instead of being retrieved and answered as if it stood completely alone.

Deliberately scoped to the live chat session (app.py's
st.session_state.messages), not src/chat_log.py's persisted disk log - that
log stays storage-only on purpose (see its own docstring), and reaching back
into a previous browser session's questions would be a different, stranger
feature than "resolve this follow-up." Clearing the chat (app.py's "Clear
chat" button, which empties st.session_state.messages) naturally clears this
memory too, since it's derived fresh from that same list every turn.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src import config

# How many prior Q&A turns to carry forward. Bounded on purpose - an
# unbounded history would make every later question's prompt grow forever
# over a long session, and a follow-up almost always refers to the last
# exchange or two, not something from 20 questions ago.
MAX_HISTORY_TURNS = 3

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    "Given the conversation history below and a follow-up question, rewrite "
    "the follow-up as a standalone question that contains all the context "
    "needed to search for an answer independently of the history. If the "
    "follow-up already stands on its own, return it unchanged. Output only "
    "the rewritten question, nothing else - no preamble, no quotes.\n\n"
    "Conversation history:\n{history}\n\n"
    "Follow-up question: {question}\n\n"
    "Standalone question:"
)


# Turns app.py's flat st.session_state.messages (one dict per chat bubble)
# into (question, answer) pairs, oldest first. Called with the messages list
# as it stood *before* the current question was appended, so "history" never
# includes the very question it's being used to resolve.
def extract_history(messages: list[dict]) -> list[tuple[str, str]]:
    history = []
    pending_question = None
    for m in messages:
        if m["role"] == "user":
            pending_question = m["content"]
        elif m["role"] == "assistant" and pending_question is not None:
            history.append((pending_question, m["content"]))
            pending_question = None
    return history


# Shared by src/generate.py's history-aware prompt too, so both places
# format the same (question, answer) pairs identically.
def format_history(history: list[tuple[str, str]]) -> str:
    recent = history[-MAX_HISTORY_TURNS:]
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in recent)


# --- Query condensation ---
# Retrieval needs a self-contained query - a bare "what about NoSQL?" has no
# topic keywords for semantic search to match against, no matter how good
# the embedding model is. Rewriting it into a standalone question *before*
# it reaches retrieve() fixes the actual failure mode (nothing relevant gets
# retrieved), not just symptoms in the final answer's wording. Skipped
# entirely - no extra LLM call - when there's no history yet, since the
# first question in a session is always already standalone.
def condense_question(question: str, history: list[tuple[str, str]]) -> str:
    if not history:
        return question
    llm = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    chain = CONDENSE_PROMPT | llm
    response = chain.invoke({"history": format_history(history), "question": question})
    return response.content.strip()
