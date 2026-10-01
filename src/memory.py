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
    "Rewrite the follow-up question to be standalone ONLY if it contains a "
    "pronoun or implicit reference pointing back to the history (\"it\", "
    "\"that\", \"the other one\", or an elliptical phrase like \"what about "
    "X?\"). If the follow-up is already a complete question on its own - "
    "even one on a completely different topic - output it unchanged. Most "
    "questions are not follow-ups; when in doubt, leave it unchanged.\n\n"
    "Output only the question itself, nothing else - no preamble, no "
    "quotes, no explanation.\n\n"
    "Example 1:\n"
    "History:\nQ: What is the capital of France?\nA: The capital of France "
    "is Paris.\n"
    "Follow-up: what about Germany?\n"
    "Output: What is the capital of Germany?\n\n"
    "Example 2:\n"
    "History:\nQ: How does photosynthesis work?\nA: Photosynthesis converts "
    "sunlight into chemical energy in plants.\n"
    "Follow-up: does it need sunlight?\n"
    "Output: Does photosynthesis need sunlight?\n\n"
    "Example 3 (new topic - NOT a follow-up, left unchanged):\n"
    "History:\nQ: What is the capital of France?\nA: The capital of France "
    "is Paris.\n"
    "Follow-up: explain how volcanoes form\n"
    "Output: explain how volcanoes form\n\n"
    "Now your turn.\n\n"
    "History:\n{history}\n\n"
    "Follow-up: {question}\n\n"
    "Output:"
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
