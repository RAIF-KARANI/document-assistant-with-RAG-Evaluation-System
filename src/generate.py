"""
Generation: turn retrieved chunks + a question into a grounded answer.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src import config
from src.memory import format_history

# The grounding instruction: tells the model to answer only from the
# supplied context and refuse instead of guessing. This prompt text is the
# entire "grounding" mechanism - it's a request to the model, not an
# enforced constraint, which is why faithfulness still needs to be measured
# later rather than assumed.
PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant answering questions using only the context "
    "below. If the answer is not contained in the context, say you don't "
    "know instead of guessing.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Used instead of PROMPT only when the caller passes a non-empty history
# (app.py's live chat, never eval/run_eval_dataset.py or
# experiments/run_experiment.py - both call generate_answer() with no
# history, so their prompts - and therefore the regression baseline - are
# completely unaffected by this). The history is for resolving references
# ("it", "that") in the question's *phrasing* only - the instruction still
# points facts back at the retrieved context, not the conversation.
PROMPT_WITH_HISTORY = ChatPromptTemplate.from_template(
    "You are a helpful assistant answering questions using only the context "
    "below. If the answer is not contained in the context, say you don't "
    "know instead of guessing.\n\n"
    "Recent conversation (only for resolving references like \"it\" or "
    "\"that\" in the question below - the Context section is still the "
    "only source of facts):\n{history}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# --- Augmentation ---
# Turns the retrieved chunks into the {context} block that gets inserted
# into the prompt below, numbered and tagged with their source file.
def format_context(chunks: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {chunk.metadata.get('source', 'unknown')})\n{chunk.page_content}"
        for i, chunk in enumerate(chunks, start=1)
    )


# --- Generation ---
# Fills the prompt with the retrieved context + question and sends the
# combined text to the local LLM (Ollama). temperature=0 makes the model
# always pick its most likely next token - deterministic, no creativity,
# which keeps answers closer to the supplied context. `history` (recent
# (question, answer) pairs, oldest first) is optional and only used by
# app.py's live chat - see PROMPT_WITH_HISTORY above for why omitting it
# (the eval harness's case) reproduces the exact original single-turn
# prompt untouched.
def generate_answer(
    question: str, chunks: list[Document], history: list[tuple[str, str]] | None = None
) -> str:
    llm = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    context = format_context(chunks)
    if history:
        chain = PROMPT_WITH_HISTORY | llm
        response = chain.invoke(
            {"context": context, "question": question, "history": format_history(history)}
        )
    else:
        chain = PROMPT | llm  # LCEL: fill the template, then call the model
        response = chain.invoke({"context": context, "question": question})
    return response.content
