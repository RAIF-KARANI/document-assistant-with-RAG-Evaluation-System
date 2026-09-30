"""
Generation: turn retrieved chunks + a question into a grounded answer.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src import config

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
# which keeps answers closer to the supplied context.
def generate_answer(question: str, chunks: list[Document]) -> str:
    llm = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    context = format_context(chunks)
    chain = PROMPT | llm  # LCEL: fill the template, then call the model
    response = chain.invoke({"context": context, "question": question})
    return response.content
