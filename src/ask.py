"""
CLI entrypoint: ask a question against the ingested documents.

    python -m src.ask "What is RAG?"
"""

import sys

from src.generate import generate_answer
from src.retrieve import retrieve


def ask(question: str) -> None:
    chunks = retrieve(question)  # retrieval stage - see src/retrieve.py
    if not chunks:
        print("No chunks retrieved — did you run `python -m src.ingest` first?")
        return

    answer = generate_answer(question, chunks)  # generation stage - see src/generate.py

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        snippet = chunk.page_content[:150].replace("\n", " ")
        print(f"  [{i}] {source} — \"{snippet}...\"")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask "your question"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    ask(question)


if __name__ == "__main__":
    main()
