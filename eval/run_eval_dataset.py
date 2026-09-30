"""
Run the RAG assistant over every question in eval/dataset.jsonl and save
the question, ground truth, retrieved chunks, and generated answer for
each one. This is the raw material Phase 3's metrics get computed from --
no scoring happens here, just recording what the pipeline actually did.

    python -m eval.run_eval_dataset
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.generate import generate_answer
from src.retrieve import retrieve

DATASET_PATH = config.PROJECT_ROOT / "eval" / "dataset.jsonl"
RESULTS_DIR = config.PROJECT_ROOT / "eval" / "results"


# Reads the hand-labeled eval set: question + ground_truth_answer +
# ground_truth_context per line. This is the "known correct" data everything
# else gets measured against.
def load_dataset(path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# Runs every question through the exact same retrieval + generation
# pipeline src/ask.py uses, and records what actually came back - not
# scored yet, just captured for Phase 3's metrics to consume.
# A question with a "scope_source" field is run the way app.py's document
# picker runs a scoped question - filtered to that one source and using
# config.SCOPED_TOP_K - so this failure mode (vague queries retrieving the
# wrong document/section) is measurable by the eval harness, not just
# manually spot-checked (see CONCEPTS.md for why that mattered).
def run(items: list[dict]) -> list[dict]:
    results = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{i}/{len(items)}] {question}")
        scope_source = item.get("scope_source")
        if scope_source:
            chunks = retrieve(question, k=config.SCOPED_TOP_K, source=scope_source)
        else:
            chunks = retrieve(question)  # retrieval stage
        answer = generate_answer(question, chunks)  # generation stage
        results.append(
            {
                "id": item["id"],
                "question": question,
                "ground_truth_answer": item["ground_truth_answer"],
                "ground_truth_context": item["ground_truth_context"],
                "unanswerable": item.get("unanswerable", False),
                "retrieved_contexts": [c.page_content for c in chunks],
                "retrieved_sources": [
                    c.metadata.get("source", "unknown") for c in chunks
                ],
                "generated_answer": answer,
            }
        )
    return results


def save_results(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{timestamp}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    items = load_dataset(DATASET_PATH)
    print(f"Loaded {len(items)} questions from {DATASET_PATH}")
    results = run(items)
    out_path = save_results(results)
    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
