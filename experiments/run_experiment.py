"""
Run one experiment config end-to-end and compare it to the baseline:
ingest (with overridden chunk_size/chunk_overlap/embedding_model, into an
isolated Chroma collection so the main assistant's data is never touched),
generate answers over eval/dataset.jsonl using the given top_k, score with
the fast/reliable metrics (faithfulness, context_recall), and append a row
to experiments/experiments.csv comparing against eval/baseline_scores.json.

    python -m experiments.run_experiment <label> [--chunk-size N] [--chunk-overlap N] [--top-k N] [--embedding-model NAME]

Examples:
    python -m experiments.run_experiment topk8 --top-k 8
    python -m experiments.run_experiment chunk1000 --chunk-size 1000 --chunk-overlap 100

Only override what you're testing - anything left unset uses the same value
as the baseline (src/config.py), so the comparison isolates one variable.
"""

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.evaluate import BASELINE_PATH, build_judge, build_metrics, print_scorecard
from eval.run_eval_dataset import DATASET_PATH, load_dataset
from ragas.dataset_schema import SingleTurnSample
from src import config
from src.generate import generate_answer
from src.ingest import build_vector_store, chunk_documents, load_documents
from src.retrieve import retrieve_from

EXPERIMENTS_DIR = config.PROJECT_ROOT / "experiments"
CHROMA_EXPERIMENTS_DIR = EXPERIMENTS_DIR / "chroma"
CSV_PATH = EXPERIMENTS_DIR / "experiments.csv"

CSV_FIELDS = [
    "timestamp",
    "label",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "embedding_model",
    "faithfulness_mean",
    "faithfulness_delta",
    "faithfulness_scored",
    "context_recall_mean",
    "context_recall_delta",
    "context_recall_scored",
    "num_questions",
    "notes",
]

# Below this fraction scored, a mean is flagged as unreliable (small-sample
# survivorship bias) rather than trusted at face value - see chunk250's
# faithfulness=1.000 from only 7/20 scored questions for why this matters.
MIN_RELIABLE_FRACTION = 0.7


# --- Stage 1: isolated ingest ---
# Only re-ingests when chunking/embedding actually changed from baseline -
# a pure top_k experiment can reuse the baseline's own chroma_db untouched,
# since top_k only affects how many chunks retrieval asks for, not what got
# indexed.
def ingest_for_experiment(label: str, chunk_size: int, chunk_overlap: int, embedding_model: str):
    is_baseline_ingest = (
        chunk_size == config.CHUNK_SIZE
        and chunk_overlap == config.CHUNK_OVERLAP
        and embedding_model == config.EMBEDDING_MODEL
    )
    if is_baseline_ingest:
        print("No chunking/embedding override - reusing the baseline chroma_db.")
        return config.COLLECTION_NAME, config.CHROMA_DIR

    collection_name = f"experiment_{label}"
    persist_directory = CHROMA_EXPERIMENTS_DIR / label
    print(f"Ingesting isolated collection '{collection_name}' at {persist_directory} ...")
    documents = load_documents(config.DATA_DIR)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_vector_store(
        chunks,
        embedding_model=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    print(f"Indexed {len(chunks)} chunks.")
    return collection_name, persist_directory


# --- Stage 2: generation over the eval dataset, with this config's retrieval ---
def generate_for_experiment(collection_name, persist_directory, top_k, embedding_model):
    items = load_dataset(DATASET_PATH)
    results = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{i}/{len(items)}] {question}")
        chunks = retrieve_from(
            question,
            k=top_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
        )
        answer = generate_answer(question, chunks)
        results.append(
            {
                "id": item["id"],
                "question": question,
                "ground_truth_answer": item["ground_truth_answer"],
                "generated_answer": answer,
                "retrieved_contexts": [c.page_content for c in chunks],
            }
        )
    return results


# --- Stage 3: score with the fast/reliable metrics only ---
async def score_for_experiment(results: list[dict]) -> dict:
    llm, embeddings = build_judge()
    metrics = build_metrics(llm, embeddings, ["faithfulness", "context_recall"])

    per_metric_scores = {name: [] for name in metrics}
    for item in results:
        sample = SingleTurnSample(
            user_input=item["question"],
            response=item["generated_answer"],
            retrieved_contexts=item["retrieved_contexts"],
            reference=item["ground_truth_answer"],
        )
        for name, metric in metrics.items():
            try:
                score = await metric.single_turn_ascore(sample)
                # Ragas can return NaN without raising (e.g. "no statements
                # generated from the answer" for a very short/refusal answer
                # - faithfulness is 0/0, mathematically undefined). A single
                # NaN silently poisons sum()/mean() in plain Python, unlike
                # pandas' .dropna().mean() in eval/evaluate.py - treat it the
                # same as a failure rather than let it corrupt the average.
                if score != score:  # NaN != NaN is the cheapest NaN check
                    print(f"    [{item['id']}] {name} returned NaN (no statements/undefined)")
                else:
                    per_metric_scores[name].append(score)
            except Exception as e:
                print(f"    [{item['id']}] {name} failed: {type(e).__name__}")

    total = len(results)
    means = {
        name: (sum(scores) / len(scores) if scores else None)
        for name, scores in per_metric_scores.items()
    }
    # Sample size travels with the mean from here on - a mean from a small
    # surviving subsample (heavy judge-failure rate) is not comparable to
    # one from a near-complete sample, even if the number itself looks fine.
    counts = {name: (len(scores), total) for name, scores in per_metric_scores.items()}
    return means, counts


def load_baseline_means() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("mean_scores", {})


def append_csv_row(row: dict) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Short name for this experiment, e.g. 'topk8'.")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Skip ingest+generation, reload experiments/<label>_results.jsonl and "
        "just re-run scoring. Useful after fixing a scoring bug without redoing "
        "the (usually much slower) generation step.",
    )
    args = parser.parse_args()

    results_path = EXPERIMENTS_DIR / f"{args.label}_results.jsonl"

    if args.rescore_only:
        if not results_path.exists():
            raise FileNotFoundError(f"{results_path} doesn't exist - can't rescore.")
        print(f"Reloading existing results from {results_path} (skipping ingest+generation)")
        with open(results_path, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
    else:
        collection_name, persist_directory = ingest_for_experiment(
            args.label, args.chunk_size, args.chunk_overlap, args.embedding_model
        )
        results = generate_for_experiment(
            collection_name, persist_directory, args.top_k, args.embedding_model
        )
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved generation results to {results_path}")

    print("Scoring (fast metrics: faithfulness, context_recall) ...")
    mean_scores, scored_counts = asyncio.run(score_for_experiment(results))

    baseline_means = load_baseline_means()
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": args.label,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "embedding_model": args.embedding_model,
        "faithfulness_mean": mean_scores.get("faithfulness"),
        "faithfulness_delta": (
            mean_scores["faithfulness"] - baseline_means["faithfulness"]
            if mean_scores.get("faithfulness") is not None and "faithfulness" in baseline_means
            else None
        ),
        "faithfulness_scored": "%d/%d" % scored_counts.get("faithfulness", (0, len(results))),
        "context_recall_mean": mean_scores.get("context_recall"),
        "context_recall_delta": (
            mean_scores["context_recall"] - baseline_means["context_recall"]
            if mean_scores.get("context_recall") is not None and "context_recall" in baseline_means
            else None
        ),
        "context_recall_scored": "%d/%d" % scored_counts.get("context_recall", (0, len(results))),
        "num_questions": len(results),
        "notes": args.notes,
    }
    append_csv_row(row)

    print(f"\n=== Experiment '{args.label}' vs. baseline ===")
    for metric in ("faithfulness", "context_recall"):
        base = baseline_means.get(metric)
        new = mean_scores.get(metric)
        scored, total = scored_counts.get(metric, (0, len(results)))
        reliable = total > 0 and (scored / total) >= MIN_RELIABLE_FRACTION
        flag = "" if reliable else "  ** UNRELIABLE - small surviving sample, do not trust this number **"
        if base is not None and new is not None:
            delta = new - base
            arrow = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            print(
                f"  {metric}: baseline={base:.3f}  experiment={new:.3f}  ({arrow} {delta:+.3f})"
                f"  [{scored}/{total} scored]{flag}"
            )
        else:
            print(f"  {metric}: baseline={base}  experiment={new}  [{scored}/{total} scored]{flag}")
    print(f"\nLogged to {CSV_PATH}")


if __name__ == "__main__":
    main()
