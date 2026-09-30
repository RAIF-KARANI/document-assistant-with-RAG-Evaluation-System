"""
Score a results file (produced by eval/run_eval_dataset.py) with Ragas
metrics, using our own local Ollama model as the judge LLM instead of a
paid cloud API. Prints a scorecard.

    python -m eval.evaluate [path/to/results.jsonl] [--fast] [--label NAME]

If no path is given, scores the most recent file in eval/results/.

Known limitations, confirmed by running the full 20-question dataset (see
eval/baseline_scores.json and CONCEPTS.md for the numbers):
- `context_precision` is excluded entirely - fails to produce parseable
  output on every attempt.
- `answer_relevancy` is included but effectively non-functional - scored
  1/20 on the full run. Kept in the default metric set for completeness,
  but --fast drops it (see below) since retrying a near-100%-failure metric
  wastes most of a run's wall-clock time for one noisy data point.

--fast scores only `faithfulness` and `context_recall` - the two metrics
confirmed reliable (75%+ and 100% success respectively). Use this for
Phase 4 experiment iterations, where a fast feedback loop matters more than
complete metric coverage on metrics already known to be mostly noise.

Without --label, results overwrite eval/baseline_scores.json (the canonical
reference Phase 4 experiments compare against). With --label NAME, results
are saved to experiments/NAME_scores.json instead, leaving the baseline
untouched - use this for every experimental config so you don't clobber the
baseline you're trying to beat.
"""

import argparse
import json
import sys
import types
from pathlib import Path

# --- Compatibility shim ---
# ragas 0.4.x unconditionally imports ChatVertexAI even for users who never
# touch Google Vertex AI. langchain-community split that integration into a
# separate package (langchain-google-vertexai) which drags in ~40MB of
# unused Google Cloud SDKs. We stub the missing module instead of installing
# all that for a dead import. Must happen before any `import ragas`.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

from src import config

RESULTS_DIR = config.PROJECT_ROOT / "eval" / "results"
BASELINE_PATH = config.PROJECT_ROOT / "eval" / "baseline_scores.json"
EXPERIMENTS_DIR = config.PROJECT_ROOT / "experiments"

# Below this, a question is flagged in the scorecard as worth a look.
FLAG_THRESHOLD = 0.5

ALL_METRICS = ["faithfulness", "context_recall", "answer_relevancy", "answer_correctness"]
FAST_METRICS = ["faithfulness", "context_recall"]  # the two confirmed reliable


def find_latest_results() -> Path:
    candidates = sorted(RESULTS_DIR.glob("*.jsonl"))
    candidates = [p for p in candidates if not p.name.endswith("_scores.jsonl")]
    if not candidates:
        raise FileNotFoundError(
            f"No results files in {RESULTS_DIR}. "
            "Run `python -m eval.run_eval_dataset` first."
        )
    return candidates[-1]


def load_results(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Judge setup ---
# Same local models the assistant itself uses (see src/config.py) - no
# separate/paid judge model. The judge grades the assistant's own retrieval
# and generation, using the exact chunks that were actually retrieved.
def build_judge() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    llm = LangchainLLMWrapper(ChatOllama(model=config.CHAT_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    )
    return llm, embeddings


def build_metrics(llm, embeddings, names: list[str]) -> dict:
    available = {}
    if "faithfulness" in names:
        available["faithfulness"] = Faithfulness(llm=llm)
    if "context_recall" in names:
        available["context_recall"] = LLMContextRecall(llm=llm)
    if "answer_relevancy" in names:
        available["answer_relevancy"] = ResponseRelevancy(llm=llm, embeddings=embeddings)
    if "answer_correctness" in names:
        answer_correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)
        answer_correctness.init(RunConfig())  # wires up its internal similarity scorer
        available["answer_correctness"] = answer_correctness
    return available


# --- Scoring ---
# Runs every metric against every question. Each (question, metric) pair is
# scored independently and failures don't abort the run - a single flaky
# judge call shouldn't lose the other 19 questions' worth of results. This
# mirrors the real failure mode of LLM-as-judge evaluation: it's failure-
# prone by nature, so the harness has to expect and record failures, not
# just the happy path.
async def score_result(item: dict, metrics: dict) -> dict:
    sample = SingleTurnSample(
        user_input=item["question"],
        response=item["generated_answer"],
        retrieved_contexts=item["retrieved_contexts"],
        reference=item["ground_truth_answer"],
    )
    scores = {}
    for name, metric in metrics.items():
        try:
            scores[name] = await metric.single_turn_ascore(sample)
        except Exception as e:
            scores[name] = None
            print(f"    [{item['id']}] {name} failed: {type(e).__name__}")
    return scores


async def run(results: list[dict], metric_names: list[str]) -> pd.DataFrame:
    llm, embeddings = build_judge()
    metrics = build_metrics(llm, embeddings, metric_names)

    rows = []
    for i, item in enumerate(results, start=1):
        print(f"[{i}/{len(results)}] scoring {item['id']}: {item['question']}")
        scores = await score_result(item, metrics)
        rows.append({"id": item["id"], "question": item["question"], **scores})
    return pd.DataFrame(rows)


# --- Aggregation & reporting ---
def print_scorecard(df: pd.DataFrame) -> None:
    metric_cols = [c for c in df.columns if c not in ("id", "question")]

    print("\n=== Scorecard (mean per metric) ===")
    for col in metric_cols:
        valid = df[col].dropna()
        n_failed = df[col].isna().sum()
        mean = valid.mean() if len(valid) else float("nan")
        print(f"  {col}: {mean:.3f}  ({len(valid)}/{len(df)} scored, {n_failed} failed)")

    print(f"\n=== Questions below {FLAG_THRESHOLD} on any metric ===")
    flagged = df[(df[metric_cols] < FLAG_THRESHOLD).any(axis=1)]
    if flagged.empty:
        print("  none")
    else:
        print(flagged[["id", "question", *metric_cols]].to_string(index=False))


# Without --label: overwrites the canonical baseline every run (Phase 3
# behavior). With --label: writes to experiments/<label>_scores.json instead
# and leaves the baseline untouched, so an experimental config can be
# compared against the baseline without destroying it.
def save_summary(df: pd.DataFrame, results_path: Path, label: str | None) -> dict:
    metric_cols = [c for c in df.columns if c not in ("id", "question")]
    summary = {
        "label": label,
        "source_results_file": results_path.name,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "embedding_model": config.EMBEDDING_MODEL,
        "chat_model": config.CHAT_MODEL,
        "mean_scores": {col: df[col].dropna().mean() for col in metric_cols},
        "scored_counts": {col: int(df[col].notna().sum()) for col in metric_cols},
        "num_questions": len(df),
    }
    if label:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EXPERIMENTS_DIR / f"{label}_scores.json"
    else:
        out_path = BASELINE_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", nargs="?", default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Score only faithfulness + context_recall (the reliable metrics).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Save to experiments/<label>_scores.json instead of overwriting the baseline.",
    )
    return parser.parse_args()


def main() -> None:
    import asyncio

    args = parse_args()
    results_path = Path(args.results_path) if args.results_path else find_latest_results()
    metric_names = FAST_METRICS if args.fast else ALL_METRICS

    print(f"Scoring {results_path} with metrics: {metric_names}")
    results = load_results(results_path)

    df = asyncio.run(run(results, metric_names))

    scores_path = results_path.with_name(results_path.stem + "_scores.jsonl")
    df.to_json(scores_path, orient="records", lines=True)
    print(f"\nSaved per-question scores to {scores_path}")

    print_scorecard(df)
    save_summary(df, results_path, args.label)


if __name__ == "__main__":
    main()
