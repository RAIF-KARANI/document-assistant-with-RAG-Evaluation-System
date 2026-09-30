"""
Regression gate: re-runs the assistant fresh over eval/dataset.jsonl right
now, scores it with the reliable metrics (faithfulness, context_recall -
see CONCEPTS.md for why only these two), and fails if either has dropped
more than REGRESSION_TOLERANCE below eval/baseline_scores.json.

This is a deliberate, slow test - it makes real local LLM calls (fresh
generation, then judge scoring), typically several minutes. It's meant to
be run on demand before/after a change you want to gate (a new chunk size,
a prompt edit, a model swap), not wired into a fast save-on-every-edit loop:

    pytest eval/test_regression.py -v -s
"""

import asyncio
import json

import pytest

from eval.evaluate import BASELINE_PATH, build_judge, build_metrics
from eval.run_eval_dataset import DATASET_PATH, load_dataset
from eval.run_eval_dataset import run as run_pipeline
from ragas.dataset_schema import SingleTurnSample

# How far below baseline a metric can drop before this counts as a
# regression rather than normal run-to-run judge noise.
REGRESSION_TOLERANCE = 0.05

RELIABLE_METRICS = ["faithfulness", "context_recall"]


def load_baseline_means() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(f"No baseline at {BASELINE_PATH} - run `python -m eval.evaluate` first.")
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["mean_scores"]


async def _score_current() -> dict:
    items = load_dataset(DATASET_PATH)
    results = run_pipeline(items)  # regenerates fresh answers against the current pipeline

    llm, embeddings = build_judge()
    metrics = build_metrics(llm, embeddings, RELIABLE_METRICS)

    per_metric = {name: [] for name in metrics}
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
                # generated" for a short/refusal answer). A single NaN
                # poisons sum()/len() in plain Python - same bug found and
                # fixed in experiments/run_experiment.py; this test never
                # ran for real until now, so it never got caught here too.
                if score == score:  # NaN != NaN
                    per_metric[name].append(score)
            except Exception:
                pass  # a single flaky judge call shouldn't fail the whole gate
    return {name: (sum(v) / len(v) if v else None) for name, v in per_metric.items()}


@pytest.fixture(scope="module")
def current_scores():
    return asyncio.run(_score_current())


@pytest.mark.parametrize("metric", RELIABLE_METRICS)
def test_metric_not_regressed(metric, current_scores):
    baseline = load_baseline_means()
    current = current_scores.get(metric)
    assert current is not None, f"{metric} produced zero scored questions this run"

    floor = baseline[metric] - REGRESSION_TOLERANCE
    assert current >= floor, (
        f"{metric} regressed: {current:.3f} vs baseline {baseline[metric]:.3f} "
        f"(tolerance {REGRESSION_TOLERANCE}, floor {floor:.3f})"
    )
