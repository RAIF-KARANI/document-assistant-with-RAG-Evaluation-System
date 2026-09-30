"""
Evaluation Dashboard - a visual view into the "RAG Evaluation System" half
of the project: the current baseline scores, every experiment run so far
compared against it, and the hand-labeled eval dataset itself.

Reads eval/baseline_scores.json, experiments/experiments.csv, and
eval/dataset.jsonl fresh on every load - this page never invents data, it
only displays whatever the eval scripts have actually produced.
"""

import json
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

ACCENT = "#7C9EFF"  # matches .streamlit/config.toml primaryColor

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1100px; }}
    [data-testid="stMetricLabel"] {{ color: #9AA1B2; font-size: 0.85rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.9rem; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
    .section-title {{
        display: flex; align-items: center; gap: 10px; margin-bottom: 2px;
    }}
    .section-title .icon {{ font-size: 1.35rem; }}
    .section-title .text {{ font-size: 1.2rem; font-weight: 700; color: #F0F1F5; }}
    .section-rule {{
        height: 3px; width: 44px; background: {ACCENT};
        border-radius: 2px; margin: 6px 0 18px 0;
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


def spacer(px: int = 26) -> None:
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


st.title("📊 Evaluation Dashboard")
st.caption(
    "How well the Document Assistant is actually measured to perform — "
    "not just \"it seems to work,\" but scored against a hand-labeled "
    "dataset with a local LLM-as-judge."
)
spacer(8)

BASELINE_PATH = config.PROJECT_ROOT / "eval" / "baseline_scores.json"
EXPERIMENTS_CSV = config.PROJECT_ROOT / "experiments" / "experiments.csv"
DATASET_PATH = config.PROJECT_ROOT / "eval" / "dataset.jsonl"
RESULTS_DIR = config.PROJECT_ROOT / "eval" / "results"

RELIABLE_METRICS = {"faithfulness", "context_recall"}


# --- Data loading - always read fresh, never cache stale eval output ---
def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiments() -> pd.DataFrame:
    if not EXPERIMENTS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENTS_CSV)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        return pd.DataFrame()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def load_chunk_count() -> int | None:
    # Uses chromadb directly (not src.retrieve.get_vector_store) so this
    # page never has to load the embedding model just to display a count -
    # count() only needs the collection's stored metadata.
    if not config.CHROMA_DIR.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return client.get_collection(config.COLLECTION_NAME).count()
    except Exception:
        return None


def load_latest_scores() -> tuple[pd.DataFrame, Path | None]:
    candidates = sorted(RESULTS_DIR.glob("*_scores.jsonl"))
    if not candidates:
        return pd.DataFrame(), None
    latest = candidates[-1]
    with open(latest, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows), latest


baseline = load_baseline()
experiments_df = load_experiments()
dataset_df = load_dataset()
scores_df, scores_path = load_latest_scores()

# --- Section 1: current baseline ---
section_header("📈", "Current baseline")

with st.container(border=True):
    if baseline is None:
        st.warning("No baseline yet — run `python -m eval.evaluate` first.")
    else:
        means = baseline.get("mean_scores", {})
        cols = st.columns(4)
        with cols[0]:
            st.metric("📐 Faithfulness", f"{means.get('faithfulness', float('nan')):.3f}")
        with cols[1]:
            st.metric("🎯 Context Recall", f"{means.get('context_recall', float('nan')):.3f}")
        with cols[2]:
            st.metric("❓ Questions", baseline.get("num_questions", "?"))
        with cols[3]:
            chunk_count = load_chunk_count()
            st.metric("🧩 Chunks indexed", chunk_count if chunk_count is not None else "?")

        st.caption(
            f"Config: chunk_size={baseline.get('chunk_size')} · "
            f"chunk_overlap={baseline.get('chunk_overlap')} · "
            f"top_k={baseline.get('top_k')} · "
            f"embedding={baseline.get('embedding_model', '').split('/')[-1]} · "
            f"chat={baseline.get('chat_model')}"
        )

        with st.expander("⚠️ Which of these numbers can you actually trust?"):
            st.markdown(
                "- **`context_recall`** — reliable. Scored successfully on "
                "essentially 100% of questions across every run this project "
                "has done.\n"
                "- **`faithfulness`** — mostly reliable, but the judge model "
                "(a local 3B Llama) sometimes fails to parse a score, and has "
                "been caught inconsistently penalizing correct \"I don't "
                "know\" refusals as if they were hallucinations. The true rate "
                "is likely a bit higher than the raw number shown.\n"
                "- **`answer_relevancy`** and **`context_precision`** are not "
                "shown here at all — direct testing found they fail to "
                "produce usable scores over 90% of the time with this local "
                "judge model. See `CONCEPTS.md` for the full investigation."
            )

spacer()

# --- Section 2: experiments ---
section_header("🧪", "Experiments vs. baseline")

with st.container(border=True):
    if experiments_df.empty:
        st.info("No experiments logged yet — run `python -m experiments.run_experiment`.")
    else:
        display_df = experiments_df.sort_values("timestamp", ascending=False).copy()

        def highlight_delta(val):
            if pd.isna(val):
                return ""
            color = "#2ECC71" if val > 0 else ("#E74C3C" if val < 0 else "")
            return f"color: {color}; font-weight: 600;" if color else ""

        styled = display_df.style.map(
            highlight_delta, subset=["faithfulness_delta", "context_recall_delta"]
        ).format(
            {
                "faithfulness_mean": "{:.3f}",
                "faithfulness_delta": "{:+.3f}",
                "context_recall_mean": "{:.3f}",
                "context_recall_delta": "{:+.3f}",
            },
            na_rep="—",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            "🟢 Green = improved vs. the baseline at the time · 🔴 Red = "
            "regressed. Rows marked INVALID/UNRELIABLE/SUPERSEDED in their "
            "notes should be read with the caveat described there — several "
            "were caused by a too-small corpus or a scoring bug, both since "
            "fixed. See `experiments/experiments.csv` notes column and "
            "`CONCEPTS.md` for the full story behind each row."
        )

        reliable = display_df[
            display_df["context_recall_scored"].astype(str).str.contains("/")
        ].copy()
        if not reliable.empty:
            # Some labels (e.g. "chunk250") appear twice - an initial buggy
            # run and its corrected rerun. Keep only the most recent row per
            # label so the chart shows one bar per config, not overlapping
            # duplicates.
            reliable = reliable.sort_values("timestamp").drop_duplicates(
                "label", keep="last"
            )
            chart_df = reliable.set_index("label")[["context_recall_mean"]]
            st.bar_chart(chart_df, height=280, color=ACCENT)
            st.caption(
                "One bar per experiment label (most recent run only, where a "
                "label was rerun after a bug fix)."
            )

spacer()

# --- Section 3: eval dataset ---
section_header("📋", "Hand-labeled eval dataset")

with st.container(border=True):
    if dataset_df.empty:
        st.info("No eval dataset found at `eval/dataset.jsonl`.")
    else:
        source_options = ["All"] + sorted(
            dataset_df["source_doc"].dropna().unique().tolist()
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            source_filter = st.selectbox("Filter by source document", source_options)
        with col2:
            answerable_filter = st.selectbox(
                "Filter by type", ["All", "Answerable", "Unanswerable"]
            )

        filtered = dataset_df.copy()
        if source_filter != "All":
            filtered = filtered[filtered["source_doc"] == source_filter]
        if answerable_filter == "Answerable":
            filtered = filtered[~filtered["unanswerable"]]
        elif answerable_filter == "Unanswerable":
            filtered = filtered[filtered["unanswerable"]]

        st.dataframe(
            filtered[["id", "question", "ground_truth_answer", "source_doc", "unanswerable"]],
            use_container_width=True,
            hide_index=True,
            height=350,
        )
        st.caption(f"Showing {len(filtered)} of {len(dataset_df)} questions.")

spacer()

# --- Section 4: latest per-question scores ---
section_header("🔍", "Latest scored run — per-question breakdown")

with st.container(border=True):
    if scores_df.empty:
        st.info("No per-question scores found yet.")
    else:
        st.caption(f"From `{scores_path.name}`")
        metric_cols = [c for c in ("faithfulness", "context_recall") if c in scores_df.columns]
        flagged = scores_df[
            (scores_df[metric_cols].fillna(1.0) < 0.5).any(axis=1)
        ] if metric_cols else pd.DataFrame()

        tab1, tab2 = st.tabs(["All questions", f"⚠️ Flagged low ({len(flagged)})"])
        with tab1:
            st.dataframe(
                scores_df[["id", "question", *metric_cols]],
                use_container_width=True,
                hide_index=True,
                height=350,
            )
        with tab2:
            if flagged.empty:
                st.success("Nothing scored below 0.5 on any metric in this run.")
            else:
                st.dataframe(
                    flagged[["id", "question", *metric_cols]],
                    use_container_width=True,
                    hide_index=True,
                )
