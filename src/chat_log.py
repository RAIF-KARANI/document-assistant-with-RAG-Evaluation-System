"""
Chat history persistence - appends each Q&A turn to a local JSONL log so
conversations survive a browser refresh or app restart.

Deliberately kept separate from generation: this log is never read back
into the prompt sent to the LLM. Multi-turn conversational memory (letting
the model resolve a follow-up question using earlier turns) is a
different, harder feature - ironically the exact "Contextual Amnesia"
research gap QueryCraft's own report names (see eval/dataset.jsonl q24) -
and isn't what this module does. This is storage for the human to look
back at, not context for the model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src import config

HISTORY_PATH = config.PROJECT_ROOT / "chat_history.jsonl"


def append_turn(
    question: str,
    answer: str,
    sources: list[str],
    scope: str | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": question,
        "answer": answer,
        "sources": sources,
        "scope": scope,
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_recent_turns(limit: int = 50) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[-limit:]


def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
