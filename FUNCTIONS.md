# Function Reference

What each file/function does. Update this whenever `src/` changes.

## app.py

Streamlit chat UI (`streamlit run app.py` — must use the project's venv,
e.g. `venv\Scripts\streamlit.exe run app.py`, or dependencies won't
resolve). Calls the exact same `retrieve()`/`generate_answer()` functions as
`src/ask.py` — no separate logic to keep in sync between the CLI and the
UI. Keeps chat history in `st.session_state`, shows retrieved chunks per
answer in an expander, and has a sidebar "Clear chat" reset. Status pills
under the title show the live config (chat model, embedding model, top_k).
No live per-message quality scoring (see PLAN.md Phase 5 for why that was
deferred).

**Sidebar "Add a document"** — `st.file_uploader` (PDF/TXT, multiple) +
"Save & rebuild index" button. On click: writes each uploaded file's bytes
into `data/my_docs/`, calls `src.ingest.rebuild_index()` (full rebuild —
cheap enough for this corpus size, no LLM calls, just local embedding), then
`src.retrieve.reset_vector_store()` so the next question in the same
running process reopens a fresh client instead of using a cached reference
to the now-replaced collection. Then `st.rerun()`s (with the success message
stashed in `st.session_state.rebuild_message` first, since `st.success()`
would otherwise be wiped out by the rerun before anyone saw it) so the
document-scope picker below refreshes to include the new file immediately.

**"🔎 Scope to a document" picker** — built from `src.retrieve.list_sources()`
(the distinct `source` metadata values actually indexed, not re-derived
filenames, so a filter built from it always matches exactly). When a single
document is selected, retrieval uses `config.SCOPED_TOP_K` (20) instead of
the corpus-wide `config.TOP_K` (4) — verified directly that this matters,
not just plausible-sounding: for a vague query like "tell me about this
project", the substantive content ranked ~11th-12th within the one
document, so scoping without also raising `k` still failed (later raised
12→20 when a different query's answer ranked 16th - see CONCEPTS.md).
Fixes a real failure a user hit live — vague queries retrieving boilerplate
(acknowledgement/certificate pages, or table-of-contents fragments) instead
of the actual content.

**Sidebar "History"** — shows the persistent log from `src/chat_log.py` as
an expander per past question (newest first), plus a "Clear history"
button. Purely a read of `load_recent_turns()`; every actual question in
the main chat calls `append_turn()` right after the answer is generated.

## src/chat_log.py

Persists each Q&A turn to a local JSONL file (`chat_history.jsonl`, project
root, gitignored) so conversations survive a browser refresh or app
restart. Deliberately never read back into `generate_answer()` — this is
storage for the human to review later, not conversational memory for the
model. (Multi-turn memory, letting the model resolve a follow-up question
using earlier turns, is the harder "Contextual Amnesia" feature QueryCraft's
own report names as a gap it wants to solve — see `eval/dataset.jsonl`
q24 — and isn't what this module does.)

- **`append_turn(question, answer, sources, scope=None)`**
  Appends one entry (UTC timestamp, question, answer, source filenames,
  and the document scope if one was selected) to the log.
- **`load_recent_turns(limit=50) -> list[dict]`**
  Reads the log back, most recent `limit` entries. Empty list if the file
  doesn't exist yet (nothing asked so far).
- **`clear_history()`**
  Deletes the log file entirely. Backs the sidebar's "Clear history" button.

## pages/1_📊_Evaluation_Dashboard.py

Second page of the Streamlit app (Streamlit auto-discovers `pages/`) — the
visual surface for the "RAG Evaluation System" half of the project.

- **`load_baseline()` / `load_experiments()` / `load_dataset()` /
  `load_latest_scores()`** — read `eval/baseline_scores.json`,
  `experiments/experiments.csv`, `eval/dataset.jsonl`, and the newest
  `eval/results/*_scores.jsonl` fresh on every page load. Nothing on this
  page is hardcoded or cached - it always reflects whatever the eval
  scripts most recently produced.
- **`load_chunk_count()`** — opens the Chroma collection directly via
  `chromadb.PersistentClient(...).get_collection(...).count()` rather than
  `src.retrieve.get_vector_store()`, specifically so the dashboard never
  has to load the embedding model just to display a number.
- **Baseline section** — `st.metric` cards for faithfulness/context_recall/
  question count/chunk count, plus an expander disclosing which metrics are
  actually reliable (pulled from the Phase 3/4 findings, not re-derived).
- **Experiments section** — the CSV as a table with faithfulness/context_recall
  deltas colored green/red via `pandas.Styler`, and a bar chart of
  `context_recall_mean` per experiment label - deduplicated to the most
  recent run per label first, since several labels (e.g. `chunk250`) were
  intentionally rerun after a bug fix and would otherwise show as
  overlapping duplicate bars.
- **Eval dataset section** — the 30-question dataset as a filterable table
  (by source document, answerable/unanswerable).
- **Latest scores section** — per-question scores from the most recent
  scored run, with an "All questions" / "⚠️ Flagged low" tab split.

## src/config.py

Central constants — no functions, just values everything else imports.

| Constant | Value | Meaning |
|---|---|---|
| `DATA_DIR` | `data/` | where source documents are read from |
| `CHROMA_DIR` | `chroma_db/` | where the vector store is persisted |
| `COLLECTION_NAME` | `"documents"` | Chroma collection name |
| `CHUNK_SIZE` | `500` | max chunk length in **characters**, not tokens |
| `CHUNK_OVERLAP` | `50` | characters repeated between consecutive chunks |
| `TOP_K` | `4` | number of chunks retrieved per question |
| `SCOPED_TOP_K` | `20` | chunks retrieved when a single document is scoped (`app.py`'s picker, or an eval question with `scope_source`) — shared so the app and eval harness can't drift apart |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | local embedding model |
| `CHAT_MODEL` | `llama3.2` | local Ollama chat model |

## src/ingest.py

Builds the vector store from scratch. Run via `python -m src.ingest`.

- **`is_toc_like(text, threshold=0.5, min_lines=4) -> bool`**
  Flags a page as table-of-contents/list-of-figures/list-of-tables style if
  ≥50% of its non-empty lines end in a bare 1-4 digit page number. Not a
  guess — validated against every page of `QueryCraft.pdf` before wiring
  in: caught exactly the 6 real TOC/list pages (fractions 0.60-0.97),
  didn't false-positive on the citation-heavy references page (0.05,
  since citation years land mid-sentence, not at line-ends). Exists
  because a section title repeated in a TOC line ("9.2 Limitations ... 43")
  can lexically out-rank that section's own real content for a question
  about the same topic — confirmed happening in this project, see
  CONCEPTS.md.

- **`load_documents(data_dir) -> list[Document]`**
  Walks `data_dir` recursively, loads every `.pdf` (via `PyPDFLoader`) and
  `.txt` (via `TextLoader`) file it finds, dropping any page `is_toc_like()`
  flags before it's ever chunked or embedded. Returns LangChain `Document`
  objects, each with `.page_content` (text) and `.metadata` (includes
  `source`, the file path).

- **`chunk_documents(documents) -> list[Document]`**
  Runs `RecursiveCharacterTextSplitter` over the loaded documents using
  `CHUNK_SIZE`/`CHUNK_OVERLAP` from config. Splits on paragraph breaks first,
  then lines, then spaces, only cutting mid-word as a last resort.

- **`build_vector_store(chunks) -> Chroma`**
  Embeds every chunk with `HuggingFaceEmbeddings` (local model) and writes
  them into a Chroma collection persisted at `CHROMA_DIR`.

- **`drop_existing_collection(collection_name, persist_directory)`**
  Deletes a previously indexed collection via Chroma's own API (not raw
  filesystem deletion — a client from earlier in the same process, e.g.
  `src.retrieve`'s cached store, may already hold an open connection to
  that persist_directory). Fixes a real bug: `Chroma.from_documents` adds
  to an existing collection of the same name rather than replacing it, so
  re-ingesting without this step silently duplicated every chunk.

- **`rebuild_index(...) -> tuple[num_documents, num_chunks]`**
  The full, safe rebuild: drop whatever's indexed now, then load + chunk +
  embed everything currently in `data_dir` from scratch. Used by both
  `main()` below and `app.py`'s sidebar upload flow — one code path, not
  two copies of the same logic.

- **`main()`**
  Calls `rebuild_index()`, prints progress. This is what executes when you
  run `python -m src.ingest` — safe to run repeatedly now, unlike before
  this fix.

## src/retrieve.py

Given a question, finds the most relevant chunks already indexed.

- **`get_vector_store() -> Chroma`**
  Opens the persisted Chroma store at `CHROMA_DIR` (must already exist —
  run `src.ingest` first). Cached in a module-level `_vector_store` so it's
  only opened once per process.

- **`reset_vector_store()`**
  Clears that cache. Must be called after `src.ingest.rebuild_index()`
  rewrites `chroma_db/` on disk within the same running process (e.g.
  `app.py`'s upload flow), or `retrieve()` would keep querying a Python
  object that still points at the old, now-replaced collection.

- **`list_sources() -> list[str]`**
  Distinct `source` metadata values currently indexed (reads real indexed
  values, not filenames re-derived from the filesystem). Backs `app.py`'s
  document-scope picker.

- **`retrieve(query, k=TOP_K, source=None) -> list[Document]`**
  Embeds `query` with the same embedding model used at ingest time, then
  runs a similarity search against the vector store, returning the `k`
  closest chunks. `source`, when given, adds a Chroma metadata filter
  (`{"source": ...}`) so the search only considers chunks from that one
  document — see `app.py`'s scope picker for why this exists.

## src/generate.py

Turns retrieved chunks + a question into an answer.

- **`PROMPT`**
  A `ChatPromptTemplate` that instructs the model to answer only from the
  given context and say "I don't know" instead of guessing.

- **`format_context(chunks) -> str`**
  Formats a list of chunks into a numbered block of text, each entry tagged
  with its source file, for insertion into the prompt.

- **`generate_answer(question, chunks) -> str`**
  Fills `PROMPT` with the formatted context + question, sends it to
  `ChatOllama` (`CHAT_MODEL`), returns the model's answer text.

## src/ask.py

CLI entrypoint. Run via `python -m src.ask "your question"`.

- **`ask(question)`**
  Calls `retrieve()` then `generate_answer()`, prints the answer followed by
  a numbered source list (file + short snippet) for each chunk used.

- **`main()`**
  Reads the question from `sys.argv`, calls `ask()`. Exits with a usage
  message if no question was given.

## eval/run_eval_dataset.py

Runs the assistant over every question in `eval/dataset.jsonl` and records
the raw results (no scoring — that's Phase 3). Run via
`python -m eval.run_eval_dataset`.

- **`load_dataset(path) -> list[dict]`**
  Reads `eval/dataset.jsonl` (one JSON object per line: `id`, `question`,
  `ground_truth_answer`, `ground_truth_context`, `source_doc`,
  `unanswerable`, optional `scope_source`).

- **`run(items) -> list[dict]`**
  For each question: calls `retrieve()` then `generate_answer()` (same
  functions `src.ask` uses), and records the question, ground truth,
  `retrieved_contexts` (chunk texts), `retrieved_sources` (which files they
  came from), and `generated_answer`. If an item has `scope_source` set,
  retrieves filtered to that one document at `config.SCOPED_TOP_K` instead
  of the corpus-wide default — the same path `app.py`'s document picker
  uses, so a question testing scoped/vague-query behavior (like q31/q32)
  is measured the way it's actually used, not just unscoped like everything
  else.

- **`save_results(results) -> Path`**
  Writes results to `eval/results/<UTC timestamp>.jsonl`, one run per file so
  past runs are never overwritten — needed later to compare against a
  baseline for regression testing (Phase 4).

- **`main()`**
  Loads the dataset, runs it, saves results, prints progress per question.

## eval/evaluate.py

Scores a results file with Ragas metrics, using our own local Ollama model
as the judge (not a paid cloud API). Run via `python -m eval.evaluate
[path]` — defaults to the most recent file in `eval/results/`.

- **Compatibility shim (top of file)** — Ragas 0.4.x unconditionally imports
  `ChatVertexAI`, which langchain-community removed into a separate package
  that pulls in ~40MB of unused Google Cloud SDKs. A fake module is injected
  into `sys.modules` before `import ragas` so that dead import succeeds
  without installing any of it.

- **`find_latest_results() -> Path`** / **`load_results(path) -> list[dict]`**
  Finds/reads the newest `eval/results/*.jsonl` produced by
  `run_eval_dataset.py` (skips `*_scores.jsonl` files).

- **`build_judge() -> (llm, embeddings)`**
  Wraps our existing local `ChatOllama`/`HuggingFaceEmbeddings` (same models
  `src/generate.py` and `src/retrieve.py` use) as Ragas-compatible judge
  objects via `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper`.

- **`build_metrics(llm, embeddings, names) -> dict`**
  Constructs `faithfulness`, `context_recall`, `answer_relevancy`,
  `answer_correctness`. **`context_precision` is deliberately excluded** —
  confirmed via direct testing that our local 3B judge model reliably fails
  to produce parseable output for that metric's prompt.
  `AnswerCorrectness.init(RunConfig())` must be called manually to wire up
  its internal similarity scorer — easy to miss, throws
  `AnswerSimilarity must be set` otherwise.

  **Actual reliability on the full 20-question run** (see
  `eval/baseline_scores.json`): `context_recall` 20/20, `faithfulness`
  15/20, `answer_correctness` 12/20, `answer_relevancy` **1/20** — this last
  one is effectively non-functional with our local judge despite looking
  "mostly fine" on a 3-question smoke test. Treat its 0.892 mean as noise
  (n=1), not a real score.

- **`score_result(item, metrics) -> dict`**
  Scores one question against every metric. Each metric call is wrapped in
  its own try/except — a single flaky judge call records `None` and a
  logged failure instead of aborting the whole run. LLM-as-judge evaluation
  is failure-prone by nature (see `CONCEPTS.md`), so the harness is built to
  expect and record that, not just the happy path.

- **`run(results) -> pd.DataFrame`**
  Loops `score_result` over every question, returns one row per question.

- **`print_scorecard(df)`**
  Prints mean per metric (with success/failure counts) and flags any
  question scoring below `FLAG_THRESHOLD` (0.5) on any metric.

- **`save_summary(df, results_path, label)`** (renamed from `save_baseline`)
  Without `--label`, overwrites `eval/baseline_scores.json` (the canonical
  reference). With `--label NAME`, writes to `experiments/NAME_scores.json`
  instead, so an experimental config never clobbers the baseline it's being
  compared against.

- **`--fast` / `--label` CLI flags** (Phase 4 addition) — `--fast` restricts
  scoring to `faithfulness` + `context_recall`, the two metrics confirmed
  reliable. Cuts a ~40 minute run down dramatically by skipping
  `answer_relevancy`'s near-100% failure/retry cost. Used by
  `experiments/run_experiment.py` for fast iteration.

## eval/test_regression.py

Regression gate: `pytest eval/test_regression.py -v -s`. Deliberately slow
(makes real local LLM calls) — re-generates fresh answers over
`eval/dataset.jsonl` right now, scores them with the reliable metrics, and
fails if either has dropped more than `REGRESSION_TOLERANCE` (0.05) below
`eval/baseline_scores.json`. Run on demand before/after a change you want to
gate, not on every save.

- **`_score_current() -> dict`** — reruns generation + fast scoring fresh
  (doesn't reuse any saved results file — a regression gate has to test the
  code as it exists *right now*, not a snapshot from earlier).
- **`test_metric_not_regressed(metric, current_scores)`** — parametrized
  over `faithfulness`/`context_recall`; asserts each hasn't dropped below
  `baseline - REGRESSION_TOLERANCE`.

## experiments/run_experiment.py

Runs one experiment config end-to-end and logs it to
`experiments/experiments.csv`, without ever touching the baseline. Run via
`python -m experiments.run_experiment <label> [--chunk-size N]
[--chunk-overlap N] [--top-k N] [--embedding-model NAME]`.

- **`ingest_for_experiment(label, chunk_size, chunk_overlap, embedding_model)`**
  Only re-ingests (into an isolated Chroma collection under
  `experiments/chroma/<label>/`) if chunking/embedding actually changed from
  the baseline config — a pure top-k experiment reuses the baseline's own
  `chroma_db/` untouched, since top-k only affects how many chunks retrieval
  asks for, not what got indexed.
- **`generate_for_experiment(...)`** — same generation loop as
  `eval/run_eval_dataset.py`, but retrieving via `src.retrieve.retrieve_from`
  (the uncached, parameterized variant) against this experiment's
  collection/top-k instead of the main assistant's cached store.
- **`score_for_experiment(results)`** — fast metrics only (faithfulness,
  context_recall), same reasoning as `--fast` above.
- **`append_csv_row(row)`** — appends one row to `experiments/experiments.csv`
  with the config, both metrics' means, and their delta vs. the current
  baseline — never overwrites previous experiments' rows.
