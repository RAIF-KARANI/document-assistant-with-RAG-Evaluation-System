# Project Plan

Document Assistant + RAG Evaluation System, merged into one project: build a
RAG pipeline, then use an evaluation harness to measure and improve it.

```
documents -> [RAG Assistant] -> answers
                  ^
         [Evaluation System] measures: retrieval quality, answer correctness,
         faithfulness, regressions when chunking/embeddings/prompts change
```

## Phase 0 — Concepts (ongoing)

Learn concepts alongside building, not all up front. Running notes live in
[CONCEPTS.md](CONCEPTS.md). Core topics: embeddings, chunking, vector
databases, retrieval, RAG generation, evaluation metrics (context
precision/recall, faithfulness, answer relevancy/correctness).

- [x] Chunking: fixed-size / recursive-character splitting, chunk overlap
- [x] Chunking: semantic chunking, hierarchical/parent-child chunking
- [x] Embeddings: what a vector actually represents, similarity search
- [x] Vector databases: how Chroma indexes and searches (HNSW, semantic index)
- [x] Retrieval: top-k, semantic search vs. keyword search
- [x] Generation: prompt construction, grounding, hallucination
- [x] Evaluation metrics: precision/recall/faithfulness/relevancy in depth

Phase 0 concept coverage complete for the full pipeline. Next new concepts
(regression testing, experiment design) come up naturally once Phase 2/3 are
actually being built.

## Phase 1 — Minimal RAG assistant — COMPLETE, fully verified end-to-end

- [x] Project scaffold (`src/`, `data/`, `eval/`, `experiments/`)
- [x] Document loading (`.pdf`, `.txt`) — `src/ingest.py`
- [x] Chunking — `RecursiveCharacterTextSplitter`, 500 char chunks / 50 overlap
- [x] Embedding + indexing — local `sentence-transformers`, stored in Chroma
- [x] Retrieval — `src/retrieve.py`, tested, returns correct chunks
- [x] Generation — `src/generate.py`, local `ChatOllama` (llama3.2), Ollama
      installed and confirmed working
- [x] CLI — `python -m src.ask "question"` — ran end-to-end:
      `python -m src.ask "What is RAG?"` returned a correct, grounded answer
      citing all 4 relevant source chunks.

Phase 1 done. No open items.

## Phase 2 — Build the evaluation dataset — COMPLETE (expanded in Phase 4)

- [x] Pick questions about the test documents with known answers — now 30
      questions in `eval/dataset.jsonl`: original 18 answerable + 2
      unanswerable spanning the sample docs, plus 9 new answerable + 1
      unanswerable covering `data/my_docs/QueryCraft.pdf` (the user's own
      62-page college project report, added as a real external document -
      see Phase 4). QueryCraft questions cover the abstract, tech stack,
      the 4 research gaps, architecture layers, a design pattern, and
      limitations/future work - written from source text I read directly
      via pypdf, not guessed.
- [x] Record `question`, `ground_truth_answer`, `ground_truth_context` per item
- [x] Store as `eval/dataset.jsonl`
- [x] `eval/run_eval_dataset.py` — runs the assistant over the whole set,
      saves `question, retrieved_contexts, generated_answer` to
      `eval/results/<timestamp>.jsonl`
- [x] First full run completed and spot-checked —
      `eval/results/20260929T120737Z.jsonl` (all 20 questions). Both
      unanswerable questions correctly got "I don't know" instead of a
      hallucinated answer; q01's generated answer matched ground truth
      almost verbatim.

## Phase 3 — Automated metrics (Ragas) — IN PROGRESS

Scope note: building the Ragas scorecard now; DeepEval's pytest-style
regression *gate* moved to Phase 4 where it actually belongs (Phase 3 =
what are the numbers, Phase 4 = did the numbers regress).

- [x] `eval/evaluate.py` built — scores `faithfulness`, `context_recall`,
      `answer_relevancy`, `answer_correctness` using our own local Ollama
      model as judge (no cloud API key)
- [x] Fixed a real compatibility bug: Ragas 0.4.x unconditionally imports
      `ChatVertexAI`; stubbed the missing module rather than installing
      ~40MB of unused Google Cloud SDKs just to satisfy a dead import
- [x] `context_precision` deliberately excluded — confirmed via direct
      testing it reliably fails to parse with our local 3B judge model
      (llama3.2), while the other four metrics mostly work
- [x] Full 20-question run completed (~40 min locally) and scorecard
      reviewed — see `eval/baseline_scores.json` and
      `eval/results/20260929T120737Z_scores.jsonl`:
      - `context_recall`: 0.817 mean, 20/20 scored — fully reliable
      - `faithfulness`: 0.858 mean, 15/20 scored (75%) — mostly reliable
      - `answer_correctness`: 0.595 mean, 12/20 scored (60%) — usable, flaky
      - `answer_relevancy`: 0.892 mean, but only **1/20 scored (5%)** —
        effectively non-functional with our local judge; the 3-question
        smoke test (1/3 succeeded) badly understated this. Treat this
        metric's number as noise, not signal, until a stronger judge model
        is used.
- [x] Pandas aggregation script — `print_scorecard()` in `eval/evaluate.py`
- [x] Saved `eval/baseline_scores.json` — mean scores + the exact config
      (chunk_size=500, overlap=50, top_k=4, embedding + chat model) that
      produced them. This is what Phase 4 experiments compare against.

Phase 3 done. Real takeaway: only `context_recall` and `faithfulness` are
trustworthy enough from this local-judge setup to lean on in Phase 4;
`answer_correctness` is usable but noisy; `answer_relevancy` and
`context_precision` need a stronger judge model before they mean anything.

## Phase 4 — Regression testing & experimentation — IN PROGRESS

- [x] `eval/evaluate.py` gained `--fast` (score only the 2 reliable metrics)
      and `--label` (save to `experiments/<label>_scores.json` instead of
      overwriting the baseline) flags — makes iteration dramatically faster
      than the ~40 min full Phase 3 run
- [x] Parameterized `src/ingest.py` (`chunk_documents`, `build_vector_store`)
      and `src/retrieve.py` (new `retrieve_from`) to accept overrides
      (chunk size/overlap, embedding model, collection name, persist dir)
      with config.py values as defaults — main assistant behavior unchanged,
      experiments get isolated Chroma collections that never touch the real
      `chroma_db/`
- [x] `eval/test_regression.py` built — `pytest eval/test_regression.py -v -s`
      regenerates fresh answers, scores with the reliable metrics, fails if
      either drops more than 0.05 below `eval/baseline_scores.json`. This is
      what "DeepEval pytest-style gate" turned into — plain pytest +
      Ragas's reliable metrics, since that combination is what's actually
      working; not yet run for real (needs a deliberate slow run)
- [x] `experiments/run_experiment.py` built — runs one config end-to-end
      (isolated ingest if needed, generation, fast scoring) and logs a row
      to `experiments/experiments.csv` with deltas vs. baseline
- [x] Experiment: top-k — `top_k=8` vs baseline `top_k=4`. Result:
      faithfulness flat (0.858→0.848), context_recall dropped notably
      (0.817→0.625). **Caveat that changes the conclusion:** our sample
      corpus has exactly 8 total chunks, so `top_k=8` = "retrieve the
      entire corpus every time," not "retrieve more selectively" — this
      experiment can't cleanly separate "more chunks hurt retrieval" from
      "a longer judge prompt hurts our weak local judge's own reliability."
      See CONCEPTS.md. **Top-k experiments need a bigger corpus to mean
      anything on this project** — logged as a real limitation, not chased
      further with this toy dataset.
- [x] Experiment: chunk size — `chunk_size=1000/overlap=100` vs baseline
      `500/50`. Result: faithfulness flat (0.858→0.856), context_recall
      dropped again (0.817→0.597). **Same root cause as the top-k
      experiment:** larger chunks collapsed the corpus to only 4 total
      chunks, so with `top_k=4` this also retrieves the entire corpus every
      time. Confirms experiment 1 wasn't a one-off — this project's 2-doc
      sample corpus (~60 lines) is too small for *any* retrieval-selectivity
      experiment to mean anything until `total_chunks >> top_k`. See
      CONCEPTS.md for the full reasoning.
- [x] **Corpus expanded**: added 9 more sample docs to `data/sample_docs/`
      (2 docs/8 chunks → 11 docs/64 chunks at chunk_size=500). Re-ingested,
      regenerated eval answers, reset `eval/baseline_scores.json` against
      the new corpus: `faithfulness` 0.893, `context_recall` 0.857 (both
      reliable metrics, full 20-question run, see CONCEPTS.md).
- [x] Experiment: top_k=8 rerun (`topk8_v2`) on the real 64-chunk corpus —
      faithfulness 0.893→0.877 (-0.015), context_recall 0.857→0.775
      (**-0.082**). Real signal this time (top_k=8 is 12.5% of the corpus,
      not saturated) — smaller than the invalid v1 drop (-0.192), confirming
      v1 was partly a saturation artifact, but a real cost remains.
- [x] Experiment: chunk_size=1000 rerun (`chunk1000_v2`) — produced a real
      29-chunk corpus. faithfulness 0.893→0.868 (-0.025), context_recall
      0.857→0.718 (**-0.138**). Also real signal, also worse than baseline.
- [x] **Conclusion: baseline config (chunk_size=500, top_k=4) beats both
      alternatives tried.** Neither bigger chunks nor more retrieved chunks
      helped — both cost faithfulness and context_recall, likely partly
      because a longer per-question context is harder for our small local
      judge to score reliably (a real, disclosed limitation, not swept
      under the rug). Full comparison table + reasoning in CONCEPTS.md;
      raw log in `experiments/experiments.csv`.
- [x] Experiment: chunk_size=250 — surfaced a real bug (NaN silently
      poisoned the naive mean in `run_experiment.py`; fixed, `eval/evaluate.py`
      was never affected since pandas already handles this). After the fix,
      this run also revealed 65% of faithfulness scoring attempts failed
      (13/20) — the surviving 7 averaged a misleading "1.000" that is
      **not a real result**, just survivorship bias in a small sample.
      `run_experiment.py` now tracks and reports scored/total counts per
      metric and flags anything under 70% scored as unreliable. Only
      `context_recall` is trustworthy from this run (20/20 scored):
      0.857 → 0.829, a small real drop. Full writeup in CONCEPTS.md.
- [x] Experiment: top_k=3 — faithfulness 0.893→0.854 (only 12/20 scored,
      below reliability threshold, directional only), context_recall
      0.857→0.803 (20/20 scored, fully reliable, real -0.054 drop). Most
      intuitive result so far (fewer chunks → higher chance of missing the
      answer). **Baseline (chunk=500, top_k=4) has now beaten every
      alternative tried** on context_recall, the one metric that's scored
      reliably across all five experiments. Full scoreboard in CONCEPTS.md.
- [x] Experiment: embedding model swap (MiniLM-L6-v2 → mpnet-base-v2) —
      **first genuine win of the session**: faithfulness 0.744→0.872
      (+0.129), context_recall 0.787→0.815 (+0.028), both with matching
      full sample sizes on both sides (no reliability caveat). Real
      tradeoff: mpnet is ~4.5x MiniLM's params — slower embedding at
      ingest and query time, larger download. Quality win is real; whether
      it's worth the latency cost is a product decision, not something
      the eval harness resolves. **Decision: keep MiniLM-L6-v2 as the
      default** — speed prioritized over the measured quality gain. No
      change to `src/config.py`; `embed_mpnet` stays logged in
      `experiments/` as a documented option if priorities change later.
- [x] Experiment: chunking strategy swap (parent-child) — built for real
      (`chunk_documents_parent_child()` / `retrieve_parent_child()`,
      `--parent-child` flag on `run_experiment.py`), tried two configs.
      **Second genuine finding, this time negative:** `top_k=4` scored
      `context_recall` 0.780→0.673 (−0.107); root-caused to child→parent
      dedup collapsing matches to ~3.09 unique parents instead of 4.
      Retried at `top_k=8` to fix it — dedup collapse confirmed fixed
      (5.62 unique parents, more breadth than baseline) but
      `context_recall` didn't recover at all (0.671) — the real cause is
      likely the 400-char child size itself matching less precisely, not
      chunk count. **Decision: not adopted** — a sound-on-paper idea that
      measurably underperformed the simple baseline twice. Full root-cause
      write-up in CONCEPTS.md.

**Phase 4 wrapped up.** Infrastructure built and proven: experiment
runner, regression gate, fast/reliable-metrics scoring. Six experiments
run, two real bugs found and fixed, one genuine win found (embedding
model swap) and consciously not adopted (speed over quality tradeoff).
- [x] Real external document added: user's own 62-page project report
      (`QueryCraft.pdf`, a natural-language-to-SQL/NoSQL interface project)
      dropped into `data/my_docs/` for real-world testing beyond the
      synthetic sample corpus. Corpus now 73 docs / 297 chunks.
- [x] Baseline regenerated + rescored against the expanded 30-question
      dataset: `faithfulness` 0.744 (24/30 scored), `context_recall` 0.787
      (30/30 scored) - both lower than the old toy-corpus baseline, as
      expected for a genuinely harder, more realistic retrieval task.
- [x] **Found a second local-judge quirk by spot-checking scores against
      raw generated text**: 3 questions scored `faithfulness=0.0` but were
      actually correct "I don't know" refusals, not hallucinations - the
      judge is inconsistent about excluding refusals (sometimes NaN,
      sometimes wrongly scored 0). Means the 0.744 baseline likely
      *understates* true faithfulness. Full writeup in CONCEPTS.md.
- [x] **Ran `eval/test_regression.py` for real.** Fixed the same
      NaN-poisons-the-mean bug found earlier in `run_experiment.py` first
      (this test had only ever been `--collect-only`-checked, never
      executed, so it never hit the bug). Full run: regenerated fresh
      answers for all 32 questions, scored, both `test_metric_not_regressed
      [faithfulness]` and `[context_recall]` **PASSED** against the current
      baseline (32m50s total). The safety net we built in Phase 4 is now
      confirmed to actually work, not just structurally valid.
- [x] Baseline already reflects the best config found so far (chunk_size=500,
      top_k=4, i.e. the original defaults held up against both challengers)

## Phase 5 — Polish (optional) — IN PROGRESS

- [x] Streamlit UI (`app.py`, `streamlit run app.py`) — chat box + expandable
      shown sources per answer, sidebar with a "Clear chat" reset. Same
      `retrieve()`/`generate_answer()` functions the CLI uses, no separate
      logic to keep in sync. Tested live in-browser: submit question →
      retrieval spinner → generation spinner → grounded answer → sources
      expander showing chunk text + file → clear chat resets correctly.
- [x] Redesigned as a proper multi-page app: `.streamlit/config.toml` for a
      real theme (not Streamlit defaults), status pills on the chat page
      showing live config, and a second page —
      `pages/1_📊_Evaluation_Dashboard.py` — giving the "RAG Evaluation
      System" half of the project its own visual surface: baseline metrics,
      a reliability disclosure (which numbers to trust), the experiments
      log as a color-coded table + bar chart, a filterable eval-dataset
      browser, and the latest per-question scores with a flagged-low tab.
      Reads `eval/baseline_scores.json`, `experiments/experiments.csv`,
      `eval/dataset.jsonl`, and the latest `eval/results/*_scores.jsonl`
      fresh on every load - never invents or caches stale numbers.
- [x] Fixed a real CSV bug found while testing the dashboard: one
      historical row in `experiments/experiments.csv` had an unquoted
      comma inside a cell, breaking column alignment for every parser
      (pandas raised `ParserError`). Rewrote the file with Python's `csv`
      module for guaranteed-correct quoting instead of hand-editing.
- [x] Polished the dashboard's spacing/color: each section is now a
      bordered card (`st.container(border=True)`) with a consistent accent-
      underlined header, spacer rhythm between cards instead of plain
      `st.divider()` lines, themed metric typography, and the experiments
      bar chart recolored to match the app's primary accent instead of
      Streamlit's default blue. Tested live in-browser.
- [x] Matched the chat page's styling to the dashboard's: same accent-
      underlined `section_header()` helper (used for the sidebar "About"
      heading), chat bubbles restyled as bordered cards with real padding
      instead of Streamlit's bare default, one shared `ACCENT` color used
      consistently across both pages. Tested live in-browser end to end
      (question → retrieval → generation → styled answer + sources).
- [x] Verified mobile responsiveness (375×812) — chat and dashboard both
      degrade cleanly (columns stack, wide tables scroll within their own
      container rather than breaking the page). No fixes needed.
- [x] **Added in-app document upload** — sidebar file uploader (PDF/TXT) +
      "Save & rebuild index" button in `app.py`, so adding a document never
      requires the CLI. Saves into `data/my_docs/`, calls the same
      `rebuild_index()` the CLI uses, resets `src.retrieve`'s cached vector
      store so the next question uses the rebuilt index. Tested end to end
      (simulated the exact upload flow, confirmed 73→74 docs / 297→298
      chunks, and that a question about the new file's content retrieved
      and cited it correctly) — then cleaned up the test file and restored
      the real corpus.
- [x] **Found and fixed a real, previously-undiscovered bug while building
      this**: re-running ingestion (`python -m src.ingest`, or now the
      upload flow) without first deleting `chroma_db/` silently *duplicated*
      every chunk on top of the existing ones — `Chroma.from_documents`
      adds to an existing collection rather than replacing it. This had
      been true since Phase 1; every prior full-corpus rebuild in this
      project worked around it by manually doing `rm -rf chroma_db` first.
      Fixed at the source: `src/ingest.py` gained `drop_existing_collection()`
      and `rebuild_index()`, and `main()` now uses them, so `python -m
      src.ingest` is safe to run repeatedly without ever needing the manual
      workaround again.
- [x] **Added document scoping** — a "🔎 Scope to a document" picker at the
      top of the chat page, backed by `src.retrieve.list_sources()` and a
      `source` filter param on `retrieve()`. Investigated a real failure
      the user hit live ("tell me about this project" retrieved
      acknowledgement/certificate boilerplate instead of the abstract) by
      checking actual retrieved chunks and their ranked scores, not
      guessing: the substantive content ranked ~11th-12th for that query,
      well outside `top_k=4`. Scoping alone didn't fix it (verified);
      scoping + a higher `top_k` when a single document is selected
      (`SCOPED_TOP_K=12` in `app.py`) did — verified both directly in
      Python and live through the actual UI on the exact failing query.
- [x] Pushed `SCOPED_TOP_K` to 20 and re-tested both queries (checked
      first that 20 would be enough — the real content ranked 16th).
      "Tell me about this project" still works. "What are the limitations
      of this project?" retrieval is now fixed (right chunk included), but
      **generation fails differently**: the model answers from a table-of-
      contents fragment that lexically matches the question ("9.2
      Limitations of the Current System") instead of the real content
      chunk sitting right next to it in the same prompt — reproduced
      identically in Python and live in the browser. This is a
      generation-side failure now, not retrieval — more `top_k` won't fix
      it. Likely real fix: exclude table-of-contents pages at ingest time
      (pure navigational noise that creates false lexical matches).
- [x] **Implemented and verified the TOC filter.** `src/ingest.py` gained
      `is_toc_like()` — flags a page if ≥50% of its lines end in a bare
      page number. Validated against every page of `QueryCraft.pdf` before
      wiring it in: caught exactly the 6 TOC/List-of-Figures/List-of-Tables
      pages, didn't false-positive on the citation-heavy references page.
      Rebuilt the index (73→67 docs, 297→283 chunks) and re-ran both
      problem queries, verified in Python and live in the browser: "tell
      me about this project" still works, and the TOC-hijacking on "what
      are the limitations" is gone — the model now correctly leads with
      the real 9.2 content, with only a milder residual imperfection
      (pulls in one loosely-related item from an adjacent section). Full
      validation methodology and before/after numbers in CONCEPTS.md.
- [x] **Closed the eval-coverage gap.** Added `scope_source` support to
      `eval/dataset.jsonl`/`eval/run_eval_dataset.py` (a question with this
      field is retrieved the way `app.py`'s document picker actually runs
      it — filtered to that source, `config.SCOPED_TOP_K=20`). Moved
      `SCOPED_TOP_K` into `src/config.py` so `app.py` and the eval harness
      share one definition instead of duplicating the constant. Added two
      scoped questions (q31/q32 — the exact two queries that exposed the
      TOC bug) to the dataset (32 questions total), regenerated, rescored.
      New baseline: `faithfulness` 0.772 (27/32 scored), `context_recall`
      0.780 (31/32 scored). **q32 (scoped) scored a perfect 1.0/1.0**,
      directly against **q28 (the same underlying question, unscoped)
      scoring only 0.5/0.286** — the scoping+TOC fix's benefit is now
      measured by the eval harness itself, not just manually spot-checked.
- [ ] Live quality badge on each answer — deferred: would mean a second
      judge-LLM call per question in the UI, and Phase 3/4 already showed
      this local judge is slow and unreliable enough that doing it live,
      per-message, isn't a good trade for a chat UI's responsiveness
- [x] **Wrote up architecture + eval results in README.** Added an
      "Architecture" section — pipeline diagram, eval harness diagram, and
      four "design decisions worth knowing" (local-only, document scoping,
      TOC filtering, chat history as storage-not-memory) each grounded in a
      real finding from this project rather than generic RAG advice.
      Refreshed the stale Status numbers (30→32 questions, old baseline
      scores→current 0.772/0.780, 297→283 chunks) and the Project layout
      tree (added `chat_log.py`, `pages/`, `test_regression.py`, etc. — all
      missing from the original scaffold-era listing).
- [x] **Persistent chat history** — `src/chat_log.py` appends each Q&A turn
      (timestamp, question, answer, sources, scope) to a local gitignored
      JSONL file, surfaced via a sidebar "History" section. Deliberately
      never read back into `generate_answer()` — separates "remember what
      was asked" from "let the model use past turns," the latter being the
      harder feature QueryCraft's own report calls "Contextual Amnesia"
      (see `eval/dataset.jsonl` q24) and a deliberate non-goal here. Tested
      directly (append/load/clear round-trip, verified file appears/
      disappears on disk correctly) before wiring into the UI.
- [x] **Multi-turn conversational memory** — the "Contextual Amnesia"
      non-goal above, revisited and built. New `src/memory.py`:
      `extract_history()` pairs up the live session's chat turns;
      `condense_question()` rewrites a bare follow-up ("and what about
      NoSQL?") into a standalone query *before* retrieval, using an extra
      local LLM call (skipped entirely when there's no history yet, so the
      first question in a session costs nothing extra). `generate_answer()`
      gained an optional `history` param for resolving pronouns in the
      final answer too - omitted entirely by `eval/` and `experiments/`,
      so the regression baseline is provably untouched (verified: same
      answer with/without the param on identical input). Tested for real in
      the browser, not just unit-level: "What database does QueryCraft
      use?" → "and what about NoSQL?" correctly rewrote the follow-up to
      "What database dialects does QueryCraft support for NoSQL
      databases?" and answered "QueryCraft supports MongoDB as a NoSQL
      database" - the actual failure mode this was built to fix, confirmed
      fixed.

## Tech stack

Python, LangChain, Chroma, sentence-transformers (embeddings, local), Ollama
(generation, local), Ragas, DeepEval, Pandas, pytest, Streamlit (optional).

## Folder structure

```
src/
  config.py    # chunk size, top-k, model names
  ingest.py    # load -> chunk -> embed -> store in Chroma
  retrieve.py  # embed query -> similarity search
  generate.py  # build prompt from retrieved chunks -> call LLM
  ask.py       # CLI entrypoint
data/          # source documents (gitignored except samples)
eval/          # evaluation dataset + results (Phase 2+)
experiments/   # experiment logs comparing configs (Phase 4+)
```

See [FUNCTIONS.md](FUNCTIONS.md) for what each function does, and
[CONCEPTS.md](CONCEPTS.md) for the running concept notes.
