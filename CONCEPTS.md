# Concept Notes

Running log of RAG concepts explained during this project, so you don't have
to scroll chat history to find them again. Newest topics at the bottom.

## Chunking — fixed-size / recursive-character splitting

Our splitter (`RecursiveCharacterTextSplitter` in `src/ingest.py`) measures
`CHUNK_SIZE`/`CHUNK_OVERLAP` in **characters**, not tokens, by default. Tokens
are roughly 4 characters each for English, so `CHUNK_SIZE = 500` is really
~120-130 tokens per chunk, not 500.

**The mid-sentence cutting problem:** a truly naive fixed-size splitter would
cut every N characters regardless of where a word or sentence is, producing
chunks that end mid-word. Two consequences:
- The embedding for a broken chunk represents a half-formed thought, so it
  matches relevant questions worse.
- If retrieved, the LLM sees a mangled fragment instead of a complete fact.

**Why our splitter mostly avoids this:** it tries separators in priority
order before cutting anywhere: `["\n\n", "\n", " ", ""]` — paragraph breaks
first, then lines, then spaces, and only cuts mid-word as an absolute last
resort. It does **not** specifically respect sentence boundaries out of the
box (no `. ` in the default separator list), so it can still split between
or even mid-sentence in some cases.

**`CHUNK_OVERLAP` as insurance:** each chunk repeats the last ~50 characters
of the previous one, so information near a boundary shows up in both
neighboring chunks instead of being lost to a bad cut.

## Chunking — other strategies

**Sentence-based** — split on real sentence boundaries (NLTK/spaCy), then
group whole sentences up to a size limit. Never cuts mid-sentence, but still
splits on an arbitrary size budget.

**Semantic chunking** — embed every individual sentence, walk through them
in order, and cut a new chunk whenever similarity to the next sentence drops
(topic shift). Produces topically coherent, variable-length chunks, at the
cost of an embedding call per sentence just to find boundaries — more
compute, less predictable chunk sizes.

**Hierarchical / structure-aware splitting** — split along the document's
own structure (Markdown/HTML headers, section numbers) instead of ignoring
it, optionally tagging each chunk with metadata like `section: "2.3
Retrieval"`.

**Parent-child (a.k.a. `ParentDocumentRetriever`) chunking** — index small
child chunks (precise matching at retrieval time), but keep a pointer to the
larger parent chunk/section each child came from. At query time, search
against the small child embeddings, but hand the LLM the full parent chunk.
Fixes the "precise retrieval vs. full context for generation" tension
directly, rather than relying on overlap as insurance.

**What we're using / what to try later:** recursive-character splitting is
our Phase 1 baseline — cheap, predictable, good enough to get a working
system to measure. Parent-child chunking is the most promising upgrade to
test in Phase 4, scored against the eval harness rather than guessed at.

## Embeddings

Every chunk gets converted from text into a vector of numbers (384 of them,
for our model) that represents its meaning — chunks about similar topics end
up with vectors pointing in similar directions, unrelated ones end up far
apart. A question gets embedded into the same space at query time, and
retrieval finds the chunks whose vectors sit closest to it. See "Vector
databases / similarity search" below for what "closest" means mechanically.

**Do you need an embedding API?** You always need *some* embedding model, but
not necessarily a paid cloud API. Two options:
- **Cloud API** (e.g. OpenAI `text-embedding-3-small`) — network call per
  request, needs an API key, costs money.
- **Local model** (what we use) — runs in-process, no network call after the
  one-time model download, free.

**What we're using:** `sentence-transformers/all-MiniLM-L6-v2`
(`EMBEDDING_MODEL` in `src/config.py`) — a small (22M param) transformer
fine-tuned specifically to produce sentence-level embeddings, output
dimension 384. Downloads once (~90MB) from Hugging Face Hub, cached locally,
then runs on CPU with no further network calls.

**How it's wired in:**
- `src/ingest.py` → `build_vector_store()` creates a `HuggingFaceEmbeddings`
  instance and passes it to `Chroma.from_documents(...)`, which embeds every
  chunk once and stores `(vector, text, metadata)`.
- `src/retrieve.py` → `get_vector_store()` creates the *same* embedder and
  passes it to `Chroma(embedding_function=...)`. `retrieve(query)` embeds the
  query with it, then Chroma does nearest-neighbor search against the stored
  chunk vectors.

**Rule that matters:** indexing and querying must use the *same* embedding
model, or the two vector spaces are unrelated and "closest" means nothing.
That's why `EMBEDDING_MODEL` is one shared constant in `config.py` rather
than something each file picks independently.

## Vector databases / semantic index

Two separate jobs, easy to conflate:
- **Embedding model:** text in, vector out. No awareness of storage or search.
- **Vector database:** stores each vector alongside its original text +
  metadata, and builds an index structure so "find the closest vectors to
  this query vector" is fast. That searchable structure is what "semantic
  index" refers to — searchable by meaning, not exact keyword match.

**Why an index, not just a list:** brute-force search (compare the query
vector to every stored vector) is exact but O(n) — fine for a handful of
chunks, too slow at millions. Vector databases pre-build an index so most
comparisons can be skipped.

**How Chroma does it:** HNSW (Hierarchical Navigable Small World) — vectors
are organized into a multi-layer graph, each connected to a handful of its
nearest neighbors. Search starts at a coarse top layer, jumps toward the
query's neighborhood, then descends into finer layers to narrow in — a
handful of graph hops instead of comparing against everything. This is
"approximate" (ANN = approximate nearest neighbor): occasionally not the
literal best match, dramatically faster in exchange.

**Similarity metric:** Chroma defaults to cosine similarity — the angle
between two vectors, ignoring magnitude. Small angle = semantically similar.

**In our code:**
- `src/ingest.py` → `Chroma.from_documents(...)` embeds each chunk, writes
  `(vector, text, metadata, id)` to a SQLite-backed store on disk at
  `chroma_db/`, and inserts the vector into the HNSW index (also persisted
  there — that's what `python -m src.ingest` actually creates).
- `src/retrieve.py` → `similarity_search(query, k)` embeds the query, walks
  the HNSW graph for the k nearest chunk vectors, returns the original text +
  metadata for those matches (never raw vectors).

Full round trip: text → (embedding model) → vector → (vector database)
stored + indexed → (vector database, at query time) nearest matches →
original text handed back. The embedding model only touches the first step.

## Retrieval pipeline & semantic search

**Common misconception worth killing early: the query does NOT get chunked.**
Chunking solves one problem — documents are too long to embed as a single
meaningful vector. A query ("What is RAG?") is already one small, focused
unit of meaning, so it's embedded whole, as-is. Documents get chunked once,
at ingest time (`src/ingest.py`); queries never get chunked, ever — there's
no chunking step in `src/retrieve.py` at all.

**The retrieval pipeline (`retrieve(query)` in `src/retrieve.py`):**
1. Query arrives as raw text, no preprocessing.
2. Query gets embedded — same model, same vector space as the chunks (see
   Embeddings section above — this is why "same model for both" matters).
3. Vector search against Chroma's HNSW index (see Vector databases section).
4. Top-`k` chunks returned as `Document` objects (text + metadata), ranked by
   similarity score.

Genuinely nothing new mechanically here — it reuses the embedding + index
machinery built for ingestion. The new concept is *why* it works:

**Semantic search vs. keyword search:** keyword/lexical search (traditional
search engines, `Ctrl+F`) matches exact words or stems — zero vocabulary
overlap means zero match, even if the content is obviously relevant to a
human. Semantic search matches on the *embedded meaning* instead: "How does
it avoid making things up?" and a chunk saying "...answer from the retrieved
text rather than from its own memory" land close together in vector space
despite sharing almost no words, because they're about the same underlying
concept. Tradeoff: fuzzier — can retrieve something superficially similar
but actually irrelevant, which is what Phase 3's context precision/recall
metrics exist to catch.

**`TOP_K` (`src/config.py`, currently 4):** how many chunks come back per
query. Real tradeoff, not a known-correct number — too low risks missing the
chunk with the answer, too high dilutes the prompt with irrelevant chunks
and costs more tokens. One of the Phase 4 experiment variables, not
something to reason out from first principles.

## Generation & grounding

**Augmentation, precisely:** retrieved chunks aren't sent alongside the
prompt — they're inserted *into* the prompt text before it reaches the LLM.
The model gets one combined string (context + instructions + question) and
has zero awareness that retrieval happened. There is no special "RAG mode"
in the LLM — RAG is entirely a prompt-construction technique; the model call
itself is an ordinary chat completion.

**How `src/generate.py` builds it:**
- `PROMPT` — a template instructing the model to answer only from the
  supplied context, and say "I don't know" instead of guessing. This
  instruction *is* the grounding mechanism — nothing more sophisticated.
- `format_context(chunks)` — numbers each retrieved chunk and tags it with
  its source file, joined into the `{context}` block.
- `generate_answer()` — `chain = PROMPT | llm` (LangChain pipe syntax: fill
  the template, then send the result to the model). `.invoke(...)` fills in
  `{context}`/`{question}`, sends the finished prompt to Ollama, returns a
  message object whose `.content` is the plain-text answer.
- `temperature=0` — collapses the model's next-token choice to "always the
  single most likely token" (deterministic, no creativity). Right setting
  for RAG: we want answers to stick to the context, not drift into creative
  variation.

**Grounding is a request, not an enforced constraint.** The LLM has its own
parametric (trained-in) knowledge about most topics and nothing *technically*
stops it from using that instead of the supplied context — "answer only from
context" relies entirely on the model's instruction-following behavior. This
is exactly why hallucination can still happen inside a working RAG system,
and exactly why Phase 3's **faithfulness** metric exists: it checks whether
the generated answer is actually supported by the retrieved chunks, rather
than assuming grounding worked just because we asked for it.

**Known gap:** the prompt never asks the model to cite which numbered source
it used. The `[1]`/`[2]` list printed by `src/ask.py` reflects what was
*retrieved*, not what the model says it actually *used* in the answer.

## Evaluation metrics (Phase 3)important

Two failure modes, two metric groups (mirrors the retrieval-vs-faithfulness
split named in `data/sample_docs/rag_overview.txt`):
- **Retrieval metrics** — did we get the right chunks? Context Precision,
  Context Recall.
- **Generation metrics** — given the chunks we got, did the model behave
  well? Faithfulness, Answer Relevancy, Answer Correctness.

**Context Precision** — of the chunks retrieved, what fraction are actually
relevant? Low precision = noise diluting the LLM's context window. Ragas
scores this ranking-aware: relevant chunks should also rank higher.

**Context Recall** — of everything actually needed to answer (per your
ground truth), how much did retrieval surface? Opposite failure from
precision: clean retrieval that still missed the one chunk with the answer.
Needs a `ground_truth_context` field, not just a ground-truth answer.

**Faithfulness** — checks the generated answer against the retrieved context
*only* (no ground truth involved): for each claim the answer makes, is it
supported by the retrieved chunks? Computed by splitting the answer into
individual claims and asking a judge LLM "supported by context: yes/no" per
claim. Low score = hallucination — this is the direct, measured version of
the "grounding is a request, not a guarantee" problem from the generation
notes above.

**Answer Relevancy** — does the answer actually address the question asked,
independent of factual correctness? A faithful-but-off-topic answer scores
low here. Typically computed by having an LLM generate synthetic questions
that the given answer would be answering, then embedding-comparing those to
the real question.

**Answer Correctness** — does the generated answer match your hand-written
`ground_truth_answer`? Blends semantic similarity with factual overlap.

**Reference-free vs. reference-based** (matters for Phase 2 dataset design):

| Metric | Needs ground truth? | Inputs |
|---|---|---|
| Context Precision | usually yes | question, retrieved_contexts |
| Context Recall | yes | retrieved_contexts, ground_truth_context |
| Faithfulness | no | generated_answer, retrieved_contexts |
| Answer Relevancy | no | question, generated_answer |
| Answer Correctness | yes | generated_answer, ground_truth_answer |

Faithfulness and Answer Relevancy check internal consistency only, so they
run on any question with no hand-labeling. Context Recall and Answer
Correctness need Phase 2's hand-labeled `ground_truth_answer` /
`ground_truth_context` to compare against.

**How it's actually computed — LLM-as-judge.** Not string matching: Ragas
and DeepEval feed the question/context/answer/ground-truth to a judge LLM
(often a stronger model) which scores or breaks down claims, as above for
faithfulness. This costs real LLM calls — a 20-question eval pass can mean
50-100+ judge calls under the hood. DeepEval leans pytest-style
(`assert faithfulness_score >= 0.7`), which is what Phase 4's regression
gate is built on.

## LLM-as-judge reliability with small local models (Phase 3, hands-on finding)

Ragas needs an LLM to act as judge for every metric (see "How it's actually
computed" above). By default it assumes a strong cloud model (GPT-4-class).
We wired it to use our own local `llama3.2` (3B) instead — no separate paid
judge, the assistant grades itself with the same model it answers with.

**This mostly works, but not uniformly — and a 3-question smoke test badly
understated how bad it gets at scale.** The full 20-question run:
`context_recall` scored 20/20 (100% reliable). `faithfulness` scored 15/20
(75%). `answer_correctness` scored 12/20 (60%). `answer_relevancy` scored
**1/20 (5%)** — it looked "mostly working" on 3 samples (1/3 succeeded) but
that was luck, not signal; at 20 questions it's revealed as effectively
non-functional with this judge model. `context_precision` fails to produce
parseable output consistently even on small samples — excluded from
`eval/evaluate.py` entirely.

**Lesson inside the lesson:** a 3-question smoke test is enough to catch
crashes and wiring bugs, but not enough to estimate a *failure rate* —
33% success on n=3 and 5% success on n=20 are both "consistent with" a huge
range of true rates. Don't trust a reliability estimate from a sample that
small; this is exactly why the full run mattered even though it took ~40
minutes.

**Why this happens:** these metrics work by asking the judge model to
produce *structured output* (JSON matching a schema, or a broken-down list
of claims) in response to a fairly demanding prompt. A 3B model is fine at
straightforward generation (our own assistant's answers) but measurably
weaker at faithfully following a complex structured-output instruction —
some prompts are just harder to comply with exactly than others. This isn't
a code bug; it's a real capability ceiling, confirmed by testing the exact
same model/setup against multiple metrics and seeing some reliably succeed
and one reliably fail.

**Design consequence:** `eval/evaluate.py` treats judge failures as
expected, not exceptional — each `(question, metric)` pair is scored
independently in its own try/except, a failure records `None` and a logged
message rather than crashing the whole run, and the scorecard reports
success/failure *counts* per metric rather than silently averaging over
whatever happened to succeed. Hiding the failure rate would make the mean
score look more trustworthy than it is.

**The honest fix, if it matters later:** use a larger local judge model
(7B+ tends to be meaningfully more reliable at structured output) or a
cloud model as judge specifically (keeping the assistant itself local) —
a real, separate decision from what model the assistant uses to answer.

## Compatibility shim: stubbing a dead import instead of installing 40MB

Ragas 0.4.x unconditionally imports `ChatVertexAI` at package load time,
even though we never use Google Vertex AI. `langchain-community` split that
integration out into its own package (`langchain-google-vertexai`), which
pulls in ~40MB of Google Cloud SDKs (`google-cloud-aiplatform`, BigQuery,
etc.) we would never use, just to satisfy one import line ragas doesn't
actually need at runtime. `eval/evaluate.py` instead injects a fake module
into `sys.modules` before `import ragas`, so the dead import succeeds
without installing any of it. A pragmatic workaround for a real upstream
bug (an import that should be optional/lazy but isn't), not a hack around
our own code.

## Phase 4, experiment 1: top_k=8 vs. baseline top_k=4 (a corpus-size lesson)

Ran `experiments/run_experiment.py topk8 --top-k 8`. Result, logged in
`experiments/experiments.csv`:
- `faithfulness`: 0.858 → 0.848 (flat, within noise)
- `context_recall`: 0.817 → **0.625** (dropped notably)

Counterintuitive at first: retrieving *more* chunks made a metric *worse*.
Two things explain it, both confirmed directly rather than guessed:

**1. Our sample corpus has exactly 8 total chunks** (`store._collection.count()`
confirmed this). `TOP_K=8` doesn't mean "retrieve more selectively" — it
means *retrieve the entire corpus, every time, for every question,
regardless of relevance*. There's no discrimination left to test; `top_k=8`
on an 8-chunk corpus is a fundamentally different experiment ("dump
everything" vs. "pick the best 4") than `top_k=8` would be on a real corpus
with hundreds of chunks.

**2. More retrieved text is a harder task for a small local judge, not just
a bigger haystack for the real answer to hide in.** `context_recall`'s judge
prompt has to reason over all retrieved text to decide whether the
reference is supported. Doubling that text (4 chunks → 8) gives our 3B
judge more to parse and more distractor content to reason past, and we
already know from Phase 3 that this judge model's reliability degrades with
prompt complexity. The drop may partly reflect genuine noise dilution in
retrieval, and partly reflect the judge itself getting confused by a longer
prompt — this setup can't fully separate the two.

**Practical takeaway (revised after experiment 2 below): this turned out
wrong.** Chunk-size experiments hit the exact same saturation problem.

## Phase 4, experiment 2: chunk_size=1000 vs. baseline chunk_size=500

Ran `experiments/run_experiment.py chunk1000 --chunk-size 1000
--chunk-overlap 100` (top_k left at the default, 4). Larger chunks collapsed
our 2 tiny sample docs into **only 4 total chunks** (down from 8 at
chunk_size=500). With `top_k=4` and only 4 chunks *existing*, this
experiment retrieves the entire corpus every time too — the same
saturation problem as experiment 1, just reached from the opposite
direction (shrinking total chunks instead of growing k).

Result: `faithfulness` flat (0.858→0.856), `context_recall` dropped again
(0.817→0.597) — consistent with experiment 1's pattern, and for the likely
same reason (a real local-judge weakness: larger/more-complete context
blocks are harder for a 3B judge to parse reliably, independent of whether
retrieval itself got better or worse).

**Generalized conclusion, revised from experiment 1's narrower one: this
project's 2-document sample corpus (a combined ~60 lines of text) is too
small to run *any* meaningful retrieval-selectivity experiment.** Whenever
`total_chunks` is close to `top_k`, every config collapses to "retrieve
everything," and there's no selection pressure left to measure. To get
experiments that actually isolate "did retrieval get better" from "did the
judge get confused by more text," the corpus needs to be large enough that
`total_chunks >> top_k` by a healthy margin (dozens of chunks at minimum,
ideally more) — a real prerequisite for Phase 4 to produce trustworthy
conclusions, not just a nice-to-have.

## Phase 4, experiments 1v2/2v2: rerun on an expanded corpus (the valid results)

Added 9 more sample docs (`data/sample_docs/`), growing the corpus from 2
docs / 8 chunks to 11 docs / **64 chunks** at the baseline chunk size. Now
`top_k=4` is ~6% of the corpus, not 50% — real selection pressure exists.
Re-ran ingestion, regenerated the eval dataset's answers, and reset
`eval/baseline_scores.json` against this corpus: `faithfulness` 0.893 (16/20
scored), `context_recall` 0.857 (20/20 scored) — both slightly higher than
the old toy-corpus baseline, and for the first time individual questions
show real recall variance (some questions score context_recall as low as
0.15-0.2) instead of every question saturating near 1.0.

Reran both experiments against this baseline:

| Config | faithfulness | Δ | context_recall | Δ |
|---|---|---|---|---|
| baseline (chunk=500, top_k=4) | 0.893 | — | 0.857 | — |
| top_k=8 (64 chunks total) | 0.877 | -0.015 | 0.775 | **-0.082** |
| chunk_size=1000 (29 chunks total) | 0.868 | -0.025 | 0.718 | **-0.138** |

**This time the drops are real signal, not corpus saturation** — top_k=8 is
still only 12.5% of a real 64-chunk corpus, and chunk_size=1000 produced a
genuine 29-chunk corpus (13.8% selection ratio), nowhere near saturated.
Both drops are smaller than the invalid v1 runs (top_k=8 v1: -0.192; here:
-0.082), confirming part of the original effect really was saturation
artifact — but a real, smaller negative effect remains even on a properly
sized corpus.

**Best-supported explanation:** both alternate configs hand the judge model
*more text per question* to reason over (8 chunks or bigger chunks, instead
of 4 smaller ones), and we already know from Phase 3 that this local 3B
judge's reliability degrades with prompt complexity/length. This doesn't
cleanly separate "the assistant's retrieval got worse" from "the judge got
worse at scoring it" — that ambiguity is a real, disclosed limit of
evaluating with a small local judge, not a gap in the experiment design.

**Conclusion for this project: the baseline config (chunk_size=500,
top_k=4) beats both alternatives tried so far on the metrics that matter.**
Neither larger chunks nor more retrieved chunks helped; both cost faithfulness
and context_recall. Worth trying smaller/opposite directions (e.g.
chunk_size=200-300, top_k=3) in a future session, and revisiting with a
stronger judge model before fully trusting the magnitude of these deltas.

## Phase 4, experiment 4: chunk_size=250 (and a real bug in the experiment tool)

Ran `chunk_size=250/overlap=25` against the 64-chunk baseline corpus
(produced a 130-chunk corpus - smaller chunks, so more of them). This run
surfaced two separate, real problems worth understanding on their own.

**Bug found: NaN silently poisons a naive mean.** `experiments/run_experiment.py`
originally computed `sum(scores) / len(scores)` in plain Python. Ragas can
legitimately *return* `NaN` (not raise an exception) when an answer has zero
extractable factual claims - e.g. "No statements were generated from the
answer" for a very short/refusal response, making faithfulness
mathematically 0/0. A single NaN in a list poisons `sum()` for the whole
list (`sum([0.5, nan, 0.8])` is `nan`, not an error). The first `chunk250`
run reported `faithfulness: nan`, useless as a result. `eval/evaluate.py`
never had this bug because pandas' `.dropna().mean()` already skips NaN
correctly - the bug was specific to the hand-rolled aggregation in the
experiment runner. Fixed by explicitly checking `score != score` (the
cheapest NaN test) and excluding it, same as a caught exception.

**Bigger problem, revealed only after fixing the bug: `chunk_size=250` has
an unusually high judge-failure rate - 13/20 (65%) of faithfulness scoring
attempts failed**, leaving only 7 successful scores. Those 7 happened to
average a perfect 1.000, which the tool's comparison printed as
"faithfulness up +0.107" - looking like a real win. **It is not.** This is
survivorship bias: whichever questions happened to produce judge-parseable
output were plausibly the "easier" ones for the judge to reason about in
the first place, while harder cases failed and got silently dropped from
the average entirely. A mean computed over 35% of the data is not
comparable to the baseline's mean over 80%+ of its data, even though both
are technically valid floats.

**Fix, generalized beyond this one run:** `run_experiment.py` now tracks and
reports `scored/total` counts per metric alongside every mean, and flags
any comparison below 70% scored as unreliable in the console output. The
real lesson: **an evaluation mean is only as meaningful as its sample
size** - a metric that fails unpredictably doesn't just add noise, it can
silently and systematically bias *which* questions survive into the
average, in a direction you can't know without checking the denominator.

**What's actually trustworthy from this run:** `context_recall` scored
20/20 both times (0.857 baseline -> 0.829 experiment, a real small drop) -
that comparison is fair. `faithfulness` at chunk_size=250 is simply
**unknown** with this local judge - not "1.000," not "worse," genuinely
unmeasured, because 65% of the attempts to measure it failed.

Also worth noting: `context_recall` itself moved slightly between the two
scoring attempts on the *identical* results (0.864 -> 0.829, both 20/20
scored) - a reminder that even the "reliable" metrics have some run-to-run
judge noise, just far less than the metrics that fail outright.

## Phase 4, experiment 5: top_k=3 vs. baseline top_k=4

Result: `faithfulness` 0.893→0.854 (only 12/20 scored, 60% - below the 70%
reliability threshold, so directional only, not fully trustworthy).
`context_recall` 0.857→0.803, **20/20 scored both sides - fully reliable**,
a real drop of 0.054.

This is the most intuitively sensible result of all four experiments so
far: retrieving *fewer* chunks (3 instead of 4) means a higher chance the
one chunk that actually had the answer didn't make the cut, so recall
dropping makes direct sense, unlike the top_k=8/chunk_size=1000 results
where *more* context somehow also hurt (likely a judge-reliability effect,
not a retrieval effect - see above).

**Running scoreboard (context_recall, the metric with consistently full
scoring across every experiment run):**

| Config | context_recall | vs. baseline |
|---|---|---|
| baseline (chunk=500, top_k=4) | 0.857 | — |
| top_k=3 | 0.803 | -0.054 |
| top_k=8 | 0.775 | -0.082 |
| chunk_size=1000 | 0.718 | -0.138 |
| chunk_size=250 | 0.829 | -0.027 |

**The baseline still wins against every alternative tried.** Interesting
shape to the results: both directions away from top_k=4 (3 and 8) hurt
recall, and both directions away from chunk_size=500 (250 and 1000) also
hurt it, with chunk_size=250 being the closest challenger (-0.027, small).
Nothing tried so far beats the original defaults - a mild but real
validation that the initial config wasn't arbitrary.

## Adding a real document (QueryCraft.pdf) and a new baseline

Added the user's own 62-page college project report as a real external
document (`data/my_docs/QueryCraft.pdf`), growing the corpus to 73
documents / 297 chunks. Added 9 answerable + 1 unanswerable question about
it to `eval/dataset.jsonl` (30 questions total now), written from source
text read directly via `pypdf` rather than guessed. Regenerated and
rescored: new baseline `faithfulness` 0.744 (24/30 scored), `context_recall`
0.787 (30/30 scored) - both notably lower than the old 20-question,
2-sample-doc baseline (0.893/0.857), which makes sense: a bigger, more
varied real corpus is a genuinely harder retrieval task than 2 tiny
synthetic docs.

**A live example of the recall-miss discussed earlier, before it was even
scored:** asking the assistant "What problem does QueryCraft solve and what
research gaps does it identify?" (manually, via `src.ask`) retrieved the
cover page and conclusion instead of the actual section 2.4 (the four
research gaps) - a real context_recall failure on real content, and the
model correctly said "I don't know the specific gaps" rather than
inventing plausible-sounding ones. Caught this by knowing the report well
enough to notice, exactly the caution flagged back in Phase 2 about
eval-writer bias - this time it worked in our favor as a sanity check.

**A second, more surprising local-judge quirk found by spot-checking the
scores against the actual generated text (q25, q28, q30 all scored
`faithfulness=0.0`):** all three were honest "I don't know" refusals, not
hallucinations - the assistant behaved exactly right. But the faithfulness
metric scored them as complete failures instead of excluding them the way
it excluded other refusals as NaN ("no statements generated") earlier in
Phase 4. So the judge is **inconsistent specifically on refusal answers**:
sometimes it correctly recognizes there's nothing to check and returns NaN,
other times it seems to manufacture some claim from a refusal sentence and
then judges that claim unsupported, scoring 0 for a *correct, faithful*
response. Confirmed by reading the raw generated answers, not assumed from
the number alone - the same discipline as everywhere else in this project:
a metric looking like a failure is a hypothesis to check against the real
text, not a fact to report directly.

**Practical consequence:** the raw `faithfulness=0.744` baseline mean
likely *understates* true faithfulness, since at least 3 of the 24 scored
questions are misclassified correct refusals, not real failures. This is
one more reason relative comparisons (delta vs. this same baseline, same
judge) are safer to trust than the absolute number in isolation - a lesson
that's come up repeatedly this phase.

## Phase 4, experiment 6: embedding model swap (the first real winner)

Swapped `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 22M params) for
`sentence-transformers/all-mpnet-base-v2` (768-dim, 110M params), same
chunk_size/top_k as baseline. Result: `faithfulness` 0.744→0.872 (+0.129,
24/30 scored both sides), `context_recall` 0.787→0.815 (+0.028, 30/30
scored both sides). **Both metrics improved, with matching/full sample
sizes on both sides** - the first experiment all session that isn't
undermined by a reliability caveat.

Mechanically sensible: mpnet is a larger, generally stronger embedding
model. Better embeddings -> the right chunks get found more often
(context_recall) -> generation has better material to stay faithful to
(faithfulness). This is the most legitimate "config B beats config A"
result produced so far.

**Real costs of adopting it, not just upside:** mpnet is ~4.5x the
parameters of MiniLM, meaning a larger one-time download, slower embedding
at ingest time, and slower query-time embedding for every question asked -
a real latency cost for a fully local setup with no GPU assumed. Whether
that tradeoff is worth a +0.129 faithfulness / +0.028 context_recall gain
depends on how latency-sensitive the actual use case is - not something
the eval harness itself can decide, since it only measures quality, not
felt speed.

<!-- Decision pending: adopt mpnet as new default embedding model in
src/config.py, or keep MiniLM for speed. -->

## A real retrieval failure, live: "tell me about this project"

The user hit this directly, not as a designed eval question. Asking the
chat UI "tell me about this project" (12-document corpus, including the
user's own `QueryCraft.pdf`) returned an answer built from acknowledgement/
certificate boilerplate — not the actual project description. The model
didn't hallucinate (it honestly said it didn't know the details), so
faithfulness held; retrieval failed it.

**Checked the actual retrieved chunks rather than guessing why:**
```
[1] "...unwavering motivation and immense patience, which sustained us
     through the challenges of this project."
[2] "...project. Her timely advice and encouragement were the foundation
     of this project..."
[3] "ACKNOWLEDGEMENT — We extend our sincere gratitude..."
[4] "...is a bona fide record of project work carried out..."
```

**Why:** "tell me about this project" is mostly generic filler words. It has
little semantic content of its own, so its embedding ends up closest to
whatever text repeats those same generic words most densely — the
acknowledgement section says "this project" over and over, while the
actual abstract describes *what QueryCraft does* using specific technical
language that shares almost no surface phrasing with the query. A vague
query has nothing distinctive to match against.

**Tested whether document-scoping alone would fix it — it didn't.** Scoping
to just `QueryCraft.pdf` returned the *exact same four chunks*. Checked
`similarity_search_with_score` at `k=15` scoped to that one document: the
actual substantive chunks (the abstract, "web interface built with Next.js
and a robust Node.js backend") ranked **11th and 12th** — present in the
corpus, just far outside `top_k=4`. So the failure wasn't really
cross-document ambiguity (in this case); it was that boilerplate can
out-rank substance *within a single document* for a vague query, and no
amount of scoping fixes a `k` that's too small to reach past it.

**What actually fixed it, verified both directly in Python and live through
the UI:** scoping to one document *and* raising `top_k` for that scoped
search (`SCOPED_TOP_K=12` in `app.py`, vs. the corpus-wide default of 4).
Scoping alone controls *which document* competes; a larger `k` controls
*how far down the ranked list* you're willing to look within it. Both
were necessary — neither alone fixed the exact failing query.

**General lesson:** when a retrieval failure looks surprising, check the
actual ranked chunks and their scores before changing anything. "Scoping
should obviously fix this" was a reasonable-sounding hypothesis that turned
out to be only half right — the fix that actually worked came from looking
at where the right content ranked, not from intuition about what a
sensible-sounding feature "should" do.

## Round 2: raising SCOPED_TOP_K to 20 fixed one query and exposed a new failure

Asked to push `SCOPED_TOP_K` from 12 to 20 and re-test both queries. Checked
first whether 20 would even be enough: `similarity_search_with_score` at
`k=30` showed the real "9.2 Limitations" content (mentioning "Token Limit
Constraints") ranked **16th** — so 20 should include it, unlike 12.

**"Tell me about this project" — still works correctly at k=20.**

**"What are the limitations of this project?" — retrieval fixed, but
generation now fails differently.** The correct chunk (rank 16, containing
the real "Token Limit Constraints" / "Visualization Limits" content) *was*
included this time. But the model answered instead from chunk [10] — a
table-of-contents fragment:
```
9.1 Conclusion         42
9.2 Limitations of the Current System      43
9.3 Future Enhancements        43
9.3.1 Privacy-First "Local LLM" Deployment     43
9.3.2 Advanced Data Visualization Engine      44
...
```
The model presented 9.3's *sub-heading titles* (Future Enhancements) as if
they were 9.2's *content* (Limitations) — reproduced identically both via
direct Python and live through the browser UI, not a one-off fluke.

**Why the TOC chunk won despite the real content also being in context:**
the TOC fragment contains the literal phrase "9.2 Limitations of the
Current System" — a near-exact lexical match to the question ("limitations
of this project"). It's pure navigational metadata, not informational
content, but it scores as an extremely strong match precisely because it
repeats the section's own title. The genuinely relevant chunk, by
contrast, only contains the *specifics* of the limitations (token limits,
visualization limits) without repeating the word "limitations" itself.

**This is now a generation-side failure, not a retrieval-side one** — the
right material was present in the prompt; the model chose the wrong
adjacent chunk to answer from. More `top_k` can't fix this, and might
make it worse (more chances for another TOC-like fragment to out-rank real
content lexically). The more targeted fix would be at ingest time: table-
of-contents pages are navigational aids for a human reader, not
informational content, and arguably shouldn't be indexed at all - they
exist purely to create exactly this kind of false-positive lexical match.
Not implemented yet - flagged as a real, evidence-backed next step rather
than guessed at.

## Round 3: implementing the TOC filter

Added `is_toc_like()` to `src/ingest.py`: a page is TOC-like if ≥50% of its
non-empty lines end in a bare 1-4 digit number (the fingerprint of a
"Section Title ... 43" style navigational line). Validated the heuristic
directly against every page of `QueryCraft.pdf` *before* wiring it in:

- Flagged exactly 6 pages: the 4-page table of contents, List of Figures,
  List of Tables (fractions 0.60-0.97).
- Correctly did **not** flag the references/bibliography page, despite it
  being full of citation years like "(2021)" - fraction 0.05, because
  those numbers land mid-sentence, not at line-ends.
- No page fell in an ambiguous middle ground between real content and TOC
  (nothing between the 0.05 references page and the 0.60 lowest TOC page).

Wired into `load_documents()` so flagged pages are dropped before they're
ever chunked or embedded - cheaper and more effective than filtering
chunks after the fact, since a TOC page's harmful lexical density survives
chunking intact (it would just produce several TOC-flavored chunks instead
of one).

**Result after rebuilding:** 73→67 documents, 297→283 chunks (exactly the
6 flagged pages removed). Re-ran both problem queries, verified both
directly in Python and live in the browser:
- "Tell me about this project" - still correct.
- "What are the limitations of this project?" - **TOC-hijacking is gone.**
  The model now correctly leads with "Token Limit Constraints" and
  "Visualization Limits" (the real 9.2 content). It still pulls in one
  loosely-related item ("Domain Rigidity," from a different section about
  problems with *other* tools, not QueryCraft's own limitations) - a much
  milder imperfection than confidently answering from the wrong section
  entirely, and consistent with everything else found this session: there
  is no single change that makes retrieval perfect, only ones that measurably
  reduce specific, identified failure modes.

## Round 4: closing the eval-coverage gap for real

The dip in Round 3's baseline (0.744→0.724 faithfulness) happened because
the 30-question dataset didn't contain any scoped/vague queries at all -
the exact failure mode being fixed was invisible to the eval harness.
Fixed properly rather than just noted:

- Added `scope_source` as an optional field on eval dataset entries.
  `eval/run_eval_dataset.py`'s `run()` checks for it and retrieves the way
  `app.py`'s document picker actually does - filtered to that source, at
  `config.SCOPED_TOP_K` instead of the corpus-wide `config.TOP_K`.
- Moved `SCOPED_TOP_K` (=20) out of `app.py` and into `src/config.py`, so
  the app and the eval harness can't silently drift to different values.
- Added `q31` ("tell me about this project") and `q32` ("what are the
  limitations of this project?") with `scope_source` set to
  `QueryCraft.pdf` - literally the two queries that exposed the TOC bug in
  the first place, not new ones invented for the occasion.

**Result, regenerated and rescored (32 questions):** baseline
`faithfulness` 0.772 (27/32 scored), `context_recall` 0.780 (31/32
scored). The interesting comparison is inside the dataset now, not just in
chat history:

| Question | Scoped? | Faithfulness | Context Recall |
|---|---|---|---|
| q28 - "two limitations of QueryCraft's current system" (unscoped, `top_k=4`) | No | 0.500 | 0.286 |
| q32 - "limitations of this project" (scoped, `top_k=20`) | Yes | **1.000** | **1.000** |

Same underlying fact, asked two ways. The unscoped version - competing
against the entire 283-chunk, 12-document corpus at `top_k=4` - still
struggles. The scoped version, now that TOC pages are filtered out and
enough chunks are retrieved within the one relevant document, scores
perfectly. This is the clean, measured version of the improvement earlier
rounds could only demonstrate by hand.

**Lesson under the lesson:** the same failure that motivated adding these
questions (a fix whose benefit didn't show up in the aggregate score)
would recur silently for any future fix in this area if the coverage gap
weren't closed. An eval dataset only measures what it happens to contain -
expanding it deliberately when a new class of behavior is discovered is
part of the eval work, not a one-time setup step.
