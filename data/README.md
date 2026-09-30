# Data

Drop the documents you want the assistant to answer questions about here
(`.pdf` or `.txt`). They're gitignored so your own files never get committed.

A set of small sample `.txt` files live in `data/sample_docs/` (11 files,
64 chunks at the default chunk size) so you can run the pipeline
immediately without supplying your own documents, and so Phase 4
retrieval-selectivity experiments (top-k, chunk size) have enough total
chunks to produce meaningful results — see CONCEPTS.md for why a tiny
corpus makes those experiments meaningless.
