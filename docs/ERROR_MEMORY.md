# Error Experience Retrieval

## Retrieval Order

1. Normalize the observed tool error and try the existing exact substring match.
2. If no exact match exists, consider global experiences and experiences from the
   same tool only. Exclude known incompatible error causes, dependency names and
   attribute names before ranking.
3. Embed the sanitized error and compare it with candidate error descriptions by
   cosine similarity. Return at most one semantic experience. Below-threshold
   scores or close alternatives with different fixes return no experience.
4. Send the experience back through the existing `error_recovery` tool result.
   It is a suggestion, not a verified repair. Input permissions, allowed execution
   changes and retry budgets remain unchanged.

Tool schema validation failures return their field errors and retry budget without
experience retrieval. LangChain invokes that callback synchronously, so it must
not perform database or embedding requests on the agent event loop. Actual tool
execution failures still retrieve experiences through a worker thread.

The semantic layer uses the same embedding configuration as document RAG, currently
`qwen3.7-text-embedding-flash`; it does not use document chunks, reranking, or the
RAG evidence judge. Similarity is not a probability that a repair is correct.

## Storage And Cost

The additive PostgreSQL table `error_memory_vectors` is created on first semantic
index access. It stores hashed entry IDs, content hashes, model profiles and
`REAL[]` vectors; no pgvector extension is required. Chat and knowledge tables are
not modified. Error patterns and fixes still use the existing error memory store.

Newly learned experiences are indexed after recording. Old eligible experiences
are indexed lazily on the first semantic miss. Changing the endpoint/model/vector
dimensions gives a different profile and triggers lazy reindexing. Changed fixes
invalidate their cached entry. Stale vectors cannot supply removed experiences.

Exact matches make no embedding request. Query embeddings use an in-process LRU
cache of 256 normalized errors. Stored experience vectors survive restarts through
PostgreSQL; query embeddings do not. Concurrent duplicate lookups share the cache.
Paths, URLs and recognized credentials are removed before embedding; only the last
2,000 normalized error characters are sent, not uploaded files or file contents.
Error text may still contain application-specific details: do not put secrets or
document contents into exception messages.

Embedding requests time out after 15 seconds per HTTP operation. Failures suspend
semantic requests for 60 seconds; exact retrieval remains available. A database
outage retains vectors in process memory and retries pending writes after 60
seconds. A process restart during that outage may require re-embedding.

## Configuration

Use the existing server-side `RAG_EMBEDDING_*` settings. Optional settings in
`.env` (also propagated by `docker/docker-compose.full.yml`):

```dotenv
ERROR_MEMORY_VECTOR_ENABLED=true
ERROR_MEMORY_MIN_SCORE=0.52
ERROR_MEMORY_AMBIGUITY_MARGIN=0.08
```

The thresholds are starting values checked with synthetic Flash embeddings, not
calibrated accuracy claims. A high CUDA similarity alone is not sufficient: for
example, an invalid GPU device ID must not retrieve an out-of-memory experience.
Validate them with representative positive and negative errors before changing
them. `false` disables only semantic retrieval, preserving exact matches.
Restart the backend after configuration or code changes. No frontend change is
required. Logs report a semantic match and its cosine score without raw errors.

## Verification

```bash
python -m unittest discover -s tests -p 'test_error_memory_vectors.py' -v
python -m unittest discover -s tests -p 'test_error_recovery.py' -v
ERROR_MEMORY_TEST_DB=1 python -m unittest discover -s tests -p 'test_error_memory_vectors.py' -v
```

The last command uses an isolated temporary PostgreSQL schema and removes only
that test schema afterward. Unit tests use synthetic embeddings, not paid API
calls. They cover paraphrases, cross-tool isolation, incompatible causes,
ambiguous/low scores, caching, outage recovery, persistence, profile changes,
credential sanitization and the actual wrapped-tool error result.
