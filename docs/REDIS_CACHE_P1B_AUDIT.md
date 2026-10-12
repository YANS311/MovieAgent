# Redis cache P1-B: hybrid recommendation payload

## Preflight findings

* User/catalog generation lookup raises into the key-builder fallback and
  returns `None`; recommendation code then skips Redis. Failed `get/add` does
  not substitute a constant generation token.
* `partial_success` is not treated as full success by existing feedback
  handlers. Feedback pages only display success for their explicit `success`
  status; some silently show no message for partial completion. The exclusion
  restore handler takes its non-`ok` path and shows the response message.
* `/agent/cache/retry/` requires login and POST, is not CSRF-exempt, and the
  configured `CsrfViewMiddleware` is enabled. A user may invalidate only their
  own cache; staff may invalidate another user or catalog scope. Repeating the
  operation is safe for data correctness and never repeats a database write,
  though each call rotates the generation again and can cause another cold
  cache period. This is semantic idempotency, not a no-op-on-replay guarantee.
* No high-severity preflight issue was found. No feedback UI change was included
  in P1-B.

## Cache payload and result contract

The previous `hybrid_recall_recommend()` cached the full result tuple containing
`Movie` instances, profile text, and source stats. The new cache value is a
plain dictionary with schema `hybrid_rec_payload_v1`, ordered `{movie_id,
rrf_score}` items, `profile_text`, and `source_stats` (with the duplicate
`profile_text` member omitted). The key family changed to
`hybrid_rec_payload` under the existing per-user/per-catalog generations, so
the old ORM-object cache family is not read. TTL remains 1800 seconds.

Cold computation still returns `(movies_list, profile_text, source_stats)` and
assigns each movie the same rounded `rrf_score`. It stores only IDs/scores and
plain profile/stat data. A cache hit validates schema and fields, performs one
`Movie.objects.filter(id__in=...)` query with the existing `prefetch_related`
relations, restores cached order and scores, and returns the same tuple shape.
It restores `source_stats['profile_text']` for callers that depend on that
field.
Live movie fields and relations are fetched from ORM, so metadata field changes
do not come from stale model snapshots. If a referenced movie was deleted or
payload fields are malformed, that cache entry is deleted best-effort and the
normal recall path recomputes. Database query errors still propagate.

The key retains `top_k` and user isolation via existing helpers. Generation
rotation makes previous payloads unreachable after user/catalog changes. The
new schema and key family prevent old tuple payloads from being accepted.

## Validation and limits

`tests/test_redis_cache_p1b_isolated.py` covers cold/hot result equivalence,
order, rounded scores, caller-used movie fields, schema rejection, deleted movie
recompute, Redis read/write failure fallback, generation invalidation, and one
batched movie query plus prefetch on hydration. These checks use AST extraction
and Mock/FakeCache only. They do not establish real ORM behavior, SQL query
counts, Redis serialization, or production performance.

No cache hit/miss counters, timing logs, metrics endpoint, or external
monitoring service were added. Existing cache helpers report backend errors;
future opt-in instrumentation could record hit/miss, serialization/hydration
duration, ORM query counts, exceptions, and endpoint percentiles without
logging user query text. No latency or hit-rate claim is made.
