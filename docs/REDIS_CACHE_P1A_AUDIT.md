# Redis cache P1-A consistency audit

Scope: online recommendation-affecting writes, post-commit invalidation behavior,
and Redis failure policy. No ORM cache-payload redesign, schema change, worker,
service startup, external datastore connection, or model execution was performed.

## Mutation entry points

| Data mutation | Entry point | Affected cache | Status |
|---|---|---|---|
| Rating create/update | `views.score_movie`, `views.movie_detail` | Current user's hybrid/CB/explain/agent-recall generation; legacy exact keys | User generation rotates with `transaction.on_commit()` after successful write |
| Rating delete | `views.admin_comment_delete`; default `UserRatingAdmin` single and bulk delete | Affected rating owner(s) | Owner is captured before delete; admin mixin schedules after commit |
| Collection add/remove/toggle | `views.add_collect`, `views.remove_collect`, `views.ajax_collect`; default `CollectAdmin` | Current/affected user | User generation rotates after commit |
| Dislike/undislike and recommendation feedback | `views.ajax_exclude_add`, `views.ajax_exclude_remove`, `agent_views.recommend_feedback_view`, `agent_views.admin_user_feedback_delete`; default `UserFeedbackAdmin` single/bulk delete | Current/affected user | User generation rotates after commit; add's delete+create is atomic |
| User profile/preferences | `views.center` profile form; `agent_views.admin_user_profile_edit/delete`; default `UserProfileAdmin`; `MemoryManager.persist_to_profile/_merge_to_profile` | Affected user's recommendation generation | User generation rotates after commit. Color-only profile changes do not affect recommendation inputs and are excluded |
| User lifecycle | `views.admin_user_add/edit/delete`; default `UserInfoAdmin` | New, edited, or deleted user | User-scoped generation rotates after successful database change |
| Catalog movie changes | `views.admin_movie_add/edit/delete`; default `MovieAdmin` | Catalog-dependent business caches for all users | Catalog generation token rotates after commit; cached values expire naturally |
| Catalog actor/genre/region changes | `views.admin_actor_add/edit/delete`; default `ActorAdmin` and `CatalogAdmin` | Catalog-dependent business caches for all users | Catalog generation token rotates after commit |
| Catalog admin bulk delete / M2M edit | Default catalog admin `delete_queryset` / `save_related` | Catalog-dependent business caches | Catalog generation token rotates after commit |
| Recent recommendation log changes | Default `RecommendLogAdmin` | Business rerank inputs for affected user(s) | Covered by user admin mixin; no online write endpoint found |
| Chat history clear/edit | `agent_views.ajax_clear_chat`, `agent_views.ajax_edit_last`, `views.chat_clear_history` | Chat history and current session memory only | Not a recommendation result cache input by itself. Slot reset uses strict session memory operations; no recommendation cache is cleared |
| Agent trace changes | Agent trace admin/agent execution | Trace display only | Not a recommendation candidate/ranking input |

### Not covered by online invalidation hooks

Static search found additional data-maintenance commands that write outside the
online/admin hooks:

| Maintenance command | Data it changes | Required invalidation if run against a live database |
|---|---|---|
| `update_maan_scores.py`, `calc_recs.py` | Replaces per-user or all `Rec` rows | Per affected user; all-user refresh requires iterating affected user IDs |
| `import_movielens.py` | Creates genres/users and replaces ratings | Catalog generation plus affected-user generations |
| `clean_data.py` | Merges movies, ratings, collections, and recommendation rows | Catalog generation plus affected users |
| `merge_and_clean.py`, `clean_metadata.py`, `patch_missing_movies.py`, `enrich_tmdb_data.py`, `import_new_movies.py` | Updates/imports movie metadata or movie records | Catalog generation |

They were not executed or changed. If these commands are used against a live
application database, their successful transaction/command completion must be
followed by per-user or catalog invalidation before declaring that cache fresh.
Direct ORM `update()/bulk_create()` or external data imports outside the listed
online/admin paths also bypass model-admin hooks. No signals were added.

## Commit and invalidation states

| Database result | Cache result | HTTP/behavior |
|---|---|---|
| Write fails or view returns JSON `status=error` | Callback discarded / transaction marked rollback | Original error response or exception; no success invalidation |
| Write commits | `on_commit` callback succeeds | Existing success response |
| Write commits | Redis invalidation fails | HTTP 200 `status=partial_success`, `database_committed=true`, `retry_mutation=false`, and `recovery=retry_cache_invalidation_only`; does not tell clients to repeat the mutation |
| Enclosing/outer transaction rolls back | Callback has not run | Database rolls back and Django discards callback |
| Nested savepoint commits | Callback remains queued | Runs only when the outermost transaction commits |
| Nested savepoint or outer transaction rolls back | Callback discarded | No invalidation for rolled-back write |

The `recommendation_write_view` wrapper puts online writes in an atomic block,
and `CommittedCacheFailureMiddleware` translates typed failures escaping an
outer transaction into the same committed-partial response. The explicit
`POST /agent/cache/retry/` endpoint retries invalidation only, with self-user or
staff authorization; it performs no database mutation. Its HTTP 503 means that
the invalidation-only retry failed and returns `retry_mutation=false`.

This is not a distributed transaction or durable outbox. If Redis remains
unavailable, old generations can remain reachable until the retry endpoint
succeeds or cache entries expire. There is no automatic retry queue or
eventual-completion guarantee in this worktree.

## Failure behavior

* Recomputable intent, hybrid recommendation, KG subgraph, movie-concept,
  multi-channel recall, query-rewrite, explain, and agent-recall cache reads/writes use fail-open helpers. A failed
  read acts as a miss; a failed write returns the computed result without
  caching. Cache-key generation failure also bypasses caching.
* Cache-key generations scope user data by user ID and recommendation family;
  hybrid and multi-channel recall keys include their result-shaping parameters
  (`top_k`, query, and weights as applicable). One successful user invalidation
  makes all parameter variants unreachable. Catalog generation
  scopes all catalog-dependent business caches without scanning/deleting keys.
* Session identity/authentication cache access was not changed to fail-open.
  A session backend error remains an error. Persisted preference memory is
  strict: failed reads/writes/deletes raise `MemoryCacheUnavailable` instead of
  silently erasing or fabricating preference state.
* Old cache values are not globally deleted. Exact legacy per-user recommendation
  keys are deleted during successful invalidation; generation-versioned old
  keys age out by TTL. Session keys are not part of either invalidation path.

## Test boundary

The isolated suites execute selected source functions with FakeCache, fake
transactions, fake responses, and AST checks. They do not initialize Django,
`AppConfig.ready()`, model imports, PyTorch, Transformers, CUDA, MySQL, Neo4j,
real Redis, or SessionMiddleware. Cookie/session persistence through the real
Django middleware stack remains an integration verification item.

Commands used for P1-A validation:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p0_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p01_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1a_isolated.py
```

The first two suites retain the prior 14 + 18 checks. P1-A adds 18 isolated
checks. Fake transaction semantics verify the callback ordering contract, not
database-driver behavior or real Django middleware behavior.
