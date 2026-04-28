# CLAUDE.md — BRN_KB_REVIEW

## Running Tests

Tests run inside Docker (they need MySQL access via `.env`).

```bash
# Rebuild and run tests (always rebuild after code changes)
docker compose -f docker-compose.test.yml build && docker compose -f docker-compose.test.yml run --rm test-runner

# Run a single test file
docker compose -f docker-compose.test.yml build && docker compose -f docker-compose.test.yml run --rm test-runner poetry run pytest tests/test_judge_aggregator.py -v --tb=short

# Run a single test class/method
docker compose -f docker-compose.test.yml build && docker compose -f docker-compose.test.yml run --rm test-runner poetry run pytest tests/test_review_runner.py::TestRunReviewWithLLMFindings -v --tb=short
```

### Test safety rules

- **Agents table is read-only.** Tests must never INSERT/UPDATE/DELETE agents. Agent lookups are mocked via `_lookup_agent` in `review_api.py` and `review_runner.py`.
- **No destructive teardowns.** Tests run against the shared production DB. Session teardown must not DROP tables or DELETE all rows. Individual fixtures clean up only the rows they created.

## Running the Service

```bash
# Rebuild and start
docker compose build && docker compose up -d

# View logs
docker compose logs -f kb-review

# Restart after code changes
docker compose build && docker compose up -d
```

## Project Structure

- `app/analysis/` — Core pipeline: `review_runner.py` (orchestrator), `analyzers.py` (LLM prompts), `judge_aggregator.py` (consensus), `pre_filter.py` (candidate pairs), `training_quality.py` (structural checks)
- `app/database_layer/` — SQLAlchemy models, schemas, config
- `app/cache_db/` — Redis client for token blacklist checks
- `app/graph_db/` — Neo4j reader
- `app/llm/` — Judge factory (LiteLLM wrapper)
- `app/api/auth.py` — Authentication (`require_auth` dependency, `CallerContext`)
- `app/api/endpoints/review_api.py` — FastAPI endpoints
- `tests/` — pytest suite (unit + integration)

## Pre-existing Issues

When you encounter pre-existing failures (broken tests, stale assertions, etc.) during your work, fix them if it's a quick fix. If it's a big effort, document them as a TODO comment in the relevant file or test with a brief explanation of what's wrong.

## Async Architecture (Phase 3)

Phase 3 (LLM judges) uses `asyncio.run()` from the sync `run_review()` entry point. Inside, judges and batches run concurrently via `asyncio.gather` + `ainvoke`. Key things to know:

- **`analyzers.py` functions are `async def`** — they use `await judge.ainvoke()`, not `judge.invoke()`. Test mocks must use `AsyncMock` for `ainvoke`.
- **DB access is sync** (SQLAlchemy) — serialized via `asyncio.Lock`. The lock is per-review, not global.
- **`asyncio.run()` creates a new event loop** in the background thread. If `run_review()` is ever called from an async context, change to `await _run_judges_parallel(...)` directly.
- **Rate limiters** are created per `(provider, model)` and shared across judges. They're created before `asyncio.run()` but work correctly inside the loop.

## Auto Mode & Sync Mode

`POST /reviews` supports two modes via the `judges` field:

- **Auto mode** (judges omitted): 3x Gemini Flash no-reasoning with Haiku fallback if Gemini is unavailable. Config constants are in `review_api.py` (`AUTO_PRIMARY_JUDGES`, `AUTO_FALLBACK_JUDGES`).
- **Custom mode** (judges provided): uses specified judges, no fallback.

Sync mode (`?wait=true`): blocks until completion (max 5 min), returns 200 with full results. On timeout returns 202 — review continues in background. Uses `run_in_executor` (not `BackgroundTasks`) because `BackgroundTasks` fires after the response is sent.

**Auto-failover** (`review_runner.py`): When all primary judges fail on probe, catches `AllJudgesFailedError`, cleans up primary artifacts (stats, issues, judge results), rebuilds judges from `fallback_judges` config, and retries Phase 3.

## Change Detection & Carry-Forward

Subsequent reviews for the same agent skip LLM calls for unchanged document pairs. Per-document SHA-256 content hashes are stored in `kb_review_doc_hashes` (keyed by `agent_id` + canonical source). On each run:

1. **Compute hashes** for all current documents (always, regardless of `carryforward` flag)
2. **Find previous compatible review** — most recent COMPLETED review with matching `analysis_types`, `similarity_threshold`, and `max_candidate_pairs` (judge selection doesn't matter)
3. **Classify documents** into unchanged / changed / new / removed
4. **Split candidate pairs** — reusable (both sources unchanged) vs new (at least one changed)
5. **Carry forward findings** — copy issues + judge results from previous review for reusable pairs (`carried_forward=True`, `original_review_id` chains to first discovery)
6. **LLM judges** only process new pairs. Ambiguity analysis only samples from changed/new docs.
7. **Upsert hashes** in Phase 5 (always stored, even when `carryforward=false`)

### Pipeline phases with carry-forward
Phase 1 → Phase 1.5 (structural) → **Change Detection** → Phase 2 (pre-filter) → **Phase 2.5 (carry-forward, 25-30%)** → Phase 3 (LLM judges, 30-85%) → Phase 4 (consensus) → Phase 5 (finalize + hash upsert)

### API controls

- `POST /reviews?carryforward=false` — force full LLM analysis, skip change detection (hashes still stored)
- `GET /reviews/{id}/issues?carried_forward=true` — only carried-forward issues
- `GET /reviews/{id}/issues?carried_forward=false` — only freshly-analyzed issues
- `GET /reviews/{id}` — summary includes `previous_review_id`, `pairs_reused`, `pairs_analyzed`, `docs_changed`, `docs_unchanged`

### Auto-failover interaction
Failover cleanup preserves carried-forward issues (`.filter(carried_forward == False)` on delete queries).

## Authentication

Triple-path auth via `require_auth` FastAPI dependency (`app/api/auth.py`):

1. **API key + X-User-Id** (`X-API-Key` + `X-User-Id` headers): Gateway-forwarded. API key proves caller is trusted; user ID provides company-scoped access.
2. **API key alone** (`X-API-Key` header): Service-to-service calls. Full access, no user context. Env: `X_API_KEY`.
3. **Direct JWT** (`Authorization: Bearer <token>`): HMAC-SHA256, same secret as Gateway/Auth Service. Company-scoped. Env: `JWT_SECRET`. Checks Redis token blacklist (`BL_{token}`) and session invalidation (`TOKEN_VERSION:{userId}`) — same keys as the Gateway.

Health check requires no auth. All other endpoints require one of the three paths.

**Redis token blacklist** (`app/cache_db/redis_client.py`): On the JWT path, tokens are checked against Redis for explicit blacklisting (logout) and session invalidation (permission changes). Same Redis instance and key format as the Gateway/Auth Service. Env: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`. Behavior when Redis is not configured depends on `APP_ENV`: in production, JWT is rejected (503 fail-closed); in dev, a warning is logged and JWT is allowed through. When Redis is configured but unreachable, JWT is always rejected (503).

**Security**: Bare `X-User-Id` without `X-API-Key` is rejected (401). Since KB Review is standalone (not behind Gateway), accepting bare `X-User-Id` would allow anyone to impersonate any user.

**Company isolation**: JWT/Gateway callers can only access agents belonging to their company. Super admins (with `super_admin_access` permission) bypass company restrictions. API key callers have full access.

**Concurrency guard**: `POST /reviews` returns 409 Conflict if the agent already has a PENDING or RUNNING review.

**Audit trail**: `created_by_user_id` is set on every review — populated from JWT/Gateway user ID, NULL for API key callers.

**Info leak prevention**: Cross-company access returns 404 (not 403) to avoid revealing resource existence.

**Swagger docs**: Disabled in production (`APP_ENV=production`). Available at `/review-api/docs` in dev.

**Test auth**: Existing tests use `dependency_overrides[require_auth]` in conftest to bypass auth (acts as API key caller). Auth-specific tests in `tests/test_auth.py`.

## Issue Status Management

Issues support four statuses: `OPEN` (default), `RESOLVED` (KB was fixed), `DISMISSED` (false positive / won't fix), `ACKNOWLEDGED` (aware but not yet fixed).

- **`PATCH /reviews/{review_id}/issues/{issue_id}`** — update issue status. Blocked during PENDING/RUNNING reviews (409 Conflict). Same auth as viewing.
- **`issues_found`** counts active issues only (OPEN + ACKNOWLEDGED). RESOLVED and DISMISSED are excluded.
- **`issues_resolved`** counts RESOLVED + DISMISSED issues.
- **Atomic count recalculation** — PATCH uses raw SQL `UPDATE ... SET col = (SELECT COUNT...)` to avoid race conditions with concurrent updates.
- **Carry-forward** copies status fields (`status`, `status_updated_by`, `status_updated_at`, `status_note`) on unchanged documents. If a user resolved an issue and docs haven't changed, the carry-forward copy stays resolved.
- **`?status=` filter** on `GET /reviews/{id}/issues` — 422 on invalid values (not silent empty). Composes with `?include_minority=`, `?carried_forward=`.
- **Finalization + stale recovery** split counts into `issues_found` (active) and `issues_resolved`.
- **Status columns**: `KbReviewIssue.status`, `status_updated_by`, `status_updated_at`, `status_note`. `KbReview.issues_resolved`.
- **Tests**: `tests/test_issue_status.py`.

## MINORITY Consensus Filtering

All findings (including MINORITY) are persisted to the DB. The issues endpoint excludes MINORITY by default — add `?include_minority=true` to include them. Sync mode responses also exclude MINORITY by default.

## Review Status Values

`PENDING` → `RUNNING` → `COMPLETED` | `FAILED` | `PARTIAL`

- **PARTIAL**: Review crashed mid-run but some findings were already persisted to the DB. Findings are queryable via the API.
- **Startup recovery**: On startup, any RUNNING or PENDING review from before the current process started is recovered — marked PARTIAL if it has persisted findings, FAILED otherwise. See `_recover_stale_reviews()` in `app/main.py`.
