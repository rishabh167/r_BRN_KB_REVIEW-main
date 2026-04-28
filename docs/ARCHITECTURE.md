# BRN_KB_REVIEW — Architecture

## Purpose

Agents in the Broadnet.ai platform can be trained with multiple KB documents that may contain conflicting, contradictory, or inconsistent information. This silently degrades RAG answer quality. This service automates detection of these issues using LLM judges.

## Service Overview

- **Standalone** — no Eureka, no Gateway dependency. Direct HTTP on port 8062.
- **Read-only Neo4j** — never writes to the graph database.
- **Shared MySQL** — uses the `brn_admin_panel` database (5 owned tables + 4 read-only).
- **Async + sync modes** — `POST /reviews` returns 202 by default (background processing); `?wait=true` blocks until completion and returns 200 with full results.
- **Triple-path auth** — API key (S2S), Gateway-forwarded user ID, or direct JWT with company-based agent isolation.

## Analysis Pipeline

```
POST /review-api/reviews
        │
        ├─ ?wait=false (default): 202 Accepted, background task
        └─ ?wait=true:  run_in_executor, return 200 on completion / 202 on timeout
        │
   Create review record (MySQL)
   Kick off run_review(id)
        │
   Phase 1 (0-10%):     Load chunks + entities from Neo4j
   URL dedup (10%):     Canonicalize URL sources, remove duplicate-variant chunks
   Phase 1.5 (10-15%):  Structural quality checks (no LLM)
   Change Detection:    Compute doc hashes, find previous review, classify docs
   Phase 2 (15-25%):    Pre-filter candidate pairs (uses deduped chunks)
   Phase 2.5 (25-30%):  Carry forward findings for unchanged pairs
   Phase 3 (30-85%):    LLM judge analysis (new pairs only, N judges × batches)
     └─ On AllJudgesFailedError: retry with fallback judges (auto mode only)
   Phase 4 (85-95%):    Multi-judge aggregation (majority voting, fresh issues only)
   Phase 5 (95-100%):   Persist results, upsert doc hashes → COMPLETED
        │
        ▼
   GET /review-api/reviews/{id}/issues → document-level findings
                                         (MINORITY excluded by default)
```

### Phase 1: Load Data

Looks up `agents.tenant_id` from MySQL, then queries Neo4j for:
- All `Document` nodes with that `tenant_id` (text, source, page, split_id, embedding)
- Entity-chunk mappings (`:MENTIONS` relationships)
- Entity-entity relationships

### URL Deduplication

Before Phase 1.5, URL-based sources are canonicalized (strip http/https, www, trailing slash) to detect duplicate pages ingested under different URLs. Duplicate chunks are removed before pair generation to avoid wasting LLM calls. The count is stored in `kb_reviews.url_duplicates_removed`.

### Phase 1.5: Training Quality

Structural checks — no LLM needed:
- **MISSING_EMBEDDINGS**: Chunks with null embedding vector (invisible to search)
- **EMPTY_CONTENT**: Chunks with empty/whitespace text (failed OCR)
- **URL_DUPLICATION**: Same page ingested under multiple URL variants (e.g., http vs https, www vs non-www)

### Phase 2: Pre-filter

Three strategies to find candidate pairs for LLM analysis:
1. **Embedding similarity**: Cosine similarity above threshold across different documents
2. **Shared entities**: Chunks from different documents sharing 2+ entities
3. **Relationship conflicts**: Same entity pair with different relationship types

Results are merged, deduplicated, and capped at `max_candidate_pairs`.

### Change Detection & Phase 2.5: Carry Forward

After Phase 1.5, the pipeline computes SHA-256 content hashes per document and compares them against stored hashes from the most recent compatible COMPLETED review for the same agent. Documents are classified as unchanged, changed, or new.

Candidate pairs are split into:
- **Reusable pairs** — both documents unchanged → findings are copied from the previous review (`carried_forward=True`, `original_review_id` chains to first discovery)
- **New pairs** — at least one document changed/new → sent to LLM judges

Config compatibility requires matching `analysis_types`, `similarity_threshold`, and `max_candidate_pairs`. Judge selection doesn't matter — findings are about content, not who found them.

When all documents are unchanged, the review completes with zero LLM calls. First review for any agent always does full analysis (no stored hashes to compare against).

Document hashes are always upserted in Phase 5, even when carry-forward is disabled via `?carryforward=false`.

### Phase 3: LLM Judges (Parallel)

All N judges run concurrently via `asyncio.gather`. Within each judge, batches also run concurrently after a probe batch validates the configuration.

**Execution model:**
- **Probe batch**: Batch 0 runs alone first — catches config/auth errors before burning API credits
- **Concurrent batches**: Remaining batches run in parallel, rate-limited by `InMemoryRateLimiter`
- **Incremental persistence**: Each batch's findings are persisted to DB immediately on completion (durable even on crash)
- **Quorum**: At least 1 judge must succeed. Failed judges are logged and skipped; if all fail, review is `FAILED`

**Rate limiting (two layers):**
- **Proactive**: `InMemoryRateLimiter` (LangChain built-in, token bucket) — one per `(provider, model)` pair, shared across judges using the same endpoint. Configured via optional `rate_limit_rpm` in JudgeConfig
- **Reactive**: `ChatOpenAI(max_retries=2)` — exponential backoff on 429 errors

**LLM details:**
- Pairs are batched (5 per LLM call) for efficiency
- Prompts include text excerpts, shared entities, and source document info
- Structured JSON output with severity, confidence, and reasoning
- JSON mode enforced via `response_format: {"type": "json_object"}` for all models
- Gemini models use temperature=1.0 (Google's recommendation); others default to 0.1
- Truncated responses are retried once automatically
- Ambiguity analysis (opt-in) runs concurrently on sampled individual chunks

**Recommended config:** 3x identical Gemini 3 Flash (no reasoning) judges. The consensus system filters noise — UNANIMOUS and MAJORITY findings are high-confidence; MINORITY findings are speculative.

**Auto-failover** (auto mode only): If all primary judges fail on the probe batch, `AllJudgesFailedError` is raised. The runner catches it, cleans up LLM-generated artifacts (judge results, LLM issues, judge stats — structural issues from Phase 1.5 and carried-forward issues are preserved), rebuilds judges from the `fallback_judges` config, and retries Phase 3. If the fallback also fails, the review is marked `FAILED` or `PARTIAL`.

### Phase 4: Aggregation

N-judge majority voting (applied only to freshly-analyzed issues — carried-forward issues retain their original consensus):
- **UNANIMOUS**: All judges agree → highest confidence
- **MAJORITY**: >50% of judges flagged → avg confidence × 0.9
- **MINORITY**: ≤50% flagged → avg confidence × 0.6
- **SINGLE_JUDGE**: Only 1 judge configured
- **STRUCTURAL**: Phase 1.5 issues (no LLM judges involved)

### Phase 5: Persist

All findings stored at document + page level (not chunks). Per-judge reasoning preserved for transparency. Document content hashes are upserted into `kb_review_doc_hashes` (sorted by source for consistent lock ordering). Hashes for removed documents are deleted. Finalization splits issue counts into `issues_found` (active: OPEN + ACKNOWLEDGED, excluding MINORITY) and `issues_resolved` (RESOLVED + DISMISSED, excluding MINORITY).

### Startup Recovery

On startup, the service records the process start time and runs `_recover_stale_reviews()` to detect reviews that were interrupted by a crash or restart. Any RUNNING or PENDING review created before the current process started is treated as orphaned — marked PARTIAL (if it has persisted non-MINORITY findings) or FAILED (if none). Recovery splits counts into `issues_found` (active) and `issues_resolved` to preserve status semantics. Using the process start time as the cutoff ensures even short-running reviews are recovered after a crash, without risking false positives on legitimately running reviews. This prevents orphaned reviews from permanently blocking agents via the concurrency guard.

## Auto Mode vs Custom Mode

`POST /reviews` supports two modes via the optional `judges` field:

- **Auto mode** (judges omitted): 3x Gemini Flash no-reasoning primary judges with 3x Haiku fallback. Config constants live in `review_api.py` (`AUTO_PRIMARY_JUDGES`, `AUTO_FALLBACK_JUDGES`). The `config_json` includes both `judges` and `fallback_judges`.
- **Custom mode** (judges provided): uses the specified judges exactly as given, no fallback. `config_json` has `judges` only.

## Sync Mode

- **Async** (default): `background_tasks.add_task(run_review, id)` → 202 Accepted
- **Sync** (`?wait=true`): `await asyncio.wait_for(loop.run_in_executor(None, run_review, id), timeout=300)` → 200 with `SyncReviewResponse` (review summary + filtered issues)
- **Sync timeout**: → 202 Accepted with current review state. The executor thread keeps running (Python threads can't be cancelled) — the review completes in the background.

Why `run_in_executor`: `BackgroundTasks` fires AFTER the response is sent, so it can't be used for sync mode. `run_in_executor` runs `run_review()` in the default thread pool. The inner `asyncio.run()` creates its own event loop in that thread — no conflict with the outer async endpoint.

## MINORITY Consensus Filtering

All findings (including MINORITY) are persisted to the DB. By default, the issues endpoint and sync responses exclude MINORITY findings. Add `?include_minority=true` to include them. This keeps the default output high-signal while preserving everything for deeper analysis.

## Document-Level Attribution

Users uploaded documents (PDFs, DOCs, URLs). All findings reference:
- **Document name** (e.g., "Product_Pricing_2024.pdf")
- **Page number** (e.g., page 3)
- **Relevant excerpt**

Chunks are internal — stored for debugging but never shown to users.

## LLM Provider Strategy

All providers expose OpenAI-compatible APIs — single `ChatOpenAI` class:
- **LiteLLM** (self-hosted proxy)
- **Fireworks** (`https://api.fireworks.ai/inference/v1`)
- **OpenRouter** (`https://openrouter.ai/api/v1`)

## Authentication

Triple-path auth implemented in `app/api/auth.py`:

```
Request → require_auth (FastAPI Depends)
            │
            ├─ X-API-Key + X-User-Id? → verify key, then DB lookup (user → company_id)
            │                           → CallerContext(gateway, company_id=N)
            │                           Company-scoped (Gateway-forwarded)
            ├─ X-API-Key alone? → verify against X_API_KEY env → CallerContext(api_key)
            │                                                     Full access
            ├─ X-User-Id alone? → 401 (bare X-User-Id rejected, prevents spoofing)
            │
            ├─ Authorization: Bearer? → decode JWT (HMAC-SHA256, same secret as Gateway)
            │                           → DB lookup (same as above)
            │                           → CallerContext(jwt, company_id=N)
            │                           Company-scoped
            └─ Nothing → 401 Unauthorized
```

**Security**: `X-User-Id` is ONLY accepted alongside a valid `X-API-Key`. Since KB Review is standalone (not behind the Gateway), a bare `X-User-Id` could be forged by anyone. Requiring the API key proves the caller is a trusted service.

**Company-based agent isolation**: JWT and Gateway callers can only access agents where `agents.company_id == user.company_id`. Super admins (role has `super_admin_access` permission in `roles_permissions` JOIN `permissions`) bypass this restriction. API key callers have full access.

**Read-only DB models**: `Users`, `Permissions`, `RolesPermissions` are mapped as read-only SQLAlchemy models for auth lookups (user → company, role → super_admin check).

**List endpoint auto-filtering**: When a JWT/Gateway caller lists reviews without specifying `agent_id`, results are automatically filtered to their company's agents via a subquery on `agents.company_id`.

**Concurrency guard**: `POST /reviews` checks for existing PENDING/RUNNING reviews for the same agent and returns 409 Conflict if one exists.

**Audit trail**: Every review records `created_by_user_id` — populated from the JWT/Gateway user ID, NULL for API key (service-to-service) callers.

**Info leak prevention**: Cross-company access returns 404 (not 403) to avoid revealing whether a resource exists for another company.

**Swagger docs**: Disabled when `APP_ENV=production` (sets `docs_url=None`, `openapi_url=None`).

## Issue Status Management

Issues support four statuses: `OPEN` (default), `RESOLVED`, `DISMISSED`, `ACKNOWLEDGED`.

- **`PATCH /reviews/{id}/issues/{id}`** — updates status, `status_updated_by` (from caller's user_id), `status_updated_at`, and optional `status_note`.
- **Authorization ordering** — company access check runs before active-status check to prevent cross-company users from distinguishing active vs completed reviews via 409 vs 404.
- **Blocked during active review** — returns 409 Conflict if review is PENDING or RUNNING (issues are still being created by the pipeline).
- **Atomic count recalculation** — after each PATCH, `issues_found` and `issues_resolved` on `kb_reviews` are recalculated via raw SQL `UPDATE ... SET col = (SELECT COUNT...)` to avoid stale reads from concurrent PATCHes.
- **`issues_found` semantic** — counts active issues only: OPEN + ACKNOWLEDGED, excluding MINORITY.
- **`issues_resolved`** — counts RESOLVED + DISMISSED, excluding MINORITY.
- **Carry-forward preserves status** — when findings are copied from a previous review (unchanged documents), all status fields are copied too. A resolved issue stays resolved across reviews as long as the underlying documents don't change.
- **No DB CHECK constraint** — status domain enforced by Pydantic validator at the API layer. MySQL CHECK constraint support varies by version.

## Celery Upgrade Path

The analysis pipeline is isolated in `review_runner.py` as `run_review(review_id)`.

To upgrade from `BackgroundTasks` to Celery:
1. Add `celery` and `redis` to dependencies
2. Create `celery_config.py` with broker URL
3. Decorate `run_review` with `@celery_app.task`
4. Replace `background_tasks.add_task(run_review, id)` with `run_review.delay(id)` (async path)
5. Sync mode (`?wait=true`) uses `run_in_executor` directly — this bypasses `BackgroundTasks` and would continue to work. Optionally switch to `run_review.apply(args=[id]).get(timeout=300)` for Celery-managed execution.

The analysis code itself doesn't change.
