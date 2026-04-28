# TODO

## Priority 1 — Next up

*(empty — pick from Priority 2)*

## Priority 2 — Production readiness

- **Post-training trigger** — Training Service calls `/reviews/auto` after KB training completes. Clients get quality checks automatically without thinking about it.
- **Nightly scheduled reviews** — Celery beat or cron job to review all active agents on a schedule. LiteLLM supports batch API (50% off) but at $0.009/review the savings are negligible — scheduling is the real value. Notes: LiteLLM/Gemini batch API is not yet available.

## Priority 3 — Future

- **Pre-training validation gate** — run review *during* training as a validation step. "We found 3 critical contradictions — do you want to proceed?" Changes the product from reactive to preventive.

## Others
- Make default 1 provider and position the 3 as MAX_QUALITY or MAX mode. Can call direct with custom and specify 1 model now.

## Done

- ~~Partial results for long runs~~ — implemented via `PARTIAL` status; findings persist per-batch as they're discovered
- ~~Gemini optimization~~ — temp=1.0, JSON mode, optimized prompts, retry on truncation
- ~~3x no-reasoning strategy~~ — validated as production config ($0.009/review, consensus filtering)
- ~~SEMANTIC_DUPLICATION moved to opt-in~~ — removed from defaults, same as AMBIGUITY
- ~~Parallel judge execution~~ — judges + batches run concurrently via `asyncio.gather` + `ainvoke`. Two-layer rate limiting: `InMemoryRateLimiter` per (provider, model) + `max_retries=2` for 429s. Probe-then-fan-out pattern, quorum logic, incremental persistence. Optional `rate_limit_rpm` in JudgeConfig.
- ~~Auto review mode~~ — `POST /reviews` with just `agent_id` (judges omitted) runs 3x Gemini no-reasoning with Haiku auto-failover. `?wait=true` for sync mode. MINORITY findings excluded by default on issues endpoint (`?include_minority=true` to see all).
- ~~Track changed documents~~ — Per-document SHA-256 content hashes in `kb_review_doc_hashes`. Subsequent reviews skip LLM calls for unchanged document pairs, carry forward findings with `carried_forward=True` and `original_review_id` chain. `?carryforward=false` forces full analysis. `?carried_forward=true/false` filters issues by origin. First review behaves identically to before.
- ~~API security~~ — Triple-path auth: API key (`X-API-Key`) for S2S, Gateway-forwarded (`X-User-Id`) for admin panel via Gateway, direct JWT (`Authorization: Bearer`) for standalone access. Company-based agent isolation. Super admin bypass via `super_admin_access` permission. `CallerContext` + `require_auth` dependency. All endpoints protected, health check excluded. Concurrency guard (409 if agent already has active review). Audit trail (`created_by_user_id` on reviews). Cross-company access returns 404 (not 403) to prevent info leaks. Swagger docs disabled in production (`APP_ENV=production`).
- ~~Redis token blacklist~~ — JWT path checks Redis for token blacklist (`BL_{token}`) and session invalidation (`TOKEN_VERSION:{userId}`), matching the Gateway's `AuthenticationFilter`. Env-sensitive: production fail-closed (503), dev warn-only. API key and Gateway paths unaffected. Aggressive timeouts (0.5s connect, 0.2s read). Connection pool capped at 20.
- ~~Issue status management~~ — `PATCH /reviews/{id}/issues/{id}` with four statuses: OPEN (default), RESOLVED, DISMISSED, ACKNOWLEDGED. `issues_found` counts active only (OPEN + ACKNOWLEDGED); `issues_resolved` counts RESOLVED + DISMISSED. Atomic SQL count recalculation. Blocked during PENDING/RUNNING (409). Status carried forward on unchanged documents. `?status=` filter on GET issues endpoint. Tests in `test_issue_status.py`.
