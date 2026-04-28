# BRN_KB_REVIEW — API Reference

Base URL: `http://localhost:8062/review-api`

## Authentication

All endpoints (except health check) require one of three auth headers:

| Header | Use Case | Access Level |
|--------|----------|--------------|
| `X-API-Key` + `X-User-Id` | Gateway-forwarded (Gateway validated JWT, forwards user ID) | Company-scoped — own company's agents only |
| `X-API-Key` alone | Service-to-service calls (Training Service, etc.) | Full access — all agents, all reviews |
| `Authorization: Bearer <jwt>` | Direct access (no Gateway) | Company-scoped — own company's agents only |

**Security**: `X-User-Id` is ONLY accepted alongside a valid `X-API-Key`. A bare `X-User-Id` returns 401.

**Error responses**:
- `401 Unauthorized` — no credentials, expired JWT, inactive user
- `403 Forbidden` — invalid API key
- `404 Not Found` — resource not found, or caller accessing another company's agent/review (returns 404 instead of 403 to avoid leaking resource existence)
- `409 Conflict` — agent already has a PENDING or RUNNING review, or attempting to update issue status while review is active
- `400 Bad Request` — malformed `X-User-Id` (not a valid integer), agent has no tenant_id, invalid provider, or empty judges list
- `422 Unprocessable Entity` — request body validation failed (e.g., `similarity_threshold > 1.0`, `max_candidate_pairs < 1` or `> 500`, invalid `analysis_types`)
- `500 Internal Server Error` — server auth misconfiguration (missing `X_API_KEY` or `JWT_SECRET` env var)

Super admins (with `super_admin_access` permission) bypass company restrictions.

---

## Health Check

```
GET /review-api/health
```

No authentication required.

```json
{"status": "healthy", "service": "Broadnet KB Review Service"}
```

---

## Start a Review

```
POST /review-api/reviews
```

```bash
# Auto mode (recommended)
curl -X POST http://localhost:8062/review-api/reviews \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"agent_id": 42}'

# Sync mode — wait for results
curl -X POST "http://localhost:8062/review-api/reviews?wait=true" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"agent_id": 42}'
```

Returns `409 Conflict` if the agent already has a PENDING or RUNNING review.

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `wait` | bool | `false` | Block until review completes (max 5 min). Returns `200` with full results on success, `202` on timeout. |
| `include_minority` | bool | `false` | Include MINORITY consensus findings in sync response (only applies when `wait=true`). |
| `carryforward` | bool | `true` | Carry forward findings for unchanged document pairs. Set to `false` to force full LLM analysis. |

### Request Body

**Auto mode — zero config (recommended):**

```json
{
  "agent_id": 42
}
```

Judges omitted → auto mode: 3x Gemini Flash no-reasoning with Haiku fallback if Gemini is unavailable.

**Custom mode — explicit judges:**

```json
{
  "agent_id": 42,
  "judges": [
    {"provider": "litellm", "model": "gemini/gemini-3-flash-preview"},
    {"provider": "litellm", "model": "gemini/gemini-3-flash-preview"},
    {"provider": "litellm", "model": "gemini/gemini-3-flash-preview"}
  ],
  "similarity_threshold": 0.85,
  "max_candidate_pairs": 50
}
```

**Custom with rate limiting:**

```json
{
  "agent_id": 42,
  "judges": [
    {"provider": "litellm", "model": "gemini/gemini-3-flash-preview", "rate_limit_rpm": 15},
    {"provider": "litellm", "model": "gemini/gemini-3-flash-preview", "rate_limit_rpm": 15},
    {"provider": "litellm", "model": "gemini/gemini-3-flash-preview", "rate_limit_rpm": 15}
  ]
}
```

All 3 judges share one rate limiter (same provider + model) capped at 15 RPM total.

**Custom — mixed models with reasoning:**

```json
{
  "agent_id": 42,
  "judges": [
    {
      "provider": "litellm",
      "model": "anthropic/claude-sonnet-4-6"
    },
    {
      "provider": "litellm",
      "model": "gemini/gemini-3-flash-preview",
      "reasoning_effort": "medium"
    },
    {
      "provider": "litellm",
      "model": "anthropic/claude-haiku-4-5"
    }
  ],
  "analysis_types": ["CONTRADICTION", "ENTITY_INCONSISTENCY", "SEMANTIC_DUPLICATION"],
  "similarity_threshold": 0.85,
  "max_candidate_pairs": 50
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | int | required | Target agent ID |
| `judges` | array | `null` | Judge configs. Omit for auto mode (3x Gemini + Haiku fallback). |
| `analysis_types` | array | `CONTRADICTION`, `ENTITY_INCONSISTENCY` | Which LLM checks to run (optional: `SEMANTIC_DUPLICATION`, `AMBIGUITY`) |
| `similarity_threshold` | float | 0.85 | Cosine similarity cutoff for pre-filtering |
| `max_candidate_pairs` | int | 50 | Max pairs sent to LLM judges |

### Judge Config

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string | yes | `litellm`, `fireworks`, or `openrouter` |
| `model` | string | yes | Model name including provider prefix (e.g., `gemini/gemini-3-flash-preview`, `anthropic/claude-haiku-4-5`) |
| `api_base` | string | no | Override base URL (falls back to env default) |
| `api_key` | string | no | Override API key (falls back to env default) |
| `temperature` | float | no | LLM temperature (default: 0.1; Gemini models default to 1.0) |
| `max_tokens` | int | no | Max output tokens (default: 4000; 16000 when reasoning_effort is set) |
| `reasoning_effort` | string | no | `low`, `medium`, or `high` — enables thinking mode for supported models |
| `rate_limit_rpm` | int | no | Rate limit in requests/min for this model. Judges sharing the same `(provider, model)` share one limiter. Omit for no rate limiting. |

### Response: `202 Accepted` (async mode)

```json
{
  "id": 1,
  "agent_id": 42,
  "status": "PENDING",
  "progress": 0,
  "total_documents": 0,
  "total_chunks": 0,
  "issues_found": 0,
  "issues_resolved": 0,
  "url_duplicates_removed": 0,
  "issues_by_type": {},
  "issues_by_severity": {},
  "judge_stats": [],
  "previous_review_id": null,
  "pairs_reused": 0,
  "pairs_analyzed": 0,
  "docs_changed": 0,
  "docs_unchanged": 0,
  "created_by_user_id": 123,
  "error_message": null,
  "started_at": null,
  "completed_at": null,
  "created_at": "2025-01-15T10:30:00"
}
```

### Response: `200 OK` (sync mode, `?wait=true`)

```json
{
  "review": {
    "id": 1,
    "agent_id": 42,
    "status": "COMPLETED",
    "progress": 100,
    "total_documents": 8,
    "total_chunks": 142,
    "issues_found": 5,
    "issues_resolved": 0,
    "url_duplicates_removed": 10,
    "previous_review_id": 1,
    "pairs_reused": 8,
    "pairs_analyzed": 2,
    "docs_changed": 1,
    "docs_unchanged": 7,
    "issues_by_type": {"CONTRADICTION": 2, "MISSING_EMBEDDINGS": 3},
    "issues_by_severity": {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 2},
    "judge_stats": [
      {
        "judge_index": 0,
        "judge_provider": "litellm",
        "judge_model": "gemini/gemini-3-flash-preview",
        "total_input_tokens": 45000,
        "total_output_tokens": 8500,
        "total_llm_calls": 12,
        "total_findings": 5,
        "duration_ms": 34200,
        "started_at": "2025-01-15T10:30:02",
        "completed_at": "2025-01-15T10:30:36"
      }
    ],
    "created_by_user_id": 123,
    "error_message": null,
    "started_at": "2025-01-15T10:30:01",
    "completed_at": "2025-01-15T10:32:45",
    "created_at": "2025-01-15T10:30:00"
  },
  "issues": [
    {
      "id": 1,
      "review_id": 1,
      "issue_type": "CONTRADICTION",
      "severity": "CRITICAL",
      "confidence": 0.95,
      "title": "Conflicting enterprise plan pricing",
      "description": "...",
      "consensus": "UNANIMOUS",
      "judges_flagged": 3,
      "judges_total": 3,
      "carried_forward": false,
      "original_review_id": null,
      "status": "OPEN",
      "status_updated_by": null,
      "status_updated_at": null,
      "status_note": null
    }
  ]
}
```

MINORITY findings are excluded from the sync response by default. Add `?include_minority=true` to include them.

If the review **times out** (>5 min), the endpoint returns `202 Accepted` with the current review summary (may show `RUNNING` status). The review continues in the background — poll `GET /reviews/{id}` for status.

If the review **fails or crashes** within the timeout, the endpoint still returns `200` with `review.status` set to `"FAILED"` (no findings) or `"PARTIAL"` (some findings persisted before the crash). Check `review.status` in the response to distinguish success from failure. `review.error_message` contains the failure reason.

---

## Get Review Status

```
GET /review-api/reviews/{review_id}
```

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8062/review-api/reviews/1
```

Company-scoped callers (JWT/Gateway) can only access reviews for their own company's agents. Returns `404` if the review doesn't exist or belongs to another company.

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `include_minority` | bool | `false` | Include MINORITY consensus findings in `issues_found`, `issues_by_type`, and `issues_by_severity` counts. |

```json
{
  "id": 1,
  "agent_id": 42,
  "status": "COMPLETED",
  "progress": 100,
  "total_documents": 8,
  "total_chunks": 142,
  "issues_found": 7,
  "issues_resolved": 0,
  "url_duplicates_removed": 10,
  "previous_review_id": 1,
  "pairs_reused": 8,
  "pairs_analyzed": 2,
  "docs_changed": 1,
  "docs_unchanged": 7,
  "issues_by_type": {
    "CONTRADICTION": 2,
    "MISSING_EMBEDDINGS": 3,
    "AMBIGUITY": 2
  },
  "issues_by_severity": {
    "CRITICAL": 1,
    "HIGH": 4,
    "MEDIUM": 2
  },
  "judge_stats": [
    {
      "judge_index": 0,
      "judge_provider": "litellm",
      "judge_model": "gemini/gemini-3-flash-preview",
      "total_input_tokens": 45000,
      "total_output_tokens": 8500,
      "total_llm_calls": 12,
      "total_findings": 5,
      "duration_ms": 34200,
      "started_at": "2025-01-15T10:30:02",
      "completed_at": "2025-01-15T10:30:36"
    }
  ],
  "created_by_user_id": 123,
  "error_message": null,
  "started_at": "2025-01-15T10:30:01",
  "completed_at": "2025-01-15T10:32:45",
  "created_at": "2025-01-15T10:30:00"
}
```

Status values: `PENDING` → `RUNNING` → `COMPLETED` / `FAILED` / `PARTIAL`

`PARTIAL` means the review crashed mid-run but some findings were already persisted and are queryable. `error_message` contains the failure reason for FAILED/PARTIAL reviews.

---

## List Reviews

```
GET /review-api/reviews?agent_id=42
```

```bash
# All reviews for an agent
curl -H "X-API-Key: $API_KEY" "http://localhost:8062/review-api/reviews?agent_id=42"

# All reviews (API key: all agents; JWT/Gateway: auto-filtered to own company)
curl -H "X-API-Key: $API_KEY" http://localhost:8062/review-api/reviews
```

Returns an array of review list items (slim summaries), newest first.

When `agent_id` is omitted by a JWT/Gateway caller, results are automatically filtered to reviews for the caller's company's agents. API key and super admin callers see all reviews.

```json
[
  {
    "id": 1,
    "agent_id": 42,
    "status": "COMPLETED",
    "progress": 100,
    "issues_found": 7,
    "issues_resolved": 0,
    "created_by_user_id": 123,
    "created_at": "2025-01-15T10:30:00"
  }
]
```

Note: `issues_found` counts active issues only (OPEN + ACKNOWLEDGED), excluding MINORITY and RESOLVED/DISMISSED. Use `GET /reviews/{review_id}?include_minority=true` for MINORITY-inclusive counts.

Use `GET /reviews/{review_id}` for the full summary (includes `issues_by_type`, `issues_by_severity`, `judge_stats`).

---

## Get Review Issues

```
GET /review-api/reviews/{review_id}/issues
```

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `issue_type` | string | | Filter: `CONTRADICTION`, `ENTITY_INCONSISTENCY`, `SEMANTIC_DUPLICATION` (opt-in), `AMBIGUITY` (opt-in), `MISSING_EMBEDDINGS`, `EMPTY_CONTENT`, `URL_DUPLICATION` |
| `severity` | string | | Filter: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` |
| `min_confidence` | float | | Minimum confidence threshold (0.0-1.0) |
| `include_minority` | bool | `false` | Include MINORITY consensus findings (excluded by default) |
| `carried_forward` | bool | | Filter by origin: `true` = only carried-forward issues, `false` = only freshly-analyzed issues. Omit for all issues (complete snapshot). |
| `status` | string | | Filter by status: `OPEN`, `RESOLVED`, `DISMISSED`, `ACKNOWLEDGED`. Returns 422 on invalid values. |

### Example

```bash
# High-confidence issues only (MINORITY excluded by default)
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8062/review-api/reviews/1/issues?issue_type=CONTRADICTION&severity=HIGH"

# Include MINORITY findings
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8062/review-api/reviews/1/issues?include_minority=true"

# Only freshly-analyzed issues (exclude carried-forward)
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8062/review-api/reviews/1/issues?carried_forward=false"

# Only carried-forward issues
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8062/review-api/reviews/1/issues?carried_forward=true"

# Only open issues (most common query)
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8062/review-api/reviews/1/issues?status=OPEN"

# Only resolved issues
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8062/review-api/reviews/1/issues?status=RESOLVED"

# Fresh open issues only (exclude carried-forward)
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8062/review-api/reviews/1/issues?status=OPEN&carried_forward=false"
```

### Response

```json
[
  {
    "id": 1,
    "review_id": 1,
    "issue_type": "CONTRADICTION",
    "severity": "CRITICAL",
    "confidence": 0.95,
    "title": "Conflicting enterprise plan pricing",
    "description": "In **Product Pricing 2024.pdf** (page 3), it states the enterprise plan is $99/month. However, in **Company FAQ.docx** (page 12), it says the enterprise plan is $149/month. Recommend updating one of these documents to reflect the current price.",
    "doc_a_name": "Product_Pricing_2024.pdf",
    "doc_a_page": 3,
    "doc_a_excerpt": "The enterprise plan is available at $99/month...",
    "doc_b_name": "Company_FAQ.docx",
    "doc_b_page": 12,
    "doc_b_excerpt": "Our enterprise tier is priced at $149/month...",
    "entities_involved": "[\"enterprise plan\", \"pricing\"]",
    "consensus": "UNANIMOUS",
    "judges_flagged": 3,
    "judges_total": 3,
    "carried_forward": false,
    "original_review_id": null,
    "status": "OPEN",
    "status_updated_by": null,
    "status_updated_at": null,
    "status_note": null,
    "judge_results": [
      {
        "judge_index": 0,
        "judge_provider": "litellm",
        "judge_model": "gemini/gemini-3-flash-preview",
        "detected": true,
        "severity": "CRITICAL",
        "confidence": 0.95,
        "reasoning": "Both documents discuss the enterprise plan pricing but state different amounts...",
        "created_at": "2025-01-15T10:32:45"
      }
    ],
    "created_at": "2025-01-15T10:32:45"
  }
]
```

---

## Update Issue Status

```
PATCH /review-api/reviews/{review_id}/issues/{issue_id}
```

Update the status of a specific issue. Blocked while the review is PENDING or RUNNING (409 Conflict).

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | string | yes | `OPEN`, `RESOLVED`, `DISMISSED`, or `ACKNOWLEDGED` |
| `note` | string | no | Optional note explaining the status change (max 1000 chars) |

### Example

```bash
# Resolve an issue
curl -X PATCH http://localhost:8062/review-api/reviews/1/issues/5 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"status": "RESOLVED", "note": "Fixed in latest training"}'

# Dismiss a false positive
curl -X PATCH http://localhost:8062/review-api/reviews/1/issues/6 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"status": "DISMISSED", "note": "Intentional difference between plans"}'

# Re-open an issue
curl -X PATCH http://localhost:8062/review-api/reviews/1/issues/5 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"status": "OPEN", "note": "Not actually fixed"}'
```

### Response: `200 OK`

Returns the updated issue.

```json
{
  "id": 5,
  "review_id": 1,
  "issue_type": "CONTRADICTION",
  "severity": "CRITICAL",
  "confidence": 0.95,
  "title": "Conflicting enterprise plan pricing",
  "status": "RESOLVED",
  "status_updated_by": 123,
  "status_updated_at": "2025-01-16T14:20:00",
  "status_note": "Fixed in latest training",
  "carried_forward": false,
  "original_review_id": null,
  "created_at": "2025-01-15T10:32:45"
}
```

### Side Effects

When an issue status changes, the review's `issues_found` and `issues_resolved` counts are atomically recalculated:
- `issues_found` = non-MINORITY issues with status OPEN or ACKNOWLEDGED
- `issues_resolved` = non-MINORITY issues with status RESOLVED or DISMISSED

### Status Values

| Status | Meaning |
|--------|---------|
| `OPEN` | Default. Issue needs attention. |
| `RESOLVED` | KB was fixed to address this issue. |
| `DISMISSED` | False positive or won't fix. |
| `ACKNOWLEDGED` | Aware of the issue but not yet fixed. Still counts as active in `issues_found`. |

### Error Responses

- `404` — review or issue not found (or cross-company access)
- `409` — review is PENDING or RUNNING
- `422` — invalid status value
