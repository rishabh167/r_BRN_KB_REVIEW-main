# BRN_KB_REVIEW — Quick Start

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- MySQL (shared `brn_admin_panel` database)
- Neo4j 5.x (with trained agent data)

## Setup

```bash
cd BRN_KB_REVIEW

# Install dependencies
poetry install

# Copy and configure environment
cp .env.example .env
# Edit .env with your DB credentials, Neo4j connection, LLM API keys, and auth settings (X_API_KEY, JWT_SECRET)
```

## Database Tables

The service requires 5 tables to be created before first use. Run the SQL scripts in `docs/DATABASE_SCHEMA.md`. It also requires SELECT access to 4 existing tables (`agents`, `users`, `roles_permissions`, `permissions`) for auth. See `docs/DATABASE_SCHEMA.md` for grants.

## Run

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8062 --reload
```

## Verify

```bash
# Health check
curl http://localhost:8062/review-api/health

# Swagger docs (disabled when APP_ENV=production)
open http://localhost:8062/review-api/docs
```

## Start a Review

Auto mode (recommended) — 3x Gemini no-reasoning with Haiku fallback:

```bash
curl -X POST http://localhost:8062/review-api/reviews \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"agent_id": 42}'
```

## Check Status

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8062/review-api/reviews/1
```

## View Issues

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8062/review-api/reviews/1/issues

# Filter by status
curl -H "X-API-Key: your-api-key" "http://localhost:8062/review-api/reviews/1/issues?status=OPEN"
```

## Resolve / Dismiss Issues

```bash
curl -X PATCH http://localhost:8062/review-api/reviews/1/issues/5 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"status": "RESOLVED", "note": "Fixed in latest training"}'
```

Statuses: `OPEN` (default), `RESOLVED` (KB was fixed), `DISMISSED` (false positive), `ACKNOWLEDGED` (aware, not yet fixed).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` on Neo4j | Check `NEO4J_URI` in `.env`, ensure Neo4j is running |
| `Access denied` on MySQL | Check `DB_USER`/`DB_PASSWORD` in `.env` |
| `Agent has no tenant_id` | The agent hasn't been trained yet — train it first |
| LLM judge returns empty | Check API key and model name for the provider |
| Review stuck at RUNNING | Check logs for LLM timeout — increase timeout or reduce `max_candidate_pairs`. Restarting the service will auto-recover it as PARTIAL/FAILED. |
