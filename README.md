# BRN_KB_REVIEW — Knowledge Base Quality Analysis

Automated quality analysis for agent knowledge bases in the Broadnet.ai Chatbot platform.

## What It Does

When clients train their AI agents with multiple documents — PDFs, web pages, FAQs, internal docs — those documents often contain conflicting, outdated, or inconsistent information. This silently degrades the quality of AI-generated answers without anyone knowing.

BRN_KB_REVIEW automatically detects these problems by running LLM-powered analysis across the entire knowledge base, producing a detailed report of every issue found, with document references, excerpts, and confidence scores.

## Business Value

- **Answer quality assurance** — Catches contradictions and inconsistencies that cause the AI to give wrong or confusing answers
- **Client trust** — Proactive quality reports show clients exactly what's in their KB and what needs fixing
- **Onboarding acceleration** — New KB uploads are validated automatically, surfacing problems before they reach end users
- **Ongoing monitoring** — Re-run reviews after document updates to ensure nothing broke
- **Transparency** — Every finding includes source documents, page numbers, and the reasoning behind the detection

## Capabilities

### LLM-Powered Detection
- **Contradictions** — Same topic, different answers across documents (e.g., conflicting pricing, policies, procedures)
- **Entity inconsistencies** — Same entity described differently in different places (e.g., product names, dates, specifications)
- **Semantic duplication** (opt-in) — Near-identical content across documents wasting KB space
- **Ambiguity** (opt-in) — Vague or unclear language that could confuse the AI

### Structural Quality Checks (No LLM Needed)
- **Missing embeddings** — Chunks invisible to search because embedding failed
- **Empty content** — Chunks with no text (failed OCR or extraction)
- **URL duplication** — Same web page ingested under multiple URL variants

### Multi-Judge Consensus
Run 1 to N LLM judges independently and aggregate results via majority voting:
- **Unanimous** — All judges agree (highest confidence)
- **Majority** — Most judges agree (high confidence)
- **Minority** — Few judges flagged (lower confidence, worth reviewing)

This filters out false positives and produces more reliable findings than any single model.

### Smart Pre-Filtering
Not every document pair needs LLM analysis. The service uses embedding similarity, shared entity graphs, and relationship conflicts to identify the pairs most likely to contain issues — keeping LLM costs proportional to actual risk, not KB size.

### Issue Status Management
Track issue lifecycle with four statuses: **Open**, **Resolved** (KB was fixed), **Dismissed** (false positive), **Acknowledged** (aware, not yet fixed). Status changes update aggregate counts in real time. Statuses are preserved automatically when issues are carried forward across reviews.

### Change Detection & Carry-Forward
On repeat reviews, the service tracks per-document content hashes and only re-analyzes pairs where at least one document has changed. Findings from unchanged document pairs are carried forward automatically, so each review is a complete snapshot of all known issues — without paying for redundant LLM calls. If nothing changed, the review completes with zero LLM calls.

### URL Canonicalization
Website-based KBs often ingest the same page under different URLs (http vs https, www vs non-www). The service automatically detects and deduplicates these before analysis, reducing noise and cost.

## Authentication

All endpoints (except health check) require authentication via one of:

- **`X-API-Key`** — service-to-service calls (full access)
- **`X-API-Key` + `X-User-Id`** — Gateway-forwarded (company-scoped access)
- **`Authorization: Bearer <jwt>`** — direct JWT (company-scoped access)

Users can only access agents belonging to their company. Super admins bypass this restriction. See [API Reference](docs/API.md) for details.

## Tech Stack

- **Python / FastAPI** — async REST API on port 8062
- **Neo4j** — reads document chunks and entity graphs (read-only)
- **MySQL** — stores reviews, issues, and judge results
- **LangChain + LiteLLM** — multi-provider LLM support (Gemini, Claude, GPT, open-source models)
- **Docker** — containerized deployment with docker-compose

## Quick Start

```bash
# Copy environment config
cp .env.example .env
# Edit .env with your DB, Neo4j, LLM API, and auth credentials

# Run with Docker
docker compose build && docker compose up -d

# Health check
curl http://localhost:8062/review-api/health

# Start a review
curl -X POST http://localhost:8062/review-api/reviews \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"agent_id": 42}'
```

## Documentation

| Doc | Description |
|-----|-------------|
| [Quick Start](docs/QUICKSTART.md) | Setup and first review |
| [API Reference](docs/API.md) | Endpoints, request/response schemas |
| [Architecture](docs/ARCHITECTURE.md) | Pipeline design, phases, provider strategy |
| [Database Schema](docs/DATABASE_SCHEMA.md) | Table definitions, user setup, rollback |
| [Costing](docs/COSTING.md) | Token usage, model pricing, cost projections |

## Running Tests

```bash
docker compose -f docker-compose.test.yml build && \
docker compose -f docker-compose.test.yml run --rm test-runner
```
