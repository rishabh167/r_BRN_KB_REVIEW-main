# KB Review Service — Costing Reference

This document covers LLM costs for the KB Review pipeline: how tokens are tracked, per-model pricing, actual usage from production runs, cost projections, and reduction strategies.

---

## 1. How Tokens Are Tracked

The service records token usage at two levels:

### Per-finding (`kb_review_judge_results` table)

| Column | Description |
|--------|-------------|
| `input_tokens` | Tokens consumed by the judge's input for this finding |
| `output_tokens` | Tokens produced by the judge's output for this finding |

Token counts are split evenly across pairs in a batch: `total_tokens // max(num_pairs, 1)`.

### Per-judge (`kb_review_judge_stats` table)

| Column | Description |
|--------|-------------|
| `total_input_tokens` | Aggregated input tokens across all batches for this judge |
| `total_output_tokens` | Aggregated output tokens across all batches for this judge |
| `total_llm_calls` | Total API calls made by this judge |
| `total_findings` | Number of findings this judge detected |
| `duration_ms` | Wall-clock execution time for this judge |

There is **no cost calculation or billing system** in the service. Token data is stored for observability; cost must be derived externally using the pricing rates below.

---

## 2. Pipeline Cost Drivers

| Parameter | Default | Effect on Cost |
|-----------|---------|----------------|
| **Batch size** | 5 pairs per LLM call | Fewer calls = lower per-call overhead, but same total tokens |
| **max_candidate_pairs** | 50 (default) | Hard cap on pairs sent to LLM; directly controls max cost |
| **similarity_threshold** | 0.85 | Lower threshold = more pairs = higher cost |
| **Number of judges** | 1-3 | Each judge runs the full pipeline independently; cost scales linearly |
| **max_tokens (output)** | 4,000 (standard) / 16,000 (reasoning) | LLM output limit per call; reasoning models include thinking tokens in this budget |
| **reasoning_effort** | None | `low`/`medium`/`high` for thinking models; `medium` recommended for Gemini |
| **Analysis types** | CONTRADICTION, ENTITY_INCONSISTENCY | SEMANTIC_DUPLICATION and AMBIGUITY are opt-in; AMBIGUITY massively increases calls (1 per chunk) |
| **Temperature** | 0.1 (standard) / 1.0 (Gemini) | Gemini requires 1.0; lowering causes unexpected behaviour |

### Per-call token profile

| Component | Estimated Tokens |
|-----------|-----------------|
| System prompt | ~550 |
| Per-pair input (2 chunks x 2,000 chars + entities) | ~800-1,500 |
| 5-pair batch total input | ~5,000-8,000 |
| Output per batch (varies by finding density) | ~200-1,500 |

---

## 3. Model Pricing

Pricing per 1 million tokens (as of February 2026).

### Anthropic Claude

| Model | Input $/M | Output $/M | Notes |
|-------|-----------|------------|-------|
| Claude Sonnet 4.6 | $3.00 | $15.00 | Latest Sonnet; same price as 4.5. Best quality for our use case |
| Claude Sonnet 4.5 | $3.00 | $15.00 | Previous Sonnet; used in our production runs |
| Claude Haiku 4.5 | $1.00 | $5.00 | Fast, good quality, 1/3 the cost of Sonnet |

Batch API gives 50% off; prompt caching gives up to 90% off cached input tokens.

### OpenAI GPT

| Model | Input $/M | Output $/M | Notes |
|-------|-----------|------------|-------|
| GPT-5.2 | $1.75 | $14.00 | Latest flagship; comparable to Sonnet on output cost |
| GPT-4.1 | $2.00 | $8.00 | Strong mid-tier; cheaper output than Sonnet |
| GPT-4.1 mini | $0.40 | $1.60 | Good budget option with solid quality |
| GPT-4.1 nano | $0.10 | $0.40 | Ultra-cheap; good for screening passes |

GPT-4o and GPT-4o-mini are legacy models — replaced by the 4.1 and 5.x families.

### Google Gemini

| Model | Input $/M | Output $/M | Notes |
|-------|-----------|------------|-------|
| Gemini 3.1 Pro | $2.00 | $12.00 | Latest Pro; long-context doubles to $4/$18 over 200K tokens |
| Gemini 3 Flash | $0.50 | $3.00 | Great quality-to-cost ratio |
| Gemini 2.5 Flash | $0.30 | $2.50 | Previous Flash; still available |
| Gemini 2.5 Flash Lite | $0.10 | $0.40 | Ultra-budget |

Batch API gives 50% off; context caching reduces cached input to 10% of base price.

### Chinese / Open-source Models

| Model | Provider | Input $/M | Output $/M | Notes |
|-------|----------|-----------|------------|-------|
| GLM-5 | Z-AI / Fireworks / DeepInfra | ~$0.80-1.00 | ~$2.56-3.20 | Strong open-source; trained on Huawei chips |
| Kimi K2.5 | Moonshot / OpenRouter / Fireworks | ~$0.45-0.60 | ~$2.20-3.00 | Very competitive pricing; open-source |
| Qwen 3.5-Plus | Alibaba / OpenRouter | ~$0.11 | ~$3.60 | Extremely cheap input; pricier output |
| Llama 4 Maverick | Fireworks / OpenRouter | $0.27 | $0.85 | Best open-source value; 1M context |
| Llama 4 Scout | Fireworks | $0.18 | $0.59 | Lighter Llama 4 variant |

Third-party providers (DeepInfra, Fireworks, Chutes) often offer lower rates than official APIs.

### Legacy Models (used in early production runs, now superseded)

| Model | Replaced By |
|-------|------------|
| Claude 3.5 Sonnet | Claude Sonnet 4.5 / 4.6 (same pricing) |
| GPT-4o / GPT-4o-mini | GPT-4.1 / GPT-5.2 |
| Gemini Flash 1.5/2.0 | Gemini 3 Flash |
| Llama 3.1 70B | Llama 4 Maverick / Scout |
| GLM-4 | GLM-5 |

---

## 4. Actual Production Data

### Agent 117 — PDF-based KB (8 docs, 209 chunks, 93 pages)

**Review #16** — Single-judge Sonnet, with ambiguity scanning:

| Metric | Value |
|--------|-------|
| LLM calls | 26 (5 pair-analysis + 21 ambiguity per-chunk) |
| Input tokens | 35,499 |
| Output tokens | 20,864 |
| Findings | 82 |
| Duration | ~6.5 min |
| **Estimated cost (Sonnet)** | **~$0.42** |

**Review #28** — Multi-judge, pairs-only (no ambiguity), 25 pairs:

| Judge | Model | LLM Calls | Input Tokens | Output Tokens | Duration | Est. Cost |
|-------|-------|-----------|-------------|--------------|----------|-----------|
| 0 | Sonnet | 5 | 21,539 | 4,575 | ~1.5 min | ~$0.13 |
| 1 | GLM-5 | 5 | 17,994 | 9,357 | ~1.6 min | ~$0.005 |
| **Total** | | **10** | **39,533** | **13,932** | **~3 min** | **~$0.14** |

### Agent 106 — Website KB (40 docs, 44 chunks, 39 pages)

**Review #40** — Single-judge Sonnet, 50 pairs:

| Metric | Value |
|--------|-------|
| LLM calls | 10 |
| Input tokens | 56,084 |
| Output tokens | 18,248 |
| Findings | 51 raw -> 18 issues |
| Duration | ~5.5 min |
| **Estimated cost (Sonnet)** | **~$0.44** |

Per-call averages: ~5,608 input tokens, ~1,825 output tokens.

### Multi-Model Comparison — Reviews #71-79 (25 pairs each, 5 LLM calls per judge)

These reviews tested multiple judge configurations across three different KBs (banking, messaging, retail).

#### Gemini 3 Flash Preview — Reasoning Level Comparison (Agent 117)

Tested three reasoning configurations with 16k token ceiling:

| Config | Review | Findings | Output Tokens | Duration | Est. Cost | Notes |
|--------|--------|----------|---------------|----------|-----------|-------|
| **High reasoning** | R74 J0 | 6 | ~15,400/call avg | ~100s/call | ~$0.18 | 3/5 calls truncated at 16k |
| **Medium reasoning** | R74 J1 | 11 | ~8,800/call avg | ~55s/call | ~$0.12 | 0 truncated; sweet spot |
| **No reasoning** | R74 J2 | 15 | ~920/call avg | ~7s/call | ~$0.02 | Fastest; was noisy at temp=0.1 but strong after prompt optimization (see below) |

**Key insight:** Medium reasoning self-regulates at ~8-10k output tokens even when given a 16k ceiling. High reasoning fills whatever budget you give it. However, after prompt optimization and temp=1.0 fix, no-reasoning performs comparably to medium reasoning (see R113/R114 analysis below).

#### Token Ceiling Experiment — Gemini Medium Reasoning (Agent 117, Review #77)

| Ceiling | Truncated Calls | Findings | Usable? |
|---------|----------------|----------|---------|
| **4k** | 5/5 | 0 | No — thinking consumes entire budget |
| **8k** | 4/5 | 4 (salvaged) | Barely — 1 finding salvaged per call |
| **16k** | 0/5 | 14 | Yes — model uses ~8k naturally |
| **No cap** | 0/5 | 8-11 | Yes but high reasoning uses 63k tokens/call |

**Result:** 16k is the optimal default. Below 8k, thinking tokens starve the JSON output. Above 16k, high reasoning expands wastefully.

#### Per-Model Cost Comparison — Actual Data (25 pairs, 1 judge)

Averaged across all available runs per model:

| Model | Runs | Avg Findings | Avg Input Tok | Avg Output Tok | Avg Cost/Review | Avg Time | Cost/Finding |
|-------|------|-------------|---------------|----------------|-----------------|----------|-------------|
| **Sonnet 4.6** | 4 | 10.2 | 25,283 | 5,956 | **$0.165** | 115s | $0.016 |
| **Gemini Med@16k** | 4 | 11.5 | 24,664 | 40,538 | **$0.028** | 244s | $0.002 |
| **Haiku 4.5** | 3 | 19.3 | 25,325 | 9,342 | **$0.058** | 85s | $0.003 |

Notes:
- Gemini Med output tokens are high (~40k) because thinking tokens are included; actual JSON content is ~2-4k.
- Haiku finds the most issues but includes more noise (duplicate/low-confidence findings).
- Sonnet has the best description quality but fewest findings.

#### Multi-Agent Results (25 pairs, 3-judge panel: Sonnet + Gemini Med + Haiku)

| Agent | KB Type | Docs | Chunks | LLM Issues | UNANIMOUS | MAJORITY | MINORITY | Total Cost |
|-------|---------|------|--------|------------|-----------|----------|----------|------------|
| 117 | Banking PDFs | 8 | 208 | 19 | 4 | 4 | 11 | ~$0.27 |
| 118 | Messaging/CPaaS | 8 | 91 | 21 | 1 | 4 | 16 | ~$0.27 |
| 119 | Retail/E-commerce | 8 | 88 | 33 | 4 | 6 | 23 | ~$0.32 |

#### Prompt & Temperature Optimization — Before/After (Agent 117, Gemini Med@16k)

After optimizing prompts (few-shot examples, concise instructions, cite-specific-values nudge) and fixing temperature from 0.1 to 1.0 (Google's recommendation for Gemini 3):

| Metric | R77 (temp=0.1, old prompts) | R92 (temp=1.0, optimized) |
|--------|---------------------------|--------------------------|
| Findings | 15 | 7 |
| Confidence range | 0.48 – 1.00 | 0.90 – 1.00 |
| Findings below 0.6 confidence | 7 (almost half) | 0 |
| Speculative/weak findings | ~7 | 0 |
| Duplicate detections | CD/FD flagged 3x | Bot capabilities 2x |

**Key finding:** temp=0.1 was causing Gemini to stretch for low-confidence findings. temp=1.0 + tighter prompts produces higher precision with zero speculative noise. Every R92 finding cites specific conflicting values and explains chatbot impact.

#### No-Reasoning Discovery — Reviews R92 vs R113 (Agent 117)

After prompt optimization, we retested Gemini 3 Flash without reasoning and found it now performs comparably to medium reasoning:

| Metric | R92 (medium reasoning) | R113 (no reasoning) |
|--------|----------------------|---------------------|
| LLM findings | 7 | 7 |
| Input tokens | 23,644 | 23,614 |
| Output tokens | 35,274 | 1,858 |
| Duration | 225s (3.7 min) | 17s |
| Est. cost | ~$0.028 | ~$0.003 |

Same finding count, nearly identical quality. No-reasoning is **19x fewer output tokens, 13x faster, ~10x cheaper.** The prompt engineering did what reasoning was compensating for — once we told the model clearly what we wanted, it doesn't need to think that hard.

R113 even found 2 issues that R92 missed (Premium Savings APY conflict, Platinum savings bonus conflict) while R92 found 2 that R113 missed. Net detection is equivalent.

#### 3x No-Reasoning Strategy — Review R114 (Agent 117)

Running the same no-reasoning model 3 times as separate judges on the same pairs, with consensus aggregation:

| Metric | 1x no-reasoning (R113) | 3x no-reasoning (R114) | 1x medium reasoning (R92) |
|--------|----------------------|----------------------|--------------------------|
| Unique findings | 7 | 10 | 7 |
| UNANIMOUS (3/3) | n/a | 6 | n/a |
| MAJORITY (2/3) | n/a | 2 | n/a |
| MINORITY (1/3) | n/a | 2 | n/a |
| Total output tokens | 1,858 | 6,557 | 35,274 |
| Duration | 17s | 57s | 225s |
| Est. cost | ~$0.003 | ~$0.009 | ~$0.028 |

**Result:** 3x no-reasoning finds **more unique issues** than 1x medium reasoning, with built-in consensus filtering, at **1/3 the cost** and **1/4 the time**. The 2 MINORITY findings were speculative (reverse-engineered math, not stated contradictions) — consensus correctly self-filtered the noise.

This is the recommended production configuration going forward.

#### Gemini 3 Flash — Configuration Notes

- **Temperature:** Must be 1.0 for Gemini models. Lowering it causes looping and degraded output (per Google's documentation).
- **Reasoning effort:** Not needed for production with optimized prompts. Available as opt-in; `medium` with 16k ceiling if used.
- **JSON mode:** `response_format: {"type": "json_object"}` enforced at the API level to reduce truncation.
- **Truncation recovery:** `_salvage_truncated_json()` extracts complete findings from cut-off responses. Retry logic re-invokes the LLM once if 0 findings are salvaged from a clearly truncated response.
- **3x judge strategy:** Run 3 identical no-reasoning judges; filter to UNANIMOUS + MAJORITY for high-confidence output. Cost: ~$0.009/review (25 pairs).

---

## 5. Pair Count by Threshold

Pair count is the primary cost driver. These are actual counts from production:

### Agent 117 (209 chunks)

| Threshold | Pairs | LLM Calls (@ batch 5) |
|-----------|-------|-----------------------|
| 0.90 | 270 | 54 |
| **0.85** | **356** | **72** |
| 0.80 | 1,139 | 228 |
| 0.75 | 2,786 | 558 |

### Agent 106 (44 chunks, pre-URL-dedup)

| Threshold | Pairs | LLM Calls (@ batch 5) |
|-----------|-------|-----------------------|
| **0.85** | **571** | **115** |

Agent 106 has far more pairs per chunk (13.0 vs 1.7) because the website KB is highly interconnected — many pages share entities, producing entity-overlap and relationship-conflict pairs beyond just embedding similarity.

---

## 6. Full-KB Cost Projections

### Agent 117 — All 356 pairs (@ 0.85 threshold)

72 LLM calls per judge. Extrapolated from Review #28 actual data (14.4x scaling):

| Config | Input Tokens | Output Tokens | Estimated Cost |
|--------|-------------|--------------|----------------|
| **Gemini 3 Flash x3 (no reasoning)** | **~1,065K** | **~29K** | **~$0.18** |
| Gemini 3 Flash (med reasoning) | ~355K | ~580K | ~$0.40 |
| Sonnet 4.5 only | ~310K | ~66K | ~$1.90 |
| Haiku 4.5 only | ~310K | ~66K | ~$0.64 |
| GPT-4.1 mini only | ~310K | ~66K | ~$0.23 |
| Llama 4 Maverick only | ~310K | ~66K | ~$0.14 |

Note: 3x Gemini runs 3 identical judges for consensus. Input triples but output stays tiny (~400 tok/call). Medium reasoning output is high (~8k/call) because thinking tokens are included.

### Agent 106 — All 571 pairs (@ 0.85, before URL dedup)

115 LLM calls. Extrapolated from Review #40 (11.5x scaling):

| | Tokens | Cost (Sonnet) |
|---|--------|---------------|
| Input | ~645K | $1.93 |
| Output | ~210K | $3.15 |
| **Total** | | **~$5.08** |

Agent 106 is more expensive per call because Sonnet produces longer output (many findings per batch in this KB).

---

## 7. Cost Estimation Rules of Thumb

### Per-pair cost

| Model | Approx. Cost per Pair | Notes |
|-------|----------------------|-------|
| Sonnet 4.5/4.6 | ~$0.007 | Measured from R71/77/78/79 |
| Haiku 4.5 | ~$0.002 | Measured from R71/78/79 |
| Gemini 3 Flash (medium reasoning) | ~$0.001 | Measured from R74/77/78/79 |
| **Gemini 3 Flash (no reasoning) x3** | **~$0.0004/pair** | **Recommended for production** — $0.009/review for 25 pairs; consensus filters noise (R113/R114) |
| Gemini 3 Flash (no reasoning) x1 | ~$0.0001 | Single run; no consensus; cheapest possible |
| GPT-5.2 | ~$0.009 | Estimated |
| GPT-4.1 mini | ~$0.001 | Estimated |
| Llama 4 Maverick | ~$0.0008 | Estimated |
| GLM-5 | ~$0.003 | Measured from R28 |

### From pages to cost

A rough estimator: **~4 pairs per page of KB content** at threshold 0.85 (but this varies hugely with entity density).

| KB Size | Est. Pairs | Gemini 3x (no reasoning) | Gemini Med 1x | Haiku 4.5 | Sonnet 4.6 |
|---------|-----------|--------------------------|---------------|-----------|------------|
| 50 pages | ~200 | **~$0.07** | ~$0.22 | ~$0.35 | ~$1.32 |
| 100 pages | ~400 | **~$0.14** | ~$0.44 | ~$0.70 | ~$2.64 |
| 500 pages | ~2,000 | **~$0.72** | ~$2.20 | ~$3.50 | ~$13.20 |

**Caveat:** The "4 pairs per page" heuristic does NOT hold for highly interconnected KBs like website scrapes. Agent 106 had **14.6 pairs per page** due to heavy entity overlap. A better predictor is `pairs = f(chunks^2 x entity_density)` rather than a linear function of pages.

---

## 8. Cross-Agent Comparison

| Metric | Agent 117 (PDFs) | Agent 106 (Website) |
|--------|-----------------|-------------------|
| Pages | 93 | 39 |
| Chunks | 208 | 44 |
| Total pairs (@ 0.85) | 356 | 571 |
| Pairs per page | 3.8 | 14.6 |
| Pairs per chunk | 1.7 | 13.0 |
| Chunks per page | 2.2 | 1.1 |
| Est. full cost (Sonnet, 1 judge) | ~$1.90 | ~$5.08 |

Website KBs are disproportionately expensive because many pages share entities (company name, product names, navigation text), creating a quadratic explosion of entity-overlap pairs.

---

## 9. Cost Reduction: URL Canonicalization

Implemented to address the URL-variant problem in website KBs (see `pre_filter.py`).

**Problem:** The scraper ingests both `broadnet.ai/solutions/healthcare` and `www.broadnet.ai/solutions/healthcare` as separate documents. These identical pages generate noise pairs between them and cause the same real issue to appear twice.

**Expected impact on Agent 106:**

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| Chunks sent to pairing | 44 | ~34 (10 URL dupes removed) |
| Candidate pairs (@ 0.85) | 571 | ~300-350 |
| LLM calls (1 judge) | 115 | ~60-70 |
| Cost (Sonnet) | ~$5.08 | ~$2.50-3.00 |
| Duplicate issues | 5 | 0 |
| New structural issues | 0 | ~3-5 URL_DUPLICATION |

Roughly a **40-50% cost reduction** for website KBs. PDF-only KBs are unaffected (filenames don't canonicalize).

---

## 10. Ambiguity Scanning Cost Impact

Ambiguity scanning (disabled by default) runs **one LLM call per chunk**, not per pair. This makes it dramatically more expensive for large KBs:

| Config | Agent 117 (209 chunks) | LLM Calls |
|--------|----------------------|-----------|
| Pairs only (25 pairs) | 5 calls | 5 |
| Pairs + ambiguity | 5 + 21 calls | 26 |

Review #16 (with ambiguity) used **5.2x more LLM calls** than Review #28 (without) for the same KB. The ambiguity scan samples up to 3 chunks per document and caps at 50 chunks total, but still adds significant cost.

---

## 11. Cost Optimization Strategies

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| **3x Gemini no-reasoning (recommended)** | ~95% vs Sonnet, ~68% vs Gemini Med | Best value: consensus filtering, more findings, 57s runtime |
| **Change detection / carry-forward** (default on) | 0-100% of LLM calls skipped on repeat reviews | First review is full cost; subsequent reviews only re-analyze changed document pairs. If nothing changed, cost is $0. |
| **Raise similarity_threshold** (0.85 -> 0.90) | ~25% fewer pairs | May miss loosely related contradictions |
| **Lower max_candidate_pairs** (default is now 50; tested at 200) | Up to 50% | Hard cap; highest-priority pairs still analyzed |
| **URL canonicalization** (website KBs) | ~40-50% for website KBs | None — pure noise reduction |
| **Disable ambiguity scanning** | ~3-5x fewer calls | Misses vague/ambiguous language issues |
| **Increase batch size** (5 -> 10) | Fewer calls, same tokens | Longer prompts may reduce accuracy |

---

## 12. Quick Cost Calculator

```
pairs = min(candidate_pairs_at_threshold, max_candidate_pairs)
calls_per_judge = ceil(pairs / 5)
total_calls = calls_per_judge x num_judges

# Standard models (no reasoning):
input_tokens  ~ calls_per_judge x 5,000
output_tokens ~ calls_per_judge x 1,200

# Reasoning models (e.g., Gemini 3 Flash with medium reasoning):
input_tokens  ~ calls_per_judge x 5,000
output_tokens ~ calls_per_judge x 8,000   (includes ~6k thinking + ~2k content)

cost = (input_tokens x input_price_per_M / 1_000_000)
     + (output_tokens x output_price_per_M / 1_000_000)
     (sum across all judges)
```

**Note:** Gemini Flash and Haiku examples below use estimated rates via LiteLLM proxy (~$0.15/$0.60/M for Gemini Flash, ~$0.80/$4.00/M for Haiku). Actual proxy rates may differ from list prices ($0.50/$3.00 for Gemini Flash, $1.00/$5.00 for Haiku) shown in section 3. Production costs in section 4 reflect actual billed amounts through the proxy.

Example: 200 pairs, 3x Gemini 3 Flash (no reasoning) — **recommended production config**:
```
calls = 40 x 3 judges = 120
input  = 120 x 5,000 = 600,000 tokens  -> 600K x $0.15/M = $0.09
output = 120 x 400   =  48,000 tokens  ->  48K x $0.60/M = $0.03
total = ~$0.12  (best value: consensus filtering, more findings, ~60s)
```

Same 200 pairs, 1x Gemini 3 Flash (medium reasoning):
```
calls = 40
input  = 40 x 5,000 = 200,000 tokens  -> 200K x $0.15/M = $0.03
output = 40 x 8,000 = 320,000 tokens  -> 320K x $0.60/M = $0.19
total = ~$0.22  (1.8x more expensive, slower, no consensus)
```

Same 200 pairs with Sonnet 4.6:
```
calls = 40
input  = 40 x 5,000 = 200,000 tokens  -> 200K x $3/M  = $0.60
output = 40 x 1,200 =  48,000 tokens  ->  48K x $15/M = $0.72
total = ~$1.32  (11x more expensive than 3x Gemini)
```

Same 200 pairs with Haiku 4.5:
```
calls = 40
input  = 200K x $0.80/M = $0.16
output =  48K x $4.00/M = $0.19
total = ~$0.35
```


