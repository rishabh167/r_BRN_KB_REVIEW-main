"""Phase 3 — LLM judge analysis of candidate pairs and individual chunks."""

import asyncio
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from . import prompts

logger = logging.getLogger("kb_review")


def _parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array from the LLM response, handling markdown fences.

    If the response was truncated (e.g. by a token limit), salvages any
    complete JSON objects that appear before the cut-off point.
    """
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fence
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Handle wrapper objects like {"findings": [...]}
            values = list(result.values())
            if len(values) == 1 and isinstance(values[0], list):
                return values[0]
            return [result]
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    # Truncation recovery: extract individual complete JSON objects
    # from a cut-off array (e.g. thinking models hitting token limits).
    salvaged = _salvage_truncated_json(text)
    if salvaged:
        logger.warning(
            f"Truncated JSON response — salvaged {len(salvaged)} complete "
            f"findings from partial output ({len(text)} chars)"
        )
        return salvaged

    logger.warning(f"Failed to parse LLM JSON response: {text[:200]}")
    return []


def _looks_truncated(text: str) -> bool:
    """Return True if the text looks like a truncated JSON response with findings.

    Detects cases where the model started producing valid findings JSON
    but was cut off by a token limit before completing.
    """
    stripped = text.strip()
    return '"detected"' in stripped and stripped.startswith("[") and not stripped.endswith("]")


def _salvage_truncated_json(text: str) -> list[dict]:
    """Extract complete JSON objects from a truncated JSON array.

    Walks through the text looking for top-level {...} blocks inside
    what should be a [...] array.  Each block that parses successfully
    is kept; the first one that fails (the truncated tail) is discarded.
    """
    start = text.find("[")
    if start == -1:
        return []

    results = []
    i = start + 1
    while i < len(text):
        # Skip whitespace and commas between objects
        while i < len(text) and text[i] in " \t\n\r,":
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        # Find matching closing brace using a depth counter
        depth = 0
        obj_start = i
        in_string = False
        escape_next = False
        for j in range(i, len(text)):
            ch = text[j]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Found a complete object
                    try:
                        obj = json.loads(text[obj_start:j + 1])
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            # Reached end of text without closing brace — truncated object
            break
    return results


def _build_pair_messages(
    pairs: list[dict],
    analysis_types: list[str],
) -> list:
    """Build the prompt messages for a pair-analysis batch (shared by sync/async)."""
    pairs_text_parts = []
    for i, pair in enumerate(pairs):
        chunk_a = pair["chunk_a"]
        chunk_b = pair["chunk_b"]
        entities = pair.get("entities", [])
        entity_str = f"Shared entities: {', '.join(entities)}" if entities else "No shared entities identified."

        pairs_text_parts.append(
            f"--- Pair {i} ---\n"
            f"**Document A:** {chunk_a['source']} (Page {chunk_a['page']})\n"
            f"```\n{(chunk_a.get('text') or '')[:2000]}\n```\n\n"
            f"**Document B:** {chunk_b['source']} (Page {chunk_b['page']})\n"
            f"```\n{(chunk_b.get('text') or '')[:2000]}\n```\n\n"
            f"{entity_str}\n"
        )

    user_msg = prompts.BATCH_PAIR_ANALYSIS_USER.format(
        pairs_text="\n".join(pairs_text_parts),
        analysis_types=", ".join(analysis_types),
    )

    return [
        SystemMessage(content=prompts.PAIR_ANALYSIS_SYSTEM),
        HumanMessage(content=user_msg),
    ]


def _process_pair_findings(
    raw_findings: list[dict],
    pairs: list[dict],
    judge_index: int,
    judge_provider: str,
    judge_model: str,
    input_tokens: int,
    output_tokens: int,
) -> list[dict]:
    """Filter and enrich raw findings with metadata (shared by sync/async)."""
    results = []
    for finding in raw_findings:
        if not finding.get("detected", True):
            continue
        if float(finding.get("confidence", 0)) < 0.5:
            continue
        pair_idx = finding.get("pair_index", 0)
        if pair_idx < 0 or pair_idx >= len(pairs):
            logger.warning(
                f"Judge {judge_index}: invalid pair_index {pair_idx} "
                f"(batch has {len(pairs)} pairs), skipping finding"
            )
            continue

        pair = pairs[pair_idx]
        results.append({
            "pair_index": pair_idx,
            "judge_index": judge_index,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "issue_type": finding.get("issue_type", "CONTRADICTION"),
            "severity": finding.get("severity", "MEDIUM"),
            "confidence": float(finding.get("confidence", 0.7)),
            "title": finding.get("title", "Potential issue detected"),
            "description": finding.get("description", ""),
            "reasoning": finding.get("reasoning", ""),
            "claim_a": finding.get("claim_a", ""),
            "claim_b": finding.get("claim_b"),
            "chunk_a": pair["chunk_a"],
            "chunk_b": pair["chunk_b"],
            "entities": pair.get("entities", []),
            "input_tokens": input_tokens // max(len(pairs), 1),
            "output_tokens": output_tokens // max(len(pairs), 1),
        })
    return results


def _extract_token_counts(response) -> tuple[int, int]:
    """Extract input/output token counts from an LLM response."""
    input_tokens = getattr(response, "usage_metadata", {}).get("input_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
    output_tokens = getattr(response, "usage_metadata", {}).get("output_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0
    return input_tokens, output_tokens


# ── Async (primary) implementations ──────────────────────────────────────


async def analyze_pair_batch(
    judge: ChatOpenAI,
    pairs: list[dict],
    analysis_types: list[str],
    judge_index: int,
    judge_provider: str,
    judge_model: str,
) -> dict:
    """Send a batch of candidate pairs to a single judge.

    Returns a dict with:
      - findings: list of finding dicts with pair_index, judge metadata, and issue details
      - input_tokens: total input tokens for this LLM call
      - output_tokens: total output tokens for this LLM call
    """
    if not pairs:
        return {"findings": [], "input_tokens": 0, "output_tokens": 0}

    messages = _build_pair_messages(pairs, analysis_types)

    try:
        response = await judge.ainvoke(messages)
        input_tokens, output_tokens = _extract_token_counts(response)
        raw_findings = _parse_json_response(response.content)

        # Retry once if the response was truncated and we got 0 findings
        if not raw_findings and _looks_truncated(response.content):
            logger.warning(
                f"Judge {judge_index}: truncated response with 0 salvaged findings "
                f"({len(response.content)} chars) — retrying"
            )
            response = await judge.ainvoke(messages)
            retry_in, retry_out = _extract_token_counts(response)
            input_tokens += retry_in
            output_tokens += retry_out
            raw_findings = _parse_json_response(response.content)
    except Exception as e:
        logger.error(f"Judge {judge_index} ({judge_provider}/{judge_model}) batch failed: {e}")
        raise
    results = _process_pair_findings(
        raw_findings, pairs, judge_index, judge_provider, judge_model,
        input_tokens, output_tokens,
    )

    logger.info(f"Judge {judge_index}: batch of {len(pairs)} pairs -> {len(results)} findings")
    return {"findings": results, "input_tokens": input_tokens, "output_tokens": output_tokens}


async def _analyze_single_chunk(
    judge: ChatOpenAI,
    chunk: dict,
    judge_index: int,
    judge_provider: str,
    judge_model: str,
) -> dict:
    """Analyze a single chunk for ambiguity. Returns partial result dict."""
    text = (chunk.get("text") or "").strip()
    if not text or len(text) < 50:
        return {"findings": [], "input_tokens": 0, "output_tokens": 0, "llm_calls": 0}

    user_msg = prompts.AMBIGUITY_USER.format(
        doc_name=chunk["source"],
        page=chunk["page"],
        text=text[:3000],
    )

    messages = [
        SystemMessage(content=prompts.AMBIGUITY_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    llm_calls = 0
    try:
        response = await judge.ainvoke(messages)
        input_tokens, output_tokens = _extract_token_counts(response)
        raw_findings = _parse_json_response(response.content)
        llm_calls += 1

        if not raw_findings and _looks_truncated(response.content):
            logger.warning(
                f"Judge {judge_index}: truncated ambiguity response for "
                f"{chunk['source']} p{chunk['page']} — retrying"
            )
            response = await judge.ainvoke(messages)
            retry_in, retry_out = _extract_token_counts(response)
            input_tokens += retry_in
            output_tokens += retry_out
            raw_findings = _parse_json_response(response.content)
            llm_calls += 1
    except Exception as e:
        logger.warning(f"Judge {judge_index} ambiguity check failed for {chunk['source']} p{chunk['page']}: {e}")
        return {"findings": [], "input_tokens": 0, "output_tokens": 0, "llm_calls": 0}

    findings = []
    for finding in raw_findings:
        if not finding.get("detected", True):
            continue
        findings.append({
            "pair_index": None,
            "judge_index": judge_index,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "issue_type": "AMBIGUITY",
            "severity": finding.get("severity", "LOW"),
            "confidence": float(finding.get("confidence", 0.5)),
            "title": finding.get("title", "Ambiguous language detected"),
            "description": finding.get("description", ""),
            "reasoning": finding.get("reasoning", ""),
            "claim_a": finding.get("claim_a", ""),
            "claim_b": None,
            "chunk_a": chunk,
            "chunk_b": None,
            "entities": [],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    return {"findings": findings, "input_tokens": input_tokens, "output_tokens": output_tokens, "llm_calls": llm_calls}


async def analyze_ambiguity_batch(
    judge: ChatOpenAI,
    chunks: list[dict],
    judge_index: int,
    judge_provider: str,
    judge_model: str,
) -> dict:
    """Analyze individual chunks for ambiguity issues concurrently.

    Returns a dict with:
      - findings: list of finding dicts
      - input_tokens: total input tokens across all LLM calls
      - output_tokens: total output tokens across all LLM calls
      - llm_calls: number of LLM calls made
    """
    # Run all chunk analyses concurrently — rate limiter on the judge
    # controls actual request throughput to the provider.
    coros = [
        _analyze_single_chunk(judge, chunk, judge_index, judge_provider, judge_model)
        for chunk in chunks
    ]
    chunk_results = await asyncio.gather(*coros, return_exceptions=True)

    all_findings = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0

    for i, result in enumerate(chunk_results):
        if isinstance(result, Exception):
            logger.warning(f"Judge {judge_index} ambiguity chunk {i} raised: {result}")
            continue
        all_findings.extend(result["findings"])
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_llm_calls += result["llm_calls"]

    logger.info(f"Judge {judge_index}: ambiguity scan of {len(chunks)} chunks -> {len(all_findings)} findings")
    return {
        "findings": all_findings,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "llm_calls": total_llm_calls,
    }
