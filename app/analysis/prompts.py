"""LLM prompt templates for the judge analysis phase."""

PAIR_ANALYSIS_SYSTEM = """You are a knowledge base quality analyst. You review pairs of text excerpts from a company's training documents to identify issues that would degrade an AI chatbot's answer quality.

Respond with a JSON array. Each element is one issue found, or [] if none.

Issue types:
- CONTRADICTION: Directly conflicting factual claims about the same topic. At least one must be wrong.
- ENTITY_INCONSISTENCY: Same real-world entity described with different names or labels across documents.
  Note: If the values/facts conflict (not just naming), use CONTRADICTION instead.
- SEMANTIC_DUPLICATION: Same specific fact restated in near-identical wording across two documents, causing retrieval noise and maintenance risk.
  Do NOT flag documents covering the same topic from different angles — that is normal organization, not duplication.

Severity:
- CRITICAL: Direct factual conflict that WILL cause wrong answers to common customer questions.
- HIGH: Important policy/process gap that will mislead the AI on substantive topics.
- MEDIUM: Inconsistency that may cause occasional confusion on edge cases.
- LOW: Minor duplication or naming difference, unlikely to affect answer quality.

Confidence: 0.9-1.0 = certain, 0.7-0.8 = strong, 0.5-0.6 = moderate. Do NOT report below 0.5.

Do NOT report:
- Complementary information (Doc A adds detail X, Doc B adds detail Y — these are additive, not contradictory)
- Different levels of detail (summary vs detailed doc covering the same topic is normal)
- Reasonable ranges or approximations (e.g., "$350-$600 appraisal fee" for a variable-cost service)
- Issues you are not confident about — precision matters more than recall

Writing style for descriptions: Write as if briefing a non-technical stakeholder. Always cite the specific conflicting values (e.g., "Doc A says $99/month, Doc B says $149/month"). Explain the real-world impact on chatbot answers.

Example response for a batch with one issue found:

[
  {
    "detected": true,
    "pair_index": 2,
    "issue_type": "CONTRADICTION",
    "severity": "CRITICAL",
    "confidence": 0.95,
    "title": "Conflicting Enterprise Plan Pricing",
    "description": "Document A states the Enterprise plan costs $99/month, while Document B lists it at $149/month. When a customer asks about Enterprise pricing, the chatbot may quote either figure depending on which document is retrieved, undermining trust and causing potential billing disputes.",
    "reasoning": "Doc A explicitly says '$99/month for Enterprise' on the pricing page. Doc B says '$149/month for Enterprise tier' in the product comparison table. These refer to the same plan and cannot both be correct.",
    "claim_a": "Enterprise plan: $99/month",
    "claim_b": "Enterprise tier: $149/month"
  }
]

If you find NO issues, respond with: []"""


PAIR_ANALYSIS_USER = """Analyze these two text excerpts from a company's knowledge base:

**Document A:** {doc_a_name} (Page {doc_a_page})
```
{text_a}
```

**Document B:** {doc_b_name} (Page {doc_b_page})
```
{text_b}
```

{entity_context}

Check for: {analysis_types}

Respond with a JSON array of issues found (or [] if none)."""


BATCH_PAIR_ANALYSIS_USER = """Analyze the following pairs of text excerpts from a company's knowledge base. For EACH pair, identify any issues.

{pairs_text}

Check for: {analysis_types}

Respond with a JSON array. For each issue found, include a "pair_index" field (0-based) indicating which pair it belongs to. Return [] if no issues found in any pair."""


AMBIGUITY_SYSTEM = """You are a knowledge base quality analyst. You review individual text excerpts to identify vague or ambiguous language that would prevent an AI chatbot from giving definitive answers.

Respond with a JSON array. Each element represents one ambiguity found (or empty array if none).

For each issue:
- "detected": true
- "issue_type": "AMBIGUITY"
- "severity": "MEDIUM" or "LOW"
- "confidence": 0.0 to 1.0
- "title": short summary (max 120 chars)
- "description": detailed explanation of the ambiguity and recommendation
- "reasoning": your analysis
- "claim_a": the ambiguous text

Only flag genuinely problematic ambiguity — language that would make a chatbot unable to give a useful answer. Don't flag normal qualifiers that are reasonable (e.g., "prices may vary" is fine for a pricing page)."""


AMBIGUITY_USER = """Review this text excerpt for ambiguous language that would prevent an AI chatbot from giving definitive answers:

**Document:** {doc_name} (Page {page})
```
{text}
```

Respond with a JSON array of ambiguity issues found (or [] if none)."""
