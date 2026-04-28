from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core import settings
from app.database_layer.db_schemas import JudgeConfig


def create_judge(config: JudgeConfig, rate_limiter=None):
    """Create a LangChain chat model for the given judge configuration.

    - provider="anthropic": uses ChatAnthropic (native SDK, no OpenAI compat layer needed)
    - all other providers (litellm, fireworks, openrouter, google): uses ChatOpenAI
      because they all expose an OpenAI-compatible API endpoint.

    Args:
        config: Judge configuration (model, provider, etc.).
        rate_limiter: Optional InMemoryRateLimiter shared across judges
            using the same (provider, model) to enforce RPM limits.
    """
    api_key = config.api_key or settings.get_provider_key(config.provider)

    # Gemini models require temperature=1.0; lowering it causes unexpected
    # behaviour (looping, degraded reasoning).  Other models default to 0.1.
    if config.temperature is not None:
        temperature = config.temperature
    elif "gemini" in config.model.lower():
        temperature = 1.0
    else:
        temperature = 0.1

    # ── Native Anthropic path ────────────────────────────────────────────
    if config.provider == "anthropic":
        kwargs = dict(
            model=config.model,
            api_key=api_key,
            temperature=temperature,
            max_retries=2,
        )
        if rate_limiter:
            kwargs["rate_limiter"] = rate_limiter
        if config.reasoning_effort:
            # Claude extended thinking — use budget_tokens instead of max_tokens
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.max_tokens or 10000,
            }
            # Extended thinking requires temperature=1
            kwargs["temperature"] = 1
        else:
            kwargs["max_tokens"] = config.max_tokens or 4000

        # ChatAnthropic does not support response_format; JSON is enforced
        # via the system prompt (prompts.py already instructs JSON output).
        return ChatAnthropic(**kwargs)

    # ── OpenAI-compatible path (litellm, fireworks, openrouter, google) ──
    api_base = config.api_base or settings.get_provider_base(config.provider)

    kwargs = dict(
        model=config.model,
        api_key=api_key,
        base_url=api_base,
        temperature=temperature,
        max_retries=2,
    )

    if rate_limiter:
        kwargs["rate_limiter"] = rate_limiter

    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
        kwargs["max_completion_tokens"] = config.max_tokens or 16000
    else:
        kwargs["max_tokens"] = config.max_tokens or 4000

    judge = ChatOpenAI(**kwargs)

    # Enforce JSON output at the API level for OpenAI-compatible providers.
    return judge.bind(response_format={"type": "json_object"})
