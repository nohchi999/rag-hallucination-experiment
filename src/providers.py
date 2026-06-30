"""
src/providers.py  (cross-model extension, added 2026-06-29)

Unified chat interface over multiple LLM providers. Normalizes every backend to
the SAME return shape the original pipeline already stores, so raw_results stay
schema-compatible:

    chat(provider, model, user_message, temperature, max_tokens, **opts)
        -> (text: str, full_response: dict)

    full_response = {
        "id", "model", "stop_reason",
        "usage": {"input_tokens", "output_tokens"},
        "raw_text",                # ALWAYS the model's text -> v2 reparse can run
        "provider",
    }

Retry/backoff is intentionally NOT here — callers (generator/judge) own the retry
loop so behaviour matches the original Anthropic generator. This module makes one
call and raises on failure.

Clients are lazily constructed so importing this module never requires every SDK
or every API key to be present — only the provider you actually use.
"""

import config

# --- lazy singletons -------------------------------------------------------
_anthropic_client = None
_openai_client = None
_google_client = None
_openrouter_client = None


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_google():
    global _google_client
    if _google_client is None:
        from google import genai
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY is not set.")
        _google_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _google_client


def _get_judge_client():
    """Judge endpoint is OpenAI-compatible (Alibaba DashScope OR OpenRouter OR any
    other). Reuse the openai SDK with the configured base_url + judge key."""
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        if not config.JUDGE_API_KEY:
            raise ValueError(
                "No judge API key set (JUDGE_API_KEY / OPENROUTER_API_KEY / DASHSCOPE_API_KEY)."
            )
        _openrouter_client = OpenAI(
            api_key=config.JUDGE_API_KEY,
            base_url=config.JUDGE_BASE_URL,
        )
    return _openrouter_client


# --- per-provider calls ----------------------------------------------------
def _chat_anthropic(model, user_message, temperature, max_tokens):
    client = _get_anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user_message}],
    )
    text = resp.content[0].text
    full = {
        "id": resp.id,
        "model": resp.model,
        "stop_reason": resp.stop_reason,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
        "raw_text": text,
        "provider": "anthropic",
    }
    return text, full


def _chat_openai(model, user_message, temperature, max_tokens):
    client = _get_openai()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_message}],
    )
    text = resp.choices[0].message.content or ""
    full = {
        "id": resp.id,
        "model": resp.model,
        "stop_reason": resp.choices[0].finish_reason,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
        "raw_text": text,
        "provider": "openai",
    }
    return text, full


def _chat_google(model, user_message, temperature, max_tokens):
    client = _get_google()
    from google.genai import types
    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        # Disable thinking so 2.5-flash behaves as a plain instruct generator
        # (comparable to Haiku) and we don't pay for thinking tokens.
        thinking_config=types.ThinkingConfig(
            thinking_budget=config.GOOGLE_THINKING_BUDGET
        ),
    )
    resp = client.models.generate_content(
        model=model,
        contents=user_message,
        config=gen_config,
    )
    text = resp.text or ""
    usage = getattr(resp, "usage_metadata", None)
    full = {
        "id": getattr(resp, "response_id", None),
        "model": model,
        "stop_reason": (
            resp.candidates[0].finish_reason.name
            if getattr(resp, "candidates", None) and resp.candidates[0].finish_reason
            else None
        ),
        "usage": {
            "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        },
        "raw_text": text,
        "provider": "google",
    }
    return text, full


def _chat_judge(model, user_message, temperature, max_tokens, provider_pin=""):
    """OpenAI-compatible judge call (DashScope direct, OpenRouter, etc.)."""
    client = _get_judge_client()
    extra_body = {}
    if provider_pin and config.JUDGE_PROVIDER == "openrouter":
        # OpenRouter-only: pin a single backend; do not silently fall back to a
        # different/quantized provider — required for judge reproducibility.
        extra_body["provider"] = {"order": [provider_pin], "allow_fallbacks": False}
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_message}],
        extra_body=extra_body or None,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    full = {
        "id": resp.id,
        "model": resp.model,
        "stop_reason": resp.choices[0].finish_reason,
        "usage": {
            "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        },
        "raw_text": text,
        "provider": config.JUDGE_PROVIDER,
        # which physical backend served it (OpenRouter exposes this; for the manifest)
        "served_by": getattr(resp, "provider", None),
    }
    return text, full


# --- usage accounting (for real-cost reporting in run manifests) -----------
USAGE_LOG = []  # list of {"provider","model","input_tokens","output_tokens"}


def reset_usage():
    USAGE_LOG.clear()


def usage_totals():
    """Aggregate recorded usage by (provider, model)."""
    agg = {}
    for u in USAGE_LOG:
        key = (u["provider"], u["model"])
        a = agg.setdefault(key, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        a["input_tokens"] += u.get("input_tokens") or 0
        a["output_tokens"] += u.get("output_tokens") or 0
        a["calls"] += 1
    return agg


# --- public dispatch -------------------------------------------------------
def chat(provider, model, user_message, temperature, max_tokens, **opts):
    """Single chat call. Raises on API/SDK error (caller handles retry)."""
    if provider == "anthropic":
        text, full = _chat_anthropic(model, user_message, temperature, max_tokens)
    elif provider == "openai":
        text, full = _chat_openai(model, user_message, temperature, max_tokens)
    elif provider == "google":
        text, full = _chat_google(model, user_message, temperature, max_tokens)
    else:
        # any non-generator provider (openrouter, dashscope, ...) -> OpenAI-compatible judge
        text, full = _chat_judge(
            model, user_message, temperature, max_tokens,
            provider_pin=opts.get("provider_pin", config.JUDGE_PROVIDER_PIN),
        )

    USAGE_LOG.append({
        "provider": provider,
        "model": full.get("model") or model,
        "input_tokens": full.get("usage", {}).get("input_tokens"),
        "output_tokens": full.get("usage", {}).get("output_tokens"),
    })
    return text, full
