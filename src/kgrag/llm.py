"""LLM provider layer: the pipeline calls generate_structured(); LLM_PROVIDER picks who answers.

- gemini (default): Google Gemini API free tier. Rate limits are per project and not
  published, so requests are paced, per-minute 429s slow the pacer down, and a per-day
  429 raises QuotaExhausted so the run stops cleanly and resumes from cache later.
- claude: Anthropic API (paid), kept so the provider can be switched back with one setting.
"""

import json
import re
import threading
import time
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from kgrag.config import settings

# $ per million tokens at paid-tier list prices: (input, output, cache write, cache read).
# On the Gemini free tier the real cost is $0; these give the paid-equivalent figure for the README.
PRICES = {
    "gemini-3.5-flash": (1.50, 9.00, 0.0, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.0, 0.30),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
}


class QuotaExhausted(Exception):
    """Daily quota used up. Stop the run; cached results are kept."""


class LLMError(Exception):
    """One attempt failed (invalid JSON, schema mismatch, refusal, truncation). Retryable."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0       # includes thinking tokens, which are billed as output
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def merge(self, other: "Usage") -> None:
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def cost(self, model: str) -> float:
        p_in, p_out, p_cw, p_cr = PRICES.get(model, (0, 0, 0, 0))
        return (self.input_tokens * p_in + self.output_tokens * p_out
                + self.cache_write_tokens * p_cw + self.cache_read_tokens * p_cr) / 1e6


def model_name() -> str:
    return settings.gemini_model if settings.llm_provider == "gemini" else settings.claude_model


def generate_structured(system: str, user: str, schema: type[BaseModel],
                        light: bool = False) -> tuple[BaseModel, Usage]:
    """Return a validated instance of `schema`. Raises LLMError (retryable) or QuotaExhausted.

    light=True: minimal reasoning, for cheap high-volume calls such as query routing.
    """
    if settings.llm_provider == "gemini":
        return _gemini(system, user, schema, light)
    if settings.llm_provider == "claude":
        return _claude(system, user, schema, light)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


# ---------------------------------------------------------------- Gemini

class _Pacer:
    """Spaces requests at least `interval` seconds apart across threads; slows down on 429."""

    def __init__(self, interval: float):
        self.interval = interval
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))

    def slow_down(self) -> None:
        with self._lock:
            self.interval = min(self.interval * 1.5, 60.0)


_pacer = _Pacer(settings.gemini_min_interval_s)
_gemini_client = None
TRANSIENT_RETRIES = 5          # per call, for 429 per-minute limits and 503/504 server errors
REQUEST_TIMEOUT_MS = 180_000   # a stuck request fails after 3 minutes instead of hanging


def _gemini_quota_info(err) -> tuple[bool, float, str | None]:
    """From a 429 body: (is daily quota, seconds to wait, quota limit if reported)."""
    body = json.dumps(err.details or {})
    daily = "PerDay" in body
    delay = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', body)
    limit = re.search(r'"quotaValue":\s*"(\d+)"', body)
    return daily, float(delay.group(1)) if delay else 60.0, limit.group(1) if limit else None


def _gemini(system: str, user: str, schema: type[BaseModel], light: bool) -> tuple[BaseModel, Usage]:
    global _gemini_client
    from google import genai
    import httpx  # the transport google-genai uses; its network errors are not APIErrors
    from google.genai import errors, types

    if _gemini_client is None:
        if not settings.gemini_api_key:
            raise SystemExit("Set GEMINI_API_KEY in .env")
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL" if light else settings.gemini_thinking_level),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    for attempt in range(1, TRANSIENT_RETRIES + 1):
        _pacer.wait()
        try:
            response = _gemini_client.models.generate_content(
                model=settings.gemini_model, contents=user, config=config)
            break
        except errors.ClientError as e:
            if e.code != 429:
                raise
            daily, delay, limit = _gemini_quota_info(e)
            if daily:
                raise QuotaExhausted(f"Gemini daily quota reached (limit: {limit or 'not reported'} requests/day)")
            print(f"  [rate limited] waiting {delay:.0f}s, slowing pacer to {_pacer.interval * 1.5:.1f}s/request")
            _pacer.slow_down()
            time.sleep(delay)
        except errors.ServerError as e:  # 503 overloaded / 504 deadline: transient on the free tier
            delay = 15 * attempt
            print(f"  [server {e.code}] waiting {delay}s before retry {attempt}/{TRANSIENT_RETRIES}")
            time.sleep(delay)
        except httpx.TransportError as e:  # connect/read timeouts, dropped connections
            delay = 15 * attempt
            print(f"  [network {type(e).__name__}] waiting {delay}s before retry {attempt}/{TRANSIENT_RETRIES}")
            time.sleep(delay)
    else:
        raise LLMError("still failing after transient-error retries")

    meta = response.usage_metadata
    usage = Usage(input_tokens=meta.prompt_token_count or 0,
                  output_tokens=(meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
    finish = response.candidates[0].finish_reason.name if response.candidates else "NO_CANDIDATES"
    if finish != "STOP":
        raise LLMError(f"finish_reason={finish}")
    try:
        return schema.model_validate_json(response.text), usage
    except ValidationError as e:
        raise LLMError(f"schema validation failed: {e.error_count()} errors") from e


# ---------------------------------------------------------------- Claude

def _claude(system: str, user: str, schema: type[BaseModel], light: bool) -> tuple[BaseModel, Usage]:
    import anthropic

    headers = ({"anthropic-workspace-id": settings.anthropic_workspace_id}
               if settings.anthropic_workspace_id else None)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None, default_headers=headers)
    try:
        response = client.beta.messages.parse(
            model=settings.claude_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low" if light else "medium"},
            output_format=schema,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
    except ValidationError as e:
        raise LLMError(f"schema validation failed: {e.error_count()} errors") from e
    u = response.usage
    usage = Usage(u.input_tokens, u.output_tokens,
                  u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0)
    if response.stop_reason in ("refusal", "max_tokens"):
        raise LLMError(f"stop_reason={response.stop_reason}")
    parsed = next((b.parsed_output for b in response.content if b.type == "text"), None)
    if parsed is None:
        raise LLMError("no parsed output")
    return parsed, usage
