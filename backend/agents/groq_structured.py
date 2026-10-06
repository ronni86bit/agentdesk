"""Shared Groq native structured-output client for AgentDesk agents.

Implements the reliability pattern introduced with the Critic: Groq native
Structured Outputs (`response_format` = strict JSON Schema generated from a
Pydantic model) instead of Instructor-based parsing. Responses are validated
against the Pydantic model before they leave this layer, and LLM attempts are
strictly bounded (default 2) with explicit, provider-aware backoff on HTTP
429. No API key is ever logged or embedded in error messages.
"""

import re

import httpx
from pydantic import BaseModel, ValidationError

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MAX_ATTEMPTS = 2  # bounded; never an unbounded retry loop
DEFAULT_BACKOFF_SECONDS = 5.0
# Hard cap on any single backoff sleep (benchmark or API process). TPM
# windows are minutes at most; longer windows (TPD) are handled by failing
# the attempt, not by sleeping through them.
MAX_BACKOFF_SECONDS = 60.0
# Groq names the exhausted bucket in the 429 body ("... on tokens per day
# (TPD)"); response headers cannot distinguish the two scopes.
_TPD_BODY_MARKER = "tokens per day"
_TPM_BODY_MARKER = "tokens per minute"
# Free Groq tiers reject any single request whose prompt+completion exceed
# the model's tokens-per-minute budget (~8k). Keep whole prompts below this
# character target (~<7k tokens at typical ratios) so one request always fits.
TARGET_PROMPT_CHARS = 15000


class StructuredOutputError(Exception):
    """A structured-output request or validation failure.

    `retry_after` carries the (optional) number of seconds the provider asked
    us to wait before the next attempt (HTTP 429), already accounting for the
    token-bucket reset window when present. `rate_limit_scope` is "tpd" or
    "tpm" when a 429 body identifies a tokens-per-day / tokens-per-minute
    limit, else None.
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        rate_limit_scope: str | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.rate_limit_scope = rate_limit_scope


def classify_rate_limit(body: str | None) -> str | None:
    """Classify a provider 429 body: "tpd", "tpm", or None."""
    if not body:
        return None
    lowered = body.lower()
    if _TPD_BODY_MARKER in lowered:
        return "tpd"
    if _TPM_BODY_MARKER in lowered:
        return "tpm"
    return None


def parse_reset_seconds(raw: str | None) -> float | None:
    """Parse Groq rate-limit reset durations like '2.077s' or '1m30s'."""
    if not raw:
        return None
    match = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)", raw.strip())
    if not match:
        return None
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2)) if match.group(2) else 0.0
    total = minutes * 60.0 + seconds
    return total if total > 0 else None


def strict_schema(model_cls: type[BaseModel]) -> dict:
    """Build a strict-mode-compatible JSON Schema from a Pydantic model.

    Adjustments (semantically equivalent): every property becomes required and
    objects get additionalProperties=false, as Structured Outputs strict mode
    demands; cosmetic title/description metadata is stripped.
    """
    schema = model_cls.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    properties = schema.get("properties", {})
    for prop in properties.values():
        prop.pop("title", None)
        prop.pop("default", None)
        items = prop.get("items")
        if isinstance(items, dict):
            items.pop("title", None)
    schema["required"] = list(properties)
    schema["additionalProperties"] = False
    return schema


def build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def request_structured_completion(
    system_prompt: str,
    user_prompt: str,
    cfg,
    schema_name: str,
    schema: dict,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Call Groq chat completions with native strict JSON-schema output.

    Returns the raw message content string. Raises StructuredOutputError on
    transport, HTTP, or unusable-body failures (with provider backoff info on
    429). The API key is sent only in the Authorization header.
    """
    payload = {
        "model": cfg.groq_model,
        "messages": build_messages(system_prompt, user_prompt),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }
    # Reasoning models spend completion tokens on hidden chain-of-thought
    # before the JSON document; left unchecked they truncate the output
    # ("max completion tokens reached"). Minimize reasoning for these
    # deterministic extraction tasks on Groq's OpenAI open-weight models.
    if "gpt-oss" in cfg.groq_model:
        payload["reasoning_effort"] = "low"
    headers = {
        "Authorization": f"Bearer {cfg.groq_api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            GROQ_CHAT_COMPLETIONS_URL,
            json=payload,
            headers=headers,
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        raise StructuredOutputError(
            f"LLM request failed: {type(exc).__name__}"
        ) from exc
    if response.status_code != 200:
        retry_after: float | None = None
        if response.status_code == 429:
            raw_retry = response.headers.get("Retry-After")
            try:
                retry_after = float(raw_retry) if raw_retry is not None else None
            except (TypeError, ValueError):
                retry_after = None
            # TPM limits may reset on a longer window than Retry-After
            # reports; the token-bucket reset header is authoritative.
            token_reset = parse_reset_seconds(
                response.headers.get("x-ratelimit-reset-tokens")
            )
            if token_reset is not None:
                retry_after = max(retry_after or 0.0, token_reset)
        # Include a truncated provider body so schema/param rejections are
        # self-describing. Provider 4xx bodies never contain our credentials.
        raise StructuredOutputError(
            f"LLM request returned HTTP {response.status_code}: "
            f"{response.text[:200]}",
            retry_after=retry_after,
            rate_limit_scope=(
                classify_rate_limit(response.text)
                if response.status_code == 429 else None
            ),
        )
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise StructuredOutputError(
            "LLM returned an unusable response body"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputError("LLM returned empty content")
    return content


def backoff_delay(error: StructuredOutputError) -> float:
    """Explicit bounded backoff for one retry of a failed attempt."""
    delay = error.retry_after if error.retry_after is not None else DEFAULT_BACKOFF_SECONDS
    return min(delay, MAX_BACKOFF_SECONDS)


def parse_structured_output(content: str, model_cls: type[BaseModel]) -> BaseModel:
    """Parse raw JSON text and validate it against a Pydantic model."""
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("output was not valid JSON") from exc
    if not isinstance(data, dict):
        raise StructuredOutputError("output was not a JSON object")
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"output failed {model_cls.__name__} validation: "
            f"{exc.error_count()} validation error(s)"
        ) from exc


def agent_system_prompt(role: str, goal: str, backstory: str) -> str:
    """Compose the system prompt from CrewAI agent role metadata."""
    return (
        f"You are {role}. Goal: {goal}\n\n{backstory}\n\n"
        "Respond with a single JSON object that exactly matches the requested "
        "schema. Do not add commentary before or after the JSON."
    )


def source_budget_chars(fixed_prompt_chars: int) -> int:
    """Compute how many characters of numbered-source text still fit.

    `fixed_prompt_chars` is everything else the request will contain
    (instructions, plan, previous report, critique, expected-output note).
    The result keeps the total prompt under TARGET_PROMPT_CHARS with a small
    floor so tiny prompts still receive usable evidence.
    """
    return max(1500, TARGET_PROMPT_CHARS - fixed_prompt_chars)
