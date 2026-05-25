"""Shared agent runner. Every agent calls run_agent() to talk to Claude.

Responsibilities:
- Build the conversation, call the Anthropic API.
- Handle the tool-use loop for agents that have tools (Curator).
- Validate the final JSON output with a Pydantic model.
- Retry up to N times on parse/validation failure, feeding the error back.
- Cache results in agent_traces keyed by (agent_name, hash(input)).
- Log a trace row for every call (success or failure).
"""

import hashlib
import json
import logging
import time
from typing import Callable, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from config import ANTHROPIC_API_KEY, DEFAULT_MODEL
from db import find_cached_trace, save_trace

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentParseError(Exception):
    """Raised when an agent fails to produce valid output after all retries."""


_client: Anthropic | None = None


def _get_anthropic() -> Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in .env")
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _canonical_json(obj) -> str:
    """Stable JSON for hashing. Sorted keys, no whitespace surprises."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _hash_input(agent_name: str, user_input: dict) -> str:
    payload = agent_name + "::" + _canonical_json(user_input)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> str:
    """Strip common LLM wrappers around a JSON object."""
    s = text.strip()
    if s.startswith("```"):
        # Remove an optional ```json fence and the closing ```
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    # If the model added prose before/after, isolate the JSON object.
    if not s.startswith("{") and "{" in s:
        s = s[s.index("{") : s.rindex("}") + 1]
    return s


def run_agent(
    agent_name: str,
    system_prompt: str,
    user_input: dict,
    output_model: type[T],
    model: str = DEFAULT_MODEL,
    tools: list[dict] | None = None,
    tool_dispatcher: Callable[[str, dict], object] | None = None,
    playlist_id: int | None = None,
    max_retries: int = 2,
    max_tool_iterations: int = 8,
    use_cache: bool = True,
    max_tokens: int = 4096,
) -> T:
    """Call Claude, validate the JSON output, return the typed result.

    On parse/validation failure, retry with the error fed back as the next
    user message. On tool_use, run the tool and feed the result back.
    """
    input_hash = _hash_input(agent_name, user_input)
    input_json = _canonical_json(user_input)

    # Cache check
    if use_cache:
        cached = find_cached_trace(agent_name, input_hash)
        if cached is not None:
            logger.info("cache hit: agent=%s hash=%s", agent_name, input_hash[:8])
            return output_model.model_validate(cached)

    client = _get_anthropic()

    # Build the initial conversation.
    user_text = (
        "Input:\n```json\n"
        + json.dumps(user_input, indent=2, ensure_ascii=False)
        + "\n```\n\n"
        "Respond with a single JSON object that matches the required schema. "
        "Do not include any text before or after the JSON."
    )
    messages: list[dict] = [{"role": "user", "content": user_text}]

    start = time.time()
    tokens_in = 0
    tokens_out = 0

    parse_attempts = 0
    final_text: str | None = None

    while True:
        # ---- Anthropic call ----
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens

        # ---- Tool use? Run tools and loop. ----
        if response.stop_reason == "tool_use" and tools:
            if tool_dispatcher is None:
                raise RuntimeError(
                    f"{agent_name}: model requested tool use but no dispatcher provided"
                )
            # Append assistant message exactly as returned (preserve tool_use ids).
            messages.append(
                {"role": "assistant", "content": response.content}
            )
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                try:
                    result = tool_dispatcher(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                except Exception as e:
                    logger.exception("tool %s failed", block.name)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"tool error: {e}",
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            max_tool_iterations -= 1
            if max_tool_iterations <= 0:
                raise AgentParseError(
                    f"{agent_name}: exceeded max tool iterations"
                )
            continue

        # ---- Final answer. Extract text. ----
        final_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                final_text += block.text

        # ---- Validate. ----
        try:
            cleaned = _extract_json(final_text)
            parsed = json.loads(cleaned)
            validated = output_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            parse_attempts += 1
            if parse_attempts > max_retries:
                latency_ms = int((time.time() - start) * 1000)
                save_trace(
                    agent_name=agent_name,
                    input_hash=input_hash,
                    input_json=input_json,
                    output_json=None,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_ms=latency_ms,
                    playlist_id=playlist_id,
                )
                raise AgentParseError(
                    f"{agent_name}: failed to produce valid output after "
                    f"{max_retries} retries. Last error: {e}\nLast text: {final_text[:500]}"
                ) from e
            # Retry with the error fed back.
            messages.append({"role": "assistant", "content": final_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed validation:\n"
                        f"{e}\n\n"
                        "Please respond with valid JSON matching the schema. "
                        "Output JSON only, no markdown fences, no prose."
                    ),
                }
            )
            continue

        # ---- Success. ----
        latency_ms = int((time.time() - start) * 1000)
        save_trace(
            agent_name=agent_name,
            input_hash=input_hash,
            input_json=input_json,
            output_json=validated.model_dump_json(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            playlist_id=playlist_id,
        )
        return validated
