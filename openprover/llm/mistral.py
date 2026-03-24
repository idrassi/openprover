"""Mistral Conversations API client for OpenProver.

Leanstral supports explicit reasoning via reasoning_effort='high'
(temperature forced to 1.0).  Reasoning tokens arrive as structured
content dicts; tool calls use the function.call.delta SSE event type.
"""

import json
import logging
import os
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

from ._base import Interrupted, archive

logger = logging.getLogger("openprover.llm")

BASE_URL = "https://api.mistral.ai"


# ── SSE helpers ──────────────────────────────────────────────────────

def _extract_sse_data(line: str) -> str | None:
    """Return the payload after 'data: ', or None for non-data lines."""
    if not line or not line.startswith("data:"):
        return None
    return line[len("data:"):].lstrip()


def _parse_content_delta(content, thinking_parts, output_parts, callback):
    """Parse the 'content' field from a message.output.delta chunk.

    When reasoning_effort='high', content is a dict like:
        {"type": "thinking", "thinking": [{"type": "text", "text": "..."}]}
    Otherwise it's a plain string.
    """
    if isinstance(content, str):
        if content:
            output_parts.append(content)
            if callback:
                callback(content, "text")
        return

    if not isinstance(content, dict):
        return

    ctype = content.get("type", "")
    if ctype == "thinking":
        for part in content.get("thinking", []):
            text = part.get("text", "")
            if text:
                thinking_parts.append(text)
                if callback:
                    callback(text, "thinking")
    else:
        # Regular output in structured form
        text = ""
        for part in content.get("content", []):
            text += part.get("text", "")
        if not text:
            text = content.get("text", "")
        if text:
            output_parts.append(text)
            if callback:
                callback(text, "text")


def _merge_tool_call_delta(acc, chunk):
    """Accumulate a function.call.delta chunk into acc[id]."""
    fc_id = chunk.get("id", "")
    entry = acc.setdefault(fc_id, {"id": fc_id, "tool_call_id": "", "name": "", "arguments": ""})
    if chunk.get("tool_call_id"):
        entry["tool_call_id"] = chunk["tool_call_id"]
    if chunk.get("name"):
        entry["name"] = chunk["name"]
    entry["arguments"] += chunk.get("arguments", "")


def _normalize_tool_calls(acc):
    """Convert accumulated tool calls to OpenAI format for the prover."""
    if not acc:
        return None
    return [
        {
            "id": tc["tool_call_id"] or tc["id"],
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"]},
        }
        for tc in acc.values()
    ]


# ── Client ───────────────────────────────────────────────────────────

class MistralClient:
    """Calls the Mistral Conversations API and archives interactions."""

    context_length = 256_000
    mistral = True  # Used by prover for tool-routing dispatch

    def __init__(self, model: str, archive_dir: Path, answer_reserve: int = 4096):
        self.model = model
        self.archive_dir = archive_dir
        self.call_count = 0
        self.total_cost = 0.0
        self.max_output_tokens = answer_reserve
        self._api_key = os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise SystemExit(
                "Error: MISTRAL_API_KEY environment variable not set.\n"
                "  Get an API key from https://console.mistral.ai/"
            )
        self._interrupted = threading.Event()
        self._soft_interrupted = threading.Event()

    # ── Interrupt interface ──────────────────────────────────────────

    def interrupt(self):
        self._interrupted.set()

    def soft_interrupt(self):
        self._soft_interrupted.set()

    def cleanup(self):
        pass

    def clear_interrupt(self):
        self._interrupted.clear()
        self._soft_interrupted.clear()

    def clear_soft_interrupt(self):
        self._soft_interrupted.clear()

    # ── HTTP helper ──────────────────────────────────────────────────

    def _request(self, payload: dict, timeout: int = 600):
        req = urllib.request.Request(
            f"{BASE_URL}/v1/conversations",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        return urllib.request.urlopen(req, timeout=timeout)

    # ── call() ───────────────────────────────────────────────────────

    def call(
        self,
        prompt: str,
        system_prompt: str,
        json_schema: dict | None = None,
        label: str = "",
        web_search: bool = False,
        stream_callback=None,
        archive_path: Path | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Single-turn call. Same interface as LLMClient.call()."""
        self.call_count += 1
        call_num = self.call_count

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      None, None, 0, archive_path)

        logger.info("[%s] calling %s%s", label, self.model,
                    " (streaming)" if stream_callback else "")

        payload = self._build_payload(
            inputs=[{"role": "user", "content": prompt}],
            instructions=system_prompt,
            max_tokens=max_tokens,
            stream=bool(stream_callback),
        )
        start = time.time()

        if self._interrupted.is_set():
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", 0, archive_path)
            raise Interrupted()

        try:
            if stream_callback:
                return self._stream(
                    payload, call_num, label, start, stream_callback, archive_path,
                    prompt=prompt, system_prompt=system_prompt, json_schema=json_schema,
                )
            else:
                return self._non_streaming(
                    payload, call_num, label, start, archive_path,
                    prompt=prompt, system_prompt=system_prompt, json_schema=json_schema,
                )
        except (urllib.error.URLError, ConnectionError) as e:
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, str(e), elapsed_ms, archive_path)
            raise RuntimeError(f"Mistral API request failed: {e}")

    # ── chat() ───────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        label: str = "",
        stream_callback=None,
        archive_path: Path | None = None,
    ) -> dict:
        """Multi-turn chat with optional tool calling.

        Args:
            messages: OpenAI-format message list.
            tools: OpenAI-format tool definitions, or None.
        """
        self.call_count += 1
        call_num = self.call_count

        # Separate system message from inputs
        instructions = ""
        inputs = []
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            else:
                inputs.append(msg)

        prompt_text = json.dumps(messages, ensure_ascii=False)
        self._archive(call_num, label, prompt_text, "", None,
                      None, None, 0, archive_path)

        logger.info("[%s] chat %s%s", label, self.model,
                    " (streaming)" if stream_callback else "")

        payload = self._build_payload(
            inputs=inputs,
            instructions=instructions,
            tools=tools,
            max_tokens=max_tokens,
            stream=bool(stream_callback),
        )
        start = time.time()

        if self._interrupted.is_set():
            self._archive(call_num, label, prompt_text, "", None,
                          None, "interrupted", 0, archive_path)
            raise Interrupted()

        try:
            if stream_callback:
                return self._stream(
                    payload, call_num, label, start, stream_callback, archive_path,
                    prompt=prompt_text, system_prompt="",
                )
            else:
                return self._non_streaming(
                    payload, call_num, label, start, archive_path,
                    prompt=prompt_text, system_prompt="",
                )
        except (urllib.error.URLError, ConnectionError) as e:
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt_text, "", None,
                          None, str(e), elapsed_ms, archive_path)
            raise RuntimeError(f"Mistral API request failed: {e}")

    # ── Payload builder ──────────────────────────────────────────────

    def _build_payload(self, *, inputs, instructions="", tools=None,
                       max_tokens=None, stream=False):
        effective_max = max_tokens if max_tokens is not None else self.max_output_tokens
        payload = {
            "model": self.model,
            "inputs": inputs,
            "instructions": instructions,
            "tools": tools or [],  # explicit [] disables auto tool calling
            "stream": stream,
            "completion_args": {
                "temperature": 1.0,
                "max_tokens": effective_max,
                "top_p": 1,
                "reasoning_effort": "high",
            },
        }
        return payload

    # ── Non-streaming path ───────────────────────────────────────────

    def _non_streaming(self, payload, call_num, label, start, archive_path,
                       *, prompt="", system_prompt="", json_schema=None):
        try:
            resp = self._request(payload)
            raw = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, f"HTTP {e.code}: {body}", elapsed_ms, archive_path)
            raise RuntimeError(f"HTTP {e.code}: {body[:1000]}")

        elapsed_ms = int((time.time() - start) * 1000)

        if self._interrupted.is_set():
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", elapsed_ms, archive_path)
            raise Interrupted()

        # Parse outputs
        result_text = ""
        thinking_text = ""
        tool_calls = None
        for entry in raw.get("outputs", []):
            if entry.get("role") != "assistant":
                continue
            result_text = entry.get("content", "")
            thinking_text = entry.get("reasoning", "")
            tc = entry.get("tool_calls")
            if tc:
                tool_calls = tc

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      raw, None, elapsed_ms, archive_path,
                      thinking=thinking_text, result_text=result_text)
        logger.info("[%s] done %dms", label, elapsed_ms)

        finish_reason = "tool_calls" if tool_calls else "stop"
        return {
            "result": result_text,
            "thinking": thinking_text,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": raw,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    # ── Streaming path ───────────────────────────────────────────────

    def _stream(self, payload, call_num, label, start, callback, archive_path,
                *, prompt="", system_prompt="", json_schema=None):
        try:
            resp = self._request(payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            elapsed_ms = int((time.time() - start) * 1000)
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, f"HTTP {e.code}: {body}", elapsed_ms, archive_path)
            raise RuntimeError(f"HTTP {e.code}: {body[:1000]}")

        thinking_parts: list[str] = []
        output_parts: list[str] = []
        tool_call_acc: dict[str, dict] = {}
        interrupted = False
        soft_interrupted = False

        for raw_line in resp:
            if self._interrupted.is_set():
                interrupted = True
                resp.close()
                break
            if self._soft_interrupted.is_set():
                soft_interrupted = True
                resp.close()
                break

            line = raw_line.decode(errors="replace").strip()
            data_str = _extract_sse_data(line)
            if data_str is None:
                continue
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = chunk.get("type", "")

            if event_type == "function.call.delta":
                _merge_tool_call_delta(tool_call_acc, chunk)
            elif event_type == "message.output.delta":
                _parse_content_delta(chunk.get("content", ""),
                                     thinking_parts, output_parts, callback)

        elapsed_ms = int((time.time() - start) * 1000)

        if interrupted:
            self._archive(call_num, label, prompt, system_prompt, json_schema,
                          None, "interrupted", elapsed_ms, archive_path)
            logger.info("[%s] interrupted after %dms", label, elapsed_ms)
            raise Interrupted()

        thinking_text = "".join(thinking_parts)
        result_text = "".join(output_parts)
        tool_calls = _normalize_tool_calls(tool_call_acc)

        if soft_interrupted:
            finish_reason = "soft_interrupted"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        self._archive(call_num, label, prompt, system_prompt, json_schema,
                      {"result": result_text, "tool_calls": tool_calls},
                      None, elapsed_ms, archive_path,
                      thinking=thinking_text, result_text=result_text)
        logger.info("[%s] done %dms finish=%s tools=%d", label, elapsed_ms,
                    finish_reason, len(tool_calls) if tool_calls else 0)

        return {
            "result": result_text,
            "thinking": thinking_text,
            "cost": 0.0,
            "duration_ms": elapsed_ms,
            "raw": {"result": result_text, "tool_calls": tool_calls},
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    # ── Archiving ────────────────────────────────────────────────────

    def _archive(self, call_num, label, prompt, system_prompt, json_schema,
                 response, error, elapsed_ms, archive_path=None,
                 *, thinking="", result_text=""):
        archive(self.model, self.archive_dir, call_num, label, prompt,
                system_prompt, json_schema, response, error, elapsed_ms,
                archive_path, thinking=thinking, result_text=result_text)
