"""Provider adapters. Each adapter turns the *canonical* message list into a
provider request, makes exactly ONE API call (no hidden retries: max_retries=0),
and returns a normalized LLMResult.

Canonical messages:
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str|None, "tool_calls": [{"id","name","args"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str, "error": str|None}
Hidden reasoning is never collected or replayed.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResult:
    text: str | None
    tool_calls: list[dict[str, Any]]
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    model_returned: str | None = None


@dataclass
class ErrorInfo:
    error_class: str
    status: int | None
    retryable: bool
    retry_after: float | None
    message: str


def classify_exception(exc: BaseException) -> ErrorInfo:
    """Split infrastructure failures (retryable) from request-level failures."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    retry_after = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            ra = resp.headers.get("retry-after")
            retry_after = float(ra) if ra else None
        except Exception:
            retry_after = None
    if "Timeout" in name or isinstance(exc, TimeoutError):
        cls, retry = "timeout", True
    elif "Connection" in name:
        cls, retry = "connection", True
    elif status == 429:
        cls, retry = "rate_limit", True
    elif isinstance(status, int) and status >= 500:
        cls, retry = "server_error", True
    elif isinstance(status, int):
        cls, retry = f"http_{status}", False
    else:
        cls, retry = name, False
    return ErrorInfo(cls, status if isinstance(status, int) else None, retry, retry_after, str(exc)[:400])


def _tool_text(m: dict[str, Any]) -> str:
    # Same convention as AgentDojo: show the error text when the tool failed.
    return m["error"] if m.get("error") else m["content"]


def _parse_json_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # Keep malformed arguments visible instead of hiding them; the tool layer will error.
    return {"_unparsed_arguments": str(raw)}


class BaseAdapter:
    provider = "base"

    def __init__(self, model: str, settings: dict[str, Any] | None = None):
        self.model = model
        self.settings = dict(settings or {})

    def effective_settings(self) -> dict[str, Any]:
        return dict(self.settings)

    def runtime_identity(self) -> dict[str, Any]:
        """Model/runtime identity for reproducibility (resolved name, digest,
        server version). Empty for providers that do not expose it."""
        return {}

    def complete(self, messages: list[dict], tools: list[dict], timeout: float) -> LLMResult:  # pragma: no cover
        raise NotImplementedError

    def sdk_version(self) -> str | None:
        return None


class _ChatCompletionsAdapter(BaseAdapter):
    """OpenAI-style Chat Completions (used for OpenAI and Groq)."""

    api_route = "chat.completions"

    def _client(self):  # pragma: no cover - provider specific
        raise NotImplementedError

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def to_provider_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role in ("system", "user"):
                out.append({"role": role, "content": m["content"]})
            elif role == "assistant":
                d: dict[str, Any] = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls"):
                    d["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                        for c in m["tool_calls"]
                    ]
                out.append(d)
            elif role == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": _tool_text(m)})
        return out

    def _clean_text(self, text: str | None) -> str | None:
        # Default: no change. Ollama overrides this to drop <think> reasoning blocks
        # so hidden reasoning never enters the trajectory (data contract).
        return text

    def _extra_body(self) -> dict[str, Any]:
        return {}

    def complete(self, messages, tools, timeout):
        client = self._client()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=self.to_provider_messages(messages),
            tools=[{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["schema"]}} for t in tools],
            tool_choice="auto",  # never forced in the real experiment
            timeout=timeout,
        )
        kwargs.update(self._extra_kwargs())
        extra_body = self._extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        calls = [{"id": c.id, "name": c.function.name, "args": _parse_json_args(c.function.arguments)}
                 for c in (msg.tool_calls or [])]
        usage = getattr(resp, "usage", None)
        return LLMResult(
            text=self._clean_text(msg.content),
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
            request_id=getattr(resp, "_request_id", None) or getattr(resp, "id", None),
            model_returned=getattr(resp, "model", None),
        )


class OpenAIAdapter(_ChatCompletionsAdapter):
    provider = "openai"

    def __init__(self, model, settings=None):
        # Defaults: provider defaults for sampling/reasoning (temperature is not forced;
        # newer OpenAI models may reject it). Override via settings and it is logged.
        super().__init__(model, {"temperature": None, "reasoning_effort": None,
                                 "parallel_tool_calls": None, **(settings or {})})

    def _client(self):
        from openai import OpenAI
        return OpenAI(max_retries=0)

    def _extra_kwargs(self):
        kw = {}
        for k in ("temperature", "reasoning_effort", "parallel_tool_calls"):
            if self.settings.get(k) is not None:
                kw[k] = self.settings[k]
        return kw

    def sdk_version(self):
        import openai
        return f"openai {openai.__version__}"


class GroqAdapter(_ChatCompletionsAdapter):
    provider = "groq"

    def __init__(self, model, settings=None):
        # Qwen on Groq: no parallel tool calls; reasoning off, as validated by api_smoke.py.
        super().__init__(model, {"temperature": None, "reasoning_effort": "none",
                                 "parallel_tool_calls": False, **(settings or {})})

    def _client(self):
        # Groq's API is OpenAI-compatible: reuse the openai SDK, so no extra package
        # (and no change to a shared virtualenv) is needed.
        from openai import OpenAI
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

    def _extra_kwargs(self):
        kw = {}
        for k in ("temperature", "reasoning_effort", "parallel_tool_calls"):
            if self.settings.get(k) is not None:
                kw[k] = self.settings[k]
        return kw

    def sdk_version(self):
        import openai
        return f"openai {openai.__version__} (OpenAI-compatible endpoint api.groq.com)"


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Cache model identity (resolved name, digest, server version) per (base_url, model)
# so the native lookup happens once per model, not once per run.
_OLLAMA_IDENTITY_CACHE: dict[tuple, dict] = {}


def _ollama_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


class OllamaAdapter(_ChatCompletionsAdapter):
    """Local open-weight models served by Ollama's OpenAI-compatible endpoint.

    No API key and no cost: the model runs on this machine. The context length is
    set on the Ollama server via the OLLAMA_CONTEXT_LENGTH environment variable
    (verify with `ollama ps`); it is recorded here for reproducibility. Thinking
    models (Gemma, Qwen) may wrap reasoning in <think>...</think>; that is stripped
    from the logged text so no hidden reasoning enters the trajectory.
    """

    provider = "ollama"

    def __init__(self, model, settings=None, base_url=None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ctx = os.getenv("OLLAMA_CONTEXT_LENGTH")
        # Full Experiment Generation Policy v1.0 (frozen defaults; overridable via settings):
        # temperature=0.0 and seed=42 are set explicitly for reproducibility. No top_p/top_k/
        # num_predict/max_tokens and no think/thinking/reasoning_effort are ever sent.
        # context_length_env documents what the Ollama server was configured with.
        super().__init__(model, {"temperature": 0.0, "seed": 42, "parallel_tool_calls": False,
                                 "context_length_env": int(ctx) if ctx and ctx.isdigit() else None,
                                 "base_url": self.base_url, **(settings or {})})

    def _client(self):
        # api_key is required by the SDK but ignored by Ollama.
        from openai import OpenAI
        return OpenAI(base_url=self.base_url, api_key=os.getenv("OLLAMA_API_KEY", "ollama"), max_retries=0)

    def _extra_kwargs(self):
        kw = {}
        for k in ("temperature", "seed", "parallel_tool_calls"):
            if self.settings.get(k) is not None:
                kw[k] = self.settings[k]
        return kw

    def _clean_text(self, text):
        if not text:
            return text
        return _THINK_BLOCK.sub("", text).strip() or None

    def effective_settings(self):
        # Record exactly what the generation policy does: the values sent plus
        # explicit markers for the parameters intentionally NOT sent.
        s = dict(self.settings)
        s["thinking_override"] = "omitted"   # no think/thinking/reasoning_effort sent
        for k in ("top_p", "top_k", "num_predict", "max_tokens"):
            s.setdefault(k, "inherited_default")
        return s

    def runtime_identity(self):
        key = (self.base_url, self.model)
        if key in _OLLAMA_IDENTITY_CACHE:
            return _OLLAMA_IDENTITY_CACHE[key]
        host = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        ident = {"requested_model": self.model, "resolved_model": None,
                 "model_digest": None, "ollama_version": None}
        try:
            tags = _ollama_get(host + "/api/tags").get("models", [])
            match = None
            for m in tags:
                if m.get("name") == self.model:
                    match = m
                    break
            if match is None:
                base = self.model.split(":")[0]
                cands = [m for m in tags if m.get("name", "").split(":")[0] == base]
                if len(cands) == 1:
                    match = cands[0]
            if match is not None:
                ident["resolved_model"] = match.get("name")
                ident["model_digest"] = match.get("digest")
            ident["ollama_version"] = _ollama_get(host + "/api/version").get("version")
        except Exception as exc:  # noqa: BLE001 - identity is best-effort metadata
            ident["identity_error"] = f"{type(exc).__name__}: {exc}"
        _OLLAMA_IDENTITY_CACHE[key] = ident
        return ident

    def sdk_version(self):
        import openai
        return f"openai {openai.__version__} (OpenAI-compatible endpoint {self.base_url})"


class AnthropicAdapter(BaseAdapter):
    provider = "anthropic"
    api_route = "messages"

    def __init__(self, model, settings=None):
        # Sonnet 5 rejects non-default temperature/top_p/top_k -> never sent.
        # Thinking is disabled so no signed thinking blocks need replaying and no
        # hidden reasoning is collected. Any change must be approved by P1 + P2.
        super().__init__(model, {"max_tokens": 4096, "thinking": "disabled", **(settings or {})})

    def _client(self):
        from anthropic import Anthropic
        return Anthropic(max_retries=0)

    @staticmethod
    def to_provider_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
        system = None
        out: list[dict] = []
        pending: list[dict] = []

        def flush():
            nonlocal pending
            if pending:
                out.append({"role": "user", "content": pending})
                pending = []

        for m in messages:
            role = m["role"]
            if role == "system":
                system = m["content"]
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": _tool_text(m) or "(empty)"}
                if m.get("error"):
                    block["is_error"] = True
                pending.append(block)  # consecutive tool results share one user turn
            elif role == "user":
                flush()
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                flush()
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]})
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
        flush()
        return system, out

    def complete(self, messages, tools, timeout):
        client = self._client()
        system, msgs = self.to_provider_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=self.settings["max_tokens"], messages=msgs,
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["schema"]} for t in tools],
            tool_choice={"type": "auto"},
            thinking={"type": "disabled"},
            timeout=timeout,
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        calls = [{"id": b.id, "name": b.name, "args": dict(b.input) if isinstance(b.input, dict) else {}}
                 for b in resp.content if b.type == "tool_use"]
        usage = getattr(resp, "usage", None)
        return LLMResult(
            text=text, tool_calls=calls,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_reason=resp.stop_reason,
            request_id=getattr(resp, "_request_id", None) or getattr(resp, "id", None),
            model_returned=resp.model,
        )

    def sdk_version(self):
        import anthropic
        return f"anthropic {anthropic.__version__}"


PROVIDER_ADAPTERS = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter,
                     "groq": GroqAdapter, "ollama": OllamaAdapter}


class MockAdapter(BaseAdapter):
    """OFFLINE test double that stands in for an LLM (no network, no cost).

    It replays AgentDojo ground-truth calls, so it must only be used to test the
    pipeline; runs made with it are labelled provider="mock". mode="obey" also
    replays the injection task calls after the payload has been delivered.
    """

    provider = "mock"

    def __init__(self, model="mock", settings=None, mode="ignore"):
        super().__init__(model, settings or {})
        self.mode = mode
        self.queue: list[dict] = []
        self.final_text = "Done."
        self._inj_calls: list[dict] = []
        self._obeyed = False
        self._n = 0

    def prepare(self, user_task, inj_task, pre_env):
        self.queue = [{"name": c.function, "args": dict(c.args)} for c in user_task.ground_truth(pre_env.model_copy(deep=True))]
        self._inj_calls = ([{"name": c.function, "args": dict(c.args)}
                            for c in inj_task.ground_truth(pre_env.model_copy(deep=True))] if inj_task else [])
        self.final_text = getattr(user_task, "GROUND_TRUTH_OUTPUT", "") or "Done."
        self._obeyed = False
        self._n = 0

    def complete(self, messages, tools, timeout):
        last = messages[-1]
        if (self.mode == "obey" and not self._obeyed and last["role"] == "tool"
                and "ignoreallpreviousinstructions" in "".join((last["content"] or "").lower().split())):
            self._obeyed = True
            self.queue = self.queue + self._inj_calls  # injected calls happen last, so they are not overwritten
        if self.queue:
            c = self.queue.pop(0)
            self._n += 1
            return LLMResult(None, [{"id": f"mock_{self._n}", "name": c["name"], "args": c["args"]}],
                             10, 5, "tool_calls", f"mock-req-{self._n}", "mock")
        return LLMResult(self.final_text, [], 10, 5, "stop", "mock-req-final", "mock")


def get_adapter(provider: str, model: str, settings=None) -> BaseAdapter:
    return PROVIDER_ADAPTERS[provider](model, settings)


ENV_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}


def has_key(provider: str) -> bool:
    # Ollama runs locally and needs no API key.
    if provider == "ollama":
        return True
    return bool(os.getenv(ENV_KEYS[provider]))
