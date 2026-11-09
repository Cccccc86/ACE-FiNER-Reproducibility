"""Meter DeepSeek chat attempts for the FiNER experiment.

Use one ledger path for the entire experiment, including pilot and retries.
Amounts are in CNY at the documented peak Flash rates, with an uncached input
reservation before each request. Use this ledger with a CNY-billed account.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
NANO_PER_CNY = 1_000_000_000
MISS_PRICE = 2_000  # ¥2 per million tokens, in nanoyuan per token
HIT_PRICE = 40      # ¥0.04 per million tokens
OUTPUT_PRICE = 8_000  # ¥8 per million tokens
PRICE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
_META_FIELDS = {
    "phase", "role", "call_id", "trial", "arm", "attempt", "seed", "row_id",
    "sample_id", "split", "condition", "run_id",
}
_SAFE_META = re.compile(r"[A-Za-z0-9_.:/-]{1,120}\Z")
_REQUEST_CONTEXT: ContextVar[dict] = ContextVar("finer_request_context", default={})


class BudgetExceeded(RuntimeError):
    """An attempt would cross the input, attempt-count or spend limit."""


class LedgerError(RuntimeError):
    """The ledger cannot safely authorize another request."""


class ProviderRequestError(RuntimeError):
    """A provider error with only sanitized, request-independent details."""

    def __init__(self, kind: str, status_code: int | None = None, *, retryable: bool = False):
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable
        status = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"DeepSeek {kind}{status}")


@contextmanager
def request_scope(**metadata):
    """Add experiment identifiers to ledger events in the current thread.

    Enter this scope inside each worker. ContextVars do not automatically
    propagate into a ThreadPoolExecutor's worker threads.
    """
    clean = {}
    for key, value in metadata.items():
        if key not in _META_FIELDS:
            raise ValueError("Unsupported request metadata field")
        if value is None:
            continue
        if type(value) is int or (
            isinstance(value, str) and _SAFE_META.fullmatch(value)
            and not value.lower().startswith(("sk-", "bearer"))
        ):
            clean[key] = value
        else:
            raise ValueError("Request metadata must be a short, non-secret identifier")
    token = _REQUEST_CONTEXT.set({**_REQUEST_CONTEXT.get(), **clean})
    try:
        yield
    finally:
        _REQUEST_CONTEXT.reset(token)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_count(value) -> bool:
    return type(value) is int and value >= 0


def _usage(response) -> dict | None:
    usage = _field(response, "usage")
    prompt = _field(usage, "prompt_tokens")
    completion = _field(usage, "completion_tokens")
    if not (_token_count(prompt) and _token_count(completion)):
        return None
    hit = _field(usage, "prompt_cache_hit_tokens")
    if hit is None:
        hit = _field(_field(usage, "prompt_tokens_details"), "cached_tokens")
    miss = _field(usage, "prompt_cache_miss_tokens")
    cache_known = _token_count(hit) and hit <= prompt and (
        miss is None or (_token_count(miss) and hit + miss == prompt)
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_cache_hit_tokens": hit if cache_known else None,
        "prompt_cache_miss_tokens": prompt - hit if cache_known else None,
    }


def _settled_cost(usage: dict) -> int:
    hit = usage["prompt_cache_hit_tokens"] or 0
    return (
        (usage["prompt_tokens"] - hit) * MISS_PRICE
        + hit * HIT_PRICE
        + usage["completion_tokens"] * OUTPUT_PRICE
    )


class _Ledger:
    """Durable, append-only attempt records with a per-path OS lock."""

    def __init__(self, path: Path, budget: int, max_attempts: int):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.thread_lock = threading.Lock()
        self.config = {
            "event": "config", "version": 1, "currency": "CNY",
            "model": MODEL, "base_url": BASE_URL,
            "budget_nano_cny": budget, "max_attempts": max_attempts,
            "peak_miss_nano_per_token": MISS_PRICE,
            "peak_hit_nano_per_token": HIT_PRICE,
            "peak_output_nano_per_token": OUTPUT_PRICE, "price_source": PRICE_SOURCE,
        }
        with self._locked():
            if not self.path.exists() or self.path.stat().st_size == 0:
                self._append(self.config)
            self._read()

    @contextmanager
    def _locked(self):
        with self.thread_lock, self.lock_path.open("a+b") as lock:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as ledger:
            ledger.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())

    def _read(self) -> tuple[dict, dict, set[str]]:
        reserves, settled, unknown = {}, {}, set()
        try:
            with self.path.open("r", encoding="utf-8") as ledger:
                header = ledger.readline()
                if not header.endswith("\n") or json.loads(header) != self.config:
                    raise LedgerError("Ledger policy differs from the configured experiment")
                for line in ledger:
                    if not line.endswith("\n"):
                        raise LedgerError("Ledger has an incomplete event and is closed to new calls")
                    record = json.loads(line)
                    attempt_id = record["attempt_id"]
                    event = record["event"]
                    if event == "reserve" and attempt_id not in reserves:
                        if not _token_count(record["reserved_nano_cny"]):
                            raise LedgerError("Invalid ledger reservation")
                        reserves[attempt_id] = record
                    elif event == "settle" and attempt_id in reserves and attempt_id not in settled and attempt_id not in unknown:
                        if not _token_count(record["cost_peak_nano_cny"]):
                            raise LedgerError("Invalid ledger cost")
                        settled[attempt_id] = record
                    elif event == "unknown" and attempt_id in reserves and attempt_id not in settled and attempt_id not in unknown:
                        unknown.add(attempt_id)
                    else:
                        raise LedgerError("Invalid ledger event sequence")
        except (OSError, KeyError, TypeError, ValueError):
            raise LedgerError("Unable to read the experiment ledger safely") from None
        return reserves, settled, unknown

    def reserve(self, input_bound: int, output_cap: int, json_mode: bool) -> str:
        with self._locked():
            reserves, settled, _ = self._read()
            if any(record["limit_violation"] for record in settled.values()):
                raise LedgerError("Provider usage crossed a reserved token bound")
            if len(reserves) >= self.config["max_attempts"]:
                raise BudgetExceeded("Experiment attempt limit reached")
            cost = input_bound * MISS_PRICE + output_cap * OUTPUT_PRICE
            committed = sum(r["cost_peak_nano_cny"] for r in settled.values()) + sum(
                r["reserved_nano_cny"] for key, r in reserves.items() if key not in settled
            )
            if committed + cost > self.config["budget_nano_cny"]:
                raise BudgetExceeded("Experiment budget cannot cover another request")
            attempt_id = uuid4().hex
            self._append({
                "event": "reserve", "attempt_id": attempt_id, "utc": _utc_now(),
                **_REQUEST_CONTEXT.get(), "model_requested": MODEL,
                "input_token_upper_bound": input_bound, "max_tokens": output_cap,
                "temperature": 0, "thinking": "disabled", "json_mode": json_mode,
                "reserved_nano_cny": cost,
            })
            return attempt_id

    def record(self, event: dict) -> None:
        with self._locked():
            self._append({**event, "utc": _utc_now(), **_REQUEST_CONTEXT.get()})

    def summary(self) -> dict:
        with self._locked():
            reserves, settled, unknown = self._read()
        spent = sum(r["cost_peak_nano_cny"] for r in settled.values())
        pending = sum(r["reserved_nano_cny"] for key, r in reserves.items() if key not in settled)
        return {
            "ledger_path": str(self.path),
            "budget_cny": self.config["budget_nano_cny"] / NANO_PER_CNY,
            "max_attempts": self.config["max_attempts"],
            "attempts": len(reserves), "settled_attempts": len(settled),
            "unresolved_attempts": len(reserves) - len(settled), "unknown_attempts": len(unknown),
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in settled.values()),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in settled.values()),
            "prompt_cache_hit_tokens_observed": sum(r["usage"]["prompt_cache_hit_tokens"] or 0 for r in settled.values()),
            "attempts_without_cache_breakdown": sum(r["usage"]["prompt_cache_hit_tokens"] is None for r in settled.values()),
            "settled_peak_cost_cny": spent / NANO_PER_CNY,
            "unresolved_reserved_cny": pending / NANO_PER_CNY,
            "committed_cny": (spent + pending) / NANO_PER_CNY,
            "remaining_cny": (self.config["budget_nano_cny"] - spent - pending) / NANO_PER_CNY,
            "usage_bound_violations": sum(r["limit_violation"] for r in settled.values()),
        }


class BudgetedDeepSeekClient:
    """Small text-only OpenAI-compatible client for the declared experiment.

    ``client`` can supply an offline fake with ``chat.completions.create``.
    The actual SDK client created here uses a finite timeout and zero SDK
    retries. Each create call reserves and sends at most one request.
    """

    def __init__(
        self, ledger_path: str | Path, *, api_key: str | None = None,
        budget_cny: int | float | str = 50, max_attempts: int = 2000,
        max_input_tokens: int = 16000, max_output_tokens: int = 1024,
        timeout_s: float = 60, client=None,
    ):
        try:
            budget = Decimal(str(budget_cny))
            if not budget.is_finite() or not 0 <= budget <= 50:
                raise ValueError("Budget must be between ¥0 and ¥50")
        except InvalidOperation:
            raise ValueError("Invalid CNY budget") from None
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if type(max_input_tokens) is not int or type(max_output_tokens) is not int or not (
            1 <= max_output_tokens <= 393216 and 1 <= max_input_tokens <= 1_000_000 - max_output_tokens
        ):
            raise ValueError("Invalid DeepSeek input or output token cap")
        if not isinstance(timeout_s, (int, float)) or not 0 < timeout_s < float("inf"):
            raise ValueError("timeout_s must be finite and positive")
        if client is None:
            from openai import OpenAI

            key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise ValueError("DEEPSEEK_API_KEY is required for live inference")
            try:
                client = OpenAI(api_key=key, base_url=BASE_URL, timeout=timeout_s, max_retries=0)
            except Exception:
                raise ProviderRequestError("client initialization failed") from None
        self._client = client
        self._ledger = _Ledger(Path(ledger_path), int(budget * NANO_PER_CNY), max_attempts)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def summary(self) -> dict:
        """Return peak-estimated settled spend plus unresolved reservations."""
        return self._ledger.summary()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def _create(
        self, *, messages: list[dict], model: str = MODEL, temperature: float = 0,
        max_tokens: int | None = None, max_completion_tokens: int | None = None,
        response_format: dict | None = None, extra_body: dict | None = None,
        stream: bool = False, n: int = 1,
    ):
        if model != MODEL or temperature != 0 or stream or n != 1:
            raise ValueError("FiNER requires one non-streaming DeepSeek Flash completion at temperature zero")
        if extra_body not in (None, {}, {"thinking": {"type": "disabled"}}):
            raise ValueError("FiNER uses text completions with thinking disabled")
        if response_format not in (None, {"type": "text"}, {"type": "json_object"}):
            raise ValueError("FiNER supports text and JSON object responses")
        if max_tokens is not None and max_completion_tokens is not None:
            raise ValueError("Supply only one output token limit")
        requested = max_tokens if max_tokens is not None else max_completion_tokens
        if requested is not None and (type(requested) is not int or requested < 1):
            raise ValueError("The requested output limit must be positive")
        output_cap = min(self.max_output_tokens, requested) if requested else self.max_output_tokens
        if not isinstance(messages, list) or not messages:
            raise ValueError("A nonempty message list is required")
        if any(
            not isinstance(m, dict) or set(m) != {"role", "content"}
            or m["role"] not in ("user", "system", "assistant")
            or not isinstance(m["content"], str)
            for m in messages
        ):
            raise ValueError("FiNER accepts text-only chat messages")
        messages = [message.copy() for message in messages]
        # Byte-based upper estimate for the text plus generous chat framing.
        # Observe provider usage to check this bound for the deployed model.
        input_bound = 512 + 32 * len(messages) + sum(len(m["content"].encode("utf-8")) for m in messages)
        if input_bound > self.max_input_tokens:
            raise BudgetExceeded(f"Request input bound {input_bound} exceeds cap {self.max_input_tokens}")
        attempt_id = self._ledger.reserve(input_bound, output_cap, response_format == {"type": "json_object"})
        started = time.perf_counter()
        params = {
            "model": MODEL, "messages": messages, "temperature": 0,
            "max_tokens": output_cap, "extra_body": {"thinking": {"type": "disabled"}},
        }
        if response_format is not None:
            params["response_format"] = response_format
        try:
            response = self._client.chat.completions.create(**params)
        except Exception as exc:
            # No raw error message, response body, request object or headers.
            status = getattr(exc, "status_code", None)
            status = status if type(status) is int and 100 <= status <= 599 else None
            transient = isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
                "APITimeoutError", "APIConnectionError",
            }
            retryable = transient or status == 429 or (status is not None and status >= 500)
            kind = "transport failure" if transient else "request failure"
            self._ledger.record({
                "event": "unknown", "attempt_id": attempt_id, "kind": kind,
                "status_code": status, "elapsed_seconds": time.perf_counter() - started,
            })
            raise ProviderRequestError(kind, status, retryable=retryable) from None

        usage = _usage(response)
        elapsed = time.perf_counter() - started
        if usage is None:
            self._ledger.record({
                "event": "unknown", "attempt_id": attempt_id,
                "kind": "missing token usage", "elapsed_seconds": elapsed,
            })
            raise ProviderRequestError("response lacked usable token usage")
        choices = _field(response, "choices") or []
        finish_reason = _field(choices[0], "finish_reason") if choices else None
        if finish_reason not in ("stop", "length", "content_filter", "tool_calls", "insufficient_system_resource", "aborted"):
            finish_reason = None
        response_model = _field(response, "model")
        if not isinstance(response_model, str) or not re.fullmatch(r"deepseek[A-Za-z0-9_.-]{0,80}", response_model, re.I):
            response_model = None
        fingerprint = _field(response, "system_fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"(?:fp_)?[a-fA-F0-9]{8,64}", fingerprint):
            fingerprint = None
        violation = usage["prompt_tokens"] > input_bound or usage["completion_tokens"] > output_cap
        self._ledger.record({
            "event": "settle", "attempt_id": attempt_id, "usage": usage,
            "cost_peak_nano_cny": _settled_cost(usage), "elapsed_seconds": elapsed,
            "finish_reason": finish_reason, "model_reported": response_model,
            "system_fingerprint": fingerprint, "limit_violation": violation,
        })
        if violation:
            raise LedgerError("Provider usage crossed a reserved token bound")
        return response


def ace_timed_llm_call(
    client, api_provider, model, prompt, role, call_id, max_tokens=4096, log_dir=None,
    sleep_seconds=15, retries_on_timeout=1000, attempt=1, use_json_mode=False,
):
    """Replacement for the upstream role call function, with one counted retry.

    Keep the upstream signature for Generator, Reflector and Curator. The
    ledger captures request metadata and timings without recording prompts,
    responses or raw provider errors. Nonblank content is returned unchanged
    for the original role parsers.
    """
    started = time.perf_counter()
    params = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if use_json_mode:
        params["response_format"] = {"type": "json_object"}
    for attempt_number in (1, 2):
        try:
            with request_scope(role=role, call_id=call_id, attempt=attempt_number):
                response = client.chat.completions.create(**params)
            break
        except ProviderRequestError as exc:
            if attempt_number == 2 or not exc.retryable:
                raise
            time.sleep(1)
    choices = _field(response, "choices") or []
    content = _field(_field(choices[0], "message"), "content") if choices else None
    finish_reason = _field(choices[0], "finish_reason") if choices else None
    if not isinstance(content, str) or not content.strip():
        raise ProviderRequestError("response was blank")
    if finish_reason not in (None, "stop"):
        raise ProviderRequestError("response did not finish normally")
    elapsed = time.perf_counter() - started
    usage = _usage(response)
    if usage is None:
        raise ProviderRequestError("response lacked usable token usage")
    return content, {
        "role": role, "call_id": call_id, "model": model,
        "prompt_time": 0.0, "response_time": elapsed, "total_time": elapsed,
        "call_time": elapsed, "prompt_length": len(prompt), "response_length": len(content),
        "prompt_num_tokens": usage["prompt_tokens"],
        "response_num_tokens": usage["completion_tokens"], "attempt": attempt_number,
    }
