"""Model provider adapters: request building and strict response parsing.

Both providers receive the same snapshot string and answer two named
probability questions in one request. Requests are sent with the standard
library (``urllib``), which never retries on its own, so every outbound
request is one recorded attempt. No provider SDK is used.

The endpoint shapes follow the providers' public documentation as read in
November 2026. **Re-check them against the current documentation before any
real run**; see ``docs/evaluation-harness.md``.

* Jev (TypeSafe System One): ``POST https://api.typesafe.ai/v1/systemone`` with
  ``model``, ``state`` and a ``questions`` object keyed by name, each
  ``{"type": "noul", "instructions": ...}``. The answer is
  ``answers.<name>.noul`` in [0, 1].
* OpenAI Decisions: ``POST https://api.openai.com/v1/decisions`` with
  ``model``, ``input`` and a ``questions`` array of
  ``{"type": "predicate", "name": ..., "instructions": ...}``. Answers are
  matched by ``name``; ``{"type": "refusal"}`` is a per-question refusal.
"""

import json
import math
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

FLAGS = ("uses_voicebank", "is_vocal_work")

# Attempt statuses.
OK = "ok"
RETRYABLE = {"rate_limited", "timeout", "transport_error", "server_error"}
STOP_PROVIDER = {"auth_error", "billing_error", "config_error"}
TERMINAL = {"invalid_response", "refusal", "input_too_large", "http_error"}
STATUSES = {OK} | RETRYABLE | STOP_PROVIDER | TERMINAL


class DuplicateKey(ValueError):
    pass


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey(key)
        result[key] = value
    return result


def strict_json(body: bytes):
    """Parse JSON, rejecting duplicate object keys and non-finite numbers."""
    def bad_constant(name):
        raise ValueError(f"non-finite number {name}")
    return json.loads(body.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=bad_constant)


def valid_probability(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0.0 <= value <= 1.0


@dataclass
class Parsed:
    status: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    returned_model: Optional[str] = None
    usage: Optional[dict] = None
    input_tokens: Optional[int] = None
    error_class: Optional[str] = None
    error_detail: Optional[str] = None


@dataclass
class HttpResult:
    """Outcome of one outbound request. ``status`` is None for transport errors."""
    status: Optional[int]
    body: Optional[bytes] = None
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None  # 'timeout' or 'transport'
    error_detail: Optional[str] = None


Transport = Callable[[str, Dict[str, str], bytes, float], HttpResult]

# Response headers worth keeping; never includes anything credential-like.
KEPT_HEADERS = ("content-type", "date", "retry-after", "x-request-id", "request-id", "openai-model", "openai-processing-ms")


def http_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> HttpResult:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResult(response.status, response.read(), _kept(response.headers))
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read() or b"", _kept(exc.headers))
    except (socket.timeout, TimeoutError) as exc:
        return HttpResult(None, error="timeout", error_detail=type(exc).__name__)
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return HttpResult(None, error="timeout", error_detail=type(reason).__name__)
        return HttpResult(None, error="transport", error_detail=f"{type(reason).__name__}: {reason}")
    except OSError as exc:
        return HttpResult(None, error="transport", error_detail=f"{type(exc).__name__}: {exc}")


def _kept(headers) -> Dict[str, str]:
    if headers is None:
        return {}
    return {k.lower(): v for k, v in headers.items() if k.lower() in KEPT_HEADERS}


def classify_http(result: HttpResult) -> Optional[Parsed]:
    """Map non-200 outcomes to statuses. Returns None for HTTP 200."""
    if result.status is None:
        if result.error == "timeout":
            return Parsed("timeout", error_class="timeout", error_detail=result.error_detail)
        return Parsed("transport_error", error_class="transport", error_detail=result.error_detail)
    code = result.status
    if code == 200:
        return None
    text = (result.body or b"")[:2000].decode("utf-8", "replace")
    if code in (401, 403):
        return Parsed("auth_error", error_class=f"http_{code}", error_detail=text)
    if code == 402:
        return Parsed("billing_error", error_class="http_402", error_detail=text)
    if code == 413 or (code == 400 and ("max_tokens_exceeded" in text or "context_length" in text)):
        return Parsed("input_too_large", error_class=f"http_{code}", error_detail=text)
    if code in (400, 404, 415, 422):
        return Parsed("config_error", error_class=f"http_{code}", error_detail=text)
    if code == 429:
        return Parsed("rate_limited", error_class="http_429", error_detail=text)
    if code >= 500:
        return Parsed("server_error", error_class=f"http_{code}", error_detail=text)
    return Parsed("http_error", error_class=f"http_{code}", error_detail=text)


def _usage(data: dict) -> Tuple[Optional[dict], Optional[int], Optional[str]]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None, "missing usage"
    tokens = usage.get("input_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        return usage, None, "invalid usage.input_tokens"
    return usage, tokens, None


class Provider:
    name = ""
    default_model = ""
    url = ""
    credential_env = ""
    # USD per million billable input tokens (documented base price, uncached).
    default_input_price = 0.0

    def build_request(self, model: str, input_text: str, questions: Dict[str, str]) -> dict:
        raise NotImplementedError

    def parse_answers(self, data: dict, names: List[str]) -> Parsed:
        raise NotImplementedError

    def headers(self, api_key: Optional[str]) -> Dict[str, str]:
        """Headers to send. Only the non-secret ones are ever recorded."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def parse(self, result: HttpResult, names: List[str]) -> Parsed:
        failure = classify_http(result)
        if failure is not None:
            return failure
        try:
            data = strict_json(result.body or b"")
        except DuplicateKey as exc:
            return Parsed("invalid_response", error_class="duplicate_key", error_detail=f"duplicate JSON key {exc}")
        except (ValueError, UnicodeDecodeError) as exc:
            return Parsed("invalid_response", error_class="parse_error", error_detail=str(exc)[:500])
        if not isinstance(data, dict):
            return Parsed("invalid_response", error_class="not_object")
        parsed = self.parse_answers(data, names)
        model = data.get("model")
        parsed.returned_model = model if isinstance(model, str) else None
        usage, tokens, usage_error = _usage(data)
        parsed.usage, parsed.input_tokens = usage, tokens
        if parsed.status == OK and usage_error:
            # Probabilities are usable, but cost must then fall back to the estimate.
            parsed.error_detail = usage_error
        return parsed


class Jev(Provider):
    name = "jev"
    default_model = "jev-1.14.0"
    url = "https://api.typesafe.ai/v1/systemone"
    credential_env = "TYPESAFE_API_KEY"
    default_input_price = 0.042

    def build_request(self, model, input_text, questions):
        return {
            "model": model,
            "state": input_text,
            "questions": {name: {"type": "noul", "instructions": text} for name, text in questions.items()},
        }

    def parse_answers(self, data, names):
        answers = data.get("answers")
        if not isinstance(answers, dict):
            return Parsed("invalid_response", error_class="missing_answers")
        probabilities = {}
        for name in names:
            answer = answers.get(name)
            if not isinstance(answer, dict):
                return Parsed("invalid_response", error_class="missing_answer", error_detail=name)
            if answer.get("type") != "noul":
                return Parsed("invalid_response", error_class="wrong_answer_type", error_detail=f"{name}: {answer.get('type')!r}")
            if not valid_probability(answer.get("noul")):
                return Parsed("invalid_response", error_class="invalid_probability", error_detail=f"{name}: {answer.get('noul')!r}")
            probabilities[name] = float(answer["noul"])
        extra = sorted(set(answers) - set(names))
        if extra:
            return Parsed("invalid_response", error_class="unexpected_answer", error_detail=", ".join(extra))
        return Parsed(OK, probabilities)


class OpenAIDecisions(Provider):
    name = "openai"
    default_model = "gpt-6-luna"
    url = "https://api.openai.com/v1/decisions"
    credential_env = "OPENAI_API_KEY"
    default_input_price = 0.10

    def build_request(self, model, input_text, questions):
        return {
            "model": model,
            "input": input_text,
            "questions": [{"type": "predicate", "name": name, "instructions": text} for name, text in questions.items()],
        }

    def parse_answers(self, data, names):
        answers = data.get("answers")
        if not isinstance(answers, list):
            return Parsed("invalid_response", error_class="missing_answers")
        by_name: Dict[str, dict] = {}
        for answer in answers:
            if not isinstance(answer, dict) or not isinstance(answer.get("name"), str):
                return Parsed("invalid_response", error_class="unnamed_answer")
            if answer["name"] in by_name:
                return Parsed("invalid_response", error_class="duplicate_answer", error_detail=answer["name"])
            by_name[answer["name"]] = answer
        refused = [n for n in names if (by_name.get(n) or {}).get("type") == "refusal"]
        if refused:
            return Parsed("refusal", error_class="refusal", error_detail=", ".join(refused))
        probabilities = {}
        for name in names:
            answer = by_name.get(name)
            if answer is None:
                return Parsed("invalid_response", error_class="missing_answer", error_detail=name)
            if answer.get("type") != "predicate":
                return Parsed("invalid_response", error_class="wrong_answer_type", error_detail=f"{name}: {answer.get('type')!r}")
            if not valid_probability(answer.get("probability")):
                return Parsed("invalid_response", error_class="invalid_probability", error_detail=f"{name}: {answer.get('probability')!r}")
            probabilities[name] = float(answer["probability"])
        extra = sorted(set(by_name) - set(names))
        if extra:
            return Parsed("invalid_response", error_class="unexpected_answer", error_detail=", ".join(extra))
        return Parsed(OK, probabilities)


PROVIDERS = {p.name: p for p in (Jev(), OpenAIDecisions())}


def synthetic_transport(provider_name: str) -> Transport:
    """Offline stand-in that answers with deterministic fake probabilities.

    The values are derived from a hash of the request and mean nothing; they
    only let the pipeline run end to end without network access.
    """
    import hashlib

    def transport(url, headers, body, timeout):
        request = json.loads(body)
        digest = hashlib.sha256(body).digest()
        names = list(request["questions"]) if isinstance(request["questions"], dict) else [q["name"] for q in request["questions"]]
        probs = {n: round(digest[i] / 255.0, 4) for i, n in enumerate(names)}
        tokens = len(body) // 3
        model = request["model"] + "-synthetic"
        if provider_name == "jev":
            data = {"model": model, "answers": {n: {"type": "noul", "noul": p} for n, p in probs.items()}, "usage": {"input_tokens": tokens, "output_tokens": 0}}
        else:
            data = {"model": model, "answers": [{"type": "predicate", "name": n, "probability": p} for n, p in probs.items()],
                    "usage": {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens}}
        return HttpResult(200, json.dumps(data).encode("utf-8"), {"content-type": "application/json"})

    return transport
