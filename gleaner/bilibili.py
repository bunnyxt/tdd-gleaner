"""Bilibili public endpoints used for input snapshots, and response classification.

Each endpoint response is classified into exactly one status:

``ok``
    The response passed validation; ``extracted`` holds the fields needed for
    the model input.
``retryable_error``
    Transport failures, non-200 HTTP statuses (including 403 and 412), rate
    limiting, and business codes that are not known to be terminal. These say
    nothing about the video itself.
``unavailable``
    A business code listed in ``UNAVAILABLE_CODES`` for that endpoint. Only
    codes known to mean the video cannot be viewed belong there.
``invalid_response``
    The body is not JSON, fields are missing or have the wrong type, or the
    returned aid does not match the requested aid.

None of these failure statuses is a content label.
"""

import json
import random
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

ENDPOINTS = {
    "view": "https://api.bilibili.com/x/web-interface/view",
    "tags": "https://api.bilibili.com/x/tag/archive/tags",
}

# Business codes treated as terminal "video unavailable" results. Keep this
# list conservative: an unknown code is retryable, never unavailable.
UNAVAILABLE_CODES = {
    "view": {-404, 62002},
    "tags": set(),
}



class Response(NamedTuple):
    http_status: Optional[int]
    body: Optional[bytes]
    error: Optional[str] = None  # transport error description, if any
    via: str = "direct"  # "direct", a worker id, or "fixture"
    user_agent: Optional[str] = None


# A transport takes (endpoint, aid) and returns the response.
Transport = Callable[[str, int], Response]


class Classified(NamedTuple):
    status: str
    error_class: Optional[str]
    error_detail: Optional[str]
    api_code: Optional[int]
    extracted: Optional[Dict[str, Any]]


def endpoint_url(endpoint: str, aid: int) -> str:
    return f"{ENDPOINTS[endpoint]}?aid={aid}"


def _invalid(error_class: str, detail: str, api_code: Optional[int] = None) -> Classified:
    return Classified("invalid_response", error_class, detail, api_code, None)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def classify(endpoint: str, aid: int, response: Response) -> Classified:
    """Classify one endpoint response for the requested aid."""
    if response.error is not None or response.http_status is None:
        return Classified("retryable_error", "transport", response.error or "no response", None, None)
    if response.http_status != 200:
        return Classified("retryable_error", "http_status", f"HTTP {response.http_status}", None, None)
    body = response.body or b""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _invalid("not_json", str(exc))
    if not isinstance(payload, dict) or not _is_int(payload.get("code")):
        return _invalid("missing_field", "response has no integer 'code'")
    code = payload["code"]
    if code != 0:
        if code in UNAVAILABLE_CODES[endpoint]:
            return Classified("unavailable", "api_code", str(payload.get("message", "")), code, None)
        return Classified("retryable_error", "api_code", str(payload.get("message", "")), code, None)
    data = payload.get("data")
    if endpoint == "view":
        return _extract_view(aid, data)
    return _extract_tags(data)


def _extract_view(aid: int, data: Any) -> Classified:
    if not isinstance(data, dict):
        return _invalid("type_error", "data is not an object", 0)
    for key in ("aid", "title", "desc"):
        if key not in data:
            return _invalid("missing_field", f"data.{key} is missing", 0)
    if not _is_int(data["aid"]):
        return _invalid("type_error", "data.aid is not an integer", 0)
    if data["aid"] != aid:
        return _invalid("aid_mismatch", f"requested {aid}, got {data['aid']}", 0)
    for key in ("title", "desc"):
        if not isinstance(data[key], str):
            return _invalid("type_error", f"data.{key} is not a string", 0)
    return Classified("ok", None, None, 0, {"aid": aid, "title": data["title"], "desc": data["desc"]})


def _extract_tags(data: Any) -> Classified:
    if data is None:
        return _invalid("missing_field", "data is missing or null", 0)
    if not isinstance(data, list):
        return _invalid("type_error", "data is not an array", 0)
    names = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "tag_name" not in item:
            return _invalid("missing_field", f"data[{index}].tag_name is missing", 0)
        if not isinstance(item["tag_name"], str):
            return _invalid("type_error", f"data[{index}].tag_name is not a string", 0)
        names.append(item["tag_name"])
    return Classified("ok", None, None, 0, {"tags": names})


class ConfigError(ValueError):
    pass


class Worker(NamedTuple):
    id: str
    url: str
    weight: int


class EndpointConfig(NamedTuple):
    workers: List[Worker]  # empty: request the public endpoint directly
    user_agents: List[str]  # empty: send the HTTP library's default


def load_endpoints_config(path: Path) -> Dict[str, EndpointConfig]:
    """Load the private endpoints configuration.

    Format (one entry per endpoint, both entries optional)::

        {
          "view": {
            "workers": [{"id": "view-1", "url": "https://...", "weight": 1, "enabled": true}],
            "user_agents": ["..."]
          },
          "tags": {...}
        }

    A worker receives the same query string as the public endpoint and must
    return the public endpoint's response unchanged. Worker URLs are never
    stored or printed; attempts record the worker id instead.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read endpoints config: {type(exc).__name__}") from None
    if not isinstance(raw, dict):
        raise ConfigError("endpoints config must be a JSON object")
    unknown = set(raw) - set(ENDPOINTS)
    if unknown:
        raise ConfigError(f"unknown endpoints in config: {', '.join(sorted(unknown))}")
    result = {}
    for endpoint in ENDPOINTS:
        entry = raw.get(endpoint, {})
        if not isinstance(entry, dict) or set(entry) - {"workers", "user_agents"}:
            raise ConfigError(f"{endpoint}: expected an object with 'workers' and/or 'user_agents'")
        workers = []
        for index, item in enumerate(entry.get("workers", [])):
            where = f"{endpoint}.workers[{index}]"
            if not isinstance(item, dict) or set(item) - {"id", "url", "weight", "enabled"}:
                raise ConfigError(f"{where}: expected id, url, weight, enabled")
            if not isinstance(item.get("id"), str) or not item["id"]:
                raise ConfigError(f"{where}: id must be a non-empty string")
            url = item.get("url")
            if not isinstance(url, str) or urllib.parse.urlsplit(url).scheme not in ("http", "https"):
                raise ConfigError(f"{where}: url must be an http(s) URL")
            weight = item.get("weight", 1)
            if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                raise ConfigError(f"{where}: weight must be a positive integer")
            if item.get("enabled", True) is True:
                workers.append(Worker(item["id"], url, weight))
            elif item.get("enabled") is not False:
                raise ConfigError(f"{where}: enabled must be a boolean")
        if entry.get("workers") and not workers:
            raise ConfigError(f"{endpoint}: no enabled worker")
        agents = entry.get("user_agents", [])
        if not isinstance(agents, list) or not all(isinstance(a, str) and a for a in agents):
            raise ConfigError(f"{endpoint}: user_agents must be a list of non-empty strings")
        result[endpoint] = EndpointConfig(workers, list(agents))
    return result


def _scrub(text: str, secrets: List[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<worker>")
    return text


def http_transport(
    config: Optional[Dict[str, EndpointConfig]] = None,
    timeout: float = 10.0,
    rng: Optional[random.Random] = None,
    opener: Callable = urllib.request.urlopen,
) -> Transport:
    """Unauthenticated HTTP transport. No cookies or credentials are sent.

    For each request a worker is drawn by weight and a User-Agent is drawn
    uniformly from the endpoint's configuration. Without workers the public
    endpoint is requested directly.
    """
    config = config or {name: EndpointConfig([], []) for name in ENDPOINTS}
    rng = rng or random.Random()

    def fetch(endpoint: str, aid: int) -> Response:
        entry = config.get(endpoint, EndpointConfig([], []))
        if entry.workers:
            worker = rng.choices(entry.workers, weights=[w.weight for w in entry.workers])[0]
            base, via = worker.url, worker.id
            secrets = [worker.url, urllib.parse.urlsplit(worker.url).netloc]
        else:
            base, via, secrets = ENDPOINTS[endpoint], "direct", []
        user_agent = rng.choice(entry.user_agents) if entry.user_agents else None
        separator = "&" if "?" in base else "?"
        headers = {"Accept": "application/json"}
        if user_agent:
            headers["User-Agent"] = user_agent
        request = urllib.request.Request(f"{base}{separator}aid={aid}", headers=headers)
        try:
            with opener(request, timeout=timeout) as handle:
                return Response(handle.status, handle.read(), None, via, user_agent)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read()
            except Exception:  # pragma: no cover - defensive
                body = None
            return Response(exc.code, body, None, via, user_agent)
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            return Response(None, None, _scrub(f"{type(exc).__name__}: {exc}", secrets), via, user_agent)

    return fetch


def fixture_transport(directory: Path) -> Transport:
    """Offline transport that serves ``<endpoint>/<aid>.json`` files with HTTP 200.

    A missing file is reported as a transport error, which lets fixtures
    simulate network failures.
    """

    def fetch(endpoint: str, aid: int) -> Response:
        path = directory / endpoint / f"{aid}.json"
        if not path.exists():
            return Response(None, None, "fixture missing", "fixture")
        return Response(200, path.read_bytes(), None, "fixture")

    return fetch
