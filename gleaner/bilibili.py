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
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional

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

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


class Response(NamedTuple):
    http_status: Optional[int]
    body: Optional[bytes]
    error: Optional[str] = None  # transport error description, if any


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


def http_transport(timeout: float = 10.0, user_agent: str = DEFAULT_USER_AGENT) -> Transport:
    """A plain unauthenticated HTTPS transport. No cookies or credentials are sent."""

    def fetch(url: str, aid: int) -> Response:
        request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return Response(handle.status, handle.read())
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read()
            except Exception:  # pragma: no cover - defensive
                body = None
            return Response(exc.code, body)
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            return Response(None, None, f"{type(exc).__name__}: {exc}")

    return fetch


def fixture_transport(directory: Path) -> Transport:
    """Offline transport that serves ``<endpoint>/<aid>.json`` files with HTTP 200.

    A missing file is reported as a transport error, which lets fixtures
    simulate network failures.
    """
    endpoint_by_url = {base: name for name, base in ENDPOINTS.items()}

    def fetch(url: str, aid: int) -> Response:
        endpoint = endpoint_by_url[url.split("?", 1)[0]]
        path = directory / endpoint / f"{aid}.json"
        if not path.exists():
            return Response(None, None, "fixture missing")
        return Response(200, path.read_bytes())

    return fetch
