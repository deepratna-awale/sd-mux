"""
HTTP transport shared by every backend.

Connect and read timeouts, retries with backoff on connection errors (and on
502/503/504 for idempotent reads), basic or bearer auth, credentials in the URL,
extra headers, and failures turned into typed errors. Nothing here assumes the
server runs on this machine: http://127.0.0.1:7860, a LAN address and an https
reverse proxy with a sub-path and a login all work the same way.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ReadTimeoutError
from urllib3.util.retry import Retry

from . import __version__
from .errors import AuthRefused, Rejected, SdmuxError, TimedOut, Unreachable

USER_AGENT = f"sdmux/{__version__}"


def normalize_url(url: str | None, default: str | None = None) -> str:
    """Turn what a user typed (host:port, full URL, trailing slash) into a base URL without a trailing slash."""
    url = (url or "").strip()
    if not url:
        if default is None:
            raise ValueError("No URL given.")
        return default
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        url = "http://" + url
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Not a valid http(s) URL: {redact_url(url)}")
    return url.rstrip("/")


def redact_url(url: str | None) -> str:
    """The URL with any user:password@ replaced by ***@, safe to log or show."""
    url = url or ""
    try:
        parts = urlsplit(url if "://" in url else f"//{url}")
    except ValueError:
        return re.sub(r"[^/@]*@", "***@", url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return url.replace(parts.netloc, f"***@{host}", 1)


# Headers the transport manages itself; extra headers may not replace them.
RESERVED_HEADERS = frozenset({"host", "content-length", "content-type", "transfer-encoding", "connection", "upgrade", "te", "trailer"})
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def parse_headers(value) -> dict[str, str]:
    """
    Extra headers (for a server behind a login proxy, e.g. Cloudflare Access service tokens).

    Takes a dict, a JSON object string, or "Name: value" lines. Raises ValueError for a malformed or
    reserved header name, or a value containing a line break.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except ValueError as e:
                raise ValueError("Extra headers look like JSON but do not parse as a JSON object.") from e
        else:
            pairs = {}
            for line in text.splitlines():
                if not line.strip():
                    continue
                name, sep, header_value = line.partition(":")
                if not sep:
                    raise ValueError("Write each extra header as Name: value, one per line.")
                pairs[name.strip()] = header_value.strip()
            value = pairs
    if not isinstance(value, dict):
        raise ValueError("Extra headers must be a dict, a JSON object or Name: value lines.")
    headers = {}
    for name, header_value in value.items():
        name = str(name).strip()
        header_value = "" if header_value is None else str(header_value).strip()
        if not _HEADER_NAME.match(name):
            raise ValueError(f"{name!r} is not a valid header name.")
        if name.lower() in RESERVED_HEADERS:
            raise ValueError(f"The {name} header is set by sdmux and cannot be replaced.")
        if any(ch in header_value for ch in "\r\n\0"):
            raise ValueError(f"The {name} header's value has a line break in it.")
        headers[name] = header_value
    return headers


class _AuthSession(requests.Session):
    """A session that also drops the extra auth headers when a redirect leaves the server's host."""

    secret_headers: tuple[str, ...] = ()

    def rebuild_auth(self, prepared_request, response):
        if self.secret_headers and self.should_strip_auth(response.request.url, prepared_request.url):
            for name in self.secret_headers:
                prepared_request.headers.pop(name, None)
        super().rebuild_auth(prepared_request, response)


def split_credentials(url: str) -> tuple[str, str | None, str | None]:
    """Move user:password@ out of a URL so it can be sent as auth instead."""
    parts = urlsplit(url)
    if parts.username is None:
        return url, None, None
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = host + (f":{parts.port}" if parts.port else "")
    clean = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return clean, unquote(parts.username), unquote(parts.password or "")


@dataclass
class Options:
    """How to reach a server. Every field has a default that suits a local install."""

    username: str | None = None
    # Secrets stay out of repr() so they never reach a log line.
    password: str | None = field(default=None, repr=False)
    token: str | None = field(default=None, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    connect_timeout: float = 5.0
    timeout: float = 30.0
    job_timeout: float = 1800.0
    retries: int = 2
    verify_tls: bool = True
    poll_interval: float = 0.5

    @classmethod
    def build(cls, options: "Options | None" = None, **overrides) -> "Options":
        opts = dataclasses.replace(options) if options else cls()
        names = {f.name for f in dataclasses.fields(cls)}
        for key, value in overrides.items():
            if key not in names:
                raise TypeError(f"Unknown connection option: {key}")
            if value is not None:
                setattr(opts, key, value)
        opts.headers = parse_headers(opts.headers)
        return opts


class Transport:
    """A requests.Session bound to one base URL, with typed errors."""

    def __init__(self, url: str, options: Options | None = None, label: str = "server") -> None:
        self.options = Options.build(options)
        clean, user, password = split_credentials(normalize_url(url))
        self.base_url = clean
        if user and not self.options.username:
            self.options.username, self.options.password = user, password
        self.label = label
        self.session = self._new_session()
        self.apply_auth()

    def _new_session(self) -> _AuthSession:
        retries = self.options.retries
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
        session = _AuthSession()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = self.options.verify_tls
        session.headers.update({"Accept": "application/json", "User-Agent": USER_AGENT})
        return session

    def apply_auth(self) -> None:
        """Basic auth from username/password, bearer from token, plus any extra headers."""
        opts = self.options
        self.session.auth = (opts.username, opts.password or "") if opts.username else None
        for name in (*self.session.secret_headers, "Authorization"):
            self.session.headers.pop(name, None)
        if opts.token:
            self.session.headers["Authorization"] = f"Bearer {opts.token}"
        self.session.headers.update(opts.headers or {})
        self.session.secret_headers = tuple(opts.headers or {})

    def auth_headers(self) -> dict[str, str]:
        """The headers carrying this connection's login, for connections the session does not make (websockets)."""
        headers: dict[str, str] = {}
        auth = self.session.auth
        if isinstance(auth, tuple) and auth[0]:
            headers["Authorization"] = "Basic " + base64.b64encode(f"{auth[0]}:{auth[1] or ''}".encode()).decode()
        for name, value in self.session.headers.items():
            if name.lower() == "authorization" or name in self.session.secret_headers:
                headers[name] = value if isinstance(value, str) else value.decode()
        cookies = "; ".join(f"{c.name}={c.value}" for c in self.session.cookies)
        if cookies:
            headers["Cookie"] = cookies
        return headers

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def request(self, method: str, path: str, timeout: float | None = None, **kwargs) -> requests.Response:
        """Send a request; transport failures and 401/403 raise typed errors, other statuses are returned."""
        url = self.url(path)
        read_timeout = timeout if timeout is not None else self.options.timeout
        try:
            response = self.session.request(method, url, timeout=(self.options.connect_timeout, read_timeout), **kwargs)
        except requests.exceptions.ConnectTimeout as e:
            raise Unreachable(f"Timed out connecting to {self.label} at {self.base_url}.", url=self.base_url) from e
        except requests.exceptions.ReadTimeout as e:
            raise TimedOut(f"{self.label} did not answer {method} {path} within {read_timeout:g}s.", url=self.base_url) from e
        except requests.exceptions.SSLError as e:
            raise Unreachable(f"TLS error talking to {self.label} at {self.base_url}: {e}", url=self.base_url) from e
        except requests.exceptions.ConnectionError as e:
            # Once retries are used up, urllib3 reports a read timeout as a connection error.
            reason = getattr(e.args[0] if e.args else None, "reason", None)
            if isinstance(reason, ReadTimeoutError):
                raise TimedOut(f"{self.label} did not answer {method} {path} within {read_timeout:g}s.", url=self.base_url) from e
            raise Unreachable(f"Could not reach {self.label} at {self.base_url}. Is it running?", url=self.base_url) from e
        except requests.exceptions.RequestException as e:
            raise SdmuxError(f"{self.label} request {method} {path} failed: {e}", url=self.base_url) from e
        if response.status_code in (401, 403):
            raise AuthRefused(
                f"{self.label} refused the request ({response.status_code}). Check the username/password or token.",
                url=self.base_url,
            )
        return response

    @staticmethod
    def json(response: requests.Response):
        try:
            return response.json()
        except ValueError as e:
            raise Rejected(f"Expected JSON but got: {response.text[:300]!r}") from e

    def check(self, response: requests.Response, what: str) -> requests.Response:
        if response.status_code >= 400:
            raise Rejected(f"{self.label} {what} failed ({response.status_code}): {response.text[:1000]}", details=response.text)
        return response

    def get_json(self, path: str, **kwargs):
        return self.json(self.check(self.request("GET", path, **kwargs), f"GET {path}"))

    def post_json(self, path: str, body=None, **kwargs):
        return self.json(self.check(self.request("POST", path, json=body, **kwargs), f"POST {path}"))

    def close(self) -> None:
        self.session.close()
