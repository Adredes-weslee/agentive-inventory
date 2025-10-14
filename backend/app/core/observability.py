r"""backend\app\core\observability.py"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable

# Prometheus client is optional in constrained environments. Provide a lightweight
# fallback so the application can still expose metrics in Prometheus text format.
try:  # pragma: no cover - exercised indirectly in environments with the package
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )
except ImportError:  # pragma: no cover - fallback implementation
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _BaseMetric:
        def __init__(self, name: str, documentation: str, labelnames: list[str] | tuple[str, ...]):
            self._name = name
            self._documentation = documentation
            self._labelnames = tuple(labelnames)
            self._lock = threading.Lock()
            _METRIC_REGISTRY.append(self)

        def _format_labels(self, label_values: tuple[str, ...]) -> str:
            if not self._labelnames:
                return ""
            pairs = [
                f'{label}="{value}"'
                for label, value in zip(self._labelnames, label_values, strict=True)
            ]
            return "{" + ",".join(pairs) + "}"

        def render(self) -> str:
            raise NotImplementedError

    _METRIC_REGISTRY: list[_BaseMetric] = []

    class _FallbackCounter(_BaseMetric):
        def __init__(self, name: str, documentation: str, labelnames: list[str] | tuple[str, ...]):
            super().__init__(name, documentation, labelnames)
            self._values: defaultdict[tuple[str, ...], float] = defaultdict(float)

        class _CounterChild:
            def __init__(self, parent: "_FallbackCounter", key: tuple[str, ...]):
                self._parent = parent
                self._key = key

            def inc(self, amount: float = 1.0) -> None:
                if amount < 0:
                    raise ValueError("Counters can only be incremented by non-negative amounts")
                with self._parent._lock:
                    self._parent._values[self._key] += amount

        def labels(self, *labelvalues: str) -> "_FallbackCounter._CounterChild":
            if len(labelvalues) != len(self._labelnames):
                raise ValueError("Incorrect number of labels provided")
            return _FallbackCounter._CounterChild(self, tuple(labelvalues))

        def render(self) -> str:
            lines = [
                f"# HELP {self._name} {self._documentation}",
                f"# TYPE {self._name} counter",
            ]
            with self._lock:
                for labels, value in self._values.items():
                    label_str = self._format_labels(labels)
                    lines.append(f"{self._name}{label_str} {value}")
            return "\n".join(lines)

    class _FallbackHistogram(_BaseMetric):
        _DEFAULT_BUCKETS = (
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
            float("inf"),
        )

        def __init__(
            self,
            name: str,
            documentation: str,
            labelnames: list[str] | tuple[str, ...],
            buckets: tuple[float, ...] | None = None,
        ) -> None:
            super().__init__(name, documentation, labelnames)
            self._buckets = tuple(buckets) if buckets is not None else self._DEFAULT_BUCKETS
            if self._buckets[-1] != float("inf"):
                self._buckets = (*self._buckets, float("inf"))
            self._bucket_counts: defaultdict[tuple[str, ...], list[int]] = defaultdict(
                lambda: [0 for _ in self._buckets]
            )
            self._sums: defaultdict[tuple[str, ...], float] = defaultdict(float)
            self._counts: defaultdict[tuple[str, ...], int] = defaultdict(int)

        class _HistogramChild:
            def __init__(self, parent: "_FallbackHistogram", key: tuple[str, ...]):
                self._parent = parent
                self._key = key

            def observe(self, amount: float) -> None:
                if amount < 0:
                    raise ValueError("Histograms cannot observe negative values")
                with self._parent._lock:
                    buckets = self._parent._bucket_counts[self._key]
                    for idx, bound in enumerate(self._parent._buckets):
                        if amount <= bound:
                            buckets[idx] += 1
                    self._parent._sums[self._key] += amount
                    self._parent._counts[self._key] += 1

        def labels(self, *labelvalues: str) -> "_FallbackHistogram._HistogramChild":
            if len(labelvalues) != len(self._labelnames):
                raise ValueError("Incorrect number of labels provided")
            return _FallbackHistogram._HistogramChild(self, tuple(labelvalues))

        def render(self) -> str:
            lines = [
                f"# HELP {self._name} {self._documentation}",
                f"# TYPE {self._name} histogram",
            ]
            with self._lock:
                for labels, buckets in self._bucket_counts.items():
                    cumulative = 0
                    for idx, bound in enumerate(self._buckets):
                        cumulative += buckets[idx]
                        bound_label = "+Inf" if math.isinf(bound) else str(bound)
                        label_pairs = self._format_labels(labels)
                        prefix = f"{self._name}_bucket"
                        if label_pairs:
                            base_labels = label_pairs[:-1]  # remove closing brace
                            label_str = f"{base_labels},le=\"{bound_label}\"}}"
                        else:
                            label_str = f"{{le=\"{bound_label}\"}}"
                        lines.append(f"{prefix}{label_str} {cumulative}")
                    sum_value = self._sums.get(labels, 0.0)
                    count_value = self._counts.get(labels, 0)
                    label_str = self._format_labels(labels)
                    lines.append(f"{self._name}_sum{label_str} {sum_value}")
                    lines.append(f"{self._name}_count{label_str} {count_value}")
            return "\n".join(lines)

    def generate_latest() -> bytes:
        payload_lines = [metric.render() for metric in _METRIC_REGISTRY]
        return ("\n".join(payload_lines) + "\n").encode("utf-8")

    Counter = _FallbackCounter 
    Histogram = _FallbackHistogram
    
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response


_REQUEST_COUNTER = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
_LATENCY_HISTOGRAM = Histogram(
    "http_request_latency_seconds", "Request latency", ["method", "path"]
)


class TokenAndRateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing auth, rate limiting, logging, and Prometheus metrics."""

    _lock: threading.Lock = threading.Lock()
    _buckets: dict[str, deque[float]] = defaultdict(deque)
    _per_minute: int = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    # IMPORTANT: During pytest runs we default to *disabling* auth even if a host
    # environment happens to have API_TOKEN set. Individual tests that verify auth
    # behavior (see test_auth_and_rate.py) explicitly monkeypatch this class
    # attribute to a non-None value before creating a TestClient.
    # This keeps the rest of the API tests independent of the developer's shell/env.
    _token: str | None = None if os.getenv("PYTEST_CURRENT_TEST") else os.getenv("API_TOKEN")
    _exempt_prefixes: tuple[str, ...] = (
        "/api/v1/health",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        request_id = (
            request.headers.get("x-request-id")
            or request.headers.get("request-id")
            or str(uuid.uuid4())
        )
        sku_id = None
        if hasattr(request, "path_params"):
            sku_value = request.path_params.get("sku_id")
            if sku_value is not None:
                sku_id = str(sku_value)

        # Optionally parse JSON body for POST/PUT/PATCH on known routes to enrich logs
        # NOTE: reading the body consumes it; we reconstruct the Request with a
        # custom `receive` so downstream still gets the original payload.
        if request.method in {"POST", "PUT", "PATCH"} and (
            path.startswith("/api/v1/procure") or path.startswith("/api/v1/approvals")
        ):
            try:
                body_bytes = await request.body()
            except Exception:
                body_bytes = b""

            if body_bytes:
                try:
                    data = json.loads(body_bytes.decode("utf-8"))
                    if isinstance(data, dict):
                        if not sku_id and isinstance(data.get("sku_id"), (str, int)):
                            sku_id = str(data["sku_id"])
                        elif (
                            not sku_id
                            and isinstance(data.get("sku_ids"), list)
                            and data["sku_ids"]
                        ):
                            sku_id = str(data["sku_ids"][0])
                except Exception:
                    # Non-JSON body; ignore
                    pass

                # Replay body to downstream app (since we consumed it)
                async def receive() -> dict:
                    return {"type": "http.request", "body": body_bytes, "more_body": False}

                request = Request(request.scope, receive)

        model_used_header = (
            request.headers.get("model_used")
            or request.headers.get("x-model-used")
            or request.headers.get("model-used")
        )

        start_perf = time.perf_counter()
        start_wall = time.time()

        def _finalize(response: Response) -> Response:
            latency = time.perf_counter() - start_perf
            status_code = getattr(response, "status_code", 500)
            path_label = path

            try:
                _REQUEST_COUNTER.labels(method, path_label, str(status_code)).inc()
                _LATENCY_HISTOGRAM.labels(method, path_label).observe(latency)
            except Exception:
                # Metrics errors should never break request handling.
                pass

            response_model_used = (
                response.headers.get("model_used")
                if hasattr(response, "headers")
                else None
            )
            model_used = response_model_used or model_used_header

            log_payload = {
                "timestamp": datetime.fromtimestamp(
                    start_wall, tz=timezone.utc
                ).isoformat(),
                "path": path,
                "method": method,
                "status": status_code,
                "latency_ms": int(latency * 1000),
                "request_id": request_id,
                "client_ip": client_ip,
                "sku_id": sku_id,
                "model_used": model_used,
            }

            try:
                print(json.dumps(log_payload))
            except Exception:
                pass

            return response

        # Token authentication
        if self._token and not path.startswith(self._exempt_prefixes):
            auth_header = request.headers.get("authorization", "")
            if auth_header != f"Bearer {self._token}":
                error_response = PlainTextResponse("Unauthorized", status_code=401)
                return _finalize(error_response)

        # Rate limiting per client IP
        if self._per_minute > 0:
            now = time.time()
            with self._lock:
                window = self._buckets[client_ip]
                while window and now - window[0] > 60.0:
                    window.popleft()
                if len(window) >= self._per_minute:
                    error_response = PlainTextResponse("Too Many Requests", status_code=429)
                    return _finalize(error_response)
                window.append(now)

        response: Response
        try:
            response = await call_next(request)
        except Exception:
            # Even if downstream fails we still want metrics/logs; re-raise after logging.
            response = PlainTextResponse(
                "Internal Server Error", status_code=500
            )
            _finalize(response)
            raise

        return _finalize(response)


def metrics_endpoint() -> Response:
    """Return Prometheus metrics payload."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

