"""Structured JSON logs and Prometheus-format metrics for the HTTP server (stdlib only).

Logs: one JSON object per line on stderr, e.g.
  {"ts": "2026-10-17T22:14:07.123Z", "level": "info", "event": "request", "method": "POST", "path": "/search", ...}
Request/response bodies, queries and questions are never logged -- only routes, status, timing and sizes.

Metrics: GET /metrics, Prometheus text exposition format 0.0.4.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone

LOGGER = "prism.server"
BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON line: ts, level, event, plus the structured `fields` passed via extra=."""

    def format(self, record):
        ts = datetime.fromtimestamp(record.created, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(record.msecs):03d}Z"
        out = {"ts": ts, "level": record.levelname.lower(), "event": record.getMessage(), **getattr(record, "fields", {})}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(quiet=False, stream=None):
    """Attach the JSON handler to the server logger (idempotent). quiet=True silences request logs."""
    log = logging.getLogger(LOGGER)
    for h in list(log.handlers):
        if getattr(h, "_prism", False):
            log.removeHandler(h)
    log.propagate = False
    if quiet:
        log.addHandler(logging.NullHandler())
        return log
    h = logging.StreamHandler(stream or sys.stderr)
    h._prism = True  # type: ignore[attr-defined]
    h.setFormatter(JsonFormatter())
    log.addHandler(h); log.setLevel(logging.INFO)
    return log


def route_label(path):
    """Bounded-cardinality route name for labels/logs: never the raw path (item ids, junk probes)."""
    path = path.split("?", 1)[0]
    if path in ("/", "/index.html"):
        return "/"
    if path in ("/health", "/metrics", "/search", "/similar", "/agent"):
        return path
    if path.startswith("/item/"):
        return "/item/{id}"
    return "other"


class Metrics:
    """Thread-safe counters, gauges and a latency histogram, rendered as Prometheus text."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.time()
        self._requests: dict[tuple[str, str, int], int] = {}
        self._latency: dict[str, list[float]] = {}   # route -> [bucket counts..., sum, count]
        self._agent: dict[tuple[str, str], int] = {}
        self._inflight = 0

    def inflight(self, delta):
        with self._lock:
            self._inflight += delta

    def observe(self, method, route, status, seconds):
        with self._lock:
            key = (method, route, int(status))
            self._requests[key] = self._requests.get(key, 0) + 1
            h = self._latency.setdefault(route, [0.0] * (len(BUCKETS) + 2))
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    h[i] += 1
            h[-2] += seconds; h[-1] += 1

    def agent_run(self, planner, grounded):
        with self._lock:
            key = (planner, "true" if grounded else "false")
            self._agent[key] = self._agent.get(key, 0) + 1

    def render(self, items=None):
        with self._lock:
            lines = ["# HELP prism_up 1 while the server is running.", "# TYPE prism_up gauge", "prism_up 1",
                     "# HELP prism_uptime_seconds Seconds since the server started.", "# TYPE prism_uptime_seconds gauge",
                     f"prism_uptime_seconds {time.time() - self._started:.3f}"]
            if items is not None:
                lines += ["# HELP prism_index_items Items in the loaded index.", "# TYPE prism_index_items gauge",
                          f"prism_index_items {items}"]
            lines += ["# HELP prism_http_inflight_requests Requests currently being handled.",
                      "# TYPE prism_http_inflight_requests gauge", f"prism_http_inflight_requests {self._inflight}",
                      "# HELP prism_http_requests_total HTTP requests by method, route and status.",
                      "# TYPE prism_http_requests_total counter"]
            for (m, r, s), n in sorted(self._requests.items()):
                lines.append(f'prism_http_requests_total{{method="{m}",route="{r}",status="{s}"}} {n}')
            lines += ["# HELP prism_http_request_duration_seconds Request latency by route.",
                      "# TYPE prism_http_request_duration_seconds histogram"]
            for r, h in sorted(self._latency.items()):
                for b, c in zip(BUCKETS, h, strict=False):
                    lines.append(f'prism_http_request_duration_seconds_bucket{{route="{r}",le="{b}"}} {int(c)}')
                lines.append(f'prism_http_request_duration_seconds_bucket{{route="{r}",le="+Inf"}} {int(h[-1])}')
                lines.append(f'prism_http_request_duration_seconds_sum{{route="{r}"}} {h[-2]:.6f}')
                lines.append(f'prism_http_request_duration_seconds_count{{route="{r}"}} {int(h[-1])}')
            lines += ["# HELP prism_agent_runs_total Agent runs by planner and grounding verdict.",
                      "# TYPE prism_agent_runs_total counter"]
            for (p, g), n in sorted(self._agent.items()):
                lines.append(f'prism_agent_runs_total{{planner="{p}",grounded="{g}"}} {n}')
        return "\n".join(lines) + "\n"
