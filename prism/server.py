"""Minimal JSON API (stdlib only). Endpoints:
  GET  /            search UI
  GET  /health
  POST /search   {"query": "...", "modality": "sat", "place": "malibu", "date_from": "...", "k": 10}
  POST /similar  {"item_id": 123, "modality": "cell", "k": 10}
  GET  /item/<id>
  POST /agent    {"question": "..."}
  GET  /metrics  Prometheus text metrics (needs the bearer token when one is set)
"""
from __future__ import annotations

import hmac
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .harness import Harness
from .observability import LOGGER, Metrics, configure_logging, route_label
from .planners import default_planner

UI = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


MAX_BODY = 64 * 1024
# Request fields accepted per endpoint, with caps so a client cannot request unbounded work.
_SEARCH_FIELDS = {"query", "modality", "place", "date_from", "date_to", "k"}
_SIMILAR_FIELDS = {"item_id", "modality", "k"}


class BadRequest(Exception):
    pass


def _clean(b, allowed):
    if not isinstance(b, dict):
        raise BadRequest("body must be a JSON object")
    extra = set(b) - allowed
    if extra:
        raise BadRequest(f"unexpected fields: {sorted(extra)}")
    k = b.get("k", 10)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise BadRequest("k must be an integer in 1..100")
    return b


def make_server(eng, host="127.0.0.1", port=8080, trace_dir="traces", token=None):
    """Build (not start) the server. token: if set, POST endpoints and /metrics require
    'Authorization: Bearer <token>'. Binds to localhost by default; port 0 picks a free port
    (see srv.server_address). Request logs go to the 'prism.server' logger as JSON (observability.py);
    srv.metrics holds the counters served at /metrics."""
    metrics = Metrics()
    log = logging.getLogger(LOGGER)

    class H(BaseHTTPRequestHandler):
        def _send_bytes(self, code, body, ctype):
            self._status, self._bytes = code, len(body)
            self.send_response(code); self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _send(self, code, obj):
            self._send_bytes(code, json.dumps(obj, default=str).encode(), "application/json")

        def _body(self):
            try:
                n = int(self.headers.get("content-length", 0))
            except ValueError:
                raise BadRequest("bad content-length") from None
            if n < 0 or n > MAX_BODY:
                raise BadRequest("body too large")
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                raise BadRequest("invalid JSON") from None

        def _authorized(self):
            if token is None:
                return True
            got = self.headers.get("authorization", "")
            return hmac.compare_digest(got, f"Bearer {token}")

        def _timed(self, handler):
            """Run a handler with metrics + one structured log line per request (no bodies, no query text)."""
            self._status, self._bytes = 500, 0
            self._fields: dict[str, Any] = {}
            t0 = time.perf_counter(); metrics.inflight(1)
            try:
                handler()
            finally:
                dt = time.perf_counter() - t0; metrics.inflight(-1)
                route = route_label(self.path)
                metrics.observe(self.command, route, self._status, dt)
                log.info("request", extra={"fields": {
                    "method": self.command, "path": route, "status": self._status, "ms": round(dt * 1e3, 2),
                    "bytes_out": self._bytes, "client": self.client_address[0], **self._fields}})

        def _fail(self, code, msg, exc=None):
            if code >= 500:
                log.error("unhandled_error", exc_info=exc or True, extra={"fields": {"path": route_label(self.path)}})
            self._send(code, {"error": msg})

        def do_GET(self):
            self._timed(self._get)

        def do_POST(self):
            self._timed(self._post)

        def _get(self):
            try:
                if self.path in ("/", "/index.html"):
                    return self._send_bytes(200, UI.encode(), "text/html; charset=utf-8")
                if self.path == "/health":
                    return self._send(200, {"ok": True, "items": len(eng)})
                if self.path == "/metrics":
                    if not self._authorized():
                        return self._send(401, {"error": "unauthorized"})
                    return self._send_bytes(200, metrics.render(items=len(eng)).encode(),
                                            "text/plain; version=0.0.4; charset=utf-8")
                if self.path.startswith("/item/"):
                    return self._send(200, eng.inspect(int(self.path.split("/")[-1])))
                self._send(404, {"error": "not found"})
            except (ValueError, KeyError, IndexError) as e:
                self._fail(400, str(e))
            except Exception:
                self._fail(500, "internal error")

        def _post(self):
            try:
                if not self._authorized():
                    return self._send(401, {"error": "unauthorized"})
                b = self._body()
                if self.path == "/search":
                    r = eng.search(**_clean(b, _SEARCH_FIELDS))
                    self._fields = {"hits": len(r["hits"]), "scanned": r["probe"]["scanned"]}
                    return self._send(200, r)
                if self.path == "/similar":
                    r = eng.similar(**_clean(b, _SIMILAR_FIELDS))
                    self._fields = {"hits": len(r["hits"]), "scanned": r["probe"]["scanned"]}
                    return self._send(200, r)
                if self.path == "/agent":
                    q = b.get("question") if isinstance(b, dict) else None
                    if not isinstance(q, str) or not q.strip() or len(q) > 2000:
                        raise BadRequest("question must be a non-empty string (max 2000 chars)")
                    # Harness is created per request, so concurrent runs share no state.
                    h = Harness(eng, trace_dir=trace_dir)
                    planner = default_planner()
                    r = planner.run(h, q)
                    metrics.agent_run(planner.name, r["grounded"])
                    self._fields = {"run": r["run_id"], "planner": planner.name, "grounded": r["grounded"],
                                    "steps": r["steps"], "claims_contradicted": len(r["claims"]["contradicted"])}
                    return self._send(200, r)
                self._send(404, {"error": "not found"})
            except (BadRequest, ValueError, KeyError, TypeError) as e:
                self._fail(400, str(e))
            except Exception:
                self._fail(500, "internal error")

        def log_message(self, *a):
            pass  # replaced by the structured request log above

    srv = ThreadingHTTPServer((host, port), H)
    srv.metrics = metrics  # type: ignore[attr-defined]
    return srv


def serve(eng, host="127.0.0.1", port=8080, trace_dir="traces", token=None, quiet=False):
    configure_logging(quiet=quiet)
    srv = make_server(eng, host, port, trace_dir, token)
    print(f"PRISM serving {len(eng):,} items on http://{host}:{srv.server_address[1]}")
    srv.serve_forever()
