"""Minimal JSON API (stdlib only). Endpoints:
  GET  /            search UI
  GET  /health
  POST /search   {"query": "...", "modality": "sat", "place": "malibu", "date_from": "...", "k": 10}
  POST /similar  {"item_id": 123, "modality": "cell", "k": 10}
  GET  /item/<id>
  POST /agent    {"question": "..."}
"""
from __future__ import annotations

import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .harness import Harness
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
    """Build (not start) the server. token: if set, POST endpoints require 'Authorization: Bearer <token>'.
    Binds to localhost by default; port 0 picks a free port (see srv.server_address)."""

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

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

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    body = UI.encode()
                    self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
                    self.send_header("content-length", str(len(body))); self.end_headers()
                    return self.wfile.write(body)
                if self.path == "/health":
                    return self._send(200, {"ok": True, "items": len(eng)})
                if self.path.startswith("/item/"):
                    return self._send(200, eng.inspect(int(self.path.split("/")[-1])))
                self._send(404, {"error": "not found"})
            except (ValueError, KeyError, IndexError) as e:
                self._send(400, {"error": str(e)})
            except Exception:
                self._send(500, {"error": "internal error"})

        def do_POST(self):
            try:
                if not self._authorized():
                    return self._send(401, {"error": "unauthorized"})
                b = self._body()
                if self.path == "/search":
                    return self._send(200, eng.search(**_clean(b, _SEARCH_FIELDS)))
                if self.path == "/similar":
                    return self._send(200, eng.similar(**_clean(b, _SIMILAR_FIELDS)))
                if self.path == "/agent":
                    q = b.get("question") if isinstance(b, dict) else None
                    if not isinstance(q, str) or not q.strip() or len(q) > 2000:
                        raise BadRequest("question must be a non-empty string (max 2000 chars)")
                    # Harness is created per request, so concurrent runs share no state.
                    h = Harness(eng, trace_dir=trace_dir)
                    return self._send(200, default_planner().run(h, q))
                self._send(404, {"error": "not found"})
            except (BadRequest, ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except Exception:
                self._send(500, {"error": "internal error"})

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), H)


def serve(eng, host="127.0.0.1", port=8080, trace_dir="traces", token=None):
    srv = make_server(eng, host, port, trace_dir, token)
    print(f"PRISM serving {len(eng):,} items on http://{host}:{srv.server_address[1]}")
    srv.serve_forever()
