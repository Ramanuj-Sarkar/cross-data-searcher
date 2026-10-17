"""HTTP endpoint tests against a real server on an ephemeral port."""
import json
import os
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

from test_core import env

from prism import server


@contextmanager
def running(token=None):
    D, _, ENG = env()
    srv = server.make_server(ENG, port=0, trace_dir=f"{D}/traces", token=token)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown(); srv.server_close()


def call(base, path, body=None, headers=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_get_endpoints():
    with running() as base:
        code, body = call(base, "/")
        assert code == 200 and b"<title>PRISM</title>" in body
        code, body = call(base, "/health")
        assert code == 200 and json.loads(body)["items"] == 16384
        code, body = call(base, "/item/5")
        assert code == 200 and json.loads(body)["id"] == 5
        assert call(base, "/item/not-a-number")[0] == 400
        assert call(base, "/item/999999999")[0] == 400
        assert call(base, "/nope")[0] == 404


def test_search_and_similar():
    with running() as base:
        code, body = call(base, "/search", {"query": "hepatocyte", "modality": "cell", "k": 5})
        hits = json.loads(body)["hits"]
        assert code == 200 and len(hits) == 5 and all(h["modality"] == "cell" for h in hits)
        code, body = call(base, "/similar", {"item_id": hits[0]["id"], "k": 3})
        assert code == 200 and hits[0]["id"] not in [h["id"] for h in json.loads(body)["hits"]]


def test_bad_requests_are_rejected():
    with running() as base:
        for body in ({"query": "x", "k": 0}, {"query": "x", "k": 101}, {"query": "x", "k": True},
                     {"query": "x", "nprobe": 10 ** 6}, {"query": "x", "bbox": [0, 1, 0, 1]},
                     {"query": "x", "date_from": "garbage"}, {"query": "x", "place": "atlantis"}, [1, 2]):
            assert call(base, "/search", body)[0] == 400, body
        assert call(base, "/similar", {"item_id": -1})[0] == 400
        assert call(base, "/search", raw=b"{not json")[0] == 400
        assert call(base, "/agent", {"question": ""})[0] == 400
        assert call(base, "/agent", {"question": "x" * 5000})[0] == 400
        assert call(base, "/search", raw=b"{" + b" " * (server.MAX_BODY + 1) + b"}")[0] == 400
        assert call(base, "/search", {"query": "x"}, headers={"content-length": "abc"})[0] in (400, 200)


def test_bearer_token():
    with running(token="s3cret") as base:
        assert call(base, "/search", {"query": "x"})[0] == 401
        assert call(base, "/search", {"query": "x"}, headers={"authorization": "Bearer wrong"})[0] == 401
        assert call(base, "/search", {"query": "x"}, headers={"authorization": "Bearer s3cret"})[0] == 200
        assert call(base, "/health")[0] == 200  # read-only endpoints stay open


def test_agent_endpoint_with_rule_planner():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with running() as base:
            code, body = call(base, "/agent", {"question": "AAV capsid sequences; also hepatocytes expressing ALB"})
            r = json.loads(body)
            assert code == 200 and r["grounded"] and r["steps"] >= 4
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
