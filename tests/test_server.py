"""HTTP endpoint tests against a real server on an ephemeral port."""
import io
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

from test_core import env

from prism import server
from prism.observability import LOGGER, JsonFormatter, configure_logging


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


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(); self.lines = []; self.setFormatter(JsonFormatter())

    def emit(self, record):
        self.lines.append(self.format(record))


@contextmanager
def captured_logs():
    log = logging.getLogger(LOGGER); h = Capture()
    old = (log.level, log.propagate)
    log.addHandler(h); log.setLevel(logging.INFO); log.propagate = False
    try:
        yield h.lines
    finally:
        log.removeHandler(h); log.setLevel(old[0]); log.propagate = old[1]


def metric(text, name, **labels):
    """Value of one sample from Prometheus text (None if absent)."""
    want = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        head, _, val = line.rpartition(" ")
        base, _, lab = head.partition("{")
        if base == name and ",".join(sorted(lab.rstrip("}").split(","))) == want if lab else base == name and not want:
            return float(val)
    return None


def test_request_logs_are_structured_and_never_contain_query_text():
    with captured_logs() as lines, running() as base:
        call(base, "/search", {"query": "very-secret-search-phrase", "modality": "cell", "k": 3})
        call(base, "/search", {"query": "x", "k": 0})
        call(base, "/item/5"); call(base, "/definitely/not/a/route")
    recs = [json.loads(line) for line in lines]
    assert [r["event"] for r in recs] == ["request"] * 4 and all(r["level"] == "info" for r in recs)
    by = {(r["method"], r["path"], r["status"]): r for r in recs}
    ok = by[("POST", "/search", 200)]
    assert ok["hits"] == 3 and ok["scanned"] > 0 and ok["ms"] >= 0 and ok["bytes_out"] > 0 and ok["ts"].endswith("Z")
    assert ("POST", "/search", 400) in by and ("GET", "/item/{id}", 200) in by and ("GET", "other", 404) in by
    assert "very-secret-search-phrase" not in "".join(lines)


def test_agent_request_log_has_run_and_verdict_but_not_the_question():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with captured_logs() as lines, running() as base:
            call(base, "/agent", {"question": "hepatocytes expressing ALB confidential-question"})
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
    rec = json.loads(lines[-1])
    assert rec["path"] == "/agent" and rec["planner"] == "rule" and rec["grounded"] is True and rec["run"]
    assert "confidential-question" not in "".join(lines)


def test_unhandled_errors_log_a_traceback_but_return_a_generic_message():
    D, _, ENG = env()
    srv = server.make_server(ENG, port=0, trace_dir=f"{D}/traces")
    ENG_search, ENG.search = ENG.search, lambda **kw: 1 / 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with captured_logs() as lines:
            code, body = call(f"http://127.0.0.1:{srv.server_address[1]}", "/search", {"query": "x"})
    finally:
        ENG.search = ENG_search; srv.shutdown(); srv.server_close()
    assert code == 500 and json.loads(body) == {"error": "internal error"}
    errs = [json.loads(line) for line in lines if json.loads(line)["level"] == "error"]
    assert errs and "ZeroDivisionError" in errs[0]["exc"]


def test_metrics_endpoint():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with running() as base:
            call(base, "/search", {"query": "x"}); call(base, "/search", {"query": "x"})
            call(base, "/search", {"query": "x", "k": 0}); call(base, "/item/1"); call(base, "/nope")
            call(base, "/agent", {"question": "hepatocytes"})
            code, body = call(base, "/metrics")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
    text = body.decode()
    assert code == 200 and "# TYPE prism_http_request_duration_seconds histogram" in text
    assert metric(text, "prism_http_requests_total", method="POST", route="/search", status="200") == 2
    assert metric(text, "prism_http_requests_total", method="POST", route="/search", status="400") == 1
    assert metric(text, "prism_http_requests_total", method="GET", route="/item/{id}", status="200") == 1
    assert metric(text, "prism_http_requests_total", method="GET", route="other", status="404") == 1
    assert metric(text, "prism_http_request_duration_seconds_count", route="/search") == 3
    assert metric(text, "prism_http_request_duration_seconds_bucket", route="/search", le="+Inf") == 3
    assert metric(text, "prism_agent_runs_total", planner="rule", grounded="true") == 1
    assert metric(text, "prism_index_items") == 16384 and metric(text, "prism_up") == 1
    assert metric(text, "prism_http_inflight_requests") == 1  # the /metrics request itself


def test_metrics_require_the_token_when_one_is_set():
    with running(token="s3cret") as base:
        assert call(base, "/metrics")[0] == 401
        assert call(base, "/metrics", headers={"authorization": "Bearer s3cret"})[0] == 200


def test_metrics_cardinality_is_bounded_by_route_labels():
    with running() as base:
        for i in range(20):
            call(base, f"/junk/{i}"); call(base, f"/item/{i}")
        text = call(base, "/metrics")[1].decode()
    routes = {line.split('route="')[1].split('"')[0] for line in text.splitlines() if 'route="' in line}
    assert routes <= {"other", "/item/{id}", "/metrics"}


def test_configure_logging_quiet_and_idempotent():
    stream = io.StringIO()
    configure_logging(stream=stream); configure_logging(stream=stream)  # no duplicate handlers
    assert len([h for h in logging.getLogger(LOGGER).handlers if getattr(h, "_prism", False)]) == 1
    logging.getLogger(LOGGER).info("hello", extra={"fields": {"a": 1}})
    assert json.loads(stream.getvalue()) == {**json.loads(stream.getvalue()), "event": "hello", "a": 1}
    configure_logging(quiet=True)
    logging.getLogger(LOGGER).info("silenced")
    assert stream.getvalue().count("\n") == 1
    logging.getLogger(LOGGER).handlers.clear(); logging.getLogger(LOGGER).propagate = True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
