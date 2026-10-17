"""Claude planner (with a mocked API) and rule-planner behaviour."""
import io
import json
import urllib.error
from test_core import env
from prism import agent as A


def tool_use(name, args, uid="u1"):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": uid, "name": name, "input": args}]}


def text(t):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": t}]}


def scripted(responses):
    """Planner whose _post replays `responses` and records the request bodies it was sent."""
    p = A.ClaudePlanner(api_key="test-key"); sent = []
    it = iter(responses)
    p._post = lambda body: (sent.append(json.loads(json.dumps(body))), next(it))[1]
    return p, sent


def harness(**kw):
    D, _, ENG = env()
    return A.Harness(ENG, trace_dir=f"{D}/traces", **kw), ENG


def test_claude_loop_grounded_answer():
    h, ENG = harness()
    hid = ENG.search("hepatocyte", modality="cell", k=1)["hits"][0]["id"]
    p, sent = scripted([tool_use("search_corpus", {"query": "hepatocyte", "modality": "cell", "k": 1}),
                        text(f"Best match is #{hid}.")])
    r = p.run(h, "find hepatocytes")
    assert r["grounded"] and r["steps"] == 1
    assert "Today is" in sent[0]["system"] and ENG.today in sent[0]["system"]
    result = sent[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "u1" and result["is_error"] is False and json.loads(result["content"])["hits"]


def test_claude_fabricated_citation_is_flagged():
    h, _ = harness()
    p, _ = scripted([text("I found item 424242 which matches.")])
    r = p.run(h, "q")
    assert not r["grounded"] and r["ungrounded_ids"] == [424242]


def test_claude_invalid_tool_input_is_reported_back_as_error():
    h, _ = harness()
    p, sent = scripted([tool_use("search_corpus", {"query": "x", "k": 999}), text("no results")])
    p.run(h, "q")
    result = sent[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "max" in json.loads(result["content"])["error"]


def test_claude_stops_on_step_budget():
    h, _ = harness(max_steps=1)
    p, _ = scripted([tool_use("search_corpus", {"query": "a"}), tool_use("search_corpus", {"query": "b"}, "u2")])
    r = p.run(h, "q")
    assert r["answer"].startswith("[stopped by policy") and "step budget" in r["answer"]


def test_claude_disallowed_tool():
    h, _ = harness(allow=["search_corpus"])
    p, sent = scripted([tool_use("inspect_item", {"item_id": 1}), text("done")])
    p.run(h, "q")
    assert "not allowed" in json.loads(sent[1]["messages"][-1]["content"][0]["content"])["error"]


def test_trace_closed_when_api_fails():
    h, _ = harness()
    p = A.ClaudePlanner(api_key="k")
    def boom(body): raise RuntimeError("api down")
    p._post = boom
    try:
        p.run(h, "q")
    except RuntimeError:
        pass
    assert h.trace.closed


def test_post_retries_transient_errors(monkeypatch=None):
    calls = []
    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError("u", 429, "rate", {}, io.BytesIO(b""))
        return io.BytesIO(b'{"ok": true}')
    real_urlopen, real_sleep = A.urllib.request.urlopen, A.time.sleep
    A.urllib.request.urlopen, A.time.sleep = fake_urlopen, lambda s: None
    try:
        assert A.ClaudePlanner(api_key="k")._post({}) == {"ok": True} and len(calls) == 3
        calls.clear()
        def always_400(req, timeout):
            calls.append(1); raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b""))
        A.urllib.request.urlopen = always_400
        try:
            A.ClaudePlanner(api_key="k")._post({})
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            assert len(calls) == 1  # 4xx other than 429 is not retried
    finally:
        A.urllib.request.urlopen, A.time.sleep = real_urlopen, real_sleep


def test_clip_keeps_json_valid():
    out = {"hits": [{"id": i, "pad": "x" * 200} for i in range(200)], "probe": {}}
    clipped = json.loads(A._clip(out, limit=3000))
    assert clipped["truncated"] and 0 < len(clipped["hits"]) < 200


# ------------------------------------------------------------------ rule planner
def test_split_clauses():
    assert A.split_clauses("A; B") == ["A", "B"]
    assert A.split_clauses("burn scars near Malibu and also hepatocytes") == ["burn scars near Malibu", "hepatocytes"]
    assert A.split_clauses("dna x plus cells y") == ["dna x", "cells y"]
    assert A.split_clauses("first and then second") == ["first", "second"]
    assert A.split_clauses("one also two") == ["one", "two"]
    assert A.split_clauses("  ;  ; ") == []
    assert A.split_clauses("a single clause") == ["a single clause"]


def test_rule_planner_routes_clauses_to_modalities():
    h, _ = harness()
    r = A.RulePlanner().run(h, "AAV capsid sequences; also hepatocytes expressing ALB")
    searches = [json.loads(l) for l in open(r["trace"]) if '"tool": "search_corpus"' in l]
    assert [(s["args"]["modality"], s["args"]["query"]) for s in searches] == [
        ("dna", "AAV capsid sequences"), ("cell", "hepatocytes expressing ALB")]
    assert r["grounded"] and h.trace.closed


def test_rule_planner_applies_place_and_year_filters():
    h, _ = harness()
    r = A.RulePlanner().run(h, "wildfire burn scars near Malibu in 2025")
    s = next(json.loads(l) for l in open(r["trace"]) if '"tool": "search_corpus"' in l)
    assert s["args"]["modality"] == "sat" and s["args"]["place"] == "malibu"
    assert (s["args"]["date_from"], s["args"]["date_to"]) == ("2025-01-01", "2025-12-31")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
