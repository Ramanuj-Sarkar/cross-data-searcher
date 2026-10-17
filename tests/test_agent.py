"""Claude planner (with a mocked API) and rule-planner behaviour."""
import io
import json
import urllib.error

from test_core import env

from prism import harness as H
from prism import planners as PL


def tool_use(name, args, uid="u1"):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": uid, "name": name, "input": args}]}


def text(t):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": t}]}


def scripted(responses):
    """Planner whose _post replays `responses` and records the request bodies it was sent."""
    p = PL.ClaudePlanner(api_key="test-key"); sent = []
    it = iter(responses)
    p._post = lambda body: (sent.append(json.loads(json.dumps(body))), next(it))[1]
    return p, sent


def harness(**kw):
    D, _, ENG = env()
    return H.Harness(ENG, trace_dir=f"{D}/traces", **kw), ENG


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
    p = PL.ClaudePlanner(api_key="k")
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
    real_urlopen, real_sleep = PL.urllib.request.urlopen, PL.time.sleep
    PL.urllib.request.urlopen, PL.time.sleep = fake_urlopen, lambda s: None
    try:
        assert PL.ClaudePlanner(api_key="k")._post({}) == {"ok": True} and len(calls) == 3
        calls.clear()
        def always_400(req, timeout):
            calls.append(1); raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b""))
        PL.urllib.request.urlopen = always_400
        try:
            PL.ClaudePlanner(api_key="k")._post({})
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            assert len(calls) == 1  # 4xx other than 429 is not retried
    finally:
        PL.urllib.request.urlopen, PL.time.sleep = real_urlopen, real_sleep


def test_clip_keeps_json_valid():
    out = {"hits": [{"id": i, "pad": "x" * 200} for i in range(200)], "probe": {}}
    clipped = json.loads(PL._clip(out, limit=3000))
    assert clipped["truncated"] and 0 < len(clipped["hits"]) < 200


# ------------------------------------------------------------------ structured citations
def _hit_and_harness():
    h, ENG = harness()
    h.start("t")
    hit = h.call("search_corpus", {"query": "burn scar", "modality": "sat", "k": 1})["hits"][0]
    return h, hit


def test_parse_claims():
    got = H.parse_claims("see #12[ndvi=0.42, Region=malibu] and #7[gc = 0.5] but not #9 or [x=1]")
    assert got == [(12, "ndvi", "0.42"), (12, "region", "malibu"), (7, "gc", "0.5")]


def test_correct_claims_verified_at_stated_precision():
    h, hit = _hit_and_harness()
    r = h.finish(f"#{hit['id']}[ndvi={hit['ndvi']}, region={hit['region'].upper()}, date={hit['date']}, "
                 f"nbr={round(hit['nbr'], 1)}]")
    assert r["grounded"] and r["claims"]["verified"] == 4 and not r["claims"]["contradicted"]


def test_wrong_claim_is_contradicted_and_ungrounds_answer():
    h, hit = _hit_and_harness()
    r = h.finish(f"#{hit['id']}[ndvi={hit['ndvi'] + 0.2:.3f}, region={hit['region']}]")
    assert not r["grounded"] and r["ungrounded_ids"] == []
    (c,) = r["claims"]["contradicted"]
    assert c["id"] == hit["id"] and c["field"] == "ndvi" and c["actual"] == hit["ndvi"]


def test_unverifiable_claims_are_reported_not_failed():
    h, hit = _hit_and_harness()
    r = h.finish(f"#{hit['id']}[elevation=900] and #{hit['id'] + 1}[ndvi=0.1]")
    reasons = {u["reason"] for u in r["claims"]["unverifiable"]}
    assert reasons == {"field not in tool results", "item not returned by any tool"}
    assert r["claims"]["checked"] == 2 and not r["claims"]["contradicted"]


def test_inspect_results_feed_nested_claims():
    h, hit = _hit_and_harness()
    d = h.call("inspect_item", {"item_id": hit["id"]})
    nir = d["band_means"]["NIR"]
    r = h.finish(f"#{hit['id']}[band_means.nir={nir}]")
    assert r["claims"]["verified"] == 1 and r["grounded"]


def test_claude_wrong_claim_flagged_end_to_end():
    h, ENG = harness()
    hit = ENG.search("hepatocyte", modality="cell", k=1)["hits"][0]
    p, _ = scripted([tool_use("search_corpus", {"query": "hepatocyte", "modality": "cell", "k": 1}),
                     text(f"#{hit['id']}[top_marker=NOT_A_GENE]")])
    r = p.run(h, "q")
    assert not r["grounded"] and r["claims"]["contradicted"][0]["field"] == "top_marker"


def test_rule_planner_answers_carry_verified_claims():
    h, _ = harness()
    r = PL.RulePlanner().run(h, "wildfire burn scars near Malibu in 2025; also AAV capsid sequences; also hepatocytes")
    assert r["grounded"] and r["claims"]["verified"] >= 10
    assert not r["claims"]["contradicted"] and not r["claims"]["unverifiable"]


# ------------------------------------------------------------------ rule planner
def test_split_clauses():
    assert PL.split_clauses("A; B") == ["A", "B"]
    assert PL.split_clauses("burn scars near Malibu and also hepatocytes") == ["burn scars near Malibu", "hepatocytes"]
    assert PL.split_clauses("dna x plus cells y") == ["dna x", "cells y"]
    assert PL.split_clauses("first and then second") == ["first", "second"]
    assert PL.split_clauses("one also two") == ["one", "two"]
    assert PL.split_clauses("  ;  ; ") == []
    assert PL.split_clauses("a single clause") == ["a single clause"]


def test_rule_planner_routes_clauses_to_modalities():
    h, _ = harness()
    r = PL.RulePlanner().run(h, "AAV capsid sequences; also hepatocytes expressing ALB")
    searches = [json.loads(l) for l in open(r["trace"]) if '"tool": "search_corpus"' in l]
    assert [(s["args"]["modality"], s["args"]["query"]) for s in searches] == [
        ("dna", "AAV capsid sequences"), ("cell", "hepatocytes expressing ALB")]
    assert r["grounded"] and h.trace.closed


def test_rule_planner_applies_place_and_year_filters():
    h, _ = harness()
    r = PL.RulePlanner().run(h, "wildfire burn scars near Malibu in 2025")
    s = next(json.loads(l) for l in open(r["trace"]) if '"tool": "search_corpus"' in l)
    assert s["args"]["modality"] == "sat" and s["args"]["place"] == "malibu"
    assert (s["args"]["date_from"], s["args"]["date_to"]) == ("2025-01-01", "2025-12-31")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
