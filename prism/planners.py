"""Planners that drive the Harness: a deterministic offline rule planner and a Claude tool-use planner.

  * planner is pluggable: Claude via the Messages API when ANTHROPIC_API_KEY is set,
    otherwise a deterministic rule planner (works offline, reproducible demos)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from . import engine as E
from .harness import TOOLS, Harness, PolicyError

_MOD_WORDS = {
    "sat": r"satellite|imagery|aerial|tile|wildfire|burn|scar|fire|ndvi|forest|vegetation|water|reservoir|urban|city|"
           r"cropland|farm|desert|cloud|land",
    "dna": r"dna|sequence|capsid|aav|plasmid|lentivir|crispr|cas9|phage|mitochondri|mtdna|genome|vector backbone",
    "cell": r"\bcells?\b|single.cell|transcriptom|scrna|express|marker|lymphocyte|hepatocyte|monocyte|neuron|"
            r"cardiomyocyte|stem cell|t cell|b cell|nk ",
}


def _year_range(text):
    m = re.search(r"\b(?:since|after)\s+(20\d\d)\b", text)
    if m:
        return f"{m.group(1)}-01-01", None
    m = re.search(r"\b(20\d\d)\b", text)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
    return None, None


def _place(text):
    t = text.lower()
    for p in sorted(list(E.PLACES) + list(E._ALIASES), key=len, reverse=True):
        if re.search(rf"\b{re.escape(p)}\b", t):
            return E._ALIASES.get(p, p)
    return None


def split_clauses(question):
    """Split a multi-intent question on ';', 'and also', 'and then', 'plus', 'also'."""
    return [c.strip() for c in re.split(r";|\band also\b|\band then\b|\bplus\b|\balso\b", question) if c.strip()]


class RulePlanner:
    """Deterministic planner: split the question into modality clauses, search each with
    filters, inspect the strongest hits, aggregate geo results, then compose a cited answer."""
    name = "rule"

    def run(self, h: Harness, question, k=8):
        h.start(question)
        try:
            return self._run(h, question, k)
        finally:
            h.close()

    def _run(self, h: Harness, question, k):
        clauses = split_clauses(question)
        jobs = []
        for c in clauses:
            mods = [m for m, rx in _MOD_WORDS.items() if re.search(rx, c.lower())]
            for m in mods or ["sat", "dna", "cell"]:
                jobs.append((c, m))
        sections = []
        for clause, mod in jobs:
            args = {"query": clause, "modality": mod, "k": k}
            if mod == "sat":
                place = _place(clause); d0, d1 = _year_range(clause)
                if place: args["place"] = place
                if d0: args["date_from"] = d0
                if d1: args["date_to"] = d1
            res = h.call("search_corpus", args)
            hits = res.get("hits", [])
            if not hits:
                sections.append(f"**{mod}** — no results for “{clause}” ({res.get('error', 'filters too narrow')}).")
                continue
            details = [h.call("inspect_item", {"item_id": x["id"]}) for x in hits[:2]]
            agg = h.call("aggregate", {"item_ids": [x["id"] for x in hits], "group_by": "region"}) if mod == "sat" else None
            sections.append(_section(mod, clause, args, res, details, agg))
        return h.finish("\n\n".join(sections))


def _fmt_hit(x):
    """Structured citation: every stated value is checked against the tool results by Harness.finish."""
    if x["modality"] == "sat":
        return f"#{x['id']}[region={x['region']}, date={x['date']}, ndvi={x['ndvi']}, nbr={x['nbr']}] (score {x['score']})"
    if x["modality"] == "dna":
        return f"#{x['id']}[gc={x['gc']}, length={x['length']}] (score {x['score']})"
    return f"#{x['id']}[top_marker={x['top_marker']}, genes_detected={x['genes_detected']}] (score {x['score']})"


def _section(mod, clause, args, res, details, agg):
    label = {"sat": "Satellite", "dna": "DNA", "cell": "Single-cell"}[mod]
    f = {k: v for k, v in args.items() if k in ("place", "date_from", "date_to")}
    lines = [f"**{label}** — “{clause}”" + (f" · filters {f}" if f else "") +
             f" · probed {res['probe']['nprobe']} lists / scanned {res['probe']['scanned']:,} vectors"]
    lines.append("Top hits: " + "; ".join(_fmt_hit(x) for x in res["hits"][:5]))
    for d in details:
        if mod == "sat":
            lines.append(f"Inspected #{d['id']}: bands {d['band_means']} → {d['uri']}")
        elif mod == "dna":
            lines.append(f"Inspected #{d['id']}: {d['sequence_head']}… → {d['uri']}")
        else:
            lines.append(f"Inspected #{d['id']}: {d['top_marker_genes']} → {d['uri']}")
    if agg:
        g = ", ".join(f"{k}: {v['count']} (mean NBR {v['mean_s2']})" for k, v in agg["groups"].items())
        lines.append(f"By region: {g}")
    return "\n".join(lines)


class ClaudePlanner:
    """Claude drives the tool loop through the Messages API (no SDK dependency)."""
    name = "claude"
    SYSTEM = ("You are an analyst agent over a multimodal corpus (satellite tiles, DNA sequences, single-cell "
              "transcriptomes). Use the tools to answer. Cite every item you mention as #<id>[field=value, ...], "
              "e.g. #12[ndvi=0.42, region=malibu], listing the measured fields you rely on; values must match the "
              "tool results exactly and only ids returned by tools may be cited. Inspect items before making claims about their properties. Be concise. "
              "Today is {today}.")

    def __init__(self, model=None, api_key=None):
        self.model = model or os.environ.get("PRISM_MODEL", "claude-sonnet-5-5")
        self.key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def _post(self, body):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                     headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 529) or attempt == 2:
                    raise RuntimeError(f"Anthropic API error {e.code}") from None
            except urllib.error.URLError as e:
                if attempt == 2:
                    raise RuntimeError(f"Anthropic API unreachable: {e.reason}") from None
            time.sleep(2 ** attempt)

    def run(self, h: Harness, question):
        h.start(question)
        try:
            return self._run(h, question)
        finally:
            h.close()

    def _run(self, h: Harness, question):
        msgs = [{"role": "user", "content": question}]
        while True:
            resp = self._post({"model": self.model, "max_tokens": 1500, "system": self.SYSTEM.format(today=h.eng.today),
                               "tools": TOOLS, "messages": msgs})
            msgs.append({"role": "assistant", "content": resp["content"]})
            uses = [b for b in resp["content"] if b["type"] == "tool_use"]
            if resp.get("stop_reason") != "tool_use" or not uses:
                text = "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text")
                return h.finish(text)
            results = []
            for u in uses:
                try:
                    out = h.call(u["name"], u["input"])
                except PolicyError as e:
                    return h.finish(f"[stopped by policy: {e}]")
                results.append({"type": "tool_result", "tool_use_id": u["id"],
                                "is_error": isinstance(out, dict) and "error" in out,
                                "content": _clip(out)})
            msgs.append({"role": "user", "content": results})


def _clip(out, limit=12000):
    """Serialize a tool result, dropping trailing hits (not bytes) so the JSON stays valid."""
    blob = json.dumps(out, default=str)
    if len(blob) <= limit or not isinstance(out, dict) or "hits" not in out:
        return blob[:limit]
    out = dict(out); hits = list(out["hits"])
    while hits and len(json.dumps({**out, "hits": hits, "truncated": True}, default=str)) > limit:
        hits.pop()
    return json.dumps({**out, "hits": hits, "truncated": True}, default=str)


def default_planner():
    return ClaudePlanner() if os.environ.get("ANTHROPIC_API_KEY") else RulePlanner()
