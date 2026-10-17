"""Auditable tool-using agent over the multimodal corpus.

Harness properties (what an enterprise deployment needs, not just a demo loop):
  * typed tool registry with JSON-schema args, validated before execution
  * policy: tool allowlist, step budget, wall-clock budget, k caps
  * every step logged (args, latency, result hash) to an append-only JSONL trace
  * grounding check: every item id cited in the final answer must have been
    returned by a tool in this run, otherwise the answer is flagged
  * planner is pluggable: Claude via the Messages API when ANTHROPIC_API_KEY is set,
    otherwise a deterministic rule planner (works offline, reproducible demos)
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from . import engine as E

TOOLS = [
    {"name": "search_corpus",
     "description": "Semantic search across satellite tiles (sat), DNA sequences (dna) and single-cell "
                    "transcriptomes (cell) with one natural-language query. Optional filters.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"},
         "modality": {"type": "string", "enum": ["sat", "dna", "cell"]},
         "place": {"type": "string", "description": f"one of {sorted(E.PLACES)}"},
         "date_from": {"type": "string", "description": "YYYY-MM-DD"},
         "date_to": {"type": "string", "description": "YYYY-MM-DD"},
         "k": {"type": "integer", "minimum": 1, "maximum": 50}},
         "required": ["query"]}},
    {"name": "find_similar",
     "description": "Nearest neighbours of an existing item, optionally in another modality.",
     "input_schema": {"type": "object", "properties": {
         "item_id": {"type": "integer"}, "modality": {"type": "string", "enum": ["sat", "dna", "cell"]},
         "k": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["item_id"]}},
    {"name": "inspect_item",
     "description": "Re-fetch one item from source and return measured properties (spectral indices, "
                    "sequence head, marker genes) plus its source URI.",
     "input_schema": {"type": "object", "properties": {"item_id": {"type": "integer"}}, "required": ["item_id"]}},
    {"name": "aggregate",
     "description": "Group item ids by region, month or modality with summary statistics.",
     "input_schema": {"type": "object", "properties": {
         "item_ids": {"type": "array", "items": {"type": "integer"}},
         "group_by": {"type": "string", "enum": ["region", "month", "modality"]}},
         "required": ["item_ids"]}},
]
_SCHEMA = {t["name"]: t["input_schema"] for t in TOOLS}
_PY = {"string": str, "integer": int, "array": list}


# Matches '#123', 'item 123', 'items #123', 'ID: 123', 'id=123' (case-insensitive).
_CITE_RX = re.compile(r"(?:#|\b(?:item|id)s?\b[\s:=#]*)(\d+)", re.I)
# Structured citation: '#123[ndvi=0.42, region=malibu]' -- field claims checked against the tool results.
_CLAIM_RX = re.compile(r"#(\d+)\s*\[([^\]]*)\]")
_NOT_FACTS = {"score", "note"}
_MIN_TOL = 0.5e-3  # catalog cards round floats to 3 decimals


def parse_claims(answer):
    """-> [(item_id, field, raw_value)] from every '#id[field=value, ...]' citation."""
    out = []
    for m in _CLAIM_RX.finditer(answer):
        for part in m.group(2).split(","):
            k, sep, v = part.partition("=")
            if sep and k.strip() and v.strip():
                out.append((int(m.group(1)), k.strip().lower(), v.strip().strip("\"'")))
    return out


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}".lower()
        if isinstance(v, dict):
            yield from _flatten(v, key + ".")
        elif not isinstance(v, (list, tuple)) and k not in _NOT_FACTS:
            yield key, v


def claim_matches(actual, raw):
    """Numbers match at the precision the claim states (never tighter than the catalog's rounding); text matches case-insensitively."""
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            claimed = float(raw)
        except ValueError:
            return False
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        return abs(actual - claimed) <= max(0.5 * 10 ** -decimals, _MIN_TOL) + 1e-9
    return str(actual).strip().lower() == raw.strip().lower()


class PolicyError(Exception):
    pass


class Harness:
    def __init__(self, eng: E.Engine, trace_dir="traces", max_steps=16, budget_s=60, allow=None):
        self.eng, self.trace_dir, self.max_steps, self.budget_s = eng, trace_dir, max_steps, budget_s
        self.allow = set(allow or _SCHEMA)
        os.makedirs(trace_dir, exist_ok=True)

    def start(self, question):
        self.run_id = uuid.uuid4().hex[:12]; self.t0 = time.perf_counter(); self.steps = 0
        self.seen_ids = set(); self.facts = {}; self.trace = open(f"{self.trace_dir}/{self.run_id}.jsonl", "w")
        self._log({"event": "start", "question": question})

    def _log(self, rec):
        rec = {"run": self.run_id, "ts": round(time.perf_counter() - self.t0, 4), **rec}
        self.trace.write(json.dumps(rec, default=str) + "\n"); self.trace.flush()

    def validate(self, name, args):
        if name not in self.allow:
            raise PolicyError(f"tool '{name}' not allowed")
        sch = _SCHEMA[name]
        for r in sch.get("required", []):
            if r not in args:
                raise PolicyError(f"{name}: missing '{r}'")
        for k, v in args.items():
            spec = sch["properties"].get(k)
            if spec is None:
                raise PolicyError(f"{name}: unexpected arg '{k}'")
            if not isinstance(v, _PY[spec["type"]]) or (spec["type"] == "integer" and isinstance(v, bool)):
                raise PolicyError(f"{name}.{k}: expected {spec['type']}")
            if spec["type"] == "array":
                item = spec.get("items", {}).get("type")
                if item and not all(isinstance(x, _PY[item]) and not isinstance(x, bool) for x in v):
                    raise PolicyError(f"{name}.{k}: items must be {item}")
                if len(v) > 1000:
                    raise PolicyError(f"{name}.{k}: too many items")
            if k in ("date_from", "date_to"):
                try:
                    E.to_day(v)
                except ValueError:
                    raise PolicyError(f"{name}.{k}: expected YYYY-MM-DD")
            if "minimum" in spec and v < spec["minimum"]:
                raise PolicyError(f"{name}.{k}: min {spec['minimum']}")
            if "enum" in spec and v not in spec["enum"]:
                raise PolicyError(f"{name}.{k}: must be one of {spec['enum']}")
            if "maximum" in spec and v > spec["maximum"]:
                raise PolicyError(f"{name}.{k}: max {spec['maximum']}")

    def call(self, name, args):
        self.steps += 1
        if self.steps > self.max_steps:
            raise PolicyError("step budget exhausted")
        if time.perf_counter() - self.t0 > self.budget_s:
            raise PolicyError("time budget exhausted")
        t = time.perf_counter()
        try:
            self.validate(name, args)
            if name == "search_corpus":
                out = self.eng.search(**args)
            elif name == "find_similar":
                out = self.eng.similar(**args)
            elif name == "inspect_item":
                out = self.eng.inspect(args["item_id"])
            else:
                out = self.eng.aggregate(args["item_ids"], args.get("group_by", "region"))
            ok = True
        except (PolicyError, ValueError, KeyError) as e:
            out, ok = {"error": str(e)}, False
        self._collect(out)
        blob = json.dumps(out, default=str)
        self._log({"event": "tool", "step": self.steps, "tool": name, "args": args, "ok": ok,
                   "ms": round((time.perf_counter() - t) * 1e3, 2),
                   "result_sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
                   "n_hits": len(out.get("hits", [])) if isinstance(out, dict) else None})
        return out

    def _collect(self, out):
        """Remember which ids tools returned and the measured fields they returned for each."""
        if isinstance(out, dict):
            for item in ([out] if "id" in out else []) + list(out.get("hits", [])):
                self.seen_ids.add(int(item["id"]))
                self.facts.setdefault(int(item["id"]), {}).update(_flatten(item))

    def check_claims(self, answer):
        verified, contradicted, unverifiable = 0, [], []
        for item_id, field, raw in parse_claims(answer):
            facts = self.facts.get(item_id)
            if facts is None:
                unverifiable.append({"id": item_id, "field": field, "reason": "item not returned by any tool"})
            elif field not in facts:
                unverifiable.append({"id": item_id, "field": field, "reason": "field not in tool results"})
            elif claim_matches(facts[field], raw):
                verified += 1
            else:
                contradicted.append({"id": item_id, "field": field, "claimed": raw, "actual": facts[field]})
        return {"checked": verified + len(contradicted) + len(unverifiable), "verified": verified,
                "contradicted": contradicted, "unverifiable": unverifiable}

    def close(self):
        """Release the trace file; safe to call more than once (e.g. after a failed run)."""
        trace = getattr(self, "trace", None)
        if trace is not None and not trace.closed:
            self._log({"event": "aborted"}); trace.close()

    def finish(self, answer):
        cited = {int(x) for x in _CITE_RX.findall(answer)}
        ungrounded = sorted(cited - self.seen_ids)
        claims = self.check_claims(answer)
        grounded = not ungrounded and not claims["contradicted"]  # a wrong stated value is as bad as a made-up id
        rec = {"event": "finish", "answer": answer, "cited": sorted(cited), "ungrounded": ungrounded,
               "claims": claims, "grounded": grounded, "steps": self.steps,
               "total_ms": round((time.perf_counter() - self.t0) * 1e3, 1)}
        self._log(rec); self.trace.close()
        return {"run_id": self.run_id, "answer": answer, "grounded": grounded,
                "ungrounded_ids": ungrounded, "claims": claims, "steps": self.steps, "total_ms": rec["total_ms"],
                "trace": f"{self.trace_dir}/{self.run_id}.jsonl"}


# ============================================================================ planners
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
