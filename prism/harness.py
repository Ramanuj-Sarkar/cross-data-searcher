"""Policy harness for the auditable tool-using agent over the multimodal corpus.

Planners (who decides which tool to call next) live in planners.py; this module is the part a deployment must trust.

Harness properties (what an enterprise deployment needs, not just a demo loop):
  * typed tool registry with JSON-schema args, validated before execution
  * policy: tool allowlist, step budget, wall-clock budget, k caps
  * every step logged (args, latency, result hash) to an append-only JSONL trace
  * grounding check: every item id cited in the final answer must have been
    returned by a tool in this run, otherwise the answer is flagged
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from typing import Any

from . import engine as E

TOOLS: list[dict[str, Any]] = [
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
_SCHEMA: dict[str, dict[str, Any]] = {t["name"]: t["input_schema"] for t in TOOLS}
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
            if spec["type"] == "array" and isinstance(v, list):
                item = spec.get("items", {}).get("type")
                if item and not all(isinstance(x, _PY[item]) and not isinstance(x, bool) for x in v):
                    raise PolicyError(f"{name}.{k}: items must be {item}")
                if len(v) > 1000:
                    raise PolicyError(f"{name}.{k}: too many items")
            if k in ("date_from", "date_to"):
                try:
                    E.to_day(v)
                except ValueError:
                    raise PolicyError(f"{name}.{k}: expected YYYY-MM-DD") from None
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
