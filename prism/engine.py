"""Query engine: encoder + sharded index + columnar catalog + raw re-fetch for provenance."""
from __future__ import annotations
import json
import re
from functools import lru_cache
import numpy as np
from . import synth
from .encoders import PrismEncoder
from .index import ShardedIVF

# (lat_min, lat_max, lon_min, lon_max)
PLACES = {
    "malibu": (33.95, 34.15, -118.95, -118.55),
    "santa monica bay": (33.70, 34.05, -118.90, -118.40),
    "angeles national forest": (34.15, 34.60, -118.45, -117.65),
    "los angeles": (33.70, 34.35, -118.70, -117.90),
    "mojave": (34.20, 35.80, -118.00, -115.00),
    "imperial valley": (32.50, 33.50, -116.10, -115.00),
    "southern california": (32.50, 35.80, -120.50, -114.50),
}
_ALIASES = {"socal": "southern california", "la": "los angeles", "mojave desert": "mojave",
            "angeles forest": "angeles national forest", "anf": "angeles national forest"}
_BY_AREA = sorted(PLACES, key=lambda p: (PLACES[p][1] - PLACES[p][0]) * (PLACES[p][3] - PLACES[p][2]))
TODAY = np.datetime64("2026-10-17")


_PLACE_RX = re.compile(r"\b(?:in|near|around|at|over)?\s*(?:the\s+)?(?:" +
                       "|".join(sorted(map(re.escape, list(PLACES) + list(_ALIASES)), key=len, reverse=True)) + r")\b", re.I)
_DATE_RX = re.compile(r"\b(?:in|since|after|during|from)?\s*20\d\d\b", re.I)


def semantic_part(query):
    """Drop place/date phrases: they are structured filters, not semantics (keeps 'Angeles National
    Forest' from pulling a burn-scar query toward the forest class)."""
    q = _DATE_RX.sub(" ", _PLACE_RX.sub(" ", query))
    return re.sub(r"\s+", " ", q).strip() or query


def resolve_place(name):
    if not name:
        return None
    key = _ALIASES.get(name.lower().strip(), name.lower().strip())
    return PLACES.get(key)


def region_of(lat, lon):
    if not np.isfinite(lat):
        return None
    for p in _BY_AREA:
        b = PLACES[p]
        if b[0] <= lat <= b[1] and b[2] <= lon <= b[3]:
            return p
    return "other"


def to_day(s):
    return int((np.datetime64(s) - synth.DAY0).astype(int))


def day_str(d):
    return str(synth.DAY0 + np.timedelta64(int(d), "D"))


class Engine:
    def __init__(self, index_dir, model_path, mmap=True):
        self.enc = PrismEncoder.load(model_path)
        self.ix = ShardedIVF.load(index_dir, mmap=mmap)
        self.cat = dict(np.load(f"{index_dir}/catalog.npz"))
        self.plan = json.load(open(f"{index_dir}/plan.json"))
        n = len(self.cat["mod"])
        self._shard = np.full(n, -1, np.int16); self._pos = np.full(n, -1, np.int64)
        for si, sh in enumerate(self.ix.shards):
            ids = np.asarray(sh.meta["ids"])
            self._shard[ids] = si; self._pos[ids] = np.arange(len(ids))

    def __len__(self):
        return len(self.ix)

    # ------------------------------------------------------------------ tools
    def search(self, query, modality=None, k=10, place=None, bbox=None, date_from=None, date_to=None, nprobe=32):
        q = self.enc.encode_text([semantic_part(query)])
        return self._search_vec(q, modality, k, place, bbox, date_from, date_to, nprobe)

    def _check_id(self, item_id):
        if isinstance(item_id, bool) or not isinstance(item_id, (int, np.integer)):
            raise ValueError(f"item_id must be an integer, got {item_id!r}")
        i = int(item_id)
        if not (0 <= i < len(self.cat["mod"])) or self.cat["mod"][i] < 0 or self._shard[i] < 0:
            raise ValueError(f"no item {item_id}")
        return i

    def similar(self, item_id, modality=None, k=10, nprobe=32):
        item_id = self._check_id(item_id)
        v = self.vector(item_id)
        hits = self._search_vec(v[None], modality, k + 1, None, None, None, None, nprobe)
        hits["hits"] = [h for h in hits["hits"] if h["id"] != item_id][:k]
        return hits

    def _search_vec(self, q, modality, k, place, bbox, date_from, date_to, nprobe):
        filters = {}
        if modality:
            mods = [modality] if isinstance(modality, str) else list(modality)
            filters["modality"] = [synth.MOD_CODE[m] for m in mods]
        box = bbox or resolve_place(place)
        if place and box is None:
            raise ValueError(f"unknown place '{place}'. Known: {sorted(PLACES)}")
        if box:
            filters["bbox"] = tuple(box)
        if date_from or date_to:
            filters["time"] = (to_day(date_from) if date_from else 0, to_day(date_to) if date_to else 10 ** 6)
        s, i, st = self.ix.search(q, k=min(int(k), 100), nprobe=nprobe, filters=filters or None)
        return {"filters": {k_: list(v) if isinstance(v, tuple) else v for k_, v in filters.items()},
                "probe": st[0], "hits": [self.card(int(x), float(sc)) for x, sc in zip(i[0], s[0])]}

    def card(self, i, score=None):
        c = self.cat; mod = synth.MODALITIES[c["mod"][i]]
        d = {"id": i, "modality": mod, "date": day_str(c["t"][i]), "uri": self.uri(i)}
        if score is not None:
            d["score"] = round(score, 4)
        if mod == "sat":
            d.update(lat=round(float(c["lat"][i]), 4), lon=round(float(c["lon"][i]), 4),
                     region=region_of(c["lat"][i], c["lon"][i]),
                     ndvi=round(float(c["s1"][i]), 3), nbr=round(float(c["s2"][i]), 3),
                     cloud_frac=round(float(c["s3"][i]), 3))
        elif mod == "dna":
            d.update(gc=round(float(c["s1"][i]), 3), length=int(c["s2"][i]))
        else:
            d.update(total_counts=int(c["s1"][i]), genes_detected=int(c["s2"][i]),
                     top_marker=synth.MARKER_GENES[int(c["s3"][i])])
        return d

    def inspect(self, item_id):
        i = self._check_id(item_id)
        d = self.card(i); raw = self.fetch_raw(i)
        if d["modality"] == "sat":
            d["band_means"] = {b: round(float(v), 3) for b, v in zip(synth.BANDS, raw.reshape(-1, 5).mean(0))}
            d["note"] = "NBR < 0.1 with SWIR > NIR is consistent with recent burn; NDVI > 0.5 dense vegetation"
        elif d["modality"] == "dna":
            d["sequence_head"] = synth.dna_to_str(raw[:60])
        else:
            norm = np.log1p(raw / max(raw.sum(), 1) * 1e4)
            top = np.argsort(-norm[:len(synth.MARKER_GENES)])[:5]
            d["top_marker_genes"] = {synth.GENES[g]: round(float(norm[g]), 2) for g in top}
        return d

    def aggregate(self, item_ids, group_by="region"):
        ids = np.asarray([self._check_id(x) for x in item_ids], np.int64)
        c = self.cat; groups = {}
        for i in ids:
            if group_by == "region":
                g = region_of(c["lat"][i], c["lon"][i]) or "n/a"
            elif group_by == "month":
                g = day_str(c["t"][i])[:7]
            elif group_by == "modality":
                g = synth.MODALITIES[c["mod"][i]]
            else:
                raise ValueError("group_by must be region|month|modality")
            groups.setdefault(g, []).append(i)
        out = {}
        for g, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            m = np.array(members)
            out[g] = {"count": len(m), "mean_s1": round(float(np.nanmean(c["s1"][m])), 3),
                      "mean_s2": round(float(np.nanmean(c["s2"][m])), 3)}
        return {"group_by": group_by, "groups": out,
                "fields": "sat: s1=NDVI s2=NBR | dna: s1=GC | cell: s1=total counts"}

    # ------------------------------------------------------------------ provenance
    def uri(self, i):
        bs = self.plan["bs"]; mod = synth.MODALITIES[self.cat["mod"][i]]
        return f"s3://prism-demo/{mod}/batch-{i // bs:06d}/item-{i % bs:04d}"

    def vector(self, i):
        sh = self.ix.shards[self._shard[i]]
        return np.asarray(sh.q8[self._pos[i]], np.float32) * self.ix.scale

    def fetch_raw(self, i):
        bs = self.plan["bs"]; b = i // bs
        return _fetch_batch(self.plan["mods"][b], bs, self.plan["seed"] * 1_000_003 + b)[i % bs]


@lru_cache(maxsize=16)
def _fetch_batch(mod, n, seed):
    return synth.GENERATORS[mod](n, seed)[0]
