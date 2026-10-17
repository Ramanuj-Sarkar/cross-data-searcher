"""python -m pytest -q  (or: python tests/test_core.py)"""
import os
import sys
import tempfile
from functools import lru_cache
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism import pipeline as P, synth, agent as A
from prism.encoders import PrismEncoder
from prism.engine import Engine

@lru_cache(maxsize=1)
def env():
    """Built lazily: spawned workers re-import the main module, so nothing heavy may run at import time."""
    d = tempfile.mkdtemp()
    PrismEncoder().fit(n_per_class=200, log=None).save(f"{d}/model.npz")
    descs = P.plan(16384, 512)
    P.ingest(descs, f"{d}/model.npz", f"{d}/stage", workers=2, log=None)
    P.build(f"{d}/stage", f"{d}/index", n_shards=3, log=lambda *_: None); P.save_plan(f"{d}/index", descs)
    return d, descs, Engine(f"{d}/index", f"{d}/model.npz")


def test_resume_is_idempotent():
    D, DESCS, ENG = env()
    s = P.ingest(DESCS, f"{D}/model.npz", f"{D}/stage", workers=2, log=None)
    assert s["items"] == 0


def test_every_item_indexed_once():
    D, DESCS, ENG = env()
    assert len(ENG) == 16384
    ids = np.concatenate([np.asarray(s.meta["ids"]) for s in ENG.ix.shards])
    assert len(np.unique(ids)) == len(ids)


def test_filters_respected():
    D, DESCS, ENG = env()
    r = ENG.search("burn scar", modality="sat", place="malibu", date_from="2025-01-01", date_to="2025-12-31", k=20)
    for h in r["hits"]:
        assert h["modality"] == "sat" and h["region"] == "malibu" and h["date"].startswith("2025")


def test_cross_modal_text_retrieval():
    D, DESCS, ENG = env()
    for mod, cname, q in [("sat", "water", "lake reservoir water"), ("dna", "crispr_cas9", "CRISPR Cas9"),
                          ("cell", "neuron", "SNAP25 neurons")]:
        ids = [h["id"] for h in ENG.search(q, modality=mod, k=10)["hits"]]
        assert np.mean(ENG.cat["label"][ids] == synth.CLASSES[mod].index(cname)) >= 0.8


def test_refetch_matches_index():
    D, DESCS, ENG = env()
    i = ENG.search("hepatocyte", modality="cell", k=1)["hits"][0]["id"]
    raw = ENG.fetch_raw(i)
    v, _ = ENG.enc.encode("cell", raw[None])
    assert float(v[0] @ ENG.vector(i)) > 0.98  # int8 storage round-trip


def test_policy_and_grounding():
    D, DESCS, ENG = env()
    h = A.Harness(ENG, trace_dir=f"{D}/traces"); h.start("t")
    assert "error" in h.call("search_corpus", {"query": "x", "k": 500})
    assert "error" in h.call("delete_everything", {})
    r = h.finish("see #999999999")
    assert not r["grounded"] and r["ungrounded_ids"] == [999999999]


def test_place_words_do_not_leak_into_semantics():
    D, DESCS, ENG = env()
    r = ENG.search("burned areas in the Angeles National Forest", modality="sat", place="angeles national forest", k=10)
    assert np.mean(ENG.cat["label"][[h["id"] for h in r["hits"]]] == synth.SAT_CLASSES.index("burn_scar")) >= 0.8


def test_similar_rejects_bad_ids():
    D, DESCS, ENG = env()
    for bad in (-1, 10 ** 9, True, "5"):
        try:
            ENG.similar(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_schema_edge_cases():
    D, DESCS, ENG = env()
    h = A.Harness(ENG, trace_dir=f"{D}/traces"); h.start("t")
    assert "error" in h.call("search_corpus", {"query": "x", "k": 0})
    assert "error" in h.call("inspect_item", {"item_id": True})
    assert "error" in h.call("search_corpus", {"query": "x", "date_from": "not-a-date"})
    assert "error" in h.call("aggregate", {"item_ids": [1, "a"]})
    h.close()


def test_grounding_id_formats():
    D, DESCS, ENG = env()
    h = A.Harness(ENG, trace_dir=f"{D}/traces"); h.start("t")
    r = h.finish("Found #11, item 22, Item #33, ID: 44 and ids=55; no id here.")
    assert r["ungrounded_ids"] == [11, 22, 33, 44, 55]


def test_today_defaults_to_catalog_max_with_override():
    D, DESCS, ENG = env()
    valid = ENG.cat["mod"] >= 0
    assert ENG.today == str(synth.DAY0 + np.timedelta64(int(ENG.cat["t"][valid].max()), "D"))
    assert Engine(f"{D}/index", f"{D}/model.npz", today="2030-01-02").today == "2030-01-02"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
