"""python -m pytest -q  (or: python tests/test_core.py)"""
import os
import sys
import tempfile
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism import harness as H
from prism import pipeline as P
from prism import synth
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
        labels = ENG.cat["label"][ids]
        # Sibling classes are deliberately confusable (shared markers/motifs, label noise): require the right
        # family >= 0.8 and the exact class to still dominate.
        family = [synth.CLASSES[mod].index(c) for c in {cname, *synth._CONFUSABLE[mod].get(cname, [])}]
        assert np.mean(np.isin(labels, family)) >= 0.8
        assert np.mean(labels == synth.CLASSES[mod].index(cname)) >= 0.5


def test_refetch_matches_index():
    D, DESCS, ENG = env()
    i = ENG.search("hepatocyte", modality="cell", k=1)["hits"][0]["id"]
    raw = ENG.fetch_raw(i)
    v, _ = ENG.enc.encode("cell", raw[None])
    assert float(v[0] @ ENG.vector(i)) > 0.98  # int8 storage round-trip


def test_policy_and_grounding():
    D, DESCS, ENG = env()
    h = H.Harness(ENG, trace_dir=f"{D}/traces"); h.start("t")
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
    h = H.Harness(ENG, trace_dir=f"{D}/traces"); h.start("t")
    assert "error" in h.call("search_corpus", {"query": "x", "k": 0})
    assert "error" in h.call("inspect_item", {"item_id": True})
    assert "error" in h.call("search_corpus", {"query": "x", "date_from": "not-a-date"})
    assert "error" in h.call("aggregate", {"item_ids": [1, "a"]})
    h.close()


def test_grounding_id_formats():
    D, DESCS, ENG = env()
    h = H.Harness(ENG, trace_dir=f"{D}/traces"); h.start("t")
    r = h.finish("Found #11, item 22, Item #33, ID: 44 and ids=55; no id here.")
    assert r["ungrounded_ids"] == [11, 22, 33, 44, 55]


def test_today_defaults_to_catalog_max_with_override():
    D, DESCS, ENG = env()
    valid = ENG.cat["mod"] >= 0
    assert ENG.today == str(synth.DAY0 + np.timedelta64(int(ENG.cat["t"][valid].max()), "D"))
    assert Engine(f"{D}/index", f"{D}/model.npz", today="2030-01-02").today == "2030-01-02"


def test_trace_retention_keeps_newest_30():
    import tempfile
    d = tempfile.mkdtemp()
    for n in range(40):  # run n was written n seconds after the epoch -> run 39 is newest
        path = f"{d}/{n:012x}.jsonl"
        open(path, "w").write("{}\n"); os.utime(path, (1_000 + n, 1_000 + n))
    open(f"{d}/notes.txt", "w").write("not a trace")
    open(f"{d}/{'0' * 12}.jsonl.bak", "w").write("not a trace either")
    _, _, ENG = env()
    h = H.Harness(ENG, trace_dir=d); h.start("t"); h.close()
    runs = sorted(f for f in os.listdir(d) if H._TRACE_RX.match(f))
    assert len(runs) == 30                                    # 29 old + the one just started
    assert f"{39:012x}.jsonl" in runs and f"{11:012x}.jsonl" in runs and f"{10:012x}.jsonl" not in runs
    assert os.path.exists(f"{d}/{h.run_id}.jsonl") and os.path.exists(f"{d}/notes.txt")
    assert os.path.exists(f"{d}/{'0' * 12}.jsonl.bak")


def test_trace_retention_can_be_disabled():
    import tempfile
    d = tempfile.mkdtemp()
    for n in range(35):
        open(f"{d}/{n:012x}.jsonl", "w").write("{}\n")
    _, _, ENG = env()
    h = H.Harness(ENG, trace_dir=d, keep_traces=None); h.start("t"); h.close()
    assert len([f for f in os.listdir(d) if H._TRACE_RX.match(f)]) == 36


def test_chunk_is_identical_to_the_same_slice_of_the_batch():
    for mod, gen in synth.GENERATORS.items():
        for n in (synth.CHUNK * 3, synth.CHUNK * 2 + 17):  # incl. a ragged final chunk
            raw, labels, meta = gen(n, 99)
            for c in range(-(-n // synth.CHUNK)):
                r2, l2, m2 = synth.gen_chunk(mod, n, 99, c)
                lo = c * synth.CHUNK
                assert np.array_equal(raw[lo:lo + len(r2)], r2, equal_nan=True) and np.array_equal(labels[lo:lo + len(l2)], l2)
                assert all(np.array_equal(meta[k][lo:lo + len(r2)], m2[k], equal_nan=True) for k in meta)


def test_fetch_raw_never_regenerates_a_batch():
    D, DESCS, ENG = env()
    from prism import engine
    engine._fetch_chunk.cache_clear()
    saved = dict(synth.GENERATORS)
    synth.GENERATORS.clear()  # any batch-level regeneration would now raise KeyError
    try:
        for i in (0, 5, 129, len(ENG) - 1):
            raw = ENG.fetch_raw(i)
            assert not raw.flags.writeable
            assert ENG.inspect(i)["id"] == i
    finally:
        synth.GENERATORS.update(saved)
    assert engine._fetch_chunk.cache_info().maxsize == 32


def test_dna_chains_are_per_class():
    labels = np.repeat(np.arange(len(synth.DNA_CLASSES)), 40)
    seqs = synth.gen_dna(len(labels), 3, labels=labels)[0]
    gc = ((seqs == 1) | (seqs == 2)).mean(1)
    assert gc[labels == synth.DNA_CLASSES.index("aav_capsid")].mean() > gc[labels == synth.DNA_CLASSES.index("human_mito")].mean()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
