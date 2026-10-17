"""Index persistence and recall regression."""
import os
import warnings

import numpy as np
import pytest
from test_core import env

from prism.index import ShardedIVF, atomic_dir

RECALL_FLOOR = 0.90   # hard failure below this
RECALL_GOAL = 0.95    # warn below this


def test_recall_at_10_regression():
    D, _, ENG = env()
    stage = f"{D}/stage"
    V = np.empty((len(ENG.cat["mod"]), 256), np.float32)
    for f in sorted(os.listdir(stage)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz"):
            z = np.load(f"{stage}/{f}"); V[z["ids"]] = z["vec"]
    rng = np.random.default_rng(1)
    qids = rng.choice(len(V), 100, replace=False)
    Q = V[qids] + rng.normal(0, 0.05, (100, 256)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    truth = np.argpartition(-(Q @ V.T), 9, axis=1)[:, :10]
    recalls = []
    for i in range(len(Q)):
        _, ids, _ = ENG.ix.search(Q[i], k=10, nprobe=32)  # engine defaults: nprobe 32, rerank 8
        recalls.append(len(set(ids[0].tolist()) & set(truth[i].tolist())) / 10)
    recall = float(np.mean(recalls))
    assert recall >= RECALL_FLOOR, f"recall@10 {recall:.3f} fell below floor {RECALL_FLOOR}"
    if recall < RECALL_GOAL:
        warnings.warn(f"recall@10 {recall:.3f} is below the {RECALL_GOAL} goal", stacklevel=2)


def test_serial_and_threaded_search_agree():
    D, _, ENG = env()
    assert ENG.ix.pool is None  # serial is the default
    threaded = ShardedIVF.load(f"{D}/index", threads=3)
    assert threaded.pool is not None
    qs = ENG.enc.encode_text(["burn scar", "hepatocyte albumin", "AAV capsid"])
    for filters in (None, {"modality": [0]}, {"modality": [0], "time": (100, 600)}):
        for q in qs:
            a = ENG.ix.search(q, k=10, nprobe=32, filters=filters)
            b = threaded.search(q, k=10, nprobe=32, filters=filters)
            assert a[1][0].tolist() == b[1][0].tolist() and a[2] == b[2]


def test_save_load_roundtrip_includes_extra_files():
    D, _, ENG = env()
    path = f"{D}/roundtrip"
    ENG.ix.save(path, extra={"catalog.npz": {"x": np.arange(3)}})
    assert np.load(f"{path}/catalog.npz")["x"].tolist() == [0, 1, 2]
    ix = ShardedIVF.load(path)
    assert len(ix) == len(ENG.ix)
    assert not os.path.exists(path + ".tmp") and not os.path.exists(path + ".old")


def test_failed_save_leaves_previous_index_intact():
    D, _, ENG = env()
    path = f"{D}/atomic"
    ENG.ix.save(path)
    before = sorted(os.listdir(path))
    with pytest.raises(RuntimeError):
        with atomic_dir(path) as tmp:
            open(f"{tmp}/partial.npy", "w").write("half")
            raise RuntimeError("crash mid-write")
    assert sorted(os.listdir(path)) == before and len(ShardedIVF.load(path)) == len(ENG.ix)
    assert not os.path.exists(path + ".tmp")


def test_overwriting_save_replaces_atomically():
    D, _, ENG = env()
    path = f"{D}/overwrite"
    ENG.ix.save(path, extra={"old.npz": {"a": np.zeros(1)}})
    ENG.ix.save(path, extra={"new.npz": {"a": np.zeros(1)}})
    assert os.path.exists(f"{path}/new.npz") and not os.path.exists(f"{path}/old.npz")
