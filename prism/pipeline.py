"""Streaming, resumable, parallel ingest.

  plan  -> batch descriptors (uri-like: modality, batch_id, n, seed)   [tiny, cheap to ship]
  map   -> worker process: fetch(descriptor) -> featurize -> encode     [CPU-bound, scales out]
  sink  -> staged per-batch .npz + manifest line (idempotent, crash-safe)
  build -> train IVF on a sample, route into shards, persist + columnar catalog

Workers only receive descriptors, never raw bytes, so the same code runs on Ray /
Dask / k8s Jobs by swapping the Pool. Resume: batches already in the manifest are skipped.
"""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
import numpy as np
from . import synth
from .encoders import PrismEncoder
from .index import ShardedIVF

_ENC = None


def _init(model_path):
    global _ENC
    _ENC = PrismEncoder.load(model_path)


def _work(desc):
    t0 = time.perf_counter()
    mod, bid, n, seed = desc["mod"], desc["batch"], desc["n"], desc["seed"]
    raw, labels, meta = synth.GENERATORS[mod](n, seed)          # stand-in for an object-store read
    t1 = time.perf_counter()
    vec, summ = _ENC.encode(mod, raw)
    t2 = time.perf_counter()
    ids = bid * desc["bs"] + np.arange(n, dtype=np.int64)
    return {"batch": bid, "mod": mod, "vec": vec.astype(np.float16), "ids": ids, "labels": labels,
            "lat": meta["lat"], "lon": meta["lon"], "t": meta["t"],
            **{k: v.astype(np.float32) for k, v in summ.items()},
            "fetch_s": t1 - t0, "encode_s": t2 - t1}


def plan(n_items, batch_size=1024, mix=(0.4, 0.3, 0.3), seed=2026):
    rng = np.random.default_rng(seed)
    nb = n_items // batch_size
    mods = rng.choice(synth.MODALITIES, nb, p=list(mix))
    return [{"mod": str(m), "batch": b, "n": batch_size, "bs": batch_size, "seed": seed * 1_000_003 + b}
            for b, m in enumerate(mods)]


def _stage(stage_dir, r):
    """Atomic write: a crash mid-write can never leave a truncated batch file behind."""
    final = f"{stage_dir}/b{r['batch']:06d}.npz"
    tmp = final + ".tmp.npz"
    np.savez(tmp, mod=np.array(r["mod"]), **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
    os.replace(tmp, final)


def ingest(descs, model_path, stage_dir, workers=2, log=print, every=50, stall_timeout=180):
    """Workers are *spawned*, not forked. Forking after the parent has run large BLAS calls is unsafe
    on macOS (Accelerate / GCD state is not fork-safe): big GEMMs hang or crash in the child while
    tiny ones work. ProcessPoolExecutor also raises when a worker dies, where Pool waits forever."""
    os.makedirs(stage_dir, exist_ok=True)
    log = log or (lambda *_: None)
    man_path = f"{stage_dir}/manifest.jsonl"
    done = set()
    if os.path.exists(man_path):
        for l in open(man_path):
            try:
                done.add(json.loads(l)["batch"])
            except (ValueError, KeyError):  # torn last line from a hard kill
                pass
    todo = [d for d in descs if d["batch"] not in done]
    log(f"ingest: {len(descs)} batches planned, {len(done)} already staged, {len(todo)} to run, {workers} workers")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")  # inherited by spawned workers: 1 BLAS thread each, no oversubscription
    t0 = time.perf_counter(); n_items = 0; per_mod = {}; lat = []; n_done = 0
    it = iter(todo); pending = set()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(model_path,)) as ex, open(man_path, "a") as man:
        def refill():
            while len(pending) < workers * 2:
                d = next(it, None)
                if d is None:
                    return
                pending.add(ex.submit(_work, d))
        refill()
        while pending:
            finished, _ = wait(pending, timeout=stall_timeout, return_when=FIRST_COMPLETED)
            if not finished:
                raise RuntimeError(f"ingest stalled: no batch finished in {stall_timeout}s "
                                   f"({n_done}/{len(todo)} done). Re-run the same command to resume.")
            for fut in finished:
                pending.discard(fut)
                try:
                    r = fut.result()
                except BrokenProcessPool as e:
                    raise RuntimeError(f"an ingest worker died after {n_done}/{len(todo)} batches "
                                       f"(crash or out-of-memory). Re-run the same command to resume.") from e
                _stage(stage_dir, r)
                man.write(json.dumps({"batch": r["batch"], "mod": r["mod"], "n": len(r["ids"])}) + "\n")
                man.flush()
                n_done += 1
                n_items += len(r["ids"]); per_mod[r["mod"]] = per_mod.get(r["mod"], 0) + len(r["ids"])
                lat.append(r["fetch_s"] + r["encode_s"])
                if n_done % every == 0:
                    el = time.perf_counter() - t0
                    log(f"  {n_done:>5}/{len(todo)} batches  {n_items:>9,} items  {n_items / el:>8,.0f} items/s")
            refill()
    el = time.perf_counter() - t0
    stats = {"items": n_items, "seconds": el, "items_per_s": n_items / max(el, 1e-9), "per_modality": per_mod,
             "batch_latency_p50_ms": float(np.percentile(lat, 50) * 1e3) if lat else None,
             "batch_latency_p95_ms": float(np.percentile(lat, 95) * 1e3) if lat else None, "workers": workers}
    return stats


def build(stage_dir, index_dir, n_shards=4, nlist=None, log=print):
    t0 = time.perf_counter()
    files = sorted(f for f in os.listdir(stage_dir) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    parts = [dict(np.load(f"{stage_dir}/{f}")) for f in files]
    N = sum(len(p["ids"]) for p in parts)
    nlist = nlist or int(2 ** round(np.log2(max(64, 4 * np.sqrt(N)))))
    rng = np.random.default_rng(0)
    sample = np.concatenate([p["vec"][rng.random(len(p["vec"])) < min(1, 150_000 / N)] for p in parts]).astype(np.float32)
    ix = ShardedIVF(n_shards=n_shards, nlist=nlist)
    ix.train(sample)
    t1 = time.perf_counter()
    # columnar catalog, dense by id
    max_id = max(int(p["ids"].max()) for p in parts) + 1
    cat = {"mod": np.full(max_id, -1, np.int8), "label": np.full(max_id, -1, np.int16)}
    for c in ("lat", "lon", "s1", "s2", "s3"):
        cat[c] = np.full(max_id, np.nan, np.float32)
    cat["t"] = np.full(max_id, -1, np.int32)
    for p in parts:
        ids = p["ids"]; code = synth.MOD_CODE[str(p["mod"])]
        mod = np.full(len(ids), code, np.int8)
        ix.add(p["vec"], {"ids": ids, "mod": mod, "lat": p["lat"], "lon": p["lon"], "t": p["t"]})
        cat["mod"][ids] = code; cat["label"][ids] = p["labels"]
        for c in ("lat", "lon", "s1", "s2", "s3", "t"):
            cat[c][ids] = p[c]
    del parts
    ix.finalize()
    ix.save(index_dir)
    np.savez(f"{index_dir}/catalog.npz", **cat)
    t2 = time.perf_counter()
    stats = {"n": N, "nlist": nlist, "n_shards": n_shards, "train_s": t1 - t0, "build_s": t2 - t1,
             "index_bytes_hot_int8": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(index_dir)
                                         for f in fs if not f.startswith(("catalog", "f16"))),
             "index_bytes_rerank_f16": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(index_dir)
                                           for f in fs if f.startswith("f16"))}
    log(f"build: {N:,} vectors, nlist={nlist}, {n_shards} shards, train {stats['train_s']:.1f}s, "
        f"build {stats['build_s']:.1f}s, hot int8 tier {stats['index_bytes_hot_int8'] / 2**20:.0f} MiB, "
        f"f16 rerank tier {stats['index_bytes_rerank_f16'] / 2**20:.0f} MiB (mmap)")
    return ix, stats


def save_plan(index_dir, descs):
    """Persist batch descriptors so any item can be re-fetched from source (provenance)."""
    json.dump({"bs": descs[0]["bs"], "seed": (descs[0]["seed"]) // 1_000_003,
               "mods": [d["mod"] for d in descs]}, open(f"{index_dir}/plan.json", "w"))
