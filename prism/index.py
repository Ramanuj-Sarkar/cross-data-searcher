"""Sharded, filtered IVF index with int8 storage.

* Shared coarse quantizer (k-means centroids) across S shards; each shard keeps its
  vectors grouped by inverted list (CSR offsets), quantized to int8 with per-dim scale
  -> 4x smaller than float32, ~256 B/vector at d=256.
* Query: score centroids once, fan out to shards on a thread pool (numpy releases the
  GIL in matmul), merge top-k.
* Metadata filters (modality, bbox, time window) are applied on the candidate set;
  if a selective filter leaves < k hits, nprobe doubles adaptively (no silent recall loss).
* Shards persist as plain .npy and reopen memory-mapped, so a node can serve an index
  larger than RAM and shards can be placed on different machines unchanged.
"""
from __future__ import annotations
import json
import os
import shutil
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import numpy as np

META_COLS = ("ids", "mod", "lat", "lon", "t")


@contextmanager
def atomic_dir(path):
    """Yield a scratch directory next to `path`; on success it replaces `path` in one rename.
    A crash mid-write leaves the previous directory untouched (plus a stale '.tmp' to clean up)."""
    path = os.path.normpath(path)
    tmp, old = path + ".tmp", path + ".old"
    for d in (tmp, old):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(tmp)
    try:
        yield tmp
        if os.path.exists(path):
            os.rename(path, old)
        os.rename(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.exists(old) and not os.path.exists(path):
            os.rename(old, path)  # swap failed half-way: restore the previous index
        raise
    shutil.rmtree(old, ignore_errors=True)


def kmeans(X, k, iters=12, seed=0, sample=200_000):
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = np.concatenate([(X[i:i + 32768] @ C.T).argmax(1) for i in range(0, len(X), 32768)])
        cnt = np.bincount(a, minlength=k)
        order = np.argsort(a, kind="stable"); sums = np.zeros_like(C)
        nz = np.flatnonzero(cnt)
        sums[nz] = np.add.reduceat(X[order], np.concatenate([[0], np.cumsum(cnt)[:-1]])[nz])
        empty = cnt == 0
        C = sums / np.maximum(cnt, 1)[:, None]
        C[empty] = X[rng.choice(len(X), empty.sum(), replace=False)]
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return C.astype(np.float32)


class Shard:
    def __init__(self, q8, offsets, meta, f16=None):
        self.q8, self.offsets, self.meta, self.f16 = q8, offsets, meta, f16

    def __len__(self):
        return len(self.q8)

    def search(self, qs, q, lists, k, filt, rerank):
        """qs: (d,) query pre-multiplied by scale; lists: centroid ids to probe."""
        o = self.offsets
        spans = [(o[l], o[l + 1]) for l in lists if o[l + 1] > o[l]]
        if not spans:
            return np.empty(0, np.float32), np.empty(0, np.int64), 0
        cand = np.concatenate([self.q8[a:b] for a, b in spans])
        pos = np.concatenate([np.arange(a, b) for a, b in spans])
        if filt is not None:
            m = filt(self.meta, pos)
            cand, pos = cand[m], pos[m]
        if len(pos) == 0:
            return np.empty(0, np.float32), np.empty(0, np.int64), 0
        s = cand.astype(np.float32) @ qs
        kk = min(k * rerank if self.f16 is not None else k, len(s))
        top = np.argpartition(-s, kk - 1)[:kk]
        if self.f16 is not None and rerank > 1:  # exact rescoring of the int8 shortlist
            p = np.sort(pos[top])
            s2 = np.asarray(self.f16[p], np.float32) @ q
            t2 = np.argpartition(-s2, min(k, len(s2)) - 1)[:k]
            return s2[t2], np.asarray(self.meta["ids"][p[t2]]), len(pos)
        return s[top], np.asarray(self.meta["ids"][pos[top]]), len(pos)


class ShardedIVF:
    def __init__(self, dim=256, n_shards=4, nlist=1024, threads=None, rerank_tier=True):
        self.dim, self.n_shards, self.nlist, self.rerank_tier = dim, n_shards, nlist, rerank_tier
        self.C = None; self.scale = None; self.shards: list[Shard] = []
        self._buf = [[] for _ in range(n_shards)]
        self.pool = ThreadPoolExecutor(threads or min(n_shards, os.cpu_count() or 2))

    # ---------------------------------------------------------------- build
    def train(self, sample):
        self.C = kmeans(sample, self.nlist)
        self.scale = (np.quantile(np.abs(sample), 0.9999, axis=0) / 127 + 1e-6).astype(np.float32)

    def add(self, vecs, meta):
        """Route rows to shards by id hash; buffer until finalize()."""
        shard_of = (meta["ids"] * 2654435761 % (2 ** 32)) % self.n_shards
        for s in range(self.n_shards):
            m = shard_of == s
            if m.any():
                self._buf[s].append((vecs[m], {c: meta[c][m] for c in META_COLS}))

    def finalize(self):
        self.shards = []
        for s in range(self.n_shards):
            V = np.concatenate([b[0] for b in self._buf[s]]).astype(np.float32)
            meta = {c: np.concatenate([b[1][c] for b in self._buf[s]]) for c in META_COLS}
            a = np.concatenate([(V[i:i + 65536] @ self.C.T).argmax(1) for i in range(0, len(V), 65536)])
            order = np.argsort(a, kind="stable")
            offsets = np.zeros(self.nlist + 1, np.int64)
            offsets[1:] = np.cumsum(np.bincount(a, minlength=self.nlist))
            q8 = np.clip(np.rint(V[order] / self.scale), -127, 127).astype(np.int8)
            f16 = V[order].astype(np.float16) if self.rerank_tier else None
            self.shards.append(Shard(q8, offsets, {c: meta[c][order] for c in META_COLS}, f16))
            self._buf[s] = []
        return self

    # ---------------------------------------------------------------- query
    def __len__(self):
        return sum(len(s) for s in self.shards)

    def search(self, Q, k=10, nprobe=16, filters=None, adaptive=True, rerank=8):
        Q = np.atleast_2d(Q).astype(np.float32)
        filt = make_filter(filters) if filters else None
        cs = Q @ self.C.T
        out_s, out_i, stats = [], [], []
        for qi in range(len(Q)):
            npb = nprobe
            while True:
                lists = np.argpartition(-cs[qi], npb - 1)[:npb]
                qs = Q[qi] * self.scale
                res = list(self.pool.map(lambda sh: sh.search(qs, Q[qi], lists, k, filt, rerank), self.shards))
                s = np.concatenate([r[0] for r in res]); i = np.concatenate([r[1] for r in res])
                scanned = sum(r[2] for r in res)
                if len(s) >= k or not adaptive or npb >= self.nlist:
                    break
                npb = min(self.nlist, npb * 2)
            top = np.argsort(-s)[:k]
            out_s.append(s[top]); out_i.append(i[top])
            stats.append({"nprobe": int(npb), "scanned": int(scanned)})
        return out_s, out_i, stats

    # ---------------------------------------------------------------- persistence
    def save(self, path, extra=None):
        """Atomically (re)write the index at `path`. `extra` maps file name -> dict of arrays, saved as
        .npz inside the same directory so e.g. the catalog lands in the same atomic swap."""
        with atomic_dir(path) as tmp:
            np.save(f"{tmp}/centroids.npy", self.C); np.save(f"{tmp}/scale.npy", self.scale)
            for si, sh in enumerate(self.shards):
                d = f"{tmp}/shard_{si:02d}"; os.makedirs(d)
                np.save(f"{d}/q8.npy", sh.q8); np.save(f"{d}/offsets.npy", sh.offsets)
                if sh.f16 is not None:
                    np.save(f"{d}/f16.npy", sh.f16)
                for c in META_COLS:
                    np.save(f"{d}/{c}.npy", sh.meta[c])
            for name, arrays in (extra or {}).items():
                np.savez(f"{tmp}/{name}", **arrays)
            with open(f"{tmp}/index.json", "w") as fh:  # written last: marks the directory complete
                json.dump({"dim": self.dim, "n_shards": self.n_shards, "nlist": self.nlist, "n": len(self)}, fh)

    @classmethod
    def load(cls, path, mmap=True):
        cfg = json.load(open(f"{path}/index.json"))
        ix = cls(cfg["dim"], cfg["n_shards"], cfg["nlist"])
        ix.C = np.load(f"{path}/centroids.npy"); ix.scale = np.load(f"{path}/scale.npy")
        mm = "r" if mmap else None
        for si in range(cfg["n_shards"]):
            d = f"{path}/shard_{si:02d}"
            meta = {c: np.load(f"{d}/{c}.npy", mmap_mode=mm) for c in META_COLS}
            f16 = np.load(f"{d}/f16.npy", mmap_mode="r") if os.path.exists(f"{d}/f16.npy") else None
            ix.shards.append(Shard(np.load(f"{d}/q8.npy", mmap_mode=mm), np.load(f"{d}/offsets.npy"), meta, f16))
        return ix


def make_filter(f):
    mods = f.get("modality"); bbox = f.get("bbox"); tw = f.get("time")

    def fn(meta, pos):
        m = np.ones(len(pos), bool)
        if mods is not None:
            m &= np.isin(meta["mod"][pos], mods)
        if bbox is not None:
            la, lo = meta["lat"][pos], meta["lon"][pos]
            m &= (la >= bbox[0]) & (la <= bbox[1]) & (lo >= bbox[2]) & (lo <= bbox[3])
        if tw is not None:
            t = meta["t"][pos]
            m &= (t >= tw[0]) & (t <= tw[1])
        return m
    return fn
