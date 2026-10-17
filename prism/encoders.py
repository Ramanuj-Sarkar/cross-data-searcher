"""Modality encoders -> one shared 256-d space.

Each modality has a native, physics/biology-aware featurizer:
  sat  : spectral indices (NDVI, NDWI, NBR), illumination-invariant chromaticity,
         texture/anisotropy, spatial heterogeneity          -> 45-d
  dna  : 5-mer + 3-mer spectra (alignment-free), GC content -> 1089-d
  cell : library-size normalized log1p expression           -> 2000-d
A linear head per modality is fit (closed-form ridge) to map native features onto
the text embedding of paired captions, CLIP-style but solvable in seconds on a CPU.
Text queries are embedded directly, so one query searches every modality.

The default text encoder is deliberately dependency-free (hashed words + char 4-grams).
The text side is pluggable (`make_text_encoder`): hashed words by default, or static embeddings via
the optional `model2vec` dependency; the choice is stored in model.npz so queries and index always agree.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

from . import synth

D = 256
TEXT_HASH_DIM = 4096
_STOP = set("a an the of in on at for to and or with from by near is are be this that these those "
            "show find me any all which what where give list get".split())


def l2n(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.maximum(n, 1e-9)).astype(np.float32)


def _tokens(text: str):
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        yield "w" + w, 1.0
        if len(w) >= 5:
            for i in range(len(w) - 3):
                yield "c" + w[i:i + 4], 0.35


class TextEncoder:
    """Hashed words + char 4-grams: dependency-free and instant, but purely lexical (no synonyms)."""
    name = "hashed"
    dim = D

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.R = (rng.standard_normal((TEXT_HASH_DIM, D)) / np.sqrt(D)).astype(np.float32)

    def encode(self, texts):
        X = np.zeros((len(texts), TEXT_HASH_DIM), np.float32)
        for r, t in enumerate(texts):
            for tok, w in _tokens(t):
                X[r, zlib.crc32(tok.encode()) % TEXT_HASH_DIM] += w
        return l2n(l2n(X) @ self.R)


DEFAULT_STATIC_MODEL = "minishlab/potion-base-8M"  # 256-d, ~30 MB, no transformer at query time


class StaticTextEncoder:
    """Static word embeddings (Model2Vec): a token-embedding lookup plus mean pooling. Understands synonyms and
    paraphrases that share no words, at near-hashed speed. Needs `pip install 'prism-demo-searcher[embeddings]'`;
    the model is fetched from the Hugging Face hub on first use."""

    def __init__(self, model_name: str = DEFAULT_STATIC_MODEL):
        try:
            from model2vec import StaticModel
        except ImportError as e:
            raise ImportError("static embeddings need the optional dependency: "
                              "pip install 'prism-demo-searcher[embeddings]'") from e
        self.name = f"model2vec:{model_name}"
        self.model = StaticModel.from_pretrained(model_name)
        self.dim = int(self.model.dim)

    def encode(self, texts):
        return l2n(np.asarray(self.model.encode(list(texts)), np.float32))


def make_text_encoder(spec: str = "hashed") -> TextEncoder | StaticTextEncoder:
    """spec: 'hashed' | 'model2vec' (default static model) | 'model2vec:<hugging-face model id>'."""
    kind, _, arg = spec.partition(":")
    if kind == "hashed" and not arg:
        return TextEncoder()
    if kind == "model2vec":
        return StaticTextEncoder(arg or DEFAULT_STATIC_MODEL)
    raise ValueError(f"unknown text encoder '{spec}' (use hashed, model2vec or model2vec:<model id>)")


# ----------------------------------------------------------------------------- native featurizers
def sat_features(tiles):
    n = tiles.shape[0]; eps = 1e-6
    B, G, R, N, S = (tiles[..., i] for i in range(5))
    ndvi = (N - R) / (N + R + eps); ndwi = (G - N) / (G + N + eps); nbr = (N - S) / (N + S + eps)
    flat = tiles.reshape(n, -1, 5)
    mean = flat.mean(1)
    feats = [mean, flat.std(1), mean / (mean.sum(1, keepdims=True) + eps)]
    for idx in (ndvi, ndwi, nbr):
        f = idx.reshape(n, -1)
        feats += [f.mean(1, keepdims=True), f.std(1, keepdims=True), np.quantile(f, [0.1, 0.5, 0.9], axis=1).T]
    bins = np.clip(((ndvi.reshape(n, -1) + 1) / 2 * 8).astype(np.int64), 0, 7)
    hist = np.bincount((bins + 8 * np.arange(n)[:, None]).ravel(), minlength=8 * n).reshape(n, 8) / bins.shape[1]
    bright = tiles.mean(-1)
    gx = np.abs(np.diff(bright, axis=2)).mean((1, 2)); gy = np.abs(np.diff(bright, axis=1)).mean((1, 2))
    tex = np.stack([gx, gy, np.abs(gx - gy) / (gx + gy + eps)], 1)
    H = tiles.shape[1] // 4
    het = np.stack([a.reshape(n, 4, H, 4, H).mean((2, 4)).reshape(n, -1).std(1) for a in (ndvi, nbr, bright)], 1)
    cloud = (bright > 0.5).reshape(n, -1).mean(1, keepdims=True)
    X = np.concatenate(feats + [hist, tex * 10, het, cloud], 1).astype(np.float32)
    summary = {"s1": ndvi.reshape(n, -1).mean(1), "s2": nbr.reshape(n, -1).mean(1), "s3": cloud[:, 0]}
    return X, summary


_W5 = 4 ** np.arange(4, -1, -1)
_W3 = 4 ** np.arange(2, -1, -1)


def _kmer(seqs, k, w):
    n, L = seqs.shape
    idx = sum(seqs[:, j:L - k + 1 + j].astype(np.int64) * w[j] for j in range(k))
    c = np.bincount((idx + (4 ** k) * np.arange(n)[:, None]).ravel(), minlength=n * 4 ** k)
    return c.reshape(n, 4 ** k) / (L - k + 1)


def dna_features(seqs):
    f5 = np.sqrt(_kmer(seqs, 5, _W5)); f3 = np.sqrt(_kmer(seqs, 3, _W3))
    gc = ((seqs == 1) | (seqs == 2)).mean(1, keepdims=True)
    X = np.concatenate([f5, f3, gc], 1).astype(np.float32)
    return X, {"s1": gc[:, 0], "s2": np.full(len(seqs), seqs.shape[1], np.float32),
               "s3": np.zeros(len(seqs), np.float32)}


_MARKER_IDX = np.arange(len(synth.MARKER_GENES))


def cell_features(counts):
    tot = counts.sum(1, keepdims=True).astype(np.float32)
    X = np.log1p(counts / np.maximum(tot, 1) * 1e4).astype(np.float32)
    top = X[:, _MARKER_IDX].argmax(1).astype(np.float32)
    return X, {"s1": tot[:, 0], "s2": (counts > 0).sum(1).astype(np.float32), "s3": top}


FEATURIZERS = {"sat": sat_features, "dna": dna_features, "cell": cell_features}


# ----------------------------------------------------------------------------- the aligned model
class PrismEncoder:
    """Per-modality standardize -> ridge head -> shared space (L2-normalized)."""

    def __init__(self, text_encoder: str = "hashed"):
        self.text_spec = text_encoder
        self._text: TextEncoder | StaticTextEncoder | None = None  # built on first use: ingest workers only need the heads
        self.heads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # mod -> (mu, sd, W)

    @property
    def text(self):
        if self._text is None:
            self._text = make_text_encoder(self.text_spec)
            self.text_spec = self._text.name
        return self._text

    @property
    def dim(self):
        """Width of the shared space (= the text encoder's embedding size)."""
        if self.heads:
            return next(iter(self.heads.values()))[2].shape[1]
        return self.text.dim

    def fit(self, n_per_class=400, alpha=1.0, seed=123, log=print):
        for mod in synth.MODALITIES:
            k = len(synth.CLASSES[mod])
            labels = np.repeat(np.arange(k), n_per_class)
            raw, _, _ = synth.GENERATORS[mod](len(labels), seed + synth.MOD_CODE[mod], labels=labels)
            X, _ = FEATURIZERS[mod](raw)
            T = self.text.encode(synth.captions_for(mod, labels, seed))
            mu, sd = X.mean(0), X.std(0) + 1e-4
            Z = (X - mu) / sd
            n, p = Z.shape
            lam = alpha * n
            if p <= n:
                W = np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ T)
            else:  # dual form when features outnumber samples
                W = Z.T @ np.linalg.solve(Z @ Z.T + lam * np.eye(n), T)
            self.heads[mod] = (mu.astype(np.float32), sd.astype(np.float32), W.astype(np.float32))
            if log:
                emb = l2n(Z @ W)
                cent = l2n(np.stack([T[labels == c].mean(0) for c in range(k)]))
                acc = ((emb @ cent.T).argmax(1) == labels).mean()
                log(f"  fit {mod:<4} native_dim={p:<5} n={n:<5} caption-centroid acc={acc:.3f}")
        return self

    def encode(self, mod, raw):
        X, summary = FEATURIZERS[mod](raw)
        mu, sd, W = self.heads[mod]
        return l2n(((X - mu) / sd) @ W), summary

    def encode_text(self, texts):
        return self.text.encode(texts)

    def save(self, path):
        arrs = {}
        for m, (mu, sd, W) in self.heads.items():
            arrs[f"{m}_mu"], arrs[f"{m}_sd"], arrs[f"{m}_W"] = mu, sd, W
        np.savez(path, text_encoder=np.array(self.text_spec), **arrs)  # type: ignore[arg-type]  # stubs reserve 'allow_pickle'

    @classmethod
    def load(cls, path):
        z = np.load(path)
        e = cls(str(z["text_encoder"]) if "text_encoder" in z.files else "hashed")  # older files: hashed
        for m in synth.MODALITIES:
            e.heads[m] = (z[f"{m}_mu"], z[f"{m}_sd"], z[f"{m}_W"])
        return e
