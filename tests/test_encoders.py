"""Text encoders: paraphrase set integrity, spec parsing, persistence, and the static-embedding wrapper.

Everything runs offline against a fake `model2vec` module; the real model is only exercised when
PRISM_TEST_EMBEDDINGS=1 (needs `pip install -e ".[embeddings]"` and network access to the Hugging Face hub).
"""
import os
import sys
import tempfile
import types

import numpy as np
import pytest
from test_core import env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402
from prism import encoders, synth  # noqa: E402
from prism import pipeline as P
from prism.encoders import (  # noqa: E402
    PrismEncoder,
    StaticTextEncoder,
    TextEncoder,
    make_text_encoder,
)
from prism.engine import Engine  # noqa: E402

FAKE_DIM = 8


class FakeStaticModel:
    """Stands in for model2vec.StaticModel: deterministic, unnormalized, dim != 256."""
    dim = FAKE_DIM

    @classmethod
    def from_pretrained(cls, name):
        m = cls(); m.loaded = name
        return m

    def encode(self, texts):
        return TextEncoder().encode(texts)[:, :FAKE_DIM] * 3.0


@pytest.fixture
def fake_model2vec(monkeypatch):
    mod = types.ModuleType("model2vec"); mod.StaticModel = FakeStaticModel
    monkeypatch.setitem(sys.modules, "model2vec", mod)
    return mod


# ------------------------------------------------------------------ paraphrase set
def test_paraphrase_set_covers_every_class_twice():
    for mod, classes in synth.CLASSES.items():
        assert set(bench.PARAPHRASE[mod]) == set(classes)
        assert all(len(qs) == 2 for qs in bench.PARAPHRASE[mod].values())


def test_paraphrases_share_no_word_with_their_class_captions():
    for mod, per_class in bench.PARAPHRASE.items():
        for cname, qs in per_class.items():
            for q in qs:
                assert bench.word_overlap(q, mod, cname) == [], (mod, cname, q)


def test_word_overlap_detects_shared_words():
    assert bench.word_overlap("dense forest canopy", "sat", "forest") == ["canopy", "dense", "forest"]
    assert bench.word_overlap("forests", "sat", "forest") == ["forest"]  # plural stemming counts as the same word
    assert bench.word_overlap("timber stands on slopes", "sat", "forest") == []


def test_paraphrases_are_disjoint_from_heldout_queries():
    held = {q for per in bench.HELDOUT.values() for qs in per.values() for q in qs}
    para = {q for per in bench.PARAPHRASE.values() for qs in per.values() for q in qs}
    assert not held & para


# ------------------------------------------------------------------ spec parsing
def test_make_text_encoder_specs(fake_model2vec):
    assert isinstance(make_text_encoder("hashed"), TextEncoder)
    default = make_text_encoder("model2vec")
    assert isinstance(default, StaticTextEncoder) and default.name == f"model2vec:{encoders.DEFAULT_STATIC_MODEL}"
    assert default.model.loaded == encoders.DEFAULT_STATIC_MODEL
    assert make_text_encoder("model2vec:org/other-model").model.loaded == "org/other-model"
    for bad in ("", "hashed:x", "bert", "model2vec2:x"):
        with pytest.raises(ValueError):
            make_text_encoder(bad)


def test_missing_optional_dependency_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "model2vec", None)  # makes `import model2vec` raise ImportError
    with pytest.raises(ImportError, match=r"prism-demo-searcher\[embeddings\]"):
        make_text_encoder("model2vec")


def test_static_encoder_outputs_unit_vectors(fake_model2vec):
    e = make_text_encoder("model2vec")
    v = e.encode(["dense forest canopy", "liver cells"])
    assert v.shape == (2, FAKE_DIM) and v.dtype == np.float32 and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


# ------------------------------------------------------------------ persistence and plumbing
def test_encoder_spec_roundtrips_and_text_model_loads_lazily(fake_model2vec):
    d = tempfile.mkdtemp()
    PrismEncoder("model2vec:fake/model").fit(n_per_class=20, log=None).save(f"{d}/m.npz")
    loaded = PrismEncoder.load(f"{d}/m.npz")
    assert loaded.text_spec == "model2vec:fake/model" and loaded._text is None  # ingest workers never load the text model
    assert loaded.dim == FAKE_DIM and loaded.text.dim == FAKE_DIM and loaded._text is not None


def test_old_model_files_without_a_text_encoder_default_to_hashed():
    d = tempfile.mkdtemp()
    PrismEncoder().fit(n_per_class=20, log=None).save(f"{d}/m.npz")
    z = dict(np.load(f"{d}/m.npz")); z.pop("text_encoder")
    np.savez(f"{d}/old.npz", **z)
    old = PrismEncoder.load(f"{d}/old.npz")
    assert old.text_spec == "hashed" and isinstance(old.text, TextEncoder) and old.dim == encoders.D


def test_index_dimension_follows_the_text_encoder(fake_model2vec):
    """A non-256 text space flows through fit -> ingest (spawned workers) -> index -> search."""
    d = tempfile.mkdtemp()
    PrismEncoder("model2vec:fake/model").fit(n_per_class=60, log=None).save(f"{d}/model.npz")
    descs = P.plan(4096, 512)
    P.ingest(descs, f"{d}/model.npz", f"{d}/stage", workers=2, log=None)
    P.build(f"{d}/stage", f"{d}/index", n_shards=2, log=lambda *_: None); P.save_plan(f"{d}/index", descs)
    eng = Engine(f"{d}/index", f"{d}/model.npz")
    assert eng.ix.dim == FAKE_DIM and eng.ix.C.shape[1] == FAKE_DIM and len(eng) == 4096
    hits = eng.search("liver hepatocyte", modality="cell", k=5)["hits"]
    assert len(hits) == 5 and all(h["modality"] == "cell" for h in hits)
    assert eng.similar(hits[0]["id"], k=3)["hits"]


def test_cli_refuses_to_mix_encoders_in_one_output_directory(fake_model2vec, monkeypatch):
    from prism import __main__ as cli
    d = tempfile.mkdtemp()
    PrismEncoder().fit(n_per_class=20, log=None).save(f"{d}/model.npz")  # hashed model already present
    monkeypatch.setattr(sys, "argv", ["prism", "build", "--items", "1024", "--out", d, "--text-encoder", "model2vec"])
    with pytest.raises(SystemExit, match="was fit with text encoder 'hashed'"):
        cli.main()


def test_default_pipeline_is_unchanged_hashed():
    _, _, ENG = env()
    assert ENG.enc.text_spec == "hashed" and ENG.enc.dim == encoders.D == ENG.ix.dim


# ------------------------------------------------------------------ real model (opt-in)
@pytest.mark.skipif(os.environ.get("PRISM_TEST_EMBEDDINGS") != "1", reason="set PRISM_TEST_EMBEDDINGS=1 (needs model2vec + network)")
def test_real_static_model_beats_hashed_on_paraphrases():
    pytest.importorskip("model2vec")
    scores = {}
    for spec in ("hashed", "model2vec"):
        d = tempfile.mkdtemp()
        PrismEncoder(spec).fit(n_per_class=200, log=None).save(f"{d}/model.npz")
        descs = P.plan(32768, 1024)
        P.ingest(descs, f"{d}/model.npz", f"{d}/stage", workers=2, log=None)
        P.build(f"{d}/stage", f"{d}/index", n_shards=2, log=lambda *_: None); P.save_plan(f"{d}/index", descs)
        q = bench.quality(Engine(f"{d}/index", f"{d}/model.npz"), bench.PARAPHRASE)
        scores[spec] = float(np.mean([v["mean_p@10"] for v in q.values()]))
    assert scores["hashed"] < 0.15                     # lexical encoder is near chance without shared words
    assert scores["model2vec"] > scores["hashed"] + 0.15, scores


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
