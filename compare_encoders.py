"""Compare text encoders on identical data: hashed words vs static embeddings.

  pip install -e ".[embeddings]"
  python compare_encoders.py                       # 131k items per encoder, writes results/encoder_comparison.json
  python compare_encoders.py --encoders hashed model2vec:minishlab/potion-base-32M

Each encoder gets its own fitted heads and index over the same synthetic corpus. Two query sets are scored with
precision@10 against the true class: HELDOUT (new phrasings, some vocabulary shared with training captions) and
PARAPHRASE (no word shared with the target class's captions -- see bench.word_overlap).
"""
import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np

import bench
from prism import pipeline as P
from prism.encoders import PrismEncoder
from prism.engine import Engine


def run_one(spec, items, workers, workdir):
    out = os.path.join(workdir, spec.replace("/", "_").replace(":", "_"))
    os.makedirs(out)
    model, stage, index = f"{out}/model.npz", f"{out}/stage", f"{out}/index"
    t = time.perf_counter(); enc = PrismEncoder(spec); enc.text  # noqa: B018  (load the text model inside the timing)
    load_s = time.perf_counter() - t
    t = time.perf_counter(); enc.fit(log=None).save(model); fit_s = time.perf_counter() - t
    descs = P.plan(items)
    P.ingest(descs, model, stage, workers=workers, log=None)
    P.build(stage, index, log=lambda *_: None); P.save_plan(index, descs)
    eng = Engine(index, model)
    all_queries = [q for qs in bench.PARAPHRASE.values() for cl in qs.values() for q in cl]
    eng.enc.encode_text(all_queries[:2])  # warm
    lat = []
    for q in all_queries:
        t = time.perf_counter(); eng.enc.encode_text([q]); lat.append(time.perf_counter() - t)
    res = {"text_encoder": eng.enc.text.name, "dim": eng.enc.dim, "load_s": load_s, "fit_s": fit_s, "items": len(eng),
           "query_encode_p50_ms": float(np.percentile(lat, 50) * 1e3),
           "heldout": bench.quality(eng, bench.HELDOUT), "paraphrase": bench.quality(eng, bench.PARAPHRASE)}
    shutil.rmtree(out, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["hashed", "model2vec"])
    ap.add_argument("--items", type=int, default=131_072)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--results", default="results/encoder_comparison.json")
    a = ap.parse_args()
    overlaps = [(m, c, q, bench.word_overlap(q, m, c)) for m, cl in bench.PARAPHRASE.items() for c, qs in cl.items() for q in qs]
    assert not any(o[3] for o in overlaps), "paraphrase queries must share no words with their class captions"
    work = tempfile.mkdtemp(prefix="prism-encoders-")
    try:
        R = {"items": a.items, "data_profile": bench.synth.PROFILE, "paraphrase_queries": len(overlaps),
             "paraphrase_word_overlap": 0, "environment": bench.environment(), "encoders": {}}
        for spec in a.encoders:
            print(f"== {spec}", flush=True)
            R["encoders"][spec] = run_one(spec, a.items, a.workers, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
    json.dump(R, open(a.results, "w"), indent=1)

    names = list(R["encoders"])
    print(f"\nprecision@10, {a.items:,} items, {len(overlaps)} paraphrase queries with zero word overlap\n")
    print(f"{'':28s}" + "".join(f"{n[:30]:>32s}" for n in names))
    for label, key in (("held-out phrasings", "heldout"), ("paraphrases (no overlap)", "paraphrase")):
        for mod in ("sat", "dna", "cell"):
            print(f"{label + ' ' + mod:28s}" + "".join(f"{R['encoders'][n][key][mod]['mean_p@10']:32.3f}" for n in names))
    print(f"{'query encode p50 (ms)':28s}" + "".join(f"{R['encoders'][n]['query_encode_p50_ms']:32.3f}" for n in names))


if __name__ == "__main__":
    main()
