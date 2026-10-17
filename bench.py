"""End-to-end scale benchmark. Writes results/bench.json.
  python bench.py --items 1000000 --workers 2
"""
import argparse
import json
import os
import shutil
import time
import numpy as np
from prism import pipeline as P, synth, agent as A
from prism.encoders import PrismEncoder
from prism.engine import Engine

# Held-out natural-language queries: NOT the training caption templates.
HELDOUT = {
    "sat": {"forest": ["wooded mountains with thick trees", "vegetation cover in the forest"],
            "water": ["lakes and reservoirs", "ocean water along the coast"],
            "urban": ["streets and buildings in the city", "urban sprawl and freeways"],
            "desert": ["dry sandy desert with no plants", "barren arid land"],
            "cropland": ["farmland with rows of crops", "agricultural fields"],
            "burn_scar": ["areas burned by recent wildfires", "charred hillsides after a fire"],
            "cloud": ["scenes blocked by clouds", "cloudy overcast imagery"]},
    "dna": {"aav_capsid": ["AAV serotype capsids for gene therapy", "adeno-associated viral vector"],
            "lentiviral_backbone": ["lentivirus vectors", "lentiviral LTR backbone"],
            "crispr_cas9": ["Cas9 genome editing constructs", "CRISPR editing"],
            "human_mito": ["mitochondrial genome", "human mtDNA"],
            "resistance_plasmid": ["antibiotic resistance plasmids", "bacterial plasmid vectors"],
            "bacteriophage": ["phage genomes", "bacteriophage DNA"]},
    "cell": {"t_cell": ["T lymphocytes", "CD3E T cells"], "b_cell": ["B lymphocytes", "MS4A1 B cells"],
             "monocyte": ["classical monocytes", "LYZ myeloid cells"], "nk_cell": ["NK cells", "natural killer lymphocytes"],
             "hepatocyte": ["liver cells expressing albumin ALB", "hepatocytes"],
             "neuron": ["brain neurons", "SNAP25 neurons"], "cardiomyocyte": ["heart muscle cells", "cardiac TNNT2 cells"],
             "hsc": ["CD34 stem cells", "hematopoietic progenitors"]},
}
DEMO_QUESTIONS = [
    "Find wildfire burn scars near Malibu in 2025",
    "Show burned areas in the Angeles National Forest since 2025; also dense forest in the Angeles National Forest",
    "AAV capsid sequences for liver gene therapy; also hepatocytes expressing ALB",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=1_000_000); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="data"); ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.fresh and os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True); os.makedirs("results", exist_ok=True)
    R = {"config": vars(a)}
    model, stage, index = f"{a.out}/model.npz", f"{a.out}/stage", f"{a.out}/index"

    t = time.perf_counter(); PrismEncoder().fit().save(model); R["fit_s"] = time.perf_counter() - t
    descs = P.plan(a.items)
    R["ingest"] = P.ingest(descs, model, stage, workers=a.workers, every=100)
    _, R["build"] = P.build(stage, index); P.save_plan(index, descs)
    eng = Engine(index, model)
    cat = eng.cat

    # --- cross-modal retrieval quality (held-out text -> each modality)
    q = {}
    for mod, classes in HELDOUT.items():
        per = {}
        for cname, qs in classes.items():
            ci = synth.CLASSES[mod].index(cname)
            per[cname] = float(np.mean([np.mean(cat["label"][[h["id"] for h in eng.search(s, modality=mod, k=10)["hits"]]] == ci) for s in qs]))
        q[mod] = {"mean_p@10": float(np.mean(list(per.values()))), "per_class": per}
    R["text_to_modality_p@10"] = q

    # --- ANN recall + latency vs exact search on the staged float vectors
    print("loading staged vectors for exact baseline")
    files = sorted(f for f in os.listdir(stage) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    V = np.empty((len(cat["mod"]), 256), np.float16)
    for f in files:
        z = np.load(f"{stage}/{f}"); V[z["ids"]] = z["vec"]
    rng = np.random.default_rng(1)
    qids = rng.choice(len(V), 200, replace=False)
    Q = V[qids].astype(np.float32) + rng.normal(0, 0.05, (200, 256)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    t = time.perf_counter()
    exact = np.zeros((len(Q), V.shape[0]), np.float32)
    for s in range(0, len(V), 200_000):
        exact[:, s:s + 200_000] = Q @ V[s:s + 200_000].astype(np.float32).T
    gt = np.argpartition(-exact, 9, axis=1)[:, :10]
    brute_ms = (time.perf_counter() - t) / len(Q) * 1e3
    del exact, V
    sweep = []
    for nprobe, rr in [(16, 1), (64, 1), (4, 8), (8, 8), (16, 8), (32, 8), (64, 8)]:
        lat = []; rec = []
        for i in range(len(Q)):
            t = time.perf_counter(); _, ids, st = eng.ix.search(Q[i], k=10, nprobe=nprobe, rerank=rr)
            lat.append(time.perf_counter() - t)
            rec.append(len(set(ids[0].tolist()) & set(gt[i].tolist())) / 10)
        sweep.append({"nprobe": nprobe, "rerank": rr, "recall@10": float(np.mean(rec)), "p50_ms": float(np.percentile(lat, 50) * 1e3),
                      "p99_ms": float(np.percentile(lat, 99) * 1e3), "qps_1thread_client": float(1 / np.mean(lat)),
                      "frac_scanned": st[0]["scanned"] / len(cat["mod"])})
        print(f"  nprobe={nprobe:<3} rerank={rr} recall@10={sweep[-1]['recall@10']:.3f} p50={sweep[-1]['p50_ms']:.1f}ms")
    R["ann"] = {"brute_force_ms_per_query": brute_ms, "sweep": sweep}

    # --- filtered queries
    lat = []
    for _ in range(50):
        t = time.perf_counter()
        eng.search("wildfire burn scar", modality="sat", place="malibu", date_from="2025-06-01", date_to="2025-12-31", k=10)
        lat.append(time.perf_counter() - t)
    R["filtered_search_p50_ms"] = float(np.percentile(lat, 50) * 1e3)

    # --- agent runs
    runs = []
    for qn in DEMO_QUESTIONS:
        r = A.RulePlanner().run(A.Harness(eng, trace_dir="results/traces"), qn)
        runs.append(r); print("\n" + qn + "\n" + r["answer"])
    R["agent"] = [{k: v for k, v in r.items()} for r in runs]
    json.dump(R, open("results/bench.json", "w"), indent=1, default=str)
    print(json.dumps({k: R[k] for k in ("ingest", "build", "ann", "filtered_search_p50_ms")}, indent=1, default=str))
    print({m: round(v["mean_p@10"], 3) for m, v in q.items()})


if __name__ == "__main__":
    main()
