"""End-to-end scale benchmark. Writes results/bench.json.
  python bench.py --items 1000000 --workers 2
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
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
HELDOUT["sat"].update({"wetland": ["marshes and swamps", "wet vegetated areas by the water"],
                       "shrubland": ["chaparral and brush", "dry scrubland with low bushes"],
                       "recovering_burn": ["land regrowing after a fire", "vegetation recovering on old burn scars"],
                       "grassland": ["pastures and meadows", "grass covered rangeland"],
                       "suburban": ["residential neighborhoods with yards", "suburbs of single family homes"],
                       "salt_flat": ["dry lake bed with salt crust", "bright playa surface"]})
HELDOUT["dna"].update({"aav_rep": ["AAV Rep replication genes", "adeno-associated virus rep helper"],
                       "cas12a_cassette": ["Cpf1 editing constructs", "Cas12a guide array"],
                       "lentiviral_gag": ["gag pol packaging constructs", "HIV gag helper plasmid"],
                       "rodent_mito": ["murine mtDNA", "mouse mitochondria genome"],
                       "phage_t4": ["T4 phage DNA", "enterobacteria phage T4"]})
HELDOUT["cell"].update({"cd8_t_cell": ["cytotoxic CD8 T lymphocytes", "CD8A T cells"],
                        "macrophage": ["tissue macrophages", "CD68 phagocytes"],
                        "plasma_cell": ["antibody secreting cells", "MZB1 plasma cells"],
                        "cd4_t_cell": ["helper T lymphocytes", "CD4 T cells"],
                        "dendritic_cell": ["antigen presenting dendritic cells", "CD1C dendritic cells"],
                        "memory_b_cell": ["CD27 memory B lymphocytes", "class switched memory B cells"]})
DEMO_QUESTIONS = [
    "Find wildfire burn scars near Malibu in 2025",
    "Show burned areas in the Angeles National Forest since 2025; also dense forest in the Angeles National Forest",
    "AAV capsid sequences for liver gene therapy; also hepatocytes expressing ALB",
]


# Floors enforced by `python bench.py --ci` (a 65k-item subset). Deliberately loose: they catch real regressions
# (a broken index, a collapsed encoder, a hung pipeline), not machine-to-machine noise.
CI_THRESHOLDS = {
    "recall@10 (nprobe 32 + rerank)": ("ann_recall", ">=", 0.90),
    "text->sat p@10": ("p_sat", ">=", 0.75),
    "text->dna p@10": ("p_dna", ">=", 0.75),
    "text->cell p@10": ("p_cell", ">=", 0.75),
    "search p50 ms": ("search_p50_ms", "<=", 50.0),
    "filtered search p50 ms": ("filtered_p50_ms", "<=", 100.0),
    "ingest items/s": ("ingest_items_per_s", ">=", 1000.0),
    "agent runs grounded": ("agent_grounded", "==", 1.0),
    "agent claims contradicted": ("agent_contradicted", "==", 0.0),
    "agent claims verified": ("agent_verified", ">=", 1.0),
    "resume recomputes only dropped batches": ("resume_exact", "==", 1.0),
    "resume output identical to original": ("resume_identical", "==", 1.0),
    "second resume recomputes nothing": ("resume_second_items", "==", 0.0),
}


def ci_metrics(R):
    sweep = next(r for r in R["ann"]["sweep"] if r["nprobe"] == 32 and r["rerank"] == 8)
    q, rs = R["text_to_modality_p@10"], R["resume"]
    return {"ann_recall": sweep["recall@10"], "search_p50_ms": sweep["p50_ms"],
            "p_sat": q["sat"]["mean_p@10"], "p_dna": q["dna"]["mean_p@10"], "p_cell": q["cell"]["mean_p@10"],
            "filtered_p50_ms": R["filtered_search_p50_ms"], "ingest_items_per_s": R["ingest"]["items_per_s"],
            "agent_grounded": float(all(a["grounded"] for a in R["agent"])),
            "agent_contradicted": float(sum(len(a["claims"]["contradicted"]) for a in R["agent"])),
            "agent_verified": float(sum(a["claims"]["verified"] for a in R["agent"])),
            "resume_exact": float(rs["recomputed_batches"] == rs["dropped_batches"]),
            "resume_identical": float(rs["identical_after_resume"]), "resume_second_items": float(rs["second_resume_items"])}


def check_thresholds(R):
    m = ci_metrics(R); rows = []
    for name, (key, op, bound) in CI_THRESHOLDS.items():
        v = m[key]
        ok = {">=": v >= bound, "<=": v <= bound, "==": v == bound}[op]
        rows.append({"check": name, "value": round(v, 4), "op": op, "bound": bound, "ok": bool(ok)})
    return rows


def resume_check(descs, model, stage, workers):
    """Simulate a crash near the end of ingest -- last 10% of batches lost, manifest tail torn -- and measure the
    resume: only the lost batches may be recomputed, byte-for-byte equal to the originals, and a second resume is a no-op."""
    man = f"{stage}/manifest.jsonl"
    lines = open(man).read().splitlines()
    keep = int(len(lines) * 0.9)
    dropped = [json.loads(line)["batch"] for line in lines[keep:]]
    original = {}
    for b in dropped:
        path = f"{stage}/b{b:06d}.npz"
        with np.load(path) as z:
            original[b] = {k: z[k].copy() for k in ("vec", "ids", "labels")}
        os.remove(path)
    with open(man, "w") as fh:
        fh.write("\n".join(lines[:keep]) + "\n" + '{"batch": 7')  # torn final line, no newline
    t = time.perf_counter()
    st = P.ingest(descs, model, stage, workers=workers, log=None)
    seconds = time.perf_counter() - t
    bs = descs[0]["bs"]
    identical = True
    for b, ref in original.items():
        with np.load(f"{stage}/b{b:06d}.npz") as z:
            identical &= all(np.array_equal(z[k], ref[k]) for k in ref)
    second = P.ingest(descs, model, stage, workers=workers, log=None)
    return {"batches_total": len(descs), "dropped_batches": len(dropped), "skipped_batches": len(descs) - len(dropped),
            "recomputed_batches": st["items"] // bs, "resume_seconds": seconds, "identical_after_resume": bool(identical),
            "second_resume_items": second["items"]}


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    """Where and on what these numbers were measured."""
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        mem = None
    return {"machine": platform.machine(), "platform": platform.platform(), "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(), "memory_bytes": mem, "python": platform.python_version(),
            "numpy": np.__version__, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out"); ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--ci", action="store_true", help="fast 65k-item subset; enforce CI_THRESHOLDS, exit 1 on failure")
    ap.add_argument("--results", help="output JSON (default results/bench.json, or results/bench_ci.json with --ci)")
    a = ap.parse_args()
    a.items = a.items or (65_536 if a.ci else 1_000_000)
    scratch = a.out is None and a.ci
    a.out = a.out or (tempfile.mkdtemp(prefix="prism-bench-") if a.ci else "data")
    a.results = a.results or ("results/bench_ci.json" if a.ci else "results/bench.json")
    if a.fresh and os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
    R = {"config": vars(a), "data_profile": synth.PROFILE, "environment": environment()}
    model, stage, index = f"{a.out}/model.npz", f"{a.out}/stage", f"{a.out}/index"

    t = time.perf_counter(); PrismEncoder().fit().save(model); R["fit_s"] = time.perf_counter() - t
    descs = P.plan(a.items)
    R["ingest"] = P.ingest(descs, model, stage, workers=a.workers, every=100)
    _, R["build"] = P.build(stage, index); P.save_plan(index, descs)
    eng = Engine(index, model)
    R["resume"] = resume_check(descs, model, stage, a.workers)
    print("resume:", R["resume"])
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
    nq = 100 if a.ci else 200
    qids = rng.choice(len(V), nq, replace=False)
    Q = V[qids].astype(np.float32) + rng.normal(0, 0.05, (nq, 256)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    t = time.perf_counter()
    exact = np.zeros((len(Q), V.shape[0]), np.float32)
    for s in range(0, len(V), 200_000):
        exact[:, s:s + 200_000] = Q @ V[s:s + 200_000].astype(np.float32).T
    gt = np.argpartition(-exact, 9, axis=1)[:, :10]
    brute_ms = (time.perf_counter() - t) / len(Q) * 1e3
    del exact, V
    sweep = []
    for nprobe, rr in ([(16, 1), (32, 8), (64, 8)] if a.ci else [(16, 1), (64, 1), (4, 8), (8, 8), (16, 8), (32, 8), (64, 8)]):
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
        r = A.RulePlanner().run(A.Harness(eng, trace_dir=f"{a.out}/traces"), qn)
        runs.append(r); print("\n" + qn + "\n" + r["answer"])
    R["agent"] = [{k: v for k, v in r.items()} for r in runs]
    R["checks"] = check_thresholds(R)
    json.dump(R, open(a.results, "w"), indent=1, default=str)
    print(json.dumps({k: R[k] for k in ("ingest", "build", "ann", "filtered_search_p50_ms", "resume")}, indent=1, default=str))
    print({m: round(v["mean_p@10"], 3) for m, v in q.items()})
    if a.ci:
        print("\nCI thresholds:")
        for c in R["checks"]:
            print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['check']}: {c['value']} (need {c['op']} {c['bound']})")
        if scratch:
            shutil.rmtree(a.out, ignore_errors=True)
        if not all(c["ok"] for c in R["checks"]):
            sys.exit(1)


if __name__ == "__main__":
    main()
