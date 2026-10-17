# PRISM — one search across siloed R&D data

**The problem.** A biotech's experimental data lives in separate systems that don't talk: sequencing output in one store, imaging in another, single-cell expression in a third. Asking "show me everything relevant to this question" means querying each silo by hand, in its own format, with no shared vocabulary and no single trail back to source. Discovery work that should take minutes takes an afternoon of glue code.

**What PRISM does.** It puts every data type — here satellite imagery, DNA sequences and single-cell RNA — into one searchable space, so a plain-language question searches all of them at once. An *auditable* agent composes the answer, and every claim is **grounding-checked**: any item it cites that the tools never actually returned gets flagged, and every step (query, latency, result hash) is logged. That provenance-and-audit layer is the headline — it's the host's own domain (an enterprise agentic harness) and the thing R&D teams can't get from grep across folders.

**Measured.** On ~1M items and 2 laptop-class CPU cores: ingest **9.3k items/s**, search **1.8 ms p50 at 0.95 recall@10**, 100% grounded agent answers.

**How it works (one paragraph).** Each data type gets a domain-aware featurizer: spectral indices (NDVI, NBR, NDWI) for imagery, k-mer spectra for DNA, normalized expression for cells. A closed-form ridge head per type aligns those features to the text embedding of paired descriptions — CLIP-style alignment that fits in ~2 seconds. Vectors land in a sharded IVF index with three tiers: int8 hot storage, a float16 rerank tier, and metadata filters, all memory-mapped. The data is synthetic but structurally realistic, and every item re-fetches from its batch descriptor for provenance.

## Results (999,424 items, 2 vCPU, 7 GB RAM, numpy only)

| Stage                                                | Result                                                        |
|------------------------------------------------------|---------------------------------------------------------------|
| Encoder fit (3 modalities)                           | 2.1 s                                                         |
| Parallel ingest (fetch → featurize → encode → stage) | **9,263 items/s**, batch p95 372 ms, 2 workers                |
| Resume after crash                                   | 976/976 staged batches skipped, 0 recomputed                  |
| Index build (k-means 4096 lists, 4 shards)           | 49 s · hot int8 tier 268 MiB · f16 rerank tier 488 MiB (mmap) |
| ANN, nprobe 32 + rerank                              | **recall@10 0.953, p50 1.8 ms** (brute force: 11.1 ms)        |
| ANN, nprobe 64 + rerank                              | recall@10 0.975, p50 2.9 ms, scans 2% of corpus               |
| Filtered search (modality + bbox + date)             | 1.1 ms p50                                                    |
| Held-out text → sat / dna / cell precision@10        | 1.00 / 1.00 / 1.00                                            |
| Agent runs (3 multi-intent questions)                | 100% grounded, 4–8 tool calls, 0.26–0.52 s                    |

The recall table shows why the rerank tier exists. With int8 only, recall plateaus at 0.79 even exhaustively, because neighbours in dense clusters are near-ties. Adding the f16 rerank of an 8× shortlist lifts nprobe 32 from 0.74 to 0.95 for about 0.3 ms more.

## Demo script (≈3 min) — framed as R&D data discovery

Open on the pain: "Your sequencing data, imaging and expression profiles live in three systems. Here's one search box over all of them."

1. **One question, two data types.** `python -m prism ask --out data "AAV capsid sequences; also hepatocytes expressing ALB"` — the agent splits it, searches DNA and single-cell separately, and returns cited hits from each. This is the "I'd have opened two tools and written a join" moment.
2. **Every answer is traceable.** `cat data/traces/<run>.jsonl` — per-step args, latency, result hash, and the grounding verdict. Point out that a fabricated citation would be caught here. This is the audit story the host cares about.
3. **Drill into one hit.** `curl localhost:8080/item/<id>` — the item re-fetches from source and returns measured properties plus its source URI, so a reviewer can verify the claim, not just trust it.
4. **It holds up at scale.** Kill an ingest mid-run and restart — it resumes from the manifest, recomputing nothing. Mention the 9.3k items/s and 1.8 ms numbers here.

(The satellite side — `"Find wildfire burn scars near Malibu in 2025"` — is the easy-to-read backup demo if the room wants something visual with no domain setup.)

## Architecture

```
descriptors ──► worker pool: fetch → featurize → encode ──► staged .npz + manifest (idempotent)
                                                                │
                              k-means coarse quantizer ◄────────┘
                                         │
                 shard 0..S  [int8 IVF lists | f16 rerank | ids, modality, lat, lon, t]  (mmap)
                                         │
 text / item query ─► centroid scores ─► thread-pool fan-out ─► filter ─► shortlist ─► f16 rerank ─► merge top-k
                                         │
          Harness: schema validation · allowlist · step + time budget · JSONL trace · grounding check
                                         │
          Planner: Claude (tool use, if ANTHROPIC_API_KEY) │ deterministic rule planner (offline)
```

| File                 | Role                                                                                                                                |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `prism/synth.py`     | Deterministic sources: 5-band tiles (7 land covers, SoCal geo), DNA families (Markov + motifs), scRNA (8 cell types, gamma-Poisson) |
| `prism/encoders.py`  | Featurizers + ridge alignment into a shared 256-d space; swappable text encoder                                                     |
| `prism/index.py`     | Sharded IVF, int8 + f16 tiers, filters, adaptive nprobe, mmap persistence                                                           |
| `prism/pipeline.py`  | Parallel, resumable ingest; index build; columnar catalog                                                                           |
| `prism/engine.py`    | Search, similar, inspect (re-fetch from source), aggregate; place gazetteer                                                         |
| `prism/agent.py`     | Tool registry, policy harness, Claude and rule planners                                                                             |
| `prism/server.py`    | `/search`, `/similar`, `/item/<id>`, `/agent` JSON API                                                                              |
| `bench.py`, `tests/` | End-to-end benchmark (`results/bench.json`) and 7 tests                                                                             |

## Run sans Agent

```bash
uv pip install numpy                       # only dependency
python -m prism build --items 1000000 --workers 8 --out data
python -m prism ask --out data "Show burned areas in the Angeles National Forest since 2025"
python -m prism serve --out data --port 8080
```
## Run with Agent

```bash
export ANTHROPIC_API_KEY=...            # Claude becomes the planner
uv pip install numpy
python -m prism build --items 1000000 --workers 8 --out data
python -m prism ask --out data "Show burned areas in the Angeles National Forest since 2025"
python -m prism serve --out data --port 8080
```

## Run Tests

```bash
uv pip install numpy
python bench.py --items 1000000 --workers 2 && python tests/test_core.py
```

## Be upfront with judges

- The data is synthetic. Precision of 1.00 reflects separable classes, not real-world difficulty. The scale, latency and recall numbers are real measurements on this machine.
- The text encoder is lexical (hashed words + char n-grams), so paraphrases with no shared vocabulary will miss. It sits behind one interface, so a sentence-embedding model drops in.
- The Claude planner path is implemented but does not show a particular advantage, which may solely be due to the synthetic data itself.

## Path to production

Real readers drop in at `synth.GENERATORS`: Sentinel-2 COGs via STAC, FASTA/BAM, AnnData. Foundation-model encoders (Prithvi / SatMAE, DNABERT-2 / Nucleotide Transformer, scGPT / Geneformer) replace the featurizers, and the ridge heads become contrastive projections. Shards go one per node behind the same `Shard.search` contract, and the pool becomes Ray or k8s Jobs.
