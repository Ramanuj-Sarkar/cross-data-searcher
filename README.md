# PRISM — one search across siloed R&D data

**The problem.** A biotech's experimental data lives in separate systems that don't talk: sequencing output in one store, imaging in another, single-cell expression in a third. Asking "show me everything relevant to this question" means querying each silo by hand, in its own format, with no shared vocabulary and no single trail back to source. Discovery work that should take minutes takes an afternoon of glue code.

**What PRISM does.** It puts every data type — here satellite imagery, DNA sequences and single-cell RNA — into one searchable space, so a plain-language question searches all of them at once. An *auditable* agent composes the answer, and every claim is **grounding-checked**: any item it cites that the tools never actually returned gets flagged, and every step (query, latency, result hash) is logged. That provenance-and-audit layer is the headline — it's the host's own domain (an enterprise agentic harness) and the thing R&D teams can't get from grep across folders.

**Measured.** On ~1M items and 2 ingest workers (Apple M-series laptop, 12 cores / 32 GB): ingest **17.4k items/s**, search **1.9 ms p50 at 0.99 recall@10**, 100% grounded agent answers.

**How it works (one paragraph).** Each data type gets a domain-aware featurizer: spectral indices (NDVI, NBR, NDWI) for imagery, k-mer spectra for DNA, normalized expression for cells. A closed-form ridge head per type aligns those features to the text embedding of paired descriptions — CLIP-style alignment that fits in ~2 seconds. Vectors land in a sharded IVF index with three tiers: int8 hot storage, a float16 rerank tier, and metadata filters, all memory-mapped. The data is synthetic but structurally realistic, and every item re-fetches from its batch descriptor for provenance.

## Results (999,424 items, 2 ingest workers, 12-core arm64 Mac / 32 GB, numpy only)

Machine, Python/numpy versions and git commit are recorded under `environment` in `results/bench.json`. Earlier numbers (9.3k items/s, 0.953 recall) came from a 2 vCPU / 7 GB box and 7 / 6 / 8 classes per modality, so they are not directly comparable.

| Stage                                                | Result                                                        |
|------------------------------------------------------|---------------------------------------------------------------|
| Encoder fit (3 modalities)                           | 3.1 s                                                         |
| Parallel ingest (fetch → featurize → encode → stage) | **17,426 items/s**, batch p95 178 ms, 2 workers               |
| Resume after crash                                   | 976/976 staged batches skipped (earlier run; covered by `test_resume_is_idempotent`) |
| Index build (k-means 4096 lists, 4 shards)           | 18 s · hot int8 tier 268 MiB · f16 rerank tier 488 MiB (mmap) |
| ANN, nprobe 32 + rerank                              | **recall@10 0.988, p50 1.9 ms** (brute force: 10.7 ms)        |
| ANN, nprobe 64 + rerank                              | recall@10 0.998, p50 2.9 ms, scans 1.7% of corpus             |
| Filtered search (modality + bbox + date)             | 2.2 ms p50                                                    |
| Held-out text → sat / dna / cell precision@10        | 0.945 / 0.975 / 0.982                                         |
| Agent runs (3 multi-intent questions)                | 100% grounded, 4–8 tool calls, 0.15–0.30 s                    |

The recall table shows why the rerank tier exists. Neighbours in dense clusters are near-ties, so int8 scoring alone caps recall. With int8 only, recall is 0.78–0.82 at nprobe 16–64; the f16 rerank of an 8× shortlist lifts nprobe 16 to 0.95 and nprobe 32 to 0.99 for under 1 ms more.

## Demo script (≈3 min) — framed as R&D data discovery

Open on the pain: "Your sequencing data, imaging and expression profiles live in three systems. Here's one search box over all of them."

1. **One question, two data types.** `python -m prism ask --out data "AAV capsid sequences; also hepatocytes expressing ALB"` — the agent splits it, searches DNA and single-cell separately, and returns cited hits from each. This is the "I'd have opened two tools and written a join" moment.
2. **Every answer is traceable.** `cat data/traces/<run>.jsonl` — per-step args, latency, result hash, and the grounding verdict. Point out that a fabricated citation would be caught here. This is the audit story the host cares about.
3. **Drill into one hit.** `curl localhost:8080/item/<id>` — the item re-fetches from source and returns measured properties plus its source URI, so a reviewer can verify the claim, not just trust it.
4. **It holds up at scale.** Kill an ingest mid-run and restart — it resumes from the manifest, recomputing nothing. Mention the ingest rate and ~2 ms search numbers from the table above here.

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
| `prism/synth.py`     | Deterministic sources: 5-band tiles (10 land covers, SoCal geo), DNA families (Markov + motifs), scRNA (11 cell types, gamma-Poisson) |
| `prism/encoders.py`  | Featurizers + ridge alignment into a shared 256-d space; swappable text encoder                                                     |
| `prism/index.py`     | Sharded IVF, int8 + f16 tiers, filters, adaptive nprobe, mmap persistence                                                           |
| `prism/pipeline.py`  | Parallel, resumable ingest; index build; columnar catalog                                                                           |
| `prism/engine.py`    | Search, similar, inspect (re-fetch from source), aggregate; place gazetteer                                                         |
| `prism/agent.py`     | Tool registry, policy harness, Claude and rule planners                                                                             |
| `prism/server.py`    | `/search`, `/similar`, `/item/<id>`, `/agent` JSON API + `static/index.html` UI                                                                           |
| `bench.py`, `tests/` | End-to-end benchmark (`results/bench.json`) and the test suite                                                                           |

## Run

```bash
pip install -e .                           # installs numpy and the `prism` command (use `-e ".[dev]"` for pytest + ruff)
prism build --items 1000000 --workers 8 --out data
prism ask --out data "Show burned areas in the Angeles National Forest since 2025"
prism serve --out data --port 8080         # binds to 127.0.0.1; see below to expose it
```

`python -m prism ...` works identically. Requires Python 3.10+.

**Optional: Claude as planner.** Set `ANTHROPIC_API_KEY` before `ask` / `serve` and Claude drives the tool loop instead of the deterministic rule planner (override the model with `PRISM_MODEL`). Without the key the offline rule planner is used.

**Serving beyond localhost.** `prism serve --host 0.0.0.0 --token <secret>` (or `PRISM_TOKEN`) requires `Authorization: Bearer <secret>` on POST endpoints. Do not expose `/agent` without a token when an API key is set.

**"Today".** The planner's current date is the newest item in the index; override with `--today YYYY-MM-DD` or `PRISM_TODAY`.

**Rebuild after upgrading.** The synthetic classes changed (see Caveats), so an index built by an older version must be rebuilt.

## Run Tests

```bash
pip install -e ".[dev]"
pytest -q                                  # unit, server, agent (mocked Claude API) and recall-regression tests
ruff check .
python bench.py --items 1000000 --workers 2   # end-to-end benchmark -> results/bench.json (includes machine + commit)
```

The recall test fails below recall@10 = 0.90 and warns below the 0.95 goal.

## Caveats

- The data is synthetic. Each modality now includes confusable sibling classes (wetland/shrubland/recovering burn, AAV Rep/Cas12a, CD8 T/macrophage/plasma cells) with overlapping signal and 4% label noise, which brings held-out precision to ~0.95–0.98 — still far easier than real data. The scale, latency and recall numbers are real measurements on this machine.
- The text encoder is lexical (hashed words + char n-grams), so paraphrases with no shared vocabulary will miss. It sits behind one interface, so a sentence-embedding model drops in.
- The Claude planner path is implemented but does not show a particular advantage, which may solely be due to the synthetic data itself.

## Path to production

Real readers drop in at `synth.GENERATORS`: Sentinel-2 COGs via STAC, FASTA/BAM, AnnData. Foundation-model encoders (Prithvi / SatMAE, DNABERT-2 / Nucleotide Transformer, scGPT / Geneformer) replace the featurizers, and the ridge heads become contrastive projections. Shards go one per node behind the same `Shard.search` contract, and the pool becomes Ray or k8s Jobs.
