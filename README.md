# PRISM — one search across siloed R&D data

**The problem.** A biotech's experimental data lives in separate systems that don't talk: sequencing output in one store, imaging in another, single-cell expression in a third. Asking "show me everything relevant to this question" means querying each silo by hand, in its own format, with no shared vocabulary and no single trail back to source. Discovery work that should take minutes takes an afternoon of glue code.

**What PRISM does.** It puts every data type — here satellite imagery, DNA sequences and single-cell RNA — into one searchable space, so a plain-language question searches all of them at once. An *auditable* agent composes the answer, and every claim is **grounding-checked**: any item it cites that the tools never actually returned gets flagged, and every step (query, latency, result hash) is logged. That provenance-and-audit layer is the headline — it's the host's own domain (an enterprise agentic harness) and the thing R&D teams can't get from grep across folders.

**Measured.** On ~1M items and 2 ingest workers (Apple M-series laptop, 12 cores / 32 GB): ingest **16.7k items/s**, search **1.6 ms p50 at 0.996 recall@10**, 100% grounded agent answers.

**How it works (one paragraph).** Each data type gets a domain-aware featurizer: spectral indices (NDVI, NBR, NDWI) for imagery, k-mer spectra for DNA, normalized expression for cells. A closed-form ridge head per type aligns those features to the text embedding of paired descriptions — CLIP-style alignment that fits in ~2 seconds. Vectors land in a sharded IVF index with three tiers: int8 hot storage, a float16 rerank tier, and metadata filters, all memory-mapped. The data is synthetic but structurally realistic, and every item re-fetches from its batch descriptor for provenance.

## Results (999,424 items, 2 ingest workers, 12-core arm64 Mac / 32 GB, numpy only)

Machine, Python/numpy versions, git commit and data profile are recorded under `environment` / `data_profile` in `results/bench.json`. **Data profile v2** (13 / 11 / 14 classes per modality, 10% label noise, weaker signals — see Caveats) is the current default; the **v1 baseline** (10 / 8 / 11 classes, 4% noise, stronger signals) is kept in `results/bench_baseline.json` for comparison. The original 2 vCPU / 7 GB numbers (9.3k items/s) are not comparable to either.

| Stage                                                | v2 (current)                                                  | v1 baseline          |
|------------------------------------------------------|---------------------------------------------------------------|----------------------|
| Encoder fit (3 modalities)                           | 2.3 s                                                         | 3.1 s                |
| Parallel ingest (fetch → featurize → encode → stage) | **16,731 items/s**, batch p95 182 ms, 2 workers               | 17,426 items/s       |
| Resume after simulated crash                         | 878/976 batches skipped, 98 recomputed in 6.3 s, byte-identical, 2nd resume = 0 items | not measured |
| Index build (k-means 4096 lists, 4 shards)           | 17 s · hot int8 tier 268 MiB · f16 rerank tier 488 MiB (mmap) | 18 s                 |
| ANN, nprobe 32 + rerank                              | **recall@10 0.996, p50 1.6 ms** (brute force: 10.0 ms)        | 0.988, 1.9 ms        |
| ANN, nprobe 64 + rerank                              | recall@10 1.000, p50 2.6 ms, scans 1.8% of corpus             | 0.998, 2.9 ms        |
| Filtered search (modality + bbox + date)             | 2.0 ms p50                                                    | 2.2 ms               |
| Held-out text → sat / dna / cell precision@10        | 0.812 / 0.891 / 0.914                                         | 0.945 / 0.975 / 0.982 |
| Agent runs (3 multi-intent questions)                | 100% grounded, 80/80 claims verified, 4–8 tool calls, 0.14–0.31 s | 100% grounded, 4–8 calls |

The recall table shows why the rerank tier exists. Neighbours in dense clusters are near-ties, so int8 scoring alone caps recall at 0.83–0.86 (nprobe 16–64); the f16 rerank of an 8× shortlist lifts nprobe 16 to 0.96 and nprobe 32 to 0.996 for under 1 ms more.

**Resume.** `bench.py` simulates a crash near the end of ingest (last 10% of batches deleted, manifest tail torn) and measures the resume: only the lost batches may be recomputed, they must equal the originals exactly, and a second resume must do nothing.

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

**Rebuild after upgrading.** The synthetic classes and signal model changed (data profile v2), so an index built by an older version must be rebuilt.

## Run Tests

```bash
pip install -e ".[dev]"
pytest -q                                  # unit, server, agent (mocked Claude API) and recall-regression tests
ruff check .
python bench.py --items 1000000 --workers 2   # end-to-end benchmark -> results/bench.json (includes machine + commit)
python bench.py --ci                       # 65k-item subset with pass/fail thresholds (what CI runs)
```

The recall test fails below recall@10 = 0.90 and warns below the 0.95 goal.

## Grounding and structured citations

Every answer is checked by the harness. Cite items as `#<id>` or, better, `#<id>[field=value, ...]`, e.g. `#12[ndvi=0.42, region=malibu]`:

- an id no tool returned makes the answer **ungrounded**;
- a field value that contradicts the stored tool results (numbers are compared at the precision the claim states, never tighter than the catalog's 3-decimal rounding; text case-insensitively; `band_means.nir` reaches nested `inspect` fields) also makes it ungrounded;
- a field or item that cannot be checked is listed under `claims.unverifiable` without failing the answer.

The rule planner emits this format itself; the Claude planner is instructed to. The verdict (`grounded`, `ungrounded_ids`, `claims`) is in the API response and the JSONL trace.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request, on Python 3.10 and 3.13: `ruff check`, `pytest`, then `python bench.py --ci` — a ~15 s, 65k-item benchmark that fails the build if any threshold in `bench.CI_THRESHOLDS` is missed (recall@10 ≥ 0.90, per-modality precision@10 ≥ 0.75, search and ingest speed floors, 100% grounded agent answers with no contradicted claims, and the crash-resume checks). Results are uploaded as the `bench-ci-py<version>` artifact.

## Caveats

- The data is synthetic. Each modality includes confusable sibling classes (e.g. wetland/water/forest, AAV capsid/Rep, CD4/CD8/T cells), 10% of items are generated from a sibling class while keeping their label, and sibling signal is blended into every item. That brings held-out precision to ~0.81–0.91 (v1: 0.95–0.98) — harder, but still easier than real data. The scale, latency and recall numbers are real measurements on this machine.
- The text encoder is lexical (hashed words + char n-grams), so paraphrases with no shared vocabulary will miss. It sits behind one interface, so a sentence-embedding model drops in.
- The Claude planner path is implemented but does not show a particular advantage, which may solely be due to the synthetic data itself.

## Path to production

Real readers drop in at `synth.GENERATORS`: Sentinel-2 COGs via STAC, FASTA/BAM, AnnData. Foundation-model encoders (Prithvi / SatMAE, DNABERT-2 / Nucleotide Transformer, scGPT / Geneformer) replace the featurizers, and the ridge heads become contrastive projections. Shards go one per node behind the same `Shard.search` contract, and the pool becomes Ray or k8s Jobs.
