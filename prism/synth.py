"""Synthetic data sources for three modalities + paired captions.

These stand in for real object-store readers (Sentinel-2 COGs, FASTA/BAM, AnnData).
Each generator is deterministic given (seed) so distributed workers can
"fetch" a batch from a descriptor alone — the same contract a real
reader would have (descriptor = URI + byte range).

Item addressability: a batch is generated as independent CHUNK-item chunks, each seeded from
(batch seed, chunk index). Any single item can therefore be re-fetched by generating just its chunk
(gen_chunk), bit-identical to the batch path, instead of regenerating the whole batch.
"""
from __future__ import annotations

import zlib

import numpy as np

# ----------------------------------------------------------------------------
# Satellite: 5-band multispectral tiles (Blue, Green, Red, NIR, SWIR), 32x32
# ----------------------------------------------------------------------------
# The last three are deliberately confusable with earlier classes (wetland~water/forest, shrubland~desert/forest,
# recovering_burn~burn_scar/forest): overlapping spectra and geography, so retrieval is not trivially separable.
SAT_CLASSES = ["forest", "water", "urban", "desert", "cropland", "burn_scar", "cloud",
               "wetland", "shrubland", "recovering_burn", "grassland", "suburban", "salt_flat"]
BANDS = ["B", "G", "R", "NIR", "SWIR"]
TILE = 32
_SPECTRA = {
    "forest":    [0.04, 0.08, 0.05, 0.45, 0.18],
    "water":     [0.08, 0.07, 0.04, 0.02, 0.01],
    "urban":     [0.15, 0.17, 0.19, 0.24, 0.30],
    "desert":    [0.20, 0.28, 0.36, 0.42, 0.50],
    "cropland":  [0.06, 0.10, 0.08, 0.38, 0.22],
    "burn_scar": [0.05, 0.06, 0.07, 0.10, 0.28],
    "cloud":     [0.70, 0.70, 0.72, 0.70, 0.55],
    "wetland":   [0.05, 0.08, 0.05, 0.28, 0.09],
    "shrubland": [0.09, 0.14, 0.16, 0.32, 0.30],
    "recovering_burn": [0.05, 0.07, 0.07, 0.22, 0.24],
    "grassland": [0.07, 0.11, 0.10, 0.34, 0.24],
    "suburban":  [0.12, 0.15, 0.16, 0.29, 0.27],
    "salt_flat": [0.30, 0.36, 0.42, 0.46, 0.44],
}
_ROAD = np.array([0.10, 0.11, 0.12, 0.14, 0.20], np.float32)
_SOIL = np.array([0.12, 0.16, 0.20, 0.25, 0.30], np.float32)
# (lat, lon, sigma) — where each land-cover class tends to appear in SoCal
_GEO = {
    "forest":    [(34.35, -118.05, 0.15)],                       # Angeles NF
    "water":     [(33.90, -118.60, 0.10), (34.40, -118.55, 0.05)],  # SM Bay, Castaic
    "urban":     [(34.02, -118.30, 0.15)],                       # LA basin
    "desert":    [(34.90, -116.50, 0.40)],                       # Mojave
    "cropland":  [(32.95, -115.55, 0.20)],                       # Imperial Valley
    "burn_scar": [(34.30, -118.10, 0.12), (34.06, -118.75, 0.06)],  # ANF, Malibu
    "cloud":     [(33.90, -117.50, 1.00)],
    "wetland":   [(33.90, -118.55, 0.12), (34.40, -118.55, 0.08)],
    "shrubland": [(34.45, -118.20, 0.30)],
    "recovering_burn": [(34.30, -118.10, 0.12), (34.06, -118.75, 0.06)],
    "grassland": [(34.20, -118.40, 0.30), (33.10, -115.70, 0.20)],
    "suburban":  [(34.10, -118.00, 0.25)],
    "salt_flat": [(35.10, -116.40, 0.30), (33.30, -115.80, 0.10)],
}

# Fraction of items whose content is drawn from a confusable class while keeping their declared label
# (annotation noise). Applied only when labels are not supplied, so training data stays clean.
LABEL_NOISE = 0.10  # v1: 0.04
_CONFUSABLE = {  # class -> siblings whose content it can borrow
    "sat": {"water": ["wetland"], "wetland": ["water", "forest"], "forest": ["shrubland", "wetland"],
            "desert": ["shrubland", "salt_flat"], "shrubland": ["desert", "forest", "grassland"],
            "burn_scar": ["recovering_burn"], "recovering_burn": ["burn_scar", "forest"],
            "cropland": ["grassland"], "grassland": ["cropland", "shrubland"], "urban": ["suburban"],
            "suburban": ["urban", "grassland"], "salt_flat": ["desert", "cloud"]},
    "dna": {"aav_capsid": ["aav_rep"], "aav_rep": ["aav_capsid"], "crispr_cas9": ["cas12a_cassette"],
            "cas12a_cassette": ["crispr_cas9"], "lentiviral_backbone": ["lentiviral_gag"],
            "lentiviral_gag": ["lentiviral_backbone"], "human_mito": ["rodent_mito"], "rodent_mito": ["human_mito"],
            "bacteriophage": ["phage_t4"], "phage_t4": ["bacteriophage"]},
    "cell": {"t_cell": ["cd8_t_cell", "cd4_t_cell"], "cd8_t_cell": ["t_cell"], "cd4_t_cell": ["t_cell"],
             "monocyte": ["macrophage", "dendritic_cell"], "macrophage": ["monocyte"], "dendritic_cell": ["monocyte"],
             "b_cell": ["plasma_cell", "memory_b_cell"], "plasma_cell": ["b_cell"], "memory_b_cell": ["b_cell"]},
}


def _content_labels(mod, labels, rng, classes):
    """Labels used to synthesize content: mostly the declared ones, with LABEL_NOISE swapped for a confusable sibling."""
    out = labels.copy()
    flip = rng.random(len(labels)) < LABEL_NOISE
    for cname, sibs in _CONFUSABLE[mod].items():
        idx = np.flatnonzero(flip & (labels == classes.index(cname)))
        if len(idx):
            out[idx] = [classes.index(sibs[i]) for i in rng.integers(0, len(sibs), len(idx))]
    return out


def _first_sibling(mod, cname):
    sibs = _CONFUSABLE[mod].get(cname)
    return sibs[0] if sibs else None


# Data profile. v1 = 10 sat / 8 dna / 11 cell classes, 4% label noise, strong signals (baseline in
# results/bench_baseline.json). v2 = more confusable siblings, 10% label noise, weaker signals.
PROFILE = "v2"
PIXEL_NOISE = 0.02      # v1: 0.008
ILLUM = (0.78, 1.22)    # v1: (0.85, 1.15)
SPECTRAL_MIX = 0.35     # max share of a sibling class's spectrum/profile blended into an item
DAY0 = np.datetime64("2024-01-01")
N_DAYS = 1004  # through 2026-10-01


def _smooth(rng, n, grid, size=TILE):
    """Low-frequency noise in [-1, 1]: random grid, bilinear-ish upsample."""
    g = rng.uniform(-1, 1, (n, grid + 1, grid + 1)).astype(np.float32)
    x = np.linspace(0, grid, size, dtype=np.float32)
    i0 = np.floor(x).astype(int).clip(0, grid - 1)
    f = (x - i0)[None, :]
    rows = g[:, i0, :] * (1 - f[..., None]) + g[:, i0 + 1, :] * f[..., None]
    out = rows[:, :, i0] * (1 - f[:, None, :]) + rows[:, :, i0 + 1] * f[:, None, :]
    return out


def _sat_chunk(n: int, rng, labels=None):
    gen = labels
    if labels is None:
        labels = rng.integers(0, len(SAT_CLASSES), n)
        gen = _content_labels("sat", labels, rng, SAT_CLASSES)
    tiles = np.empty((n, TILE, TILE, 5), np.float32)
    lat = np.empty(n, np.float32); lon = np.empty(n, np.float32)
    for ci, cname in enumerate(SAT_CLASSES):
        idx = np.flatnonzero(gen == ci)
        m = len(idx)
        if m == 0:
            continue
        spec = np.array(_SPECTRA[cname], np.float32)[None, :]
        sib = _first_sibling("sat", cname)
        if sib:  # blend a random share of the sibling's spectrum into each item
            w = rng.uniform(0, SPECTRAL_MIX, (m, 1)).astype(np.float32)
            spec = (1 - w) * spec + w * np.array(_SPECTRA[sib], np.float32)[None, :]
        base = spec * rng.normal(1, 0.08, (m, 5)).astype(np.float32)
        tex = _smooth(rng, m, 8 if cname in ("forest", "urban") else 4)
        amp = {"water": 0.08, "desert": 0.15, "forest": 0.25}.get(cname, 0.18)
        t = base[:, None, None, :] * (1 + amp * tex[..., None])
        if cname == "urban":  # road grid
            yy, xx = np.mgrid[:TILE, :TILE]
            period = rng.integers(6, 11, m)
            mask = ((yy[None] % period[:, None, None]) < 2) | ((xx[None] % period[:, None, None]) < 2)
            t = np.where(mask[..., None], _ROAD, t)
        elif cname == "cropland":  # striped fields
            period = rng.integers(4, 9, m)
            coord = np.where(rng.random(m)[:, None, None] < 0.5, np.mgrid[:TILE, :TILE][0][None], np.mgrid[:TILE, :TILE][1][None])
            bare = ((coord // period[:, None, None]) % 2 == 1)
            t = np.where(bare[..., None], _SOIL * (1 + 0.1 * tex[..., None]), t)
        elif cname in ("burn_scar", "recovering_burn"):  # patchy remnants of forest (more regrowth when recovering)
            frac = _smooth(rng, m, 3)
            forest = np.array(_SPECTRA["forest"], np.float32)
            t = np.where((frac > (0.45 if cname == "burn_scar" else 0.0))[..., None], forest * (1 + 0.2 * tex[..., None]), t)
        elif cname == "cloud":  # partial cloud over random ground
            cov = (_smooth(rng, m, 3) + 1) / 2
            ground = np.array(_SPECTRA["urban"], np.float32)
            a = np.clip(cov * 1.6 - 0.2, 0, 1)[..., None]
            t = a * t + (1 - a) * ground
        illum = rng.uniform(*ILLUM, (m, 1, 1, 1)).astype(np.float32)
        tiles[idx] = t * illum + rng.normal(0, PIXEL_NOISE, t.shape).astype(np.float32)
        # geo
        centers = _GEO[cname]
        pick = rng.integers(0, len(centers), m)
        c = np.array(centers, np.float32)[pick]
        lat[idx] = c[:, 0] + rng.normal(0, 1, m) * c[:, 2]
        lon[idx] = c[:, 1] + rng.normal(0, 1, m) * c[:, 2]
    np.clip(tiles, 0, 1, out=tiles)
    days = rng.integers(0, N_DAYS, n).astype(np.int32)
    return tiles, labels.astype(np.int16), {"lat": lat, "lon": lon, "t": days}


# ----------------------------------------------------------------------------
# DNA: synthetic sequences from family-specific Markov chains + planted motifs
# ----------------------------------------------------------------------------
# aav_rep and cas12a_cassette share GC content and planted motifs with aav_capsid / crispr_cas9.
DNA_CLASSES = ["aav_capsid", "lentiviral_backbone", "crispr_cas9", "human_mito",
               "resistance_plasmid", "bacteriophage", "aav_rep", "cas12a_cassette",
               "lentiviral_gag", "rodent_mito", "phage_t4"]
_SHARES_MOTIFS = {"aav_rep": "aav_capsid", "cas12a_cassette": "crispr_cas9", "lentiviral_gag": "lentiviral_backbone",
                  "rodent_mito": "human_mito", "phage_t4": "bacteriophage"}
CHUNK = 128            # items per independently seeded chunk (see module docstring)
MOTIF_COUNT = (2, 6)   # v1: (4, 9) motif insertions per sequence
MOTIF_MUT = 0.10       # v1: 0.04
BASE_MUT = 0.03        # v1: 0.01
_GC = {"aav_capsid": 0.55, "lentiviral_backbone": 0.45, "crispr_cas9": 0.50,
       "human_mito": 0.44, "resistance_plasmid": 0.52, "bacteriophage": 0.49,
       "aav_rep": 0.55, "cas12a_cassette": 0.50,
       "lentiviral_gag": 0.45, "rodent_mito": 0.44, "phage_t4": 0.49}
DNA_LEN = 1000
_MOTIF_LEN = 24
_FAM_CACHE: dict = {}


def _fam(name):
    if name not in _FAM_CACHE:
        r = np.random.default_rng(zlib.crc32(name.encode()))
        gc = _GC[name]
        p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
        T = np.stack([r.dirichlet(p * 25 + 0.5) for _ in range(4)])
        motifs = r.integers(0, 4, (5, _MOTIF_LEN)).astype(np.uint8)
        if name in _SHARES_MOTIFS:  # 4 of 5 motifs inherited from the sibling family
            motifs[:4] = _fam(_SHARES_MOTIFS[name])[2][:4]
        _FAM_CACHE[name] = (p, np.cumsum(T, 1), motifs)
    return _FAM_CACHE[name]


_CHAINS = None


def _chains():
    """All families' Markov chains stacked, so one 1000-step loop serves every class in a chunk."""
    global _CHAINS
    if _CHAINS is None:
        fams = [_fam(c) for c in DNA_CLASSES]
        _CHAINS = (np.stack([np.cumsum(f[0]) for f in fams]).astype(np.float32),
                   np.stack([f[1] for f in fams]).astype(np.float32).reshape(-1, 4))  # row = class * 4 + previous base
    return _CHAINS


def _dna_chunk(n: int, rng, labels=None):
    gen = labels
    if labels is None:
        labels = rng.integers(0, len(DNA_CLASSES), n)
        gen = _content_labels("dna", labels, rng, DNA_CLASSES)
    init, rows = _chains()
    u = rng.random((DNA_LEN, n), dtype=np.float32)
    steps = np.empty((DNA_LEN, n), np.uint8)
    steps[0] = (u[0][:, None] > init[gen]).sum(1)
    base = gen.astype(np.int64) * 4
    for j in range(1, DNA_LEN):
        steps[j] = (u[j][:, None] > rows[base + steps[j - 1]]).sum(1)
    seqs = np.ascontiguousarray(steps.T)
    for ci, cname in enumerate(DNA_CLASSES):
        idx = np.flatnonzero(gen == ci); m = len(idx)
        if m == 0:
            continue
        motifs = _fam(cname)[2]
        s = seqs[idx]
        r_ = np.arange(m)[:, None]
        for _ in range(int(rng.integers(*MOTIF_COUNT))):
            mi = rng.integers(0, len(motifs), m)
            pos = rng.integers(0, DNA_LEN - _MOTIF_LEN, m)
            mot = motifs[mi].copy()
            mut = rng.random(mot.shape) < MOTIF_MUT
            mot[mut] = rng.integers(0, 4, mut.sum())
            s[r_, pos[:, None] + np.arange(_MOTIF_LEN)] = mot
        seqs[idx] = s
    mut = rng.random(seqs.shape) < BASE_MUT
    seqs[mut] = rng.integers(0, 4, mut.sum())
    days = rng.integers(0, N_DAYS, n).astype(np.int32)
    nan = np.full(n, np.nan, np.float32)
    return seqs, labels.astype(np.int16), {"lat": nan, "lon": nan.copy(), "t": days}


def dna_to_str(codes) -> str:
    return "".join("ACGT"[c] for c in codes)


# ----------------------------------------------------------------------------
# Single-cell RNA-seq: gamma-Poisson counts over 2000 genes, 8 cell types
# ----------------------------------------------------------------------------
# cd8_t_cell, macrophage and plasma_cell share markers and most of their expression program with t_cell, monocyte, b_cell.
CELL_CLASSES = ["t_cell", "b_cell", "monocyte", "nk_cell", "hepatocyte", "neuron",
                "cardiomyocyte", "hsc", "cd8_t_cell", "macrophage", "plasma_cell",
                "cd4_t_cell", "dendritic_cell", "memory_b_cell"]
_SIBLING = {"cd8_t_cell": "t_cell", "macrophage": "monocyte", "plasma_cell": "b_cell",
            "cd4_t_cell": "t_cell", "dendritic_cell": "monocyte", "memory_b_cell": "b_cell"}
MARKER_FOLD = 15       # v1: 40-fold marker up-regulation
_MARKERS = {
    "t_cell": ["CD3E", "CD3D", "IL7R", "CD2"],
    "b_cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "monocyte": ["CD14", "LYZ", "S100A8", "FCN1"],
    "nk_cell": ["NKG7", "GNLY", "KLRD1", "PRF1"],
    "hepatocyte": ["ALB", "APOA1", "TTR", "SERPINA1"],
    "neuron": ["SNAP25", "SYT1", "RBFOX3", "STMN2"],
    "cardiomyocyte": ["TNNT2", "MYH7", "ACTC1", "MYL7"],
    "hsc": ["CD34", "KIT", "PROM1", "THY1"],
    "cd8_t_cell": ["CD3E", "CD8A", "CD8B", "GZMK"],
    "macrophage": ["CD68", "CD14", "C1QA", "MSR1"],
    "plasma_cell": ["CD79A", "MZB1", "JCHAIN", "SDC1"],
    "cd4_t_cell": ["CD3E", "CD4", "IL7R", "CCR7"],
    "dendritic_cell": ["CD14", "CD1C", "FCER1A", "CLEC10A"],
    "memory_b_cell": ["CD19", "CD27", "MS4A1", "AIM2"],
}
N_GENES = 2000
MARKER_GENES = list(dict.fromkeys(g for c in CELL_CLASSES for g in _MARKERS[c]))  # shared markers listed once
GENES = MARKER_GENES + [f"GENE{i:04d}" for i in range(N_GENES - len(MARKER_GENES))]
_CELL_PROFILES = None


def _profiles():
    global _CELL_PROFILES
    if _CELL_PROFILES is None:
        r = np.random.default_rng(7)
        mu = r.lognormal(-1, 1.5, N_GENES)
        mu[: len(MARKER_GENES)] = np.median(mu) * 0.3
        P = []; progs: dict[str, np.ndarray] = {}
        pool = np.arange(len(MARKER_GENES), N_GENES)
        for c in CELL_CLASSES:
            mult = np.ones(N_GENES)
            mult[[GENES.index(g) for g in _MARKERS[c]]] = MARKER_FOLD
            prog = r.choice(pool, 60, replace=False)
            if c in _SIBLING:  # keep 50 of the sibling's 60 program genes
                prog = np.concatenate([r.choice(progs[_SIBLING[c]], 50, replace=False), prog[:10]])
            progs[c] = prog
            mult[prog] *= 4
            q = mu * mult
            P.append(q / q.sum())
        _CELL_PROFILES = np.array(P, np.float64)
    return _CELL_PROFILES


def _cell_chunk(n: int, rng, labels=None):
    gen = labels
    if labels is None:
        labels = rng.integers(0, len(CELL_CLASSES), n)
        gen = _content_labels("cell", labels, rng, CELL_CLASSES)
    prof = _profiles()
    P = prof[gen]
    sib = np.array([CELL_CLASSES.index(sc) if (sc := _first_sibling("cell", c)) else -1 for c in CELL_CLASSES])[gen]
    w = (rng.uniform(0, SPECTRAL_MIX, n) * (sib >= 0))[:, None]  # blend part of the sibling's expression profile
    P = (1 - w) * P + w * prof[np.where(sib >= 0, sib, gen)]
    lib = rng.lognormal(np.log(2500), 0.5, n)[:, None]
    rate = P * lib * rng.gamma(2.0, 0.5, P.shape)
    counts = rng.poisson(rate).astype(np.int32)
    days = rng.integers(0, N_DAYS, n).astype(np.int32)
    nan = np.full(n, np.nan, np.float32)
    return counts, labels.astype(np.int16), {"lat": nan, "lon": nan.copy(), "t": days}


# ----------------------------------------------------------------------------
# Captions (paired text used to align every modality into the text space)
# ----------------------------------------------------------------------------
CAPTIONS = {
    "sat": {
        "_mod": ["satellite image", "multispectral satellite tile", "aerial imagery",
                 "remote sensing scene", "earth observation image"],
        "forest": ["dense forest canopy", "healthy green vegetation", "tree cover in mountain woodland",
                   "high NDVI vegetated area", "conifer and chaparral forest", "lush vegetation"],
        "water": ["open water body", "reservoir or lake surface", "coastal ocean water",
                  "dark water with low reflectance", "bay and shoreline water", "high NDWI water"],
        "urban": ["dense urban development", "city blocks and road grid", "built-up area with buildings",
                  "impervious surfaces and streets", "downtown urban core", "residential neighborhoods and freeways"],
        "desert": ["arid desert terrain", "bare sand and rock", "dry barren land", "sparse desert scrub",
                   "bright soil with no vegetation", "Mojave desert landscape"],
        "cropland": ["agricultural cropland fields", "irrigated farmland rows", "crop fields in a striped pattern",
                     "cultivated agriculture parcels", "farm fields with crops", "row crops on farmland"],
        "burn_scar": ["wildfire burn scar", "charred burned area after fire", "fire damaged vegetation",
                      "post fire burn severity", "scorched land from a wildfire", "low NBR burned forest"],
        "wetland": ["wetland marsh with standing water", "swampy vegetated shoreline", "marshland and wet vegetation",
                    "seasonally flooded wetland", "riparian marsh near open water", "waterlogged vegetated land"],
        "shrubland": ["shrubland with sparse bushes", "chaparral scrub on dry hillsides", "dry brush and low shrubs",
                      "semi-arid shrub vegetation", "scrubland between desert and forest", "sparse woody shrubs"],
        "recovering_burn": ["recovering burn area with regrowth", "regrowing vegetation after a wildfire",
                            "old burn scar with green recovery", "post fire vegetation regrowth",
                            "partially recovered burned hillside", "burned land recolonized by plants"],
        "grassland": ["open grassland and pasture", "rolling grass-covered hills", "meadow with low green vegetation",
                      "savanna-like grass and scattered shrubs", "rangeland with seasonal grass", "grazing land"],
        "suburban": ["suburban housing with yards", "low-density residential neighborhoods", "tree-lined suburban streets",
                     "subdivision of houses and lawns", "urban fringe with gardens", "residential area with green lots"],
        "salt_flat": ["dry salt flat playa", "bright evaporite lake bed", "white alkali flats in the desert",
                      "barren playa with salt crust", "dry lakebed with pale soil", "salt pan terrain"],
        "cloud": ["cloud covered scene", "thick cloud cover obscuring ground", "bright white clouds",
                  "cloudy image with haze", "overcast sky occluding land", "cloud contamination"],
    },
    "dna": {
        "_mod": ["DNA sequence", "nucleotide sequence", "genomic sequence read", "DNA fragment", "sequencing read"],
        "aav_capsid": ["AAV capsid gene", "adeno-associated virus capsid", "AAV vector cap sequence",
                       "gene therapy AAV serotype capsid", "recombinant AAV capsid variant"],
        "lentiviral_backbone": ["lentiviral vector backbone", "lentivirus transfer plasmid", "HIV derived lentiviral LTR",
                                "lentiviral packaging sequence", "integrating lentiviral vector"],
        "crispr_cas9": ["CRISPR Cas9 cassette", "Cas9 nuclease expression construct", "guide RNA scaffold and Cas9",
                        "genome editing CRISPR construct", "SpCas9 gene editing sequence"],
        "human_mito": ["human mitochondrial DNA", "mtDNA genome fragment", "mitochondrial genome sequence",
                       "human mitochondria D-loop region", "mitochondrial gene sequence"],
        "resistance_plasmid": ["E. coli plasmid with antibiotic resistance", "ampicillin resistance plasmid",
                               "bacterial cloning vector", "beta-lactamase resistance gene plasmid",
                               "kanamycin resistance plasmid"],
        "aav_rep": ["AAV Rep gene", "adeno-associated virus Rep replication protein", "AAV rep sequence for replication",
                    "gene therapy AAV rep helper cassette", "recombinant AAV Rep78 and Rep52"],
        "cas12a_cassette": ["Cas12a editing cassette", "Cpf1 nuclease CRISPR construct", "CRISPR Cas12a crRNA array",
                            "genome editing with Cas12a", "Cas12a Cpf1 gene editing sequence"],
        "lentiviral_gag": ["lentiviral gag-pol packaging gene", "HIV gag polyprotein lentivirus construct",
                           "lentivirus Gag Pol packaging plasmid", "gag pol rev helper sequence", "lentiviral helper plasmid"],
        "rodent_mito": ["mouse mitochondrial DNA", "rodent mtDNA genome fragment", "murine mitochondrial genome",
                        "rat mitochondria sequence", "mouse mitochondrial gene sequence"],
        "phage_t4": ["T4 bacteriophage genome", "enterobacteria phage T4 DNA", "T4 phage lytic virus sequence",
                     "myovirus phage that infects E. coli", "phage T4 structural genes"],
        "bacteriophage": ["bacteriophage genome", "lambda phage DNA", "phage that infects bacteria",
                          "viral phage sequence", "bacteriophage lytic genes"],
    },
    "cell": {
        "_mod": ["single-cell transcriptome", "scRNA-seq cell profile", "single cell gene expression",
                 "cell expression profile", "single-cell RNA sequencing"],
        "t_cell": ["T cell", "CD3E positive T lymphocyte", "T cells expressing CD3D and IL7R",
                   "cytotoxic or helper T lymphocyte", "adaptive immune T cell"],
        "b_cell": ["B cell", "CD19 positive B lymphocyte", "MS4A1 CD79A B cells",
                   "antibody producing B lymphocyte lineage", "B cells in blood"],
        "monocyte": ["monocyte", "CD14 positive classical monocyte", "myeloid monocytes expressing LYZ",
                     "innate immune monocyte", "S100A8 high myeloid cell"],
        "nk_cell": ["natural killer cell", "NK cell expressing NKG7 and GNLY", "cytotoxic NK lymphocyte",
                    "KLRD1 positive natural killer", "innate lymphoid NK cells"],
        "hepatocyte": ["hepatocyte", "liver hepatocyte expressing ALB", "liver parenchymal cell",
                       "APOA1 and TTR high hepatocytes", "liver cells"],
        "neuron": ["neuron", "neuronal cell expressing SNAP25", "brain neuron with SYT1",
                   "RBFOX3 positive neurons", "nerve cells from brain tissue"],
        "cardiomyocyte": ["cardiomyocyte", "heart muscle cell expressing TNNT2", "cardiac myocyte with MYH7",
                          "heart cardiomyocytes", "ACTC1 positive cardiac muscle cell"],
        "cd8_t_cell": ["CD8 positive cytotoxic T cell", "CD8A and CD3E expressing T lymphocyte", "cytotoxic CD8 T cells",
                       "GZMK CD8B T cells", "killer T lymphocyte"],
        "macrophage": ["macrophage", "CD68 positive tissue macrophage", "C1QA macrophages",
                       "myeloid macrophage expressing CD14", "phagocytic macrophage cell"],
        "plasma_cell": ["plasma cell", "antibody secreting plasma cell", "MZB1 and JCHAIN plasma cells",
                        "SDC1 positive plasma B cell", "immunoglobulin producing plasma cells"],
        "cd4_t_cell": ["CD4 positive helper T cell", "CD4 and IL7R expressing T lymphocyte", "helper T cells",
                       "CCR7 CD4 T cells", "T helper lymphocyte"],
        "dendritic_cell": ["dendritic cell", "CD1C positive dendritic cells", "antigen presenting dendritic cell",
                           "FCER1A myeloid dendritic cell", "CLEC10A dendritic cells"],
        "memory_b_cell": ["memory B cell", "CD27 positive memory B lymphocyte", "class-switched memory B cells",
                          "AIM2 memory B cell", "long-lived memory B lymphocyte"],
        "hsc": ["hematopoietic stem cell", "CD34 positive stem and progenitor cell", "HSC expressing KIT and PROM1",
                "bone marrow stem cell", "blood stem progenitor cells"],
    },
}
CLASSES = {"sat": SAT_CLASSES, "dna": DNA_CLASSES, "cell": CELL_CLASSES}
_CHUNK_FNS = {"sat": _sat_chunk, "dna": _dna_chunk, "cell": _cell_chunk}


def gen_chunk(mod: str, n: int, seed: int, c: int, labels=None):
    """Chunk `c` (items c*CHUNK .. ) of the n-item batch `seed`: (raw, labels, meta). Identical to the same
    slice of GENERATORS[mod](n, seed), at 1/(n/CHUNK) of the cost."""
    lo = c * CHUNK
    if not 0 <= lo < n:
        raise IndexError(f"chunk {c} outside batch of {n}")
    size = min(CHUNK, n - lo)
    sub = None if labels is None else np.asarray(labels)[lo:lo + size]
    return _CHUNK_FNS[mod](size, np.random.default_rng([seed, c]), sub)


def _batch(mod):
    def gen(n: int, seed: int, labels=None):
        parts = [gen_chunk(mod, n, seed, c, labels) for c in range(-(-n // CHUNK))]
        meta = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), meta
    return gen


gen_sat, gen_dna, gen_cell = _batch("sat"), _batch("dna"), _batch("cell")
GENERATORS = {"sat": gen_sat, "dna": gen_dna, "cell": gen_cell}
MODALITIES = ["sat", "dna", "cell"]
MOD_CODE = {m: i for i, m in enumerate(MODALITIES)}


def captions_for(mod: str, labels, seed: int):
    rng = np.random.default_rng(seed)
    C = CAPTIONS[mod]; names = CLASSES[mod]; out = []
    for lab in labels:
        ph = C[names[lab]]
        a, b = rng.choice(len(ph), 2, replace=False)
        out.append(f"{C['_mod'][rng.integers(len(C['_mod']))]} of {ph[a]}, {ph[b]}")
    return out
