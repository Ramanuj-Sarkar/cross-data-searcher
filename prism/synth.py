"""Synthetic data sources for three modalities + paired captions.

These stand in for real object-store readers (Sentinel-2 COGs, FASTA/BAM, AnnData).
Each generator is deterministic given (seed) so distributed workers can
"fetch" a batch from a descriptor alone — the same contract a real
reader would have (descriptor = URI + byte range).
"""
from __future__ import annotations
import zlib
import numpy as np

# ----------------------------------------------------------------------------
# Satellite: 5-band multispectral tiles (Blue, Green, Red, NIR, SWIR), 32x32
# ----------------------------------------------------------------------------
SAT_CLASSES = ["forest", "water", "urban", "desert", "cropland", "burn_scar", "cloud"]
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
}
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


def gen_sat(n: int, seed: int, labels=None):
    rng = np.random.default_rng(seed)
    if labels is None:
        labels = rng.integers(0, len(SAT_CLASSES), n)
    tiles = np.empty((n, TILE, TILE, 5), np.float32)
    lat = np.empty(n, np.float32); lon = np.empty(n, np.float32)
    for ci, cname in enumerate(SAT_CLASSES):
        idx = np.flatnonzero(labels == ci)
        m = len(idx)
        if m == 0:
            continue
        base = np.array(_SPECTRA[cname], np.float32)[None, :] * rng.normal(1, 0.08, (m, 5)).astype(np.float32)
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
        elif cname == "burn_scar":  # patchy remnants of forest
            frac = _smooth(rng, m, 3)
            forest = np.array(_SPECTRA["forest"], np.float32)
            t = np.where((frac > 0.45)[..., None], forest * (1 + 0.2 * tex[..., None]), t)
        elif cname == "cloud":  # partial cloud over random ground
            cov = (_smooth(rng, m, 3) + 1) / 2
            ground = np.array(_SPECTRA["urban"], np.float32)
            a = np.clip(cov * 1.6 - 0.2, 0, 1)[..., None]
            t = a * t + (1 - a) * ground
        illum = rng.uniform(0.85, 1.15, (m, 1, 1, 1)).astype(np.float32)
        tiles[idx] = t * illum + rng.normal(0, 0.008, t.shape).astype(np.float32)
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
DNA_CLASSES = ["aav_capsid", "lentiviral_backbone", "crispr_cas9", "human_mito",
               "resistance_plasmid", "bacteriophage"]
_GC = {"aav_capsid": 0.55, "lentiviral_backbone": 0.45, "crispr_cas9": 0.50,
       "human_mito": 0.44, "resistance_plasmid": 0.52, "bacteriophage": 0.49}
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
        _FAM_CACHE[name] = (p, np.cumsum(T, 1), motifs)
    return _FAM_CACHE[name]


def gen_dna(n: int, seed: int, labels=None):
    rng = np.random.default_rng(seed)
    if labels is None:
        labels = rng.integers(0, len(DNA_CLASSES), n)
    seqs = np.empty((n, DNA_LEN), np.uint8)
    for ci, cname in enumerate(DNA_CLASSES):
        idx = np.flatnonzero(labels == ci); m = len(idx)
        if m == 0:
            continue
        p, cT, motifs = _fam(cname)
        s = np.empty((m, DNA_LEN), np.uint8)
        s[:, 0] = rng.choice(4, m, p=p)
        u = rng.random((m, DNA_LEN), dtype=np.float32)
        for j in range(1, DNA_LEN):
            s[:, j] = (u[:, j, None] > cT[s[:, j - 1]]).sum(1)
        rows = np.arange(m)[:, None]
        for _ in range(int(rng.integers(4, 9))):
            mi = rng.integers(0, len(motifs), m)
            pos = rng.integers(0, DNA_LEN - _MOTIF_LEN, m)
            mot = motifs[mi].copy()
            mut = rng.random(mot.shape) < 0.04
            mot[mut] = rng.integers(0, 4, mut.sum())
            s[rows, pos[:, None] + np.arange(_MOTIF_LEN)] = mot
        seqs[idx] = s
    mut = rng.random(seqs.shape) < 0.01
    seqs[mut] = rng.integers(0, 4, mut.sum())
    days = rng.integers(0, N_DAYS, n).astype(np.int32)
    nan = np.full(n, np.nan, np.float32)
    return seqs, labels.astype(np.int16), {"lat": nan, "lon": nan.copy(), "t": days}


def dna_to_str(codes) -> str:
    return "".join("ACGT"[c] for c in codes)


# ----------------------------------------------------------------------------
# Single-cell RNA-seq: gamma-Poisson counts over 2000 genes, 8 cell types
# ----------------------------------------------------------------------------
CELL_CLASSES = ["t_cell", "b_cell", "monocyte", "nk_cell", "hepatocyte", "neuron",
                "cardiomyocyte", "hsc"]
_MARKERS = {
    "t_cell": ["CD3E", "CD3D", "IL7R", "CD2"],
    "b_cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "monocyte": ["CD14", "LYZ", "S100A8", "FCN1"],
    "nk_cell": ["NKG7", "GNLY", "KLRD1", "PRF1"],
    "hepatocyte": ["ALB", "APOA1", "TTR", "SERPINA1"],
    "neuron": ["SNAP25", "SYT1", "RBFOX3", "STMN2"],
    "cardiomyocyte": ["TNNT2", "MYH7", "ACTC1", "MYL7"],
    "hsc": ["CD34", "KIT", "PROM1", "THY1"],
}
N_GENES = 2000
MARKER_GENES = [g for c in CELL_CLASSES for g in _MARKERS[c]]
GENES = MARKER_GENES + [f"GENE{i:04d}" for i in range(N_GENES - len(MARKER_GENES))]
_CELL_PROFILES = None


def _profiles():
    global _CELL_PROFILES
    if _CELL_PROFILES is None:
        r = np.random.default_rng(7)
        mu = r.lognormal(-1, 1.5, N_GENES)
        mu[: len(MARKER_GENES)] = np.median(mu) * 0.3
        P = []
        for ci, c in enumerate(CELL_CLASSES):
            mult = np.ones(N_GENES)
            mult[[GENES.index(g) for g in _MARKERS[c]]] = 40
            prog = r.choice(np.arange(len(MARKER_GENES), N_GENES), 60, replace=False)
            mult[prog] *= 4
            q = mu * mult
            P.append(q / q.sum())
        _CELL_PROFILES = np.array(P, np.float64)
    return _CELL_PROFILES


def gen_cell(n: int, seed: int, labels=None):
    rng = np.random.default_rng(seed)
    if labels is None:
        labels = rng.integers(0, len(CELL_CLASSES), n)
    P = _profiles()[labels]
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
        "hsc": ["hematopoietic stem cell", "CD34 positive stem and progenitor cell", "HSC expressing KIT and PROM1",
                "bone marrow stem cell", "blood stem progenitor cells"],
    },
}
CLASSES = {"sat": SAT_CLASSES, "dna": DNA_CLASSES, "cell": CELL_CLASSES}
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
