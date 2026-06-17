# Textile Fiber Classification via Near-Infrared Spectroscopy

> **From 228-pixel lab spectrometer to 2–4 LEDs — wavelength selection for portable NIR textile identification.**

Near-infrared (NIR) spectroscopy dataset for textile fiber classification, collected with a **TI DLP NIRScan nano** spectrometer (900–1700 nm, 228 bands, 188 samples, 6 classes). The project investigates feature selection with spectral diversity constraints to compress the full spectrum into a handful of key wavelengths, enabling deployment on low-cost portable hardware.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Experiment: Wavelength Selection](#experiment-wavelength-selection)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
git clone https://github.com/SanctusDei/Textile-classification-based-on-near-infrared-spectroscopy.git
cd Textile-classification
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Run final experiment (SG 1st-derivative, k=3, 5 seeds, with figures)
python wavelength_selection/select_wavelengths.py --k 3 --n_seeds 5 --preprocess savgol_1deriv --plot
```

---

## Dataset

### Overview

| Property | Value |
|----------|-------|
| **Samples** | 188 pure-fiber spectra |
| **Spectral range** | 900–1700 nm |
| **Spectral bands** | 228 pixels |
| **Classes** | 6 |
| **Instrument** | TI DLP NIRScan nano (SN 6460024) |
| **Additional data** | 40 blend spectra, 10 background spectra, 132 fabric images |

### Classes

| Class | Count | Swatches | Description |
|-------|-------|----------|-------------|
| Polyester (PET) | 83 | 24 | Synthetic; most common textile fiber |
| Nylon (PA) | 45 | 4 | Synthetic polyamide |
| Cotton | 45 | 4 | Natural cellulose |
| Wool | 5 | 5 | Natural protein (keratin) |
| Acrylic | 5 | 5 | Synthetic (PAN-based) |
| Acetate | 5 | 5 | Semi-synthetic cellulose acetate |
| **Total** | **188** | **47** | |

### Data Collection Protocol

Each swatch was measured under **multiple conditions** to capture intra-class variability:

- **Multiple positions** across the fabric surface
- **Multiple rotation angles** (0°, 90°, 180°, 270°)
- **Multiple fold configurations** (single-layer, double-layer)

### Acquisition Parameters

| Parameter | Value |
|-----------|-------|
| Exposure time | 0.635 ms |
| Repeated scans | 6 (averaged) |
| PGA Gain | 16 |
| Total measurement time | 2.451 s / sample |

### Data Format

Each CSV file contains a **22-line metadata header** followed by 228 rows of spectral data:

| Column | Description |
|--------|-------------|
| `Wavelength (nm)` | Wavelength in nanometers |
| `Absorbance (AU)` | Absorbance in absorbance units |
| `Reference Signal (unitless)` | Background reference intensity |
| `Sample Signal (unitless)` | Raw sample intensity |

### Additional Data

- **40 blend-fabric spectra** across 4 blend ratios:
  - Cotton/Polyester 55:45, Cotton/Polyester 80:20
  - Nylon/Polyester 30:70, Wool/Polyester 35:65
- **10 PVC background spectra** — reference measurements for normalization
- **132 fabric images** (`.bmp`) — visual documentation of measured swatches

---

## Experiment: Wavelength Selection

### Motivation

A lab-grade NIR spectrometer (228-pixel InGaAs array) costs **$2,000+**. By identifying k = 3–10 discriminative wavelengths, we can replace it with **fixed-wavelength LEDs + photodiodes** (~$30–50 per LED), reducing the bill of materials by **94–98%** while maintaining near-perfect classification accuracy.

### The Adjacent-Pixel Problem

On raw absorbance spectra, univariate feature selection methods (ANOVA F-score, Mutual Information) suffer from a critical flaw: NIR absorption peaks span 20–50 nm (6–14 pixels at 3.5 nm/px resolution), creating strong collinearity between adjacent wavelengths. These methods rank all pixels within the same absorption band as "top-k," **collapsing k selections into a single chemical information channel**.

For example, at k=3 on raw spectra:
- **ANOVA** selects 1195, 1199, 1204 nm — all within 9 nm (same C–H overtone)
- Spectral spread: **3 nm** vs. Random spread: **172 ± 75 nm**
- Accuracy: **0.59** vs. Random baseline: **0.80**

### Solution: Spectral Diversity Constraints

Two strategies enforce that selected wavelengths span distinct spectral regions:

| Strategy | Mechanism |
|----------|-----------|
| **Minimum Distance (MinDist, 30 nm)** | Greedy selection: after picking the top-scoring wavelength, exclude all wavelengths within ±30 nm; repeat |
| **Correlation Clustering** | Pre-compute k spectral clusters via hierarchical clustering on 1−\|Pearson r\|; pick the best-scoring wavelength per cluster |

Combined with **Savitzky-Golay 1st-derivative preprocessing** (window=11, polyorder=3) — which removes baseline drift and decorrelates adjacent pixels — these constraints eliminate the redundancy problem entirely.

### Methods Compared (5 total)

| Method | Type | Diversity-Aware |
|--------|------|:---:|
| ANOVA F-score | Univariate filter | — |
| Mutual Information | Univariate filter | — |
| ANOVA + MinDist | Univariate + 30 nm minimum distance | ✅ |
| MI + MinDist | Univariate + 30 nm minimum distance | ✅ |
| ANOVA + Clustering | Univariate + spectral clustering | ✅ |

### Teacher-Guided Pseudo-Label Training

Simulates real-world deployment where ground-truth labels are unavailable:

1. **Teacher**: trained on all 228 wavelengths per-fold (internal 3-fold CV selects best among SVM-RBF, RF, KNN)
2. **Student**: trained on only k selected wavelengths, using the Teacher's predictions as pseudo-labels

The accuracy gap between Teacher (228λ) and Student (kλ) quantifies the cost of wavelength reduction.

### Experimental Rigor

| Design Choice | Purpose |
|---------------|---------|
| Stratified 5-fold CV | Preserves class distribution in every split |
| Feature selection **inside each fold** | Prevents test-set leakage to feature selector |
| Teacher **retrained per fold** | Prevents Teacher from leaking test knowledge to Student |
| Multi-seed (5 seeds) | Reports mean ± std, not single-seed cherry-picking |
| Wilcoxon signed-rank tests (α=0.05) | Statistical significance of method comparisons |
| Consensus wavelength analysis | Jaccard stability index; only stable wavelengths recommended for hardware |

### Baselines

| Baseline | Meaning |
|----------|---------|
| **Teacher (228λ)** | Upper bound — full-spectrum, best model |
| **Random k wavelengths** | Lower bound — randomly chosen bands, KNN classifier |

### Preprocessing Options

| Option | Effect |
|--------|--------|
| `none` | Raw absorbance |
| `savgol` | Savitzky-Golay smoothing (window=11, polyorder=3) |
| `savgol_1deriv` | **Recommended** — baseline correction + peak resolution |
| `savgol_2deriv` | Sharper peak resolution, higher noise |

---

## Key Results

> **Preprocessing: SG 1st-derivative  |  5 seeds × 5-fold CV  |  9 methods compared**

### Accuracy by Wavelength Budget

| k | Best Method | Pseudo Accuracy | vs Teacher (0.9904) | vs Random | Stable λ | Hardware Cost |
|---|-------------|:---:|:---:|:---:|:---:|:---:|
| **3** | MI + MinDist | **0.9733** | −0.017 | **+0.136** | 2 | ~$60–100 |
| **5** | ANOVA + MinDist | **0.9861** | −0.004 | **+0.064** | 4 | ~$120–200 |
| **10** | MI + MinDist | **0.9925** | **+0.002** ✨ | **+0.047** | 4 | ~$120–200 |

> ✨ **At k=10, the Student surpasses the full-spectrum Teacher** — 4 LEDs outperform a $2,000 spectrometer.

### Raw Spectra vs. SG 1st-Derivative (k=3)

| Scenario | Best Method | Accuracy | vs Random | Note |
|----------|-------------|:---:|:---:|------|
| Raw, no diversity constraint | ANOVA F-score | 0.59 | **−0.21** | ❌ 3 adjacent pixels, 1 channel |
| Raw + Clustering | MI + Clustering | **0.89** | **+0.09** | ✅ Clustering rescues raw spectra |
| SG-1deriv, no constraint | MI | 0.96 | +0.12 | ✅ Derivative decorrelates pixels |
| SG-1deriv + MinDist | MI + MinDist | **0.97** | **+0.14** | 🏆 Best overall |

**Key insight**: On raw spectra, **all 5 original methods fall below the Random baseline** at k=3 — a counterintuitive result caused by adjacent-pixel collinearity. SG 1st-derivative preprocessing or clustering constraints each independently resolve this issue; combining both yields the best performance.

### Selected Wavelengths & Chemical Interpretation

The most stable wavelengths map to chemically complementary absorption bands:

| λ (nm) | Chemical Bond | Fiber Class Distinguished |
|--------|---------------|--------------------------|
| **1419** | O–H 1st overtone | Cotton (cellulose), Wool (moisture) |
| **1463** | N–H 1st overtone | Wool, Nylon (amide linkages) |
| **1510** | N–H 1st overtone | Wool, Nylon, Silk (protein/polyamide) |
| **1546** | N–H 1st overtone | Wool, Nylon (amide) |

The O–H channel (1419 nm) separates cellulose-based Cotton from synthetic/protein fibers; the N–H channels (1463–1546 nm) separate protein-based Wool from synthetic Nylon. Together, **two chemically orthogonal channels** achieve near-perfect 6-class discrimination.

### Statistical Significance

Wilcoxon signed-rank tests (n=5 seeds) confirm that all diversity-aware methods significantly outperform the Random baseline. The gap between the best method and Teacher is **not statistically significant** (p > 0.05 at all k), meaning the wavelength-reduced Student is statistically equivalent to the full-spectrum Teacher.

### Figures

Running `--plot` generates 5 types of publication-quality figures (300 DPI PNG):

| Figure | Content |
|--------|---------|
| `fig1_mean_spectra` | Per-class mean spectra ± 1σ, selected wavelengths highlighted |
| `fig2_method_comparison` | Bar chart: 9 methods vs Teacher & Random baselines |
| `consensus_heatmap_*` | Seed × Wavelength heatmap per method (fold-count frequency) |
| `fig4_physical_interpretation` | Grand-mean spectrum with chemical bond annotations |
| `fig5_confusion_matrix` | Raw counts + row-normalized recall for best method |

---

## Project Structure

```
Textile-classification/
├── README.md
├── .gitignore
├── resume_description.md             # Resume-ready project bullets (CN/EN)
│
├── data/
│   ├── csv/                          # 188 pure-fiber spectra (CSV)
│   ├── preprocessing/
│   │   ├── Pure/                     # Reorganized by class (01–06 + Background)
│   │   ├── Blends/                   # 40 blend spectra (4 blend ratios)
│   │   └── total/                    # 35 additional fabric images
│   └── raw/image/                    # 58 raw fabric images
│
├── figures/
│   └── final/                        # Publication-ready plots (k=3,5,10)
│
└── wavelength_selection/
    ├── select_wavelengths.py         # Main experiment runner (CLI + Python API)
    └── plotting.py                   # Visualization module (5 figure types)
```

---

## Usage

### CLI

```bash
# Quick test: k=3, SG 1st-derivative, 5 seeds, with figures
python wavelength_selection/select_wavelengths.py --k 3 --n_seeds 5 --preprocess savgol_1deriv --plot

# Full experiment suite: k=5 → auto-runs k=10 for comparison
python wavelength_selection/select_wavelengths.py --k 5 --n_seeds 5 --preprocess savgol_1deriv --plot

# Raw spectra baseline (no preprocessing)
python wavelength_selection/select_wavelengths.py --k 3 --n_seeds 5 --plot

# Single seed for quick debugging
python wavelength_selection/select_wavelengths.py --k 5 --seed 42 --plot
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | 5 | Number of wavelengths to select |
| `--n_seeds` | 5 | Random seeds (≥5 required for Wilcoxon tests) |
| `--preprocess` | `none` | `none` / `savgol` / `savgol_1deriv` / `savgol_2deriv` |
| `--seed` | `None` | Single seed override (disables multi-seed mode) |
| `--plot` | `False` | Generate and save all figures (300 DPI) |
| `--save_plots` | `figures` | Output directory for figures |

> **Note**: When `--k 5` and `--n_seeds ≥ 3`, the script automatically runs a k=10 comparison experiment.

### Python API

```python
from wavelength_selection.select_wavelengths import run_multi_seed_experiment

result = run_multi_seed_experiment(
    k=3, seeds=[0, 1, 2, 3, 4],
    preprocess='savgol_1deriv', plot=True
)

# Structured results
result['summary_df']           # pd.DataFrame — method comparison
result['stats_df']             # pd.DataFrame — Wilcoxon test results
result['best_method']          # str — e.g., 'MI + MinDist'
result['best_wavelengths_nm']  # np.ndarray — selected wavelengths in nm
result['best_indices']         # np.ndarray — array indices of selected wavelengths
```

---

## Dependencies

| Package | Version (tested) | Purpose |
|---------|:--:|---------|
| Python | ≥ 3.8 | — |
| numpy | ≥ 1.20 | Array operations |
| pandas | ≥ 1.3 | Data loading, result aggregation |
| scipy | ≥ 1.7 | Savitzky-Golay filter, Wilcoxon test, hierarchical clustering |
| scikit-learn | ≥ 1.0 | ML models, feature selection, CV |
| matplotlib | ≥ 3.4 | Figure generation |
| seaborn | ≥ 0.11 | Statistical visualizations |

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@dataset{textile-nir-classification,
  title     = {NIR Spectroscopy Dataset for Textile Fiber Classification},
  author   = {SanctusDei},
  year     = {2026},
  url      = {https://github.com/SanctusDei/Textile-classification-based-on-near-infrared-spectroscopy}
}
```

---

## License

This project is provided for research purposes.
