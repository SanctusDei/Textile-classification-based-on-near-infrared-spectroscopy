# Textile Fiber Classification via Near-Infrared Spectroscopy

> **Wavelength selection for portable NIR textile identification — from lab spectrometer to low-cost multi-wavelength sensor.**

Near-infrared (NIR) spectroscopy dataset for textile fiber classification, collected with a **TI DLP NIRScan nano** spectrometer. The project investigates feature selection methods to compress 228 spectral bands into a handful of key wavelengths, enabling deployment on low-cost portable hardware.

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
# Clone the repository
git clone https://github.com/SanctusDei/Textile-classification-based-on-near-infrared-spectroscopy.git
cd Textile-classification

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Run a quick experiment (3 wavelengths, 5 seeds, with figures)
python wavelength_selection/select_wavelengths.py --k 5 --n_seeds 5 --plot
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

This protocol ensures the dataset reflects real-world measurement variance, not just idealized lab conditions.

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
  - Cotton/Polyester 55:45
  - Cotton/Polyester 80:20
  - Nylon/Polyester 30:70
  - Wool/Polyester 35:65
- **10 PVC background spectra** — reference measurements for normalization
- **132 fabric images** (`.bmp`) — visual documentation of measured swatches

---

## Experiment: Wavelength Selection

### Motivation

A lab-grade NIR spectrometer (228-pixel InGaAs array) costs **$2,000+** and requires a computer for data processing. By identifying a small subset of discriminative wavelengths (k = 5–10), we can replace the full spectrometer with a handful of **fixed-wavelength LEDs + photodiodes**, reducing the bill of materials to **$150–$500** while maintaining classification accuracy.

### Methods

Five feature selection methods are compared:

| Method | Type | Description |
|--------|------|-------------|
| **ANOVA F-score** | Univariate filter | One-way ANOVA; assumes normality, fast |
| **Mutual Information** | Univariate filter | Captures non-linear dependencies; no distribution assumption |
| **RFE (Linear SVM)** | Wrapper | Recursive Feature Elimination; considers feature interactions |
| **L1 LogisticRegression** | Embedded | L1 penalty induces sparsity; multi-class via one-vs-rest |
| **Random Forest Imp.** | Embedded | Tree-based mean decrease in impurity |

### Teacher-Guided Pseudo-Label Training

A two-stage training paradigm simulates real-world deployment where ground-truth labels are unavailable:

1. **Teacher**: trained on **all 228 wavelengths** (per-fold, to prevent leakage). Internal 3-fold CV selects the best model among SVM-RBF, RandomForest, and KNN.
2. **Student**: trained on only **k selected wavelengths**, using the Teacher's predictions as pseudo-labels.

The accuracy gap between Teacher (228λ) and Student (kλ) quantifies the cost of wavelength reduction.

### Experimental Design (Rigor)

| Design Choice | Purpose |
|---------------|---------|
| **Stratified 5-fold CV** | Preserves class distribution in every split |
| **Feature selection inside each fold** | Prevents test-set leakage to feature selector |
| **Teacher retrained per fold** | Prevents Teacher from leaking test knowledge to Student |
| **Multi-seed experiments (5 seeds)** | Reports mean ± std, not single-seed cherry-picking |
| **Wilcoxon signed-rank tests** | Statistical significance of method comparisons (α = 0.05) |
| **Consensus wavelength analysis** | Jaccard stability index across folds; only stable wavelengths recommended for hardware |

### Baselines

| Baseline | Meaning |
|----------|---------|
| **Teacher (228λ)** | Upper bound — full-spectrum classification with best model |
| **Random k wavelengths** | Lower bound — no intelligent selection, randomly chosen bands |

### Preprocessing Options

- **None** (raw absorbance)
- **Savitzky-Golay smoothing** (window=11, polyorder=3)
- **Savitzky-Golay 1st derivative** (baseline correction, resolves overlapping peaks)
- **Savitzky-Golay 2nd derivative** (sharper peak resolution, higher noise)

---

## Key Results

### Accuracy vs. Wavelength Count

| k | Best Method | Pseudo Accuracy | Teacher Gap | Compression |
|---|-------------|-----------------|-------------|-------------|
| 5 | RFE (Linear SVM) | ~0.84 | ~−0.06 | 97.8% |
| 10 | RFE (Linear SVM) | ~0.87 | ~−0.03 | 95.6% |

*Results reported as mean across 5 seeds × 5 folds. Full results depend on preprocessing choice — run the experiment to reproduce.*

### Selected Wavelengths & Physical Interpretation

The most frequently selected wavelengths correspond to chemically meaningful absorption bands:

| Wavelength (nm) | Chemical Bond | Relevance |
|-----------------|---------------|-----------|
| ~1150–1250 | C–H 2nd overtone | Polymer backbone (PET, PA) |
| ~1400–1450 | O–H 1st overtone | Cellulose/water (Cotton) |
| ~1450–1550 | N–H 1st overtone | Amide bonds (Wool, Nylon, Silk) |
| ~1600–1700 | C–H 1st overtone | CH₃, CH₂, aromatic (PET); O–H (Cotton) |

These wavelength assignments are consistent with the known NIR spectroscopy of textile polymers: polyester is distinguished by aromatic C–H overtones, cotton by O–H from cellulose, and protein-based fibers (wool) by N–H from amide linkages.

### Figures

Running with `--plot` generates 5 publication-quality figures (300 DPI):

| Figure | Content |
|--------|---------|
| `fig1_mean_spectra` | Per-class mean spectra ± 1σ, selected wavelengths highlighted |
| `fig2_method_comparison` | Bar chart: 5 methods vs Teacher & Random baselines |
| `consensus_heatmap_*` | Seed × Wavelength heatmap per method (fold-count frequency) |
| `fig4_physical_interpretation` | Grand-mean spectrum with chemical bond annotations |
| `fig5_confusion_matrix` | Raw + normalized confusion matrix for best method |

---

## Project Structure

```
Textile-classification/
├── README.md                         # This file
├── .gitignore                        # Excludes code artifacts
├── resume_description.md             # Resume-ready project bullets (CN/EN)
│
├── data/
│   ├── csv/                          # 188 pure-fiber spectra (CSV, 22-line header + 228 rows)
│   ├── preprocessing/
│   │   ├── Pure/                     # Reorganized by class (01–06 + Background)
│   │   ├── Blends/                   # 40 blend spectra (4 blend ratios)
│   │   └── total/                    # 35 additional fabric images
│   └── raw/image/                    # 58 raw fabric images
│
├── figures/                          # Generated plots (300 DPI PNG)
│   ├── fig1_mean_spectra_k*.png
│   ├── fig2_method_comparison_k*.png
│   ├── consensus_heatmap_*_*wl.png
│   ├── fig4_physical_interpretation_k*.png
│   └── fig5_confusion_matrix_*_k*.png
│
└── wavelength_selection/             # Experiment code
    ├── select_wavelengths.py         # Main experiment runner + CLI
    └── plotting.py                   # Visualization module (5 figure types)
```

---

## Usage

### CLI

```bash
# Basic: k=5 wavelengths, 5 seeds, no figures
python wavelength_selection/select_wavelengths.py

# Aggressive compression: only 3 wavelengths
python wavelength_selection/select_wavelengths.py --k 3 --n_seeds 5

# With preprocessing and figures
python wavelength_selection/select_wavelengths.py --k 5 --preprocess savgol_1deriv --plot

# Single seed for quick debugging
python wavelength_selection/select_wavelengths.py --k 5 --seed 42 --plot
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | 5 | Number of wavelengths to select |
| `--n_seeds` | 5 | Random seeds for multi-seed experiments (≥5 for Wilcoxon tests) |
| `--preprocess` | `none` | `none` / `savgol` / `savgol_1deriv` / `savgol_2deriv` |
| `--seed` | `None` | Single seed override (disables multi-seed) |
| `--plot` | `False` | Generate and save all 5 figures |
| `--save_plots` | `figures` | Output directory for figures |

### Python API

```python
from wavelength_selection.select_wavelengths import run_multi_seed_experiment

result = run_multi_seed_experiment(k=5, seeds=[0, 1, 2, 3, 4], plot=True)

# Access results
print(result['summary_df'])           # Method comparison table
print(result['stats_df'])             # Wilcoxon test results
print(result['best_wavelengths_nm'])  # Selected wavelengths in nm
print(result['best_method'])          # Best feature selection method
```

---

## Dependencies

| Package | Version (tested) | Purpose |
|---------|------------------|---------|
| Python | ≥ 3.8 | — |
| numpy | ≥ 1.20 | Array operations |
| pandas | ≥ 1.3 | Data loading, result aggregation |
| scipy | ≥ 1.7 | Savitzky-Golay filter, Wilcoxon test |
| scikit-learn | ≥ 1.0 | ML models, feature selection, CV |
| matplotlib | ≥ 3.4 | Figure generation |
| seaborn | ≥ 0.11 | Statistical visualizations |

Install with:
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

This project is provided for research purposes. See individual files for attribution details.
