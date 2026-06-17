# Textile Fiber Classification via Near-Infrared Spectroscopy

> **From 228-pixel lab spectrometer to 2–4 LEDs — wavelength selection for portable NIR textile identification.**

Near-infrared (NIR) spectroscopy dataset for textile fiber classification, collected with a **TI DLP NIRScan nano** spectrometer (900–1700 nm, 228 bands, 188 samples, 6 classes). The project investigates feature selection methods to compress the full spectrum into a handful of key wavelengths, enabling deployment on low-cost portable hardware.

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
| **Samples** | 173 pure-fiber spectra (after filtering single-switch classes) |
| **Spectral range** | 900–1700 nm |
| **Spectral bands** | 228 pixels |
| **Classes** | 3 (Cotton, Nylon, Polyester) |
| **Unique swatches** | 35 physical fabric specimens |
| **Instrument** | TI DLP NIRScan nano (SN 6460024) |
| **Additional data** | 40 blend spectra, 10 background spectra, 132 fabric images |

### Classes

| Class | Spectra | Swatches | Description |
|-------|:--:|:--:|-------------|
| Polyester (PET) | 83 | 24 (+1 Std) | Synthetic; most common textile fiber |
| Nylon (PA) | 45 | 4 (+1 Std) | Synthetic polyamide |
| Cotton | 45 | 4 (+1 Std) | Natural cellulose |
| **Total** | **173** | **35** | |

> **Note**: Acetate, Acrylic, and Wool (5 spectra each, single "Std" swatch) are excluded — with only one physical specimen per class, specimen-level cross-validation is impossible.

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
- **132 fabric images** (`.jpg`) — visual documentation of measured swatches (across `data/raw/image/`, `data/preprocessing/total/`, and `data/preprocessing/Blends/images/`)

---

## Experiment: Wavelength Selection

### Motivation

A lab-grade NIR spectrometer (228-pixel InGaAs array) costs **$2,000+**. By identifying k = 3–10 discriminative wavelengths, we can replace it with **fixed-wavelength LEDs + photodiodes** (~$30–50 per LED), reducing the bill of materials by **94–98%** while maintaining near-perfect classification accuracy.

### The Adjacent-Pixel Problem

On raw absorbance spectra, univariate feature selection methods (ANOVA F-score, Mutual Information) suffer from a critical flaw: NIR absorption peaks span 20–50 nm (6–14 pixels at 3.5 nm/px resolution), creating strong collinearity between adjacent wavelengths. These methods rank all pixels within the same absorption band as "top-k," **collapsing k selections into a single chemical information channel**.

Concretely, without countermeasures, ANOVA on raw spectra tends to select multiple wavelengths from the same C–H overtone band (e.g., 1190–1210 nm), yielding an effective spectral spread of only a few nanometers — compared to ~170 nm for random selection. This redundancy degrades classification accuracy because the model sees only one chemical dimension instead of k complementary ones.

### Mitigating Adjacent-Pixel Collinearity

**Savitzky-Golay 1st-derivative preprocessing** (window=11, polyorder=3) is the primary mitigation in the current pipeline: it removes baseline drift and decorrelates adjacent pixels, largely resolving the redundancy problem before feature selection begins.

For stronger guarantees, two additional strategies are implemented in the code as optional extensions (not active in the default 5-method suite):

| Strategy | Mechanism |
|----------|-----------|
| **Minimum Distance (MinDist, 30 nm)** | Greedy selection: after picking the top-scoring wavelength, exclude all wavelengths within ±30 nm; repeat |
| **Correlation Clustering** | Pre-compute k spectral clusters via hierarchical clustering on 1−\|Pearson r\|; pick the best-scoring wavelength per cluster |

The diversity-aware methods (`select_anova_mindist`, `select_mi_mindist`, `_select_anova_per_cluster`, `_select_mi_per_cluster`) are defined in `select_wavelengths.py` and can be registered into the selection method suite for experiments that require explicit spectral diversity constraints.

### Methods Compared (5 total)

| Method | Type | Description |
|--------|------|-------------|
| ANOVA F-score | Univariate filter | Fast, assumes normality |
| Mutual Information | Univariate filter | Captures non-linear dependencies |
| RFE (Linear SVM) | Wrapper | Recursive feature elimination, considers interactions |
| L1 LogisticRegression | Embedded | Sparsity via L1 penalty |
| Random Forest Imp. | Embedded | Tree-based feature importance |

### Teacher-Guided Pseudo-Label Training

Simulates real-world deployment where ground-truth labels are unavailable:

1. **Teacher**: trained on all 228 wavelengths per-fold (internal 3-fold CV selects best among SVM-RBF, RF, KNN)
2. **Student**: trained on only k selected wavelengths, using the Teacher's predictions as pseudo-labels

The accuracy gap between Teacher (228λ) and Student (kλ) quantifies the cost of wavelength reduction.

### Experimental Rigor

| Design Choice | Purpose |
|---------------|---------|
| **Swatch-level GroupKFold (5-fold)** | All spectra from a physical swatch stay in the same fold — zero data leakage |
| Feature selection **inside each fold** | Prevents test-set leakage to feature selector |
| Teacher **retrained per fold** | Prevents Teacher from leaking test knowledge to Student |
| Multi-seed (5 seeds) | Reports mean ± std, not single-seed cherry-picking |
| Wilcoxon signed-rank tests (α=0.05) | Statistical significance of method comparisons |
| Consensus wavelength analysis | Jaccard stability index; only stable wavelengths recommended for hardware |

> **Why swatch-level CV matters**: Without it, spectra from the same fabric piece appear in both train and test (e.g., Cotton_C01_Pos1 in train, Cotton_C01_Pos2 in test). The model learns to recognize the specific cloth rather than the material class, inflating Random baseline accuracy and undermining the validity of wavelength selection.

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

> **Preprocessing: SG 1st-derivative  |  5 seeds × swatch-level 5-fold CV  |  3 classes (Cotton, Nylon, PET)  |  5 methods compared**

### Accuracy by Wavelength Budget

Results vary by preprocessing and method choice. Running with `savgol_1deriv` preprocessing and multi-seed evaluation produces stable, reproducible rankings. The Teacher baseline (full 228λ) typically achieves >0.98 accuracy; the best k-wavelength Student typically closes the gap to within <0.02 at k=5, demonstrating that 2–4 LEDs can approach lab-spectrometer performance.

### Preprocessing Impact

| Preprocessing | Effect on Feature Selection |
|---------------|----------------------------|
| `none` (raw) | Adjacent-pixel collinearity degrades univariate rankings |
| `savgol` | Smoothing reduces noise, preserves peak shapes |
| `savgol_1deriv` | **Recommended** — removes baseline drift, sharpens peaks |
| `savgol_2deriv` | Further sharpens but amplifies noise |

### Selected Wavelengths & Chemical Interpretation

The most stable wavelengths typically map to chemically complementary absorption bands:

| λ (nm) | Chemical Bond | Fiber Class Distinguished |
|--------|---------------|--------------------------|
| **~1420** | O–H 1st overtone | Cotton (cellulose) |
| **~1465** | N–H 1st overtone | Nylon (amide linkages) |
| **~1510** | N–H 1st overtone | Nylon (polyamide) |
| **~1545** | N–H 1st overtone | Nylon (amide) |

The O–H channel separates cellulose-based Cotton from synthetic fibers; the N–H channels distinguish Nylon from PET. Together, **two chemically orthogonal channels** achieve near-perfect 3-class discrimination.

### Statistical Significance

Wilcoxon signed-rank tests (n=5 seeds, α=0.05) compare each method against the Random baseline and the Teacher. With ≥5 seeds, the tests reliably identify methods that significantly outperform random selection.

### Figures

Running `--plot` generates 5 types of publication-quality figures (300 DPI PNG):

| Figure | Filename Pattern | Content |
|--------|-----------------|---------|
| 1 | `fig1_mean_spectra_k{N}.png` | Per-class mean spectra ± 1σ, selected wavelengths highlighted |
| 2 | `fig2_method_comparison_k{N}.png` | Bar chart: 5 methods vs Teacher & Random baselines |
| 3 | `consensus_heatmap_{Method}_{N}wl.png` | Seed × Wavelength heatmap per method (fold-count frequency) |
| 4 | `fig4_physical_interpretation_k{N}.png` | Grand-mean spectrum with chemical bond annotations |
| 5 | `fig5_confusion_matrix_{Method}_k{N}.png` | Raw counts + row-normalized recall for best method |

---

## Project Structure

```
Textile-classification/
├── README.md
├── .gitignore
├── resume_description.md             # Resume-ready project bullets (CN/EN)
│
├── data/
│   ├── csv/                          # 188 pure-fiber spectra (228-band CSV)
│   ├── preprocessing/
│   │   ├── Pure/                     # Spectra reorganized by class (7 subdirs: 3 kept + Acetate/Acrylic/Wool/Background)
│   │   ├── Blends/                   # 40 blend spectra (4 blend ratios) + 5 fabric images
│   │   └── total/                    # 35 additional fabric images
│   └── raw/image/                    # 60 raw fabric images
│
├── figures/                          # Output directory for generated plots (300 DPI PNG)
│
├── results/                          # Experiment outputs (CSV summary + statistics + JSON config)
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

# Full experiment suite: k=5 → auto-runs k=3 for comparison
python wavelength_selection/select_wavelengths.py --k 5 --n_seeds 5 --preprocess savgol_1deriv --plot

# Raw spectra baseline (no preprocessing)
python wavelength_selection/select_wavelengths.py --k 3 --n_seeds 5 --plot

# Single seed for quick debugging
python wavelength_selection/select_wavelengths.py --k 5 --seed 42 --plot

# Run only specific methods
python wavelength_selection/select_wavelengths.py --k 3 --methods "ANOVA F-score,Mutual Information" --plot
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | 5 | Number of wavelengths to select |
| `--n_seeds` | 5 | Random seeds (≥5 required for Wilcoxon tests) |
| `--preprocess` | `none` | `none` / `savgol` / `savgol_1deriv` / `savgol_2deriv` |
| `--seed` | `None` | Single seed override (disables multi-seed mode) |
| `--methods` | all 5 | Comma-separated method names to include |
| `--plot` | `False` | Generate and save all figures (300 DPI) |
| `--save_plots` | `figures` | Output directory for figures |
| `--save_results` | `True` | Save experiment results to CSV/JSON |
| `--no_save_results` | — | Disable result saving |
| `--results_dir` | `results` | Directory for saved results |
| `--compare_k` | `None` | Run a second experiment with this k value |

> **Note**: When `--k 5` and `--n_seeds ≥ 3`, the script automatically runs a k=3 comparison experiment (unless `--compare_k` is explicitly set).

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
result['best_method']          # str — e.g., 'ANOVA F-score'
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
