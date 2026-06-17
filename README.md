  # Textile Fiber Classification via Near-Infrared Spectroscopy

  > **From 228-point lab spectrometer to 2–4 LEDs — wavelength selection for portable NIR textile identification.**

  Near-infrared (NIR) spectroscopy dataset for textile fiber classification, collected with a **TI DLP NIRScan nano** spectrometer (900–1700 nm, 228 sampling points). The full dataset contains 188 pure-fiber spectra across 6 classes, plus 40 blend spectra and 10 background spectra. The wavelength selection experiment uses 173 spectra from 3 classes. The project investigates feature selection methods to compress the full spectrum into a handful of key wavelengths, enabling deployment on low-cost portable hardware.

  ---

  ## Table of Contents

  - [Quick Start](#quick-start)
  - [Dataset](#dataset)
  - [Experiment: Wavelength Selection](#experiment-wavelength-selection)
  - [Key Results](#key-results)
  - [Usage](#usage)
  - [Citation](#citation)
  - [License](#license)

  ---

  ## Quick Start

  ```bash
  git clone https://github.com/SanctusDei/Textile-classification-based-on-near-infrared-spectroscopy.git
  cd Textile-classification
  pip install numpy pandas scipy scikit-learn matplotlib seaborn

  # Run final experiment (SG 1st-derivative, k=5, 5 seeds, with figures)
  python wavelength_selection/select_wavelengths.py --k 5 --n_seeds 5 --preprocess savgol_1deriv --plot
  ```

  ---

  ## Dataset

  ### Full Dataset

  | Property | Value |
  |----------|-------|
  | **Pure-fiber spectra** | 188 (6 classes) |
  | **Blend-fabric spectra** | 40 (4 blend ratios) |
  | **Background spectra** | 10 (PVC reference) |
  | **Fabric images** | 60 (`.jpg`, in `data/raw/image/`) |
  | **Spectral range** | 900–1700 nm |
  | **Sampling points** | 228 |
  | **Instrument** | TI DLP NIRScan nano |

  | Class | Spectra | Swatches | Description |
  |-------|:--:|:--:|-------------|
  | Polyester (PET) | 83 | 25 | Synthetic; most common textile fiber |
  | Nylon (PA) | 45 | 5 | Synthetic polyamide |
  | Cotton | 45 | 5 | Natural cellulose |
  | Acetate | 5 | 1 | Semi-synthetic cellulose acetate |
  | Acrylic | 5 | 1 | Synthetic (PAN-based) |
  | Wool | 5 | 1 | Natural protein (keratin) |
  | **Total** | **188** | **38** | |

  ### Experiment Subset

  The wavelength selection experiment uses a filtered subset of the full dataset:

  | Property | Value |
  |----------|-------|
  | **Samples** | 173 pure-fiber spectra |
  | **Classes** | 3 (Cotton, Nylon, Polyester) |
  | **Unique swatches** | 35 physical fabric specimens |

  > **Note**: Acetate, Acrylic, and Wool are excluded from the experiment — with only 1 physical swatch each, specimen-level GroupKFold cross-validation is impossible (the same swatch cannot appear in both train and test).

  ### Data Collection Protocol

  Each swatch was measured under **multiple conditions** to capture intra-class variability:

  - **Multiple positions** across the fabric surface
  - **Multiple rotation angles** (0°, 90°)
  - **Multiple fold configurations** (Single, 2-Fold, 4-Fold)

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

  ### Data Not Used in This Experiment

  The following data are included in the repository but not used by the wavelength selection pipeline:

  - **40 blend-fabric spectra** (`data/preprocessing/Blends/csv/`) — 4 blend ratios:
    - Cotton/Polyester 55:45, Cotton/Polyester 80:20
    - Nylon/Polyester 30:70, Wool/Polyester 35:65
  - **10 PVC background spectra** (`data/preprocessing/Pure/07_Background/csv/`) — reference measurements for normalization
  - **60 fabric images** (`data/raw/image/`) — visual documentation of measured swatches

  ---

  ## Experiment: Wavelength Selection

  ### Motivation

  A lab-grade NIR spectrometer (228-point InGaAs array) costs **$2,000+**. By identifying k = 3–10 discriminative wavelengths, we can replace it with **fixed-wavelength LEDs + photodiodes** (~$30–50 per LED), reducing the bill of materials by **94–98%** while maintaining near-perfect classification accuracy.

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

  ### Baselines

  | Baseline | Meaning |
  |----------|---------|
  | **Teacher (228λ)** | Upper bound — full-spectrum, best model |
  | **Random k wavelengths** | Lower bound — randomly chosen sampling points, KNN classifier |

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

  Results with `savgol_1deriv` preprocessing, 5 seeds × swatch-level 5-fold CV, 3 classes:

  | k | Teacher (228λ) | Random k | Best Method | Best Acc. (Pseudo) | Gap vs Teacher |
  |---|:---:|:---:|-------------|:---:|:---:|
  | **10** | 0.9657 ± 0.0112 | 0.9181 ± 0.0169 | L1 LogisticRegression | 0.9836 ± 0.0062 | **+0.0179** |
  | **5** | 0.9657 ± 0.0112 | 0.8720 ± 0.0251 | L1 LogisticRegression | 0.9836 ± 0.0068 | **+0.0179** |
  | **3** | 0.9573 ± 0.0059 | 0.8531 ± 0.0413 | RFE (Linear SVM) | 0.9648 ± 0.0093 | **+0.0074** |

  L1 LogisticRegression dominates at both k=5 and k=10, achieving identical pseudo-label accuracy (0.9836) while exceeding the teacher by +0.0179. Mutual Information ranks second (0.9747 at k=10, 0.9457 at k=5). RFE suffers from severe selection instability (Jaccard = 0.00) despite reasonable accuracy. ANOVA F-score and Random Forest consistently underperform — ANOVA only sees the O–H band, and RF importance is unreliable with correlated spectral features. At k=3, RFE (0.9648) edges out L1 LogReg (0.9610). **k=5 is the sweet spot**: 0.9836 accuracy with 5 LEDs, matching k=10's performance exactly at half the hardware cost. The student exceeding the teacher across all k budgets suggests the full-spectrum teacher overfits, and L1-driven feature selection acts as a powerful regularizer.

  > **Note on stability metrics**: Strict consensus (intersection across all folds) underestimates true stability because adjacent wavelengths within the same chemical absorption band are interchangeable. Selection frequency (%) — how often each wavelength appears across all seeds × folds — is the more informative metric. For example, at k=5, L1 LogReg has only 1.0 strict-consensus wavelengths on average but top-5 wavelengths appear at 48–100% frequency, each from a distinct chemical band.

  ### Preprocessing Impact

  | Preprocessing | Effect on Feature Selection |
  |---------------|----------------------------|
  | `none` (raw) | Adjacent-point collinearity degrades univariate rankings |
  | `savgol` | Smoothing reduces noise, preserves peak shapes |
  | `savgol_1deriv` | **Recommended** — removes baseline drift, sharpens peaks |
  | `savgol_2deriv` | Further sharpens but amplifies noise |

  ### Selected Wavelengths & Chemical Interpretation

  **k=5 — L1 LogisticRegression, best method (Pseudo: 0.9836):**

  Selection frequency across 25 runs (5 seeds × 5 folds):

  | λ (nm) | Freq. | Chemical Bond | Distinguishes |
  |--------|:-----:|---------------|---------------|
  | **1655** | 100% | C–H 1st overtone | PET (aromatic CH); O–H (Cotton) |
  | **1522** | 64% | N–H 1st overtone | Nylon (amide) |
  | **1413** | 60% | O–H 1st overtone | Cotton (cellulose/water); C–H combination |
  | **1658** | 56% | C–H 1st overtone | PET (aromatic CH); O–H (Cotton) |
  | **1519** | 48% | N–H 1st overtone | Nylon (amide) |

  Three chemically distinct bands cover all three classes: C–H 1st overtone (~1655 nm) for PET, N–H 1st overtone (~1520 nm) for Nylon, and O–H 1st overtone (~1415 nm) for Cotton. The 1658 nm and 1519 nm picks are adjacent sampling points within the same bands — the L1 penalty reliably selects from the correct band but the exact index varies by ±5 nm across folds.

  **k=10 — L1 LogisticRegression, best method (Pseudo: 0.9836):**

  Selection frequency across 25 runs (5 seeds × 5 folds):

  | λ (nm) | Freq. | Chemical Bond | Distinguishes |
  |--------|:-----:|---------------|---------------|
  | **1655** | 100% | C–H 1st overtone | PET (aromatic CH); O–H (Cotton) |
  | **1416** | 96% | O–H 1st overtone | Cotton (cellulose/water); C–H combination |
  | **1413** | 84% | O–H 1st overtone | Cotton (cellulose/water); C–H combination |
  | **1658** | 84% | C–H 1st overtone | PET (aromatic CH); O–H (Cotton) |
  | **1522** | 80% | N–H 1st overtone | Nylon (amide) |

  With k=10, L1 LogReg consistently selects from all three chemical bands (C–H, O–H, N–H) at ≥80% frequency. However, k=10 achieves identical accuracy to k=5 (0.9836) — the extra 5 wavelengths are largely redundant adjacent sampling points within the same absorption bands. This confirms that **five chemically distinct wavelengths are sufficient** for robust 3-class textile discrimination.

  **k=3 — RFE (Linear SVM), best method (Pseudo: 0.9648):**

  Selection frequency across 15 runs (3 seeds × 5 folds):

  | λ (nm) | Freq. | Chemical Bond | Distinguishes |
  |--------|:-----:|---------------|---------------|
  | **1510** | 47% | N–H 1st overtone | Nylon (amide) |
  | **1173** | 40% | C–H 2nd overtone | PET, PA (polymer backbone) |
  | **1516** | 33% | N–H 1st overtone | Nylon (amide) |

  Only 3 wavelengths exceed 30% selection frequency, and two belong to the same N–H band. The O–H band (Cotton marker) is missed entirely, making Cotton vs PET discrimination fragile. With k=3, accuracy drops to 0.9648 and selection frequencies are low (≤47%), indicating that **k=3 is insufficient for robust 3-class discrimination**.

  ### Statistical Significance

  Wilcoxon signed-rank tests (n=5 seeds, α=0.05) compare each method against the Random baseline and the Teacher.

  | Experiment | Outcome |
  |------------|---------|
  | **k=10** | All p ≥ 0.0625 (not significant). L1 LogReg vs Teacher p=0.0625; MI vs Random p=0.0625 |
  | **k=5** | All p ≥ 0.0625 (not significant). L1 LogReg vs Teacher p=0.0625; MI vs Random p=0.0625 |
  | **k=3** | Skipped — Wilcoxon requires ≥ 5 seeds (k=3 auto-compare uses only 3 seeds) |

  ---

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
  | `--n_seeds` | 5 | Random seeds (≥5 for Wilcoxon, ≥6 for α=0.05 significance) |
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

  ## Future Work & Limitations

  ### Limitations

  - **Class coverage**: Only 3 of 6 pure-fiber classes (Cotton, Nylon, Polyester) were used due to the single-switch constraint. Acetate, Acrylic, and Wool lack sufficient physical specimens for swatch-level cross-validation.
  - **Blend fabrics**: 40 blend-fiber spectra are present in the dataset but not yet included in the wavelength selection experiment. Blends (e.g., Cotton/Polyester 55:45) present a harder multi-label or out-of-distribution challenge.
  - **Dataset scale**: 173 spectra from 35 swatches is modest; larger-scale collection across more fabric types and conditions would improve generalizability.
  - **Hardware validation**: Selected wavelengths are validated statistically via cross-validation but have not yet been tested on physical LED + photodiode hardware.

  ### Future Work

  - **Hardware prototyping**: Build and test a portable device with the selected k LEDs to validate real-world classification accuracy.
  - **Blend classification**: Extend the pipeline to handle blend fabrics, potentially via multi-label soft classification or regression-based composition prediction.
  - **Transfer learning**: Investigate whether wavelengths selected on pure fabrics generalize to blends.
  - **Additional classes**: Collect more swatches for Acetate, Acrylic, and Wool to enable full 6-class discrimination.

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
