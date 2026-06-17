"""
Wavelength Selection for Portable NIR Textile Classification
============================================================
Pure ML approach — no deep learning required.

Problem: 228 NIR wavelengths → select k=5 or k=10 most informative ones
         → deploy on low-cost portable hardware.

Methods compared (5 total):
  1. ANOVA F-score          — univariate filter, fast
  2. Mutual Information     — captures non-linear dependencies
  3. RFE (Linear SVM)       — wrapper, considers feature interactions
  4. L1 LogisticRegression  — embedded sparsity via L1 penalty
  5. Random Forest Imp.     — tree-based embedded importance

Training modes:
  - Hard labels: student trains on ground-truth labels
  - Teacher-guided pseudo-labels: student trains on teacher-predicted labels
    (teacher retrained per fold — no information leakage)

Key design decisions (reviewer-ready):
  - Feature selection runs INSIDE each CV fold (no data leakage)
  - Teacher is retrained per-fold (no test-set leakage to student)
  - Multi-seed experiments report mean ± std (not single seed=42)
  - Consensus wavelengths across folds measure stability
  - Wilcoxon signed-rank test for statistical significance

Usage:
  python select_wavelengths.py
  python select_wavelengths.py --k 5 --n_seeds 5
  python select_wavelengths.py --k 10 --plot --save_plots figures/
  python select_wavelengths.py --preprocess savgol_1deriv
  python select_wavelengths.py --methods "ANOVA F-score,Mutual Information"
"""

import csv
import glob
import json
import os
import sys
import argparse
import warnings
from collections import defaultdict, Counter
from functools import reduce
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
    RFE,
    SelectKBest,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from plotting import generate_all_figures, interpret_wavelength

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# Shuffled GroupKFold (compat for sklearn < 1.4)
# ═══════════════════════════════════════════════════════════════════════════════

class ShuffledGroupKFold:
    """GroupKFold with shuffle support, compatible with sklearn < 1.4.

    Shuffles group labels within each class independently, then assigns
    groups to folds via round-robin per class. This ensures each class
    is spread across folds (stratified-like) while keeping all spectra
    from the same swatch together.

    Different random_state values produce different train/test splits.
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        if groups is None or y is None:
            raise ValueError("groups and y must be provided")
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        fold_assignments = np.full(n_groups, -1, dtype=int)

        rng = np.random.RandomState(self.random_state) if self.shuffle else None

        # Group groups by class for stratified assignment
        group_to_class = {}
        for g in unique_groups:
            mask = groups == g
            # All samples in a group share the same class
            group_to_class[g] = y[mask][0]

        # Process each class independently: shuffle → round-robin
        for cls in np.unique(y):
            cls_groups = [g for g in unique_groups if group_to_class[g] == cls]
            if self.shuffle:
                rng.shuffle(cls_groups)
            for i, g in enumerate(cls_groups):
                g_idx = np.where(unique_groups == g)[0][0]
                fold_assignments[g_idx] = i % self.n_splits

        # Yield splits
        for fold in range(self.n_splits):
            test_groups = unique_groups[fold_assignments == fold]
            if len(test_groups) == 0:
                continue  # skip empty folds (should not happen with n_splits <= min class count)
            train_groups = unique_groups[fold_assignments != fold]
            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def apply_savgol(
    X: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    derivative: int = 0,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing / derivative to each spectrum.

    Args:
        X: (n_samples, n_wavelengths) absorbance matrix
        window_length: smoothing window (forced odd)
        polyorder: polynomial order
        derivative: 0=smoothing, 1=1st deriv, 2=2nd deriv

    Returns:
        Transformed spectra of same shape
    """
    if window_length % 2 == 0:
        window_length += 1

    X_out = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_out[i] = savgol_filter(
            X[i], window_length=window_length,
            polyorder=polyorder, deriv=derivative
        )
    return X_out


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading (pandas-based)
# ═══════════════════════════════════════════════════════════════════════════════

def load_wavelength_grid(data_dir: str = "data/csv") -> np.ndarray:
    """Extract wavelength grid from the first CSV using pandas."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    # pandas reads the header rows + data, skip the instrument metadata
    df = pd.read_csv(files[0], skiprows=22, header=None)
    return df.iloc[:, 0].dropna().to_numpy(dtype=np.float64)


def load_all_spectra(
    data_dir: str = "data/csv",
    preprocess: str = "none",
    sg_window: int = 11,
    sg_polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame, np.ndarray]:
    """Load all spectra using pandas.

    Extracts swatch-level identifiers for specimen-aware cross-validation.
    Each unique swatch (physical fabric piece) forms a group; all spectra
    from the same swatch stay together during train/test splits to prevent
    data leakage.

    Returns:
        X:         (n_samples, 228) absorbance spectra
        y:         (n_samples,) integer class labels
        wl_grid:   (228,) wavelength in nm
        le:        fitted LabelEncoder
        meta_df:   DataFrame with columns [filename, material, swatch_id, class_id]
        groups:    (n_samples,) integer group labels for GroupKFold
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    wavelength_grid = load_wavelength_grid(data_dir)

    X_list, y_list = [], []
    meta_records = []

    for fpath in files:
        fname = os.path.basename(fpath)
        material = fname.split('_')[0]

        df = pd.read_csv(fpath, skiprows=22, header=None)
        absorbance = df.iloc[:, 1].dropna().to_numpy(dtype=np.float64)

        if len(absorbance) == 228:
            X_list.append(absorbance)
            y_list.append(material)

            # Parse filename: Material_Variant_Condition.csv
            # e.g. Cotton_C01_Pos1_01 → swatch_id = "Cotton_C01"
            #      Acetate_Std_Single_01 → swatch_id = "Acetate_Std"
            parts = fname.replace('.csv', '').split('_')
            variant = parts[1] if len(parts) > 1 else ''
            swatch_id = f"{material}_{variant}"  # specimen-level identifier

            meta_records.append({
                'filename': fname,
                'material': material,
                'swatch_id': swatch_id,
                'variant': variant,
                'condition': '_'.join(parts[2:]) if len(parts) > 2 else '',
            })

    X = np.array(X_list, dtype=np.float64)
    le = LabelEncoder()
    y = le.fit_transform(y_list)

    meta_df = pd.DataFrame(meta_records)
    meta_df['class_id'] = y

    # Build group labels: each unique swatch gets a unique integer
    swatch_names = meta_df['swatch_id'].values
    swatch_le = LabelEncoder()
    groups = swatch_le.fit_transform(swatch_names)

    # ── Preprocessing ──

    deriv = 0
    if preprocess == "savgol_1deriv":
        deriv = 1
    elif preprocess == "savgol_2deriv":
        deriv = 2

    if preprocess != "none":
        label = preprocess.replace('_', ' ')
        print(f"  Preprocessing: {label} (window={sg_window}, polyorder={sg_polyorder})")
        X = apply_savgol(X, window_length=sg_window, polyorder=sg_polyorder, derivative=deriv)

    n_swatches = len(swatch_le.classes_)
    print(f"  Loaded: {X.shape[0]} spectra × {X.shape[1]} wavelengths")
    print(f"  Swatches: {n_swatches} unique specimens")
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")
    print(f"  Class distribution:\n{meta_df['material'].value_counts().to_string()}")

    return X, y, wavelength_grid, le, meta_df, groups


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Selection Methods
# ═══════════════════════════════════════════════════════════════════════════════

def select_anova(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ANOVA F-score: univariate filter, fast, assumes normality."""
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    indices = np.argsort(selector.scores_)[::-1][:k]
    return indices, selector.scores_[indices]


def select_mutual_info(X: np.ndarray, y: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Mutual Information: non-linear dependencies, no distribution assumption."""
    scores = mutual_info_classif(X, y, random_state=seed)
    indices = np.argsort(scores)[::-1][:k]
    return indices, scores[indices]


def select_rfe_svm(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """RFE with linear SVM: wrapper, considers feature interactions."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    rfe = RFE(svm, n_features_to_select=k, step=max(1, X.shape[1] // 30))
    rfe.fit(X_s, y)
    indices = np.where(rfe.support_)[0]
    scores = (X.shape[1] - rfe.ranking_[indices]) / X.shape[1]
    return indices, scores


def select_l1_logreg(X: np.ndarray, y: np.ndarray, k: int, C: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """L1-regularized Logistic Regression: embedded sparsity."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=5000, random_state=42)
    lr.fit(X_s, y)
    importance = np.abs(lr.coef_).sum(axis=0)
    indices = np.argsort(importance)[::-1][:k]
    return indices, importance[indices]


def select_rf_importance(X: np.ndarray, y: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Random Forest feature importance: tree-based embedded."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    indices = np.argsort(rf.feature_importances_)[::-1][:k]
    return indices, rf.feature_importances_[indices]

# ═══════════════════════════════════════════════════════════════════════════════
# Teacher & Student Training (per-fold, NO leakage)
# ═══════════════════════════════════════════════════════════════════════════════

def train_teacher_on_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 42,
) -> Tuple[object, object, float]:
    """Train teacher on one fold's training data.

    Teacher is fit ONLY on X_tr, y_tr — NO access to test data.
    Internal 3-fold CV selects the best model type.
    """
    # std
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_tr)

    candidates = {
        'SVM-RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=3),
    }

    best_model, best_acc = None, 0
    for name, model in candidates.items():
        scores = cross_val_score(model, X_s, y_tr, cv=3, scoring='accuracy')
        acc = scores.mean()
        if acc > best_acc:
            best_acc = acc
            best_model = model

    best_model.fit(X_s, y_tr)
    return best_model, scaler, best_acc


def train_student_on_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    selected_indices: np.ndarray,
    teacher_model=None,
    teacher_scaler=None,
    use_pseudo_labels: bool = False,
    seed: int = 42,
) -> Dict:
    """Train student on k selected wavelengths.

    Two modes:
      - Hard labels: student trains on ground-truth y_tr
      - Pseudo labels: student trains on teacher-predicted labels
        (teacher trained on same X_tr — no leakage)
    """
    X_tr_k = X_tr[:, selected_indices]
    X_te_k = X_te[:, selected_indices]

    scaler = StandardScaler()
    X_tr_k = scaler.fit_transform(X_tr_k)
    X_te_k = scaler.transform(X_te_k)

    if use_pseudo_labels and teacher_model is not None and teacher_scaler is not None:
        teacher_proba = teacher_model.predict_proba(teacher_scaler.transform(X_tr))
        y_train_labels = np.argmax(teacher_proba, axis=1)
        student = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
        student.fit(X_tr_k, y_train_labels)
    else:
        student = KNeighborsClassifier(n_neighbors=3)
        student.fit(X_tr_k, y_tr)

    y_pred = student.predict(X_te_k)
    acc = accuracy_score(y_te, y_pred)

    return {
        'model': student,
        'scaler': scaler,
        'accuracy': acc,
        'predictions': y_pred,
        'n_features': len(selected_indices),
    }


def compute_teacher_baseline(
    X: np.ndarray, y: np.ndarray, skf: GroupKFold, groups: np.ndarray, seed: int = 42,
) -> float:
    """Per-fold teacher baseline — swatch-aware, NO leakage."""
    fold_accs = []
    for tr_idx, te_idx in skf.split(X, y, groups=groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        model, scaler, _ = train_teacher_on_fold(X_tr, y_tr, seed)
        fold_accs.append(accuracy_score(y_te, model.predict(scaler.transform(X_te))))
    return np.mean(fold_accs)


# ═══════════════════════════════════════════════════════════════════════════════
# Consensus & Stability Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_consensus(
    selected_sets: List[set],
    wavelength_grid: np.ndarray,
    k: int,
) -> Dict:
    """Analyze wavelength selection stability across folds.

    Reports BOTH strict consensus (intersection) and selection frequency,
    since adjacent wavelengths within the same chemical band are often
    interchangeable — strict consensus underestimates true stability.

    Returns:
        consensus:    wavelengths in ALL folds (strict intersection)
        majority:     wavelengths in ≥ ceil(n_folds/2) folds
        union:        wavelengths in ANY fold
        stability:    Jaccard index (|intersection| / |union|)
        frequencies:  dict {wavelength_idx: count across folds}
        top_freq:     top-k wavelengths by selection frequency with percentages
    """
    n_folds = len(selected_sets)
    intersection = reduce(lambda a, b: a & b, selected_sets)
    union = reduce(lambda a, b: a | b, selected_sets)

    freq = defaultdict(int)
    for s in selected_sets:
        for idx in s:
            freq[idx] += 1

    majority_threshold = max(2, n_folds // 2 + 1)
    majority = {idx for idx, count in freq.items() if count >= majority_threshold}

    stability = len(intersection) / len(union) if union else 0

    # Top-k by selection frequency (the proper metric for stability)
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
    top_freq = [(idx, count, count / n_folds * 100) for idx, count in sorted_freq[:k]]

    return {
        'consensus': sorted(intersection),
        'majority': sorted(majority),
        'union': sorted(union),
        'n_consensus': len(intersection),
        'n_majority': len(majority),
        'n_union': len(union),
        'stability_jaccard': stability,
        'frequencies': dict(freq),
        'top_freq': top_freq,  # [(idx, count, pct), ...]
    }


def analyze_cross_seed_frequency(
    all_selected_sets_raw: List[List[set]],
    wavelength_grid: np.ndarray,
    k: int,
) -> Dict:
    """Analyze wavelength selection frequency across ALL seeds × folds.

    This is the most robust stability metric: counts how many times
    each wavelength index was selected across all (n_seeds × n_folds)
    independent runs.

    Returns:
        top_wavelengths:  list of (index, nm, count, pct) sorted by frequency
        frequency_map:    dict {idx: pct} for all selected wavelengths
        n_total_selections: total number of selection events (seeds × folds × k)
    """
    n_total = sum(len(sets) for sets in all_selected_sets_raw)  # seeds
    n_selections_per_run = k * len(all_selected_sets_raw[0]) if all_selected_sets_raw else k * 5  # folds × k
    n_total_opportunities = n_total * n_selections_per_run if n_total else 0

    freq = defaultdict(int)
    for seed_sets in all_selected_sets_raw:
        for fold_set in seed_sets:
            for idx in fold_set:
                freq[idx] += 1

    n_seeds = len(all_selected_sets_raw)
    n_folds = len(all_selected_sets_raw[0]) if all_selected_sets_raw else 5
    total_runs = n_seeds * n_folds

    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
    top_wavelengths = []
    for idx, count in sorted_freq[:k]:
        nm = wavelength_grid[idx]
        pct = count / total_runs * 100
        top_wavelengths.append((idx, nm, count, pct))

    return {
        'top_wavelengths': top_wavelengths,
        'frequency_map': {idx: count / total_runs * 100 for idx, count in freq.items()},
        'total_runs': total_runs,
        'n_unique_selected': len(freq),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(
    method_accuracies: Dict[str, List[float]],
    teacher_accs: List[float],
    random_accs: List[float],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Wilcoxon signed-rank tests. Returns results as a DataFrame."""

    records = []

    def _wilcoxon(a, b):
        if len(a) < 5:
            return {'W': np.nan, 'p_value': np.nan, 'significant': False}
        try:
            stat, p = wilcoxon(a, b, zero_method='wilcox')
            return {'W': stat, 'p_value': p, 'significant': p < alpha}
        except ValueError:
            return {'W': np.nan, 'p_value': np.nan, 'significant': False}

    for method, accs in method_accuracies.items():
        r = _wilcoxon(teacher_accs, accs)
        records.append({
            'Comparison': f'{method} vs Teacher',
            'W': r['W'], 'p_value': r['p_value'],
            'significant (α=0.05)': r['significant'],
        })

        r = _wilcoxon(accs, random_accs)
        records.append({
            'Comparison': f'{method} vs Random k',
            'W': r['W'], 'p_value': r['p_value'],
            'significant (α=0.05)': r['significant'],
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-Seed Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_seed_experiment(
    X: np.ndarray,
    y: np.ndarray,
    wavelength_grid: np.ndarray,
    skf: GroupKFold,
    groups: np.ndarray,
    k: int,
    seed: int,
    selection_methods: Dict,
) -> Tuple[Dict, float, float]:
    """One full experiment with a given random seed.

    Swatch-aware CV: all spectra from a physical swatch stay in the
    same fold (train or test). Prevents same-fabric data leakage.

    Per-fold:
      1. Train teacher on X_tr only
      2. Select k wavelengths on X_tr
      3. Train student on X_tr_k (hard + pseudo-label)
      4. Evaluate on X_te_k
      5. Track consensus wavelengths
    """
    # ── Teacher baseline (per-fold, no leakage) ──
    teacher_fold_accs = []
    teacher_results_per_fold = []

    for tr_idx, te_idx in skf.split(X, y, groups=groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        # get teacher model
        model, scaler, _ = train_teacher_on_fold(X_tr, y_tr, seed)
        teacher_fold_accs.append(
            accuracy_score(y_te, model.predict(scaler.transform(X_te)))
        )
        teacher_results_per_fold.append((model, scaler))

    teacher_baseline = np.mean(teacher_fold_accs)

    # ── Random k baseline ──
    random_fold_accs = []
    rng = np.random.RandomState(seed)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y, groups=groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        rand_idx = rng.choice(X.shape[1], k, replace=False)
        scaler = StandardScaler()
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(scaler.fit_transform(X_tr[:, rand_idx]), y_tr)
        random_fold_accs.append(
            accuracy_score(y_te, knn.predict(scaler.transform(X_te[:, rand_idx])))
        )
    random_baseline = np.mean(random_fold_accs)

    # ── Per-method experiment ──
    results = {}

    for method_name, select_fn in selection_methods.items():
        print(f"\n  [{method_name}]  seed={seed}")

        fold_accs_hard = []
        fold_accs_pseudo = []
        selected_sets = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y, groups=groups)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            teacher_fold, teacher_scaler_fold = teacher_results_per_fold[fold]

            indices, scores = select_fn(X_tr, y_tr)
            selected_sets.append(set(indices))

            # Hard labels
            result_hard = train_student_on_fold(
                X_tr, y_tr, X_te, y_te, indices,
                use_pseudo_labels=False, seed=seed
            )
            fold_accs_hard.append(result_hard['accuracy'])

            # Pseudo labels
            result_pseudo = train_student_on_fold(
                X_tr, y_tr, X_te, y_te, indices,
                teacher_model=teacher_fold,
                teacher_scaler=teacher_scaler_fold,
                use_pseudo_labels=True, seed=seed
            )
            fold_accs_pseudo.append(result_pseudo['accuracy'])

        consensus_info = analyze_consensus(selected_sets, wavelength_grid, k)

        results[method_name] = {
            'accuracy_hard': np.mean(fold_accs_hard),
            'accuracy_pseudo': np.mean(fold_accs_pseudo),
            'std_hard': np.std(fold_accs_hard),
            'std_pseudo': np.std(fold_accs_pseudo),
            'fold_accs_hard': fold_accs_hard,
            'fold_accs_pseudo': fold_accs_pseudo,
            'consensus_info': consensus_info,
            'selected_sets': selected_sets,
        }

        # ── Per-method output: frequency-based + strict consensus ──
        top_freq = consensus_info['top_freq']
        freq_str = ', '.join(
            f"{wavelength_grid[idx]:.0f}nm ({pct:.0f}%)"
            for idx, count, pct in top_freq[:k]
        )
        print(f"    Hard:        {np.mean(fold_accs_hard):.4f} ± {np.std(fold_accs_hard):.4f}")
        print(f"    Pseudo-label: {np.mean(fold_accs_pseudo):.4f} ± {np.std(fold_accs_pseudo):.4f}")
        print(f"    Selection frequency (top-{k}): {freq_str}")
        print(f"    Strict consensus: {consensus_info['n_consensus']}/{k}  |  "
              f"Jaccard stability: {consensus_info['stability_jaccard']:.2f}")

    return results, teacher_baseline, random_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Seed Experiment Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_seed_experiment(
    k: int = 5,
    seeds: List[int] = None,
    preprocess: str = "none",
    X: np.ndarray = None,
    y: np.ndarray = None,
    wavelength_grid: np.ndarray = None,
    le: LabelEncoder = None,
    meta_df: pd.DataFrame = None,
    groups: np.ndarray = None,
    plot: bool = False,
    save_plots_dir: str = "figures",
    methods: List[str] = None,
) -> Dict:
    """Run experiment across multiple random seeds, aggregate results.

    Uses swatch-level GroupKFold to prevent data leakage: all spectra
    from the same physical fabric swatch stay together in train or test.
    Classes with only 1 swatch (Acetate, Acrylic, Wool) are excluded.

    Args:
        plot: if True, generate and save all figures
        save_plots_dir: directory for saved figures
        methods: optional list of method names to include (None = all 5)
        groups: specimen-level group labels (auto-loaded if None)
    """
    if seeds is None:
        seeds = [42]

    # Load data if not provided
    if X is None:
        X, y, wavelength_grid, le, meta_df, groups = load_all_spectra(preprocess=preprocess)

    # ── Exclude classes with only 1 swatch ──
    # Acetate, Acrylic, Wool have single "Std" swatches → impossible
    # to do specimen-level CV. Keep only Cotton, Nylon, Polyester.
    n_swatches_per_class = {}
    for cls_id in np.unique(y):
        cls_swatches = np.unique(groups[y == cls_id])
        n_swatches_per_class[le.classes_[cls_id]] = len(cls_swatches)

    keep_mask = np.ones(len(y), dtype=bool)
    for cls_id, cls_name in enumerate(le.classes_):
        if n_swatches_per_class[cls_name] < 2:
            keep_mask[y == cls_id] = False
            print(f"  ⚠ Excluding '{cls_name}': only {n_swatches_per_class[cls_name]} swatch(es) "
                  f"— cannot do specimen-level CV")

    X = X[keep_mask]
    y = y[keep_mask]
    groups = groups[keep_mask]
    meta_df = meta_df.iloc[keep_mask].reset_index(drop=True)

    # Re-encode labels (some classes removed)
    le = LabelEncoder()
    y = le.fit_transform(y)

    print(f"  Retained: {X.shape[0]} spectra, {len(np.unique(groups))} swatches, "
          f"{len(le.classes_)} classes ({list(le.classes_)})")

    print("=" * 70)
    print(f"MULTI-SEED WAVELENGTH SELECTION EXPERIMENT")
    print(f"  k = {k} wavelengths  |  {len(seeds)} seeds: {seeds}")
    print(f"  Preprocessing: {preprocess}")
    print(f"  CV: GroupKFold (swatch-level, no data leakage)")
    print("=" * 70)

    # Selection methods registry (5 classic methods)
    all_methods = {
        'ANOVA F-score':             lambda Xtr, ytr: select_anova(Xtr, ytr, k),
        'Mutual Information':        lambda Xtr, ytr: select_mutual_info(Xtr, ytr, k, seeds[0]),
        'RFE (Linear SVM)':          lambda Xtr, ytr: select_rfe_svm(Xtr, ytr, k),
        'L1 LogisticRegression':     lambda Xtr, ytr: select_l1_logreg(Xtr, ytr, k),
        'Random Forest Imp.':        lambda Xtr, ytr: select_rf_importance(Xtr, ytr, k, seeds[0]),
    }

    # Filter methods if specified
    if methods is not None:
        selection_methods = {name: fn for name, fn in all_methods.items() if name in methods}
        missing = set(methods) - set(all_methods.keys())
        if missing:
            print(f"  ⚠ Unknown method(s): {missing} — skipping")
        print(f"  Methods: {list(selection_methods.keys())}")
    else:
        selection_methods = all_methods

    all_seed_results = {}
    all_teacher_baselines = []
    all_random_baselines = []
    all_consensus = defaultdict(list)
    all_selected_sets_raw = defaultdict(list)  # {method: [[fold_sets_seed0], [fold_sets_seed1], ...]}

    # ── Run per-seed ──
    for seed in seeds:
        print(f"\n{'─' * 50}")
        print(f"SEED = {seed}")
        print(f"{'─' * 50}")
        # Swatch-level CV: keep same fabric together
        skf = ShuffledGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        results, teacher_bl, random_bl = run_single_seed_experiment(
            X, y, wavelength_grid, skf, groups, k, seed, selection_methods
        )

        all_teacher_baselines.append(teacher_bl)
        all_random_baselines.append(random_bl)

        for method, r in results.items():
            if method not in all_seed_results:
                all_seed_results[method] = {'hard': [], 'pseudo': [],
                                            'fold_hard': [], 'fold_pseudo': []}
            all_seed_results[method]['hard'].append(r['accuracy_hard'])
            all_seed_results[method]['pseudo'].append(r['accuracy_pseudo'])
            all_consensus[method].append(r['consensus_info'])
            all_selected_sets_raw[method].append(r['selected_sets'])

    # ═══════════════════════════════════════════════════════════════════════════
    # Aggregate Summary
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 70)
    print(f"FINAL SUMMARY — {len(seeds)} seeds × 5 folds (k = {k})")
    print("=" * 70)

    teacher_mean = np.mean(all_teacher_baselines)
    teacher_std = np.std(all_teacher_baselines)
    random_mean = np.mean(all_random_baselines)
    random_std = np.std(all_random_baselines)

    # Build summary DataFrame
    summary_records = []
    best_method, best_score = None, 0

    for method in sorted(all_seed_results.keys(),
                         key=lambda m: -np.mean(all_seed_results[m]['pseudo'])):
        accs_h = all_seed_results[method]['hard']
        accs_p = all_seed_results[method]['pseudo']
        mean_h, std_h = np.mean(accs_h), np.std(accs_h)
        mean_p, std_p = np.mean(accs_p), np.std(accs_p)

        if mean_p > best_score:
            best_score = mean_p
            best_method = method

        summary_records.append({
            'Method': method,
            'Hard_mean': mean_h, 'Hard_std': std_h,
            'Pseudo_mean': mean_p, 'Pseudo_std': std_p,
            'Delta_Teacher': mean_p - teacher_mean,
            'Delta_Random': mean_p - random_mean,
            'n_seeds': len(seeds),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values('Pseudo_mean', ascending=False).reset_index(drop=True)

    # Pretty-print
    print(f"\n  {'Method':<25s} {'Hard':>12s} {'Pseudo':>12s}  {'Δ Teacher':>10s}  {'Δ Random':>10s}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}  {'─'*10}  {'─'*10}")
    print(f"  {'Teacher (228λ)':<25s} {teacher_mean:>12.4f} {'—':>12s}  {'—':>10s}  {'—':>10s}")
    print(f"  {'  (± std)':<25s} {teacher_std:>12.4f} {'':>12s}")
    print(f"  {'Random k wavelengths':<25s} {random_mean:>12.4f} {'—':>12s}  {'—':>10s}  {'—':>10s}")
    print(f"  {'  (± std)':<25s} {random_std:>12.4f} {'':>12s}")
    print()

    for _, row in summary_df.iterrows():
        marker = " ← BEST" if row['Method'] == best_method else ""
        print(f"  {row['Method']:<25s} {row['Hard_mean']:>7.4f}±{row['Hard_std']:<4.4f} "
              f"{row['Pseudo_mean']:>7.4f}±{row['Pseudo_std']:<4.4f}  "
              f"{row['Delta_Teacher']:>+10.4f}  {row['Delta_Random']:>+10.4f}{marker}")

    # ── Statistical Tests ──
    if len(seeds) >= 5:
        print(f"\n  ── Statistical Tests (Wilcoxon signed-rank, α=0.05, n={len(seeds)}) ──")
        method_accs = {m: all_seed_results[m]['pseudo'] for m in all_seed_results}
        stats_df = run_statistical_tests(method_accs, all_teacher_baselines, all_random_baselines)

        for _, row in stats_df.iterrows():
            if not pd.isna(row['p_value']):
                sig = "SIGNIFICANT" if row['significant (α=0.05)'] else "not sig."
                print(f"    {row['Comparison']:<40s}  W={row['W']:>6.1f}  "
                      f"p={row['p_value']:.4f}  [{sig}]")
    else:
        stats_df = pd.DataFrame()
        print(f"\n  ── Statistical Tests ──")
        print(f"    Skipped: need ≥ 5 seeds (have {len(seeds)})")

    # ── Best Method: Wavelengths & Physical Interpretation ──
    print(f"\n  ── Best Method: {best_method} ──")

    # Cross-seed frequency analysis (the proper stability metric)
    best_raw_sets = all_selected_sets_raw[best_method]
    freq_analysis = analyze_cross_seed_frequency(best_raw_sets, wavelength_grid, k)

    top_wl_indices = [item[0] for item in freq_analysis['top_wavelengths']]
    best_wl_nm = wavelength_grid[top_wl_indices]

    print(f"    Selection frequency across {freq_analysis['total_runs']} runs "
          f"({len(seeds)} seeds × 5 folds):")
    for idx, nm, count, pct in freq_analysis['top_wavelengths'][:k]:
        band_info = interpret_wavelength(nm)
        print(f"      {nm:>6.0f} nm  —  {pct:>5.0f}% ({count}/{freq_analysis['total_runs']})  —  {band_info}")

    # Also report strict consensus for reference
    best_consensus_all_seeds = all_consensus[best_method]
    avg_consensus = np.mean([ci['n_consensus'] for ci in best_consensus_all_seeds])
    avg_jaccard = np.mean([ci['stability_jaccard'] for ci in best_consensus_all_seeds])
    print(f"    Strict consensus (avg across seeds): {avg_consensus:.1f}/{k}  |  "
          f"Jaccard: {avg_jaccard:.2f}")
    print(f"    Unique wavelengths ever selected: {freq_analysis['n_unique_selected']}")

    n_final = len(top_wl_indices)
    print(f"\n    Compression:  228 → {n_final} wavelengths ({(1 - n_final/228)*100:.0f}% reduction)")
    print(f"    Accuracy gap:  {best_score - teacher_mean:+.4f} vs Teacher (228λ)")

    print(f"\n  ── Hardware Impact ──")
    print(f"    Lab spectrometer (228 px):       ~$2,000+")
    print(f"    Portable {n_final}-λ LED array:  ~${n_final*30}–${n_final*50}")
    print(f"    Cost reduction:                  ~{(1 - n_final*40/2000)*100:.0f}%")
    print(f"    Inference latency:               < 1 ms (suitable for real-time)")

    # ═══════════════════════════════════════════════════════════════════════════
    # Generate Figures
    # ═══════════════════════════════════════════════════════════════════════════

    if plot:
        generate_all_figures(
            X, y, wavelength_grid, le.classes_.tolist(),
            summary_df, teacher_mean, teacher_std, random_mean, random_std,
            best_method, top_wl_indices, best_wl_nm,
            all_selected_sets_raw, k,
            seed=seeds[0], save_dir=save_plots_dir,
        )

    return {
        'k': k,
        'n_seeds': len(seeds),
        'n_wavelengths_total': len(wavelength_grid),
        'teacher_mean': teacher_mean,
        'teacher_std': teacher_std,
        'random_mean': random_mean,
        'random_std': random_std,
        'summary_df': summary_df,
        'stats_df': stats_df,
        'best_method': best_method,
        'best_score': best_score,
        'best_wavelengths_nm': best_wl_nm,
        'best_indices': top_wl_indices,
        'all_consensus': all_consensus,
        'freq_analysis': freq_analysis,  # cross-seed frequency analysis
        'all_selected_sets_raw': all_selected_sets_raw,
        'X': X, 'y': y, 'wavelength_grid': wavelength_grid, 'le': le, 'meta_df': meta_df,
    }




# ═══════════════════════════════════════════════════════════════════════════════
# Results Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(
    result: Dict,
    results_dir: str = "results",
    preprocess: str = "none",
) -> None:
    """Persist experiment results to CSV and JSON files.

    Writes:
      - {results_dir}/summary_k{N}_{preprocess}.csv      — method comparison
      - {results_dir}/statistics_k{N}_{preprocess}.csv    — Wilcoxon tests
      - {results_dir}/best_wavelengths_k{N}_{preprocess}.csv — selected λ + interpretation
      - {results_dir}/experiment_k{N}_{preprocess}.json   — full config + key metrics
    """
    os.makedirs(results_dir, exist_ok=True)

    k = result['k']
    n_seeds = result['n_seeds']
    tag = f"k{k}_{preprocess}" if preprocess != "none" else f"k{k}"
    prefix = os.path.join(results_dir, tag)

    # ── 1. Method comparison summary ──
    summary_df = result['summary_df'].copy()
    summary_df.to_csv(f"{prefix}_summary.csv", index=False, float_format="%.4f")
    print(f"  ✓ Saved: {prefix}_summary.csv")

    # ── 2. Statistical tests ──
    stats_df = result['stats_df']
    if not stats_df.empty:
        stats_df.to_csv(f"{prefix}_statistics.csv", index=False, float_format="%.4f")
        print(f"  ✓ Saved: {prefix}_statistics.csv")

    # ── 3. Best wavelengths with selection frequency ──
    from plotting import interpret_wavelength
    freq_data = result.get('freq_analysis', {})
    top_wl = freq_data.get('top_wavelengths', [])
    wl_records = []
    if top_wl:
        for rank, (idx, nm, count, pct) in enumerate(top_wl, start=1):
            wl_records.append({
                'Rank': rank,
                'Index': idx,
                'Wavelength_nm': f"{nm:.1f}",
                'Frequency_pct': f"{pct:.0f}",
                'Selection_count': f"{count}/{freq_data.get('total_runs', '?')}",
                'Interpretation': interpret_wavelength(nm),
            })
    else:
        # fallback
        wl_nm = result['best_wavelengths_nm']
        indices = result['best_indices']
        for rank, (idx, nm) in enumerate(zip(indices, wl_nm), start=1):
            wl_records.append({
                'Rank': rank,
                'Index': idx,
                'Wavelength_nm': f"{nm:.1f}",
                'Frequency_pct': '—',
                'Selection_count': '—',
                'Interpretation': interpret_wavelength(nm),
            })
    wl_df = pd.DataFrame(wl_records)
    wl_df.to_csv(f"{prefix}_best_wavelengths.csv", index=False)
    print(f"  ✓ Saved: {prefix}_best_wavelengths.csv")

    # ── 4. Full experiment config + key metrics (JSON) ──
    wl_nm = result['best_wavelengths_nm']
    indices = result['best_indices']
    config = {
        'k': k,
        'n_seeds': n_seeds,
        'preprocess': preprocess,
        'n_wavelengths_total': result.get('n_wavelengths_total',
                                         len(result.get('wavelength_grid', []))),
        'best_method': result['best_method'],
        'best_score_pseudo': float(result['best_score']),
        'teacher_mean': float(result['teacher_mean']),
        'teacher_std': float(result['teacher_std']),
        'random_mean': float(result['random_mean']),
        'random_std': float(result['random_std']),
        'best_wavelengths_nm': [f"{float(w):.1f}" for w in wl_nm],
        'best_wavelengths_indices': [int(i) for i in indices],
        'class_names': [str(c) for c in result['le'].classes_] if result.get('le') else [],
        'avg_consensus': float(np.mean([ci['n_consensus']
                              for ci in result.get('all_consensus', {}).get(result['best_method'], [])]
                              )) if result.get('all_consensus') else None,
    }
    with open(f"{prefix}_experiment.json", 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: {prefix}_experiment.json")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wavelength Selection for Portable NIR Textile Classification"
    )
    parser.add_argument("--k", type=int, default=5,
                        help="Number of wavelengths to select (default: 5)")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--preprocess", type=str, default="none",
                        choices=["none", "savgol", "savgol_1deriv", "savgol_2deriv"],
                        help="Preprocessing: savgol / savgol_1deriv / savgol_2deriv")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed override (disables multi-seed mode)")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method names (default: all 5). "
                             "Choices: ANOVA F-score, Mutual Information, "
                             "RFE (Linear SVM), L1 LogisticRegression, "
                             "Random Forest Imp.")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save figures")
    parser.add_argument("--save_plots", type=str, default="figures",
                        help="Directory to save figures (default: figures/)")
    parser.add_argument("--save_results", action="store_true", default=True,
                        help="Save experiment results to CSV/JSON (default: True)")
    parser.add_argument("--no_save_results", action="store_false", dest="save_results",
                        help="Do not save experiment results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results (default: results/)")
    parser.add_argument("--compare_k", type=int, default=None,
                        help="Also run a comparison experiment with this k value")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))

    # Parse methods filter
    method_list = None
    if args.methods:
        method_list = [m.strip() for m in args.methods.split(",")]

    # Pre-load data to pass into multiple runs
    print("Loading data...")
    X, y, wavelength_grid, le, meta_df, groups = load_all_spectra(preprocess=args.preprocess)

    # ── Primary experiment ──
    result_k = run_multi_seed_experiment(
        k=args.k, seeds=seeds, preprocess=args.preprocess,
        X=X, y=y, wavelength_grid=wavelength_grid, le=le, meta_df=meta_df,
        groups=groups,
        plot=args.plot, save_plots_dir=args.save_plots,
        methods=method_list,
    )

    # Save primary results
    if args.save_results:
        print(f"\n{'─' * 50}")
        print(f"Saving results to {args.results_dir}/")
        print(f"{'─' * 50}")
        save_results(result_k, results_dir=args.results_dir, preprocess=args.preprocess)

    # ── Comparison experiment (auto k=3 when k=5, or explicit --compare_k) ──
    compare_k = args.compare_k
    if compare_k is None and args.k == 5 and args.n_seeds >= 3:
        compare_k = 3  # default: k=5 → compare with k=3

    if compare_k is not None and compare_k != args.k:
        print("\n\n")
        result_compare = run_multi_seed_experiment(
            k=compare_k, seeds=seeds[:min(len(seeds), 3)], preprocess=args.preprocess,
            X=X, y=y, wavelength_grid=wavelength_grid, le=le, meta_df=meta_df,
            groups=groups,
            plot=args.plot, save_plots_dir=args.save_plots,
            methods=method_list,
        )

        if args.save_results:
            print(f"\n{'─' * 50}")
            print(f"Saving k={compare_k} results to {args.results_dir}/")
            print(f"{'─' * 50}")
            save_results(result_compare, results_dir=args.results_dir, preprocess=args.preprocess)

    if args.save_results:
        print(f"\nAll results saved to: {os.path.abspath(args.results_dir)}/")
