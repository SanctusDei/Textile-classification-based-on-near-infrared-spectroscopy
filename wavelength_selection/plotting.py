"""
Visualization module for wavelength selection experiments.
=========================================================
Publication-quality figures at 300 DPI using matplotlib + seaborn.

Figures:
  1. Mean spectra per class with selected wavelengths highlighted
  2. Method comparison bar chart (vs Teacher/Random baselines)
  3. Consensus heatmaps (wavelength × seed fold-frequency)
  4. Physical interpretation (chemical bond annotations on grand-mean spectrum)
  5. Confusion matrix (counts + normalized recall)

Usage:
  from plotting import plot_mean_spectra, plot_method_comparison, ...

  # Or call the convenience wrapper to generate all 5 figures:
  from plotting import generate_all_figures
  generate_all_figures(X, y, wavelength_grid, le, summary_df, ...)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Global style ─────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════════════════════════
# Chemical bond interpretation helper
# ═══════════════════════════════════════════════════════════════════════════════════

def interpret_wavelength(wl_nm: float) -> str:
    """Provide physical interpretation of a NIR wavelength.

    Maps wavelength regions to chemical bond overtones commonly observed
    in polymer and natural fiber NIR spectra.

    Args:
        wl_nm: wavelength in nanometers (900–1700)

    Returns:
        string describing the dominant chemical bond vibration
    """
    if 900 <= wl_nm <= 950:
        return "C–H 3rd overtone (aliphatic)"
    elif 1000 <= wl_nm <= 1050:
        return "N–H 2nd overtone (amide/protein)"
    elif 1100 <= wl_nm <= 1250:
        return "C–H 2nd overtone (polymer backbone — PET, PA)"
    elif 1350 <= wl_nm <= 1450:
        return "O–H 1st overtone (cellulose/water — Cotton); C–H combination"
    elif 1450 <= wl_nm <= 1550:
        return "N–H 1st overtone (amide — Wool, Nylon, Silk)"
    elif 1550 <= wl_nm <= 1700:
        return "C–H 1st overtone (CH3, CH2, aromatic — PET); O–H (Cotton)"
    else:
        return "—"


# ═══════════════════════════════════════════════════════════════════════════════════
# Figure 1: Mean spectra per class
# ═══════════════════════════════════════════════════════════════════════════════════

def plot_mean_spectra(
    X: np.ndarray,
    y: np.ndarray,
    wavelength_grid: np.ndarray,
    class_names: list,
    selected_wl_indices: Optional[np.ndarray] = None,
    selected_wl_labels: Optional[list] = None,
    title: str = "Mean NIR Absorbance Spectra by Textile Class",
    save_path: Optional[str] = None,
):
    """Figure 1: Mean spectrum per class ± 1 std, selected wavelengths highlighted.

    Args:
        X: (n_samples, n_wavelengths) absorbance matrix
        y: (n_samples,) integer class labels
        wavelength_grid: (n_wavelengths,) wavelength in nm
        class_names: list of class name strings (from LabelEncoder.classes_)
        selected_wl_indices: array of selected wavelength indices to mark with vertical lines
        selected_wl_labels:  annotation labels for the markers
        title: plot title
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("tab10", len(class_names))

    for cls_id, cls_name in enumerate(class_names):
        mask = y == cls_id
        mean_spec = X[mask].mean(axis=0)
        std_spec = X[mask].std(axis=0)

        ax.plot(wavelength_grid, mean_spec, color=palette[cls_id],
                label=f'{cls_name} (n={mask.sum()})', linewidth=1.5)
        ax.fill_between(wavelength_grid,
                        mean_spec - std_spec, mean_spec + std_spec,
                        color=palette[cls_id], alpha=0.1)

    # Mark selected wavelengths
    if selected_wl_indices is not None and len(selected_wl_indices) > 0:
        sel_nm = wavelength_grid[selected_wl_indices]
        y_lim = ax.get_ylim()
        for i, (idx, nm) in enumerate(zip(selected_wl_indices, sel_nm)):
            ax.axvline(x=nm, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            label = selected_wl_labels[i] if selected_wl_labels else f'{nm:.0f}'
            ax.annotate(label, xy=(nm, y_lim[1] * 0.95),
                        rotation=90, fontsize=7, color='red',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance (AU)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim(wavelength_grid[0], wavelength_grid[-1])

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"  Figure saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════════
# Figure 2: Method comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════════

def plot_method_comparison(
    summary_df: pd.DataFrame,
    teacher_mean: float,
    teacher_std: float,
    random_mean: float,
    random_std: float,
    k: int,
    save_path: Optional[str] = None,
):
    """Figure 2: Bar chart comparing all methods vs baselines.

    Args:
        summary_df: DataFrame with columns [Method, Pseudo_mean, Pseudo_std, n_seeds]
        teacher_mean, teacher_std: full-spectrum teacher baseline
        random_mean, random_std: random-k baseline
        k: number of selected wavelengths
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = summary_df['Method'].tolist()
    means = summary_df['Pseudo_mean'].tolist()
    stds = summary_df['Pseudo_std'].tolist()

    colors = sns.color_palette("Blues_r", len(methods))
    x = np.arange(len(methods))

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='gray',
                  linewidth=0.8, capsize=4, width=0.55)

    # Baseline lines
    ax.axhline(y=teacher_mean, color='#d62728', linestyle='-', linewidth=1.5,
               label=f'Teacher (228λ): {teacher_mean:.4f} ± {teacher_std:.4f}')
    ax.axhspan(teacher_mean - teacher_std, teacher_mean + teacher_std,
               color='#d62728', alpha=0.08)
    ax.axhline(y=random_mean, color='#7f7f7f', linestyle=':', linewidth=1.5,
               label=f'Random k={k}: {random_mean:.4f} ± {random_std:.4f}')

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                f'{mean:.4f}', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (Pseudo-label mode)')
    n_seeds = summary_df['n_seeds'].iloc[0]
    ax.set_title(f'Method Comparison — k={k} Wavelengths\n'
                 f'({len(methods)} methods, {n_seeds} seeds × 5 folds)')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0.3, 1.05)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"  Figure saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════════
# Figure 3: Consensus heatmaps
# ═══════════════════════════════════════════════════════════════════════════════════

def plot_consensus_heatmap(
    all_selected_sets: Dict[str, list],
    wavelength_grid: np.ndarray,
    k: int,
    save_dir: str = "figures",
):
    """Figure 3: Heatmap — how often each wavelength was selected per seed × fold.

    Grid: rows = seeds (0..n_seeds-1), cols = wavelength range (zoomed to active region).
    Cell color = number of folds (0–5) that selected that wavelength.

    One heatmap per method.

    Args:
        all_selected_sets: {method_name: [[fold_sets_seed0], [fold_sets_seed1], ...]}
            where each fold_sets is a list of 5 sets of wavelength indices
        wavelength_grid: (n_wavelengths,) wavelength in nm
        k: number of selected wavelengths
        save_dir: directory for saved figures
    """
    os.makedirs(save_dir, exist_ok=True)

    for method_name, seed_fold_sets in all_selected_sets.items():
        n_seeds = len(seed_fold_sets)
        n_wl = len(wavelength_grid)
        freq_matrix = np.zeros((n_seeds, n_wl))

        for seed_idx, fold_sets in enumerate(seed_fold_sets):
            for s in fold_sets:
                for idx in s:
                    freq_matrix[seed_idx, idx] += 1

        fig, ax = plt.subplots(figsize=(14, 2 + n_seeds * 0.3))

        # Zoom to active wavelength region
        active_cols = np.where(freq_matrix.sum(axis=0) > 0)[0]
        if len(active_cols) == 0:
            plt.close(fig)
            continue

        pad = 5
        col_start = max(0, active_cols[0] - pad)
        col_end = min(n_wl, active_cols[-1] + pad + 1)
        freq_zoomed = freq_matrix[:, col_start:col_end]
        wl_zoomed = wavelength_grid[col_start:col_end]

        sns.heatmap(freq_zoomed, ax=ax, cmap='YlOrRd', annot=False,
                    xticklabels=[f'{w:.0f}' for w in wl_zoomed],
                    yticklabels=[f'Seed {i}' for i in range(n_seeds)],
                    cbar_kws={'label': 'Fold count (max 5)'},
                    linewidths=0.5, linecolor='white', vmin=0, vmax=5)

        ax.set_title(f'{method_name} — Wavelength Selection Frequency (k={k})')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=90, labelsize=6)

        fig.tight_layout()
        safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
        fname = f'{save_dir}/consensus_heatmap_{safe_name}_{k}wl.png'
        fig.savefig(fname)
        print(f"  Figure saved: {fname}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════════
# Figure 4: Physical interpretation
# ═══════════════════════════════════════════════════════════════════════════════════

def plot_physical_interpretation(
    X: np.ndarray,
    y: np.ndarray,
    wavelength_grid: np.ndarray,
    class_names: list,
    top_wl_indices: np.ndarray,
    k: int,
    save_path: Optional[str] = None,
):
    """Figure 4: Grand-mean spectrum with selected wavelengths annotated by chemical bond.

    Overlays all class means (faded) and annotates each selected wavelength
    with its chemical bond assignment. Best for justifying wavelength choices
    in a paper's Discussion section.

    Args:
        X: (n_samples, n_wavelengths) absorbance matrix
        y: (n_samples,) integer class labels
        wavelength_grid: (n_wavelengths,) wavelength in nm
        class_names: list of class name strings
        top_wl_indices: indices of selected wavelengths to annotate
        k: number of selected wavelengths
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    grand_mean = X.mean(axis=0)
    ax.plot(wavelength_grid, grand_mean, color='#333333', linewidth=1.2, alpha=0.6)

    # Overlay per-class means (faded)
    palette = sns.color_palette("tab10", len(class_names))
    for cls_id, cls_name in enumerate(class_names):
        mask = y == cls_id
        ax.plot(wavelength_grid, X[mask].mean(axis=0),
                color=palette[cls_id], linewidth=0.6, alpha=0.4, label=f'{cls_name}')

    # Annotate selected wavelengths with chemical bonds
    sel_nm = wavelength_grid[top_wl_indices]
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min

    for i, (idx, nm) in enumerate(zip(top_wl_indices, sel_nm)):
        bond_info = interpret_wavelength(nm)

        ax.axvline(x=nm, color='#d62728', linestyle='--', alpha=0.6, linewidth=1.0)

        # Alternate annotation height to avoid overlapping text
        offset = y_max + y_span * 0.02 * (1 if i % 2 == 0 else 1.8)
        ax.annotate(
            f'{nm:.0f} nm\n{bond_info}',
            xy=(nm, grand_mean[idx]),
            xytext=(nm, offset),
            fontsize=6, color='#8b0000',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8,
                            connectionstyle='arc3,rad=0.1'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5',
                      edgecolor='#d62728', alpha=0.85),
        )

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance (AU)')
    ax.set_title(f'Physical Interpretation of Selected Wavelengths (k={k})')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.set_xlim(wavelength_grid[0], wavelength_grid[-1])

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"  Figure saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════════
# Figure 5: Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix_best(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list,
    wavelength_grid: np.ndarray,
    best_method_name: str,
    best_indices: np.ndarray,
    k: int,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """Figure 5: Confusion matrix (raw counts + normalized recall) for best method.

    Trains a fresh KNN student on a 75/25 stratified split using the selected
    wavelengths, then plots both raw and row-normalized confusion matrices.

    Args:
        X: (n_samples, n_wavelengths) absorbance matrix
        y: (n_samples,) integer class labels
        class_names: list of class name strings
        wavelength_grid: (n_wavelengths,) wavelength in nm (unused, kept for API consistency)
        best_method_name: name of the best-performing method (for title)
        best_indices: selected wavelength indices for the best method
        k: number of selected wavelengths
        seed: random seed for the train/test split
        save_path: if provided, save figure to this path
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    X_tr_k = StandardScaler().fit_transform(X_tr[:, best_indices])
    X_te_k = StandardScaler().fit_transform(X_te[:, best_indices])

    student = KNeighborsClassifier(n_neighbors=3)
    student.fit(X_tr_k, y_tr)
    y_pred = student.predict(X_te_k)

    cm = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{best_method_name} — Confusion Matrix (Counts, k={k})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized (recall)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    axes[1].set_title(f'{best_method_name} — Normalized (Recall, k={k})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    acc = accuracy_score(y_te, y_pred)
    fig.suptitle(f'Best Method: {best_method_name}  |  Test Accuracy: {acc:.4f}  |  k={k}',
                 fontsize=13, fontweight='bold', y=1.02)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"  Figure saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════════
# Convenience: generate all figures at once
# ═══════════════════════════════════════════════════════════════════════════════════

def generate_all_figures(
    X: np.ndarray,
    y: np.ndarray,
    wavelength_grid: np.ndarray,
    class_names: list,
    summary_df: pd.DataFrame,
    teacher_mean: float,
    teacher_std: float,
    random_mean: float,
    random_std: float,
    best_method: str,
    best_indices: np.ndarray,
    best_wl_nm: np.ndarray,
    all_selected_sets_raw: Dict[str, list],
    k: int,
    seed: int = 42,
    save_dir: str = "figures",
):
    """Generate all 5 publication-quality figures in one call.

    This is the main entry point called from the experiment runner.
    Creates the save_dir if it doesn't exist.

    Args:
        X, y, wavelength_grid, class_names: dataset
        summary_df: method comparison DataFrame from run_multi_seed_experiment
        teacher_mean, teacher_std: teacher baseline stats
        random_mean, random_std: random-k baseline stats
        best_method: name of the best method
        best_indices: selected wavelength indices for the best method
        best_wl_nm: selected wavelengths in nm
        all_selected_sets_raw: {method: [[fold_sets_seed0], ...]}
        k: number of selected wavelengths
        seed: seed for confusion matrix split
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  ── Generating Figures ──")

    # Figure 1: Mean spectra with best wavelengths highlighted
    wl_labels = [f'{w:.0f} nm\n{interpret_wavelength(w)}' for w in best_wl_nm]
    plot_mean_spectra(
        X, y, wavelength_grid, class_names,
        selected_wl_indices=best_indices,
        selected_wl_labels=wl_labels,
        title=f'Mean NIR Spectra — Best Wavelengths (k={k})',
        save_path=f'{save_dir}/fig1_mean_spectra_k{k}.png',
    )

    # Figure 2: Method comparison bar chart
    plot_method_comparison(
        summary_df, teacher_mean, teacher_std, random_mean, random_std, k,
        save_path=f'{save_dir}/fig2_method_comparison_k{k}.png',
    )

    # Figure 3: Consensus heatmaps (one per method)
    if all_selected_sets_raw:
        plot_consensus_heatmap(
            all_selected_sets_raw, wavelength_grid, k,
            save_dir=save_dir,
        )

    # Figure 4: Physical interpretation plot
    plot_physical_interpretation(
        X, y, wavelength_grid, class_names, best_indices, k,
        save_path=f'{save_dir}/fig4_physical_interpretation_k{k}.png',
    )

    # Figure 5: Confusion matrix for best method
    safe_name = best_method.replace(" ", "_")
    plot_confusion_matrix_best(
        X, y, class_names, wavelength_grid, best_method, best_indices, k,
        seed=seed,
        save_path=f'{save_dir}/fig5_confusion_matrix_{safe_name}_k{k}.png',
    )

    print(f"\n  All figures saved to: {save_dir}/")
