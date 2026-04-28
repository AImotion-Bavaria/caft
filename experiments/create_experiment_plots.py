"""
Generate comparative plots for experiment results.

Reads the experiment-level summary CSV and creates bar charts with error bars
comparing all methods across key metrics. Plots are saved in
    <experiment>/Evaluations/figures/

Usage:
    python create_experiment_plots.py
    python create_experiment_plots.py <path_to_experiment_dir>
"""
from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
DEFAULT_EXP_DIR = (
    Path(__file__).resolve().parent
    # / "artifacts" / "results" / "_random_blobs_balanced" / "Exp_1"
    / "artifacts" / "results" / "_steel_plates" / "Exp_1"
)

SPLITS = ["Train", "Validation", "Test"]

# Set to True to also generate plots per individual DATA_SPLIT_STATE
PLOT_SPLIT_STATES = False

# Method display order (same as create_experiment_summary.py)
_METHOD_ORDER = {
    "CE": 0,
    "CE_weighted": 1,
    "AEC": 2,
    "RWWCE1": 3, "RWWCE2": 4, "RWWCE3": 5,
    "constraint_aware": 6,
    "threshold_tuning": 7,
}


def _find_family(clean: str, family_dict: dict) -> tuple[str, str]:
    """Return (family_key, variant_suffix) for a cleaned method name.

    First tries exact match, then _CM<N> suffix removal, then longest-prefix
    match against the keys in *family_dict*.
    """
    # 1. Exact match
    if clean in family_dict:
        return clean, ""
    # 2. _CM<N> suffix
    m = re.search(r"_CM(\d+)$", clean)
    if m:
        base = clean[: m.start()]
        if base in family_dict:
            return base, m.group(0)
    # 3. Longest-prefix match (e.g. "constraint_aware3" → "constraint_aware")
    best = ""
    for key in family_dict:
        if clean.startswith(key) and len(key) > len(best):
            best = key
    if best:
        return best, clean[len(best):]
    return clean, ""


def _method_sort_key(name: str) -> tuple:
    clean = name.removeprefix("test_runs_")
    base, suffix = _find_family(clean, _METHOD_ORDER)
    m = re.search(r"_CM(\d+)$", clean)
    cm_num = int(m.group(1)) if m else 0
    family = _METHOD_ORDER.get(base, 99)
    return (family, cm_num, base, suffix)


def _short_label(name: str) -> str:
    """Strip 'test_runs_' prefix for plot labels."""
    return name.removeprefix("test_runs_")


# ─── Color palette ───────────────────────────────────────────────────────────
# Assign a color family per method base, with CM variants as lighter shades.
_FAMILY_COLORS = {
    "CE":               "#2c3e50",
    "CE_weighted":      "#7f8c8d",
    "AEC":              "#c0392b",
    "RWWCE1":           "#2980b9",
    "RWWCE2":           "#27ae60",
    "RWWCE3":           "#8e44ad",
    "constraint_aware": "#e67e22",
    "threshold_tuning": "#16a085",
}

_CM_LIGHTNESS = {0: 1.0, 1: 1.0, 2: 0.65, 3: 0.40}


def _lighten(hex_color: str, factor: float) -> str:
    """Lighten a hex color towards white by *factor* (0 = original, 1 = white)."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    mix = 1.0 - factor
    r2 = int(r * mix + 255 * factor)
    g2 = int(g * mix + 255 * factor)
    b2 = int(b * mix + 255 * factor)
    return f"#{r2:02x}{g2:02x}{b2:02x}"


def _method_color(name: str) -> str:
    clean = name.removeprefix("test_runs_")
    base, _ = _find_family(clean, _FAMILY_COLORS)
    m = re.search(r"_CM(\d+)$", clean)
    cm = int(m.group(1)) if m else 0
    base_color = _FAMILY_COLORS.get(base, "#555555")
    lighten_factor = _CM_LIGHTNESS.get(cm, 0.0)
    if cm == 1:
        return base_color
    return _lighten(base_color, 1.0 - lighten_factor)


# ─── Data loading ────────────────────────────────────────────────────────────

def load_experiment_data(exp_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the experiment CSV and return (mean_df, std_df) indexed by method.

    Returns a tuple (df_mean, df_std, methods, split_states) where df_mean has
    one row per method with Mean values, df_std the Std values, methods is the
    sorted list, and split_states is the list of DATA_SPLIT_STATE labels found.
    """
    csv_path = exp_dir / "Evaluations" / f"summary_{exp_dir.name}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run create_experiment_summary.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    methods = sorted(
        df[~df["DataSplit"].isin(["Mean", "Std"])]["Method"].unique(),
        key=_method_sort_key,
    )

    df_mean = df[df["DataSplit"] == "Mean"].set_index("Method").reindex(methods)
    df_std = df[df["DataSplit"] == "Std"].set_index("Method").reindex(methods)

    # Discover individual DATA_SPLIT_STATE labels (everything except Mean/Std)
    split_states = sorted(
        [s for s in df["DataSplit"].unique() if s not in ("Mean", "Std")]
    )

    return df_mean, df_std, methods, split_states


def load_split_state_data(
    exp_dir: Path, state: str, methods: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_vals, df_zeros) for a single DATA_SPLIT_STATE.

    df_vals contains the actual values for that state; df_zeros is a
    zero-filled DataFrame with the same shape (no std for a single state).
    """
    csv_path = exp_dir / "Evaluations" / f"summary_{exp_dir.name}.csv"
    df = pd.read_csv(csv_path)
    df_state = df[df["DataSplit"] == state].set_index("Method").reindex(methods)
    df_zeros = pd.DataFrame(0.0, index=df_state.index, columns=df_state.columns)
    return df_state, df_zeros


# ─── Plot helpers ────────────────────────────────────────────────────────────

def _fmt_bar_label(v: float) -> str:
    """Format a value for display on a bar."""
    if np.isnan(v):
        return ""
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}"


def _annotate_bars(ax: plt.Axes, bars, values: np.ndarray,
                   errors: np.ndarray | None = None,
                   horizontal: bool = False, fontsize: float = 7.5):
    """Place value labels above error bars (or bar top if no errors)."""
    for idx, (bar, v) in enumerate(zip(bars, values)):
        txt = _fmt_bar_label(v)
        if not txt:
            continue
        err = 0.0
        if errors is not None and not np.isnan(errors[idx]):
            err = errors[idx]
        if horizontal:
            x = bar.get_width() + err
            y = bar.get_y() + bar.get_height() / 2
            offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
            ax.text(x + offset, y, f" {txt}", va="center", ha="left",
                    fontsize=fontsize, color="#333333")
        else:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height() + err
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.text(x, y + offset, txt, va="bottom", ha="center",
                    fontsize=fontsize, color="#333333", rotation=90)

def _bar_chart(
    ax: plt.Axes,
    methods: list[str],
    values: np.ndarray,
    errors: np.ndarray | None,
    title: str,
    ylabel: str,
    colors: list[str],
    horizontal: bool = False,
):
    """Draw a single bar chart on *ax*."""
    x = np.arange(len(methods))
    labels = [_short_label(m) for m in methods]
    err_kw = dict(ecolor="#333333", capsize=3, linewidth=1)

    if horizontal:
        bars = ax.barh(x, values, xerr=errors, color=colors, edgecolor="white",
                       linewidth=0.5, error_kw=err_kw)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(ylabel, fontsize=11)
        ax.invert_yaxis()
        _annotate_bars(ax, bars, values, errors=errors, horizontal=True)
    else:
        bars = ax.bar(x, values, yerr=errors, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.7, error_kw=err_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        _annotate_bars(ax, bars, values, errors=errors, horizontal=False)
        # Reserve headroom for rotated bar annotations so they don't collide with titles
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.18)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─── Figure generators ───────────────────────────────────────────────────────

def plot_global_metrics(df_mean, df_std, methods, split, fig_dir):
    """Accuracy, Precision (macro), Recall (macro), F1-Score — one figure."""
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    titles = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"]
    colors = [_method_color(m) for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Global Metrics — {split}", fontsize=15, fontweight="bold", y=0.98)

    for ax, metric, title in zip(axes.flat, metrics, titles):
        col = f"{split}_{metric}"
        vals = df_mean[col].values.astype(float)
        errs = df_std[col].values.astype(float)
        errs = np.where(np.isnan(errs), 0, errs)
        _bar_chart(ax, methods, vals, errs, title, metric, colors)

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0)
    fig.savefig(fig_dir / f"global_metrics_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_metrics(df_mean, df_std, methods, split, fig_dir):
    """Average Cost and AEC side-by-side."""
    metrics = ["Average_Cost", "AEC"]
    titles = ["Average Cost", "AEC (Average Expected Cost)"]
    colors = [_method_color(m) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(f"Cost Metrics — {split}", fontsize=15, fontweight="bold", y=1.02)

    for ax, metric, title in zip(axes, metrics, titles):
        col = f"{split}_{metric}"
        vals = df_mean[col].values.astype(float)
        errs = df_std[col].values.astype(float)
        errs = np.where(np.isnan(errs), 0, errs)
        _bar_chart(ax, methods, vals, errs, title, metric, colors)

    fig.tight_layout()
    fig.savefig(fig_dir / f"cost_metrics_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_metrics(df_mean, df_std, methods, split, class_k, fig_dir):
    """Per-class detail: Precision, Recall, TP, TN, FN, FP, FN_rate, FP_rate."""
    metrics = [
        f"Precision_{class_k}",
        f"Recall_{class_k}",
        f"TP_{class_k}",
        f"TN_{class_k}",
        f"FN_{class_k}",
        f"FP_{class_k}",
        f"FN_{class_k}_rate",
        f"FP_{class_k}_rate",
        f"TP_{class_k}_rate",  # True Positive Rate
    ]
    titles = [
        f"Precision (class {class_k})",
        f"Recall (class {class_k})",
        f"TP (class {class_k})",
        f"TN (class {class_k})",
        f"FN count (class {class_k})",
        f"FP count (class {class_k})",
        f"FN rate (class {class_k})",
        f"FP rate (class {class_k})",
        f"True Positive Rate (class {class_k})",
    ]
    colors = [_method_color(m) for m in methods]

    fig, axes = plt.subplots(2, 5, figsize=(22, 11))
    fig.suptitle(f"Class {class_k} Metrics — {split}", fontsize=15,
                 fontweight="bold", y=0.98)

    for ax, metric, title in zip(axes.flat[:9], metrics, titles):
        col = f"{split}_{metric}"
        vals = df_mean[col].values.astype(float)
        errs = df_std[col].values.astype(float)
        errs = np.where(np.isnan(errs), 0, errs)
        _bar_chart(ax, methods, vals, errs, title, metric, colors)

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0)
    fig.savefig(fig_dir / f"class_{class_k}_metrics_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fp_cross_class(df_mean, df_std, methods, split, fig_dir):
    """Cross-class false positive counts: FP_i_j for all i≠j."""
    pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    colors = [_method_color(m) for m in methods]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(f"Cross-Class False Positives — {split}", fontsize=15,
                 fontweight="bold", y=0.98)

    for ax, (i, j) in zip(axes.flat, pairs):
        metric = f"FP_{i}_{j}"
        col = f"{split}_{metric}"
        vals = df_mean[col].values.astype(float)
        errs = df_std[col].values.astype(float)
        errs = np.where(np.isnan(errs), 0, errs)
        _bar_chart(ax, methods, vals, errs,
                   f"FP_{i}→{j}  (class {i} predicted as {j})",
                   "Count", colors)

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0)
    fig.savefig(fig_dir / f"fp_cross_class_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(df_mean, df_std, methods, split, fig_dir):
    """Heatmap: methods (rows) × key metrics (columns), normalised per column."""
    key_metrics = [
        "Accuracy", "Precision", "Recall", "F1_Score",
        "Average_Cost", "AEC",
        "Precision_2", "Recall_2", "FP_2_rate", "FN_2_rate",
    ]
    labels = [_short_label(m) for m in methods]
    cols = [f"{split}_{m}" for m in key_metrics]

    data = df_mean[cols].values.astype(float)
    data_std = df_std[cols].values.astype(float)

    # Normalize each column to [0, 1] for colouring
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normed = (data - col_min) / col_range

    # For cost/error metrics, invert so lower = better = darker green
    invert_cols = ["Average_Cost", "AEC", "FP_2_rate", "FN_2_rate"]
    for idx, m in enumerate(key_metrics):
        if m in invert_cols:
            normed[:, idx] = 1.0 - normed[:, idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(normed, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels(key_metrics, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells with mean (bold) ± std (smaller)
    for i in range(len(labels)):
        for j in range(len(key_metrics)):
            val = data[i, j]
            std = data_std[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="black")
            else:
                if abs(val) > 10:
                    mean_txt = f"{val:.1f}"
                    std_txt = f"±{std:.1f}"
                else:
                    mean_txt = f"{val:.3f}"
                    std_txt = f"±{std:.3f}"
                ax.text(j, i - 0.12, mean_txt, ha="center", va="center",
                        fontsize=8, color="black")
                ax.text(j, i + 0.18, std_txt, ha="center", va="center",
                        fontsize=6, color="#333333")

    ax.set_title(f"Method Comparison Heatmap — {split}\n"
                 "(green = better, red = worse; cost metrics inverted)",
                 fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.6, label="Normalised score (higher = better)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"summary_heatmap_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_class2_heatmap(df_mean, df_std, methods, split, fig_dir):
    """Heatmap: Accuracy, cost metrics, and all class-2 metrics."""
    key_metrics = [
        "Accuracy",
        "Average_Cost", "AEC",
        "Precision_2", "Recall_2",
        "TP_2", "TN_2", "FN_2", "FP_2",
        "FN_2_rate", "FP_2_rate",
        "FP_0_2", "FP_1_2",
    ]
    labels = [_short_label(m) for m in methods]
    cols = [f"{split}_{m}" for m in key_metrics]

    data = df_mean[cols].values.astype(float)
    data_std = df_std[cols].values.astype(float)

    # Normalize each column to [0, 1] for colouring
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normed = (data - col_min) / col_range

    # For cost/error metrics, invert so lower = better = darker green
    invert_cols = {
        "Average_Cost", "AEC",
        "FN_2", "FP_2", "FN_2_rate", "FP_2_rate",
        "FP_0_2", "FP_1_2",
    }
    for idx, m in enumerate(key_metrics):
        if m in invert_cols:
            normed[:, idx] = 1.0 - normed[:, idx]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(normed, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels(key_metrics, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells with mean (bold) ± std (smaller)
    for i in range(len(labels)):
        for j in range(len(key_metrics)):
            val = data[i, j]
            std = data_std[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="black")
            else:
                if abs(val) > 10:
                    mean_txt = f"{val:.1f}"
                    std_txt = f"±{std:.1f}"
                else:
                    mean_txt = f"{val:.3f}"
                    std_txt = f"±{std:.3f}"
                ax.text(j, i - 0.12, mean_txt, ha="center", va="center",
                        fontsize=8, color="black")
                ax.text(j, i + 0.18, std_txt, ha="center", va="center",
                        fontsize=6, color="#333333")

    ax.set_title(f"Cost & Class 2 Heatmap — {split}\n"
                 "(green = better, red = worse; cost/error metrics inverted)",
                 fontsize=11, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.6, label="Normalised score (higher = better)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"cost_class2_heatmap_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_cm_cell_keys(num_classes=3):
    """Return a num_classes×num_classes list-of-lists of summary-CSV metric keys
    that map to each confusion-matrix cell  (row=true, col=predicted).

    Diagonal: TP_k.  Off-diagonal (i,j): FP_i_j  (true i predicted as j).
    """
    keys = []
    for i in range(num_classes):
        row = []
        for j in range(num_classes):
            row.append(f"TP_{i}" if i == j else f"FP_{i}_{j}")
        keys.append(row)
    return keys


def plot_confusion_matrix_grid(df_mean, df_std, methods, split, fig_dir,
                               num_classes=3):
    """One confusion-matrix heatmap per method, arranged in a grid.

    Each cell shows  mean ± std  (counts, averaged over runs).
    Colour intensity reflects the mean count (log-scale friendly).
    """
    from matplotlib.colors import LogNorm, Normalize

    cm_keys = _build_cm_cell_keys(num_classes)
    n_methods = len(methods)
    ncols = min(n_methods, 4)
    nrows = -(-n_methods // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.2 * nrows + 0.8),
                             squeeze=False)
    fig.suptitle(f"Confusion Matrices (Mean ± Std) — {split}",
                 fontsize=15, fontweight="bold", y=1.0)

    class_labels = [f"Class {k}" for k in range(num_classes)]

    for idx, method in enumerate(methods):
        ax = axes.flat[idx]
        label = _short_label(method)

        # Build mean / std matrices
        mean_mat = np.zeros((num_classes, num_classes))
        std_mat  = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                col = f"{split}_{cm_keys[i][j]}"
                mean_mat[i, j] = df_mean.loc[method, col]
                std_mat[i, j]  = df_std.loc[method, col]

        # Colour by mean counts (use linear scale starting at 0)
        vmax = max(np.nanmax(mean_mat), 1)
        im = ax.imshow(mean_mat, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")

        # Annotate each cell with  mean ± std
        for i in range(num_classes):
            for j in range(num_classes):
                m = mean_mat[i, j]
                s = std_mat[i, j]
                if np.isnan(m):
                    txt = "N/A"
                elif m >= 10:
                    txt = f"{m:.1f}\n±{s:.1f}"
                else:
                    txt = f"{m:.2f}\n±{s:.2f}"
                # Choose text colour for readability
                text_color = "white" if m > 0.65 * vmax else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_methods, nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / f"confusion_matrices_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_grid_rates(df_mean, df_std, methods, split, fig_dir,
                                     num_classes=3):
    """Same grid as plot_confusion_matrix_grid but with row-normalised rates.

    Each cell shows  rate ± std_rate  where rate = count / row_sum.
    Diagonal cells are green-shaded (correct), off-diagonal red-shaded (errors).
    """
    cm_keys = _build_cm_cell_keys(num_classes)
    n_methods = len(methods)
    ncols = min(n_methods, 4)
    nrows = -(-n_methods // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.0 * nrows + 1.0),
                             squeeze=False)
    fig.suptitle(f"Confusion Matrices \u2014 Row-Normalised Rates (Mean \u00b1 Std) \u2014 {split}",
                 fontsize=17, fontweight="bold", y=1.0)

    class_labels = [f"Class {k}" for k in range(num_classes)]

    for idx, method in enumerate(methods):
        ax = axes.flat[idx]
        label = _short_label(method)

        # Build mean matrix
        mean_mat = np.zeros((num_classes, num_classes))
        std_mat  = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                col = f"{split}_{cm_keys[i][j]}"
                mean_mat[i, j] = df_mean.loc[method, col]
                std_mat[i, j]  = df_std.loc[method, col]

        # Row-normalise to get rates
        row_sums = mean_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        rate_mat = mean_mat / row_sums
        # Approximate std of rate via delta method: std_rate ≈ std_count / row_sum
        std_rate_mat = std_mat / row_sums

        # Custom colouring: green diagonal, red off-diagonal
        # Build RGB image manually
        rgb = np.zeros((num_classes, num_classes, 3))
        for i in range(num_classes):
            for j in range(num_classes):
                r = rate_mat[i, j]
                if i == j:
                    # Green channel scales with rate (correct classification)
                    rgb[i, j] = [1 - 0.7 * r, 1 - 0.15 * r, 1 - 0.7 * r]  # white → green
                else:
                    # Red channel scales with rate (misclassification)
                    rgb[i, j] = [1 - 0.15 * r, 1 - 0.7 * r, 1 - 0.7 * r]  # white → red

        ax.imshow(rgb, aspect="equal")

        # Annotate each cell with  rate ± std_rate
        for i in range(num_classes):
            for j in range(num_classes):
                r = rate_mat[i, j]
                sr = std_rate_mat[i, j]
                if np.isnan(r):
                    txt = "N/A"
                else:
                    txt = f"{r:.1%}\n±{sr:.1%}"
                text_color = "white" if r > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=13, fontweight="bold", color=text_color)

        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_labels, fontsize=11)
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(class_labels, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(label, fontsize=13, fontweight="bold")

    for idx in range(n_methods, nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / f"confusion_matrices_rates_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_cards_global(df_mean, df_std, methods, split, fig_dir):
    """Grid of metric-summary cards (one per method) — global + cost metrics.

    Same layout as confusion-matrix grid so the plots can be compared
    side by side.  Each card is a small heatmap (rows = metrics) with
    the actual Mean ± Std printed in each cell.
    """
    metric_keys = [
        "Accuracy", "Precision", "Recall", "F1_Score",
        "Average_Cost", "AEC",
    ]
    metric_labels = [
        "Accuracy", "Precision", "Recall", "F1-Score",
        "Avg Cost", "AEC",
    ]
    # For these metrics, lower is better (invert colour scale)
    lower_is_better = {"Average_Cost", "AEC"}

    n_metrics = len(metric_keys)
    n_methods = len(methods)
    ncols = min(n_methods, 4)
    nrows = -(-n_methods // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.0 * ncols, 0.7 * n_metrics * nrows + 2.0),
                             squeeze=False)
    fig.suptitle(f"Global Metric Cards (Mean ± Std) — {split}",
                 fontsize=17, fontweight="bold", y=1.0)

    # Collect all mean values per metric across methods for normalisation
    all_vals = {}
    for mk in metric_keys:
        col = f"{split}_{mk}"
        all_vals[mk] = df_mean[col].reindex(methods).values.astype(float)

    for idx, method in enumerate(methods):
        ax = axes.flat[idx]
        label = _short_label(method)

        means = np.array([df_mean.loc[method, f"{split}_{mk}"] for mk in metric_keys])
        stds  = np.array([df_std.loc[method, f"{split}_{mk}"] for mk in metric_keys])

        # Normalise each metric to [0,1] across methods for colouring
        normed = np.zeros(n_metrics)
        for j, mk in enumerate(metric_keys):
            vmin = np.nanmin(all_vals[mk])
            vmax = np.nanmax(all_vals[mk])
            rng = vmax - vmin if vmax != vmin else 1.0
            normed[j] = (means[j] - vmin) / rng
            if mk in lower_is_better:
                normed[j] = 1.0 - normed[j]

        # Build a 2-D array for imshow (n_metrics rows × 1 col)
        data_2d = normed.reshape(-1, 1)
        ax.imshow(data_2d, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        # Annotate
        for j in range(n_metrics):
            m = means[j]
            s = stds[j]
            if np.isnan(m):
                txt = "N/A"
            elif abs(m) >= 10:
                txt = f"{m:.2f} ± {s:.2f}"
            else:
                txt = f"{m:.3f} ± {s:.3f}"
            text_color = "white" if normed[j] > 0.7 or normed[j] < 0.3 else "black"
            ax.text(0, j, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels(metric_labels, fontsize=10)
        ax.set_xticks([])
        ax.set_title(label, fontsize=13, fontweight="bold")

    for idx in range(n_methods, nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / f"metric_cards_global_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_cards_class2(df_mean, df_std, methods, split, fig_dir):
    """Grid of metric-summary cards — class 2 detail metrics.

    Same layout as confusion-matrix grid.
    """
    metric_keys = [
        "Accuracy",
        "Precision_2", "Recall_2",
        "FN_2_rate", "FP_2_rate",
        "FP_0_2", "FP_1_2",
    ]
    metric_labels = [
        "Accuracy",
        "Precision", "Recall",
        "FN rate", "FP rate",
        "FP 0→2", "FP 1→2",
    ]
    # For these metrics, lower is better
    lower_is_better = {"FN_2_rate", "FP_2_rate", "FP_0_2", "FP_1_2"}

    n_metrics = len(metric_keys)
    n_methods = len(methods)
    ncols = min(n_methods, 4)
    nrows = -(-n_methods // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.0 * ncols, 0.7 * n_metrics * nrows + 2.0),
                             squeeze=False)
    fig.suptitle(f"Class 2 Metric Cards (Mean ± Std) — {split}",
                 fontsize=17, fontweight="bold", y=1.0)

    all_vals = {}
    for mk in metric_keys:
        col = f"{split}_{mk}"
        all_vals[mk] = df_mean[col].reindex(methods).values.astype(float)

    for idx, method in enumerate(methods):
        ax = axes.flat[idx]
        label = _short_label(method)

        means = np.array([df_mean.loc[method, f"{split}_{mk}"] for mk in metric_keys])
        stds  = np.array([df_std.loc[method, f"{split}_{mk}"] for mk in metric_keys])

        normed = np.zeros(n_metrics)
        for j, mk in enumerate(metric_keys):
            vmin = np.nanmin(all_vals[mk])
            vmax = np.nanmax(all_vals[mk])
            rng = vmax - vmin if vmax != vmin else 1.0
            normed[j] = (means[j] - vmin) / rng
            if mk in lower_is_better:
                normed[j] = 1.0 - normed[j]

        data_2d = normed.reshape(-1, 1)
        ax.imshow(data_2d, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        for j in range(n_metrics):
            m = means[j]
            s = stds[j]
            if np.isnan(m):
                txt = "N/A"
            elif abs(m) >= 10:
                txt = f"{m:.2f} ± {s:.2f}"
            else:
                txt = f"{m:.3f} ± {s:.3f}"
            text_color = "white" if normed[j] > 0.7 or normed[j] < 0.3 else "black"
            ax.text(0, j, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels(metric_labels, fontsize=10)
        ax.set_xticks([])
        ax.set_title(label, fontsize=13, fontweight="bold")

    for idx in range(n_methods, nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / f"metric_cards_class2_{split}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_comparison(df_mean, df_std, methods, fig_dir):
    """For each key metric, show Train / Val / Test side-by-side (grouped bars).

    Uses the same per-method colour scheme as all other plots, with
    hatching patterns to distinguish the three splits.
    """
    key_metrics = [
        ("Accuracy", "Accuracy"),
        ("F1_Score", "F1-Score (macro)"),
        ("AEC", "AEC"),
        ("Average_Cost", "Average Cost"),
        ("Recall_2", "Recall (class 2)"),
        ("FP_2_rate", "FP rate (class 2)"),
    ]
    split_hatches = {"Train": "", "Validation": "//", "Test": "xx"}
    split_alpha   = {"Train": 1.0, "Validation": 0.70, "Test": 0.45}
    method_colors = [_method_color(m) for m in methods]
    labels = [_short_label(m) for m in methods]
    x = np.arange(len(methods))
    width = 0.25

    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    fig.suptitle("Train / Validation / Test Comparison", fontsize=15,
                 fontweight="bold", y=0.99)

    for ax, (metric, title) in zip(axes.flat, key_metrics):
        for k, split in enumerate(SPLITS):
            col = f"{split}_{metric}"
            vals = df_mean[col].values.astype(float)
            errs = df_std[col].values.astype(float)
            errs = np.where(np.isnan(errs), 0, errs)
            bars = ax.bar(
                x + (k - 1) * width, vals, width, yerr=errs,
                color=method_colors,
                alpha=split_alpha[split],
                hatch=split_hatches[split],
                edgecolor="white", linewidth=0.5,
                error_kw=dict(ecolor="#333", capsize=2, linewidth=0.8),
            )
            _annotate_bars(ax, bars, vals, errors=errs, fontsize=6.5)

        # Build a legend for the three splits using dummy patches
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="#888888", alpha=split_alpha[s],
                  hatch=split_hatches[s], edgecolor="white", label=s)
            for s in SPLITS
        ]
        ax.legend(handles=legend_handles, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)
    fig.savefig(fig_dir / "split_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Split-comparison style helpers ──────────────────────────────────────────

def _split_comparison_grid(
    df_mean, df_std, methods, metrics_list, suptitle, filename,
    fig_dir, nrows=None, ncols=None, figsize=None,
):
    """Generic split-comparison plot: per-method colours, hatching per split.

    Parameters
    ----------
    metrics_list : list of (metric_key, display_title)
    """
    from matplotlib.patches import Patch

    n = len(metrics_list)
    if nrows is None or ncols is None:
        ncols = min(n, 2)
        nrows = -(-n // ncols)  # ceil division
    if figsize is None:
        figsize = (7 * ncols, 5.5 * nrows + 1)

    split_hatches = {"Train": "", "Validation": "//", "Test": "xx"}
    split_alpha   = {"Train": 1.0, "Validation": 0.70, "Test": 0.45}
    method_colors = [_method_color(m) for m in methods]
    labels = [_short_label(m) for m in methods]
    x = np.arange(len(methods))
    width = 0.25

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.99)

    for idx, (metric, title) in enumerate(metrics_list):
        ax = axes.flat[idx]
        for k, split in enumerate(SPLITS):
            col = f"{split}_{metric}"
            vals = df_mean[col].values.astype(float)
            errs = df_std[col].values.astype(float)
            errs = np.where(np.isnan(errs), 0, errs)
            bars = ax.bar(
                x + (k - 1) * width, vals, width, yerr=errs,
                color=method_colors,
                alpha=split_alpha[split],
                hatch=split_hatches[split],
                edgecolor="white", linewidth=0.5,
                error_kw=dict(ecolor="#333", capsize=2, linewidth=0.8),
            )
            _annotate_bars(ax, bars, vals, errors=errs, fontsize=6.5)

        legend_handles = [
            Patch(facecolor="#888888", alpha=split_alpha[s],
                  hatch=split_hatches[s], edgecolor="white", label=s)
            for s in SPLITS
        ]
        ax.legend(handles=legend_handles, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0)
    fig.savefig(fig_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_accuracy(df_mean, df_std, methods, fig_dir):
    """Single-plot split comparison for Accuracy."""
    _split_comparison_grid(
        df_mean, df_std, methods,
        [("Accuracy", "Accuracy")],
        suptitle="Accuracy — Train / Validation / Test",
        filename="split_accuracy.png",
        fig_dir=fig_dir,
        nrows=1, ncols=1, figsize=(12, 5.5),
    )


def plot_split_cost(df_mean, df_std, methods, fig_dir):
    """Split comparison for cost metrics (Average Cost + AEC)."""
    _split_comparison_grid(
        df_mean, df_std, methods,
        [("Average_Cost", "Average Cost"), ("AEC", "AEC")],
        suptitle="Cost Metrics — Train / Validation / Test",
        filename="split_cost_metrics.png",
        fig_dir=fig_dir,
        nrows=1, ncols=2, figsize=(15, 5.5),
    )


def plot_split_class2_precision_recall(df_mean, df_std, methods, fig_dir):
    """Split comparison: class-2 Precision & Recall."""
    _split_comparison_grid(
        df_mean, df_std, methods,
        [("Precision_2", "Precision (class 2)"),
         ("Recall_2", "Recall (class 2)")],
        suptitle="Class 2 — Precision & Recall",
        filename="split_class2_precision_recall.png",
        fig_dir=fig_dir,
        nrows=1, ncols=2, figsize=(15, 5.5),
    )


def plot_split_class2_tp_tn(df_mean, df_std, methods, fig_dir):
    """Split comparison: class-2 TP & TN."""
    _split_comparison_grid(
        df_mean, df_std, methods,
        [("TP_2", "TP (class 2)"),
         ("TN_2", "TN (class 2)")],
        suptitle="Class 2 — True Positives & True Negatives",
        filename="split_class2_tp_tn.png",
        fig_dir=fig_dir,
        nrows=1, ncols=2, figsize=(15, 5.5),
    )


def plot_split_class2_fn_fp(df_mean, df_std, methods, fig_dir):
    """Split comparison: class-2 FN, FP (counts) and FN_rate, FP_rate."""
    _split_comparison_grid(
        df_mean, df_std, methods,
        [("FN_2", "FN count (class 2)"),
         ("FP_2", "FP count (class 2)"),
         ("FN_2_rate", "FN rate (class 2)"),
         ("FP_2_rate", "FP rate (class 2)")],
        suptitle="Class 2 — False Negatives & False Positives",
        filename="split_class2_fn_fp.png",
        fig_dir=fig_dir,
        nrows=2, ncols=2, figsize=(15, 10),
    )


def plot_split_class2_cross_fp(df_mean, df_std, methods, fig_dir):
    """Split comparison: cross-class FP into class 2 (FP_0_2 + FP_1_2)."""
    _split_comparison_grid(
        df_mean, df_std, methods,
        [("FP_0_2", "FP 0→2 (class 0 predicted as 2)"),
         ("FP_1_2", "FP 1→2 (class 1 predicted as 2)")],
        suptitle="Class 2 — Cross-Class False Positives",
        filename="split_class2_cross_fp.png",
        fig_dir=fig_dir,
        nrows=1, ncols=2, figsize=(15, 5.5),
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXP_DIR
    exp_dir = exp_dir.resolve()

    fig_dir = exp_dir / "Evaluations" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Experiment: {exp_dir}")
    print(f"Output:     {fig_dir}")
    print("=" * 60)

    df_mean, df_std, methods, split_states = load_experiment_data(exp_dir)

    # ── Per-split figures (aggregated Mean ± Std across all split states) ──
    for split in SPLITS:
        print(f"\n--- {split} ---")

        print(f"  Global metrics ...")
        plot_global_metrics(df_mean, df_std, methods, split, fig_dir)

        print(f"  Cost metrics ...")
        plot_cost_metrics(df_mean, df_std, methods, split, fig_dir)

        for k in (0, 1, 2):
            print(f"  Class {k} metrics ...")
            plot_class_metrics(df_mean, df_std, methods, split, k, fig_dir)

        print(f"  Cross-class FP ...")
        plot_fp_cross_class(df_mean, df_std, methods, split, fig_dir)

        print(f"  Summary heatmap ...")
        plot_summary_heatmap(df_mean, df_std, methods, split, fig_dir)

        print(f"  Cost & Class 2 heatmap ...")
        plot_cost_class2_heatmap(df_mean, df_std, methods, split, fig_dir)

        print(f"  Confusion matrices (counts) ...")
        plot_confusion_matrix_grid(df_mean, df_std, methods, split, fig_dir)

        print(f"  Confusion matrices (rates) ...")
        plot_confusion_matrix_grid_rates(df_mean, df_std, methods, split, fig_dir)

        print(f"  Metric cards (global) ...")
        plot_metric_cards_global(df_mean, df_std, methods, split, fig_dir)

        print(f"  Metric cards (class 2) ...")
        plot_metric_cards_class2(df_mean, df_std, methods, split, fig_dir)

    # ── Cross-split comparisons ──
    print(f"\n--- Split Comparisons ---")

    print(f"  Overview ...")
    plot_split_comparison(df_mean, df_std, methods, fig_dir)

    print(f"  Accuracy ...")
    plot_split_accuracy(df_mean, df_std, methods, fig_dir)

    print(f"  Cost metrics ...")
    plot_split_cost(df_mean, df_std, methods, fig_dir)

    print(f"  Class 2: Precision & Recall ...")
    plot_split_class2_precision_recall(df_mean, df_std, methods, fig_dir)

    print(f"  Class 2: TP & TN ...")
    plot_split_class2_tp_tn(df_mean, df_std, methods, fig_dir)

    print(f"  Class 2: FN & FP (count + rate) ...")
    plot_split_class2_fn_fp(df_mean, df_std, methods, fig_dir)

    print(f"  Class 2: Cross-class FP ...")
    plot_split_class2_cross_fp(df_mean, df_std, methods, fig_dir)

    # ── Per DATA_SPLIT_STATE figures ──
    if not PLOT_SPLIT_STATES:
        print("\nSkipping per-split-state plots (PLOT_SPLIT_STATES = False).")
    for state in (split_states if PLOT_SPLIT_STATES else []):
        print(f"\n{'='*60}")
        print(f"  Generating plots for {state} ...")
        state_fig_dir = fig_dir / state
        state_fig_dir.mkdir(parents=True, exist_ok=True)

        df_state, df_state_zeros = load_split_state_data(exp_dir, state, methods)

        for split in SPLITS:
            print(f"    {state} / {split} ...")

            plot_global_metrics(df_state, df_state_zeros, methods, split, state_fig_dir)
            plot_cost_metrics(df_state, df_state_zeros, methods, split, state_fig_dir)

            for k in (0, 1, 2):
                plot_class_metrics(df_state, df_state_zeros, methods, split, k, state_fig_dir)

            plot_fp_cross_class(df_state, df_state_zeros, methods, split, state_fig_dir)
            plot_summary_heatmap(df_state, df_state_zeros, methods, split, state_fig_dir)
            plot_cost_class2_heatmap(df_state, df_state_zeros, methods, split, state_fig_dir)
            plot_confusion_matrix_grid(df_state, df_state_zeros, methods, split, state_fig_dir)
            plot_confusion_matrix_grid_rates(df_state, df_state_zeros, methods, split, state_fig_dir)
            plot_metric_cards_global(df_state, df_state_zeros, methods, split, state_fig_dir)
            plot_metric_cards_class2(df_state, df_state_zeros, methods, split, state_fig_dir)

    n_files = len(list(fig_dir.rglob("*.png")))
    print(f"\n{'='*60}")
    print(f"Done! {n_files} figures saved to {fig_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
