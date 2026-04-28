"""
Summarize experiment results across runs, data splits, and experiments.

Creates three levels of summary (each as .txt, .csv, .xlsx):
  Level 1 – Per test_runs folder   → Evaluations/ with runs 1-5, mean, std
  Level 2 – Per DATA_SPLIT_STATE   → Evaluations/ across all methods
  Level 3 – Per Experiment (Exp_1) → Evaluations/ across all data splits per method

Usage:
    python create_experiment_summary.py
    python create_experiment_summary.py <path_to_experiment_dir>
"""
from __future__ import annotations

import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
DEFAULT_EXP_DIR = (
    Path(__file__).resolve().parent
    # / "artifacts" / "results" / "_random_blobs_balanced" / "Exp_1"
    / "artifacts" / "results" / "_steel_plates" / "Exp_3"
)

METRICS = [
    # ── Global ──
    "Accuracy",
    "Precision",       # macro avg
    "Recall",          # macro avg
    "F1_Score",        # macro avg
    "Average_Cost",
    "AEC",
    # ── Class 2 ──
    "Precision_2", "Recall_2", "TP_2", "TN_2",
    "FN_2", "FP_2", "FN_2_rate", "FP_2_rate", "TP_2_rate",
    "FP_0_2", "FP_1_2",
    # ── Class 1 ──
    "Precision_1", "Recall_1", "TP_1", "TN_1",
    "FN_1", "FP_1", "FN_1_rate", "FP_1_rate", "TP_1_rate",
    "FP_0_1", "FP_2_1",
    # ── Class 0 ──
    "Precision_0", "Recall_0", "TP_0", "TN_0",
    "FN_0", "FP_0", "FN_0_rate", "FP_0_rate", "TP_0_rate",
    "FP_1_0", "FP_2_0",
]

SPLITS = ["Train", "Validation", "Test"]


# ═════════════════════════════════════════════════════════════════════════════
#  Parsing
# ═════════════════════════════════════════════════════════════════════════════

def parse_result_file(filepath: Path) -> dict:
    """Parse a single result .txt file.

    If *Phase 2* exists the metrics are taken from that section.
    If *(eval cost)* blocks exist, Average_Cost and AEC come from there.

    Returns
    -------
    dict  {split: {metric: value}}
    """
    content = filepath.read_text(encoding="utf-8")

    # Use Phase 2 results when present (constraint-aware or threshold tuning)
    phase2_match = re.search(r"--- Phase 2 (Final Evaluation|Evaluation \(tau=)", content)
    if phase2_match:
        content = content[phase2_match.start():]

    results = {}
    for split in SPLITS:
        results[split] = _extract_split_metrics(content, split)
    return results


def _extract_split_metrics(content: str, split: str) -> dict:
    """Extract all requested metrics for one data split."""
    metrics: dict = {}

    std_header = f"---------------{split} Metrics Summary---------------"
    eval_header = f"---------------{split} (eval cost) Metrics Summary---------------"

    has_eval_cost = eval_header in content
    std_section = _extract_section(content, std_header)

    # Threshold tuning Phase 2 only has (eval cost) headers — use them for everything
    if std_section is None and has_eval_cost:
        std_section = _extract_section(content, eval_header)

    if std_section is None:
        return {m: np.nan for m in METRICS}

    # ── Average Cost & AEC: prefer (eval cost) section ──
    if has_eval_cost:
        ec_section = _extract_section(content, eval_header)
        metrics["Average_Cost"] = _float(ec_section, r"Average cost:\s*([\d.]+)")
        metrics["AEC"]          = _float(ec_section, r"AEC:\s*([\d.]+)")
    else:
        metrics["Average_Cost"] = _float(std_section, r"Average cost:\s*([\d.]+)")
        metrics["AEC"]          = _float(std_section, r"AEC:\s*([\d.]+)")

    # ── Accuracy ──
    metrics["Accuracy"] = _float(std_section, r"Accuracy:\s*([\d.]+)")

    # ── Macro-avg Precision / Recall / F1 ──
    m = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", std_section)
    if m:
        metrics["Precision"] = float(m.group(1))
        metrics["Recall"]    = float(m.group(2))
        metrics["F1_Score"]  = float(m.group(3))
    else:
        metrics["Precision"] = metrics["Recall"] = metrics["F1_Score"] = np.nan

    # ── Per-class Precision & Recall from classification report ──
    for k in (0, 1, 2):
        ck = re.search(rf"^\s*{k}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                        std_section, re.MULTILINE)
        if ck:
            metrics[f"Precision_{k}"] = float(ck.group(1))
            metrics[f"Recall_{k}"]    = float(ck.group(2))
        else:
            metrics[f"Precision_{k}"] = metrics[f"Recall_{k}"] = np.nan

    # ── Confusion-matrix derived metrics (all classes) ──
    cm = _extract_cm(std_section)
    if cm is not None:
        for k in (0, 1, 2):
            tp_k = cm[k][k]
            fn_k = sum(cm[k][j] for j in range(3) if j != k)   # count
            fp_k = sum(cm[i][k] for i in range(3) if i != k)   # count
            row_sum = sum(cm[k])          # TP_k + FN_k (actual class k)
            col_sum = sum(cm[i][k] for i in range(3))  # TP_k + FP_k (pred k)

            metrics[f"TP_{k}"] = float(tp_k)
            # TN_k: instances not class k that are correctly not predicted as k
            tn_k = sum(cm[i][j] for i in range(3) for j in range(3)
                       if i != k and j != k)
            metrics[f"TN_{k}"] = float(tn_k)
            metrics[f"FN_{k}"] = float(fn_k)
            metrics[f"FP_{k}"] = float(fp_k)
            # FN_k_rate: fraction of actual-k missed = FN_k / (TP_k + FN_k)
            metrics[f"FN_{k}_rate"] = fn_k / row_sum if row_sum > 0 else np.nan
            # FP_k_rate: classical False Positive Rate = FP_k / (FP_k + TN_k)
            fp_tn_sum = fp_k + tn_k
            metrics[f"FP_{k}_rate"] = fp_k / fp_tn_sum if fp_tn_sum > 0 else np.nan
            # True Positive Rate: TP_{k}_rate = TP_k / (TP_k + FN_k)
            metrics[f"TP_{k}_rate"] = tp_k / row_sum if row_sum > 0 else np.nan
            # Override report-based Recall/Precision with exact CM values
            metrics[f"Recall_{k}"]    = tp_k / row_sum if row_sum > 0 else np.nan
            metrics[f"Precision_{k}"] = tp_k / col_sum if col_sum > 0 else np.nan
            # Cross-class FP entries: FP_i_k for each i != k
            for i in range(3):
                if i != k:
                    metrics[f"FP_{i}_{k}"] = float(cm[i][k])
    else:
        for k in (0, 1, 2):
            metrics[f"TP_{k}"] = np.nan
            metrics[f"TN_{k}"] = np.nan
            metrics[f"FN_{k}"] = np.nan
            metrics[f"FP_{k}"] = np.nan
            metrics[f"FN_{k}_rate"] = np.nan
            metrics[f"FP_{k}_rate"] = np.nan
            for i in range(3):
                if i != k:
                    metrics[f"FP_{i}_{k}"] = np.nan

    return metrics


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_section(content: str, header: str) -> str | None:
    """Return text between *header* and the next section separator."""
    idx = content.find(header)
    if idx == -1:
        return None
    start = idx + len(header)
    nxt = re.search(r"\n-{10,}", content[start:])
    return content[start : start + nxt.start()] if nxt else content[start:]


def _float(text: str | None, pattern: str) -> float:
    if text is None:
        return np.nan
    m = re.search(pattern, text)
    return float(m.group(1)) if m else np.nan


def _extract_cm(text: str | None):
    """Parse the first 3×3 integer confusion matrix."""
    if text is None:
        return None
    rows = re.findall(r"\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]", text)
    if len(rows) >= 3:
        return [[int(v) for v in rows[i]] for i in range(3)]
    return None


_METHOD_ORDER = {
    "CE": 0,
    "CE_weighted": 1,
    "AEC": 2,
    "RWWCE1": 3, "RWWCE2": 4, "RWWCE3": 5,
    "constraint_aware": 6,
}


def _method_sort_key(name: str) -> tuple:
    """Sort key: CE, CE_weighted, AEC*, RWWCE*, constraint_aware*.

    Within each family, CM variants are ordered by number (CM1, CM2, CM3).
    The *name* is the test_runs folder name **without** the ``test_runs_`` prefix.
    """
    clean = name.removeprefix("test_runs_")
    m = re.search(r"_CM(\d+)$", clean)
    if m:
        base = clean[:m.start()]
        cm_num = int(m.group(1))
    else:
        base = clean
        cm_num = 0
    family = _METHOD_ORDER.get(base, 99)
    return (family, cm_num, base)


# ═════════════════════════════════════════════════════════════════════════════
#  DataFrame construction
# ═════════════════════════════════════════════════════════════════════════════

def build_summary_df(all_results: list[dict], row_labels: list[str]) -> pd.DataFrame:
    """One row per result plus Mean / Std rows at the bottom."""
    records = []
    for label, res in zip(row_labels, all_results):
        row = {"Run": label}
        for s in SPLITS:
            for m in METRICS:
                row[f"{s}_{m}"] = res[s].get(m, np.nan)
        records.append(row)

    df = pd.DataFrame(records)

    num_cols = [c for c in df.columns if c != "Run"]
    mean_r = {"Run": "Mean"}
    std_r  = {"Run": "Std"}
    for col in num_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        mean_r[col] = vals.mean()
        std_r[col]  = vals.std()   # ddof=1 (pandas default)

    df = pd.concat([df, pd.DataFrame([mean_r, std_r])], ignore_index=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Output helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_val(v) -> str:
    """Format a single value for text output."""
    if pd.isna(v):
        return f"{'N/A':>15}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):>15d}"
    if float(v) == int(v) and abs(v) < 1e6:
        return f"{int(v):>15d}"
    return f"{v:>15.4f}"


def _format_txt(df: pd.DataFrame, title: str, label_col: str = "Run",
                label_width: int = 30) -> str:
    """Render the DataFrame as a readable text block (one sub-table per split)."""
    lines = [title, "=" * max(len(title), 60), ""]

    for split in SPLITS:
        lines.append(f"--- {split} ---")
        cols = [c for c in df.columns if c.startswith(f"{split}_")]
        short = [c.replace(f"{split}_", "") for c in cols]

        hdr = f"{'':>{label_width}}" + "".join(f"{n:>15}" for n in short)
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for _, row in df.iterrows():
            lbl = str(row[label_col])[:label_width]
            vals = "".join(_fmt_val(row[c]) for c in cols)
            lines.append(f"{lbl:>{label_width}}{vals}")

        lines.append("")

    return "\n".join(lines)


def save_summary(df: pd.DataFrame, out_dir: Path, name: str,
                 label_col: str = "Run", label_width: int = 30):
    """Write .csv, .xlsx, .txt into *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.4f")

    # Excel
    try:
        df.to_excel(out_dir / f"{name}.xlsx", index=False)
    except ModuleNotFoundError:
        print("    (openpyxl not installed – .xlsx skipped)")

    # TXT
    txt = _format_txt(df, name, label_col=label_col, label_width=label_width)
    (out_dir / f"{name}.txt").write_text(txt, encoding="utf-8")

    print(f"  -> {out_dir.relative_to(out_dir.parents[3])}/{name}  (.csv, .xlsx, .txt)")


# ═════════════════════════════════════════════════════════════════════════════
#  File discovery
# ═════════════════════════════════════════════════════════════════════════════

def _run_number(path: Path) -> int:
    m = re.search(r"(\d+)\.txt$", path.name)
    return int(m.group(1)) if m else 0


def find_result_files(folder: Path) -> list[Path]:
    """Return *_results*.txt files sorted by trailing run number."""
    return sorted(folder.glob("*_results*.txt"), key=_run_number)


# ═════════════════════════════════════════════════════════════════════════════
#  Level 1 – per test_runs folder
# ═════════════════════════════════════════════════════════════════════════════

def summarize_test_runs(test_runs_dir: Path) -> pd.DataFrame | None:
    """Summarise individual runs (1–5) inside one test_runs folder."""
    files = find_result_files(test_runs_dir)
    if not files:
        print(f"  WARN: no result files in {test_runs_dir.name}")
        return None

    results, labels = [], []
    for f in files:
        labels.append(f"Run_{_run_number(f)}")
        results.append(parse_result_file(f))

    df = build_summary_df(results, labels)
    save_summary(df, test_runs_dir / "Evaluations",
                 f"summary_{test_runs_dir.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  Level 2 – per DATA_SPLIT_STATE
# ═════════════════════════════════════════════════════════════════════════════

def summarize_data_split(state_dir: Path) -> pd.DataFrame | None:
    """Aggregate all methods inside one DATA_SPLIT_STATE.

    Each method is represented by its mean across 5 runs.
    """
    print(f"\n{'─'*60}")
    print(f"DATA_SPLIT_STATE: {state_dir.name}")
    print(f"{'─'*60}")

    tr_dirs = sorted(
        (d for d in state_dir.iterdir()
         if d.is_dir() and d.name.startswith("test_runs")),
        key=lambda d: _method_sort_key(d.name)
    )

    means, labels = [], []
    std_data: dict[str, dict] = {}   # label -> {split: {metric: std_val}}
    for tr_dir in tr_dirs:
        df = summarize_test_runs(tr_dir)
        if df is None:
            continue
        mean_row = df[df["Run"] == "Mean"].iloc[0]
        std_row  = df[df["Run"] == "Std"].iloc[0]
        lbl = tr_dir.name
        labels.append(lbl)
        result = {
            s: {m: mean_row[f"{s}_{m}"] for m in METRICS}
            for s in SPLITS
        }
        means.append(result)
        std_data[lbl] = {
            s: {m: std_row[f"{s}_{m}"] for m in METRICS}
            for s in SPLITS
        }

    if not means:
        return None

    df_state = build_summary_df(means, labels)
    save_summary(df_state, state_dir / "Evaluations",
                 f"summary_{state_dir.name}", label_width=35)

    # ── NEW: wide mean±std .txt ──────────────────────────────────────────
    _write_data_split_mean_std_txt(state_dir, df_state, std_data, labels)

    return df_state


# ─────────────────────────────────────────────────────────────────────────────
#  Level 2 helper – wide mean±std .txt
# ─────────────────────────────────────────────────────────────────────────────

def _write_data_split_mean_std_txt(
    state_dir: Path,
    df_state: pd.DataFrame,
    std_data: dict[str, dict],
    labels: list[str],
):
    """Write a new .txt with paired Metric_mean / Metric_std columns."""
    out_dir = state_dir / "Evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    LW = 35

    # Build column names: Metric_mean, Metric_std alternating
    wide_cols: list[str] = []
    for m in METRICS:
        wide_cols.append(f"{m}_mean")
        wide_cols.append(f"{m}_std")

    lines: list[str] = [
        f"Mean ± Std Summary: {state_dir.name}",
        "=" * 80, ""
    ]

    for split in SPLITS:
        lines.append(f"--- {split} ---")
        hdr = f"{'':>{LW}}" + "".join(f"{c:>15}" for c in wide_cols)
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for lbl in labels:
            row_vals = ""
            mean_row = df_state[df_state["Run"] == lbl]
            if mean_row.empty:
                continue
            mean_row = mean_row.iloc[0]
            std_dict = std_data.get(lbl, {}).get(split, {})
            for m in METRICS:
                mv = mean_row[f"{split}_{m}"]
                sv = std_dict.get(m, np.nan)
                row_vals += _fmt_val(mv) + _fmt_val(sv)
            lines.append(f"{lbl:>{LW}}{row_vals}")

        lines.append("")

    fname = f"summary_{state_dir.name}_mean_std"
    (out_dir / f"{fname}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {out_dir.relative_to(out_dir.parents[3])}/{fname}.txt")


# ═════════════════════════════════════════════════════════════════════════════
#  Level 3 – per Experiment
# ═════════════════════════════════════════════════════════════════════════════

def summarize_experiment(exp_dir: Path):
    """For each method, show per-state values and mean/std across states."""
    # Preferred order: 42, 5, 0
    STATE_ORDER = ["DATA_SPLIT_STATE_42", "DATA_SPLIT_STATE_5", "DATA_SPLIT_STATE_0"]

    state_dirs_all = sorted(
        d for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("DATA_SPLIT_STATE")
    )

    # Collect per-method, per-state mean values
    # method_data[method][state_label] = {split: {metric: val}}
    method_data: dict[str, dict[str, dict]] = {}
    # method_std[method][state_label] = {split: {metric: std_val}}
    method_std: dict[str, dict[str, dict]] = {}
    # method_runs[method] = list of {col: val} dicts from ALL individual runs
    method_runs: dict[str, list[dict]] = {}
    state_labels: list[str] = []

    for state_dir in state_dirs_all:
        state_label = state_dir.name
        state_labels.append(state_label)
        df_state = summarize_data_split(state_dir)
        if df_state is None:
            continue
        for _, row in df_state.iterrows():
            method = row["Run"]
            if method == "Mean":
                continue
            if method == "Std":
                # map Std row back to each method (skip – handled below)
                continue
            method_data.setdefault(method, {})[state_label] = {
                s: {m: row[f"{s}_{m}"] for m in METRICS}
                for s in SPLITS
            }
        # Std row = std across methods (not what we need).
        # Instead, read per-method std from Level 1 results.
        for tr_dir in sorted(
            (d for d in state_dir.iterdir()
             if d.is_dir() and d.name.startswith("test_runs")),
            key=lambda d: _method_sort_key(d.name)
        ):
            l1_csv = tr_dir / "Evaluations" / f"summary_{tr_dir.name}.csv"
            if not l1_csv.exists():
                continue
            df_l1 = pd.read_csv(l1_csv)
            std_row = df_l1[df_l1["Run"] == "Std"]
            if std_row.empty:
                continue
            std_row = std_row.iloc[0]
            method_std.setdefault(tr_dir.name, {})[state_label] = {
                s: {m: std_row.get(f"{s}_{m}", np.nan) for m in METRICS}
                for s in SPLITS
            }
            # Collect individual runs for overall Mean/Std
            run_rows = df_l1[df_l1["Run"].str.startswith("Run_")]
            for _, rr in run_rows.iterrows():
                run_dict = {}
                for s in SPLITS:
                    for m in METRICS:
                        col = f"{s}_{m}"
                        run_dict[col] = rr.get(col, np.nan)
                method_runs.setdefault(tr_dir.name, []).append(run_dict)

    if not method_data:
        print("  WARN: no data at experiment level")
        return

    # Re-order state labels to preferred order
    ordered_states = [s for s in STATE_ORDER if s in state_labels]
    # append any unexpected states at the end
    for s in state_labels:
        if s not in ordered_states:
            ordered_states.append(s)

    # ── Build the experiment-level DataFrame (method-by-method) ──
    records = []
    for method in sorted(method_data, key=_method_sort_key):
        for sl in ordered_states:
            if sl not in method_data[method]:
                continue
            r = method_data[method][sl]
            row = {"Method": method, "DataSplit": sl}
            for s in SPLITS:
                for m in METRICS:
                    row[f"{s}_{m}"] = r[s].get(m, np.nan)
            records.append(row)

        # Mean / Std rows across ALL individual runs (not split-means)
        mean_row = {"Method": method, "DataSplit": "Mean"}
        std_row  = {"Method": method, "DataSplit": "Std"}
        if method in method_runs:
            all_cols = [f"{s}_{m}" for s in SPLITS for m in METRICS]
            for col in all_cols:
                vals = [r.get(col, np.nan) for r in method_runs[method]]
                arr = pd.Series(vals).dropna()
                mean_row[col] = arr.mean() if len(arr) > 0 else np.nan
                std_row[col]  = arr.std()  if len(arr) > 0 else np.nan
        records.append(mean_row)
        records.append(std_row)

    df_exp = pd.DataFrame(records)

    # Column order
    metric_cols = [f"{s}_{m}" for s in SPLITS for m in METRICS]
    ordered = ["Method", "DataSplit"] + [c for c in metric_cols if c in df_exp.columns]
    df_exp = df_exp[ordered]

    # ── Save CSV (method-by-method, as before) ──
    out_dir = exp_dir / "Evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"summary_{exp_dir.name}"

    df_exp.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.4f")

    # ── Save method-by-method txt (as before) ──
    lines_method = [
        f"Experiment Summary: {exp_dir.name}",
        "=" * 60, ""
    ]
    for method in sorted(method_data, key=_method_sort_key):
        mdf = df_exp[df_exp["Method"] == method]
        lines_method.append(f"Method: {method}")
        lines_method.append("-" * 60)

        for split in SPLITS:
            lines_method.append(f"  --- {split} ---")
            cols = [c for c in mdf.columns if c.startswith(f"{split}_")]
            short = [c.replace(f"{split}_", "") for c in cols]

            hdr = f"  {'':>25}" + "".join(f"{n:>15}" for n in short)
            lines_method.append(hdr)
            lines_method.append("  " + "-" * (len(hdr) - 2))

            for _, row in mdf.iterrows():
                lbl = str(row["DataSplit"])[:25]
                vals = "".join(_fmt_val(row[c]) for c in cols)
                lines_method.append(f"  {lbl:>25}{vals}")
            lines_method.append("")
        lines_method.append("")

    (out_dir / f"{name}.txt").write_text("\n".join(lines_method), encoding="utf-8")
    try:
        df_exp.to_excel(out_dir / f"{name}.xlsx", index=False)
    except ModuleNotFoundError:
        pass
    print(f"\n  -> Experiment summary (method-by-method): {out_dir}/{name}  (.csv, .xlsx, .txt)")

    # ══════════════════════════════════════════════════════════════════════
    #  NEW: Detailed overview .txt  (split-first, then states, then mean)
    # ══════════════════════════════════════════════════════════════════════
    _write_experiment_detailed_txt(exp_dir, method_data, method_std, ordered_states, out_dir)

    # ══════════════════════════════════════════════════════════════════════
    #  NEW: Multi-sheet Excel (one sheet per split + summary sheet)
    # ══════════════════════════════════════════════════════════════════════
    _write_experiment_excel(exp_dir, method_data, method_std, ordered_states, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Detailed .txt  (DATA_SPLIT style, split-first)
# ─────────────────────────────────────────────────────────────────────────────

def _state_table_lines(method_data: dict, state_label: str,
                       split: str, label_width: int = 40) -> list[str]:
    """Build text lines for one state × one split (all methods as rows)."""
    methods = sorted(method_data, key=_method_sort_key)
    metric_names = METRICS
    lines: list[str] = []

    hdr = f"{'':>{label_width}}" + "".join(f"{n:>15}" for n in metric_names)
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for method in methods:
        vals_dict = method_data[method].get(state_label, {}).get(split, {})
        vals = "".join(_fmt_val(vals_dict.get(m, np.nan)) for m in metric_names)
        lines.append(f"{method:>{label_width}}{vals}")

    return lines


def _mean_table_lines(method_data: dict, ordered_states: list[str],
                      split: str, label_width: int = 40) -> list[str]:
    """Build the mean-across-states table for one split."""
    methods = sorted(method_data, key=_method_sort_key)
    metric_names = METRICS
    lines: list[str] = []

    hdr = f"{'':>{label_width}}" + "".join(f"{n:>15}" for n in metric_names)
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for method in methods:
        per_metric: dict[str, list[float]] = {m: [] for m in metric_names}
        for sl in ordered_states:
            vals_dict = method_data[method].get(sl, {}).get(split, {})
            for m in metric_names:
                v = vals_dict.get(m, np.nan)
                if not pd.isna(v):
                    per_metric[m].append(v)

        mean_vals = ""
        for m in metric_names:
            arr = per_metric[m]
            mean_vals += _fmt_val(np.mean(arr) if arr else np.nan)

        lines.append(f"{method:>{label_width}}{mean_vals}")

    return lines


def _write_experiment_detailed_txt(exp_dir: Path, method_data: dict,
                                   method_std: dict,
                                   ordered_states: list[str], out_dir: Path):
    """Write the detailed overview txt (split-first, each state, then mean)."""
    LW = 40
    lines: list[str] = [
        f"Experiment Detailed Overview: {exp_dir.name}",
        "=" * 80, ""
    ]

    for split in SPLITS:
        lines.append(f"{'='*80}")
        lines.append(f"  {split.upper()}")
        lines.append(f"{'='*80}")
        lines.append("")

        for sl in ordered_states:
            lines.append(f"--- {sl} ---")
            lines.extend(_state_table_lines(method_data, sl, split, LW))
            lines.append("")

        lines.append(f"--- Mean across Data Splits ---")
        lines.extend(_mean_table_lines(method_data, ordered_states, split, LW))
        lines.append("")
        lines.append("")

    fname = f"summary_{exp_dir.name}_detailed"
    (out_dir / f"{fname}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {out_dir.relative_to(out_dir.parents[1])}/{fname}.txt")

    # ── Also write the mean±std variant ──
    _write_experiment_detailed_mean_std_txt(exp_dir, method_data, method_std,
                                           ordered_states, out_dir)


def _write_experiment_detailed_mean_std_txt(
    exp_dir: Path, method_data: dict, method_std: dict,
    ordered_states: list[str], out_dir: Path,
):
    """Write detailed overview with paired Metric_mean / Metric_std columns.

    Per-state rows show mean (across 5 runs) and std (across 5 runs).
    The final 'Mean across Data Splits' row averages both mean and std
    across states.
    """
    LW = 40
    # Build paired column headers
    wide_cols: list[str] = []
    for m in METRICS:
        wide_cols.append(f"{m}_mean")
        wide_cols.append(f"{m}_std")

    methods = sorted(method_data, key=_method_sort_key)

    lines: list[str] = [
        f"Experiment Detailed Overview (Mean ± Std): {exp_dir.name}",
        "=" * 80, ""
    ]

    for split in SPLITS:
        lines.append(f"{'='*80}")
        lines.append(f"  {split.upper()}")
        lines.append(f"{'='*80}")
        lines.append("")

        for sl in ordered_states:
            lines.append(f"--- {sl} ---")
            hdr = f"{'':>{LW}}" + "".join(f"{c:>15}" for c in wide_cols)
            lines.append(hdr)
            lines.append("-" * len(hdr))

            for method in methods:
                mean_dict = method_data[method].get(sl, {}).get(split, {})
                std_dict  = method_std.get(method, {}).get(sl, {}).get(split, {})
                row_vals = ""
                for m in METRICS:
                    row_vals += _fmt_val(mean_dict.get(m, np.nan))
                    row_vals += _fmt_val(std_dict.get(m, np.nan))
                lines.append(f"{method:>{LW}}{row_vals}")
            lines.append("")

        # Mean across Data Splits
        lines.append(f"--- Mean across Data Splits ---")
        hdr = f"{'':>{LW}}" + "".join(f"{c:>15}" for c in wide_cols)
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for method in methods:
            row_vals = ""
            for m in METRICS:
                mean_vals, std_vals = [], []
                for sl in ordered_states:
                    mv = method_data[method].get(sl, {}).get(split, {}).get(m, np.nan)
                    sv = method_std.get(method, {}).get(sl, {}).get(split, {}).get(m, np.nan)
                    if not pd.isna(mv):
                        mean_vals.append(mv)
                    if not pd.isna(sv):
                        std_vals.append(sv)
                row_vals += _fmt_val(np.mean(mean_vals) if mean_vals else np.nan)
                row_vals += _fmt_val(np.mean(std_vals) if std_vals else np.nan)
            lines.append(f"{method:>{LW}}{row_vals}")
        lines.append("")
        lines.append("")

    fname = f"summary_{exp_dir.name}_detailed_mean_std"
    (out_dir / f"{fname}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {out_dir.relative_to(out_dir.parents[1])}/{fname}.txt")


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-sheet Excel (one sheet per split + summary)
# ─────────────────────────────────────────────────────────────────────────────

def _build_split_df(method_data: dict, ordered_states: list[str],
                    split: str) -> pd.DataFrame:
    """One DataFrame per split: rows = methods, columns = state × metric."""
    methods = sorted(method_data, key=_method_sort_key)
    records = []

    for method in methods:
        for sl in ordered_states:
            row = {"Method": method, "DataSplit": sl}
            vals_dict = method_data[method].get(sl, {}).get(split, {})
            for m in METRICS:
                row[m] = vals_dict.get(m, np.nan)
            records.append(row)

        # Mean row
        mean_row = {"Method": method, "DataSplit": "Mean"}
        for m in METRICS:
            vals = []
            for sl in ordered_states:
                v = method_data[method].get(sl, {}).get(split, {}).get(m, np.nan)
                if not pd.isna(v):
                    vals.append(v)
            mean_row[m] = np.mean(vals) if vals else np.nan
        records.append(mean_row)

        # Std row
        std_row = {"Method": method, "DataSplit": "Std"}
        for m in METRICS:
            vals = []
            for sl in ordered_states:
                v = method_data[method].get(sl, {}).get(split, {}).get(m, np.nan)
                if not pd.isna(v):
                    vals.append(v)
            std_row[m] = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
        records.append(std_row)

    return pd.DataFrame(records)


def _build_summary_df(method_data: dict, ordered_states: list[str]) -> pd.DataFrame:
    """Summary sheet: one row per method, columns = split × metric (mean only)."""
    methods = sorted(method_data, key=_method_sort_key)
    records = []
    for method in methods:
        row = {"Method": method}
        for split in SPLITS:
            for m in METRICS:
                vals = []
                for sl in ordered_states:
                    v = method_data[method].get(sl, {}).get(split, {}).get(m, np.nan)
                    if not pd.isna(v):
                        vals.append(v)
                row[f"{split}_{m}"] = np.mean(vals) if vals else np.nan
        records.append(row)
    return pd.DataFrame(records)


def _build_summary_wide_df(
    method_data: dict, method_std: dict, ordered_states: list[str]
) -> pd.DataFrame:
    """Summary_MeanStd sheet: paired mean/std columns per metric.

    Mean = mean-of-means across states (same as Summary sheet).
    Std  = mean-of-within-run-stds across states (average run-level variability).
    """
    methods = sorted(method_data, key=_method_sort_key)
    records = []
    for method in methods:
        row: dict = {"Method": method}
        for split in SPLITS:
            for m in METRICS:
                mean_vals, std_vals = [], []
                for sl in ordered_states:
                    mv = method_data[method].get(sl, {}).get(split, {}).get(m, np.nan)
                    sv = method_std.get(method, {}).get(sl, {}).get(split, {}).get(m, np.nan)
                    if not pd.isna(mv):
                        mean_vals.append(mv)
                    if not pd.isna(sv):
                        std_vals.append(sv)
                row[f"{split}_{m}_mean"] = np.mean(mean_vals) if mean_vals else np.nan
                row[f"{split}_{m}_std"]  = np.mean(std_vals) if std_vals else np.nan
        records.append(row)
    return pd.DataFrame(records)


def _write_experiment_excel(exp_dir: Path, method_data: dict,
                            method_std: dict,
                            ordered_states: list[str], out_dir: Path):
    """Write multi-sheet Excel: Train | Validation | Test | Summary | Summary_MeanStd."""
    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:
        print("    (openpyxl not installed – multi-sheet .xlsx skipped)")
        return

    fname = f"summary_{exp_dir.name}_detailed.xlsx"
    with pd.ExcelWriter(out_dir / fname, engine="openpyxl") as writer:
        for split in SPLITS:
            df = _build_split_df(method_data, ordered_states, split)
            df.to_excel(writer, sheet_name=split, index=False,
                        float_format="%.4f")

        df_sum = _build_summary_df(method_data, ordered_states)
        df_sum.to_excel(writer, sheet_name="Summary", index=False,
                        float_format="%.4f")

        df_wide = _build_summary_wide_df(method_data, method_std, ordered_states)
        df_wide.to_excel(writer, sheet_name="Summary_MeanStd", index=False,
                         float_format="%.4f")

    print(f"  -> {out_dir.relative_to(out_dir.parents[1])}/{fname}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXP_DIR
    exp_dir = exp_dir.resolve()

    print("=" * 60)
    print(f"Experiment: {exp_dir}")
    print("=" * 60)

    if not exp_dir.exists():
        print(f"ERROR: directory not found: {exp_dir}")
        sys.exit(1)

    summarize_experiment(exp_dir)

    print("\n" + "=" * 60)
    print("Done! All summaries created.")
    print("=" * 60)
