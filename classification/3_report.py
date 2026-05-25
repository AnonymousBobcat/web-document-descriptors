"""
Generate harmonized-vs-raw descriptor comparison outputs for ARR submission.

Reads:
  results/data_check/class_counts_<task>.csv
  results/classifier/harmonized_descriptors/metrics.csv
  results/classifier/harmonized_descriptors/retrieval.csv
  results/classifier/descriptors/metrics.csv
  results/classifier/descriptors/retrieval.csv

Writes (in results/report/compare/):
  compare_f1.png            - 3-panel figure: macro-F1 vs vocab size,
                              harmonized vs raw, TF-IDF curve
  compare_f1.pdf            - same figure as vector PDF
  compare_main_table.tex    - main F1 table with both variants + TF-IDF
  compare_sensitivity.tex   - full sweep across all configs and both variants
  retrieval_compare_<task>_r<rec>.tex
                            - per-class retrieval, 3-way comparison
                              (Harm. desc., Raw desc., TF-IDF) at 80% and 20%
                              recall, with weighted-mean row
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# ---------- config ----------
DATA_CHECK_DIR = "results/data_check"
CLASSIFIER_DIR = "results/classifier"
OUT_DIR = "results/report/compare"
TASKS = ["label", "topic", "format"]
TASK_DISPLAY = {"label": "FineWeb Edu", "topic": "Topic", "format": "Format"}

# the two descriptor-key runs to compare
VARIANTS = [
    ("harmonized_descriptors", "Harmonized"),
    ("descriptors", "Raw"),
]
VARIANT_KEYS = [k for k, _ in VARIANTS]
VARIANT_NAMES = dict(VARIANTS)

# default configs picked out for the main table (smallest min-freq = full vocab)
DEFAULT_DESC = "descriptors_min1"
DEFAULT_TFIDF = "tfidf_max50000"

# colors and markers per variant; TF-IDF as neutral reference
STYLE = {
    "harmonized_descriptors": {"color": "#1f77b4", "marker": "o"},
    "descriptors": {"color": "#d62728", "marker": "s"},
    "tfidf": {"color": "#555555", "marker": "D"},
}

os.makedirs(OUT_DIR, exist_ok=True)


# ---------- load ----------
def load_variant(key):
    base = f"{CLASSIFIER_DIR}/{key}"
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"missing {base}/ -- run the classifier with --descriptor-key {key}"
        )
    m = pd.read_csv(f"{base}/metrics.csv")
    r = pd.read_csv(f"{base}/retrieval.csv")
    m["variant"] = key
    r["variant"] = key
    return m, r


print("Loading classifier outputs...")
metrics_list, retrieval_list = [], []
for key in VARIANT_KEYS:
    m, r = load_variant(key)
    metrics_list.append(m)
    retrieval_list.append(r)
    print(f"  {key}: {len(m)} metric rows, {len(r)} retrieval rows")

metrics = pd.concat(metrics_list, ignore_index=True)
retrieval = pd.concat(retrieval_list, ignore_index=True)

# per-class test support, joined from data_check outputs
supports = {}
for task in TASKS:
    df = pd.read_csv(f"{DATA_CHECK_DIR}/class_counts_{task}.csv")
    supports[task] = dict(zip(df["class"], df["test_n"]))


# ---------- helpers ----------
def fmt_int(x):
    return f"{int(x):,}"


def fmt_f1(mean, lo, hi):
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def fmt_pct(x):
    return f"{x:.1f}\\%"


def fmt_lift(x):
    return f"{x:.1f}"


def latex_escape(s):
    return str(s).replace("&", r"\&").replace("_", r"\_")


# ---------- Figure: macro-F1 vs vocab size, 3 panels ----------
def write_compare_figure():
    """3 panels (one per task). For each panel:
    - one descriptor curve per variant (harmonized, raw) across THRESHOLDS,
      with CI band
    - TF-IDF as a horizontal dashed line at its single macro-F1 point
      (CI band as shaded horizontal strip)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)

    for ax, task in zip(axes, TASKS):
        sub = metrics[metrics["task"] == task]

        # descriptor curves, one per variant
        for variant, label in VARIANTS:
            s = sub[
                (sub["variant"] == variant) & (sub["feat"] == "descriptors")
            ].sort_values("vocab_size")
            if len(s) == 0:
                continue
            style = STYLE[variant]
            ax.plot(
                s["vocab_size"],
                s["macro_f1"],
                marker=style["marker"],
                color=style["color"],
                label=f"Descriptors ({label})",
                linewidth=1.8,
                markersize=6,
            )
            ax.fill_between(
                s["vocab_size"],
                s["macro_f1_ci_lo"],
                s["macro_f1_ci_hi"],
                alpha=0.15,
                color=style["color"],
                linewidth=0,
            )

        # TF-IDF curve (use harmonized-side rows; TF-IDF doesn't depend on
        # descriptor key, so the two variants' TF-IDF rows are identical).
        tfidf_rows = sub[
            (sub["variant"] == "harmonized_descriptors") & (sub["feat"] == "tfidf")
        ].sort_values("vocab_size")
        if len(tfidf_rows) > 0:
            style = STYLE["tfidf"]
            ax.plot(
                tfidf_rows["vocab_size"],
                tfidf_rows["macro_f1"],
                marker=style["marker"],
                color=style["color"],
                linestyle="--",
                label="TF-IDF",
                linewidth=1.8,
                markersize=6,
            )
            ax.fill_between(
                tfidf_rows["vocab_size"],
                tfidf_rows["macro_f1_ci_lo"],
                tfidf_rows["macro_f1_ci_hi"],
                alpha=0.15,
                color=style["color"],
                linewidth=0,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Vocabulary size (log scale)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(TASK_DISPLAY[task])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, frameon=True)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{OUT_DIR}/compare_f1.{ext}"
        plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote compare_f1.png and compare_f1.pdf")


# ---------- Table: main F1 with both variants ----------
def write_compare_main_table():
    """One row per (task, feature config). Descriptors broken out by variant;
    TF-IDF shown once per task (identical across variants in practice)."""
    rows = []
    for task in TASKS:
        # descriptors, one row per variant
        for variant, label in VARIANTS:
            sub = metrics[
                (metrics["task"] == task)
                & (metrics["variant"] == variant)
                & (metrics["config"] == DEFAULT_DESC)
            ]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "Task": TASK_DISPLAY[task],
                    "Features": f"Descriptors ({label})",
                    "Macro-F1": fmt_f1(
                        r["macro_f1"], r["macro_f1_ci_lo"], r["macro_f1_ci_hi"]
                    ),
                    "W-F1": f"{r['weighted_f1']:.2f}",
                    "$|V|$": fmt_int(r["vocab_size"]),
                }
            )

        # TF-IDF row (use harmonized side; it's the same TF-IDF features)
        sub = metrics[
            (metrics["task"] == task)
            & (metrics["variant"] == "harmonized_descriptors")
            & (metrics["config"] == DEFAULT_TFIDF)
        ]
        if len(sub) > 0:
            r = sub.iloc[0]
            rows.append(
                {
                    "Task": TASK_DISPLAY[task],
                    "Features": "TF-IDF",
                    "Macro-F1": fmt_f1(
                        r["macro_f1"], r["macro_f1_ci_lo"], r["macro_f1_ci_hi"]
                    ),
                    "W-F1": f"{r['weighted_f1']:.2f}",
                    "$|V|$": fmt_int(r["vocab_size"]),
                }
            )

    df = pd.DataFrame(rows)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & Features & Macro-F1 [95\% CI] & W-F1 & $|V|$ \\",
        r"\midrule",
    ]
    prev_task = None
    for _, r in df.iterrows():
        if prev_task is not None and r["Task"] != prev_task:
            lines.append(r"\midrule")
        lines.append(
            f"{r['Task']} & {r['Features']} & {r['Macro-F1']} & "
            f"{r['W-F1']} & {r['$|V|$']} \\\\"
        )
        prev_task = r["Task"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Classification F1 comparing harmonized vs.\ raw descriptors, "
        r"with TF-IDF as a text-based reference. Confidence intervals from "
        r"1000 bootstrap resamples of the test set. Descriptor configs use "
        r"minimum frequency 1 (full vocabulary); TF-IDF capped at "
        r"50{,}000 features. $|V|$ = vocabulary size.}",
        r"\label{tab:compare_main}",
        r"\end{table}",
    ]
    with open(f"{OUT_DIR}/compare_main_table.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote compare_main_table.tex ({len(df)} rows)")


# ---------- Table: full sensitivity sweep, both variants + TF-IDF ----------
def write_compare_sensitivity_table():
    rows = []
    for task in TASKS:
        # descriptor sweep, one block per variant
        for variant, label in VARIANTS:
            sub = metrics[
                (metrics["task"] == task)
                & (metrics["variant"] == variant)
                & (metrics["feat"] == "descriptors")
            ].sort_values("vocab_size")
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "Task": TASK_DISPLAY[task],
                        "Variant": label,
                        "Config": r["config"].replace("_", r"\_"),
                        "$|V|$": fmt_int(r["vocab_size"]),
                        "Macro-F1": fmt_f1(
                            r["macro_f1"],
                            r["macro_f1_ci_lo"],
                            r["macro_f1_ci_hi"],
                        ),
                        "W-F1": f"{r['weighted_f1']:.2f}",
                    }
                )

        # TF-IDF sweep (one block per task; doesn't depend on descriptor key,
        # so we read from the harmonized run)
        sub = metrics[
            (metrics["task"] == task)
            & (metrics["variant"] == "harmonized_descriptors")
            & (metrics["feat"] == "tfidf")
        ].sort_values("vocab_size")
        for _, r in sub.iterrows():
            rows.append(
                {
                    "Task": TASK_DISPLAY[task],
                    "Variant": "TF-IDF",
                    "Config": r["config"].replace("_", r"\_"),
                    "$|V|$": fmt_int(r["vocab_size"]),
                    "Macro-F1": fmt_f1(
                        r["macro_f1"],
                        r["macro_f1_ci_lo"],
                        r["macro_f1_ci_hi"],
                    ),
                    "W-F1": f"{r['weighted_f1']:.2f}",
                }
            )

    df = pd.DataFrame(rows)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllrcc}",
        r"\toprule",
        r"Task & Variant & Config & $|V|$ & Macro-F1 [95\% CI] & W-F1 \\",
        r"\midrule",
    ]
    prev_task, prev_variant = None, None
    for _, r in df.iterrows():
        if prev_task is not None and (
            r["Task"] != prev_task or r["Variant"] != prev_variant
        ):
            lines.append(r"\midrule")
        lines.append(
            f"{r['Task']} & {r['Variant']} & {r['Config']} & "
            f"{r['$|V|$']} & {r['Macro-F1']} & {r['W-F1']} \\\\"
        )
        prev_task = r["Task"]
        prev_variant = r["Variant"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Sensitivity to feature configuration. Descriptor blocks "
        r"sweep minimum token frequency; TF-IDF block sweeps maximum "
        r"vocabulary size (`tfidf\_full' = no cap). CIs from 1000 bootstrap "
        r"resamples of the test set.}",
        r"\label{tab:compare_sensitivity}",
        r"\end{table}",
    ]
    with open(f"{OUT_DIR}/compare_sensitivity.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote compare_sensitivity.tex ({len(df)} rows)")


# ---------- Table: per-class retrieval, 3-way comparison ----------
def write_retrieval_compare_table(task, recall_level):
    """Per-class retrieval comparison at one recall level for one task.
    Three blocks side-by-side: descriptors-Harmonized, descriptors-Raw, TF-IDF.
    Each block reports Visited%, Lift, Precision. Final column is test support N.
    Sorted ascending by harmonized-descriptor Visited% (best classes first).
    A weighted-mean row (by test support) is appended.
    """
    rec_df = retrieval[retrieval["target_recall"] == recall_level]
    rec_df = rec_df[rec_df["task"] == task]

    # pull each block; TF-IDF is read from the harmonized run (identical text features)
    desc_h = rec_df[
        (rec_df["variant"] == "harmonized_descriptors")
        & (rec_df["config"] == DEFAULT_DESC)
    ].set_index("class")
    desc_r = rec_df[
        (rec_df["variant"] == "descriptors") & (rec_df["config"] == DEFAULT_DESC)
    ].set_index("class")
    tfidf = rec_df[
        (rec_df["variant"] == "harmonized_descriptors")
        & (rec_df["config"] == DEFAULT_TFIDF)
    ].set_index("class")

    classes = sorted(
        set(desc_h.index) & set(desc_r.index) & set(tfidf.index),
        key=lambda c: desc_h.loc[c, "kept_pct"],
    )

    if not classes:
        print(
            f"  skipped retrieval compare for {task} @ recall={recall_level}: "
            f"no overlapping classes (check DEFAULT_DESC/DEFAULT_TFIDF)"
        )
        return

    rows = []
    for c in classes:
        h, r_, t = desc_h.loc[c], desc_r.loc[c], tfidf.loc[c]
        rows.append(
            {
                "Class": latex_escape(c),
                "V_h": fmt_pct(h["kept_pct"]),
                "L_h": fmt_lift(h["lift"]),
                "P_h": fmt_pct(h["precision"] * 100),
                "V_r": fmt_pct(r_["kept_pct"]),
                "L_r": fmt_lift(r_["lift"]),
                "P_r": fmt_pct(r_["precision"] * 100),
                "V_t": fmt_pct(t["kept_pct"]),
                "L_t": fmt_lift(t["lift"]),
                "P_t": fmt_pct(t["precision"] * 100),
                "N": fmt_int(supports[task].get(c, 0)),
            }
        )

    # weighted means by test support
    total_n = sum(supports[task].get(c, 0) for c in classes)
    if total_n > 0:

        def wmean(df_, col):
            return (
                sum(df_.loc[c, col] * supports[task].get(c, 0) for c in classes)
                / total_n
            )

        rows.append(
            {
                "Class": r"\textbf{Weighted mean}",
                "V_h": fmt_pct(wmean(desc_h, "kept_pct")),
                "L_h": fmt_lift(wmean(desc_h, "lift")),
                "P_h": fmt_pct(wmean(desc_h, "precision") * 100),
                "V_r": fmt_pct(wmean(desc_r, "kept_pct")),
                "L_r": fmt_lift(wmean(desc_r, "lift")),
                "P_r": fmt_pct(wmean(desc_r, "precision") * 100),
                "V_t": fmt_pct(wmean(tfidf, "kept_pct")),
                "L_t": fmt_lift(wmean(tfidf, "lift")),
                "P_t": fmt_pct(wmean(tfidf, "precision") * 100),
                "N": fmt_int(total_n),
            }
        )

    df = pd.DataFrame(rows)
    rec_pct = int(recall_level * 100)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l|ccc|ccc|ccc|c}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{Desc. (Harm.)} "
        r"& \multicolumn{3}{c|}{Desc. (Raw)} "
        r"& \multicolumn{3}{c|}{TF-IDF} & \\",
        r"Class & Vis. & Lift & Prec. "
        r"& Vis. & Lift & Prec. "
        r"& Vis. & Lift & Prec. & N \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        if r["Class"].startswith(r"\textbf"):
            lines.append(r"\midrule")
        lines.append(
            f"{r['Class']} & {r['V_h']} & {r['L_h']} & {r['P_h']} "
            f"& {r['V_r']} & {r['L_r']} & {r['P_r']} "
            f"& {r['V_t']} & {r['L_t']} & {r['P_t']} & {r['N']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Per-class retrieval at {rec_pct}\% recall for "
        rf"{TASK_DISPLAY[task]}, comparing harmonized descriptors, raw "
        rf"descriptors, and TF-IDF. Vis.\ = fraction of test set scanned to "
        rf"reach target recall (lower is better); Lift = precision $/$ base "
        rf"rate (higher is better); Prec.\ = precision at that operating "
        rf"point; N = test support. Descriptor configs use minimum frequency "
        rf"1; TF-IDF capped at 50{{,}}000 features. Sorted ascending by "
        rf"harmonized-descriptor Vis.}}",
        rf"\label{{tab:retrieval_compare_{task}_r{rec_pct}}}",
        r"\end{table}",
    ]
    out_path = f"{OUT_DIR}/retrieval_compare_{task}_r{rec_pct}.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.basename(out_path)} ({len(classes)} classes)")


# ---------- run ----------
print(f"\nWriting comparison outputs to {OUT_DIR}/")
write_compare_figure()
write_compare_main_table()
write_compare_sensitivity_table()
for task in TASKS:
    for r in (0.80, 0.20):
        write_retrieval_compare_table(task, r)
print(f"\nDone.")
