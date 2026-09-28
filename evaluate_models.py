"""
evaluate_models.py
================================================================================
Combines CHECKPOINT/model1/, CHECKPOINT/model2/, CHECKPOINT/model3/ (produced
independently by run_phase7_model1.py / model2.py / model3.py, possibly on
three different machines) and computes every measure discussed for this
study. No GPU needed — this script only reads saved CSVs and does statistics.

Run this AFTER all three model scripts have finished (or after however many
folds you managed to complete — it will compute what it can and clearly flag
anything based on incomplete data).

USAGE:
    Put CHECKPOINT/model1/, CHECKPOINT/model2/, CHECKPOINT/model3/ (each
    downloaded from wherever it was trained) in the same folder as this
    script, then:
        pip install -q pandas numpy scipy scikit-learn matplotlib
        python evaluate_models.py

OUTPUT:
  results/classification_summary.csv        — mean/std accuracy & F1 per model
  results/significance_tests.json           — paired t-test / Wilcoxon / Cohen's d
  results/fragmentation_confidence_correlation.csv
  results/code_mixed_fragmentation_ttest.csv
  results/fragmentation_stratified_accuracy.csv
  results/audited_subset_check.csv          — only if a human-audit file is found
  results/merged_predictions_wide.csv       — one row per sentence, all models side by side
  results/image/model_comparison.png
  results/image/fragmentation_confidence_scatter.png
  results/image/fragmentation_stratified_accuracy.png
  results/image/correlation_shrinkage.png
"""

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, ttest_ind, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "CHECKPOINT"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = RESULTS_DIR / "image"
RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS_EXPECTED = 5

# checkpoint dirname -> friendly label/color, used when label_map.json
# doesn't specify one (kept in sync with run_phase7_model{1,2,3}.py)
KNOWN_MODEL_INFO = {
    "model1": {"label": "mBERT (baseline)",  "color": "#e94560"},
    "model2": {"label": "MuRIL (baseline)",  "color": "#0f3460"},
    "model3": {"label": "mBERT (adapted)",   "color": "#53d8fb"},
    "model4": {"label": "XLM-R (base)",      "color": "#f5a623"},
}
REFERENCE_MODEL_DIR = "model1"  # used as the fragmentation stratification reference


# ── Loading ──────────────────────────────────────────────────────────────────
def load_model(dirname: str):
    """Returns (info dict, fold_metrics df, concatenated predictions df) or None if missing."""
    model_dir = CHECKPOINT_DIR / dirname
    if not model_dir.exists():
        print(f"  [{dirname}] NOT FOUND at {model_dir} — skipping this model entirely.")
        return None

    label_map_path = model_dir / "label_map.json"
    meta = {}
    if label_map_path.exists():
        with open(label_map_path, encoding="utf-8") as f:
            meta = json.load(f)

    fold_metrics_path = model_dir / "fold_metrics.csv"
    fold_metrics = pd.read_csv(fold_metrics_path) if fold_metrics_path.exists() else pd.DataFrame()

    pred_frames = []
    for fold_num in range(1, N_FOLDS_EXPECTED + 1):
        p = model_dir / f"fold{fold_num}_predictions.csv"
        if p.exists():
            pred_frames.append(pd.read_csv(p, encoding="utf-8-sig"))
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    n_folds_found = len(fold_metrics) if not fold_metrics.empty else 0
    friendly = KNOWN_MODEL_INFO.get(dirname, {"label": meta.get("model_name", dirname), "color": "#888888"})
    info = {
        "dirname": dirname,
        "model_name": meta.get("model_name", dirname),
        "label": friendly["label"],
        "color": friendly["color"],
        "dataset_name": meta.get("dataset_name", "unknown"),
        "n_folds_found": n_folds_found,
    }
    status = "COMPLETE" if n_folds_found == N_FOLDS_EXPECTED else f"PARTIAL ({n_folds_found}/{N_FOLDS_EXPECTED} folds)"
    print(f"  [{dirname}] {info['label']} — {status}, {len(predictions):,} prediction rows loaded")
    return info, fold_metrics, predictions


print("=" * 70)
print("  LOADING CHECKPOINTS")
print("=" * 70)

models = {}
for dirname in ["model1", "model2", "model3", "model4"]:
    result = load_model(dirname)
    if result is not None:
        info, fold_metrics, predictions = result
        models[dirname] = {"info": info, "fold_metrics": fold_metrics, "predictions": predictions}

if not models:
    raise SystemExit("No model checkpoints found under CHECKPOINT/. Nothing to evaluate.")


# ── 1. Classification summary (mean/std per model) ──────────────────────────
print("\n" + "=" * 70)
print("  1. CLASSIFICATION SUMMARY")
print("=" * 70)

summary_rows = []
for dirname, m in models.items():
    fm = m["fold_metrics"]
    if fm.empty:
        continue
    row = {
        "model_dir": dirname,
        "model": m["info"]["label"],
        "dataset": m["info"]["dataset_name"],
        "n_folds": len(fm),
        "mean_accuracy": fm["accuracy"].mean(), "std_accuracy": fm["accuracy"].std(),
        "mean_macro_f1": fm["macro_f1"].mean(), "std_macro_f1": fm["macro_f1"].std(),
        "mean_macro_precision": fm["macro_precision"].mean(), "std_macro_precision": fm["macro_precision"].std(),
        "mean_macro_recall": fm["macro_recall"].mean(), "std_macro_recall": fm["macro_recall"].std(),
        "mean_train_time_min": fm["train_time_min"].mean() if "train_time_min" in fm.columns else None,
    }
    summary_rows.append(row)
    print(f"  {m['info']['label']:<28} Macro-F1={row['mean_macro_f1']:.4f}±{row['std_macro_f1']:.4f}  "
          f"Acc={row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f}  ({row['n_folds']}/{N_FOLDS_EXPECTED} folds)")

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(RESULTS_DIR / "classification_summary.csv", index=False)


# ── 2. Paired significance tests (adapted vs baseline, MuRIL vs baseline) ───
print("\n" + "=" * 70)
print("  2. SIGNIFICANCE TESTS")
print("=" * 70)

def get_f1_array(dirname):
    if dirname not in models or models[dirname]["fold_metrics"].empty:
        return None
    fm = models[dirname]["fold_metrics"].sort_values("fold")
    return fm.set_index("fold")["macro_f1"]

def run_paired_tests(name_a, dirname_a, name_b, dirname_b):
    fa, fb = get_f1_array(dirname_a), get_f1_array(dirname_b)
    if fa is None or fb is None:
        print(f"  {name_a} vs {name_b}: skipped (missing model)")
        return None
    common_folds = sorted(set(fa.index) & set(fb.index))
    if len(common_folds) < 2:
        print(f"  {name_a} vs {name_b}: skipped (only {len(common_folds)} matching folds)")
        return None
    a_vals = fa.loc[common_folds].values
    b_vals = fb.loc[common_folds].values
    diff   = a_vals - b_vals
    n      = len(common_folds)

    t_stat, p_ttest = ttest_rel(a_vals, b_vals)

    # FIX Issue-3: Wilcoxon signed-rank requires n >= 6 to produce a valid
    # p-value (with n=5 scipy returns p=1.0 or raises ValueError).
    # Guard against this and flag clearly in output.
    WILCOXON_MIN_N = 6
    if n >= WILCOXON_MIN_N:
        try:
            _, p_wilcoxon = wilcoxon(a_vals, b_vals)
        except Exception:
            p_wilcoxon = float("nan")
    else:
        p_wilcoxon = float("nan")
        print(f"  [NOTE] Wilcoxon skipped for '{name_a} vs {name_b}': "
              f"n={n} < minimum required n={WILCOXON_MIN_N}. "
              f"Interpret paired t-test only.")

    # FIX Issue-4: Cohen's d from n=5 differences is very noisy.
    # Compute a 10,000-sample bootstrap 95% CI around d to quantify uncertainty.
    cohens_d_raw = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0
    rng = np.random.default_rng(42)
    boot_d = []
    for _ in range(10_000):
        samp = rng.choice(diff, size=n, replace=True)
        d_b = float(samp.mean() / samp.std()) if samp.std() > 0 else 0.0
        boot_d.append(d_b)
    d_ci_lo, d_ci_hi = float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5))

    result = {
        "comparison":          f"{name_a} vs {name_b}",
        "n_folds_compared":    n,
        "mean_diff":           round(float(diff.mean()), 6),
        "cohens_d":            round(cohens_d_raw, 4),
        "cohens_d_ci95_lo":    round(d_ci_lo, 4),   # bootstrap 95% CI lower bound
        "cohens_d_ci95_hi":    round(d_ci_hi, 4),   # bootstrap 95% CI upper bound
        "cohens_d_note":       f"Estimated from n={n} folds; wide CI reflects small sample.",
        "paired_ttest_p":      round(float(p_ttest), 6),
        "wilcoxon_p":          round(float(p_wilcoxon), 6) if not np.isnan(p_wilcoxon) else None,
        "wilcoxon_note":       None if n >= WILCOXON_MIN_N else f"Not computed: n={n} < {WILCOXON_MIN_N}",
        "significant_ttest":   bool(p_ttest < 0.05),
        "significant_wilcoxon": bool(p_wilcoxon < 0.05) if not np.isnan(p_wilcoxon) else None,
    }
    sig_flag = "significant" if result["significant_ttest"] else "not significant"
    print(f"  {name_a} vs {name_b}: Δ={result['mean_diff']:+.4f}  "
          f"d={result['cohens_d']:+.4f} [95%CI {d_ci_lo:+.3f}..{d_ci_hi:+.3f}]  "
          f"p={result['paired_ttest_p']:.4f} ({sig_flag}, n={n} folds)")
    return result


sig_results = {}
if "model3" in models and "model1" in models:
    sig_results["adapted_vs_baseline"] = run_paired_tests(
        models["model3"]["info"]["label"], "model3", models["model1"]["info"]["label"], "model1")
if "model2" in models and "model1" in models:
    sig_results["muril_vs_baseline"] = run_paired_tests(
        models["model2"]["info"]["label"], "model2", models["model1"]["info"]["label"], "model1")
if "model4" in models and "model1" in models:
    sig_results["xlmr_vs_baseline"] = run_paired_tests(
        models["model4"]["info"]["label"], "model4", models["model1"]["info"]["label"], "model1")

with open(RESULTS_DIR / "significance_tests.json", "w", encoding="utf-8") as f:
    json.dump(sig_results, f, indent=2, ensure_ascii=False)


# ── 3. Fragmentation-confidence correlation (per model) ─────────────────────
print("\n" + "=" * 70)
print("  3. FRAGMENTATION-CONFIDENCE CORRELATION")
print("=" * 70)

frag_corr_rows = []
for dirname, m in models.items():
    preds = m["predictions"]
    if preds.empty or "fragmentation_ratio" not in preds.columns:
        continue
    r, p = pearsonr(preds["fragmentation_ratio"], preds["confidence"])
    frag_corr_rows.append({
        "model_dir": dirname, "model": m["info"]["label"],
        "pearson_r": round(float(r), 4), "p_value": round(float(p), 8),
        "n": len(preds), "significant": bool(p < 0.05),
    })
    print(f"  {m['info']['label']:<28} r={r:+.4f}  p={p:.2e}  (n={len(preds):,})")

df_frag_corr = pd.DataFrame(frag_corr_rows)
df_frag_corr.to_csv(RESULTS_DIR / "fragmentation_confidence_correlation.csv", index=False)


# ── 4. Code-mixed vs monolingual fragmentation (Welch's t-test, per model) ──
print("\n" + "=" * 70)
print("  4. CODE-MIXED VS MONOLINGUAL FRAGMENTATION")
print("=" * 70)

code_mixed_rows = []
for dirname, m in models.items():
    preds = m["predictions"]
    if preds.empty or "is_code_mixed" not in preds.columns:
        continue
    cm_frag = preds.loc[preds["is_code_mixed"], "fragmentation_ratio"]
    mono_frag = preds.loc[~preds["is_code_mixed"], "fragmentation_ratio"]
    if len(cm_frag) < 2 or len(mono_frag) < 2:
        print(f"  {m['info']['label']:<28} skipped — not enough sentences in one group")
        continue
    t_stat, p_val = ttest_ind(cm_frag, mono_frag, equal_var=False)
    code_mixed_rows.append({
        "model_dir": dirname, "model": m["info"]["label"],
        "n_code_mixed": len(cm_frag), "n_monolingual": len(mono_frag),
        "mean_frag_code_mixed": round(float(cm_frag.mean()), 4),
        "mean_frag_monolingual": round(float(mono_frag.mean()), 4),
        "welch_t": round(float(t_stat), 4), "p_value": round(float(p_val), 8),
        "significant": bool(p_val < 0.05),
    })
    print(f"  {m['info']['label']:<28} code-mixed={cm_frag.mean():.3f}  "
          f"monolingual={mono_frag.mean():.3f}  t={t_stat:.2f}  p={p_val:.2e}")

df_code_mixed = pd.DataFrame(code_mixed_rows)
df_code_mixed.to_csv(RESULTS_DIR / "code_mixed_fragmentation_ttest.csv", index=False)


# ── 5. Fragmentation-stratified accuracy gain ───────────────────────────────
print("\n" + "=" * 70)
print("  5. FRAGMENTATION-STRATIFIED ACCURACY")
print(f"     (quartiles defined by {REFERENCE_MODEL_DIR}'s fragmentation ratio,")
print("      applied consistently across all models so the comparison is fair)")
print("=" * 70)

strat_table = None
if REFERENCE_MODEL_DIR in models and not models[REFERENCE_MODEL_DIR]["predictions"].empty:
    ref_preds = models[REFERENCE_MODEL_DIR]["predictions"][["text", "fragmentation_ratio"]].drop_duplicates("text")
    try:
        ref_preds["frag_quartile"] = pd.qcut(ref_preds["fragmentation_ratio"], 4,
                                              labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    except ValueError as e:
        print(f"  Could not compute quartiles ({e}) — skipping this analysis.")
        ref_preds = None

    if ref_preds is not None:
        quartile_map = ref_preds.set_index("text")["frag_quartile"]

        strat_rows = []
        for dirname, m in models.items():
            preds = m["predictions"]
            if preds.empty:
                continue
            joined = preds.copy()
            joined["frag_quartile"] = joined["text"].map(quartile_map)
            joined = joined.dropna(subset=["frag_quartile"])
            coverage = len(joined) / len(preds) if len(preds) else 0
            for q, grp in joined.groupby("frag_quartile", observed=True):
                strat_rows.append({
                    "quartile": q, "model_dir": dirname, "model": m["info"]["label"],
                    "n": len(grp), "accuracy": grp["correct"].mean(),
                })
            print(f"  {m['info']['label']:<28} joined {len(joined):,}/{len(preds):,} "
                  f"rows to reference quartiles ({coverage*100:.1f}% coverage)")

        strat_table = pd.DataFrame(strat_rows)
        if not strat_table.empty:
            pivot = strat_table.pivot_table(index="quartile", columns="model", values="accuracy", observed=True)
            pivot = pivot.reindex(["Q1_low", "Q2", "Q3", "Q4_high"])
            print("\n  Accuracy by fragmentation quartile:")
            print(pivot.round(4).to_string())

            # Explicit delta column: adapted minus baseline, per quartile
            base_label = models[REFERENCE_MODEL_DIR]["info"]["label"]
            if "model3" in models:
                adapted_label = models["model3"]["info"]["label"]
                if adapted_label in pivot.columns and base_label in pivot.columns:
                    pivot["delta_adapted_minus_baseline"] = pivot[adapted_label] - pivot[base_label]
                    print("\n  Delta (adapted - baseline) by quartile — should be largest for Q4_high "
                          "if the mechanism is real:")
                    print(pivot["delta_adapted_minus_baseline"].round(4).to_string())
            pivot.to_csv(RESULTS_DIR / "fragmentation_stratified_accuracy.csv")
else:
    print(f"  Reference model {REFERENCE_MODEL_DIR} not available — skipping this analysis.")


# ── 6. Human-audited subset consistency check (optional) ────────────────────
print("\n" + "=" * 70)
print("  6. HUMAN-AUDITED SUBSET CHECK (optional)")
print("=" * 70)

audit_path = DATA_PROCESSED / "human_audit_sample.csv"
audit_rows = []
if audit_path.exists():
    audit_df = pd.read_csv(audit_path, encoding="utf-8-sig")
    text_col = next((c for c in audit_df.columns if "text" in c.lower()), None)
    label_col = next((c for c in audit_df.columns
                       if c.lower() in {"human_label", "verified_label", "audited_label",
                                         "manual_label", "gold_label", "sentiment_label", "label"}
                       and c != text_col), None)
    if text_col is None or label_col is None:
        print(f"  Found {audit_path.name} but couldn't identify text/label columns "
              f"(columns present: {list(audit_df.columns)}). Rename them or edit this "
              f"script's column-detection list, then re-run this section.")
    else:
        audit_df = audit_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "gold_label"})
        audit_df["text"] = audit_df["text"].astype(str).str.strip()
        for dirname, m in models.items():
            preds = m["predictions"]
            if preds.empty:
                continue
            joined = preds.merge(audit_df, on="text", how="inner")
            if joined.empty:
                continue
            from sklearn.metrics import accuracy_score, f1_score
            acc_audited = accuracy_score(joined["gold_label"], joined["pred_label"])
            f1_audited = f1_score(joined["gold_label"], joined["pred_label"], average="macro", zero_division=0)
            acc_full = preds["correct"].mean()
            audit_rows.append({
                "model_dir": dirname, "model": m["info"]["label"], "n_audited_matched": len(joined),
                "accuracy_on_audited_subset": round(float(acc_audited), 4),
                "macro_f1_on_audited_subset": round(float(f1_audited), 4),
                "accuracy_on_full_set": round(float(acc_full), 4),
                "direction_agrees": None,  # filled in below once both models are computed
            })
            print(f"  {m['info']['label']:<28} audited acc={acc_audited:.4f}  "
                  f"full-set acc={acc_full:.4f}  (n_matched={len(joined)})")
        if audit_rows:
            pd.DataFrame(audit_rows).to_csv(RESULTS_DIR / "audited_subset_check.csv", index=False)
else:
    print(f"  No {audit_path.name} found — skipping. This is optional; if you finish "
          f"the human-audit sample later, re-run just this section.")


# ── 7. Merged wide predictions table (one row per sentence, all models) ────
print("\n" + "=" * 70)
print("  7. MERGED PREDICTIONS TABLE")
print("=" * 70)

merged = None
for dirname, m in models.items():
    preds = m["predictions"]
    if preds.empty:
        continue
    cols = preds[["text", "true_label", "pred_label", "correct", "confidence",
                  "fragmentation_ratio", "is_code_mixed", "fold"]].copy()
    cols = cols.rename(columns={c: f"{dirname}_{c}" for c in cols.columns if c != "text"})
    merged = cols if merged is None else merged.merge(cols, on="text", how="outer")

if merged is not None:
    merged.to_csv(RESULTS_DIR / "merged_predictions_wide.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved {len(merged):,} rows to merged_predictions_wide.csv")


# ── Plots ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  GENERATING FIGURES")
print("=" * 70)

def style_dark_axes(ax):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

# 8a. Model comparison bar chart
if not df_summary.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#1a1a2e")
    for ax, (metric_name, mean_col, std_col) in zip(
        axes, [("Macro-F1", "mean_macro_f1", "std_macro_f1"), ("Accuracy", "mean_accuracy", "std_accuracy")]
    ):
        style_dark_axes(ax)
        colors = [KNOWN_MODEL_INFO.get(d, {}).get("color", "#888888") for d in df_summary["model_dir"]]
        x = np.arange(len(df_summary))
        bars = ax.bar(x, df_summary[mean_col], yerr=df_summary[std_col], capsize=6,
                       color=colors, edgecolor="white", linewidth=0.5, alpha=0.85,
                       error_kw=dict(ecolor="white", elinewidth=1.5))
        ax.set_xticks(x)
        ax.set_xticklabels(df_summary["model"], fontsize=9, color="white", rotation=12, ha="right")
        ax.set_title(metric_name, fontsize=13, color="white", fontweight="bold")
        for bar, mean_v, std_v in zip(bars, df_summary[mean_col], df_summary[std_col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_v + 0.003,
                    f"{mean_v:.4f}", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")
    fig.suptitle("Model Comparison — Macro-F1 and Accuracy", fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "model_comparison.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved model_comparison.png")

# 8b. Fragmentation-confidence scatter, one panel per model
plot_models = [(d, m) for d, m in models.items() if not m["predictions"].empty]
if plot_models:
    fig, axes = plt.subplots(1, len(plot_models), figsize=(5.5 * len(plot_models), 5), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("#1a1a2e")
    for ax, (dirname, m) in zip(axes, plot_models):
        style_dark_axes(ax)
        preds = m["predictions"]
        sample = preds.sample(min(3000, len(preds)), random_state=42)
        ax.scatter(sample["fragmentation_ratio"], sample["confidence"],
                    s=6, alpha=0.25, color=KNOWN_MODEL_INFO.get(dirname, {}).get("color", "#53d8fb"))
        row = df_frag_corr[df_frag_corr["model_dir"] == dirname]
        if not row.empty:
            r_val, p_val = row.iloc[0]["pearson_r"], row.iloc[0]["p_value"]
            ax.set_title(f"{m['info']['label']}\nr={r_val:+.4f}, p={p_val:.1e}",
                         fontsize=10, color="white")
        ax.set_xlabel("Fragmentation ratio (subwords/word)", fontsize=9, color="white")
        ax.set_ylabel("Confidence", fontsize=9, color="white")
    fig.suptitle("Fragmentation vs. Prediction Confidence", fontsize=12, color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "fragmentation_confidence_scatter.png", dpi=300,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved fragmentation_confidence_scatter.png")

# 8c. Fragmentation-stratified accuracy grouped bar chart
if strat_table is not None and not strat_table.empty:
    pivot_plot = strat_table.pivot_table(index="quartile", columns="model", values="accuracy", observed=True)
    pivot_plot = pivot_plot.reindex(["Q1_low", "Q2", "Q3", "Q4_high"])
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#1a1a2e")
    style_dark_axes(ax)
    n_models_plot = len(pivot_plot.columns)
    width = 0.8 / max(1, n_models_plot)
    x = np.arange(len(pivot_plot.index))
    for i, col in enumerate(pivot_plot.columns):
        color = None
        for d, info in KNOWN_MODEL_INFO.items():
            if info["label"] == col:
                color = info["color"]
        ax.bar(x + i * width, pivot_plot[col], width=width, label=col, color=color, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x + width * (n_models_plot - 1) / 2)
    ax.set_xticklabels(pivot_plot.index, color="white")
    ax.set_ylabel("Accuracy", color="white")
    ax.set_title(f"Accuracy by Fragmentation Quartile\n(quartiles from {REFERENCE_MODEL_DIR})",
                 color="white", fontweight="bold")
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "fragmentation_stratified_accuracy.png", dpi=300,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved fragmentation_stratified_accuracy.png")

# 8d. |r| shrinkage bar chart
if not df_frag_corr.empty:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1a2e")
    style_dark_axes(ax)
    colors = [KNOWN_MODEL_INFO.get(d, {}).get("color", "#888888") for d in df_frag_corr["model_dir"]]
    ax.bar(df_frag_corr["model"], df_frag_corr["pearson_r"].abs(), color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("|Pearson r|  (fragmentation vs. confidence)", color="white")
    ax.set_title("Fragmentation-Confidence Correlation Strength\n(lower = fragmentation predicts confidence less)",
                 color="white", fontweight="bold", fontsize=10)
    ax.set_xticklabels(df_frag_corr["model"], rotation=12, ha="right", color="white")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "correlation_shrinkage.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved correlation_shrinkage.png")

print("\n" + "=" * 70)
print("  DONE — see results/ for tables and results/image/ for figures")
print("=" * 70)
