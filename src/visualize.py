"""
visualize.py — All plots and figures.

Main figures:
  Fig 1: Recovery heatmap (error class × ICL strategy) — the paper's key result
  Fig 2: Accuracy by model family at fixed size (3B tier)
  Fig 3: Accuracy vs size tier per strategy
  Fig 4: Error distribution stacked bars per model
  Fig 5: Robustness ratio comparison
  Fig 6: JS divergence heatmap (baseline vs each strategy)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Global style
PALETTE_FAMILY = {"llama": "#4C72B0", "qwen": "#DD8452", "phi": "#55A868", "gemma": "#C44E52"}
PALETTE_TIER   = {"1-2B": "#9ecae1", "3B": "#3182bd", "7-9B": "#08519c"}
STRATEGY_COLORS = {
    "S0": "#b0b0b0", "S1": "#fdae6b", "S2": "#fd8d3c",
    "S3": "#e6550d", "S4": "#a63603", "S5": "#74c476", "S6": "#238b45",
    "S5_RANDOM": "#9ecae1", "S5_CORRECT_ONLY": "#31a354",
}

ERROR_CLASS_LABELS = {
    "E1": "Arithmetic\nSlip", "E2": "Distractor\nCapture",
    "E3": "Premise-Order\nSensitivity", "E4": "Step\nSkipping",
    "E5": "Hallucinated\nPremise", "E6": "Format\nError",
    "E7": "Logic\nReversal",
}
STRATEGY_LABELS = {
    "S1": "Zero-shot\nCoT", "S2": "Few-shot\n(k=3)", "S3": "Few-shot\nCoT (k=3)",
    "S4": "Few-shot\nCoT (k=5)", "S5": "Error-Targeted\nICL★", "S6": "Self-\nConsistency",
    "S5_RANDOM": "Random-Target\nICL", "S5_CORRECT_ONLY": "Targeted\nCorrect-Only",
}


def _save(fig, figures_dir: Optional[str], filename: str, dpi: int = 150):
    if figures_dir:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        path = Path(figures_dir) / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure: {path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# Fig 1: Recovery Heatmap (key result)
# ─────────────────────────────────────────────
def plot_recovery_heatmap(
    recovery_df: pd.DataFrame,
    figures_dir: Optional[str] = None,
    title: str = "Per-Error-Class × Per-ICL-Strategy Recovery Rate",
):
    """
    Main contribution figure.
    recovery_df: rows=error_class (E1–E7), cols=strategy (S1–S6), values=recovery rate.
    """
    from src.config import ERROR_CLASS_KEYS
    ec_order = [ec for ec in ERROR_CLASS_KEYS if ec in recovery_df.index]
    strat_order = [s for s in ["S1", "S2", "S3", "S4", "S5", "S6"] if s in recovery_df.columns]

    data = recovery_df.loc[ec_order, strat_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        data,
        ax=ax,
        annot=True, fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Recovery Rate (0=never fixed, 1=always fixed)"},
        xticklabels=[STRATEGY_LABELS.get(s, s) for s in strat_order],
        yticklabels=[ERROR_CLASS_LABELS.get(e, e) for e in ec_order],
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("ICL Strategy", fontsize=11)
    ax.set_ylabel("Error Class (from S0 baseline)", fontsize=11)

    # Highlight the novel strategy column (S5)
    if "S5" in strat_order:
        s5_idx = strat_order.index("S5")
        ax.add_patch(plt.Rectangle(
            (s5_idx, 0), 1, len(ec_order),
            fill=False, edgecolor="gold", lw=3, label="Novel (S5)"
        ))
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    _save(fig, figures_dir, "fig1_recovery_heatmap.pdf")


# ─────────────────────────────────────────────
# Fig 2: Family comparison at fixed size (3B)
# ─────────────────────────────────────────────
def plot_family_comparison(
    accuracy_df: pd.DataFrame,
    size_tier: str = "3B",
    strategy_keys: List[str] = None,
    figures_dir: Optional[str] = None,
):
    """Bar chart: accuracy by family, grouped by strategy, at a fixed size tier."""
    strategy_keys = strategy_keys or ["S0", "S1", "S3", "S5", "S6"]
    # Filter to size tier
    mask = accuracy_df.index.get_level_values("size_tier") == size_tier
    df = accuracy_df[mask].reset_index()

    if df.empty:
        logger.warning(f"No data for size_tier={size_tier}")
        return

    # Average over datasets per model
    id_cols = ["model_name", "family"]
    val_cols = [s for s in strategy_keys if s in df.columns]
    df_agg = df.groupby(id_cols)[val_cols].mean().reset_index()

    x = np.arange(len(df_agg))
    width = 0.8 / len(val_cols)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strat in enumerate(val_cols):
        offset = (i - len(val_cols) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_agg[strat], width=width,
                      color=STRATEGY_COLORS.get(strat, "#aaa"),
                      label=STRATEGY_LABELS.get(strat, strat), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df_agg["model_name"], fontsize=10)
    ax.set_ylabel("Mean Accuracy (avg over datasets)", fontsize=11)
    ax.set_title(f"Model Family Comparison — {size_tier} tier", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=len(val_cols), loc="upper left")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, figures_dir, f"fig2_family_comparison_{size_tier}.pdf")


# ─────────────────────────────────────────────
# Fig 3: Accuracy vs size tier per strategy
# ─────────────────────────────────────────────
def plot_scaling_curves(
    accuracy_df: pd.DataFrame,
    families: List[str] = None,
    figures_dir: Optional[str] = None,
):
    """Line plots: accuracy vs parameter count, one line per family."""
    families = families or ["llama", "qwen", "phi", "gemma"]
    tier_order = ["1-2B", "3B", "7-9B"]
    tier_x = {"1-2B": 1.5, "3B": 3.0, "7-9B": 8.0}

    strategies = ["S0", "S3", "S5", "S6"]
    fig, axes = plt.subplots(1, len(strategies), figsize=(14, 4), sharey=True)

    for ax, strat in zip(axes, strategies):
        df = accuracy_df.reset_index()
        if strat not in df.columns:
            continue
        for fam in families:
            fam_df = df[df["family"] == fam].groupby("size_tier")[strat].mean()
            fam_df = fam_df.reindex(tier_order)
            xs = [tier_x.get(t, t) for t in fam_df.index if not np.isnan(fam_df[t])]
            ys = [fam_df[t] for t in fam_df.index if not np.isnan(fam_df[t])]
            if xs:
                ax.plot(xs, ys, marker="o", color=PALETTE_FAMILY.get(fam, "gray"),
                        label=fam.capitalize(), linewidth=2, markersize=7)
        ax.set_title(STRATEGY_LABELS.get(strat, strat).replace("\n", " "), fontsize=10)
        ax.set_xlabel("~Params (B)", fontsize=9)
        ax.set_xticks([1.5, 3.0, 8.0])
        ax.set_xticklabels(["1-2B", "3B", "7-9B"], fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    axes[0].set_ylabel("Mean Accuracy", fontsize=11)
    axes[0].legend(fontsize=8)
    fig.suptitle("Accuracy vs Model Size by Strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, figures_dir, "fig3_scaling_curves.pdf")


# ─────────────────────────────────────────────
# Fig 4: Error distribution stacked bars
# ─────────────────────────────────────────────
def plot_error_distribution(
    error_dist_df: pd.DataFrame,
    strategy: str = "S0",
    figures_dir: Optional[str] = None,
):
    """Stacked bar: error class fractions per model at baseline strategy."""
    from src.config import ERROR_CLASS_KEYS
    ec_colors = {
        "E1": "#e41a1c", "E2": "#ff7f00", "E3": "#fdbf6f",
        "E4": "#984ea3", "E5": "#4daf4a", "E6": "#377eb8", "E7": "#a65628", "EU": "#999999",
    }
    df = error_dist_df[error_dist_df["strategy"] == strategy]
    if df.empty:
        logger.warning(f"No error distribution data for strategy {strategy}")
        return

    pivot = df.pivot_table(index="model", columns="error_class", values="fraction",
                           aggfunc="mean").fillna(0)
    ec_cols = [ec for ec in ERROR_CLASS_KEYS + ["EU"] if ec in pivot.columns]
    pivot = pivot[ec_cols]

    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(pivot))
    for ec in ec_cols:
        vals = pivot[ec].values
        ax.bar(pivot.index, vals, bottom=bottom,
               color=ec_colors.get(ec, "#aaa"),
               label=ERROR_CLASS_LABELS.get(ec, ec).replace("\n", " "), alpha=0.85)
        bottom += vals

    ax.set_ylabel("Fraction of Errors", fontsize=11)
    ax.set_title(f"Error Class Distribution — {strategy} baseline", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, figures_dir, f"fig4_error_distribution_{strategy}.pdf")


# ─────────────────────────────────────────────
# Fig 5: Robustness ratio
# ─────────────────────────────────────────────
def plot_robustness_ratios(
    robustness_df: pd.DataFrame,
    figures_dir: Optional[str] = None,
):
    """Bar chart of robustness ratios across models and strategies."""
    if robustness_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    strategies = robustness_df["strategy"].unique()
    models = robustness_df["model"].unique()
    x = np.arange(len(models))
    width = 0.8 / len(strategies)

    for i, strat in enumerate(strategies):
        strat_df = robustness_df[robustness_df["strategy"] == strat]
        means = [strat_df[strat_df["model"] == m]["robustness_ratio"].mean()
                 for m in models]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width,
               color=STRATEGY_COLORS.get(strat, "#aaa"),
               label=STRATEGY_LABELS.get(strat, strat).replace("\n", " "), alpha=0.85)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Perfect robustness")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, fontsize=8)
    ax.set_ylabel("Robustness Ratio (perturbed / clean)", fontsize=11)
    ax.set_title("Robustness Ratio: Perturbed vs Clean Benchmarks", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, figures_dir, "fig5_robustness_ratios.pdf")


# ─────────────────────────────────────────────
# Fig 6: JS divergence matrix
# ─────────────────────────────────────────────
def plot_js_divergence(
    all_results: Dict,
    model_keys: List[str],
    strategy_keys: List[str],
    dataset_keys: List[str],
    model_configs: Dict,
    figures_dir: Optional[str] = None,
):
    """Heatmap of JS divergence between S0 error distribution and each strategy."""
    from src.metrics import error_distribution, js_divergence

    rows = []
    for model_key in model_keys:
        cfg = model_configs.get(model_key, {})
        # Compute S0 distribution
        s0_results = []
        for dk in dataset_keys:
            s0_results += all_results.get(model_key, {}).get("S0", {}).get(dk, [])
        s0_dist = error_distribution(s0_results)

        for strat in strategy_keys:
            if strat == "S0":
                continue
            strat_results = []
            for dk in dataset_keys:
                strat_results += all_results.get(model_key, {}).get(strat, {}).get(dk, [])
            strat_dist = error_distribution(strat_results)
            jsd = js_divergence(s0_dist, strat_dist)
            rows.append({
                "model":    cfg.get("name", model_key),
                "strategy": strat,
                "jsd":      jsd,
            })

    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="model", columns="strategy", values="jsd")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="Blues",
                vmin=0, vmax=0.5,
                cbar_kws={"label": "JS Divergence from S0"})
    ax.set_title("JS Divergence of Error Distribution vs S0 Baseline",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("ICL Strategy")
    ax.set_ylabel("Model")
    plt.tight_layout()
    _save(fig, figures_dir, "fig6_js_divergence.pdf")


# ─────────────────────────────────────────────
# Summary table (for report)
# ─────────────────────────────────────────────
def print_summary_table(accuracy_df: pd.DataFrame, strategy_keys: List[str]):
    """Print a clean accuracy summary table."""
    cols = [s for s in strategy_keys if s in accuracy_df.columns]
    summary = accuracy_df[cols].groupby(level="model_name").mean()
    print("\n=== Mean Accuracy per Model (averaged over datasets) ===")
    print(summary.to_string(float_format="{:.3f}".format))
    print()
    print("=== Mean Accuracy per Strategy (averaged over models) ===")
    print(summary.mean(axis=0).to_string(float_format="{:.3f}".format))
