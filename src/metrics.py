"""
metrics.py — All evaluation metrics.

Key outputs:
  accuracy_table()      — per-model × per-strategy × per-dataset accuracy
  recovery_delta()      — per-error-class × per-strategy recovery improvement over S0
  robustness_ratio()    — paired clean vs perturbed accuracy (Mirzadeh 2024 style)
  error_distribution()  — error class frequencies per model/strategy
  js_divergence()       — JS divergence between error distributions
  full_metrics_report() — combined DataFrame ready for visualisation
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Basic accuracy
# ─────────────────────────────────────────────
def accuracy(results: List[Dict]) -> float:
    """Simple exact-match accuracy."""
    if not results:
        return float("nan")
    return sum(r["correct"] for r in results) / len(results)


def accuracy_table(
    all_results: Dict,
    model_keys: List[str],
    strategy_keys: List[str],
    dataset_keys: List[str],
    model_configs: Dict,
) -> pd.DataFrame:
    """
    Build accuracy table: rows = model × dataset, cols = strategy.

    all_results: {model_key → {strategy_key → {dataset_key → [results]}}}
    Returns DataFrame with MultiIndex rows and strategy columns.
    """
    rows = []
    for model_key in model_keys:
        cfg = model_configs.get(model_key, {})
        model_name = cfg.get("name", model_key)
        family = cfg.get("family", "unknown")
        size_tier = cfg.get("size_tier", "unknown")
        for dataset_key in dataset_keys:
            row = {
                "model_key":  model_key,
                "model_name": model_name,
                "family":     family,
                "size_tier":  size_tier,
                "dataset":    dataset_key,
            }
            for strat in strategy_keys:
                results = (all_results
                           .get(model_key, {})
                           .get(strat, {})
                           .get(dataset_key, []))
                row[strat] = accuracy(results)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index(["model_name", "size_tier", "family", "dataset"])
    return df


# ─────────────────────────────────────────────
# Recovery delta
# ─────────────────────────────────────────────
def recovery_delta(
    all_results: Dict,
    model_keys: List[str],
    strategy_keys: List[str],
    dataset_keys: List[str],
    model_configs: Dict,
    baseline_strategy: str = "S0",
) -> pd.DataFrame:
    """
    Per-error-class × per-strategy recovery rate (Δ accuracy over S0 baseline).

    For each error class Ek:
      recovery(Sk, Ek) = fraction of items coded Ek in S0 that are now correct in Sk

    Returns DataFrame: rows = error_class, cols = strategy (S1–S6), values = mean Δ.
    """
    from src.config import ERROR_CLASS_KEYS

    # Collect per-error-class accuracy for each strategy
    # Structure: {error_class → {strategy → [0/1 correct]}}
    ec_strategy_correct: Dict[str, Dict[str, List[int]]] = {
        ec: {s: [] for s in strategy_keys if s != baseline_strategy}
        for ec in ERROR_CLASS_KEYS + ["EU"]
    }

    for model_key in model_keys:
        model_results = all_results.get(model_key, {})
        # Get baseline results with error codes
        baseline_all = {}
        for dataset_key in dataset_keys:
            b_results = model_results.get(baseline_strategy, {}).get(dataset_key, [])
            for r in b_results:
                baseline_all[r["id"]] = r

        # For each non-baseline strategy, look up which error class the item had in baseline
        for strat in strategy_keys:
            if strat == baseline_strategy:
                continue
            for dataset_key in dataset_keys:
                strat_results = model_results.get(strat, {}).get(dataset_key, [])
                for r in strat_results:
                    baseline_r = baseline_all.get(r["id"])
                    if baseline_r is None:
                        continue
                    ec = baseline_r.get("error_class")
                    if ec is None or ec == "E0":
                        continue
                    if ec not in ec_strategy_correct:
                        ec_strategy_correct[ec] = {}
                    if strat not in ec_strategy_correct[ec]:
                        ec_strategy_correct[ec][strat] = []
                    ec_strategy_correct[ec][strat].append(int(r["correct"]))

    # Compute mean recovery per error class × strategy
    rows = []
    for ec in ERROR_CLASS_KEYS:
        row = {"error_class": ec}
        for strat in strategy_keys:
            if strat == baseline_strategy:
                continue
            vals = ec_strategy_correct[ec].get(strat, [])
            row[strat] = np.nanmean(vals) if vals else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("error_class")
    return df


def recovery_heatmap_data(
    recovery_df: pd.DataFrame,
    model_configs: Dict,
) -> pd.DataFrame:
    """
    Prepare the main heatmap DataFrame.
    rows = error class (E1–E7), cols = ICL strategy (S1–S6).
    Values = mean recovery rate across all models.
    """
    from src.config import ERROR_CLASSES, ICL_STRATEGIES
    # Rename indices
    ec_names = {k: v["name"] for k, v in ERROR_CLASSES.items()
                if k in recovery_df.index}
    strat_names = {k: v["name"].split("(")[0].strip() for k, v in ICL_STRATEGIES.items()
                   if k in recovery_df.columns}
    df = recovery_df.rename(index=ec_names, columns=strat_names)
    return df


# ─────────────────────────────────────────────
# Robustness ratio (paired)
# ─────────────────────────────────────────────
def robustness_ratio(
    clean_results: List[Dict],
    perturbed_results: List[Dict],
) -> float:
    """
    Mirzadeh-style robustness ratio: acc_perturbed / acc_clean.
    Requires paired items (matching original_id).
    """
    clean_by_id = {r["original_id"]: r for r in clean_results}
    paired_clean, paired_perturbed = [], []
    for r in perturbed_results:
        orig_id = r.get("original_id", r["id"])
        if orig_id in clean_by_id:
            paired_clean.append(clean_by_id[orig_id])
            paired_perturbed.append(r)

    if not paired_clean:
        logger.warning("No paired items found for robustness ratio.")
        return float("nan")

    acc_clean = accuracy(paired_clean)
    acc_perturbed = accuracy(paired_perturbed)
    ratio = acc_perturbed / acc_clean if acc_clean > 0 else float("nan")
    return ratio


def robustness_table(
    all_results: Dict,
    model_keys: List[str],
    strategy_keys: List[str],
    paired_datasets: Dict[str, str],
    model_configs: Dict,
) -> pd.DataFrame:
    """
    Build robustness ratio table for all paired datasets.
    paired_datasets: {perturbed_key → clean_key}
    """
    rows = []
    for model_key in model_keys:
        cfg = model_configs.get(model_key, {})
        for strat in strategy_keys:
            strat_results = all_results.get(model_key, {}).get(strat, {})
            for perturbed_key, clean_key in paired_datasets.items():
                clean = strat_results.get(clean_key, [])
                perturbed = strat_results.get(perturbed_key, [])
                ratio = robustness_ratio(clean, perturbed)
                rows.append({
                    "model":        cfg.get("name", model_key),
                    "family":       cfg.get("family", ""),
                    "size_tier":    cfg.get("size_tier", ""),
                    "strategy":     strat,
                    "clean_dataset":     clean_key,
                    "perturbed_dataset": perturbed_key,
                    "robustness_ratio":  ratio,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Error distribution & JS divergence
# ─────────────────────────────────────────────
def error_distribution(results: List[Dict]) -> Dict[str, float]:
    """Return normalised error class distribution (excludes E0 correct)."""
    from src.config import ERROR_CLASS_KEYS
    counts = {ec: 0 for ec in ERROR_CLASS_KEYS + ["EU"]}
    total = 0
    for r in results:
        ec = r.get("error_class")
        if ec and ec != "E0":
            counts[ec] = counts.get(ec, 0) + 1
            total += 1
    if total == 0:
        return {ec: 0.0 for ec in counts}
    return {ec: v / total for ec, v in counts.items()}


def js_divergence(dist_p: Dict[str, float], dist_q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two error distributions."""
    from scipy.spatial.distance import jensenshannon
    keys = sorted(set(dist_p) | set(dist_q))
    p = np.array([dist_p.get(k, 0.0) for k in keys])
    q = np.array([dist_q.get(k, 0.0) for k in keys])
    # Normalise
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    return float(jensenshannon(p, q))


# ─────────────────────────────────────────────
# Statistical significance
# ─────────────────────────────────────────────
def mcnemar_test(
    results_a: List[Dict],
    results_b: List[Dict],
) -> Tuple[float, float]:
    """
    McNemar's test for paired binary outcomes.
    Returns (chi2_statistic, p_value).
    Assumes results_a and results_b are aligned (same items, same order).
    """
    from scipy.stats import chi2
    n_01, n_10 = 0, 0
    for a, b in zip(results_a, results_b):
        ca, cb = a["correct"], b["correct"]
        if not ca and cb:
            n_01 += 1
        elif ca and not cb:
            n_10 += 1
    # Continuity-corrected McNemar
    denom = n_01 + n_10
    if denom == 0:
        return 0.0, 1.0
    chi2_stat = (abs(n_01 - n_10) - 1) ** 2 / denom
    p_val = float(1 - chi2.cdf(chi2_stat, df=1))
    return chi2_stat, p_val


# ─────────────────────────────────────────────
# Full metrics report
# ─────────────────────────────────────────────
def full_metrics_report(
    all_results: Dict,
    model_keys: List[str],
    strategy_keys: List[str],
    dataset_keys: List[str],
    model_configs: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all metrics and return as dict of DataFrames.
    Keys: 'accuracy', 'recovery', 'robustness', 'error_dist'
    """
    from src.config import DATASETS

    logger.info("Computing accuracy table...")
    acc_df = accuracy_table(all_results, model_keys, strategy_keys,
                            dataset_keys, model_configs)

    logger.info("Computing recovery delta...")
    rec_df = recovery_delta(all_results, model_keys, strategy_keys,
                            dataset_keys, model_configs)

    logger.info("Computing robustness ratios...")
    paired = {k: DATASETS[k]["paired_with"]
              for k in dataset_keys
              if DATASETS.get(k, {}).get("paired_with")}
    rob_df = robustness_table(all_results, model_keys, strategy_keys,
                              paired, model_configs)

    logger.info("Computing error distributions...")
    err_rows = []
    for model_key in model_keys:
        cfg = model_configs.get(model_key, {})
        for strat in strategy_keys:
            for dataset_key in dataset_keys:
                results = (all_results
                           .get(model_key, {})
                           .get(strat, {})
                           .get(dataset_key, []))
                dist = error_distribution(results)
                for ec, frac in dist.items():
                    err_rows.append({
                        "model":    cfg.get("name", model_key),
                        "strategy": strat,
                        "dataset":  dataset_key,
                        "error_class": ec,
                        "fraction": frac,
                    })
    err_df = pd.DataFrame(err_rows)

    return {
        "accuracy":   acc_df,
        "recovery":   rec_df,
        "robustness": rob_df,
        "error_dist": err_df,
    }


def save_metrics(metrics: Dict[str, pd.DataFrame], output_dir: str):
    """Save all metric DataFrames to CSV."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in metrics.items():
        df.to_csv(out / f"{name}.csv")
    logger.info(f"Metrics saved to {output_dir}")
