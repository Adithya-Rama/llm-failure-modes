"""
checkpointing.py — Robust checkpoint save/load for Colab.

Handles: Drive path creation, atomic writes, resume-from-partial,
         full results aggregation from checkpoint files.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _atomic_write(path: Path, data: list):
    """Write JSON atomically to avoid corrupt files on Colab disconnect."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def save_checkpoint(path: Path, results: List[Dict]):
    """Save a list of result dicts to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_write(path, results)
        logger.debug(f"Checkpoint saved: {path} ({len(results)} items)")
    except Exception as e:
        logger.error(f"Checkpoint save failed: {e}")


def load_checkpoint(path: Path) -> Tuple[List[Dict], Set[str]]:
    """
    Load checkpoint file if it exists.
    Returns (completed_results, set_of_done_ids).
    """
    path = Path(path)
    if not path.exists():
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results if "id" in r}
        logger.info(f"Resumed from {path}: {len(results)} items already done")
        return results, done_ids
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Corrupt checkpoint at {path}, starting fresh. Error: {e}")
        return [], set()


def get_checkpoint_path(
    checkpoint_dir: str,
    model_key: str,
    strategy_key: str,
    dataset_key: str,
) -> Path:
    """Standard checkpoint path for a model × strategy × dataset run."""
    return Path(checkpoint_dir) / f"{model_key}__{strategy_key}__{dataset_key}.json"


def load_all_checkpoints(checkpoint_dir: str) -> Dict:
    """
    Scan checkpoint_dir and load all completed runs.
    Returns: {model_key → {strategy_key → {dataset_key → [results]}}}
    """
    checkpoint_dir = Path(checkpoint_dir)
    all_results: Dict = {}

    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint dir not found: {checkpoint_dir}")
        return all_results

    for ckpt_file in sorted(checkpoint_dir.glob("*.json")):
        stem = ckpt_file.stem
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        model_key, strategy_key, dataset_key = parts

        results, _ = load_checkpoint(ckpt_file)
        if not results:
            continue

        if model_key not in all_results:
            all_results[model_key] = {}
        if strategy_key not in all_results[model_key]:
            all_results[model_key][strategy_key] = {}
        all_results[model_key][strategy_key][dataset_key] = results

    # Summary
    total = sum(
        len(v)
        for m in all_results.values()
        for s in m.values()
        for v in s.values()
    )
    logger.info(f"Loaded {total} total results from {checkpoint_dir}")
    return all_results


def checkpoint_status(
    checkpoint_dir: str,
    model_keys: List[str],
    strategy_keys: List[str],
    dataset_keys: List[str],
) -> Dict:
    """
    Print/return a status summary of which runs are complete.
    Returns dict with counts.
    """
    checkpoint_dir = Path(checkpoint_dir)
    status = {}
    for mk in model_keys:
        status[mk] = {}
        for sk in strategy_keys:
            status[mk][sk] = {}
            for dk in dataset_keys:
                ckpt = checkpoint_dir / f"{mk}__{sk}__{dk}.json"
                if ckpt.exists():
                    results, _ = load_checkpoint(ckpt)
                    status[mk][sk][dk] = len(results)
                else:
                    status[mk][sk][dk] = 0
    return status


def print_checkpoint_status(
    checkpoint_dir: str,
    model_keys: List[str],
    strategy_keys: List[str],
    dataset_keys: List[str],
):
    """Human-readable checkpoint status table."""
    status = checkpoint_status(checkpoint_dir, model_keys, strategy_keys, dataset_keys)
    header = f"{'Model':<20} {'Strategy':<8} " + " ".join(f"{dk:<15}" for dk in dataset_keys)
    print(header)
    print("-" * len(header))
    for mk in model_keys:
        for sk in strategy_keys:
            row = f"{mk:<20} {sk:<8} "
            row += " ".join(f"{status[mk][sk][dk]:<15}" for dk in dataset_keys)
            print(row)


def merge_results_for_analysis(checkpoint_dir: str) -> Dict:
    """Convenience: load all checkpoints into analysis-ready structure."""
    return load_all_checkpoints(checkpoint_dir)


def export_full_results(all_results: Dict, output_path: str):
    """Save the full nested results dict as a single JSON for archival."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, default=str)
    logger.info(f"Full results exported to {path}")


def filter_all_results_to_records(all_results: Dict, datasets: Dict[str, List[Dict]]) -> Dict:
    """
    Return a copy of nested results restricted to the currently loaded records.

    This keeps analysis fair when an old checkpoint contains more items than
    the active N_SAMPLES configuration.
    """
    allowed_ids = {
        dataset_key: {r["id"] for r in records}
        for dataset_key, records in datasets.items()
    }
    filtered: Dict = {}
    for model_key, model_results in all_results.items():
        filtered[model_key] = {}
        for strategy_key, strategy_results in model_results.items():
            filtered[model_key][strategy_key] = {}
            for dataset_key, results in strategy_results.items():
                allowed = allowed_ids.get(dataset_key)
                if allowed is None:
                    filtered[model_key][strategy_key][dataset_key] = results
                else:
                    filtered[model_key][strategy_key][dataset_key] = [
                        r for r in results if r.get("id") in allowed
                    ]
    return filtered


def rescore_results(results: List[Dict], dataset_key: str, recode_errors: bool = True) -> Dict:
    """
    Recompute pred_answer/correct from saved raw_output.

    Useful after improving answer extraction: expensive model generations can be
    reused while scoring, error labels, and downstream metrics are refreshed.
    Mutates the provided result dicts and returns a compact before/after summary.
    """
    from src.config import DATASETS
    from src.data_loader import extract_model_answer, answers_match
    from src.taxonomy import code_batch

    answer_type = DATASETS.get(dataset_key, {}).get("answer_type", "numeric")
    before_correct = sum(1 for r in results if r.get("correct"))
    changed = 0

    for result in results:
        old_pred = result.get("pred_answer")
        old_correct = result.get("correct")
        pred = extract_model_answer(result.get("raw_output") or "", answer_type)
        correct = answers_match(pred, result.get("gold_answer", ""), answer_type)
        result["pred_answer"] = pred
        result["correct"] = correct
        if old_pred != pred or old_correct != correct:
            changed += 1
        if recode_errors:
            result["error_class"] = None
            result.pop("judge_justification", None)

    if recode_errors:
        code_batch(results, answer_type)

    after_correct = sum(1 for r in results if r.get("correct"))
    n = len(results)
    return {
        "n": n,
        "changed": changed,
        "before_correct": before_correct,
        "after_correct": after_correct,
        "before_accuracy": before_correct / n if n else 0.0,
        "after_accuracy": after_correct / n if n else 0.0,
    }


def rescore_checkpoint(path: Path, recode_errors: bool = True, save: bool = True) -> Dict:
    """Rescore one checkpoint JSON file and optionally save it in place."""
    path = Path(path)
    results, _ = load_checkpoint(path)
    if not results:
        return {"path": str(path), "n": 0, "changed": 0}

    parts = path.stem.split("__")
    dataset_key = parts[-1] if len(parts) >= 3 else results[0].get("dataset", "")
    summary = rescore_results(results, dataset_key, recode_errors=recode_errors)
    summary["path"] = str(path)
    if save:
        save_checkpoint(path, results)
    return summary


def rescore_all_checkpoints(
    checkpoint_dir: str,
    model_keys: Optional[List[str]] = None,
    strategy_keys: Optional[List[str]] = None,
    dataset_keys: Optional[List[str]] = None,
    recode_errors: bool = True,
    save: bool = True,
) -> List[Dict]:
    """Rescore all matching checkpoint files in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    summaries = []
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint dir not found: {checkpoint_dir}")
        return summaries

    for ckpt_file in sorted(checkpoint_dir.glob("*.json")):
        parts = ckpt_file.stem.split("__")
        if len(parts) != 3:
            continue
        model_key, strategy_key, dataset_key = parts
        if model_keys and model_key not in model_keys:
            continue
        if strategy_keys and strategy_key not in strategy_keys:
            continue
        if dataset_keys and dataset_key not in dataset_keys:
            continue
        summaries.append(rescore_checkpoint(
            ckpt_file,
            recode_errors=recode_errors,
            save=save,
        ))
    return summaries
