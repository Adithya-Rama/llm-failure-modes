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
