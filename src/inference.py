"""
inference.py — Core inference engine.

Responsibilities:
  - Run a model on a list of records under a given ICL strategy
  - Handle self-consistency (S6) majority voting
  - Checkpoint every N items to Drive
  - Resume from existing checkpoint on Colab reconnect
  - OOM recovery with graceful degradation
"""

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Answer majority vote (for S6 self-consistency)
# ─────────────────────────────────────────────
def majority_vote(answers: List[Optional[str]], answer_type: str) -> Optional[str]:
    """Return the most common non-None answer."""
    from collections import Counter
    from src.data_loader import answers_match

    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    # For numeric: cluster near-equal answers together
    if answer_type == "numeric":
        try:
            nums = [float(a.replace(",", "")) for a in valid]
            # Round to avoid float drift
            rounded = [round(n, 4) for n in nums]
            c = Counter(rounded)
            best_val = c.most_common(1)[0][0]
            return str(int(best_val) if best_val == int(best_val) else best_val)
        except ValueError:
            pass

    c = Counter(valid)
    return c.most_common(1)[0][0]


# ─────────────────────────────────────────────
# Single-record inference
# ─────────────────────────────────────────────
def run_single(
    model,
    tokenizer,
    model_cfg: Dict,
    record: Dict,
    strategy_key: str,
    error_class: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Run inference on a single record.
    Returns a result dict with keys: id, strategy, raw_output(s), pred_answer, correct.
    """
    from src.config import ICL_STRATEGIES, DATASETS
    from src.prompts import build_prompt
    from src.models import format_prompt, generate_response
    from src.data_loader import extract_model_answer, answers_match

    dataset_cfg = DATASETS.get(record["dataset"], {})
    answer_type = dataset_cfg.get("answer_type", "numeric")
    family = model_cfg["family"]
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    if answer_type == "trivalent":
        # FOLIO-style prompts are long, especially under few-shot CoT. A 256
        # token cap often truncates before the True/False/Unknown verdict.
        max_new_tokens = max(max_new_tokens, 384)

    strategy = ICL_STRATEGIES[strategy_key]

    try:
        messages = build_prompt(record, strategy_key, answer_type,
                                error_class=error_class, seed=seed)
        prompt_text = format_prompt(messages, tokenizer, family)

        if strategy_key == "S6":
            # Self-consistency: n=5 samples
            n_samples = strategy.get("n_samples", 5)
            temperature = strategy.get("temperature", 0.7)
            # Generate sequentially to keep memory bounded on T4/L4 Colab GPUs.
            # num_return_sequences=n_samples multiplies KV-cache memory.
            raw_outputs = []
            for _ in range(n_samples):
                raw_outputs.extend(generate_response(
                    model, tokenizer, prompt_text,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    n_return=1,
                ))
            preds = [extract_model_answer(r, answer_type) for r in raw_outputs]
            pred_answer = majority_vote(preds, answer_type)
            raw_output = raw_outputs[0]  # store first for taxonomy coding
        else:
            raw_outputs = generate_response(
                model, tokenizer, prompt_text,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            raw_output = raw_outputs[0]
            pred_answer = extract_model_answer(raw_output, answer_type)

        correct = answers_match(pred_answer, record["gold_answer"], answer_type)

        return {
            "id":           record["id"],
            "original_id":  record.get("original_id", record["id"]),
            "dataset":      record["dataset"],
            "strategy":     strategy_key,
            "question":     record["question"],
            "gold_answer":  record["gold_answer"],
            "metadata":     record.get("metadata", {}),
            "pred_answer":  pred_answer,
            "correct":      correct,
            "raw_output":   raw_output,
            "error_class":  None,  # filled by taxonomy coder
            "error_msg":    None,
        }

    except torch.cuda.OutOfMemoryError:
        logger.error(f"OOM on record {record['id']} — clearing cache and skipping.")
        torch.cuda.empty_cache()
        return _error_result(record, strategy_key, "OOM")
    except Exception as e:
        logger.error(f"Error on record {record['id']}: {e}")
        return _error_result(record, strategy_key, str(e))


def _error_result(record: Dict, strategy_key: str, error_msg: str) -> Dict:
    return {
        "id":          record["id"],
        "original_id": record.get("original_id", record["id"]),
        "dataset":     record["dataset"],
        "strategy":    strategy_key,
        "question":    record["question"],
        "gold_answer": record["gold_answer"],
        "metadata":    record.get("metadata", {}),
        "pred_answer": None,
        "correct":     False,
        "raw_output":  None,
        "error_class": None,
        "error_msg":   error_msg,
    }


# ─────────────────────────────────────────────
# Batch inference with checkpointing
# ─────────────────────────────────────────────
def run_experiment(
    model,
    tokenizer,
    model_cfg: Dict,
    model_key: str,
    records: List[Dict],
    strategy_key: str,
    checkpoint_dir: str,
    checkpoint_every: int = 50,
    error_classes: Optional[Dict[str, str]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run a full experiment: model × dataset × strategy.

    error_classes: {record_id → error_class} for S5 (from baseline coding).
                   None for all other strategies.

    Returns list of result dicts (one per record).
    Saves checkpoints to checkpoint_dir/model_key/strategy_key/dataset_key.json
    """
    from src.checkpointing import load_checkpoint, save_checkpoint
    from tqdm import tqdm

    if not records:
        return []

    dataset_key = records[0]["dataset"]
    ckpt_key = f"{model_key}__{strategy_key}__{dataset_key}"
    ckpt_path = Path(checkpoint_dir) / f"{ckpt_key}.json"

    # Resume from checkpoint
    completed, done_ids = load_checkpoint(ckpt_path)
    active_ids = {r["id"] for r in records}
    completed = [r for r in completed if r.get("id") in active_ids]
    done_ids = {r["id"] for r in completed if "id" in r}
    remaining = [r for r in records if r["id"] not in done_ids]
    logger.info(
        f"[{model_cfg['name']} | {strategy_key} | {dataset_key}] "
        f"{len(completed)} done, {len(remaining)} remaining"
    )

    if not remaining:
        return completed

    iterator = tqdm(remaining, desc=f"{model_cfg['name']}|{strategy_key}|{dataset_key}",
                    disable=not verbose)
    batch = []

    for record in iterator:
        error_class = None
        if error_classes and strategy_key in ("S5", "S5_RANDOM", "S5_CORRECT_ONLY"):
            error_class = error_classes.get(record["id"])

        result = run_single(model, tokenizer, model_cfg, record,
                            strategy_key, error_class=error_class, seed=seed)
        completed.append(result)
        batch.append(result)

        if len(batch) >= checkpoint_every:
            save_checkpoint(ckpt_path, completed)
            batch = []
            if verbose:
                n_correct = sum(r["correct"] for r in completed)
                acc = n_correct / len(completed)
                iterator.set_postfix({"acc": f"{acc:.3f}"})

    # Final save
    if batch:
        save_checkpoint(ckpt_path, completed)

    return completed


# ─────────────────────────────────────────────
# Multi-strategy runner
# ─────────────────────────────────────────────
def run_all_strategies(
    model,
    tokenizer,
    model_cfg: Dict,
    model_key: str,
    datasets: Dict[str, List[Dict]],
    strategy_keys: List[str],
    checkpoint_dir: str,
    checkpoint_every: int = 50,
    baseline_errors: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run all strategy × dataset combinations for one model.

    baseline_errors: {dataset_key → {record_id → error_class}}
                     required if "S5" is in strategy_keys

    Returns: {strategy_key → {dataset_key → [result dicts]}}
    """
    all_results = {}

    for strategy_key in strategy_keys:
        all_results[strategy_key] = {}
        for dataset_key, records in datasets.items():
            if not records:
                continue
            ec_map = None
            if baseline_errors and strategy_key in ("S5", "S5_RANDOM", "S5_CORRECT_ONLY"):
                ec_map = baseline_errors.get(dataset_key, {})

            results = run_experiment(
                model, tokenizer, model_cfg, model_key,
                records, strategy_key,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                error_classes=ec_map,
                seed=seed,
                verbose=verbose,
            )
            all_results[strategy_key][dataset_key] = results
            logger.info(
                f"  {strategy_key} | {dataset_key}: "
                f"acc={_acc(results):.3f} ({len(results)} items)"
            )

    return all_results


def _acc(results: List[Dict]) -> float:
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


# ─────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────
class Timer:
    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self.start
        logger.info(f"[{self.label}] {elapsed:.1f}s")
        self.elapsed = elapsed
