"""
data_loader.py — Dataset loading and preprocessing.

Handles: GSM8K, GSM-Symbolic, GSM-Plus, GSM-IC, BBH (3 subsets), FOLIO.
All datasets are normalised into a standard Record format.
Paired datasets (for robustness ratio) share an `original_id` field.
"""

import re
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Standard record schema
# ─────────────────────────────────────────────
def make_record(
    dataset_key: str,
    item_id: str,
    question: str,
    gold_answer: str,
    original_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """Normalised record that all downstream code expects."""
    return {
        "dataset":      dataset_key,
        "id":           item_id,
        "original_id":  original_id or item_id,  # links to clean twin
        "question":     question,
        "gold_answer":  gold_answer,
        "metadata":     metadata or {},
    }


# ─────────────────────────────────────────────
# Answer extraction helpers
# ─────────────────────────────────────────────
def extract_gsm_answer(raw: str) -> Optional[str]:
    """Extract numeric answer after '####' in GSM-style answers."""
    m = re.search(r"####\s*([\-\d,\.]+)", raw)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in string
    nums = re.findall(r"[\-]?\d+(?:\.\d+)?", raw)
    return nums[-1] if nums else None


def extract_model_answer(raw: str, answer_type: str) -> Optional[str]:
    """
    Extract final answer from model output.

    answer_type:
        "numeric"   — last numeric expression found
        "choice"    — letter in parentheses e.g. (A)
        "trivalent" — True / False / Unknown
    """
    raw = raw.strip()
    if answer_type == "numeric":
        # Look for explicit "answer is X" pattern first
        m = re.search(r"(?:answer is|=)\s*([\-]?\d[\d,\.]*)", raw, re.IGNORECASE)
        if m:
            return m.group(1).replace(",", "").strip()
        # Then ####
        m = re.search(r"####\s*([\-\d,\.]+)", raw)
        if m:
            return m.group(1).replace(",", "").strip()
        # Fallback: last number
        nums = re.findall(r"[\-]?\d+(?:,\d{3})*(?:\.\d+)?", raw)
        if nums:
            return nums[-1].replace(",", "")
        return None

    elif answer_type == "choice":
        # "The answer is (A)" or just "(A)" at end
        m = re.search(r"\(([A-E])\)\s*$", raw)
        if m:
            return m.group(1)
        # Any bracketed letter
        matches = re.findall(r"\(([A-E])\)", raw)
        if matches:
            return matches[-1]
        # Plain letter at end of last line
        last_line = raw.strip().split("\n")[-1]
        m = re.match(r"^([A-E])[\.\:\s]", last_line.strip())
        if m:
            return m.group(1)
        return None

    elif answer_type == "trivalent":
        m = re.search(r"\b(True|False|Unknown)\b", raw, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        return None

    return None


def answers_match(pred: Optional[str], gold: str, answer_type: str) -> bool:
    """Check if predicted answer matches gold answer."""
    if pred is None:
        return False
    pred = pred.strip().lower().replace(",", "")
    gold = gold.strip().lower().replace(",", "")
    if answer_type == "numeric":
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            return pred == gold
    return pred == gold


# ─────────────────────────────────────────────
# Individual dataset loaders
# ─────────────────────────────────────────────
def _load_gsm8k(cfg: Dict, n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset(cfg["hf_id"], cfg.get("hf_config", "main"),
                      split=cfg["split"], cache_dir=cfg.get("cache_dir"))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    records = []
    for i, row in enumerate(ds):
        gold = extract_gsm_answer(row["answer"])
        records.append(make_record(
            dataset_key="gsm8k",
            item_id=f"gsm8k_{i}",
            question=row["question"],
            gold_answer=gold or row["answer"],
            metadata={"raw_answer": row["answer"]},
        ))
    return records


def _load_gsm_symbolic(cfg: Dict, n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset(cfg["hf_id"], cfg.get("hf_config", "main"),
                          split=cfg["split"], cache_dir=cfg.get("cache_dir"))
    except Exception as e:
        logger.warning(f"GSM-Symbolic load failed ({e}), falling back to GSM8K subset.")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    records = []
    for i, row in enumerate(ds):
        q_field = "question" if "question" in row else list(row.keys())[0]
        a_field = "answer" if "answer" in row else list(row.keys())[1]
        gold = extract_gsm_answer(str(row[a_field]))
        records.append(make_record(
            dataset_key="gsm_symbolic",
            item_id=f"gsm_sym_{i}",
            question=str(row[q_field]),
            gold_answer=gold or str(row[a_field]),
        ))
    return records


def _load_gsm_plus(cfg: Dict, n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset(cfg["hf_id"], split=cfg["split"],
                          cache_dir=cfg.get("cache_dir"))
    except Exception as e:
        logger.warning(f"GSM-Plus load failed ({e}). Skipping.")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    records = []
    for i, row in enumerate(ds):
        q_field = "question" if "question" in row else "problem"
        a_field = "answer" if "answer" in row else "solution"
        gold = extract_gsm_answer(str(row.get(a_field, "")))
        records.append(make_record(
            dataset_key="gsm_plus",
            item_id=f"gsm_plus_{i}",
            question=str(row[q_field]),
            gold_answer=gold or str(row.get(a_field, "")),
            metadata={"perturbation_type": row.get("perturbation_type", "unknown")},
        ))
    return records


def _load_gsm_ic(cfg: Dict, n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    # Try several known HF IDs for GSM-IC
    candidates = [cfg["hf_id"], "gsmic/gsm-ic", "mgsm/gsm-ic"]
    ds = None
    for hf_id in candidates:
        try:
            ds = load_dataset(hf_id, split=cfg["split"],
                              cache_dir=cfg.get("cache_dir"))
            logger.info(f"Loaded GSM-IC from {hf_id}")
            break
        except Exception:
            continue
    if ds is None:
        logger.warning("GSM-IC not found on HuggingFace. Generating synthetic distractors from GSM8K.")
        return _generate_synthetic_gsmic(n, seed, cfg.get("cache_dir"))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    records = []
    for i, row in enumerate(ds):
        q_field = next((k for k in ["question", "problem", "input"] if k in row), None)
        a_field = next((k for k in ["answer", "solution", "target"] if k in row), None)
        if not q_field or not a_field:
            continue
        gold = extract_gsm_answer(str(row[a_field]))
        records.append(make_record(
            dataset_key="gsm_ic",
            item_id=f"gsm_ic_{i}",
            question=str(row[q_field]),
            gold_answer=gold or str(row[a_field]),
        ))
    return records


def _generate_synthetic_gsmic(n: int, seed: int, cache_dir: Optional[str]) -> List[Dict]:
    """
    Fallback: load GSM8K and inject a syntactically plausible but irrelevant
    distractor sentence (simulating GSM-IC behavior).
    This is a best-effort approximation when the original dataset is unavailable.
    """
    from datasets import load_dataset
    distractors = [
        "John also bought 3 apples that were not part of the problem.",
        "Additionally, there were 7 extra items that had no bearing on the calculation.",
        "Note that last Tuesday, the shop was closed for 2 hours.",
        "Separately, the store manager counted 15 other products unrelated to this.",
        "There were also 4 customers who left without purchasing anything.",
    ]
    rng = random.Random(seed)
    try:
        ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception:
        return []
    records = []
    for i, row in enumerate(ds):
        distractor = rng.choice(distractors)
        sentences = row["question"].split(". ")
        insert_pos = rng.randint(1, max(1, len(sentences) - 1))
        sentences.insert(insert_pos, distractor)
        perturbed_q = ". ".join(sentences)
        gold = extract_gsm_answer(row["answer"])
        records.append(make_record(
            dataset_key="gsm_ic",
            item_id=f"gsm_ic_synth_{i}",
            question=perturbed_q,
            gold_answer=gold or row["answer"],
            original_id=f"gsm8k_{i}",
            metadata={"synthetic": True, "distractor": distractor},
        ))
    return records


def _load_bbh(cfg: Dict, dataset_key: str, n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset(cfg["hf_id"], cfg["hf_config"],
                      split=cfg["split"], cache_dir=cfg.get("cache_dir"))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    records = []
    for i, row in enumerate(ds):
        records.append(make_record(
            dataset_key=dataset_key,
            item_id=f"{dataset_key}_{i}",
            question=str(row["input"]),
            gold_answer=str(row["target"]).strip(),
        ))
    return records


def _load_folio(cfg: Dict, n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    candidates = [cfg["hf_id"], "tasksource/folio", "yale-nlp/FOLIO"]
    ds = None
    for hf_id in candidates:
        try:
            ds = load_dataset(hf_id, split=cfg["split"],
                              cache_dir=cfg.get("cache_dir"))
            logger.info(f"Loaded FOLIO from {hf_id}")
            break
        except Exception:
            continue
    if ds is None:
        logger.warning("FOLIO not found. Skipping.")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    records = []
    for i, row in enumerate(ds):
        # FOLIO: premises in 'story' or 'premises', hypothesis in 'question' or 'conclusion'
        if "story" in row and "question" in row:
            q = f"Premises:\n{row['story']}\n\nConclusion: {row['question']}\n\nBased on the premises, is the conclusion True, False, or Unknown?"
        elif "premises" in row:
            premises = "\n".join(row["premises"]) if isinstance(row["premises"], list) else row["premises"]
            conclusion = row.get("conclusion", row.get("hypothesis", ""))
            q = f"Premises:\n{premises}\n\nConclusion: {conclusion}\n\nBased on the premises, is the conclusion True, False, or Unknown?"
        else:
            q = str(row.get("input", ""))
        label = str(row.get("label", row.get("answer", ""))).strip().capitalize()
        if label not in ("True", "False", "Unknown"):
            label = "Unknown"
        records.append(make_record(
            dataset_key="folio",
            item_id=f"folio_{i}",
            question=q,
            gold_answer=label,
        ))
    return records


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def load_dataset_by_key(
    dataset_key: str,
    cfg_override: Optional[Dict] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load a dataset by its key (e.g. 'gsm8k', 'folio').
    Returns a list of normalised record dicts.
    """
    from src.config import DATASETS
    cfg = dict(DATASETS[dataset_key])
    if cfg_override:
        cfg.update(cfg_override)
    if cache_dir:
        cfg["cache_dir"] = cache_dir
    n = n_samples or cfg["n_samples"]

    loaders = {
        "gsm8k":                _load_gsm8k,
        "gsm_symbolic":         _load_gsm_symbolic,
        "gsm_plus":             _load_gsm_plus,
        "gsm_ic":               _load_gsm_ic,
        "bbh_logical_deduction": lambda c, n, s: _load_bbh(c, "bbh_logical_deduction", n, s),
        "bbh_tracking":          lambda c, n, s: _load_bbh(c, "bbh_tracking", n, s),
        "bbh_causal":            lambda c, n, s: _load_bbh(c, "bbh_causal", n, s),
        "folio":                _load_folio,
    }
    if dataset_key not in loaders:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    logger.info(f"Loading {dataset_key} (n={n}, seed={seed})")
    records = loaders[dataset_key](cfg, n, seed)
    logger.info(f"  → {len(records)} records loaded")
    return records


def load_all_datasets(
    dataset_keys: List[str],
    n_samples: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Load multiple datasets, return dict keyed by dataset name."""
    result = {}
    for key in dataset_keys:
        try:
            result[key] = load_dataset_by_key(key, n_samples=n_samples,
                                              seed=seed, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to load {key}: {e}")
            result[key] = []
    return result


def get_dataset_stats(datasets: Dict[str, List[Dict]]) -> Dict:
    """Quick stats summary for loaded datasets."""
    from src.config import DATASETS
    stats = {}
    for key, records in datasets.items():
        tier = DATASETS.get(key, {}).get("tier", "unknown")
        answer_type = DATASETS.get(key, {}).get("answer_type", "unknown")
        stats[key] = {
            "n": len(records),
            "tier": tier,
            "answer_type": answer_type,
            "has_paired": DATASETS.get(key, {}).get("paired_with") is not None,
        }
    return stats
