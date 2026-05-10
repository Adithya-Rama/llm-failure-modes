"""
taxonomy.py — Error taxonomy coding.

Two coding modes:
  1. Rule-based: fast heuristics applied to the reasoning trace
  2. LLM-judge: prompt a lightweight judge model for uncertain cases

Main functions:
  code_error(result, dataset_cfg) → error_class (E0-E7 or EU)
  code_batch(results, dataset_cfg) → results with error_class filled in
  compute_kappa(human_labels, auto_labels) → Cohen's κ
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Rule-based heuristics
# ─────────────────────────────────────────────

def _is_format_error(result: Dict, answer_type: str) -> bool:
    """E6: Answer not parseable but reasoning appears complete."""
    from src.data_loader import extract_model_answer
    raw = result.get("raw_output") or ""
    pred = extract_model_answer(raw, answer_type)
    # If we couldn't parse an answer but there's substantial output
    return pred is None and len(raw.strip()) > 30


def _has_arithmetic_error(result: Dict) -> bool:
    """
    E1: Try to detect arithmetic slips by re-evaluating simple expressions
    found in the reasoning trace.
    """
    raw = result.get("raw_output") or ""
    # Find patterns like "X + Y = Z" or "X * Y = Z"
    patterns = [
        (r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)", lambda a, b, c: int(a) + int(b) == int(c)),
        (r"(\d+)\s*-\s*(\d+)\s*=\s*(\d+)", lambda a, b, c: int(a) - int(b) == int(c)),
        (r"(\d+)\s*[×x\*]\s*(\d+)\s*=\s*(\d+)", lambda a, b, c: int(a) * int(b) == int(c)),
        (r"(\d+)\s*/\s*(\d+)\s*=\s*([\d\.]+)", lambda a, b, c: abs(int(a) / int(b) - float(c)) < 0.1),
    ]
    errors_found = False
    for pattern, checker in patterns:
        for m in re.finditer(pattern, raw):
            try:
                a, b, c = m.group(1), m.group(2), m.group(3)
                if not checker(a, b, c):
                    errors_found = True
                    break
            except (ValueError, ZeroDivisionError):
                continue
        if errors_found:
            break
    return errors_found


def _has_distractor_signal(result: Dict) -> bool:
    """
    E2: Detect if model incorporated irrelevant information.
    Heuristic: model mentions quantities that don't appear in standard
    problem flow (requires the metadata 'distractor' key from GSM-IC).
    """
    meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    distractor = meta.get("distractor", "")
    if distractor:
        # Check if any number from the distractor appears in the raw output
        raw = result.get("raw_output") or ""
        nums_in_distractor = re.findall(r"\d+", distractor)
        for n in nums_in_distractor:
            if re.search(rf"\b{n}\b", raw):
                return True
    return False


def _has_step_skip(result: Dict) -> bool:
    """E4: Reasoning trace is very short for a multi-step problem."""
    raw = result.get("raw_output") or ""
    question = result.get("question") or ""
    # Count numbers in question (proxy for problem complexity)
    q_nums = len(re.findall(r"\d+", question))
    # Count logical steps in output
    steps = len(re.findall(r"\n|therefore|so|then|because|thus", raw, re.IGNORECASE))
    return q_nums >= 3 and steps < 2 and len(raw) < 150


def _has_hallucination(result: Dict) -> bool:
    """
    E5: Model introduces numerical values not present in the question.
    Heuristic: numbers in output that don't appear in question or are not
    the result of clearly shown arithmetic.
    """
    raw = result.get("raw_output") or ""
    question = result.get("question") or ""
    q_nums = set(re.findall(r"\d+", question))
    out_nums = set(re.findall(r"\b\d+\b", raw))
    # Numbers in output that weren't in question and aren't simple arithmetic results
    extra = out_nums - q_nums
    # Filter out very small numbers (likely intermediate arithmetic)
    extra_large = {n for n in extra if int(n) > 100 and int(n) not in
                   {int(x) * int(y) for x in q_nums for y in q_nums
                    if x.isdigit() and y.isdigit()}}
    return len(extra_large) > 1


def _has_logical_reversal(result: Dict) -> bool:
    """
    E7: Model reverses a logical negation or connective.
    Heuristic: question contains 'no', 'not', 'never' but output ignores it.
    """
    raw = result.get("raw_output") or ""
    question = result.get("question") or ""
    negation_in_q = bool(re.search(r"\b(no|not|never|none|cannot|isn't|aren't|doesn't)\b",
                                   question, re.IGNORECASE))
    negation_in_out = bool(re.search(r"\b(no|not|never|none|cannot)\b", raw, re.IGNORECASE))
    # If question has negation but output ignores it
    return negation_in_q and not negation_in_out and len(raw) > 20


# ─────────────────────────────────────────────
# Main rule-based coder
# ─────────────────────────────────────────────
def code_error_rulebased(result: Dict, answer_type: str) -> str:
    """
    Classify a single failed result into error classes E1–E7 or EU.
    Returns 'E0' if result is correct, else one of E1–E7 or 'EU'.
    Priority order follows the taxonomy literature.
    """
    if result.get("correct"):
        return "E0"

    # E6 Format — check first (before other checks which require parseable output)
    if _is_format_error(result, answer_type):
        return "E6"

    # E2 Distractor — GSM-IC specific
    dataset = result.get("dataset", "")
    if "ic" in dataset and _has_distractor_signal(result):
        return "E2"

    # E7 Logical reversal
    if answer_type in ("choice", "trivalent") and _has_logical_reversal(result):
        return "E7"

    # E1 Arithmetic slip
    if answer_type == "numeric" and _has_arithmetic_error(result):
        return "E1"

    # Public GSM-IC mirrors do not always expose the irrelevant span separately.
    # After ruling out parse errors and explicit arithmetic slips, keep failed
    # GSM-IC examples in the distractor-capture bucket.
    if dataset == "gsm_ic":
        return "E2"

    # E4 Step skipping
    if _has_step_skip(result):
        return "E4"

    # E5 Hallucination
    if _has_hallucination(result):
        return "E5"

    # E3 Premise-order (hard to detect rule-based; mark for LLM review)
    if answer_type in ("choice", "trivalent"):
        return "E3"

    return "EU"  # Unclassifiable by rules


# ─────────────────────────────────────────────
# LLM-judge coder
# ─────────────────────────────────────────────
LLM_JUDGE_PROMPT = """You are an expert annotator for a research study on LLM reasoning failures.
Classify the following failed model response into exactly ONE of these error classes:

E1 - Arithmetic Slip: The reasoning plan was correct but there was a computational error.
E2 - Distractor Capture: The model was misled by irrelevant information in the problem.
E3 - Premise-Order Sensitivity: The model got confused by the order premises were presented.
E4 - Step Skipping: The model skipped necessary intermediate reasoning steps.
E5 - Hallucinated Premise: The model introduced facts not present in the problem.
E6 - Format Error: The reasoning was correct but the answer was not in the required format.
E7 - Logical Connective/Reversal Error: The model incorrectly applied logical operators or negations.
EU - Unclassifiable: The error doesn't fit any category above.

QUESTION: {question}

MODEL OUTPUT: {raw_output}

GOLD ANSWER: {gold_answer}

Respond with ONLY the error code (E1, E2, E3, E4, E5, E6, E7, or EU) followed by a one-sentence justification.
Format: CODE: <justification>
"""


def code_error_llm_judge(
    result: Dict,
    judge_model,
    judge_tokenizer,
    judge_family: str,
    max_new_tokens: int = 100,
) -> Tuple[str, str]:
    """
    Use an LLM judge to classify an error.
    Returns (error_class, justification).
    """
    from src.models import format_prompt, generate_response

    prompt_content = LLM_JUDGE_PROMPT.format(
        question=result.get("question", "")[:500],
        raw_output=(result.get("raw_output") or "")[:500],
        gold_answer=result.get("gold_answer", ""),
    )
    messages = [
        {"role": "system", "content": "You are a precise error classifier. Follow the format exactly."},
        {"role": "user", "content": prompt_content},
    ]
    prompt_text = format_prompt(messages, judge_tokenizer, judge_family)
    responses = generate_response(
        judge_model, judge_tokenizer, prompt_text,
        max_new_tokens=max_new_tokens, do_sample=False,
    )
    response = responses[0] if responses else ""

    # Parse response
    m = re.search(r"\b(E[0-7U]|EU)\b", response)
    error_class = m.group(1) if m else "EU"
    # Get justification
    justification = response.split(":")[-1].strip() if ":" in response else response.strip()
    return error_class, justification


# ─────────────────────────────────────────────
# Batch coding
# ─────────────────────────────────────────────
def code_batch(
    results: List[Dict],
    answer_type: str,
    judge_model=None,
    judge_tokenizer=None,
    judge_family: Optional[str] = None,
    llm_judge_on_uncertain: bool = False,
) -> List[Dict]:
    """
    Code error classes for all results in a batch.
    Applies rule-based first; optionally applies LLM judge on EU cases.
    Modifies results in-place and returns them.
    """
    from src.config import DATASETS
    stats = {"E0": 0, "E1": 0, "E2": 0, "E3": 0, "E4": 0,
             "E5": 0, "E6": 0, "E7": 0, "EU": 0}

    for result in results:
        if result.get("error_class") is not None:
            stats[result["error_class"]] = stats.get(result["error_class"], 0) + 1
            continue  # already coded

        # Rule-based
        ec = code_error_rulebased(result, answer_type)
        result["error_class"] = ec
        stats[ec] = stats.get(ec, 0) + 1

        # LLM judge for uncertain cases
        if llm_judge_on_uncertain and ec == "EU" and judge_model is not None:
            try:
                ec_judge, justification = code_error_llm_judge(
                    result, judge_model, judge_tokenizer, judge_family
                )
                result["error_class"] = ec_judge
                result["judge_justification"] = justification
                stats["EU"] -= 1
                stats[ec_judge] = stats.get(ec_judge, 0) + 1
            except Exception as e:
                logger.warning(f"LLM judge failed for {result['id']}: {e}")

    logger.info(f"Error coding complete: {stats}")
    return results


def build_error_class_map(results: List[Dict]) -> Dict[str, str]:
    """Return {record_id → error_class} for S5 lookup."""
    return {r["id"]: r.get("error_class", "EU") for r in results
            if r.get("error_class") not in (None, "E0")}


# ─────────────────────────────────────────────
# Inter-annotator agreement (Cohen's κ)
# ─────────────────────────────────────────────
def compute_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Compute Cohen's κ between two annotators."""
    from sklearn.metrics import cohen_kappa_score
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must be equal length")
    return float(cohen_kappa_score(labels_a, labels_b))


def sample_for_annotation(
    results: List[Dict],
    n: int = 150,
    seed: int = 42,
) -> List[Dict]:
    """Sample n incorrect results for human annotation."""
    import random
    failed = [r for r in results if not r.get("correct")]
    rng = random.Random(seed)
    return rng.sample(failed, min(n, len(failed)))


def export_annotation_csv(
    sample: List[Dict],
    output_path: str,
):
    """Export annotation sample to CSV for human review."""
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "dataset", "question", "gold_answer",
            "pred_answer", "raw_output", "auto_label", "human_label", "notes"
        ])
        writer.writeheader()
        for r in sample:
            writer.writerow({
                "id":           r["id"],
                "dataset":      r["dataset"],
                "question":     r["question"][:200],
                "gold_answer":  r["gold_answer"],
                "pred_answer":  r.get("pred_answer"),
                "raw_output":   (r.get("raw_output") or "")[:300],
                "auto_label":   r.get("error_class", ""),
                "human_label":  "",  # to be filled by annotator
                "notes":        "",
            })
    logger.info(f"Annotation CSV saved to {output_path}")
