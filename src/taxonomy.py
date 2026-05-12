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


def _normalise_math_text(raw: str) -> str:
    """
    Strip LaTeX delimiters and normalise math operators so plain-text
    arithmetic regexes can match expressions inside LaTeX.

    Handles: \\[ ... \\], \\( ... \\), $ ... $, \\text{...},
             \\times / \\cdot → *, \\frac{a}{b} → a/b approximation.
    """
    # Remove LaTeX environment delimiters
    text = re.sub(r"\\\[|\\\]|\\\(|\\\)", " ", raw)
    text = re.sub(r"\$+", " ", text)
    # Normalise multiplication operators to *
    text = re.sub(r"\\times\b|\\cdot\b", "*", text)
    # Remove LaTeX commands that wrap text (e.g. \text{Total sold})
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    # Remove remaining LaTeX commands (e.g. \frac, \left, \right)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Remove LaTeX grouping braces
    text = re.sub(r"[{}]", " ", text)
    return text


def _has_arithmetic_error(result: Dict) -> bool:
    """
    E1: Detect arithmetic slips by re-evaluating simple expressions in the
    reasoning trace.  Handles plain text AND LaTeX-formatted math.

    Also catches the case where the model's stated final answer does not
    match the last intermediate computed value in its own trace — a common
    pattern when a model computes correctly but then copies the wrong number.
    """
    raw = result.get("raw_output") or ""
    # Work on both original and LaTeX-normalised text
    texts_to_check = [raw, _normalise_math_text(raw)]

    patterns = [
        # Binary: A + B = C
        (r"([\d,]+(?:\.\d+)?)\s*\+\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)",
         lambda a, b, c: abs((_n(a) + _n(b)) - _n(c)) < max(0.5, 0.001 * abs(_n(c)))),
        # Binary: A - B = C
        (r"([\d,]+(?:\.\d+)?)\s*-\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)",
         lambda a, b, c: abs((_n(a) - _n(b)) - _n(c)) < max(0.5, 0.001 * abs(_n(c)))),
        # Binary: A * B = C  (also handles × already normalised)
        (r"([\d,]+(?:\.\d+)?)\s*[x*×]\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)",
         lambda a, b, c: abs((_n(a) * _n(b)) - _n(c)) < max(0.5, 0.001 * abs(_n(c)))),
        # Division: A / B = C
        (r"([\d,]+(?:\.\d+)?)\s*/\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)",
         lambda a, b, c: _safe_div_check(a, b, c)),
        # Three-term addition: A + B + C = D
        (r"([\d,]+(?:\.\d+)?)\s*\+\s*([\d,]+(?:\.\d+)?)\s*\+\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)",
         lambda a, b, c, d=None: True),  # handled separately below
    ]

    for text in texts_to_check:
        # Three-term sums handled separately
        for m in re.finditer(
            r"([\d,]+(?:\.\d+)?)\s*\+\s*([\d,]+(?:\.\d+)?)\s*\+\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)",
            text
        ):
            try:
                a, b, c, d = m.group(1), m.group(2), m.group(3), m.group(4)
                expected = _n(a) + _n(b) + _n(c)
                if abs(expected - _n(d)) > max(0.5, 0.001 * abs(expected)):
                    return True
            except (ValueError, ZeroDivisionError):
                continue

        # Binary patterns
        for pattern, checker in patterns[:4]:
            for m in re.finditer(pattern, text):
                try:
                    a, b, c = m.group(1), m.group(2), m.group(3)
                    if not checker(a, b, c):
                        return True
                except (ValueError, ZeroDivisionError, IndexError):
                    continue

    # Extra check: pred_answer doesn't match the last intermediate result
    # shown in the trace (model computed correctly but wrote wrong final answer)
    pred = result.get("pred_answer")
    if pred is not None:
        try:
            pred_val = float(str(pred).replace(",", ""))
            # Extract all "= <number>" occurrences
            intermediates = re.findall(
                r"=\s*\$?\s*([\-]?\d[\d,]*(?:\.\d+)?)", raw
            )
            if len(intermediates) >= 2:
                try:
                    last_computed = float(intermediates[-1].replace(",", ""))
                    second_last = float(intermediates[-2].replace(",", ""))
                    # If pred differs from last computed by more than 1 but
                    # matches an earlier intermediate, that's a copy-paste slip
                    if (abs(pred_val - last_computed) > 1.0
                            and abs(pred_val - second_last) < 1.0):
                        return True
                except ValueError:
                    pass
        except (ValueError, AttributeError):
            pass

    return False


def _n(s: str) -> float:
    """Parse a numeric string (strips commas and spaces)."""
    return float(str(s).replace(",", "").strip())


def _safe_div_check(a: str, b: str, c: str) -> bool:
    """Return True (no error) if A/B ≈ C."""
    try:
        denom = _n(b)
        if denom == 0:
            return True  # skip division by zero
        return abs(_n(a) / denom - _n(c)) < max(0.1, 0.001 * abs(_n(c)))
    except (ValueError, ZeroDivisionError):
        return True


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
    """
    E4: Model skips a required intermediate step.

    Two detection modes:
    1. Short-response mode: output is short AND has few computation lines for
       a complex problem (original heuristic, relaxed thresholds).
    2. Early-stop mode: the model's predicted answer appears as an intermediate
       computed value *before* the final step, indicating the model stopped
       mid-calculation (e.g. computed monthly cost but not annual).
    """
    raw = result.get("raw_output") or ""
    question = result.get("question") or ""
    pred = result.get("pred_answer")

    q_nums = len(re.findall(r"\d+", question))
    if q_nums < 3:
        return False  # trivial problem — step skip not meaningful

    # Mode 1: Very short output with almost no computation steps
    computation_lines = re.findall(
        r"[\d,]+(?:\.\d+)?\s*[+\-×x*/]\s*[\d,]+(?:\.\d+)?\s*=", raw
    )
    if len(raw) < 600 and len(computation_lines) < 2:
        return True

    # Mode 2: pred_answer matches an intermediate (not final) computed value.
    # This catches "forgot to multiply by 12" style errors.
    if pred is not None:
        try:
            pred_val = float(str(pred).replace(",", ""))
            if abs(pred_val) < 0.001:
                return False  # zero answer not meaningful here

            # Find all "= <number>" occurrences in the output
            intermediates = re.findall(
                r"=\s*\$?\s*([\-]?\d[\d,]*(?:\.\d+)?)", raw
            )
            # If there are multiple intermediate values and pred equals one
            # of the non-final ones, the model stopped too early.
            if len(intermediates) >= 3:
                for inter in intermediates[:-2]:   # skip last two (final answer)
                    try:
                        if abs(float(inter.replace(",", "")) - pred_val) < 0.5:
                            return True
                    except ValueError:
                        continue
        except (ValueError, AttributeError):
            pass

    return False


def _has_hallucination(result: Dict) -> bool:
    """
    E5: Model introduces numerical values not present in the question and not
    derivable from question numbers via standard arithmetic.

    We are conservative here to avoid false positives (e.g. compound interest
    results that appear large but ARE derivable).  Only flag when:
    - ≥ 2 large numbers (> 500) appear in the output that are not in the question
    - AND those numbers cannot be derived by any of: +, -, *, /, % of question nums
    - AND the model makes explicit factual claims (not just calculations)
    """
    raw = result.get("raw_output") or ""
    question = result.get("question") or ""

    q_nums_str = re.findall(r"\d+(?:\.\d+)?", question)
    q_nums = set(q_nums_str)
    try:
        q_vals = [float(n) for n in q_nums_str]
    except ValueError:
        q_vals = []

    out_nums_str = re.findall(r"\b\d+(?:\.\d+)?\b", raw)
    out_nums = set(out_nums_str)

    extra = out_nums - q_nums

    # Build the set of derivable values: products, sums, diffs, ratios,
    # percentages, and squares of all pairs of question numbers.
    derivable: set = set()
    for x in q_vals:
        for y in q_vals:
            derivable.update([
                round(x + y, 2), round(x - y, 2), round(x * y, 2),
                round(x / y, 2) if y != 0 else None,
                round(x * y / 100, 2),          # percentage
                round(x + x * y / 100, 2),      # base + percent
                round(x * (1 + y / 100) ** 2, 2),  # 2-year compound
                round(x * (1 + y / 100) ** 3, 2),  # 3-year compound
            ])
        derivable.add(round(x ** 2, 2))
    derivable.discard(None)

    def _is_derivable(n_str: str) -> bool:
        try:
            val = round(float(n_str), 2)
            if val <= 0:
                return True
            return any(abs(val - d) < max(1.0, 0.01 * abs(val)) for d in derivable if d)
        except ValueError:
            return True

    # Only count large "phantom" numbers (>500) that are not derivable
    phantom = [
        n for n in extra
        if (lambda v: v > 500)(float(n)) and not _is_derivable(n)
    ]

    if len(phantom) < 2:
        return False

    # Additionally require the model makes an explicit factual assertion
    # (not just calculation).  Hallucinations typically look like
    # "the store charges X" or "originally there were Y".
    hallucination_phrases = re.search(
        r"\b(originally|initially|assume|given that|stated that|according to|"
        r"the problem says|we know that|it is known)\b",
        raw, re.IGNORECASE
    )
    return hallucination_phrases is not None


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

    # Runtime/model failures are not reasoning failures. Keep them out of the
    # substantive taxonomy so a broken model load does not masquerade as E2/E4.
    if result.get("error_msg") and not (result.get("raw_output") or "").strip():
        return "EU"

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

    # For wrong numeric answers that pass all other checks, default to E1.
    # A model that gives a wrong number after showing reasoning most likely
    # made an arithmetic or step-level error — labelling it EU hides signal.
    if answer_type == "numeric":
        return "E1"

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
