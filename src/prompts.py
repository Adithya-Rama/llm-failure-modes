"""
prompts.py — Prompt construction for all 7 ICL strategies.

Key structures:
  build_prompt(record, strategy_key, error_class, exemplar_bank, answer_type)
    → returns a messages list ready for format_prompt()

  EXEMPLAR_BANK: hand-crafted (error, correction) pairs for each of the
    7 error classes, used exclusively by the novel S5 strategy.
"""

from typing import List, Dict, Optional

# ─────────────────────────────────────────────
# System prompts per answer type
# ─────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "numeric": (
        "You are a careful math reasoning assistant. "
        "Work through problems step by step. "
        "End your response with 'The answer is [number].' on a new line."
    ),
    "choice": (
        "You are a careful logical reasoning assistant. "
        "Work through problems step by step. "
        "End your response with the answer in parentheses, e.g. (A)."
    ),
    "trivalent": (
        "You are a careful logical reasoning assistant. "
        "You will be given premises and a conclusion. "
        "Determine if the conclusion is True, False, or Unknown based only on the premises. "
        "End your response with exactly one of: True, False, or Unknown."
    ),
}


# ─────────────────────────────────────────────
# Standard few-shot exemplar pools (correct examples, no errors)
# Used by S2, S3, S4
# ─────────────────────────────────────────────
STANDARD_EXEMPLARS = {
    "numeric": [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4 more. How many eggs does she have left over each day?",
            "answer_only": "9",
            "cot": (
                "Janet starts with 16 eggs per day.\n"
                "She eats 3 for breakfast: 16 - 3 = 13.\n"
                "She uses 4 for muffins: 13 - 4 = 9.\n"
                "The answer is 9."
            ),
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer_only": "3",
            "cot": (
                "Blue fiber: 2 bolts.\n"
                "White fiber: half of 2 = 1 bolt.\n"
                "Total: 2 + 1 = 3 bolts.\n"
                "The answer is 3."
            ),
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer_only": "70000",
            "cot": (
                "Original value: $80,000.\n"
                "After 150% increase: $80,000 × 2.5 = $200,000.\n"
                "Total cost: $80,000 + $50,000 = $130,000.\n"
                "Profit: $200,000 - $130,000 = $70,000.\n"
                "The answer is 70000."
            ),
        },
        {
            "question": "There are 15 trees in the grove. Grove workers will plant trees today. After they are done, there will be 21 trees. How many trees did the workers plant today?",
            "answer_only": "6",
            "cot": (
                "Start: 15 trees. End: 21 trees.\n"
                "Trees planted: 21 - 15 = 6.\n"
                "The answer is 6."
            ),
        },
        {
            "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "answer_only": "39",
            "cot": (
                "Total chocolates: 32 + 42 = 74.\n"
                "After eating 35: 74 - 35 = 39.\n"
                "The answer is 39."
            ),
        },
    ],
    "choice": [
        {
            "question": (
                "The following are five objects: a purple ball, a red cube, a blue cylinder, a yellow cone, and a green sphere.\n"
                "The red cube is to the left of the purple ball. The blue cylinder is to the right of the green sphere.\n"
                "The yellow cone is between the red cube and the blue cylinder.\n"
                "What is the order of objects from left to right?\n"
                "(A) Red cube, yellow cone, purple ball, green sphere, blue cylinder\n"
                "(B) Red cube, purple ball, yellow cone, green sphere, blue cylinder\n"
                "(C) Green sphere, red cube, yellow cone, purple ball, blue cylinder\n"
                "(D) Red cube, yellow cone, green sphere, purple ball, blue cylinder\n"
                "(E) Green sphere, blue cylinder, red cube, yellow cone, purple ball"
            ),
            "answer_only": "C",
            "cot": (
                "Let me work through the constraints:\n"
                "- Red cube is left of purple ball.\n"
                "- Blue cylinder is right of green sphere.\n"
                "- Yellow cone is between red cube and blue cylinder.\n"
                "Order: Green sphere, red cube, yellow cone, purple ball, blue cylinder.\n"
                "The answer is (C)."
            ),
        },
    ],
    "trivalent": [
        {
            "question": (
                "Premises:\n"
                "All birds can fly. Penguins are birds. Tweety is a penguin.\n\n"
                "Conclusion: Tweety can fly.\n\n"
                "Based on the premises, is the conclusion True, False, or Unknown?"
            ),
            "answer_only": "True",
            "cot": (
                "From the premises: All birds can fly, and Tweety is a bird (penguin is a bird).\n"
                "Therefore, Tweety can fly.\n"
                "True"
            ),
        },
        {
            "question": (
                "Premises:\n"
                "Some cats are black. All black animals are nocturnal. Felix is a cat.\n\n"
                "Conclusion: Felix is nocturnal.\n\n"
                "Based on the premises, is the conclusion True, False, or Unknown?"
            ),
            "answer_only": "Unknown",
            "cot": (
                "We know some cats are black, but not all. Felix is a cat, but we don't know if Felix is black.\n"
                "If Felix is black, then Felix is nocturnal. But we cannot determine this from the premises.\n"
                "Unknown"
            ),
        },
    ],
}


# ─────────────────────────────────────────────
# ERROR-TARGETED EXEMPLAR BANK (S5 — novel contribution)
# Format: {error_class: [{question, wrong_reasoning, correct_reasoning, gold_answer}]}
# These are used to show the model both the error AND its correction.
# ─────────────────────────────────────────────
EXEMPLAR_BANK: Dict[str, List[Dict]] = {

    "E1": [  # Arithmetic Slip
        {
            "question": "A store had 140 apples. They sold 37 in the morning and 48 in the afternoon. How many apples remain?",
            "error_demonstration": (
                "Apples sold: 37 + 48 = 84. [ARITHMETIC ERROR: 37+48=85, not 84]\n"
                "Remaining: 140 - 84 = 56. This is WRONG due to the arithmetic slip."
            ),
            "correct_reasoning": (
                "Apples sold in morning: 37. Apples sold in afternoon: 48.\n"
                "Total sold: 37 + 48 = 85. (Check: 37+40=77, 77+8=85 ✓)\n"
                "Remaining: 140 - 85 = 55.\n"
                "The answer is 55."
            ),
            "gold_answer": "55",
        },
        {
            "question": "Maria earns $18 per hour and works 6 hours a day, 5 days a week. What does she earn in a week?",
            "error_demonstration": (
                "Daily earnings: 18 × 6 = 98. [ARITHMETIC ERROR: 18×6=108]\n"
                "Weekly: 98 × 5 = 490. This is WRONG."
            ),
            "correct_reasoning": (
                "Daily earnings: $18 × 6 = $108. (18×6 = 18×5 + 18 = 90+18 = 108 ✓)\n"
                "Weekly earnings: $108 × 5 = $540.\n"
                "The answer is 540."
            ),
            "gold_answer": "540",
        },
    ],

    "E2": [  # Distractor Capture
        {
            "question": (
                "Tom has 24 marbles. He gives 8 to his friend. "
                "His sister also collected 15 stamps last week which she keeps in a separate box. "
                "How many marbles does Tom have now?"
            ),
            "error_demonstration": (
                "Tom has 24 marbles and gives 8 away. His sister has 15 stamps.\n"
                "Remaining: 24 - 8 + 15 = 31. [ERROR: included the irrelevant stamps!]"
            ),
            "correct_reasoning": (
                "The stamps are irrelevant — they belong to Tom's sister and are not marbles.\n"
                "Focus only on Tom's marbles: 24 - 8 = 16.\n"
                "The answer is 16."
            ),
            "gold_answer": "16",
        },
        {
            "question": (
                "A factory produces 200 widgets per day. The manager's car gets 30 miles per gallon. "
                "The factory runs 5 days a week. How many widgets does it produce in a week?"
            ),
            "error_demonstration": (
                "200 widgets/day × 5 days = 1000. Also the car gets 30 mpg (irrelevant) so... 1000/30? "
                "[ERROR: distractor pulled into calculation]"
            ),
            "correct_reasoning": (
                "The car's fuel efficiency is irrelevant to widget production.\n"
                "Weekly widgets: 200 × 5 = 1000.\n"
                "The answer is 1000."
            ),
            "gold_answer": "1000",
        },
    ],

    "E3": [  # Premise-Order Sensitivity
        {
            "question": (
                "Conclusion: All students passed. "
                "Some students studied hard. "
                "Everyone who studied hard passed. "
                "Is the conclusion True, False, or Unknown?"
            ),
            "error_demonstration": (
                "The first thing mentioned is the conclusion. If all students passed... "
                "[ERROR: reasoning from conclusion backward, confused by ordering]"
            ),
            "correct_reasoning": (
                "Let me reorder the premises logically:\n"
                "P1: Some students studied hard.\n"
                "P2: Everyone who studied hard passed.\n"
                "Conclusion to check: All students passed.\n"
                "P1+P2 only guarantee that studying students passed, not ALL students.\n"
                "Unknown"
            ),
            "gold_answer": "Unknown",
        },
    ],

    "E4": [  # Step Skipping
        {
            "question": "A car travels 60 mph for 2 hours, then 40 mph for 3 hours. What is the total distance?",
            "error_demonstration": (
                "Total distance ≈ 50 mph × 5 hours = 250 miles. "
                "[ERROR: skipped computing each leg separately, used wrong average]"
            ),
            "correct_reasoning": (
                "Leg 1: 60 mph × 2 hours = 120 miles.\n"
                "Leg 2: 40 mph × 3 hours = 120 miles.\n"
                "Total distance: 120 + 120 = 240 miles.\n"
                "The answer is 240."
            ),
            "gold_answer": "240",
        },
        {
            "question": "Sam buys 3 notebooks at $4 each and 2 pens at $1.50 each. He pays with a $20 bill. How much change does he get?",
            "error_demonstration": (
                "Total ≈ $15, change ≈ $5. [ERROR: jumped to approximation, skipped individual calculations]"
            ),
            "correct_reasoning": (
                "Notebooks: 3 × $4 = $12.\n"
                "Pens: 2 × $1.50 = $3.00.\n"
                "Total: $12 + $3 = $15.\n"
                "Change: $20 - $15 = $5.\n"
                "The answer is 5."
            ),
            "gold_answer": "5",
        },
    ],

    "E5": [  # Hallucinated Premise
        {
            "question": "All mammals breathe air. Whales are mammals. Do whales breathe air?",
            "error_demonstration": (
                "Whales live in water and breathe through gills. Since they breathe through gills, "
                "they do not breathe air. [ERROR: introduced false fact about gills not in premises]"
            ),
            "correct_reasoning": (
                "Stick to what the premises state:\n"
                "P1: All mammals breathe air.\n"
                "P2: Whales are mammals.\n"
                "Conclusion: Whales breathe air. (Even if external knowledge differs, we reason from premises only.)\n"
                "True"
            ),
            "gold_answer": "True",
        },
    ],

    "E6": [  # Format Error
        {
            "question": "If a pizza is cut into 8 slices and you eat 3, what fraction remains?",
            "error_demonstration": (
                "5 out of 8 slices remain, which is five eighths or about 62.5 percent. "
                "[FORMAT ERROR: didn't give a clean numeric answer as required]"
            ),
            "correct_reasoning": (
                "Slices remaining: 8 - 3 = 5.\n"
                "As a fraction: 5/8.\n"
                "As a decimal: 0.625.\n"
                "The answer is 5 (remaining slices)."
            ),
            "gold_answer": "5",
        },
    ],

    "E7": [  # Logical Connective / Reversal Error
        {
            "question": "No reptiles are warm-blooded. Snakes are reptiles. Are snakes warm-blooded?",
            "error_demonstration": (
                "Warm-blooded animals include mammals. Snakes are reptiles, which are like warm-blooded. "
                "[ERROR: reversed the negation — 'no reptiles are warm-blooded' became confused]"
            ),
            "correct_reasoning": (
                "P1: No reptiles are warm-blooded. (Universal negation)\n"
                "P2: Snakes are reptiles.\n"
                "Therefore: Snakes are not warm-blooded.\n"
                "False"
            ),
            "gold_answer": "False",
        },
        {
            "question": "If it is raining, the ground is wet. The ground is wet. Is it raining?",
            "error_demonstration": (
                "The ground is wet, and wet ground means it rained, so it must be raining. "
                "[ERROR: affirming the consequent — wet ground has other causes]"
            ),
            "correct_reasoning": (
                "P1: Rain → Wet ground. (If raining, ground is wet)\n"
                "P2: Ground is wet.\n"
                "This is affirming the consequent — the ground could be wet for other reasons (sprinklers, etc.).\n"
                "We cannot conclude it is raining.\n"
                "Unknown"
            ),
            "gold_answer": "Unknown",
        },
    ],
}


# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────
def _get_exemplars(answer_type: str, k: int, use_cot: bool, seed: int = 42) -> List[Dict]:
    """Sample k standard exemplars for the given answer type."""
    import random
    pool = STANDARD_EXEMPLARS.get(answer_type, STANDARD_EXEMPLARS["numeric"])
    rng = random.Random(seed)
    chosen = rng.choices(pool, k=k) if k > len(pool) else rng.sample(pool, min(k, len(pool)))
    return chosen


def _exemplar_to_messages(exemplar: Dict, use_cot: bool) -> List[Dict]:
    """Convert a standard exemplar dict to user/assistant message pair."""
    user = {"role": "user", "content": exemplar["question"]}
    content = exemplar["cot"] if use_cot else f"The answer is {exemplar['answer_only']}."
    assistant = {"role": "assistant", "content": content}
    return [user, assistant]


def _error_exemplar_to_messages(ex: Dict) -> List[Dict]:
    """
    Convert an error-targeted exemplar to messages.
    Shows: question → error demonstration (labeled) → correction.
    """
    user_content = (
        ex["question"] + "\n\n"
        "[Note: Here is an example of a common error and its correction:]\n"
        f"INCORRECT reasoning: {ex['error_demonstration']}\n"
        f"CORRECT reasoning: {ex['correct_reasoning']}"
    )
    user = {"role": "user", "content": user_content}
    assistant = {"role": "assistant", "content": ex["correct_reasoning"]}
    return [user, assistant]


def _error_exemplar_correct_only_to_messages(ex: Dict) -> List[Dict]:
    """Convert a targeted exemplar to messages without showing the wrong trace."""
    user_content = (
        ex["question"] + "\n\n"
        "[Note: Here is a corrected example for this kind of problem:]\n"
        f"CORRECT reasoning: {ex['correct_reasoning']}"
    )
    user = {"role": "user", "content": user_content}
    assistant = {"role": "assistant", "content": ex["correct_reasoning"]}
    return [user, assistant]


def _choose_error_target_class(
    strategy_key: str,
    error_class: Optional[str],
    record_id: str,
    seed: int,
) -> Optional[str]:
    """Choose the exemplar bank for S5 and its ablations."""
    if strategy_key in ("S5", "S5_CORRECT_ONLY"):
        return error_class
    if strategy_key == "S5_RANDOM":
        import random
        classes = sorted(EXEMPLAR_BANK.keys())
        if error_class in classes and len(classes) > 1:
            classes = [c for c in classes if c != error_class]
        rng = random.Random(f"{seed}:{record_id}:S5_RANDOM")
        return rng.choice(classes) if classes else None
    return None


def build_prompt(
    record: Dict,
    strategy_key: str,
    answer_type: str,
    error_class: Optional[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a messages list for a given record and ICL strategy.

    record:        normalised record from data_loader
    strategy_key:  one of S0–S6
    answer_type:   "numeric" | "choice" | "trivalent"
    error_class:   required for S5 (error-targeted ICL)
    seed:          for reproducible exemplar sampling

    Returns: list of {"role": ..., "content": ...} dicts
    """
    from src.config import ICL_STRATEGIES

    strategy = ICL_STRATEGIES[strategy_key]
    system = SYSTEM_PROMPTS.get(answer_type, SYSTEM_PROMPTS["numeric"])
    question = record["question"]

    messages = [{"role": "system", "content": system}]

    k = strategy.get("k_shots", 0)
    use_cot = strategy.get("use_cot", False)

    if strategy_key == "S0":
        # Zero-shot: just the question
        messages.append({"role": "user", "content": question})

    elif strategy_key == "S1":
        # Zero-shot CoT: question + "Let's think step by step"
        messages.append({
            "role": "user",
            "content": question + "\n\nLet's think step by step."
        })

    elif strategy_key in ("S2", "S3", "S4"):
        # Standard few-shot (answer-only or CoT)
        exemplars = _get_exemplars(answer_type, k, use_cot, seed=seed)
        for ex in exemplars:
            messages.extend(_exemplar_to_messages(ex, use_cot))
        messages.append({"role": "user", "content": question})

    elif strategy_key in ("S5", "S5_RANDOM", "S5_CORRECT_ONLY"):
        # Error-targeted ICL plus ablations.
        target_class = _choose_error_target_class(
            strategy_key, error_class, record.get("id", ""), seed
        )
        if target_class is None or target_class not in EXEMPLAR_BANK:
            # Fallback to standard few-shot CoT if no error class known
            exemplars = _get_exemplars(answer_type, 3, True, seed=seed)
            for ex in exemplars:
                messages.extend(_exemplar_to_messages(ex, True))
        else:
            bank = EXEMPLAR_BANK[target_class]
            import random
            rng = random.Random(seed)
            chosen = rng.choices(bank, k=min(3, len(bank)))
            for ex in chosen:
                if strategy_key == "S5_CORRECT_ONLY":
                    messages.extend(_error_exemplar_correct_only_to_messages(ex))
                else:
                    messages.extend(_error_exemplar_to_messages(ex))
        messages.append({"role": "user", "content": question})

    elif strategy_key == "S6":
        # Self-consistency: zero-shot CoT base (sampling done at inference time)
        messages.append({
            "role": "user",
            "content": question + "\n\nLet's think step by step."
        })

    else:
        raise ValueError(f"Unknown strategy key: {strategy_key}")

    return messages
