"""
config.py — Central configuration for LLM Failure Modes project.
All model IDs, dataset specs, ICL strategies, error taxonomy, and paths live here.
Edit RUN_CONFIG at the bottom to control what actually gets executed.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ─────────────────────────────────────────────
# PATHS  (override via env vars for Colab/Drive)
# ─────────────────────────────────────────────
DRIVE_ROOT       = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/llm-failure-modes")
CHECKPOINT_DIR   = os.path.join(DRIVE_ROOT, "checkpoints")
RESULTS_DIR      = os.path.join(DRIVE_ROOT, "results")
FIGURES_DIR      = os.path.join(DRIVE_ROOT, "figures")
CACHE_DIR        = os.path.join(DRIVE_ROOT, "hf_cache")
LOCAL_SRC        = os.environ.get("LOCAL_SRC", "/content/llm-failure-modes/src")

# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────
# Each entry: name (short label), hf_id (HuggingFace model ID),
# family (for grouping), size_tier, dtype, max_new_tokens, requires_auth
MODELS = {
    # ── 1–2B tier ──────────────────────────────────────────────────────────
    "llama-1b": {
        "name":          "Llama-3.2-1B",
        "hf_id":         "meta-llama/Llama-3.2-1B-Instruct",
        "family":        "llama",
        "size_tier":     "1-2B",
        "requires_auth": True,
        "max_new_tokens": 256,
    },
    "qwen-1.5b": {
        "name":          "Qwen2.5-1.5B",
        "hf_id":         "Qwen/Qwen2.5-1.5B-Instruct",
        "family":        "qwen",
        "size_tier":     "1-2B",
        "requires_auth": False,
        "max_new_tokens": 256,
    },
    "gemma-2b": {
        "name":          "Gemma-2-2B",
        "hf_id":         "google/gemma-2-2b-it",
        "family":        "gemma",
        "size_tier":     "1-2B",
        "requires_auth": True,
        "max_new_tokens": 256,
    },

    # ── 3B tier (core novelty: fixed size, different families) ─────────────
    "llama-3b": {
        "name":          "Llama-3.2-3B",
        "hf_id":         "meta-llama/Llama-3.2-3B-Instruct",
        "family":        "llama",
        "size_tier":     "3B",
        "requires_auth": True,
        "max_new_tokens": 256,
    },
    "qwen-3b": {
        "name":          "Qwen2.5-3B",
        "hf_id":         "Qwen/Qwen2.5-3B-Instruct",
        "family":        "qwen",
        "size_tier":     "3B",
        "requires_auth": False,
        "max_new_tokens": 256,
    },
    "phi-3.5": {
        "name":          "Phi-3.5-mini",
        "hf_id":         "microsoft/Phi-3.5-mini-instruct",
        "family":        "phi",
        "size_tier":     "3B",
        "requires_auth": False,
        "max_new_tokens": 256,
    },

    # ── 7–9B tier ──────────────────────────────────────────────────────────
    "llama-8b": {
        "name":          "Llama-3.1-8B",
        "hf_id":         "meta-llama/Llama-3.1-8B-Instruct",
        "family":        "llama",
        "size_tier":     "7-9B",
        "requires_auth": True,
        "max_new_tokens": 256,
    },
    "qwen-7b": {
        "name":          "Qwen2.5-7B",
        "hf_id":         "Qwen/Qwen2.5-7B-Instruct",
        "family":        "qwen",
        "size_tier":     "7-9B",
        "requires_auth": False,
        "max_new_tokens": 256,
    },
    "gemma-9b": {
        "name":          "Gemma-2-9B",
        "hf_id":         "google/gemma-2-9b-it",
        "family":        "gemma",
        "size_tier":     "7-9B",
        "requires_auth": True,
        "max_new_tokens": 256,
    },
}

# ─────────────────────────────────────────────
# DATASET CONFIGS
# ─────────────────────────────────────────────
# answer_type: "numeric" | "choice" | "trivalent"
# paired_with: dataset key of the clean twin (for robustness ratio)
DATASETS = {
    # ── Easy tier: paired-perturbation arithmetic ───────────────────────────
    "gsm8k": {
        "hf_id":        "gsm8k",
        "hf_config":    "main",
        "split":        "test",
        "n_samples":    250,
        "answer_type":  "numeric",
        "tier":         "easy",
        "paired_with":  None,
        "q_field":      "question",
        "a_field":      "answer",
        "answer_regex": r"####\s*(\-?[\d,\.]+)",
    },
    "gsm_symbolic": {
        "hf_id":        "apple/GSM-Symbolic",
        "hf_config":    "main",
        "split":        "test",
        "n_samples":    150,
        "answer_type":  "numeric",
        "tier":         "easy",
        "paired_with":  "gsm8k",
        "q_field":      "question",
        "a_field":      "answer",
        "answer_regex": r"####\s*(\-?[\d,\.]+)",
    },
    "gsm_plus": {
        "hf_id":        "qintongli/GSM-Plus",
        "hf_config":    None,
        "split":        "test",
        "n_samples":    150,
        "answer_type":  "numeric",
        "tier":         "easy",
        "paired_with":  "gsm8k",
        "q_field":      "question",
        "a_field":      "answer",
        "answer_regex": r"####\s*(\-?[\d,\.]+)",
    },
    "gsm_ic": {
        # GSM-IC: irrelevant context distractor insertion (Shi et al. 2023)
        "hf_id":        "Aditi-5/GSM-IC",
        "hf_config":    None,
        "split":        "test",
        "n_samples":    150,
        "answer_type":  "numeric",
        "tier":         "easy",
        "paired_with":  "gsm8k",
        "q_field":      "question",
        "a_field":      "answer",
        "answer_regex": r"####\s*(\-?[\d,\.]+)",
    },

    # ── Medium tier: BIG-Bench Hard subsets ────────────────────────────────
    "bbh_logical_deduction": {
        "hf_id":        "lukaemon/bbh",
        "hf_config":    "logical_deduction_five_objects",
        "split":        "test",
        "n_samples":    150,
        "answer_type":  "choice",
        "tier":         "medium",
        "paired_with":  None,
        "q_field":      "input",
        "a_field":      "target",
        "answer_regex": r"\(([A-E])\)",
    },
    "bbh_tracking": {
        "hf_id":        "lukaemon/bbh",
        "hf_config":    "tracking_shuffled_objects_three_objects",
        "split":        "test",
        "n_samples":    150,
        "answer_type":  "choice",
        "tier":         "medium",
        "paired_with":  None,
        "q_field":      "input",
        "a_field":      "target",
        "answer_regex": r"\(([A-E])\)",
    },
    "bbh_causal": {
        "hf_id":        "lukaemon/bbh",
        "hf_config":    "causal_judgement",
        "split":        "test",
        "n_samples":    150,
        "answer_type":  "choice",
        "tier":         "medium",
        "paired_with":  None,
        "q_field":      "input",
        "a_field":      "target",
        "answer_regex": r"\b(Yes|No)\b",
    },

    # ── Hard tier: formal first-order logic ────────────────────────────────
    "folio": {
        "hf_id":        "yale-nlp/folio",
        "hf_config":    None,
        "split":        "validation",
        "n_samples":    200,
        "answer_type":  "trivalent",
        "tier":         "hard",
        "paired_with":  None,
        "q_field":      "story",
        "a_field":      "label",
        "answer_regex": r"\b(True|False|Unknown)\b",
    },
}

# ─────────────────────────────────────────────
# ERROR TAXONOMY (7 classes)
# ─────────────────────────────────────────────
ERROR_CLASSES = {
    "E1": {
        "name":        "Arithmetic Slip",
        "description": "Correct reasoning plan but arithmetic computation error",
        "source":      "Mirzadeh et al. 2024; Li et al. 2024",
    },
    "E2": {
        "name":        "Distractor Capture",
        "description": "Irrelevant context misleads answer selection",
        "source":      "Shi et al. 2023",
    },
    "E3": {
        "name":        "Premise-Order Sensitivity",
        "description": "Performance degrades when premises are reordered",
        "source":      "Chen et al. 2024",
    },
    "E4": {
        "name":        "Step Skipping",
        "description": "Multi-step reasoning abbreviated, intermediate steps missing",
        "source":      "Dziri et al. 2023",
    },
    "E5": {
        "name":        "Hallucinated Premise",
        "description": "Model introduces facts not present in the problem",
        "source":      "Dziri et al. 2023; FLARE 2025",
    },
    "E6": {
        "name":        "Format Error",
        "description": "Correct reasoning but answer not parseable in expected format",
        "source":      "GSM-Plus perturbation types",
    },
    "E7": {
        "name":        "Logical Connective / Reversal Error",
        "description": "Incorrect application of logical operators or reversal of known facts",
        "source":      "Berglund et al. 2023; FOLIO",
    },
    "E0": {
        "name":        "Correct",
        "description": "No error — model answered correctly",
        "source":      "N/A",
    },
    "EU": {
        "name":        "Unclassifiable",
        "description": "Cannot determine error type from trace",
        "source":      "N/A",
    },
}

ERROR_CLASS_KEYS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]  # excludes E0/EU

# ─────────────────────────────────────────────
# ICL STRATEGIES
# ─────────────────────────────────────────────
ICL_STRATEGIES = {
    "S0": {
        "name":        "Zero-shot",
        "description": "Direct question, no examples or CoT instruction",
        "source":      "Brown et al. 2020",
        "k_shots":     0,
        "use_cot":     False,
        "novel":       False,
    },
    "S1": {
        "name":        "Zero-shot CoT",
        "description": "Add 'Let's think step by step' after question",
        "source":      "Kojima et al. 2022",
        "k_shots":     0,
        "use_cot":     True,
        "novel":       False,
    },
    "S2": {
        "name":        "Few-shot (k=3, answer-only)",
        "description": "3 examples with question+answer, no reasoning trace",
        "source":      "Brown et al. 2020",
        "k_shots":     3,
        "use_cot":     False,
        "novel":       False,
    },
    "S3": {
        "name":        "Few-shot CoT (k=3)",
        "description": "3 examples with full reasoning chains",
        "source":      "Wei et al. 2022",
        "k_shots":     3,
        "use_cot":     True,
        "novel":       False,
    },
    "S4": {
        "name":        "Few-shot CoT (k=5)",
        "description": "5 examples with full reasoning chains",
        "source":      "Wei et al. 2022",
        "k_shots":     5,
        "use_cot":     True,
        "novel":       False,
    },
    "S5": {
        "name":        "Error-Targeted ICL (novel)",
        "description": "Exemplars matched to model's anticipated error class — "
                       "shows the error being made and its correction",
        "source":      "This work (novel contribution)",
        "k_shots":     3,
        "use_cot":     True,
        "novel":       True,
    },
    "S5_RANDOM": {
        "name":        "Random-Target ICL (ablation)",
        "description": "Uses error-targeted exemplar format, but samples from a random error class",
        "source":      "This work (ablation)",
        "k_shots":     3,
        "use_cot":     True,
        "novel":       False,
    },
    "S5_CORRECT_ONLY": {
        "name":        "Error-Targeted Correct-Only ICL (ablation)",
        "description": "Uses matched corrective exemplars without showing the wrong reasoning",
        "source":      "This work (ablation)",
        "k_shots":     3,
        "use_cot":     True,
        "novel":       False,
    },
    "S6": {
        "name":        "Self-Consistency (n=5)",
        "description": "Sample n=5 responses, majority vote on final answer",
        "source":      "Wang et al. 2022",
        "k_shots":     0,
        "use_cot":     True,
        "n_samples":   5,
        "temperature": 0.7,
        "novel":       False,
    },
}

ICL_STRATEGY_KEYS = list(ICL_STRATEGIES.keys())

# ─────────────────────────────────────────────
# QUANTISATION CONFIG
# ─────────────────────────────────────────────
QUANT_CONFIG = {
    "load_in_4bit":             True,
    "bnb_4bit_quant_type":      "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype":   "bfloat16",
}

# Generation defaults (per strategy overrides in icl_strategies.py)
GENERATION_DEFAULTS = {
    "do_sample":        False,
    "temperature":      1.0,
    "top_p":            1.0,
    "repetition_penalty": 1.1,
}

# ─────────────────────────────────────────────
# EXPERIMENT RUN CONFIG  ← edit this to control what runs
# ─────────────────────────────────────────────
# FULL_SCOPE_REFERENCE_DO_NOT_RUN_BY_DEFAULT:
# The original ambitious grid is preserved here for documentation/reporting.
#   models = [
#       "llama-3b", "qwen-3b", "phi-3.5",
#       "llama-8b", "qwen-7b", "gemma-9b",
#       "llama-1b", "qwen-1.5b", "gemma-2b",
#   ]
#   datasets = [
#       "gsm8k", "gsm_symbolic", "gsm_plus", "gsm_ic",
#       "bbh_logical_deduction", "bbh_tracking", "folio",
#   ]
#   strategies = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]

RUN_CONFIG = {
    # ── Active model grid ──────────────────────────────────────────────────
    # With 220 Colab Pro A100 credits (~73 GPU-hours), running 9 models ×
    # 5 datasets × 4 strategies (S0, S1, S3, S5) + S6/ablations on a
    # small subset fits comfortably (~55-60 GPU-hours ≈ 165-180 credits).
    #
    # Auth-gated models (llama-*, gemma-*) require HF_TOKEN set in Colab
    # Secrets.  No-auth models (qwen-*, phi-3.5) run without a token.
    "models": [
        # 1-2B tier — one per family for scaling curve
        "qwen-1.5b",    # no auth
        "llama-1b",     # needs HF token
        # gemma-2b excluded: same tier as qwen-1.5b, needs auth, adds 20+ min
        # 3B tier — key family-comparison tier
        "qwen-3b",      # no auth — primary model
        "phi-3.5",      # no auth — math-focused, tests H6
        "llama-3b",     # needs HF token — family contrast at 3B
        # 7-9B tier — scaling check
        "qwen-7b",      # no auth
        "llama-8b",     # needs HF token
        # gemma-9b excluded: needs auth; llama-8b covers the 7-9B tier
    ],

    # ── Datasets ───────────────────────────────────────────────────────────
    # 5 datasets covering all difficulty tiers and error classes.
    "datasets": [
        "gsm8k",                # easy / clean arithmetic (E1 baseline)
        "gsm_ic",               # easy / distractor capture (E2 target)
        "gsm_symbolic",         # easy / perturbation robustness (paired with gsm8k)
        "bbh_logical_deduction",# medium / logical reasoning (E3/E7)
        "folio",                # hard / formal FOL (E5/E7)
    ],

    # ── Main strategies ────────────────────────────────────────────────────
    # S0 (baseline) → S1 (zero-shot CoT) → S3 (few-shot CoT k=3) → S5 (novel)
    "strategies": ["S0", "S1", "S3", "S5"],

    # ── Ablation strategies (run on small subset only) ─────────────────────
    # S5_RANDOM: control — error-targeted format but wrong error class
    # S5_CORRECT_ONLY: ablation — correct class, no wrong-trace shown
    # S6: self-consistency — 5× inference cost, run on 50-item subset
    "optional_ablation_strategies": ["S5_RANDOM", "S5_CORRECT_ONLY", "S6"],
    "optional_ablation_datasets": ["gsm8k", "gsm_ic"],
    "optional_ablation_model": "qwen-3b",
    "optional_ablation_samples": 50,

    # ── Phase-2 scope ──────────────────────────────────────────────────────
    # Phase 2 (S1/S3/S5) runs on all models — that's where the heatmap data
    # comes from.  All models listed here will run S1, S3, S5.
    "phase2_models": [
        "qwen-1.5b", "llama-1b",
        "qwen-3b", "phi-3.5", "llama-3b",
        "qwen-7b", "llama-8b",
    ],

    # ── Infrastructure ─────────────────────────────────────────────────────
    # Checkpoint every N items.  50 is safe for Colab (saves ~every 4-8 min).
    "checkpoint_every": 50,

    # Samples per dataset.  100 gives meaningful statistics without blowing
    # the credit budget.  Use 10 for smoke tests.
    "max_samples": 100,

    # Reproducibility seed
    "seed": 42,

    # Run rule-based error coding automatically after S0 baseline
    "run_error_coding": True,

    # Items to sample for human annotation (Cohen's κ validation)
    "kappa_sample_size": 150,
}
