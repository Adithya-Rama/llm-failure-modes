# Project Context — LLM Failure Modes
## COMP6242 Deep Learning, Semester 1 2026

---

## What This Project Is

A controlled empirical study on **when small open-weight LLMs fail and whether in-context learning can fix it**. The core output is a **per-error-class × per-ICL-strategy recovery heatmap** — a measurement that does not exist in the literature for open-weight 1B–8B models at fixed parameter count across families.

**Submitted abstract (final):**
> Large language models sometimes fail in the most common ways, be it an arithmetic slip, hallucinated reasoning, or irrelevant context. Even though they fail, we never ask when and where do these models silently break down, and are these models capable of being fixed without retraining and through prompting alone. This project investigates this in a controlled prompting framework with three model families, Llama, Qwen, and Phi/Gemma at parameter sizes 1–2B, 3B, 7–9B. Task difficulty is varied across arithmetic benchmarks (GSM-Symbolic, GSM-Plus, GSM-IC), BIG-Bench Hard logic subsets, and formal logic from FOLIO. Error taxonomy is coupled with reasoning to measure how well different in-context learning strategies recover each error type. ICL strategies and a novel error-targeted condition are used that provide corrected exemplars matched to the failure class. The main output is a recovery heatmap, giving a clear picture of what prompting can and cannot fix.

---

## Project Idea Alignment

Maps to **Project Idea #5** from course notes: "Evaluate a small open model ... on tasks such as arithmetic, logic puzzles, or compositional reasoning. Identify systematic failures by designing adversarial prompts. Explore in-context learning as a mechanism for mitigating failures."

**COMP6242 novel contribution over base idea:**
1. Seven-class error taxonomy applied systematically (not just pass/fail)
2. Per-error-class recovery rates across 7 ICL strategies (cross-tabulation unreported in literature)
3. Novel **error-targeted ICL** (S5): exemplars matched to model's anticipated error class
4. Family-at-fixed-size comparison (3B tier: Llama vs Qwen vs Phi) — isolated family effect
5. Cohen's κ validation of error coding (inter-annotator agreement)

---

## Literature Grounding (key papers to cite)

| Paper | Relevance |
|---|---|
| Mirzadeh et al. 2024 — GSM-Symbolic (ICLR 2025) | Paired-perturbation benchmark; robustness ratio method |
| Li et al. 2024 — GSM-Plus (ACL 2024) | 8 perturbation families, E1/E2/E6 taxonomy support |
| Shi et al. 2023 — GSM-IC (ICML 2023) | Distractor capture (E2) benchmark |
| Chen et al. 2024 — Premise Order (ICML 2024) | E3 (premise-order sensitivity) baseline |
| Dziri et al. 2023 — Faith and Fate (NeurIPS 2023) | E4 step skipping; compositional reasoning limits |
| Berglund et al. 2023 — Reversal Curse (ICLR 2024) | E7 (logical reversal) |
| Wei et al. 2022 — Chain of Thought (NeurIPS 2022) | S3/S4 (few-shot CoT) |
| Kojima et al. 2022 — Zero-shot CoT (NeurIPS 2022) | S1 strategy |
| Wang et al. 2022 — Self-Consistency (ICLR 2023) | S6 strategy |
| Huang et al. 2023 — Cannot Self-Correct Yet | Why intrinsic self-correction fails |
| Chen et al. 2025 — Revisiting CoT (arXiv:2506.14641) | Key prior: CoT = format aligner, not reasoner, in small models |
| Schaeffer et al. 2023 — Emergent Abilities Mirage | Why we don't make scaling claims from 3 sizes |

---

## Codebase Structure

```
llm-failure-modes/
├── main.ipynb              ← Master Colab notebook (orchestrates everything)
├── generate_notebook.py    ← Regenerates main.ipynb from source
├── requirements.txt
├── README.md
├── CONTEXT.md              ← This file
└── src/
    ├── config.py           ← Model registry, dataset configs, ICL strategy defs
    ├── data_loader.py      ← Dataset loading + answer extraction utilities
    ├── models.py           ← Model loading (NF4 4-bit quant), chat template handling
    ├── prompts.py          ← Prompt builders for all 7 strategies + exemplar bank
    ├── inference.py        ← Inference engine: batch runs, checkpointing, S6 self-consistency
    ├── taxonomy.py         ← Error coding: rule-based heuristics + LLM-judge fallback
    ├── metrics.py          ← Accuracy, recovery delta, robustness ratio, JS divergence, McNemar
    ├── visualize.py        ← All 6 figures (heatmap, family bars, scaling, error dist, etc.)
    └── checkpointing.py    ← Drive checkpoint save/load, resume logic
```

---

## Experimental Design Summary

### Models (9 cells)
| Key | Name | Family | Size Tier | Auth Needed |
|---|---|---|---|---|
| llama-1b | Llama-3.2-1B-Instruct | llama | 1-2B | Yes (HF token) |
| qwen-1.5b | Qwen2.5-1.5B-Instruct | qwen | 1-2B | No |
| gemma-2b | Gemma-2-2B-IT | gemma | 1-2B | Yes |
| llama-3b | Llama-3.2-3B-Instruct | llama | 3B | Yes |
| qwen-3b | Qwen2.5-3B-Instruct | qwen | 3B | No |
| phi-3.5 | Phi-3.5-mini-Instruct | phi | 3B | No |
| llama-8b | Llama-3.1-8B-Instruct | llama | 7-9B | Yes |
| qwen-7b | Qwen2.5-7B-Instruct | qwen | 7-9B | No |
| gemma-9b | Gemma-2-9B-IT | gemma | 7-9B | Yes |

### Datasets (7 total)
| Key | Tier | Benchmark | Perturbation Type |
|---|---|---|---|
| gsm8k | Easy | GSM8K | None (clean baseline) |
| gsm_symbolic | Easy | GSM-Symbolic | Name/number substitution |
| gsm_plus | Easy | GSM-Plus | 8 perturbation types |
| gsm_ic | Easy | GSM-IC | Distractor insertion |
| bbh_logical_deduction | Medium | BBH | Logical deduction |
| bbh_tracking | Medium | BBH | Object tracking |
| folio | Hard | FOLIO | First-order logic |

### ICL Strategies (7)
| Key | Name | Novel? |
|---|---|---|
| S0 | Zero-shot | No |
| S1 | Zero-shot CoT | No |
| S2 | Few-shot answer-only (k=3) | No |
| S3 | Few-shot CoT (k=3) | No |
| S4 | Few-shot CoT (k=5) | No |
| **S5** | **Error-Targeted ICL** | **Yes — key novel contribution** |
| S6 | Self-Consistency (n=5) | No |

### Error Taxonomy (7 classes)
| Code | Name | Key Diagnostic |
|---|---|---|
| E1 | Arithmetic Slip | Correct plan, wrong computation |
| E2 | Distractor Capture | Irrelevant context incorporated |
| E3 | Premise-Order Sensitivity | Order change causes failure |
| E4 | Step Skipping | Missing intermediate steps |
| E5 | Hallucinated Premise | Facts not in problem introduced |
| E6 | Format Error | Correct reasoning, wrong format |
| E7 | Logical Connective/Reversal | Negation/connective misapplied |

---

## Key Hypotheses (from abstract + literature)

1. **H1 (uneven recovery):** Recovery rates are highly non-uniform across error classes — no single ICL strategy uniformly helps all error types.
2. **H2 (arithmetic → self-consistency):** E1 (arithmetic slips) is best recovered by S6 (self-consistency majority vote).
3. **H3 (distractor → error-targeted):** E2/E3 (distractors, premise order) are best addressed by S5 (error-targeted ICL).
4. **H4 (persistent failures):** E2 distractor capture remains largely unrecovered across all strategies for small models (per Mirzadeh et al.).
5. **H5 (CoT = format aligner):** S3/S4 (few-shot CoT) helps E6 (format errors) most, but may hurt E4 (step skipping) in models < 3B (per Chen et al. 2025).
6. **H6 (family effect at 3B):** At fixed 3B size, Phi-3.5 shows lower E1 rates than Llama-3.2-3B and Qwen2.5-3B (consistent with Phi-3.5 technical report's math focus).

---

## Metrics to Report

| Metric | What It Measures | Where Used |
|---|---|---|
| Exact-match accuracy | Overall performance | Table 1, Fig 2, Fig 3 |
| Per-class recovery Δ | ICL rescue rate per error type | **Fig 1 (heatmap)** — main result |
| Paired robustness ratio | acc_perturbed / acc_clean | Table 2, Fig 5 |
| Error distribution JS divergence | How much ICL shifts error profile | Fig 6 |
| Cohen's κ | Inter-annotator agreement on error coding | Reported in §Evaluation |
| McNemar's p-value | Statistical significance of S0 vs S5 | Footnotes / Appendix |

---

## Compute Plan (Colab Pro)

| Model tier | Memory (NF4 4-bit) | Strategy | Est. time per model |
|---|---|---|---|
| 1-2B | ~2-3 GB | bf16 or 4-bit | ~15–30 min |
| 3B | ~3-4 GB | NF4 4-bit | ~30–60 min |
| 7-9B | ~5-6 GB | NF4 4-bit | ~60–90 min |

Total budget (9 models × 7 strategies × ~500 items): ~15–20 GPU-hours. Spread over multiple Colab sessions with checkpointing. Recommended: run one model per session.

---

## What Has Been Done

- [x] Literature review (Mirzadeh, Dziri, Shi, Chen, Berglund, Wei, Kojima, Wang, Madaan)
- [x] Project registration submitted (19 Apr 2026)
- [x] Codebase designed and implemented:
  - config.py, data_loader.py, models.py, prompts.py
  - inference.py, taxonomy.py, metrics.py, visualize.py, checkpointing.py
  - main.ipynb (Colab-ready, Drive-integrated, checkpoint-safe)
- [x] Exemplar bank for all 7 error classes (S5 novel strategy)
- [x] Error-targeted ICL prompt builder
- [x] Rule-based + LLM-judge error taxonomy coder
- [x] Recovery heatmap visualization (Fig 1)
- [x] Family comparison, scaling curves, error distribution, robustness, JS divergence plots

## What's Remaining

- [ ] Run experiments (Phase 1: S0 baseline, Phase 2: S1–S6 ICL strategies)
- [ ] Human annotation of 150-item sample + Cohen's κ computation
- [ ] Statistical significance tests (McNemar's paired test)
- [ ] Ablation: exemplar order sensitivity (3 seeds on S3)
- [ ] Write final report (due 14 June 2026)
- [ ] Record 5-minute video presentation (due 31 May 2026)

---

## Potential Extensions (Future Work / Report Section)

1. **Dynamic error-targeted ICL**: Instead of using baseline error predictions, detect the error class in real-time from the first reasoning attempt, then re-prompt with a targeted exemplar.
2. **Cross-dataset error transfer**: Does an error class identified on GSM8K predict failures on FOLIO? Test the taxonomy's generalizability.
3. **Exemplar quality ablation**: Compare hand-crafted S5 exemplars vs. auto-generated ones (using GPT-4o-mini to generate corrective examples).
4. **LLM-judge calibration**: Test multiple judge models for error coding and compare κ.
5. **BBEH (BIG-Bench Extra Hard)**: Run on the newer, harder Kazemi et al. 2025 extension — our 7-9B models likely haven't saturated it.
6. **Instruction fine-tuning vs. base models**: Compare instruct vs. base variants to test whether RLHF changes which error classes dominate.
7. **Quantization effect on error distribution**: Does NF4 4-bit change which errors occur vs. bf16? Validate on a calibration subset.

---

## Report Outline (suggested)

1. **Introduction** — Why failure modes matter; research question; contributions
2. **Related Work** — GSM-Symbolic, GSM-Plus, GSM-IC, BBH, FOLIO, CoT literature, ICL recovery literature
3. **Methodology** — Error taxonomy, experimental design, models, datasets, strategies, evaluation metrics
4. **Results** — Recovery heatmap (main), accuracy tables, robustness ratios, family comparison, error distributions
5. **Discussion** — Which hypotheses confirmed; persistent failures; family effects; limitations
6. **Conclusion & Future Work**
7. **References**
8. **Appendix** — Exemplar bank samples, annotation guidelines, Cohen's κ details

---

*Last updated: project implementation complete, awaiting experimental runs.*
