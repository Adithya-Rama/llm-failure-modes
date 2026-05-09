# LLM Failure Modes — COMP6242 Group Project

> Systematic failure modes in small open-weight LLMs: a controlled study of error taxonomy and in-context learning recovery.

## Quick Start (Google Colab)

1. **Upload `main.ipynb`** to Google Drive
2. **Push this repo to GitHub**, then edit `GITHUB_REPO` in Cell 1 of the notebook
3. **Set `HF_TOKEN`** in Colab Secrets (sidebar key icon) — needed for Llama and Gemma models
4. **Run cells top to bottom**. Results checkpoint to Drive automatically.

## Repository Structure

```
├── main.ipynb              ← Colab notebook (upload to Drive)
├── requirements.txt        ← Dependencies (auto-installed in notebook)
├── CONTEXT.md              ← Full project context, design decisions, next steps
└── src/
    ├── config.py           ← All models, datasets, strategies, error taxonomy
    ├── data_loader.py      ← Dataset loading (GSM8K/Plus/IC/Symbolic, BBH, FOLIO)
    ├── models.py           ← HuggingFace model loading with NF4 4-bit quantisation
    ├── prompts.py          ← Prompt builders + error-targeted exemplar bank
    ├── inference.py        ← Inference engine with checkpointing + self-consistency
    ├── taxonomy.py         ← 7-class error coder (rule-based + LLM-judge)
    ├── metrics.py          ← Accuracy, recovery delta, robustness ratio, kappa
    ├── visualize.py        ← All figures including the recovery heatmap
    └── checkpointing.py    ← Drive checkpoint save/load + resume logic
```

## Models

| Tier | Models | HF Auth? |
|------|--------|----------|
| 1–2B | Llama-3.2-1B, Qwen2.5-1.5B, Gemma-2-2B | Llama + Gemma need token |
| 3B   | Llama-3.2-3B, Qwen2.5-3B, Phi-3.5-mini | Llama needs token |
| 7–9B | Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B | Llama + Gemma need token |

## Datasets

| Dataset | Tier | Source |
|---------|------|--------|
| GSM8K | Easy | HuggingFace `gsm8k` |
| GSM-Symbolic | Easy | `apple/GSM-Symbolic` |
| GSM-Plus | Easy | `qintongli/GSM-Plus` |
| GSM-IC | Easy | Shi et al. 2023 (auto-fallback if unavailable) |
| BBH (3 subsets) | Medium | `lukaemon/bbh` |
| FOLIO | Hard | `yale-nlp/folio` |

## ICL Strategies

- **S0** Zero-shot
- **S1** Zero-shot CoT
- **S2** Few-shot answer-only (k=3)
- **S3** Few-shot CoT (k=3)
- **S4** Few-shot CoT (k=5)
- **S5** ⭐ **Error-Targeted ICL** (novel) — exemplars matched to model's anticipated error class
- **S6** Self-Consistency (n=5 samples, majority vote)

## Error Taxonomy

`E1` Arithmetic Slip · `E2` Distractor Capture · `E3` Premise-Order Sensitivity  
`E4` Step Skipping · `E5` Hallucinated Premise · `E6` Format Error · `E7` Logic Reversal

## Requirements

- Google Colab Pro (A100 recommended for 7–9B models)
- HuggingFace account + token (for Llama/Gemma gated models)
- ~20 GB Google Drive space for checkpoints + model cache

## Citation / Declaration

This project uses Claude (Anthropic) for code assistance and literature review synthesis, as declared per course requirements.
