"""
models.py — Model loading, quantisation, and chat-template utilities.

Handles: Llama-3.x, Qwen2.5, Phi-3.5, Gemma-2, Mistral.
All families use tokenizer.apply_chat_template for prompt formatting.
Memory management: always call unload_model() before loading a new one.
"""

import gc
import logging
import torch
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Quantisation helpers
# ─────────────────────────────────────────────
def get_bnb_config():
    """NF4 double-quantisation config (fits 7-9B in ~5-6 GB VRAM)."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _device_map(size_tier: str) -> str:
    """Auto device map; CPU fallback if no CUDA."""
    if torch.cuda.is_available():
        return "auto"
    logger.warning("No CUDA found — loading on CPU (slow, reduce n_samples!)")
    return "cpu"


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_model(
    model_key: str,
    use_quantisation: bool = True,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple:
    """
    Load a model + tokenizer by config key.
    Returns (model, tokenizer, model_cfg).

    use_quantisation: set False for 1-2B models on A100 to avoid overhead.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import MODELS

    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Options: {list(MODELS.keys())}")

    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    size_tier = cfg["size_tier"]

    logger.info(f"Loading {cfg['name']} ({hf_id}) | size_tier={size_tier}")

    # Tokenizer
    tok_kwargs = dict(
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    if hf_token:
        tok_kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **tok_kwargs)
    _fix_tokenizer(tokenizer, cfg["family"])

    # Model
    model_kwargs = dict(
        trust_remote_code=True,
        device_map=_device_map(size_tier),
        cache_dir=cache_dir,
    )
    if hf_token:
        model_kwargs["token"] = hf_token

    if use_quantisation and torch.cuda.is_available():
        model_kwargs["quantization_config"] = get_bnb_config()
        logger.info("  Using NF4 4-bit quantisation")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        logger.info("  Using bf16 (no quantisation)")

    # Phi-3.5 / Phi-3 models raise DynamicCache.from_legacy_cache errors
    # with recent Transformers versions when flash-attention is used.
    # Force eager (standard) attention to sidestep the issue entirely.
    family = cfg["family"]
    if family == "phi":
        model_kwargs["attn_implementation"] = "eager"
        logger.info("  Phi family: using eager attention (DynamicCache workaround)")

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"  Loaded: {n_params:.2f}B parameters")
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"  GPU memory: {mem_gb:.2f} GB allocated")

    return model, tokenizer, cfg


def unload_model(model, tokenizer):
    """Free GPU memory — call between model evaluations."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Model unloaded, GPU cache cleared.")


# ─────────────────────────────────────────────
# Tokenizer fixes per family
# ─────────────────────────────────────────────
def _fix_tokenizer(tokenizer, family: str):
    """Apply family-specific tokenizer patches."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma uses right-padding; others use left for batch inference
    if family == "gemma":
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"


# ─────────────────────────────────────────────
# Prompt formatting (chat template)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. "
    "Think step by step before giving your final answer."
)

def format_prompt(
    messages: list,
    tokenizer,
    family: str,
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply the model's chat template to a list of message dicts.
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    # Some older tokenizers (Phi-3 variants) don't support system role in chat template
    if family == "phi":
        # Phi-3.5 supports system role natively
        pass
    # Gemma-2 doesn't support system role — merge into first user turn
    elif family == "gemma":
        messages = _merge_system_into_user(messages)

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as e:
        logger.warning(f"apply_chat_template failed ({e}), falling back to manual format")
        text = _manual_format(messages, family)

    return text


def _merge_system_into_user(messages: list) -> list:
    """Merge system message into first user message for Gemma."""
    out = []
    system_content = ""
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"] + "\n\n"
        else:
            if system_content and msg["role"] == "user":
                out.append({"role": "user", "content": system_content + msg["content"]})
                system_content = ""
            else:
                out.append(msg)
    return out


def _manual_format(messages: list, family: str) -> str:
    """Last-resort manual formatting when apply_chat_template fails."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if family == "llama":
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        elif family == "qwen":
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        elif family == "phi":
            parts.append(f"<|{role}|>\n{content}<|end|>")
        else:
            parts.append(f"{role.upper()}: {content}")
    if family == "llama":
        return "<|begin_of_text|>" + "".join(parts) + "<|start_header_id|>assistant<|end_header_id|>\n"
    elif family == "qwen":
        return "".join(parts) + "<|im_start|>assistant\n"
    return "".join(parts) + "\nAssistant:"


# ─────────────────────────────────────────────
# Raw token generation
# ─────────────────────────────────────────────
@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,   # 1.0 = disabled; >1.0 hurts Llama-3.x quality
    n_return: int = 1,
) -> list:
    """
    Generate n_return responses for a single prompt.
    Returns list of decoded strings (length = n_return).
    """
    # NOTE: 2048 was too small — S3/S4/S5 with 5 CoT exemplars + a long
    # FOLIO problem can exceed 2500 tokens, silently truncating the actual
    # question. 4096 covers all combinations in our 9×7×7 grid.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    if input_len >= 4096:
        logger.warning(
            f"Input length hit 4096 token limit — prompt may have been truncated. "
            f"Consider reducing few-shot exemplars or shortening the question."
        )

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample or (n_return > 1),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=n_return,
    )
    # Only apply repetition_penalty if explicitly > 1.0 — otherwise it
    # degrades greedy decoding for Llama-3.x and Qwen.
    if repetition_penalty and repetition_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if do_sample or n_return > 1:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    outputs = model.generate(**inputs, **gen_kwargs)
    # Decode only newly generated tokens
    responses = []
    for out in outputs:
        new_tokens = out[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(text.strip())
    return responses


def get_model_info(model_key: str) -> Dict:
    """Return model config dict without loading the model."""
    from src.config import MODELS
    return MODELS[model_key]
