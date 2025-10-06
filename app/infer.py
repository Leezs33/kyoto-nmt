# app/infer.py
from typing import Dict, Tuple, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 给每个方向准备多个候选仓库（按顺序尝试）
MODEL_CANDIDATES: Dict[str, List[str]] = {
    "ja-en": [
        "Helsinki-NLP/opus-mt-ja-en",
        "staka/fugumt-ja-en",
    ],
    "en-ja": [
        "Helsinki-NLP/opus-mt-en-ja",
        "staka/fugumt-en-ja",
    ],
}

_tok: Dict[str, AutoTokenizer] = {}
_mod: Dict[str, AutoModelForSeq2SeqLM] = {}

def _load_pair(name: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """优先 safetensors；失败再用普通权重。"""
    try:
        tok = AutoTokenizer.from_pretrained(name, use_safetensors=True)
        mod = AutoModelForSeq2SeqLM.from_pretrained(name, use_safetensors=True)
        print(f"[infer] loaded with safetensors: {name}")
        return tok, mod
    except Exception as e1:
        print(f"[infer] safetensors not available for {name}, fallback. detail: {e1}")
        tok = AutoTokenizer.from_pretrained(name)  # 普通权重
        mod = AutoModelForSeq2SeqLM.from_pretrained(name)
        print(f"[infer] loaded with default weights: {name}")
        return tok, mod

def _ensure(direction: str):
    if direction not in MODEL_CANDIDATES:
        raise ValueError(f"Unsupported direction: {direction}")
    if direction in _tok and direction in _mod:
        return
    last_err = None
    for name in MODEL_CANDIDATES[direction]:
        try:
            _tok[direction], _mod[direction] = _load_pair(name)
            print(f"[infer] using model for {direction}: {name}")
            return
        except Exception as e:
            print(f"[infer] candidate failed: {name} -> {e}")
            last_err = e
    raise RuntimeError(f"All candidates failed for {direction}. last error: {last_err}")

def translate(text: str, direction: str = "ja-en", max_new_tokens: int = 128) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    _ensure(direction)
    tok = _tok[direction]
    mod = _mod[direction]
    inputs = tok(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = mod.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True)
