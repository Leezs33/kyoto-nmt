from typing import Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_MAP: Dict[str, str] = {
    "ja-en": "Helsinki-NLP/opus-mt-ja-en",
    "en-ja": "Helsinki-NLP/opus-mt-en-ja",
}

_tok: Dict[str, AutoTokenizer] = {}
_mod: Dict[str, AutoModelForSeq2SeqLM] = {}

def _ensure(direction: str):
    if direction not in MODEL_MAP:
        raise ValueError(f"Unsupported direction: {direction}")
    if direction not in _tok or direction not in _mod:
        name = MODEL_MAP[direction]
        _tok[direction] = AutoTokenizer.from_pretrained(name, use_safetensors=True)
        _mod[direction] = AutoModelForSeq2SeqLM.from_pretrained(name, use_safetensors=True)

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
