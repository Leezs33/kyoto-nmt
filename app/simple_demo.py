from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import torch

MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"

# 懒加载：服务先起来，第一次翻译时再下载/加载模型
_tokenizer = None
_model = None
def _ensure_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def translate(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    _ensure_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=128, num_beams=4)
    return _tokenizer.decode(out[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("# 日→英 翻译小工具 (MarianMT)\n先跑通MVP，第一次点击时会自动加载模型")
    inp = gr.Textbox(label="日文", lines=5, placeholder="例：京都は四季折々の風景が美しい。")
    btn = gr.Button("翻译")
    out = gr.Textbox(label="英文", lines=5)
    btn.click(fn=translate, inputs=inp, outputs=out)

if __name__ == "__main__":
    # 用本机环回地址 & 改个端口，避免冲突
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
