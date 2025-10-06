from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .infer import translate

app = FastAPI(title="Kyoto NMT API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateReq(BaseModel):
    text: str = Field(..., description="要翻译的文本")
    direction: str = Field("ja-en", pattern="^(ja-en|en-ja)$")
    max_new_tokens: int = Field(128, ge=8, le=256)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/translate")
def translate_api(req: TranslateReq):
    return {"translation": translate(req.text, req.direction, req.max_new_tokens)}

# 也可提供固定路径，方便前端调用
@app.post("/translate/ja-en")
def ja_en(req: TranslateReq):
    return {"translation": translate(req.text, direction="ja-en", max_new_tokens=req.max_new_tokens)}

@app.post("/translate/en-ja")
def en_ja(req: TranslateReq):
    return {"translation": translate(req.text, direction="en-ja", max_new_tokens=req.max_new_tokens)}
