import sys
from contextlib import asynccontextmanager
import csv
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel

import torch
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from gensim.models.doc2vec import Doc2Vec

BASE_DIR = Path(__file__).resolve().parent

# 2stage 不是 package，要先讓 python 找得到才 import 得到裡面的模組
sys.path.append(str(BASE_DIR.parent / "2stage"))
from cleanFile import clean_title
from net import D2V_PATH, load_model
from tokenizer import tokenize_titles

INFER_EPOCHS = 200


@lru_cache(maxsize=1)
def get_models():
    """doc2vec + 分類器只載入一次，之後每個 request 都拿快取。"""
    d2v = Doc2Vec.load(D2V_PATH)
    model, classes = load_model()
    return d2v, model, classes


def predict_board(title):
    """原始標題 -> 清理 -> 斷詞 -> doc2vec 向量 -> 分類，回傳 (板名, 機率)。"""
    d2v, model, classes = get_models()

    tokens = tokenize_titles([clean_title(title.strip().lower())])[0]
    vector = d2v.infer_vector(tokens, epochs=INFER_EPOCHS)
    with torch.no_grad():
        probs = model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0)).softmax(dim=1)[0]

    idx = int(probs.argmax())
    return classes[idx]



CSV_PATH = Path(__file__).parent / "user-labeled-titles.csv"
CSV_HEADERS = [ "title", "label"]

def append_to_csv(title: str, label: str) -> None:
    """把使用者回饋寫入 CSV。檔案不存在會自動建立並寫入表頭。"""
    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "title": title,
            "label": label,
        })


app = FastAPI()
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/model/prediction")
def prediction(title: str = ""):
    title = title.strip()
    if not title:
        return {"title": title, "label": None, "score": None, "result": "請輸入標題"}

    label = predict_board(title)
    return {
        "title": title,
        "label": label,
        "result": label,
    }



class FeedbackRequest(BaseModel):
    title: str
    label: str

@app.post("/api/model/feedback")
def feedback(data: FeedbackRequest):
    title = data.title.strip()
    label = data.label.strip()

    if not title:
        return {"title": title, "label": label, "result": "請輸入標題"}

    append_to_csv(title, label)

    return {
        "title": title,
        "label": label,
        "result": "已收到回饋",
    }