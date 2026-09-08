from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/model/prediction")
def predict(title: str = ""):
    return {"title": title, "result": f"你輸入的是：{title}"}
