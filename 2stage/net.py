"""模型的路徑、結構與存讀檔，訓練（multiClassification.py）與網站（fastapi/main.py）共用。

結構放在同一個地方，兩邊才不會改到不一致而載不回權重。
"""

import os

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(__file__)
D2V_PATH = os.path.join(BASE_DIR, "doc2vec.model")
CLF_PATH = os.path.join(BASE_DIR, "classifier.pt")


def build_model(input_dim, n_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, n_classes),
    )


def save_model(model, classes, path=CLF_PATH):
    """存權重和類別對照表；特徵維度看得出來，不用另外存。"""
    torch.save({"state_dict": model.state_dict(), "classes": list(classes)}, path)
    return path


def load_model(path=CLF_PATH):
    """回傳 (model, classes)；model 已經是 eval 模式。"""
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    classes = ckpt["classes"]
    model = build_model(state_dict["0.weight"].shape[1], len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes
