import torch
import torch.nn as nn
import numpy as np
import csv
import os
import random
from torch.utils.data import Dataset, DataLoader, random_split
from embeding import labels, docs, model as d2v_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

classes = sorted(set(labels))
label2idx = {label: i for i, label in enumerate(classes)}

X = np.array([d2v_model.dv[i] for i in range(len(docs))], dtype=float)
y = np.array([label2idx[label] for label in labels], dtype=int)


class TitleDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


ds = TitleDataset(X, y)

# 切割 8:2
train_ds, val_ds = random_split(ds, [0.8, 0.2])

loader = DataLoader(train_ds, shuffle=True, batch_size=64)

model = nn.Sequential(
    nn.Linear(40, 32),
    nn.ReLU(),
    nn.Linear(32, 50),
    nn.ReLU(),
    nn.Linear(50, 9),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = 0.0
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)

    print("epoch", epoch, "loss", total_loss / len(train_ds))

# 評估準確率
model.eval()
loader = DataLoader(val_ds, shuffle=True, batch_size=64)
correct_count = 0
with torch.no_grad():
    for xb, yb in loader:
        logits = model(xb)
        pred = logits.argmax(dim=1)
        for i in range(len(yb)):
            if pred[i] == yb[i]:
                correct_count += 1

correct_rate = correct_count / len(val_ds)
print("Accuracy", correct_rate * 100, "%")


# 用真實的文章標題來測試
# 走完整流程：原始標題 -> cleanFile 清理 -> tokenizer 斷詞 -> doc2vec 向量 -> 分類模型
# （tokenizer 在 import 時就會載入 CKIP 模型，所以放在這裡才 import，不影響前面的訓練）
from cleanFile import clean_title
from tokenizer import tokenize_titles

test_titles = [
    "[分享] 大谷翔平今天又轟出全壘打",
    "[問卦] 女友生日該送什麼禮物比較好",
    "[閒聊] 這季新番大家覺得哪部最好看",
    "[討論] 立法院昨天的表決結果",
    "[情報] 全聯今天衛生紙特價買一送一",
    "[新聞] 國軍漢光演習今日登場",
    "[菜單] 預算三萬求推薦電競主機",
    "[請益] 台積電這個價位可以進場嗎",
    "[請益] 面試上外商軟體工程師該怎麼談薪水",
]

cleaned = [clean_title(title.strip().lower()) for title in test_titles]
tokens_list = tokenize_titles(cleaned)

for title, tokens in zip(test_titles, tokens_list):
    vector = d2v_model.infer_vector(tokens, epochs=50)
    logits = model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0))
    pred = logits.argmax(dim=1).item()
    print(title, "->", classes[pred])
