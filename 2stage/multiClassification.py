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
        correct_count += (pred == yb).sum().item()

correct_rate = correct_count / len(val_ds)
print("Accuracy", correct_rate * 100, "%")
