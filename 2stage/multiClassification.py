import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from embeding import X, y

torch.manual_seed(42)

#
# 標籤轉成數字索引
#
CLASSES = sorted(set(y.tolist()))
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
y_idx = np.array([class_to_idx[label] for label in y], dtype=np.int64)


#
# 資料集
#
class TitleDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


dataset = TitleDataset(X, y_idx)

#
# 切成 8:2
#
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

BATCH_SIZE = 256
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

print(f"類別數 {len(CLASSES)}: {CLASSES}")
print(f"特徵維度 {X.shape[1]}")
print(f"訓練集 {len(train_set)}  驗證集 {len(val_set)}")




model = nn.Sequential(
    nn.Linear(40, 60),  
    nn.ReLU(),
    nn.Linear(60, 50),  
    nn.ReLU(),
    nn.Linear(50, 9)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


#
# 評估
#
def evaluate(loader):
    model.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = logits.argmax(dim=1)
            for pred, target in zip(preds, yb):
                if pred == target:
                    correct += 1
                n += 1
    return correct / n


#
# 訓練
#
for epoch in range(20):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)

    print(f"epoch {epoch + 1:2d}  train_loss {total_loss / len(train_set):.4f}")


val_acc = evaluate(val_loader)
print(f"val_acc {val_acc:.4f}")
