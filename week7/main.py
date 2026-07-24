import torch
import torch.nn as nn
import numpy as np
import csv
import os
from torch.utils.data import Dataset,DataLoader




# 讀取csv
def load_csv_as_array(file_path):
    sex = []
    hight = []   # 性別 + 身高
    weight = []    # 體重
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            gender, height, w = row
            gender = 1 if gender.strip().lower() == "male" else 0
            sex.append([gender])
            hight.append([float(height)])
            weight.append([float(w)])
    list1 = np.array(sex, dtype=float)     
    list2 = np.array(hight, dtype=float)
    list3 = np.array(weight, dtype=float)
    return list1, list2, list3

# 計算 z 分數
def calculate_zScore_list(prices_list):
    # 平均數
    avg_price = sum(prices_list) / len(prices_list)

    # 標準差
    sd_price = 0
    for price in prices_list:
        sd_price += (price - avg_price) ** 2
    sd_price = (sd_price / len(prices_list)) ** 0.5

    # 計算 z-score:(X - 平均數) / 標準差
    zScores = []
    for price in prices_list:
        zScore = (price - avg_price) / sd_price
        zScores.append(zScore)

    return zScores, avg_price, sd_price


# 處理excel
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gender-height-weight.csv")
sex ,high,weight  = load_csv_as_array(csv_path) 

# 處理excel
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gender-height-weight.csv")
sex ,high,weight  = load_csv_as_array(csv_path) 
# 身高z分數
hzScores, havg_price, hsd_price = calculate_zScore_list(high)
# 體重z分數
wzScores, wavg_price, wsd_price = calculate_zScore_list(weight)


class MyDataset(Dataset):
    def __init__(self,x ,y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

ds = MyDataset(np.hstack([sex, hzScores]), np.array(wzScores))
loader = DataLoader(ds, shuffle=True,batch_size=64)


model = nn.Sequential(
    nn.Linear(2, 5),  
    nn.ReLU(),
    nn.Linear(5, 1),    
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(50):
    total_loss = 0.0
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)

    epoch_loss = total_loss / len(ds)
    rmse = epoch_loss ** 0.5 * wsd_price[0]
    print(f"epoch {epoch+1}, loss={epoch_loss:.4f}, RMSE={rmse:.2f} lbs")

