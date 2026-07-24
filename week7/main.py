import torch
import torch.nn as nn
import numpy as np
import csv
import os
from torch.utils.data import Dataset,DataLoader,random_split




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

# 切割
train_ds, val_ds = random_split(ds, [0.8, 0.2])

loader = DataLoader(train_ds, shuffle=True,batch_size=64)


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

# 評估
        
total_loss = 0
loader = DataLoader(val_ds, shuffle=True,batch_size=64)
for xb, yb in loader:
    logits = model(xb)
    loss = criterion(logits, yb)
    total_loss += loss.item() * len(xb)


avg_loss = total_loss / len(val_ds)
avg_loss_pounds = (avg_loss ** 0.5) * wsd_price[0]
print("Average Loss in Weight ", avg_loss_pounds)

print("---------------model 1 finish-----------------------")

# 讀取 titanic.csv
def load_titanic_csv(file_path):
    survived = []
    pclass = []
    sex = []   # 1 是女, 0 是男
    age = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            row_survived = row[1]
            row_pclass = row[2]
            row_sex = row[4]
            row_age = row[5]
            if row_age.strip() == "":
                row_age = 0.0
            else :
                row_age = float(row_age)
            survived.append([float(row_survived)])
            pclass.append([float(row_pclass)])
            sex.append([1.0 if row_sex.strip().lower() == "female" else 0.0])
            age.append([float(row_age)])
    survived = np.array(survived, dtype=float)
    pclass = np.array(pclass, dtype=float)
    sex = np.array(sex, dtype=float)
    age = np.array(age, dtype=float)
    return survived, pclass, sex, age


titanic_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic.csv")
t_survived, t_pclass, t_sex, t_age = load_titanic_csv(titanic_csv_path)
# age z分數
azScores, avg_price, asd_price = calculate_zScore_list(t_age)


class MyDataset(Dataset):
    def __init__(self,x ,y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

ds = MyDataset(np.hstack([t_pclass, t_sex, azScores]), np.array(t_survived))


# 切割
train_ds, val_ds = random_split(ds, [0.8, 0.2])

loader = DataLoader(train_ds, shuffle=True,batch_size=64)

model = nn.Sequential(
    nn.Linear(3, 8),  
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()    
)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



for epoch in range(50):
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 評估準確率
loader = DataLoader(val_ds, shuffle=True,batch_size=1)
correct_count = 0
threshold = 0.5
for xb, yb in loader:
    logits = model(xb)
    survival_status = 0
    if logits > threshold:
        survival_status = 1
    if survival_status == yb:
        correct_count += 1

correct_rate = correct_count / len(val_ds)
print(correct_rate * 100,"%" )