import numpy as np
import csv
import os

class ReLU:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * self.mask


class Sigmoid:

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1 - self.y)


class Linear:

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = x
        return self.y

    def backward(self, dout):
        return dout


class Layer:
    def __init__(self, w, b, activation):
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.activation = activation
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        z = np.dot(x, self.w) + self.b
        return self.activation.forward(z)

    def backward(self, dout):
        dL_dz = self.activation.backward(dout)
        # 對W梯度
        self.dw = np.outer(self.x, dL_dz)
        # 對B的梯度
        self.db = dL_dz
        # 對X的梯度往外傳
        return dL_dz @ self.w.T


class MSELoss:
    def get_loss(self, expects, outputs):
        expects = np.array(expects, dtype=float)
        outputs = np.array(outputs, dtype=float)
        return np.mean((expects - outputs) ** 2)

    def get_output_gradients(self, outputs, expects):
        outputs = np.array(outputs, dtype=float)
        expects = np.array(expects, dtype=float)
        return 2 * (outputs - expects) / outputs.size


class BCELoss:
    def get_loss(self, expects, outputs):
        expects = np.array(expects, dtype=float)
        outputs = np.array(outputs, dtype=float)
        return -np.sum(expects * np.log(outputs) + (1 - expects) * np.log(1 - outputs))

    def get_output_gradients(self, outputs, expects):
        outputs = np.array(outputs, dtype=float)
        expects = np.array(expects, dtype=float)
        eps = 1e-12
        outputs = np.clip(outputs, eps, 1 - eps)
        return (outputs - expects) / (outputs * (1 - outputs))


class Network:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        y = input
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self, output_gradients):
        delta = output_gradients
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def zero_grad(self, learning_rate):
        for layer in self.layers:
            layer.w -= learning_rate * layer.dw
            layer.b -= learning_rate * layer.db


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


# 還原 z 分數:X = zScore * 標準差 + 平均數
def restore_zScore_list(zScores, avg_price, sd_price):
    prices = []
    for zScore in zScores:
        price = zScore * sd_price + avg_price
        prices.append(price)
    return prices

def initNetwork1():
    return [
        Layer(w=np.random.randn(2, 3) * 0.5, b=np.full(3, 0.1),  activation=ReLU()),
        Layer(w=np.random.randn(3, 1) * 0.5, b=np.full(1, 0.1),  activation=Linear()),
    ]


# 處理excel
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gender-height-weight.csv")
sex ,high,weight  = load_csv_as_array(csv_path) 
# 轉成z分數
# 身高z分數
hzScores, havg_price, hsd_price = calculate_zScore_list(high)
# 體重z分數
wzScores, wavg_price, wsd_price = calculate_zScore_list(weight)

# ----- Model 1 -----
print("----- Model 1 -----")
nn1 = Network(initNetwork1())
xs = np.hstack([sex,hzScores])
es = np.array(wzScores)
loss_fn1 = MSELoss()
learning_rate1 = 0.001

for epoch in range(10):
    losses = []
    for x, y in zip(xs, es):
        outputs = nn1.forward(x)
        loss = loss_fn1.get_loss(y, outputs)
        output_gradients = loss_fn1.get_output_gradients(outputs, y)
        nn1.backward(output_gradients)
        nn1.zero_grad(learning_rate1)

loss_sum=0
for x, y in zip(xs, es):
    outputs=nn1.forward(x)
    loss=loss_fn1.get_loss(y, outputs)
    loss_sum+=loss

avg_loss = loss_sum / len(xs)

print(avg_loss)
# 換成正確數值
avg_loss_pounds = (avg_loss ** 0.5) * wsd_price[0]
print(f"平均誤差:{avg_loss_pounds:.2f} 磅")