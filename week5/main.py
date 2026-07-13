import numpy as np
import random


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


def initNetwork1():
    return [
        Layer(w=[[0.5, 0.6], [0.2, -0.6]], b=[0.3, 0.25], activation=ReLU()),
        Layer(w=[[0.8], [-0.5]],            b=[0.6],        activation=Linear()),
        Layer(w=[[0.6, -0.3]],              b=[0.4, 0.75],  activation=Linear()),
    ]


def initNetwork2():
    return [
        Layer(w=[[0.5, 0.6], [0.2, -0.6]], b=[0.3, 0.25], activation=ReLU()),
        Layer(w=[[0.8], [0.4]],             b=[-0.5],       activation=Sigmoid()),
    ]

def initNetwork3():
    return [
        Layer(w=[[0.5, 0.6], [0.2, -0.6]], b=[0.3, 0.25], activation=ReLU()),
        Layer(w=[[0.8], [-0.5]],            b=[0.6],        activation=Linear()),
    ]

def initNetwork4():
    return [
        Layer(w=[[0.5, 0.6]], b=[0.3, 0.25], activation=ReLU()),   
        Layer(w=[[0.8], [-0.5]], b=[-0.5], activation=Linear()), 
    ]


# ----- Model 1 -----
# print("----- Model 1 -----")
# nn1 = Network(initNetwork1())
# expects1 = np.array([0.8, 1])
# loss_fn1 = MSELoss()
# learning_rate1 = 0.01

# outputs = nn1.forward(np.array([1.5, 0.5]))
# loss = loss_fn1.get_loss(expects1, outputs)
# output_gradients = loss_fn1.get_output_gradients(outputs, expects1)
# nn1.backward(output_gradients)
# nn1.zero_grad(learning_rate1)

# print(f"Loss: {loss:.6f} | Output: {np.round(outputs, 4)}")
# for i, layer in enumerate(nn1.layers):
#     print(f"  Layer {i} w:\n{np.round(layer.w, 4)}")
#     print(f"  Layer {i} b: {np.round(layer.b, 4)}")


# ----- Model 2 -----
# print("----- Model 2 -----")
# nn2 = Network(initNetwork2())
# expects2 = np.array([1])
# loss_fn2 = BCELoss()
# learning_rate2 = 0.1

# for epoch in range(1000):
#     outputs = nn2.forward(np.array([0.75, 1.25]))
#     loss = loss_fn2.get_loss(expects2, outputs)
#     output_gradients = loss_fn2.get_output_gradients(outputs, expects2)
#     nn2.backward(output_gradients)
#     nn2.zero_grad(learning_rate2)
#     print(f"Final Loss: {loss:.6f} | Final Output: {np.round(outputs, 4)}")
    

# print(f"Final Loss: {loss:.6f} | Final Output: {np.round(outputs, 4)}")
# for i, layer in enumerate(nn2.layers):
#     print(f"  Layer {i} w:\n{np.round(layer.w, 4)}")
#     print(f"  Layer {i} b: {np.round(layer.b, 4)}")

print("----- Model 3 -----")

def generate_data(n):
    data = []
    for _ in range(n):
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)
        expect = 2 * x1 - x2 + 3
        data.append(([x1, x2], expect))
    return data


train_data = generate_data(100)



# 1. 打算用第一個回歸模型來修改
# 2. 發現網路一輸出有兩個，但要求y只有一個，解法:新增網路三
# 3. 發現output，輸出為nan，查看輸出output值超級大，可能是學習率的關係，解法，先調整學習率至0.01，但還是炸掉，在調整至0.001
# 4. 觀察發現Loss持續下降
# 5. 但如果學習綠率太小調整至帶小到0.0001，會導致Loss下降速度變慢，解法:調整學習率至0.001為最佳


nn3 = Network(initNetwork3())
loss_fn3 = MSELoss()
learning_rate3 = 0.001

losses = []
for epoch in range(100):          
    losses = []                  
    for x, y in train_data: 
        outputs = nn3.forward(np.array(x))
        loss = loss_fn3.get_loss(np.array([y]), outputs)
        output_gradients = loss_fn3.get_output_gradients(outputs, np.array([y]))
        nn3.backward(output_gradients)
        nn3.zero_grad(learning_rate3)
        losses.append(loss)

    average_loss = np.mean(losses)  
    print(f"Epoch {epoch}: Average Loss = {average_loss:.6f}")

# print("----- Model 4 -----")

# def generate_data(n):
#     data = []
#     for _ in range(n):
#         x = random.uniform(-10, 10)
#         expect = x ** 2
#         data.append(([x], expect))
#     return data


# train_data = generate_data(1000)



# # 1. 發現0.01學習率一樣會炸掉，調整至0.001，但平均loss很大，調整至0.0001還可以，但一樣無法降下來。
# # 2. 後續不管多加幾個epoch，或是調整學習率，Loss大概都在50~60區間，可能原因是神經網路太簡單(?



# nn4 = Network(initNetwork4())
# loss_fn4 = MSELoss()
# learning_rate4 = 0.0001

# losses = []
# for epoch in range(1000):          
#     losses = []                  
#     for x, y in train_data: 
#         outputs = nn4.forward(np.array(x))
#         loss = loss_fn4.get_loss(np.array([y]), outputs)
#         output_gradients = loss_fn4.get_output_gradients(outputs, np.array([y]))
#         nn4.backward(output_gradients)
#         nn4.zero_grad(learning_rate4)
#         losses.append(loss)

#     average_loss = np.mean(losses)  
#     print(f"Epoch {epoch}: Average Loss = {average_loss:.6f}")


